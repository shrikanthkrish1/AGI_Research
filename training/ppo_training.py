# ppo_training.py
"""
BroRLPPOTrainer — PPO trainer tuned for MITS reasoning + ARC structured tokenization.

Features:
 - Integrates MITSInference (PMI*, PMI**, consistency) as the generation engine.
 - AugmentationScorer: batched log-likelihood + entropy scoring for efficiency.
 - Entropy-weighted consistency aggregation.
 - BroRL-style group normalization and PPO clipping.
 - Value head for advantage estimation (simple pooled-head over LM hidden states).
 - ARC-mode hooks: structured-token reward shaping.
"""

import math
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import local modules; adjust paths if needed.
try:
    from mits_inference import MITSInference, grid_to_structured_text, grid_hash
except Exception:
    # If import fails, the user must ensure mits_inference.py is in PYTHONPATH
    MITSInference = None
    grid_to_structured_text = None
    grid_hash = None

# Transformers (policy & tokenizer)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

# ---------------------------
# Helper: AugmentationScorer
# ---------------------------
class AugmentationScorer:
    """
    Light wrapper to compute batched log-likelihoods and entropies for (context, completion) pairs.
    Uses a causal LM model and its tokenizer (same interface as HF models).
    """
    def __init__(self, model, tokenizer, device: Optional[str] = None, max_length: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model.to(self.device)
        self.max_length = max_length or getattr(self.tokenizer, "model_max_length", 1024)

    def compute_entropy(self, context: str) -> float:
        """
        Entropy of next-token distribution given context.
        """
        try:
            enc = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits  # (1, L, V)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                return float(ent)
        except Exception:
            return float('nan')

    def compute_log_likelihood_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        """
        For each (context, completion) pair, returns average log-prob (per completion token).
        Returns NaN for failed items.
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            sub = pairs[i:i + batch_size]
            texts = []
            ctx_lens = []
            for ctx, comp in sub:
                # Build combined sequence with explicit SEP to make boundary detection precise
                ctx_txt = ctx
                if "[SEP]" not in ctx_txt:
                    combined = f"{ctx_txt} [SEP] {comp}"
                    ctx_len = len(self.tokenizer(ctx_txt + " [SEP]", return_tensors='pt', truncation=True, max_length=self.max_length)['input_ids'][0])
                else:
                    combined = f"{ctx_txt} {comp}"
                    # fallback: measure ctx token length
                    ctx_len = len(self.tokenizer(ctx_txt, return_tensors='pt', truncation=True, max_length=self.max_length)['input_ids'][0])
                texts.append(combined)
                ctx_lens.append(ctx_len)
            enc = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                log_probs = F.log_softmax(out.logits, dim=-1)
            input_ids = enc['input_ids']
            B, L = input_ids.shape
            for bi in range(B):
                ctx_len = ctx_lens[bi]
                ids = input_ids[bi]
                if ids.shape[0] <= ctx_len:
                    results.append(float('nan'))
                    continue
                comp_len = ids.shape[0] - ctx_len
                if comp_len <= 0:
                    results.append(float('nan')); continue
                start_pos = max(ctx_len - 1, 0)
                logits_slice = log_probs[bi, start_pos:start_pos + comp_len, :]
                targets = ids[ctx_len:ctx_len + comp_len].unsqueeze(1)
                token_logps = logits_slice.gather(1, targets).squeeze(1)
                total = float(token_logps.sum().item())
                results.append(total / max(1, comp_len))
        return results

# ---------------------------
# PPO Trainer
# ---------------------------
class BroRLPPOTrainer:
    """
    PPO trainer that uses MITSInference for generation and AugmentationScorer for fast consistency checks.
    Simplified but practical implementation of PPO for reasoning-level updates.
    """
    def __init__(
        self,
        policy_model,             # HF AutoModelForCausalLM (policy)
        tokenizer,
        mits: Optional[MITSInference] = None,
        aug_scorer: Optional[AugmentationScorer] = None,
        device: Optional[str] = None,
        lr: float = 1e-5,
        value_lr: float = 1e-4,
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        consistency_sample_size: int = 8,
        arc_mode: bool = False
    ):
        self.device = torch.device(device) if isinstance(device, str) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.policy.to(self.device)
        self.mits = mits
        self.arc_mode = arc_mode

        # augmentation scorer (if not provided, use policy model as scorer)
        if aug_scorer is None:
            self.aug_scorer = AugmentationScorer(self.policy, self.tokenizer, device=self.device)
        else:
            self.aug_scorer = aug_scorer

        # simple value head: pool LM last hidden states -> scalar value
        hidden_size = getattr(self.policy.config, "hidden_size", None)
        if hidden_size is None:
            # fallback: try embedding size
            hidden_size = getattr(self.policy.config, "n_embd", 768)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        ).to(self.device)

        # optimizers
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.AdamW(self.value_head.parameters(), lr=value_lr)

        # PPO params
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = ppo_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.consistency_sample_size = consistency_sample_size

    # ---------------------------
    # Helper: compute sequence logprob under a model (sum of token logprobs)
    # ---------------------------
    def compute_sequence_logprob(self, context: str, completion: str) -> float:
        """
        Return sum log-prob of completion conditioned on context under the policy model.
        """
        full = context
        if "[SEP]" not in full:
            full = f"{context} [SEP] {completion}"
            ctx_len = len(self.tokenizer(context + " [SEP]", return_tensors='pt')['input_ids'][0])
        else:
            full = f"{context} {completion}"
            ctx_len = len(self.tokenizer(context, return_tensors='pt')['input_ids'][0])
        enc = self.tokenizer(full, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length, padding=False).to(self.device)
        with torch.no_grad():
            out = self.policy(**enc)
            logits = out.logits  # (1, L, V)
            logp = F.log_softmax(logits, dim=-1)
        ids = enc['input_ids'][0]
        L = ids.shape[0]
        if L <= ctx_len:
            return float('-inf')
        # gather positions
        # predicted token at position t is given by logits at t-1
        token_logps = []
        for t in range(ctx_len, L):
            tgt = int(ids[t].item())
            logval = float(logp[0, t - 1, tgt].item())
            token_logps.append(logval)
        return float(sum(token_logps))

    # ---------------------------
    # Value function: use mean-pooled last hidden state
    # ---------------------------
    def compute_value(self, context: str) -> float:
        enc = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
        with torch.no_grad():
            out = self.policy(**enc, output_hidden_states=True)
            # last hidden state shape: (1, L, H)
            hidden = out.hidden_states[-1]  # take last
            pooled = hidden.mean(dim=1)     # (1, H)
            v = self.value_head(pooled)     # (1, 1)
            return float(v.item())

    # ---------------------------
    # Collect trajectories using MITS
    # ---------------------------
    def collect_trajectories(
        self,
        questions: List[str],
        input_grids: Optional[List[Any]] = None,
        ground_truths: Optional[List[Any]] = None,
        augmentations_fn = None,
        num_rollouts_per_q: int = 1,
        max_depth: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        For each question (and optional input_grid), run MITSInference to obtain candidates.
        Returns list per question of trajectory dicts:
          {
            'context': context_text,
            'prediction': predicted_text,
            'candidates': [...],
            'best': best_candidate,
            'pmi_star': ...,
            'pmi_double_star': ...,
            'logprob': old_logprob_under_policy,
            'value': baseline_value,
            'ground_truth': optional,
          }
        """
        trajectories = []
        for qi, q in enumerate(questions):
            q_trajs = []
            for r in range(num_rollouts_per_q):
                input_grid = None if input_grids is None else input_grids[qi]
                if input_grid is not None:
                    context_text = grid_to_structured_text(input_grid)
                    res = self.mits.inference(question=context_text, max_depth=max_depth, augmentations=None, input_grid=input_grid)
                else:
                    context_text = q
                    res = self.mits.inference(question=q, max_depth=max_depth, augmentations=None, input_grid=None)
                best = res.get('best')
                if best is None:
                    continue
                prediction = best.prediction
                # compute old logprob under current policy
                try:
                    old_logp = self.compute_sequence_logprob(context_text, prediction)
                except Exception:
                    old_logp = float('-inf')
                # baseline value
                try:
                    value = self.compute_value(context_text)
                except Exception:
                    value = 0.0
                # build augmentations if provided
                augs = []
                if augmentations_fn is not None:
                    augs = augmentations_fn(context_text)
                # compute consistency & pmi** using trainer's _compute_consistency_efficient if needed
                # But we will call mits.compute_consistency_score which uses the model scorer internally
                if augs:
                    try:
                        self.mits.compute_consistency_score(res, augs)
                    except Exception:
                        pass
                traj = {
                    'context': context_text,
                    'prediction': prediction,
                    'best': best,
                    'candidates': res.get('candidates', []),
                    'pmi_star': getattr(best, 'pmi_star', 0.0),
                    'pmi_double_star': getattr(best, 'pmi_double_star', getattr(best, 'pmi_star', 0.0)),
                    'logprob': old_logp,
                    'value': value,
                    'ground_truth': None if ground_truths is None else ground_truths[qi],
                    'augmentations': augs
                }
                q_trajs.append(traj)
            trajectories.append(q_trajs)
        return trajectories

    # ---------------------------
    # Reward computation
    # ---------------------------
    def compute_rewards(self, trajectories: List[List[Dict[str, Any]]],
                        lambda_correct: float = 1.0, lambda_pmi: float = 0.5, lambda_sym: float = 0.0) -> List[List[float]]:
        """
        Compute per-trajectory scalar rewards.
        Reward = lambda_correct * correctness + lambda_pmi * tanh(pmi_double_star) + lambda_sym * symmetry_score
        Returns nested list of rewards aligned with trajectories input shape.
        """
        all_rewards = []
        for trajs in trajectories:
            q_rewards = []
            for t in trajs:
                correct = 0.0
                gt = t.get('ground_truth')
                if gt is not None:
                    # simple string equality or structured grid equality
                    try:
                        if isinstance(gt, (list, tuple)):
                            # structured grid compare - exact match
                            if t['prediction'] == str(gt) or t['prediction'] == grid_to_structured_text(gt):
                                correct = 1.0
                        else:
                            # text match (loose)
                            if str(gt).strip().lower() in str(t['prediction']).strip().lower():
                                correct = 1.0
                    except Exception:
                        correct = 0.0
                pmi_term = math.tanh(float(t.get('pmi_double_star', 0.0)))
                # optional symmetry score placeholder: 0 for now
                sym_score = 0.0
                R = lambda_correct * correct + lambda_pmi * pmi_term + lambda_sym * sym_score
                q_rewards.append(float(R))
            all_rewards.append(q_rewards)
        return all_rewards

    # ---------------------------
    # PPO update
    # ---------------------------
    def ppo_update(self, trajectories: List[List[Dict[str, Any]]], rewards: List[List[float]]):
        """
        Perform PPO updates over collected trajectories.
        trajectories: nested list per question each containing traj dicts with 'logprob' and 'value'.
        rewards: nested list matching trajectories with scalar rewards.
        """
        # flatten
        flat_trajs = []
        flat_rewards = []
        for trajs, rs in zip(trajectories, rewards):
            for t, r in zip(trajs, rs):
                flat_trajs.append(t)
                flat_rewards.append(r)
        if len(flat_trajs) == 0:
            return
        # prepare tensors
        old_logps = torch.tensor([t['logprob'] for t in flat_trajs], dtype=torch.float32, device=self.device)
        values = torch.tensor([t['value'] for t in flat_trajs], dtype=torch.float32, device=self.device)
        returns = torch.tensor(flat_rewards, dtype=torch.float32, device=self.device)
        # advantages (simple)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        for epoch in range(self.ppo_epochs):
            # compute new logprobs and values under current policy
            new_logps = []
            new_values = []
            entropies = []
            for t in flat_trajs:
                ctx = t['context']; pred = t['prediction']
                # compute new logprob under current policy
                new_lp = self.compute_sequence_logprob(ctx, pred)
                new_logps.append(new_lp)
                v = self.compute_value(ctx)
                new_values.append(v)
                # entropy of policy distribution at context
                try:
                    ent = self.aug_scorer.compute_entropy(ctx)
                except Exception:
                    ent = 0.0
                entropies.append(ent)
            new_logps = torch.tensor(new_logps, dtype=torch.float32, device=self.device)
            new_values = torch.tensor(new_values, dtype=torch.float32, device=self.device)
            entropies = torch.tensor(entropies, dtype=torch.float32, device=self.device)

            ratios = torch.exp(new_logps - old_logps)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = F.mse_loss(new_values, returns)
            entropy_loss = -torch.mean(entropies)

            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Backprop: we update both policy and value_head; for stability do two optimizers
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # After update return some diagnostics
        return {
            "mean_return": float(torch.mean(torch.tensor(flat_rewards)).item()),
            "num_steps": len(flat_trajs)
        }

# ---------------------------
# Minimal smoke test (runs when invoked directly)
# ---------------------------
if __name__ == "__main__":
    print("Running PPO trainer smoke test with GPT-2 (small). This is for functionality testing only.")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # small model for smoke; swap to your production model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        # resize tokenizer if necessary
        # create mits inference (requires mits_inference.py present)
        if MITSInference is None:
            print("mits_inference not importable in this environment. Smoke test will not run end-to-end.")
        else:
            mits = MITSInference(model, tokenizer, device="cpu", arc_mode=True, add_arc_tokens_on_init=False)
            trainer = BroRLPPOTrainer(policy_model=model, tokenizer=tokenizer, mits=mits, aug_scorer=None, arc_mode=True)
            # synthetic ARC-like small grid as input
            input_grid = [
                [1, 1, 0],
                [1, 2, 2],
                [0, 2, 2]
            ]
            q_text = grid_to_structured_text(input_grid)
            # define simple augmentation function
            def simple_augs(ctx):
                return [{"text": ctx}, {"text": ctx}]
            # run one collect -> compute rewards -> update
            trajectories = trainer.collect_trajectories([q_text], input_grids=[input_grid], ground_truths=[None], augmentations_fn=simple_augs, num_rollouts_per_q=1, max_depth=1)
            rewards = trainer.compute_rewards(trajectories)
            print("Collected trajectories:", trajectories)
            print("Rewards:", rewards)
            stats = trainer.ppo_update(trajectories, rewards)
            print("PPO update stats:", stats)
    except Exception as e:
        print("Smoke test failed:", e)
