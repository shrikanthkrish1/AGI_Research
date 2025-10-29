# mits_inference.py
"""
MITSInference — research-faithful MITS tree + Structured Reasoning Tokenization (SRT)

Features:
 - Structured tokenization for ARC grid tasks (SRT)
 - Optional automatic addition of structured tokens into tokenizer
 - PatternMemory (LRU) for caching solved input->output templates
 - Corrected PMI computation via single-tokenization and vectorized log-prob gathering
 - Batch log-likelihood scoring and entropy-weighted consistency scoring
 - Safe handling of truncation (returns NaN or conservative values when needed)

Usage:
    from mits_inference import MITSInference
    mits = MITSInference(model, tokenizer, device='cuda', arc_mode=True, add_arc_tokens=True)
    res = mits.inference(question_text_or_structured_text, augmentations=[{'text':...}, ...])
"""

import logging
import math
import time
import hashlib
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


# ---------------------------
# Lightweight PatternMemory
# ---------------------------
class PatternMemory:
    """A small LRU cache mapping grid_hash -> solved output / transform metadata."""
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._store = OrderedDict()

    def _prune(self):
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def store(self, key: str, output_grid: Any, transform: Optional[Any] = None, meta: Optional[Dict] = None):
        entry = {"output_grid": output_grid, "transform": transform, "meta": meta or {}, "timestamp": time.time()}
        if key in self._store:
            self._store.pop(key)
        self._store[key] = entry
        self._prune()

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        val = self._store.get(key)
        if val is not None:
            # promote to end (most recent)
            self._store.pop(key)
            self._store[key] = val
        return val

    def contains(self, key: str) -> bool:
        return key in self._store

    def clear(self):
        self._store.clear()


# ---------------------------
# Structured tokenization helpers (ARC-focused)
# ---------------------------

def canonical_color_map(grid: List[List[int]]) -> Dict[int, str]:
    flat = [int(c) for row in grid for c in row]
    if len(flat) == 0:
        return {}
    unique, counts = np.unique(flat, return_counts=True)
    order = sorted(zip(-counts, unique), key=lambda x: (x[0], x[1]))
    ordered_vals = [v for _, v in order]
    return {int(v): f"COLOR_{i}" for i, v in enumerate(ordered_vals)}


def grid_hash(grid: List[List[int]]) -> str:
    arr = np.array(grid, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def grid_to_structured_text(input_grid: List[List[int]],
                            output_grid: Optional[List[List[int]]] = None,
                            include_delta: bool = False,
                            normalize_colors: bool = True,
                            include_size: bool = True) -> str:
    """
    Convert numeric grid → structured token text.
    Returns single string for tokenizer.
    """
    H = len(input_grid)
    W = len(input_grid[0]) if H > 0 else 0
    cmap = canonical_color_map(input_grid) if normalize_colors else None

    def tok_for(r: int, c: int, val: int) -> str:
        if cmap is not None:
            color_tok = cmap.get(int(val), f"COLOR_{val}")
        else:
            color_tok = f"COLOR_{val}"
        return f"R{r}C{c}_{color_tok}"

    parts = []
    parts.append("[START_GRID]")
    if include_size:
        parts.append(f"[SIZE]_{H}x{W}")
    for r in range(H):
        row_tokens = [tok_for(r, c, input_grid[r][c]) for c in range(W)]
        parts.append("[ROW] " + " ".join(row_tokens))
    parts.append("[END_GRID]")

    if output_grid is not None:
        parts.append("[SEP]")
        parts.append("[OUTPUT]")
        for r in range(H):
            row_tokens = [tok_for(r, c, output_grid[r][c]) for c in range(W)]
            parts.append("[ROW] " + " ".join(row_tokens))
        parts.append("[END_OUTPUT]")

    if include_delta and output_grid is not None:
        parts.append("[DELTA]")
        for r in range(H):
            for c in range(W):
                inv = int(input_grid[r][c]); outv = int(output_grid[r][c])
                if inv == outv:
                    parts.append(f"NOP_R{r}C{c}")
                else:
                    parts.append(f"DELTA_R{r}C{c}_FROM_{inv}_TO_{outv}")
        parts.append("[END_DELTA]")

    return " ".join(parts)


def build_arc_special_tokens(max_h: int = 30, max_w: int = 30, max_color_tokens: int = 50, include_delta_tokens: bool = True):
    toks = set()
    toks.update(["[START_GRID]", "[END_GRID]", "[ROW]", "[SIZE]", "[SEP]", "[OUTPUT]", "[END_OUTPUT]", "[DELTA]", "[END_DELTA]"])
    for r in range(max_h):
        toks.add(f"[R{r}]")
    for c in range(max_w):
        toks.add(f"[C{c}]")
    for i in range(max_color_tokens):
        toks.add(f"COLOR_{i}")
    if include_delta_tokens:
        toks.add("NOP")
        toks.add("DELTA")
    return sorted(list(toks))


# ---------------------------
# Candidate dataclass
# ---------------------------
@dataclass
class MITSCandidate:
    reasoning_steps: List[str]
    pmi_star: float = 0.0
    normalized_pmi: float = 0.0
    consistency_score: float = 0.0
    pmi_double_star: float = 0.0
    prediction: str = ""
    meta: Dict[str, Any] = None


# ---------------------------
# Main class
# ---------------------------
class MITSInference:
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        beam_width: int = 8,
        top_k: int = 32,
        arc_mode: bool = False,
        add_arc_tokens_on_init: bool = False,
        pattern_memory_max_size: int = 2048,
    ):
        """
        model: HF causal LM (AutoModelForCausalLM)
        tokenizer: HF tokenizer
        arc_mode: enable SRT & ARC structured tokenization on inference when text appears structured
        add_arc_tokens_on_init: whether to add ARC special tokens to tokenizer and resize model embeddings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model.to(self.device)
        self.beam_width = beam_width
        self.top_k = top_k
        self.entropy_history: List[float] = []
        self.arc_mode = arc_mode
        self.pattern_memory = PatternMemory(max_size=pattern_memory_max_size)

        # optionally add ARC special tokens (one-time call)
        if add_arc_tokens_on_init:
            try:
                new_tokens = build_arc_special_tokens(max_color_tokens=32)
                to_add = [t for t in new_tokens if t not in self.tokenizer.get_vocab()]
                if to_add:
                    self.tokenizer.add_tokens(to_add)
                    # resize model embeddings if possible
                    try:
                        self.model.resize_token_embeddings(len(self.tokenizer))
                        logger.info(f"Added {len(to_add)} ARC tokens and resized model embeddings.")
                    except Exception as e:
                        logger.warning(f"Added tokens but failed to resize model embeddings: {e}")
            except Exception as e:
                logger.error(f"Failed to add ARC tokens on init: {e}")

    # ---------------------------
    # SRT Preprocess (ARC structural token injection)
    # ---------------------------
    def preprocess_arc_text(self, text: str) -> str:
        """
        Accepts raw question / structured text; injects structural markers for improved tokenization.
        If text already looks like structured text (contains [START_GRID]) it's returned as-is.
        """
        try:
            if not text or "[START_GRID]" in text:
                return text
            s = text
            # common heuristics: presence of double brackets or 'grid' indicates ARC-like
            if "[[" in s and "]]" in s:
                # replace [[ ... ]] blocks with explicit markers
                s = s.replace("[[", "[START_GRID] ").replace("]]", " [END_GRID]")
            # mark common keywords
            s = s.replace("grid", "<GRID>")
            s = s.replace("color", "<COLOR>")
            s = s.replace("transform", "<TRANSFORM>")
            # put step markers
            import re
            s = re.sub(r"(Step\s*\d+[:\-]?)", lambda m: f"<STEP> {m.group(1)} </STEP>", s, flags=re.I)
            # collapse multiple spaces
            s = " ".join(s.split())
            return s
        except Exception as e:
            logger.debug(f"SRT preprocessing error: {e}")
            return text

    # ---------------------------
    # Entropy helper
    # ---------------------------
    def compute_entropy(self, text: str) -> float:
        """Compute entropy of next-token distribution for given text (softmax over logits at last position)."""
        try:
            txt = self.preprocess_arc_text(text) if self.arc_mode else text
            enc = self.tokenizer(txt, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits  # (1, L, V)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                # keep history
                self.entropy_history.append(ent)
                if len(self.entropy_history) > 10000:
                    self.entropy_history = self.entropy_history[-10000:]
                return float(ent)
        except Exception as e:
            logger.debug(f"compute_entropy failed: {e}")
            return float('nan')

    def get_dynamic_sample_count(self, base_samples: int = 4, cap: int = 32) -> int:
        if not self.entropy_history:
            return base_samples
        recent = self.entropy_history[-64:]
        mean = float(np.mean(recent))
        var = float(np.var(recent))
        factor = 1.0 + min(var / (mean + 1e-9), 2.0)
        cnt = int(max(1, min(cap, round(base_samples * factor))))
        return cnt

    # ---------------------------
    # PMI computation (single-tokenize, vectorized)
    # ---------------------------
    def compute_step_pmi(self, context_with_q: str, previous_steps: List[str], current_step: str) -> float:
        """
        PMI(current_step; context_with_q) = log p(current_step | context_with_q + prev) - log p(current_step | BOS + prev)
        Uses single-tokenization per sequence to avoid token alignment errors. Returns scalar log-diff.
        """
        try:
            # prepare text for both full and base contexts
            ctx = (context_with_q + " " + " ".join(previous_steps)).strip()
            full = (ctx + " " + current_step).strip()
            bos = self.tokenizer.bos_token or ""
            base_ctx = (bos + " " + " ".join(previous_steps)).strip()
            base_full = (base_ctx + " " + current_step).strip()

            # preprocess if arc_mode
            if self.arc_mode:
                full_txt = self.preprocess_arc_text(full)
                base_full_txt = self.preprocess_arc_text(base_full)
                ctx_txt = self.preprocess_arc_text(ctx)
                base_ctx_txt = self.preprocess_arc_text(base_ctx)
            else:
                full_txt, base_full_txt, ctx_txt, base_ctx_txt = full, base_full, ctx, base_ctx

            # single tokenizations
            enc_full = self.tokenizer(full_txt, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            enc_ctx = self.tokenizer(ctx_txt, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            enc_base_full = self.tokenizer(base_full_txt, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            enc_base_ctx = self.tokenizer(base_ctx_txt, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)

            # detect truncation
            if enc_full['input_ids'].shape[1] >= self.tokenizer.model_max_length or enc_ctx['input_ids'].shape[1] >= self.tokenizer.model_max_length:
                logger.debug("Truncation detected in compute_step_pmi (full/context) -> returning 0.0 PMI")
                return 0.0
            if enc_base_full['input_ids'].shape[1] >= self.tokenizer.model_max_length or enc_base_ctx['input_ids'].shape[1] >= self.tokenizer.model_max_length:
                # baseline truncation fallback
                logger.debug("Truncation detected in compute_step_pmi (base) -> baseline logp set to 0")
                log_p_base = 0.0
            else:
                # compute base logp
                out_base = self.model(**enc_base_full)
                log_probs_base = F.log_softmax(out_base.logits, dim=-1)  # (1, L, V)
                base_ctx_len = enc_base_ctx['input_ids'].shape[1]
                ids_base = enc_base_full['input_ids'][0]
                step_ids_base = ids_base[base_ctx_len:]
                if step_ids_base.numel() == 0:
                    log_p_base = 0.0
                else:
                    # pick logits positions for step: start at base_ctx_len-1
                    start_pos_b = max(base_ctx_len - 1, 0)
                    logits_slice_b = log_probs_base[0, start_pos_b:start_pos_b + step_ids_base.shape[0], :]
                    token_logps_b = logits_slice_b.gather(1, step_ids_base.unsqueeze(1)).squeeze(1)
                    log_p_base = float(token_logps_b.sum().item())

            # compute full logp
            out_full = self.model(**enc_full)
            log_probs_full = F.log_softmax(out_full.logits, dim=-1)
            ctx_len = enc_ctx['input_ids'].shape[1]
            ids_full = enc_full['input_ids'][0]
            step_ids_full = ids_full[ctx_len:]
            if step_ids_full.numel() == 0:
                return 0.0
            start_pos = max(ctx_len - 1, 0)
            logits_slice = log_probs_full[0, start_pos:start_pos + step_ids_full.shape[0], :]
            token_logps = logits_slice.gather(1, step_ids_full.unsqueeze(1)).squeeze(1)
            log_p_full = float(token_logps.sum().item())

            # optional weighting by step type
            weight = 1.0
            if "<STEP>" in current_step or self.arc_mode:
                weight = 1.05
            return (log_p_full - log_p_base) * weight
        except Exception as e:
            logger.debug(f"compute_step_pmi failed: {e}")
            return 0.0

    # ---------------------------
    # Batched log-likelihood scoring
    # ---------------------------
    def compute_log_likelihood(self, context: str, completion: str) -> float:
        """
        Compute average log-prob per completion token (completion conditioned on context).
        Returns NaN on failure/truncation.
        """
        try:
            ctx = self.preprocess_arc_text(context) if self.arc_mode else context
            full = f"{ctx} [SEP] {completion}" if "[SEP]" not in ctx else f"{ctx} {completion}"
            enc_full = self.tokenizer(full, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            enc_ctx = self.tokenizer(ctx + " [SEP]", return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            ctx_len = enc_ctx['input_ids'].shape[1]
            ids = enc_full['input_ids']
            if ids.shape[1] <= ctx_len:
                return float('nan')
            with torch.no_grad():
                out = self.model(**enc_full)
                log_probs = F.log_softmax(out.logits, dim=-1)
                targets = ids[0, ctx_len:].to(self.device)
                start_logits_pos = max(ctx_len - 1, 0)
                logits_slice = log_probs[0, start_logits_pos:start_logits_pos + targets.shape[0], :]
                token_logps = logits_slice.gather(1, targets.unsqueeze(1)).squeeze(1)
                total_logp = float(token_logps.sum().item())
                return total_logp / max(1, targets.shape[0])
        except Exception as e:
            logger.debug(f"compute_log_likelihood failed: {e}")
            return float('nan')

    def compute_log_likelihood_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        """
        Compute NLLs (avg per completion token) for a list of (context, completion) pairs in batches.
        Returns list of floats (NaN for failures).
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            sub = pairs[i:i + batch_size]
            texts = []
            ctx_lens = []
            for ctx, comp in sub:
                ctx_txt = self.preprocess_arc_text(ctx) if self.arc_mode else ctx
                combined = f"{ctx_txt} [SEP] {comp}" if "[SEP]" not in ctx_txt else f"{ctx_txt} {comp}"
                texts.append(combined)
                enc_ctx = self.tokenizer(ctx_txt + " [SEP]", return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length)
                ctx_lens.append(enc_ctx['input_ids'].shape[1])
            enc = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=self.tokenizer.model_max_length).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                log_probs = F.log_softmax(out.logits, dim=-1)  # (B, L, V)
            input_ids = enc['input_ids']  # (B, L)
            B, L = input_ids.shape
            for bi in range(B):
                ctx_len = ctx_lens[bi]
                ids = input_ids[bi]
                if ids.shape[0] <= ctx_len:
                    results.append(float('nan'))
                    continue
                # compute token logprobs for completion tokens
                # logits positions for tokens: use [ctx_len-1 .. ctx_len-1 + (len(completion_tokens)-1)]
                start_pos = max(ctx_len - 1, 0)
                # We will gather across available positions up to end-of-sequence
                # Compute how many tokens are considered as completion tokens
                comp_len = ids.shape[0] - ctx_len
                if comp_len <= 0:
                    results.append(float('nan'))
                    continue
                logits_slice = log_probs[bi, start_pos:start_pos + comp_len, :]  # (comp_len, V)
                targets = ids[ctx_len:ctx_len + comp_len].unsqueeze(1)  # (comp_len, 1)
                token_logps = logits_slice.gather(1, targets).squeeze(1)  # (comp_len,)
                total = float(token_logps.sum().item())
                results.append(total / max(1, comp_len))
        return results

    # ---------------------------
    # Consistency scoring (entropy-weighted)
    # ---------------------------
    def compute_consistency_score(self, result: Dict[str, Any], augmentations: List[Dict[str, Any]], tau: float = 0.7) -> None:
        """
        Compute consistency score for the best candidate using augmentation log-likelihoods weighted by entropies.
        Updates candidate.consistency_score and candidate.pmi_double_star in-place.
        """
        best: MITSCandidate = result.get("best")
        if best is None:
            return
        orig_pred = best.prediction
        if not augmentations:
            best.consistency_score = 1.0
            best.pmi_double_star = best.pmi_star
            return

        pairs = []
        for aug in augmentations:
            ctx = aug.get('text', '')
            if ctx:
                pairs.append((ctx, orig_pred))
        if len(pairs) == 0:
            best.consistency_score = 0.0
            best.pmi_double_star = best.pmi_star
            return

        nlls = np.array(self.compute_log_likelihood_batch(pairs), dtype=float)
        finite_mask = np.isfinite(nlls)
        if finite_mask.sum() == 0:
            best.consistency_score = 0.0
            best.pmi_double_star = best.pmi_star
            return
        nlls = nlls[finite_mask]
        # entropies
        entropies = np.array([self.compute_entropy(pairs[i][0]) for i in range(len(pairs))])[finite_mask]
        scores = -nlls  # higher better
        s_mean = float(np.mean(scores)); s_std = float(np.std(scores) if np.std(scores) > 1e-8 else 1.0)
        norm_scores = (scores - s_mean) / s_std
        weights = np.exp((-entropies / tau))
        weights = weights / (weights.sum() + 1e-12)
        w_mean = float((weights * norm_scores).sum())
        w_var = float((weights * (norm_scores - w_mean)**2).sum())
        w_std = float(math.sqrt(w_var)) if w_var > 1e-12 else 1.0
        threshold = w_mean + w_std
        consistency = float((weights * (norm_scores <= threshold)).sum())
        consistency = max(0.0, min(1.0, consistency))
        best.consistency_score = consistency
        best.pmi_double_star = best.pmi_star * consistency

    # ---------------------------
    # Prediction extraction (ARC-safe)
    # ---------------------------
    def extract_prediction(self, text: str) -> str:
        """
        Try to extract grid / answer from a reasoning step text.
        """
        try:
            import re
            m = re.search(r"\[\[[\d,\s\[\]]+\]\]", text)
            if m:
                return m.group(0)
            # lines: look for "Output:" or "Answer:" or last meaningful line
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in reversed(lines):
                if line.lower().startswith(("output", "answer")):
                    parts = line.split(":")
                    return parts[-1].strip()
            return lines[-1] if lines else text.strip()
        except Exception:
            return text.strip()

    # ---------------------------
    # Inference (beam search)
    # ---------------------------
    def inference(self, question: str, max_depth: int = 3, augmentations: Optional[List[Dict]] = None, input_grid: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        Run MITS beam search on `question`. If `input_grid` provided (list of lists), pattern memory is checked and structured tokens used.
        Returns dictionary: {"candidates": [MITSCandidate,...], "best": MITSCandidate, "prediction": ...}
        """
        # If pattern memory hit (when input_grid provided)
        if input_grid is not None:
            key = grid_hash(input_grid)
            cached = self.pattern_memory.retrieve(key)
            if cached:
                cand = MITSCandidate(reasoning_steps=[], pmi_star=0.0, normalized_pmi=0.0, prediction=cached["output_grid"])
                cand.pmi_double_star = cand.pmi_star
                return {"candidates": [cand], "best": cand, "prediction": cached["output_grid"], "cached": True}

        # Prepare initial context text
        if input_grid is not None:
            context_with_q = grid_to_structured_text(input_grid)
            self.arc_mode = True  # strengthen arc mode when structured input used
        else:
            context_with_q = question
            if self.arc_mode:
                context_with_q = self.preprocess_arc_text(question)

        beams: List[MITSCandidate] = [MITSCandidate(reasoning_steps=[], pmi_star=0.0, normalized_pmi=0.0, prediction="")]
        for depth in range(max_depth):
            new_candidates: List[MITSCandidate] = []
            for cand in beams:
                reasoning = cand.reasoning_steps.copy()
                context = (context_with_q + " " + " ".join(reasoning)).strip()
                # entropy-based dynamic sampling
                try:
                    ent = self.compute_entropy(context)
                except Exception:
                    ent = float('nan')
                self.entropy_history.append(ent)
                num_samples = self.get_dynamic_sample_count(base_samples=4, cap=32)
                # generate candidates - keep it simple: sample num_samples continuations
                try:
                    input_enc = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
                    gen_outputs = self.model.generate(
                        **input_enc,
                        max_new_tokens=32,
                        do_sample=True,
                        top_k=min(self.top_k, 100),
                        num_return_sequences=num_samples,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                except Exception as e:
                    logger.debug(f"generation failed: {e}")
                    continue

                for gen in gen_outputs:
                    try:
                        decoded = self.tokenizer.decode(gen, skip_special_tokens=True)
                        # extract the new step relative to context
                        new_step = decoded.replace(context, "").strip()
                        if not new_step:
                            continue
                        pmi_val = self.compute_step_pmi(context_with_q, reasoning, new_step)
                        total_pmi = cand.pmi_star + pmi_val
                        new_steps = reasoning + [new_step]
                        norm_pmi = total_pmi / max(1, len(new_steps))
                        new_candidates.append(MITSCandidate(reasoning_steps=new_steps, pmi_star=total_pmi, normalized_pmi=norm_pmi, prediction=new_step))
                    except Exception as e:
                        logger.debug(f"failed handling generated sequence: {e}")
                        continue

            if not new_candidates:
                break
            new_candidates.sort(key=lambda c: c.normalized_pmi, reverse=True)
            beams = new_candidates[:self.beam_width]

        # Finalize best
        if not beams:
            empty = MITSCandidate(reasoning_steps=[], pmi_star=0.0, normalized_pmi=0.0, prediction="")
            return {"candidates": [], "best": empty, "prediction": ""}

        best = max(beams, key=lambda c: c.normalized_pmi)
        # build human-friendly prediction
        prediction_text = self.extract_prediction(best.prediction)
        best.prediction = prediction_text

        result = {"candidates": beams, "best": best, "prediction": prediction_text}

        # compute consistency and PMI** if augmentations provided
        if augmentations:
            try:
                self.compute_consistency_score(result, augmentations)
            except Exception as e:
                logger.debug(f"consistency scoring failed: {e}")
                best.consistency_score = 0.0
                best.pmi_double_star = best.pmi_star

        return result
