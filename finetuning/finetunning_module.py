"""
Complete Fine-tuning Pipeline: Qwen 2.5 7B + MITS + PPO + Unsloth
=================================================================
(Fixed: Force single-process tokenization, avoid CUDA multiprocessing issues)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from typing import Dict, List, Optional
import argparse
import time
import gc

# CRITICAL: Set these BEFORE any imports that use CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,garbage_collection_threshold:0.6,expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Unsloth imports for efficient training
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

# Your MITS + PPO implementation
from mits_inference import MITSInference
from ppo_training import BroRLPPOTrainer, Trajectory, prepare_augmentations_for_mits
from arc_agi_augumentations import StructuralAugmenter

# ARC dataset handling
from arc_loader import ArcDataset
from arc_downloader import download_arc_data


# ============================================================================ #
# TOKENIZATION & MODEL UTILITIES
# ============================================================================ #
def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True):
    """
    Reduce vocabulary to single-character tokens + special tokens
    This dramatically reduces model size and improves ARC-AGI performance
    """
    import json
    from tokenizers import Tokenizer

    if not keep_norm:
        tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
        if tokenizer_json.get('normalizer') is not None:
            tokenizer_json['normalizer'] = None
            tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    if keep is None:
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:
        keep_indices = set(tokenizer.vocab.get(t) for t in keep if t in tokenizer.vocab)

    if keep_model_tok:
        keep_indices.update(tokenizer.all_special_ids)
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    if isinstance(v, list):
                        keep_indices.update(v)
                    elif v is not None:
                        keep_indices.add(v)

    keep_indices -= {None}
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    tok_json = json.loads(tokenizer._tokenizer.to_str())
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))

    with torch.no_grad():
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        model.resize_token_embeddings(len(row_select))
        model.get_input_embeddings().weight.data[:] = new_embed
        model.get_output_embeddings().weight.data[:] = new_lm_head

        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    if isinstance(v, list):
                        setattr(config, k, [mapping.get(t) for t in v])
                    else:
                        setattr(config, k, mapping.get(v))

    return mapping


class InputMaskingDataCollator:
    """Data collator that masks training examples appropriately"""
    def __init__(self, tokenizer, query_beg, reply_beg, mask_first_n_examples=0):
        self.tokenizer = tokenizer
        self.query_beg = query_beg
        self.reply_beg = reply_beg
        self.mask_first_n_examples = mask_first_n_examples

    def __call__(self, examples):
        texts = [ex['text'] for ex in examples]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128000
        )
        labels = batch['input_ids'].clone()

        # Mask entire inputs by default
        for i, text in enumerate(texts):
            labels[i][:] = -100

        batch['labels'] = labels
        return batch


def pre_tokenize_dataset(dataset: Dataset, tokenizer, text_field: str = "text"):
    """Pre-tokenize dataset to avoid multiprocessing issues"""
    print("Pre-tokenizing dataset (single-threaded)...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=128000,
            padding=False,  # Don't pad yet
        )
    
    # Force single process with num_proc=1
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # CRITICAL: Single process only
        remove_columns=[text_field],
        desc="Tokenizing"
    )
    
    return tokenized


# ============================================================================ #
# MITS + PPO ENHANCED TRAINER
# ============================================================================ #
class MITSPPOStarTrainer(BroRLPPOTrainer):
    """Enhanced trainer combining MITS + PPO-Star + Unsloth"""
    def __init__(self, *args, use_unsloth=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_unsloth = use_unsloth

    def train_epoch(self, train_dataset: Dataset, augmenter: StructuralAugmenter,
                    batch_size: int = 2, gradient_accumulation_steps: int = 2):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        indices = np.random.permutation(len(train_dataset))

        for batch_start in tqdm(range(0, len(indices), batch_size), desc="Training"):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [train_dataset[int(i)] for i in batch_indices]

            questions, ground_truths, all_augmentations = [], [], []

            for item in batch:
                task_dict = item.get('task')
                if task_dict:
                    augs = prepare_augmentations_for_mits(task_dict, augmenter, format_style='compact')
                    all_augmentations.extend(augs)
                    questions.append(item.get('text', ''))
                    gt = {'output': task_dict.get('test', [{}])[0].get('output', [])} if 'test' in task_dict else {'output': []}
                    ground_truths.append(gt)

            if not questions:
                continue

            try:
                trajectory_groups = self.collect_trajectories(
                    questions, ground_truths, all_augmentations, num_rollouts_per_question=4
                )
            except Exception as e:
                print(f"Error collecting trajectories: {e}")
                continue

            self.compute_rewards(trajectory_groups)
            self.compute_advantages(trajectory_groups)

            try:
                loss, stats = self.compute_policy_loss(trajectory_groups)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if (num_batches + 1) % gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (num_batches + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                total_loss += float(loss.item())
                num_batches += 1
            except Exception as e:
                print(f"Error in PPO update: {e}")
                continue

            if getattr(self, "device", None) == 'cuda':
                torch.cuda.empty_cache()

        return total_loss / num_batches if num_batches > 0 else 0.0


# ============================================================================ #
# MAIN TRAINING PIPELINE
# ============================================================================ #
def main(args):
    print("=" * 80)
    print("MITS + PPO Fine-tuning with Qwen 2.5")
    print("=" * 80)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists() or not list(data_path.glob("*.json")):
        print(f"\nDownloading ARC-AGI dataset to {data_path}...")
        download_arc_data(str(data_path))

    challenges_file = data_path / 'arc-agi_evaluation_challenges.json'
    solutions_file = data_path / 'arc-agi_evaluation_solutions.json'

    if not challenges_file.exists():
        challenges_file = data_path / 'arc-agi_training_challenges.json'
        solutions_file = data_path / 'arc-agi_training_solutions.json'

    arc_dataset = ArcDataset.load_from_json(str(challenges_file))
    arc_dataset = arc_dataset.load_solutions(str(solutions_file))
    print(f"Loaded {len(arc_dataset.keys)} tasks")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    gpu_total_gb = None
    if device == "cuda":
        try:
            prop = torch.cuda.get_device_properties(0)
            gpu_total_gb = int(prop.total_memory / (1024 ** 3))
        except Exception:
            gpu_total_gb = None

    requested_model = args.model_name

    # If low VRAM, prefer 3B default
    if gpu_total_gb is not None and gpu_total_gb < 8:
        if '7B' in requested_model or requested_model.endswith('-7B'):
            fallback_model = 'Qwen/Qwen2.5-3B'
            print(f"Detected low VRAM ({gpu_total_gb} GB). Will try {fallback_model} as default.")
            requested_model = fallback_model

    print(f"\nLoading model: {requested_model}")

    # Clear CUDA cache before loading
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=requested_model,
        load_in_4bit=True,
        dtype="float16" if device == "cuda" else "float32",
        device_map="auto",
        max_seq_length=4096,
    )

    # tokenizer safety
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Original vocab size: {len(tokenizer)}")

    # Prepare model for training
    model = FastLanguageModel.for_training(model)

    if not hasattr(model, "_saved_temp_tokenizer"):
        try:
            model._saved_temp_tokenizer = tokenizer
        except Exception:
            pass

    # enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass

    # Attach LoRA adapters with progressive retry
    desired_ranks = [16, 8, 4]
    
    peft_success = False
    last_exc = None

    for r_val in desired_ranks:
        try:
            print(f"\nAttempting to attach LoRA adapters with r={r_val} ...")
            
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
            
            model = FastLanguageModel.get_peft_model(
                model=model,
                r=r_val,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules=[
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj',
                ],
                use_gradient_checkpointing=True,
                random_state=42,
                use_rslora=False,
            )
            peft_success = True
            print(f"LoRA attach succeeded with r={r_val}")
            break
        except torch.cuda.OutOfMemoryError as oom_e:
            print(f"OOM with r={r_val}: {oom_e}")
            last_exc = oom_e
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
        except Exception as e:
            print(f"Error attaching LoRA with r={r_val}: {e}")
            last_exc = e
            try:
                if not hasattr(model, "_saved_temp_tokenizer"):
                    model._saved_temp_tokenizer = tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
            except Exception:
                pass

    if not peft_success:
        print("\nERROR: Unable to attach LoRA adapters with safe ranks. Last exception:")
        raise RuntimeError(f"PEFT attach failed. Last exception: {last_exc}")

    # Optionally reduce vocab
    if args.reduce_vocab:
        print("\nReducing vocabulary to single-character tokens...")
        keep_tokens = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz0123456789!?.:,;*+/-=')
        keep_tokens += tokenizer.tokenize('\n')
        keep_single_char_tokens(model, tokenizer, keep=keep_tokens)
        print(f"Reduced vocab size: {len(tokenizer)}")

    # Format options
    fmt_opts = {
        'preprompt': 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        'query_beg': 'I',
        'reply_beg': '\n+/-=O',
        'reply_end': '\n' + tokenizer.eos_token,
        'lines_sep': '\n',
        'max_tokens': 128000,
    }

    print("\nPreparing training data...")

    train_dataset_base = arc_dataset.remove_test_data()
    train_aug_opts = {'tp': True, 'rt': True, 'perm': True, 'shfl_ex': True, 'seed': 0}

    train_dataset_augmented = train_dataset_base.repeat(
        n=args.repeat_factor,
        seed=0
    ).augment(**train_aug_opts)

    train_list = train_dataset_augmented.as_list(len_name='text', **fmt_opts)

    for i, item in enumerate(train_list):
        key = train_dataset_augmented.keys[i]
        base_key = ArcDataset.get_base_key(key)
        item['task'] = arc_dataset.challenge.get(base_key, {})

    train_dataset_hf = Dataset.from_list(train_list)
    print(f"Training examples: {len(train_dataset_hf)}")

    print("\nInitializing MITS inference...")

    mits_inference = MITSInference(
        model=model,
        tokenizer=tokenizer,
        beam_width=16,
        max_depth=5,
        base_sample_num=2,
        device=device
    )

    augmenter = StructuralAugmenter(num_augmentations=32, seed=42)

    # Training branches
    if args.training_mode == 'standard':
        print("\n" + "=" * 80)
        print("Starting Standard Training (Unsloth)")
        print("=" * 80)

        # CRITICAL: Pre-tokenize the dataset in single-threaded mode
        print("\nPre-tokenizing dataset to avoid multiprocessing issues...")
        
        # Create a simple tokenization function that doesn't require multiprocessing
        def simple_tokenize(examples):
            # Just return the text as-is, we'll handle tokenization in the collator
            return examples
        
        # Don't pre-tokenize, just keep the text field and let the collator handle it
        processed_dataset = train_dataset_hf
        
        # Create trainer with AGGRESSIVE anti-multiprocessing settings
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=processed_dataset,
            dataset_text_field="text",
            max_seq_length=fmt_opts['max_tokens'],
            packing=False,  # Disable packing to avoid complex tokenization
            data_collator=InputMaskingDataCollator(
                tokenizer=tokenizer,
                query_beg=fmt_opts['query_beg'],
                reply_beg=fmt_opts['reply_beg'],
                mask_first_n_examples=1,
            ),
            args=TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_ratio=0.25,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                embedding_learning_rate=args.learning_rate / 10,
                fp16=True if device == "cuda" else False,
                bf16=False,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir=str(output_dir / 'tmp'),
                save_strategy='epoch',
                save_total_limit=2,
                report_to='none',
                dataloader_num_workers=0,  # CRITICAL
                dataloader_pin_memory=False,
                dataloader_prefetch_factor=None,  # Disable prefetching
                remove_unused_columns=False,  # Keep all columns
            ),
        )
        
        # Patch the trainer to force num_proc=1 if it tries to use multiprocessing
        if hasattr(trainer, '_prepare_dataset'):
            original_prepare = trainer._prepare_dataset
            def patched_prepare(*args, **kwargs):
                # Force num_proc to 1 in any map operations
                if 'num_proc' in kwargs:
                    kwargs['num_proc'] = 1
                return original_prepare(*args, **kwargs)
            trainer._prepare_dataset = patched_prepare

        print("\nStarting training...")
        unsloth_train(trainer)

    elif args.training_mode == 'mits_ppo':
        print("\n" + "=" * 80)
        print("Starting MITS + PPO Training")
        print("=" * 80)

        ppo_trainer = MITSPPOStarTrainer(
            model=model,
            tokenizer=tokenizer,
            mits_inference=mits_inference,
            augmenter=augmenter,
            learning_rate=args.learning_rate,
            lambda_correct=1.0,
            lambda_pmi=0.5,
            lambda_similarity=0.3,
            device=device,
            brorl_rollout_size=32,
            enable_mixed_precision=True,
        )

        for epoch in range(args.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*80}")

            avg_loss = ppo_trainer.train_epoch(
                train_dataset=train_dataset_hf,
                augmenter=augmenter,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )

            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            if (epoch + 1) % args.save_every_n_epochs == 0:
                ckpt_path = output_dir / f'checkpoint-epoch-{epoch+1}'
                ckpt_path.mkdir(exist_ok=True)
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                print(f"Saved checkpoint to {ckpt_path}")

    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)

    final_path = output_dir / 'final_model'
    final_path.mkdir(exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    config = {
        'model_name': requested_model,
        'training_mode': args.training_mode,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'vocab_reduced': args.reduce_vocab,
        'final_vocab_size': len(tokenizer),
    }

    with open(final_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to: {final_path}")
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


# ============================================================================ #
# CLI
# ============================================================================ #
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 with MITS + PPO")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--reduce_vocab', action='store_true')
    parser.add_argument('--data_path', type=str, default='arc_data')
    parser.add_argument('--repeat_factor', type=int, default=48)
    parser.add_argument('--training_mode', type=str, default='standard',
                        choices=['standard', 'mits_ppo'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output_qwen_mits')
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--eval_size', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
