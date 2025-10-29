# run_qwen2p5_mits_ppo.py
"""
Run Qwen-2.5 (7B) + MITS + PPO training using structural augmentations.
Optimized for free GPUs (Kaggle / Colab) with safe auto-checkpoint & resume.
"""

import os
import argparse
import time
import gc
from pathlib import Path
import torch
from datasets import Dataset

# === project imports ===
from finetunning_module import (
    FastLanguageModel,
    MITSPPOStarTrainer,
)
from arc_loader import ArcDataset
from arc_downloader import download_arc_data
from arc_agi_augumentations import StructuralAugmenter
from mits_inference import MITSInference


# -------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--output_dir', type=str, default='qwen_mits_output')
    parser.add_argument('--data_path', type=str, default='arc_data')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--repeat_factor', type=int, default=12)
    parser.add_argument('--use_4bit', action='store_true')
    parser.add_argument('--checkpoint_interval_minutes', type=int, default=15)
    return parser.parse_args()


# -------------------------------------------------------------
# Main Training Routine
# -------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # 1) Download and prepare ARC dataset
    print("📦 Checking dataset...")
    download_arc_data(str(data_path))

    challenges_file = data_path / 'arc-agi_training_challenges.json'
    solutions_file = data_path / 'arc-agi_training_solutions.json'
    if not challenges_file.exists():
        challenges_file = data_path / 'arc-agi_evaluation_challenges.json'
        solutions_file = data_path / 'arc-agi_evaluation_solutions.json'

    arc_dataset = ArcDataset.load_from_json(str(challenges_file))
    arc_dataset = arc_dataset.load_solutions(str(solutions_file))
    print(f"✅ Loaded {len(arc_dataset.keys)} ARC tasks")

    # 2) Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # 3) Resume or load base model
    resume_path = output_dir / "last_checkpoint"
    if resume_path.exists():
        print(f"🔁 Resuming from {resume_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(resume_path),
            load_in_4bit=args.use_4bit,
            dtype="float16" if device == 'cuda' else "float32",
            device_map="auto",
        )
    else:
        print("🚀 Loading base model:", args.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=args.use_4bit,
            dtype="float16" if device == 'cuda' else "float32",
            device_map="auto",
            max_seq_length=4096,
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = FastLanguageModel.for_training(model)
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass

    # 4) Attach LoRA/PEFT
    desired_ranks = [32, 16, 8]
    peft_success, last_exc = False, None
    for r in desired_ranks:
        try:
            print(f"🧩 Attaching LoRA rank={r} ...")
            model = FastLanguageModel.get_peft_model(
                model=model,
                r=r,
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
            break
        except Exception as e:
            print(f"❌ LoRA rank={r} failed: {e}")
            last_exc = e
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
    if not peft_success:
        raise RuntimeError(f"LoRA attach failed: {last_exc}")

    # 5) Build MITS + Augmenter
    mits_inference = MITSInference(
        model=model,
        tokenizer=tokenizer,
        beam_width=12,
        max_depth=4,
        base_sample_num=2,
        device=device,
    )
    augmenter = StructuralAugmenter(num_augmentations=32, seed=42)

    # 6) Prepare training dataset
    fmt_opts = {
        'preprompt': 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        'query_beg': 'I',
        'reply_beg': '\n+/-=O',
        'reply_end': '\n' + tokenizer.eos_token,
        'lines_sep': '\n',
        'max_tokens': 128000,
    }

    train_dataset_base = arc_dataset.remove_test_data()
    train_dataset_aug = train_dataset_base.repeat(
        n=args.repeat_factor, seed=0
    ).augment(tp=True, rt=True, perm=True, shfl_ex=True)
    train_list = train_dataset_aug.as_list(len_name='text', **fmt_opts)
    for i, item in enumerate(train_list):
        key = train_dataset_aug.keys[i]
        base_key = ArcDataset.get_base_key(key)
        item['task'] = arc_dataset.challenge.get(base_key, {})
    train_dataset_hf = Dataset.from_list(train_list)
    print(f"📚 Training examples: {len(train_dataset_hf)}")

    # 7) Instantiate PPO-Star trainer
    ppo_trainer = MITSPPOStarTrainer(
        model=model,
        tokenizer=tokenizer,
        mits_inference=mits_inference,
        augmenter=augmenter,
        device=device,
        learning_rate=1e-4,
        lambda_correct=1.0,
        lambda_pmi=0.5,
        lambda_similarity=0.3,
    )

    # -------------------------------------------------------------
    # 8) Training Loop with Auto-Checkpoint and Resume Safety
    # -------------------------------------------------------------
    start_time = time.time()
    last_checkpoint_time = start_time

    for epoch in range(args.epochs):
        print(f"\n=== 🏁 EPOCH {epoch+1}/{args.epochs} ===")
        avg_loss = ppo_trainer.train_epoch(
            train_dataset=train_dataset_hf,
            augmenter=augmenter,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save after each epoch
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"💾 Saved epoch checkpoint: {ckpt_dir}")

        # Auto-save every 2 epochs
        if (epoch + 1) % 2 == 0:
            auto_ckpt_dir = output_dir / f"auto_checkpoint_epoch_{epoch+1}"
            auto_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(auto_ckpt_dir))
            tokenizer.save_pretrained(str(auto_ckpt_dir))
            print(f"🧠 Auto-checkpoint (every 2 epochs): {auto_ckpt_dir}")

        # Time-based quick-save
        now = time.time()
        if (now - last_checkpoint_time) / 60.0 >= args.checkpoint_interval_minutes:
            quick_path = output_dir / "last_checkpoint"
            quick_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(quick_path))
            tokenizer.save_pretrained(str(quick_path))
            print(f"⚡ Quick checkpoint saved to {quick_path}")
            last_checkpoint_time = now

    # Final save
    final_dir = output_dir / "final_model"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("✅ Training complete. Final model saved to:", final_dir)


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
