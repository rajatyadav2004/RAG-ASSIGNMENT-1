"""
finetune.py
===========
Fine-tuning a Small LLM using LoRA (Parameter-Efficient Fine-Tuning)
for Technical Question Answering.

Assignment 2 — Generative AI & LLMs Course Project

Features:
  - Loads 600 Q&A pairs from dataset.json
  - Formats data as instruction-following prompts
  - Fine-tunes GPT-2 (small, 117M params) using LoRA via PEFT
  - Evaluates on a held-out test split using BLEU and ROUGE
  - Compares base model vs fine-tuned model on sample questions
  - Saves the fine-tuned adapter to ./finetuned_model/
"""

import os
import json
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ─── 1. IMPORTS ────────────────────────────────────────────────────────────────
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download NLTK data for BLEU computation
nltk.download("punkt", quiet=True)

# ─── 2. CONFIGURATION ──────────────────────────────────────────────────────────
BASE_MODEL        = "gpt2"             # Base LLM (117M params, CPU-friendly)
DATASET_FILE      = "dataset.json"     # Q&A training data
OUTPUT_DIR        = "./finetuned_model" # Where to save LoRA adapter
MAX_SEQ_LENGTH    = 256                # Max token length per sample
TRAIN_RATIO       = 0.85               # 85% train, 15% test
NUM_EPOCHS        = 3                  # Training epochs
BATCH_SIZE        = 4                  # Per-device batch size
LEARNING_RATE     = 2e-4               # AdamW learning rate
LORA_R            = 8                  # LoRA rank (r)
LORA_ALPHA        = 32                 # LoRA scaling factor
LORA_DROPOUT      = 0.1               # LoRA dropout rate
SEED              = 42                 # Reproducibility seed

torch.manual_seed(SEED)


# ─── 3. DATASET LOADING ────────────────────────────────────────────────────────
def load_dataset(filepath: str) -> tuple[list, list]:
    """
    Load Q&A pairs from JSON and split into train/test sets.

    Returns:
        (train_samples, test_samples)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle deterministically
    np.random.seed(SEED)
    np.random.shuffle(data)

    split_idx = int(len(data) * TRAIN_RATIO)
    train = data[:split_idx]
    test  = data[split_idx:]

    print(f"[Dataset] Total: {len(data)} | Train: {len(train)} | Test: {len(test)}")
    return train, test


# ─── 4. PROMPT FORMATTING ──────────────────────────────────────────────────────
def format_prompt(sample: dict) -> str:
    """
    Format a Q&A pair as an instruction-following prompt.

    Expected JSON keys: "input" (question), "output" (answer)
    """
    return (
        f"### Question:\n{sample['input']}\n\n"
        f"### Answer:\n{sample['output']}\n"
        f"<|endoftext|>"
    )


def prepare_hf_dataset(samples: list) -> Dataset:
    """Convert list of dicts to a HuggingFace Dataset with formatted prompts."""
    formatted = [{"text": format_prompt(s)} for s in samples]
    return Dataset.from_list(formatted)


# ─── 5. MODEL & TOKENIZER SETUP ────────────────────────────────────────────────
def load_base_model(model_name: str):
    """Load the base LLM and its tokenizer."""
    print(f"[Model] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False  # Required for gradient checkpointing
    return model, tokenizer


# ─── 6. LORA CONFIGURATION ─────────────────────────────────────────────────────
def apply_lora(model):
    """
    Apply LoRA adapters to the model's attention layers.

    LoRA injects trainable rank-decomposition matrices (A, B) into Q and V
    projection layers, reducing trainable parameters dramatically.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,                           # Rank of update matrices
        lora_alpha=LORA_ALPHA,              # Scaling factor (alpha/r = scaling)
        target_modules=["c_attn"],          # GPT-2 combined Q,K,V projection
        lora_dropout=LORA_DROPOUT,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── 7. TRAINING ───────────────────────────────────────────────────────────────
def train_model(model, tokenizer, train_dataset: Dataset):
    """Configure and run the SFTTrainer for LoRA fine-tuning."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,     # Simulate larger batch size
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,                         # Disable for CPU compatibility
        bf16=False,
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",                   # Disable W&B etc.
        optim="adamw_torch"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    print("\n[Training] Starting LoRA fine-tuning...")
    start_time = __import__("time").time()
    trainer.train()
    elapsed = __import__("time").time() - start_time
    print(f"[Training] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save LoRA adapter (not full model)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[Training] LoRA adapter saved to '{OUTPUT_DIR}'")

    return trainer


# ─── 8. INFERENCE ──────────────────────────────────────────────────────────────
def generate_answer(question: str, model, tokenizer, max_new_tokens: int = 100) -> str:
    """Generate an answer for a given question using the current model."""
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("### Answer:")[-1].strip()
    return answer


# ─── 9. EVALUATION METRICS ─────────────────────────────────────────────────────
def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score."""
    ref_tokens  = nltk.word_tokenize(reference.lower())
    hyp_tokens  = nltk.word_tokenize(hypothesis.lower())
    smoother    = SmoothingFunction().method1
    score       = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
    return round(score, 4)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def evaluate_model(model, tokenizer, test_samples: list) -> dict:
    """
    Evaluate the model on test samples using BLEU and ROUGE.

    Returns:
        dict with average BLEU and ROUGE scores
    """
    model.eval()
    bleu_scores   = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    print(f"\n[Eval] Evaluating on {len(test_samples)} test samples...")
    for i, sample in enumerate(test_samples[:50]):  # Limit for speed
        ref  = sample["output"]
        hyp  = generate_answer(sample["input"], model, tokenizer)

        bleu   = compute_bleu(ref, hyp)
        rouge  = compute_rouge(ref, hyp)

        bleu_scores.append(bleu)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])

        if i < 3:  # Print first 3 for inspection
            print(f"\n  Sample {i+1}:")
            print(f"    Q  : {sample['input'][:80]}...")
            print(f"    Ref: {ref[:80]}...")
            print(f"    Hyp: {hyp[:80]}...")
            print(f"    BLEU={bleu:.4f} | ROUGE-1={rouge['rouge1']:.4f} | ROUGE-L={rouge['rougeL']:.4f}")

    metrics = {
        "avg_bleu"  : round(np.mean(bleu_scores), 4),
        "avg_rouge1": round(np.mean(rouge1_scores), 4),
        "avg_rouge2": round(np.mean(rouge2_scores), 4),
        "avg_rougeL": round(np.mean(rougeL_scores), 4),
    }
    print(f"\n[Eval] Results: {metrics}")
    return metrics


# ─── 10. BASE vs FINE-TUNED COMPARISON ────────────────────────────────────────
def compare_base_vs_finetuned(
    base_model, finetuned_model, tokenizer, questions: list[str]
):
    """Side-by-side comparison of base vs fine-tuned model outputs."""
    print("\n" + "="*70)
    print("Comparison: Base GPT-2 vs LoRA Fine-Tuned GPT-2")
    print("="*70)

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 50)

        base_ans = generate_answer(q, base_model, tokenizer, max_new_tokens=80)
        print(f"Base Model    : {base_ans[:200]}")

        ft_ans = generate_answer(q, finetuned_model, tokenizer, max_new_tokens=80)
        print(f"Fine-Tuned    : {ft_ans[:200]}")


# ─── 11. MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  LoRA Fine-Tuning — Technical QA Language Model")
    print("="*70 + "\n")

    # Load dataset
    train_samples, test_samples = load_dataset(DATASET_FILE)

    # Prepare HuggingFace dataset
    train_hf = prepare_hf_dataset(train_samples)

    # Load base model
    base_model, tokenizer = load_base_model(BASE_MODEL)

    # ── Evaluate BASE model before fine-tuning ─────────────────────────────────
    print("\n[Phase 1] Evaluating BASE model (before fine-tuning)...")
    base_metrics = evaluate_model(base_model, tokenizer, test_samples)

    # ── Apply LoRA and fine-tune ───────────────────────────────────────────────
    print("\n[Phase 2] Applying LoRA adapters...")
    lora_model = apply_lora(base_model)

    trainer = train_model(lora_model, tokenizer, train_hf)

    # ── Evaluate FINE-TUNED model ──────────────────────────────────────────────
    print("\n[Phase 3] Evaluating FINE-TUNED model (after LoRA training)...")
    ft_metrics = evaluate_model(lora_model, tokenizer, test_samples)

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Metric':<15} {'Base Model':>12} {'Fine-Tuned':>12} {'Improvement':>12}")
    print("-"*60)
    for key in ["avg_bleu", "avg_rouge1", "avg_rouge2", "avg_rougeL"]:
        base_val = base_metrics[key]
        ft_val   = ft_metrics[key]
        delta    = ft_val - base_val
        sign     = "+" if delta >= 0 else ""
        print(f"  {key:<13} {base_val:>12.4f} {ft_val:>12.4f} {sign}{delta:>11.4f}")
    print("="*60)

    # ── Side-by-side output comparison ────────────────────────────────────────
    comparison_questions = [
        "What is virtual memory and why is it used?",
        "Explain LoRA fine-tuning in simple terms.",
        "What is the ACID property of databases?",
    ]
    compare_base_vs_finetuned(base_model, lora_model, tokenizer, comparison_questions)

    print("\n[Done] Fine-tuning pipeline completed successfully.")
    print(f"       LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
