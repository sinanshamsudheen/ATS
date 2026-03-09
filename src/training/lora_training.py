"""
LoRA fine-tuning pipeline for ATS Phi model.

Provides importable functions used by ``scripts/train_model.py``.
"""

import json
from typing import Tuple

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def load_configs(
    lora_path: str = "configs/lora_config.yaml",
    train_path: str = "configs/training_config.yaml",
) -> Tuple[dict, dict]:
    """Load LoRA and training YAML configs."""
    with open(lora_path, "r") as f:
        lora_cfg = yaml.safe_load(f)
    with open(train_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    return lora_cfg, train_cfg


def get_quantization_config(train_cfg: dict):
    """Create BitsAndBytes quantization config (or *None*)."""
    if not train_cfg.get("use_4bit", False):
        return None
    compute_dtype = getattr(torch, train_cfg.get("bnb_4bit_compute_dtype", "float16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=train_cfg.get("use_double_quant", True),
    )


# ------------------------------------------------------------------
# Model + LoRA
# ------------------------------------------------------------------

def load_model_and_tokenizer(train_cfg: dict, bnb_config):
    """Load the base causal-LM and tokenizer."""
    model_name = train_cfg["model_name"]
    print(f"Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        model_name = train_cfg.get("fallback_model", "microsoft/phi-2")
        print(f"Primary model failed, using fallback: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    return model, tokenizer, model_name


def apply_lora(model, lora_cfg: dict, bnb_config):
    """Attach LoRA adapters and return the PEFT model."""
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


# ------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------

def tokenize_dataset(data: list, tokenizer, max_seq_length: int = 2048):
    """Tokenise the formatted dataset and return a HF ``Dataset``."""
    texts = [s["text"] for s in data]

    def _tokenize_fn(examples):
        tok = tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(_tokenize_fn, batched=True, remove_columns=["text"])


# ------------------------------------------------------------------
# Training entry point
# ------------------------------------------------------------------

def run_training(
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    output_dir: str | None = None,
):
    """End-to-end training: load → LoRA → tokenise → train → save."""
    print("=" * 60)
    print("ATS Phi Fine-Tuning with LoRA")
    print("=" * 60)

    lora_cfg, train_cfg = load_configs()

    print(f"\nPyTorch {torch.__version__}  |  CUDA {'available' if torch.cuda.is_available() else 'NOT available'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

    bnb_config = get_quantization_config(train_cfg)
    if bnb_config:
        print("4-bit NF4 quantization enabled")

    # 1. Model
    print("\n[1/6] Loading model and tokenizer...")
    model, tokenizer, model_name = load_model_and_tokenizer(train_cfg, bnb_config)

    # 2. LoRA
    print("\n[2/6] Applying LoRA adapters...")
    model = apply_lora(model, lora_cfg, bnb_config)

    # 3. Data
    print("\n[3/6] Loading dataset...")
    t_path = train_data_path or train_cfg["train_dataset"]
    v_path = val_data_path or train_cfg["validation_dataset"]
    with open(t_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(v_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    print(f"  Train: {len(train_data)}  |  Validation: {len(val_data)}")

    # 4. Tokenise
    print("\n[4/6] Tokenizing...")
    max_sl = train_cfg.get("max_seq_length", 2048)
    train_ds = tokenize_dataset(train_data, tokenizer, max_sl)
    val_ds = tokenize_dataset(val_data, tokenizer, max_sl)

    # 5. Trainer
    print("\n[5/6] Configuring trainer...")
    out = output_dir or train_cfg["output_dir"]
    training_args = TrainingArguments(
        output_dir=out,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_steps=train_cfg.get("eval_steps", 50),
        save_steps=train_cfg.get("save_steps", 100),
        optim=train_cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        logging_dir=train_cfg.get("logging_dir", f"{out}/logs"),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 6. Train
    print("\n[6/6] Starting training...")
    print(f"  Epochs: {training_args.num_train_epochs}  |  Effective batch: "
          f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    trainer.train()

    print("\nSaving LoRA adapter weights...")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results['eval_loss']:.4f}")
    print(f"\nTraining complete — adapter saved to {out}")
