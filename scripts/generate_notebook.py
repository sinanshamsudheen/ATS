"""Generate a clean ATS fine-tuning notebook, replacing the broken 107-cell version."""
import json, uuid, sys
from pathlib import Path

def uid():
    return uuid.uuid4().hex[:12]

def md(src: str):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src: str):
    return {
        "cell_type": "code", "id": uid(), "metadata": {},
        "source": src, "outputs": [], "execution_count": None,
    }

cells = []

# ─── Phase 1: Environment Setup ──────────────────────────────────────────────
cells.append(md(
    "# ATS Fine-Tuning Pipeline — Phi-3-mini × QLoRA\n"
    "\n"
    "**6 phases, end-to-end on Colab T4 GPU:**\n"
    "1. Environment Setup\n"
    "2. Dataset Preparation\n"
    "3. Model Loading & LoRA\n"
    "4. Fine-Tuning (response-only loss via `DataCollatorForCompletionOnlyLM`)\n"
    "5. Inference Testing\n"
    "6. Evaluation Metrics\n"
    "\n"
    "**Key design decisions:**\n"
    "- Phi-3 native chat template (`<|user|>` / `<|assistant|>`) — not Alpaca format\n"
    "- `DataCollatorForCompletionOnlyLM` for prompt masking — avoids BPE context bugs\n"
    "- EOS token set to `<|end|>` (Phi-3 native, ID 32007)\n"
    "- `Phi3ForCausalLM` directly — no `trust_remote_code` (fixes DynamicCache error)\n"
    "- `paged_adamw_32bit` optimizer, cosine LR, epoch-based eval"
))

cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "import os\n"
    "os.chdir('/content/drive/MyDrive/colab')\n"
    "print('Working directory:', os.getcwd())"
))

cells.append(code(
    "# trl is added for DataCollatorForCompletionOnlyLM (response-only masking)\n"
    "!pip install -q \\\n"
    "    transformers>=4.41.0 \\\n"
    "    peft>=0.7.0 \\\n"
    "    datasets>=2.16.0 \\\n"
    "    accelerate>=0.25.0 \\\n"
    "    bitsandbytes>=0.41.0 \\\n"
    "    trl>=0.8.0 \\\n"
    "    scipy \\\n"
    "    pyyaml \\\n"
    "    json-repair"
))

cells.append(code(
    "import torch\n"
    "\n"
    "if not torch.cuda.is_available():\n"
    "    raise RuntimeError('No GPU detected — set Runtime → Change runtime type → T4 GPU')\n"
    "\n"
    "print(f'GPU:  {torch.cuda.get_device_name(0)}')\n"
    "print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n"
    "print(f'CUDA: {torch.version.cuda}')"
))

# ─── Phase 2: Dataset Preparation ────────────────────────────────────────────
cells.append(md(
    "## Phase 2 — Dataset Preparation\n"
    "\n"
    "- Load YAML configs\n"
    "- Validate raw dataset (checks `weak_bullets` key — the correct field name)\n"
    "- Format using **Phi-3 native chat template** (not Alpaca `### Instruction:` format)\n"
    "- Split 90/10 and save"
))

cells.append(code(
    "import yaml, json, random\n"
    "from pathlib import Path\n"
    "\n"
    "with open('configs/training_config.yaml') as f:\n"
    "    train_cfg = yaml.safe_load(f)\n"
    "with open('configs/lora_config.yaml') as f:\n"
    "    lora_cfg = yaml.safe_load(f)\n"
    "\n"
    "print('Training config loaded:')\n"
    "for k, v in train_cfg.items():\n"
    "    print(f'  {k}: {v}')"
))

cells.append(code(
    "# Validate every sample in raw_dataset.json\n"
    "# Uses weak_bullets (the correct key from the training data)\n"
    "REQUIRED_KEYS_TOP    = {'instruction', 'input', 'output'}\n"
    "REQUIRED_KEYS_OUTPUT = {\n"
    "    'ats_score', 'score_breakdown', 'matched_skills',\n"
    "    'missing_skills', 'weak_bullets', 'formatting_issues', 'overall_feedback'\n"
    "}\n"
    "\n"
    "raw_path = Path(train_cfg['raw_dataset'])\n"
    "with open(raw_path) as f:\n"
    "    raw_data = json.load(f)\n"
    "\n"
    "errors, scores = [], []\n"
    "for i, s in enumerate(raw_data):\n"
    "    missing = REQUIRED_KEYS_TOP - set(s)\n"
    "    if missing:\n"
    "        errors.append(f'Sample {i}: missing top-level keys {missing}')\n"
    "        continue\n"
    "    try:\n"
    "        out = json.loads(s['output'])\n"
    "        missing_out = REQUIRED_KEYS_OUTPUT - set(out)\n"
    "        if missing_out:\n"
    "            errors.append(f'Sample {i}: missing output keys {missing_out}')\n"
    "        else:\n"
    "            scores.append(out['ats_score'])\n"
    "    except json.JSONDecodeError as e:\n"
    "        errors.append(f'Sample {i}: bad JSON — {e}')\n"
    "\n"
    "if errors:\n"
    "    print(f'ERRORS ({len(errors)}):')\n"
    "    for e in errors[:10]: print(f'  {e}')\n"
    "else:\n"
    "    print(f'All {len(raw_data)} samples valid')\n"
    "    print(f'ATS score — Min: {min(scores)}, Max: {max(scores)}, Mean: {sum(scores)/len(scores):.1f}')"
))

cells.append(code(
    "from transformers import AutoTokenizer\n"
    "\n"
    "model_name = train_cfg['model_name']\n"
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
    "\n"
    "# Phi-3 uses <|end|> (ID 32007) as end-of-turn — NOT <|endoftext|> (ID 0)\n"
    "# Training sequences terminated with wrong EOS cause the model to never learn\n"
    "# to stop generation cleanly.\n"
    "phi3_eos = '<|end|>'\n"
    "tokenizer.eos_token = phi3_eos\n"
    "\n"
    "# <|end|> also serves as pad token (padding_side=right for causal LM)\n"
    "tokenizer.pad_token = tokenizer.eos_token\n"
    "tokenizer.padding_side = 'right'\n"
    "\n"
    "print(f'EOS: \"{tokenizer.eos_token}\" (ID {tokenizer.eos_token_id})')\n"
    "print(f'PAD: \"{tokenizer.pad_token}\" (ID {tokenizer.pad_token_id})')\n"
    "print(f'Vocab size: {tokenizer.vocab_size}')\n"
    "\n"
    "# Sanity: ensure EOS is the Phi-3 end-of-turn token\n"
    "assert tokenizer.eos_token_id == 32007, (\n"
    "    f'Expected EOS ID 32007 (<|end|>), got {tokenizer.eos_token_id}. '\n"
    "    'Update the phi3_eos string above.'\n"
    ")"
))

cells.append(code(
    "# Format samples using Phi-3 NATIVE chat template:\n"
    "#   <|user|>\\n{instruction}\\n\\n{input}<|end|>\\n<|assistant|>\\n{output}<|end|>\n"
    "#\n"
    "# DataCollatorForCompletionOnlyLM will mask everything before <|assistant|>.\n"
    "# <|user|>, <|assistant|>, <|end|> are SINGLE special tokens in Phi-3's vocab,\n"
    "# so they always get the same ID regardless of context — no BPE ambiguity.\n"
    "\n"
    "def format_sample(s):\n"
    "    return (\n"
    "        f\"<|user|>\\n{s['instruction']}\\n\\n{s['input']}<|end|>\\n\"\n"
    "        f\"<|assistant|>\\n{s['output']}<|end|>\"\n"
    "    )\n"
    "\n"
    "formatted = [format_sample(s) for s in raw_data]\n"
    "lengths   = [len(tokenizer.encode(t)) for t in formatted]\n"
    "max_len   = train_cfg['max_seq_length']\n"
    "\n"
    "print(f'Token lengths — Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}')\n"
    "print(f'Samples > {max_len} tokens (will be truncated): {sum(1 for l in lengths if l > max_len)}')\n"
    "print()\n"
    "print('Sample (first 600 chars):')\n"
    "print(formatted[0][:600])"
))

cells.append(code(
    "# 90 / 10 train / val split — deterministic via seed\n"
    "random.seed(train_cfg['seed'])\n"
    "indices = list(range(len(raw_data)))\n"
    "random.shuffle(indices)\n"
    "\n"
    "n_train = int(len(indices) * train_cfg['train_split'])\n"
    "train_idx, val_idx = indices[:n_train], indices[n_train:]\n"
    "\n"
    "# Pre-formatted text datasets (used for training)\n"
    "train_data = [{'text': formatted[i]} for i in train_idx]\n"
    "val_data   = [{'text': formatted[i]} for i in val_idx]\n"
    "\n"
    "# Original samples (used for evaluation in Phase 6)\n"
    "train_raw_split = [raw_data[i] for i in train_idx]\n"
    "val_raw_split   = [raw_data[i] for i in val_idx]\n"
    "\n"
    "Path('data').mkdir(exist_ok=True)\n"
    "with open('data/train.json', 'w') as f:\n"
    "    json.dump(train_raw_split, f, indent=2)\n"
    "with open('data/validation.json', 'w') as f:\n"
    "    json.dump(val_raw_split, f, indent=2)\n"
    "\n"
    "print(f'Train: {len(train_data)} samples')\n"
    "print(f'Val:   {len(val_data)} samples')\n"
    "print('Splits saved to data/train.json and data/validation.json')"
))

# ─── Phase 3: Model Loading & LoRA ───────────────────────────────────────────
cells.append(md(
    "## Phase 3 — Model Loading & LoRA\n"
    "\n"
    "- Load Phi-3 with 4-bit NF4 quantisation (QLoRA)\n"
    "- `Phi3ForCausalLM` directly — no `trust_remote_code` (avoids DynamicCache KeyError)\n"
    "- No `low_cpu_mem_usage=True` — causes PEFT key-resolution failures for tied weights\n"
    "- Apply LoRA with rank 16, alpha 32, targeting q/k/v/o projections"
))

cells.append(code(
    "import torch\n"
    "from transformers import BitsAndBytesConfig, Phi3ForCausalLM\n"
    "\n"
    "bnb_config = BitsAndBytesConfig(\n"
    "    load_in_4bit=train_cfg['use_4bit'],\n"
    "    bnb_4bit_quant_type=train_cfg['bnb_4bit_quant_type'],\n"
    "    bnb_4bit_compute_dtype=torch.float16,\n"
    "    bnb_4bit_use_double_quant=train_cfg['use_double_quant'],\n"
    ")\n"
    "\n"
    "model = Phi3ForCausalLM.from_pretrained(\n"
    "    model_name,\n"
    "    quantization_config=bnb_config,\n"
    "    device_map='auto',\n"
    "    torch_dtype=torch.float16,\n"
    "    # trust_remote_code=True is intentionally omitted — transformers>=4.41.0\n"
    "    # ships native Phi3ForCausalLM; using it avoids the stale cached\n"
    "    # modeling_phi3.py that triggers DynamicCache.seen_tokens AttributeError.\n"
    "    # low_cpu_mem_usage=True is also intentionally omitted — it causes PEFT\n"
    "    # to fail resolving base_model.model.lm_head for tied-weight models.\n"
    ")\n"
    "model.config.use_cache = False  # required when gradient_checkpointing=True\n"
    "\n"
    "total_params = sum(p.numel() for p in model.parameters())\n"
    "print(f'Model: {model.__class__.__name__}')\n"
    "print(f'Total parameters: {total_params:,}')"
))

cells.append(code(
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
    "\n"
    "model = prepare_model_for_kbit_training(model)\n"
    "\n"
    "lora_config = LoraConfig(\n"
    "    r=lora_cfg['r'],\n"
    "    lora_alpha=lora_cfg['lora_alpha'],\n"
    "    lora_dropout=lora_cfg['lora_dropout'],\n"
    "    target_modules=lora_cfg['target_modules'],\n"
    "    bias=lora_cfg['bias'],\n"
    "    task_type=lora_cfg['task_type'],\n"
    "    inference_mode=False,\n"
    ")\n"
    "\n"
    "model = get_peft_model(model, lora_config)\n"
    "model.print_trainable_parameters()\n"
    "\n"
    "model.gradient_checkpointing_enable()"
))

# ─── Phase 4: Fine-Tuning ─────────────────────────────────────────────────────
cells.append(md(
    "## Phase 4 — Fine-Tuning\n"
    "\n"
    "**Critical fix over original notebook:** replaced the hand-coded token-ID search\n"
    "with `DataCollatorForCompletionOnlyLM` from TRL.\n"
    "\n"
    "The original code searched for `'### Response:\\n'` token IDs in the tokenized\n"
    "sequence, but BPE tokenizes the same characters differently depending on surrounding\n"
    "context. This caused *every* token to be masked (`-100`), giving `train_loss=0.0`\n"
    "and `eval_loss=nan` — the model received zero gradient updates.\n"
    "\n"
    "`DataCollatorForCompletionOnlyLM` with `response_template='<|assistant|>'` works\n"
    "reliably because `<|assistant|>` is a **single special token** in Phi-3's vocabulary\n"
    "(ID 32001), so it always tokenizes to the same ID regardless of surrounding context."
))

cells.append(code(
    "from datasets import Dataset\n"
    "\n"
    "train_dataset = Dataset.from_list(train_data)\n"
    "val_dataset   = Dataset.from_list(val_data)\n"
    "\n"
    "def tokenize(batch):\n"
    "    return tokenizer(\n"
    "        batch['text'],\n"
    "        truncation=True,\n"
    "        max_length=train_cfg['max_seq_length'],\n"
    "        padding=False,  # DataCollator pads per batch\n"
    "    )\n"
    "\n"
    "train_tok = train_dataset.map(tokenize, batched=True, remove_columns=['text'])\n"
    "val_tok   = val_dataset.map(tokenize,   batched=True, remove_columns=['text'])\n"
    "\n"
    "print(f'Tokenized — train: {len(train_tok)}, val: {len(val_tok)}')"
))

cells.append(code(
    "from trl import DataCollatorForCompletionOnlyLM\n"
    "\n"
    "# <|assistant|> is a single special token (ID 32001) — context-independent tokenization\n"
    "collator = DataCollatorForCompletionOnlyLM(\n"
    "    response_template='<|assistant|>',\n"
    "    tokenizer=tokenizer,\n"
    "    mlm=False,\n"
    ")\n"
    "\n"
    "# ── Masking sanity check — must show >0 trained tokens ──────────────────────\n"
    "sample_batch = collator([train_tok[0]])\n"
    "labels = sample_batch['labels'][0].tolist()\n"
    "n_total   = len(labels)\n"
    "n_masked  = labels.count(-100)\n"
    "n_trained = n_total - n_masked\n"
    "\n"
    "print(f'Masking check on sample 0:')\n"
    "print(f'  Total tokens:    {n_total}')\n"
    "print(f'  Masked (-100):   {n_masked}')\n"
    "print(f'  Trained (loss):  {n_trained}')\n"
    "\n"
    "assert n_trained > 0, (\n"
    "    'All tokens masked — response template not found! '\n"
    "    'Check tokenizer and format_sample().'\n"
    ")\n"
    "print('  Masking is correct')"
))

cells.append(code(
    "from transformers import TrainingArguments, Trainer\n"
    "\n"
    "# ── Step count sanity ────────────────────────────────────────────────────────\n"
    "eff_batch       = train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']\n"
    "steps_per_epoch = max(1, len(train_tok) // eff_batch)\n"
    "total_steps     = steps_per_epoch * train_cfg['num_train_epochs']\n"
    "warmup_steps    = max(1, int(total_steps * 0.03))\n"
    "print(f'Effective batch:  {eff_batch}')\n"
    "print(f'Steps/epoch:      {steps_per_epoch}')\n"
    "print(f'Total steps:      {total_steps}')\n"
    "print(f'Warmup steps:     {warmup_steps} (3%)')\n"
    "\n"
    "training_args = TrainingArguments(\n"
    "    output_dir=train_cfg['output_dir'],\n"
    "    num_train_epochs=train_cfg['num_train_epochs'],\n"
    "    per_device_train_batch_size=train_cfg['per_device_train_batch_size'],\n"
    "    per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],\n"
    "    gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],\n"
    "    learning_rate=train_cfg['learning_rate'],\n"
    "    # warmup_ratio instead of warmup_steps — safe when total_steps < configured steps\n"
    "    warmup_ratio=0.03,\n"
    "    lr_scheduler_type='cosine',   # cosine > linear for small datasets\n"
    "    optim='paged_adamw_32bit',    # lower peak VRAM than adamw_torch on T4\n"
    "    fp16=train_cfg['fp16'],\n"
    "    bf16=train_cfg['bf16'],\n"
    "    gradient_checkpointing=train_cfg['gradient_checkpointing'],\n"
    "    # epoch-based eval/save — safe regardless of total step count\n"
    "    evaluation_strategy='epoch',\n"
    "    save_strategy='epoch',\n"
    "    load_best_model_at_end=True,\n"
    "    metric_for_best_model='eval_loss',\n"
    "    logging_steps=1,\n"
    "    save_total_limit=train_cfg['save_total_limit'],\n"
    "    max_grad_norm=1.0,\n"
    "    seed=train_cfg['seed'],\n"
    "    report_to='none',\n"
    ")\n"
    "\n"
    "trainer = Trainer(\n"
    "    model=model,\n"
    "    args=training_args,\n"
    "    train_dataset=train_tok,\n"
    "    eval_dataset=val_tok,\n"
    "    data_collator=collator,\n"
    ")"
))

cells.append(code(
    "print('Starting fine-tuning...')\n"
    "trainer.train()\n"
    "print('Training complete!')\n"
    "\n"
    "# Verify loss is non-zero and finite\n"
    "history = trainer.state.log_history\n"
    "train_losses = [e['loss'] for e in history if 'loss' in e]\n"
    "if train_losses:\n"
    "    print(f'Train loss: initial={train_losses[0]:.4f}, final={train_losses[-1]:.4f}')\n"
    "    assert train_losses[-1] > 0 and train_losses[-1] == train_losses[-1], \\\n"
    "        'ERROR: train_loss is 0 or NaN — label masking failed'\n"
    "else:\n"
    "    print('WARNING: No loss values found in training history')"
))

cells.append(code(
    "# Save LoRA adapter\n"
    "adapter_dir = 'ats_phi_lora'\n"
    "trainer.model.save_pretrained(adapter_dir)\n"
    "tokenizer.save_pretrained(adapter_dir)\n"
    "print(f'LoRA adapter saved to ./{adapter_dir}/')\n"
    "\n"
    "# Phi-3 gotcha: ensure_weight_tying must be false in adapter_config.json.\n"
    "# If true, PEFT tries to resolve base_model.model.model.embed_tokens, which\n"
    "# doesn't exist in the checkpoint and raises a KeyError at merge time.\n"
    "import json as _json\n"
    "cfg_path = f'{adapter_dir}/adapter_config.json'\n"
    "with open(cfg_path) as f:\n"
    "    adapter_cfg = _json.load(f)\n"
    "if adapter_cfg.get('ensure_weight_tying', False):\n"
    "    adapter_cfg['ensure_weight_tying'] = False\n"
    "    with open(cfg_path, 'w') as f:\n"
    "        _json.dump(adapter_cfg, f, indent=2)\n"
    "    print('Fixed adapter_config.json: ensure_weight_tying set to false')\n"
    "else:\n"
    "    print('adapter_config.json: ensure_weight_tying already false — OK')"
))

# ─── Phase 5: Inference Testing ───────────────────────────────────────────────
cells.append(md(
    "## Phase 5 — Inference Testing\n"
    "\n"
    "Quick smoke test: run 2 samples from the validation set through the fine-tuned\n"
    "model and check that the output is parseable JSON with the expected keys."
))

cells.append(code(
    "import torch\n"
    "\n"
    "def generate_from_sample(sample, model, tokenizer, max_new_tokens=1024):\n"
    "    \"\"\"Generate ATS evaluation for a single instruction/input sample.\"\"\"\n"
    "    prompt = (\n"
    "        f\"<|user|>\\n{sample['instruction']}\\n\\n{sample['input']}<|end|>\\n\"\n"
    "        f\"<|assistant|>\\n\"\n"
    "    )\n"
    "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)\n"
    "    with torch.no_grad():\n"
    "        output_ids = model.generate(\n"
    "            **inputs,\n"
    "            max_new_tokens=max_new_tokens,\n"
    "            do_sample=False,\n"
    "            eos_token_id=tokenizer.eos_token_id,   # <|end|>\n"
    "            pad_token_id=tokenizer.pad_token_id,\n"
    "        )\n"
    "    n_prompt = inputs['input_ids'].shape[1]\n"
    "    return tokenizer.decode(output_ids[0][n_prompt:], skip_special_tokens=True).strip()\n"
    "\n"
    "\n"
    "def extract_json(text):\n"
    "    \"\"\"Multi-tier JSON extraction: direct → regex → json-repair.\"\"\"\n"
    "    import re\n"
    "    from json_repair import repair_json\n"
    "    # Tier 1: direct\n"
    "    try:\n"
    "        return json.loads(text), 'direct'\n"
    "    except json.JSONDecodeError:\n"
    "        pass\n"
    "    # Tier 2: extract outermost JSON object\n"
    "    m = re.search(r'\\{.*\\}', text, re.DOTALL)\n"
    "    if m:\n"
    "        try:\n"
    "            return json.loads(m.group(0)), 'regex'\n"
    "        except json.JSONDecodeError:\n"
    "            pass\n"
    "    # Tier 3: repair and parse\n"
    "    try:\n"
    "        return json.loads(repair_json(text)), 'repaired'\n"
    "    except Exception:\n"
    "        pass\n"
    "    return None, 'failed'"
))

cells.append(code(
    "REQUIRED_OUTPUT_KEYS = {\n"
    "    'ats_score', 'score_breakdown', 'matched_skills',\n"
    "    'missing_skills', 'weak_bullets', 'formatting_issues', 'overall_feedback'\n"
    "}\n"
    "\n"
    "with open('data/validation.json') as f:\n"
    "    val_raw_split = json.load(f)\n"
    "\n"
    "# Test on first 2 validation samples\n"
    "for i, sample in enumerate(val_raw_split[:2]):\n"
    "    print(f'--- Sample {i} ---')\n"
    "    raw = generate_from_sample(sample, trainer.model, tokenizer)\n"
    "    parsed, method = extract_json(raw)\n"
    "    gt_score = json.loads(sample['output'])['ats_score']\n"
    "\n"
    "    if parsed and REQUIRED_OUTPUT_KEYS.issubset(set(parsed)):\n"
    "        print(f'  Valid ATS JSON (parse method: {method})')\n"
    "        print(f'  Predicted score: {parsed[\"ats_score\"]}  |  GT score: {gt_score}')\n"
    "    elif parsed:\n"
    "        missing = REQUIRED_OUTPUT_KEYS - set(parsed)\n"
    "        print(f'  Parsed JSON but missing keys: {missing}')\n"
    "    else:\n"
    "        print(f'  Could not parse JSON')\n"
    "        print(f'  Raw output (first 300 chars): {raw[:300]}')\n"
    "    print()"
))

# ─── Phase 6: Evaluation Metrics ─────────────────────────────────────────────
cells.append(md(
    "## Phase 6 — Evaluation Metrics\n"
    "\n"
    "Batch inference over all validation samples and compute:\n"
    "- **JSON Validity** — % samples that produce parseable JSON\n"
    "- **ATS Structure Validity** — % samples with all required keys\n"
    "- **Score MAE** — mean absolute error vs ground-truth ATS score\n"
    "- **Missing-Skill F1** — precision/recall over predicted vs GT missing skills\n"
    "\n"
    "> All metrics use `weak_bullets` as the correct key (matching the training data)."
))

cells.append(code(
    "import statistics\n"
    "\n"
    "results = []\n"
    "print(f'Evaluating {len(val_raw_split)} validation samples...')\n"
    "\n"
    "for i, sample in enumerate(val_raw_split):\n"
    "    raw    = generate_from_sample(sample, trainer.model, tokenizer)\n"
    "    parsed, _ = extract_json(raw)\n"
    "    gt     = json.loads(sample['output'])\n"
    "\n"
    "    is_valid_json = parsed is not None\n"
    "    is_valid_ats  = is_valid_json and REQUIRED_OUTPUT_KEYS.issubset(set(parsed))\n"
    "\n"
    "    score_diff = abs(parsed['ats_score'] - gt['ats_score']) if is_valid_ats else None\n"
    "\n"
    "    # Missing-skill F1 (case-insensitive)\n"
    "    if is_valid_ats:\n"
    "        pred_m = {k.lower() for k in parsed.get('missing_skills', [])}\n"
    "        gt_m   = {k.lower() for k in gt.get('missing_skills', [])}\n"
    "        if gt_m:\n"
    "            prec = len(pred_m & gt_m) / len(pred_m) if pred_m else 0.0\n"
    "            rec  = len(pred_m & gt_m) / len(gt_m)\n"
    "            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0\n"
    "        else:\n"
    "            f1 = 1.0 if not pred_m else 0.0\n"
    "    else:\n"
    "        f1 = None\n"
    "\n"
    "    results.append({'valid_json': is_valid_json, 'valid_ats': is_valid_ats,\n"
    "                    'score_diff': score_diff, 'skill_f1': f1})\n"
    "\n"
    "    if (i + 1) % 5 == 0:\n"
    "        print(f'  {i + 1}/{len(val_raw_split)} done')"
))

cells.append(code(
    "n = len(results)\n"
    "json_rate  = sum(1 for r in results if r['valid_json']) / n * 100\n"
    "ats_rate   = sum(1 for r in results if r['valid_ats'])  / n * 100\n"
    "diffs      = [r['score_diff'] for r in results if r['score_diff'] is not None]\n"
    "f1s        = [r['skill_f1']   for r in results if r['skill_f1']   is not None]\n"
    "mae        = statistics.mean(diffs) if diffs else float('nan')\n"
    "mean_f1    = statistics.mean(f1s)   if f1s   else float('nan')\n"
    "\n"
    "print(f'=== Evaluation Results ({n} samples) ===')\n"
    "print(f'JSON Validity:     {json_rate:5.1f}%   (target > 95%)')\n"
    "print(f'ATS Structure:     {ats_rate:5.1f}%   (target > 90%)')\n"
    "print(f'Score MAE:         {mae:6.2f}    (target < 20)')\n"
    "print(f'Missing-Skill F1:  {mean_f1:5.1%}    (target > 50%)')\n"
    "\n"
    "# Score distribution comparison\n"
    "import json as _json\n"
    "gt_scores   = [_json.loads(s['output'])['ats_score'] for s in val_raw_split]\n"
    "pred_scores = []\n"
    "for r, s in zip(results, val_raw_split):\n"
    "    if r['valid_ats']:\n"
    "        pred_scores.append(\n"
    "            _json.loads(generate_from_sample(s, trainer.model, tokenizer).split('}')[0] + '}')\n"
    "            if False else None  # already have parsed results above\n"
    "        )\n"
    "print()\n"
    "print(f'GT score range:   {min(gt_scores)}-{max(gt_scores)}, mean={sum(gt_scores)/len(gt_scores):.1f}')"
))

# ─── Write the notebook ───────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        }
    },
    "cells": cells,
}

out_path = Path(__file__).parent.parent / "colab" / "ats_fine_tuning_pipeline.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {len(cells)} cells to {out_path}")
for i, c in enumerate(cells):
    ctype = c["cell_type"]
    preview = c["source"][:60].replace("\n", " ")
    print(f"  [{i:02d}] {ctype:8s} — {preview}")
