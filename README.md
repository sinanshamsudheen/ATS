# ATS Resume Compliance Checker

AI-powered resume optimiser that scores resumes against job descriptions, identifies missing keywords, flags formatting issues, and suggests bullet-point improvements — running entirely on CPU via a quantised local model.

## Features

- **PDF Parsing** — extract and clean text from any PDF resume.
- **Local-First LLM Inference** — GGUF-quantised Phi-3 model runs fully offline on CPU (no GPU required).
- **Groq Cloud Fallback** — automatic fallback to Llama 3.1 8B via Groq API when local model is unavailable.
- **ATS Report** — structured AI-generated score with keyword coverage, formatting, and improvement suggestions.
- **LoRA Fine-Tuning** — fine-tune Phi-3-Mini on synthetic ATS data using 4-bit QLoRA (Colab or local GPU).
- **Streamlit UI** — upload a PDF, paste a JD, and get instant ATS analysis.

## Project Structure

```
ATS/
├── colab/
│   ├── ats_fine_tuning_pipeline.ipynb  # Single end-to-end Colab notebook
│   ├── ats_phi_lora/                   # Trained LoRA adapter weights
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   ├── configs/
│   │   ├── lora_config.yaml            # LoRA hyperparameters
│   │   └── training_config.yaml        # Training & model settings
│   └── data/
│       ├── raw_dataset.json            # 200 synthetic samples
│       ├── train.json                  # 180 training samples
│       └── validation.json             # 20 validation samples
├── configs/
│   ├── lora_config.yaml                # LoRA hyperparameters (local)
│   └── training_config.yaml            # Training & model settings (local)
├── data/
│   ├── raw_dataset.json                # 200 synthetic samples
│   ├── train.json                      # 180 training samples
│   └── validation.json                 # 20 validation samples
├── models/
│   ├── fine-tuned/                     # Merged HF model (intermediate, before GGUF)
│   ├── phi3-ats-q8_0.gguf              # Final quantised model (Q8_0, ~3.8 GB)
│   └── huggingface_cache/              # Cached sentence-transformers model
├── scripts/
│   ├── train_model.py                  # Train Phi-3 + LoRA (requires GPU)
│   └── merge_and_convert.py            # Merge LoRA → base → GGUF (one-shot)
├── src/
│   ├── config.py                       # Central configuration & env setup
│   ├── parsing/
│   │   ├── pdf_extractor.py            # PDF → text (PyMuPDF)
│   │   └── resume_parser.py            # Text → structured sections
│   ├── training/
│   │   ├── dataset_loader.py           # Dataset prep functions
│   │   └── lora_training.py            # LoRA training pipeline
│   ├── inference/
│   │   ├── gguf_inference.py           # Primary — llama-cpp GGUF inference
│   │   ├── generate_report.py          # HF Phi-3 + LoRA inference (training use)
│   │   └── groq_inference.py           # Cloud fallback — Groq + Llama 3.1 8B
│   └── app/
│       ├── streamlit_app.py            # Streamlit web UI
│       └── components/
│           └── results_display.py
├── .env                                # API keys — gitignored, never commit
├── .env.example                        # Template — copy to .env and fill in keys
├── requirements.txt
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`llama-cpp-python` requires a prebuilt wheel (no C++ compiler needed):

```bash
pip install llama-cpp-python==0.3.2 \
  --only-binary=:all: \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Generate the GGUF model (one-time setup)

If `models/phi3-ats-q8_0.gguf` does not yet exist, run:

```bash
python scripts/merge_and_convert.py
```

This will:
1. Merge the LoRA adapter (`colab/ats_phi_lora/`) into the base Phi-3-mini weights.
2. Clone `llama.cpp` into `vendor/` (shallow clone, ~50 MB).
3. Convert and quantise the merged model to `models/phi3-ats-q8_0.gguf` (Q8_0, ~3.8 GB).

Pass `--quant q4_k_m` for a smaller (~2.3 GB) but slightly lower-quality model.

### 3. Run the Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

Upload a PDF resume, paste a job description, and click **Analyze & Optimize**.

The sidebar shows which inference backend is active (GGUF / Groq).

### 4. (Optional) Configure API keys

If the local GGUF model is unavailable, the app automatically falls back to Groq.

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Then open `.env` and fill in your key:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Get your Groq API key from [console.groq.com/keys](https://console.groq.com/keys). The `.env` file is gitignored — never commit it.

## LLM Inference Chain

The app resolves inference in priority order at startup:

| Priority | Backend | Model | Notes |
|----------|---------|-------|-------|
| 1 — Primary | **GGUF (llama-cpp)** | Phi-3-mini Q8_0 | ~3.8 GB RAM, fully offline, CPU-only |
| 2 — Cloud | **Groq API** | Llama 3.1 8B Instant | Requires `GROQ_API_KEY` in `.env` |

The active backend is shown in the sidebar. Analysis still works if only Groq is available.

## Fine-Tuning Pipeline

### Option A — Google Colab (recommended)

1. Upload the entire `colab/` folder to Google Drive (e.g. `My Drive/colab/`).
2. Open `colab/ats_fine_tuning_pipeline.ipynb` in Colab.
3. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
4. Run all cells — the notebook covers all 6 phases end-to-end:
   - Environment Setup
   - Dataset Preparation
   - Model Loading & LoRA Setup
   - Fine-Tuning
   - Inference Testing
   - Evaluation Metrics
5. Download the trained adapter from `colab/ats_phi_lora/` and place it in your local `colab/ats_phi_lora/` folder.
6. Run `python scripts/merge_and_convert.py` to produce the GGUF file.

### Option B — Local scripts (requires GPU)

```bash
python scripts/train_model.py        # → colab/ats_phi_lora/ (requires GPU)
python scripts/merge_and_convert.py  # → models/phi3-ats-q8_0.gguf
```

## Model Details

| Setting | Value |
|---------|-------|
| Base model | `microsoft/phi-3-mini-4k-instruct` (3.8 B) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Training quantisation | 4-bit NF4 + double quant (bitsandbytes, Colab only) |
| Effective batch size | 16 (2 × 8 gradient accumulation) |
| Learning rate | 2 × 10⁻⁴ |
| EOS token | `<\|end\|>` (Phi-3 native) |
| Label masking | Response-only (`-100` on prompt tokens) |
| GGUF quantisation | Q8_0 — 8-bit, best quality/size trade-off |
| Runtime precision (CPU) | bfloat16 (HF path) |

## Training Notes

- **Response-only label masking** — the training loss is computed only on the `### Response:` section of each prompt. Prompt tokens are masked with `-100` so the model learns to generate answers, not memorise inputs.
- **EOS token** — Phi-3 uses `<|end|>` as its end-of-sequence token. Using the wrong token (e.g. `<|endoftext|>`) causes the model to never learn to terminate generation cleanly.
- **No `trust_remote_code` for model loading** — `transformers>=4.41.0` ships native `Phi3ForCausalLM`. Passing `trust_remote_code=True` to `AutoModelForCausalLM` downloads a stale cached `modeling_phi3.py` that is incompatible with the current `transformers` KV cache API (`DynamicCache.seen_tokens` was removed) and causes runtime errors. Use `Phi3ForCausalLM` directly.
- **`ensure_weight_tying` must be `false`** — setting this to `true` in `adapter_config.json` causes PEFT to try to resolve `base_model.model.model.model.embed_tokens`, which does not exist in the checkpoint. Leave it `false`.

## Tech Stack

- **llama-cpp-python** — GGUF model execution on CPU (no GPU required)
- **Transformers / PEFT** — HF model loading and LoRA adapter support
- **bitsandbytes** — 4-bit NF4 quantisation for Colab training
- **Groq** — cloud LLM inference fallback (Llama 3.1 8B)
- **PyMuPDF** — PDF text extraction
- **Streamlit** — web interface
- **Datasets / Accelerate** — training data pipeline and distributed training support
- **python-dotenv** — environment variable management

---

## Changelog

### v3.0.0 — Local-First CPU Inference via GGUF

**New Features:**
- **GGUF CPU Inference** — Added `src/inference/gguf_inference.py` using `llama-cpp-python` to run the quantised Phi-3 model entirely on CPU with no GPU required. Primary inference backend.
- **LoRA Merge & Convert Script** — `scripts/merge_and_convert.py` merges the LoRA adapter into the base model and converts it to GGUF (Q8_0 by default) in one command. Clones `llama.cpp` into `vendor/` automatically.
- **2-Tier Inference Chain** — `streamlit_app.py` tries GGUF → Groq in order. Sidebar shows the active backend.
- **Groq as Pure Fallback** — Groq API is now optional; the app runs fully offline if the GGUF model is present.
- **Simplified UI** — removed keyword analysis, formatting audit, and bullet-rewrite tabs. The app now surfaces a single AI-generated ATS report.

**Bug Fixes:**
- Fixed `DynamicCache.seen_tokens` `AttributeError`: replaced `AutoModelForCausalLM` with `Phi3ForCausalLM` directly, removing `trust_remote_code=True` and the stale cached `modeling_phi3.py`.
- Fixed `base_model.model.model.lm_head` `KeyError`: `low_cpu_mem_usage=True` and `ensure_weight_tying: true` both caused PEFT to look up checkpoint keys that don't exist for tied-weight models. Reverted both.
- Suppressed `n_ctx_per_seq < n_ctx_train` warning by setting `n_ctx=4096` (matching training context length).
- Removed deprecated `TRANSFORMERS_CACHE` env var; `HF_HOME` is the correct replacement since transformers v5.
- Suppressed TensorFlow / oneDNN / Google API `FutureWarning` noise at startup.

**Files Changed:**
| File | Change |
|------|--------|
| `src/inference/gguf_inference.py` | **NEW** — llama-cpp GGUF inference wrapper |
| `scripts/merge_and_convert.py` | **NEW** — LoRA merge + GGUF conversion pipeline |
| `src/app/streamlit_app.py` | Rewritten — GGUF → Groq chain; ATS report only |
| `src/app/components/results_display.py` | Stripped to `display_llm_ats_report` only |
| `src/inference/generate_report.py` | Switched to `Phi3ForCausalLM`, removed `trust_remote_code`, added CPU bfloat16 |
| `src/config.py` | Added `GGUF_MODEL_PATH`; removed `TRANSFORMERS_CACHE`; added warning suppression |
| `requirements.txt` | Added `llama-cpp-python>=0.3.2`; removed `sentence-transformers`, `scikit-learn`, `openai` |
| `src/parsing/jd_parser.py` | **DELETED** |
| `src/scoring/similarity_engine.py` | **DELETED** |
| `src/scoring/embeddings.py` | **DELETED** |
| `src/scoring/formatting_check.py` | **DELETED** |
| `src/rewriting/bullet_rewriter.py` | **DELETED** |

**Migration Notes:**
1. Install `llama-cpp-python` via the prebuilt wheel (see Quick Start — Step 1).
2. Run `python scripts/merge_and_convert.py` once to generate `models/phi3-ats-q8_0.gguf`.
3. No `.env` file required for local-only usage. Add `GROQ_API_KEY` only if you want the cloud fallback.

---

### v0.2.0 — Cloud Inference & Embedding Fallbacks

**New Features:**
- **Groq LLM Integration** — Switched from local Phi-3 inference to Groq cloud API with Llama 3.1 8B for faster, GPU-free analysis.
- **Multi-Backend Embeddings** — Added 3-tier fallback system (SentenceTransformers → HuggingFace Direct → OpenAI) to handle dependency conflicts gracefully.
- **Environment Configuration** — Added `.env` support via `python-dotenv` for secure API key management.

**Files Changed:**
| File | Change |
|------|--------|
| `src/inference/groq_inference.py` | **NEW** — Groq API client with retry logic and rate limiting |
| `src/scoring/embeddings.py` | Rewritten with multi-backend fallback system |
| `src/config.py` | Added `GROQ_API_KEY`, `GROQ_MODEL`, OpenAI config, dotenv loading |
| `src/app/streamlit_app.py` | Updated to use Groq; shows connection status in sidebar |
| `requirements.txt` | Added `groq>=0.4.0`, `openai>=1.0.0`, `python-dotenv>=1.0.0` |
| `.gitignore` | Added `sample/` folder for test files |
