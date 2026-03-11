# ATS Resume Compliance Checker

AI-powered resume optimiser that scores resumes against job descriptions, identifies missing keywords, flags formatting issues, and suggests bullet-point improvements.

## Features

- **PDF Parsing** — extract and clean text from any PDF resume.
- **Keyword Matching** — semantic similarity between resume and JD using sentence-transformers.
- **Formatting Audit** — detect ATS-unfriendly elements (images, encryption, low text density).
- **Bullet Quality Analysis** — heuristic scoring for action verbs, metrics, and specificity.
- **Groq LLM Integration** — fast cloud inference using Llama 3.1 8B via Groq API.
- **Multi-Backend Embeddings** — automatic fallback: SentenceTransformers → HuggingFace → OpenAI.
- **LoRA Fine-Tuning** — fine-tune Phi-3-Mini on synthetic ATS data using 4-bit QLoRA.
- **Streamlit UI** — upload a PDF, paste a JD, and get instant analysis.

## Project Structure

```
ATS/
├── colab/
│   ├── ats_fine_tuning_pipeline.ipynb  # Single end-to-end Colab notebook
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
│   ├── fine-tuned/                     # Saved LoRA adapter (after training)
│   └── huggingface_cache/              # Cached sentence-transformers model
├── scripts/
│   ├── generate_dataset.py             # Generate synthetic ATS dataset
│   ├── prepare_dataset.py              # Validate, format & split dataset
│   ├── train_model.py                  # Train Phi-3 + LoRA
│   └── run_inference.py                # Run inference with fine-tuned model
├── src/
│   ├── config.py                       # Central configuration
│   ├── parsing/
│   │   ├── pdf_extractor.py            # PDF → text (PyMuPDF)
│   │   ├── resume_parser.py            # Text → structured sections
│   │   └── jd_parser.py               # JD keyword extraction
│   ├── scoring/
│   │   ├── embeddings.py               # Multi-backend embeddings (ST/HF/OpenAI)
│   │   ├── similarity_engine.py        # Semantic keyword matching
│   │   └── formatting_check.py         # ATS format compliance
│   ├── rewriting/
│   │   └── bullet_rewriter.py          # Heuristic bullet analysis
│   ├── training/
│   │   ├── dataset_loader.py           # Dataset prep functions
│   │   └── lora_training.py            # LoRA training pipeline
│   ├── inference/
│   │   ├── generate_report.py          # Local Phi-3 model loading & generation
│   │   └── groq_inference.py           # Groq + Llama 3.1 cloud inference
│   └── app/
│       ├── streamlit_app.py            # Streamlit web UI
│       └── components/
│           └── results_display.py
├── requirements.txt
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
# Optional: OpenAI fallback for embeddings
OPENAI_API_KEY=sk_your_openai_key_here
```

Get your Groq API key from [console.groq.com/keys](https://console.groq.com/keys).

### 3. Run the Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

Upload a PDF resume, paste a job description, and click **Analyze & Optimize**.

## Fine-Tuning Pipeline

### Option A — Google Colab (recommended)

1. Upload the entire `colab/` folder to Google Drive (e.g. `My Drive/colab/`).
2. Open `colab/ats_fine_tuning_pipeline.ipynb` in Colab.
3. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
4. Run all cells top to bottom — the notebook is self-contained across all 6 phases:
   - Environment Setup
   - Dataset Preparation
   - Model Loading & LoRA Setup
   - Fine-Tuning
   - Inference Testing
   - Evaluation Metrics

### Option B — Local scripts

```bash
python scripts/generate_dataset.py   # → data/raw_dataset.json (200 samples)
python scripts/prepare_dataset.py    # → data/train.json + data/validation.json
python scripts/train_model.py        # → models/fine-tuned/ (requires GPU)
python scripts/run_inference.py --resume path/to/resume.txt --job-desc path/to/jd.txt
```

## Model Details

| Setting | Value |
|---|---|
| Base model | `microsoft/phi-3-mini-4k-instruct` (3.8 B) |
| Fallback | `microsoft/phi-2` (2.7 B) |
| Quantisation | 4-bit NF4 + double quant (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Effective batch size | 16 (2 × 8 grad accum) |
| Learning rate | 2 × 10⁻⁴ |
| EOS token | `<\|end\|>` (Phi-3 native) |
| Label masking | Response-only (`-100` on prompt tokens) |
| Min. transformers | `>=4.41.0` (native Phi-3 support, no `trust_remote_code` for model) |

## Training Notes

- **Response-only label masking** — the training loss is computed only on the `### Response:` section of each prompt. Prompt tokens are masked with `-100` so the model learns to generate answers, not memorise inputs.
- **EOS token** — Phi-3 uses `<|end|>` as its end-of-sequence token. Using the wrong token (e.g. `<|endoftext|>`) causes the model to never learn to terminate generation cleanly.
- **No `trust_remote_code` for model loading** — `transformers>=4.41.0` ships native `Phi3ForCausalLM`. Passing `trust_remote_code=True` to `AutoModelForCausalLM` downloads a stale `modeling_phi3.py` that is incompatible with the current `transformers` cache API and causes runtime errors.

## Tech Stack

- **Groq** — fast LLM inference (Llama 3.1 8B)
- **Transformers / PEFT / bitsandbytes** — model loading, LoRA, 4-bit quantisation
- **Sentence-Transformers / HuggingFace** — semantic embeddings (`all-MiniLM-L6-v2`)
- **OpenAI** — optional embedding fallback
- **PyMuPDF** — PDF text extraction
- **Streamlit** — web interface
- **scikit-learn** — cosine similarity
- **Datasets / Accelerate** — training data pipeline and distributed training support
- **python-dotenv** — environment variable management

## Embedding Backends

The app automatically selects the best available embedding backend:

| Priority | Backend | Requirements |
|----------|---------|-------------|
| 1 | SentenceTransformers | `sentence-transformers` installed |
| 2 | HuggingFace Direct | `transformers` installed (automatic fallback) |
| 3 | OpenAI API | `OPENAI_API_KEY` set in `.env` |

## LLM Inference

| Mode | Model | Requirements |
|------|-------|-------------|
| **Cloud (default)** | Llama 3.1 8B via Groq | `GROQ_API_KEY` in `.env` |
| Local (optional) | Phi-3-mini + LoRA | GPU with 8GB+ VRAM |

---

## Changelog

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

**Migration Notes:**
1. Create a `.env` file with your `GROQ_API_KEY`
2. Optionally add `OPENAI_API_KEY` for embedding fallback
3. No GPU required — all inference runs via Groq cloud
