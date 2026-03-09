# ATS Resume Compliance Checker

AI-powered resume optimiser that scores resumes against job descriptions, identifies missing keywords, flags formatting issues, and suggests bullet-point improvements.

## Features

- **PDF Parsing** — extract and clean text from any PDF resume.
- **Keyword Matching** — semantic similarity between resume and JD using sentence-transformers.
- **Formatting Audit** — detect ATS-unfriendly elements (images, encryption, low text density).
- **Bullet Quality Analysis** — heuristic scoring for action verbs, metrics, and specificity.
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
│   │   ├── embeddings.py               # SentenceTransformer embeddings
│   │   ├── similarity_engine.py        # Semantic keyword matching
│   │   └── formatting_check.py         # ATS format compliance
│   ├── rewriting/
│   │   └── bullet_rewriter.py          # Heuristic bullet analysis
│   ├── training/
│   │   ├── dataset_loader.py           # Dataset prep functions
│   │   └── lora_training.py            # LoRA training pipeline
│   ├── inference/
│   │   └── generate_report.py          # Model loading & generation
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

### 2. Run the Streamlit app

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

- **Transformers / PEFT / bitsandbytes** — model loading, LoRA, 4-bit quantisation
- **Sentence-Transformers** — semantic embeddings (`all-MiniLM-L6-v2`)
- **PyMuPDF** — PDF text extraction
- **Streamlit** — web interface
- **scikit-learn** — cosine similarity
- **Datasets / Accelerate** — training data pipeline and distributed training support
