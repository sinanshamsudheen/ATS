# ATS Resume Compliance Checker

AI-powered resume optimiser that scores resumes against job descriptions, identifies missing keywords, flags formatting issues, and suggests bullet-point improvements.

## Features

- **PDF Parsing** — extract and clean text from any PDF resume.
- **Keyword Matching** — semantic similarity between resume and JD using sentence-transformers.
- **Formatting Audit** — detect ATS-unfriendly elements (images, encryption, low text density).
- **Bullet Quality Analysis** — heuristic scoring for action verbs, metrics, and specificity.
- **LoRA Fine-Tuning** — fine-tune a Phi-3-mini model on synthetic ATS data using 4-bit QLoRA.
- **Streamlit UI** — upload a PDF, paste a JD, and get instant analysis.

## Project Structure

```
ATS/
├── configs/
│   ├── lora_config.yaml          # LoRA hyperparameters
│   └── training_config.yaml      # Training & model settings
├── data/
│   ├── raw_dataset.json          # 200 synthetic samples
│   ├── train.json                # 180 training samples
│   └── validation.json           # 20 validation samples
├── models/
│   ├── ats_phi_lora/             # Saved LoRA adapter (after training)
│   └── huggingface_cache/        # Cached sentence-transformers model
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_model_loading_and_lora_setup.ipynb
│   ├── 04_fine_tuning_training.ipynb
│   ├── 05_inference_testing.ipynb
│   └── 06_evaluation_metrics.ipynb
├── scripts/
│   ├── generate_dataset.py       # Generate synthetic ATS dataset
│   ├── prepare_dataset.py        # Validate, format & split dataset
│   ├── train_model.py            # Train Phi + LoRA
│   └── run_inference.py          # Run inference with fine-tuned model
├── src/
│   ├── config.py                 # Central configuration
│   ├── parsing/
│   │   ├── pdf_extractor.py      # PDF → text (PyMuPDF)
│   │   ├── resume_parser.py      # Text → structured sections
│   │   └── jd_parser.py          # JD keyword extraction
│   ├── scoring/
│   │   ├── embeddings.py         # SentenceTransformer embeddings
│   │   ├── similarity_engine.py  # Semantic keyword matching
│   │   └── formatting_check.py   # ATS format compliance
│   ├── rewriting/
│   │   └── bullet_rewriter.py    # Heuristic bullet analysis
│   ├── training/
│   │   ├── dataset_loader.py     # Dataset prep functions
│   │   └── lora_training.py      # LoRA training pipeline
│   ├── inference/
│   │   └── generate_report.py    # Model loading & generation
│   └── app/
│       ├── streamlit_app.py      # Streamlit web UI
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

### Generate & prepare dataset

```bash
python scripts/generate_dataset.py   # → data/raw_dataset.json (200 samples)
python scripts/prepare_dataset.py    # → data/train.json + data/validation.json
```

### Train (requires GPU)

```bash
python scripts/train_model.py
```

Saves the LoRA adapter to `models/ats_phi_lora/`.

### Inference

```bash
python scripts/run_inference.py --resume path/to/resume.txt --job-desc path/to/jd.txt
```

Or use the interactive notebooks in `notebooks/` for step-by-step exploration.

## Model Details

| Setting | Value |
|---|---|
| Base model | `microsoft/phi-3-mini-4k-instruct` (3.8 B) |
| Fallback | `microsoft/phi-2` (2.7 B) |
| Quantisation | 4-bit NF4 + double quant |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Effective batch size | 16 (2 × 8 grad accum) |
| Learning rate | 2 × 10⁻⁴ |

## Tech Stack

- **Transformers / PEFT / bitsandbytes** — model loading, LoRA, quantisation
- **Sentence-Transformers** — semantic embeddings (all-MiniLM-L6-v2)
- **PyMuPDF** — PDF text extraction
- **Streamlit** — web interface
- **scikit-learn** — cosine similarity
