# ATS Resume Compliance Checker

**Micro Project: Concepts In Natural Language Processing (AMT302)**

Sree Chitra Thirunal College of Engineering, Thiruvananthapuram

**Project Members**
| Name | Roll No |
|------|---------|
| Abhay Krishna | 302 |
| D Vaibhav | 333 |
| Muhammed Sinan D | 347 |
| Paul Johnston | 351 |

**Academic Year: 2023–2027**

---

## Introduction

Applicant Tracking Systems (ATS) are widely used by organizations to automatically filter and rank resumes during recruitment. These systems evaluate resumes based on formatting compatibility, keyword relevance, and how well the content matches the job description. As a result, many qualified candidates may be rejected during the initial screening stage because their resumes are not optimized for ATS parsing.

The ATS Resume Compliance Checker is designed to help users analyze and improve their resumes so that they perform better in ATS-based screening systems. The system processes resumes in PDF format and compares them with job descriptions to evaluate compatibility. By identifying missing keywords, formatting issues, and weak experience descriptions, the tool provides suggestions that help users optimize their resumes for automated screening systems.

---

## Methodology

The system follows a structured pipeline to analyze resume quality and determine its compatibility with ATS systems. The main stages are outlined below:

### Resume Extraction

The uploaded resume is processed using a PDF extraction module (`src/parsing/pdf_extractor.py`) that extracts and cleans text from the document using PyMuPDF. Important sections such as skills, education, and experience are identified through pattern-based parsing (`src/parsing/resume_parser.py`) for further analysis.

### Job Description Analysis

The job description is analyzed to identify important keywords and skill requirements expected for the role. A deterministic keyword scorer (`src/scoring/keyword_scorer.py`) containing approximately 300 technical terms extracts relevant skills from the job description text.

### Semantic Similarity Calculation

The system uses a fine-tuned Phi-3 language model to semantically compare the resume content against job requirements. The model generates embeddings and calculates how well the resume matches the job description, accounting for synonyms and related terms that keyword matching alone would miss.

### Formatting Audit

The system checks for ATS-unfriendly elements such as:
- Missing section headers (Experience, Education, Skills)
- Absence of standard bullet point formatting
- Missing contact information (email, phone)
- Overall structural organization

### Bullet Point Quality Evaluation

Experience bullet points are evaluated to determine whether they contain:
- Strong action verbs (e.g., "Developed", "Implemented", "Led")
- Measurable outcomes and quantifiable results
- Clear descriptions of contributions and impact

Weak bullets are identified and rewrite suggestions are generated.

### Report Generation

The results of the analysis are compiled into a structured ATS compliance report highlighting:
- **ATS Score** (0–100): Overall compatibility score
- **Score Breakdown**: keyword_coverage (50%), bullet_quality (25%), formatting (15%), structure (10%)
- **Matched Skills**: Keywords found in both resume and job description
- **Missing Skills**: Important keywords from the job description not found in the resume
- **Formatting Issues**: Structural problems that may affect ATS parsing
- **Suggested Improvements**: Specific recommendations for optimization

---

## Results

The system generates an ATS Compliance Report with the following structure:

**ATS Compliance Score**

| ATS Score |
|-----------|
| 0 – 100   |

**Analysis Summary**

| Evaluation Component | Description |
|----------------------|-------------|
| Keyword Match | Percentage of job description keywords found in the resume |
| Missing Skills | Technical skills and keywords absent from the resume |
| Formatting Issues | Structural problems affecting ATS readability |
| Suggested Improvements | Actionable recommendations for resume optimization |

---

## Technical Implementation

### Project Structure

```
ATS/
├── colab/
│   ├── ats_fine_tuning_pipeline.ipynb  # End-to-end training notebook
│   ├── configs/
│   │   ├── lora_config.yaml            # LoRA hyperparameters
│   │   └── training_config.yaml        # Training settings
│   └── data/
│       ├── train.json                  # Training samples
│       └── validation.json             # Validation samples
├── models/                             # Generated at runtime
│   └── phi3-ats-q8_0.gguf             # Quantised model (~3.8 GB)
├── scripts/
│   ├── train_model.py                  # Local training script
│   ├── merge_and_convert.py            # LoRA merge + GGUF conversion
│   └── fix_dataset.py                  # Dataset repair utility
├── src/
│   ├── config.py                       # Configuration & environment
│   ├── parsing/
│   │   ├── pdf_extractor.py            # PDF text extraction
│   │   └── resume_parser.py            # Section parsing
│   ├── scoring/
│   │   └── keyword_scorer.py           # Skill extraction & matching
│   ├── inference/
│   │   ├── _prompts.py                 # Shared prompt templates
│   │   ├── gguf_inference.py           # Local CPU inference
│   │   └── groq_inference.py           # Cloud inference fallback
│   └── app/
│       ├── streamlit_app.py            # Web interface
│       └── components/
│           └── results_display.py      # Report visualisation
├── requirements.txt
└── .env.example
```

### Technologies Used

| Category | Libraries |
|----------|-----------|
| LLM Inference | llama-cpp-python, transformers, groq |
| Fine-Tuning | peft (LoRA), bitsandbytes, trl, accelerate |
| PDF Processing | PyMuPDF |
| Web Interface | Streamlit |
| Data Processing | pandas, numpy, datasets |

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/phi-3-mini-4k-instruct (3.8B parameters) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj, k_proj, o_proj |
| Quantisation | Q8_0 (8-bit) |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

pip install llama-cpp-python==0.3.2 \
  --only-binary=:all: \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Generate the Model (One-Time Setup)

```bash
python scripts/merge_and_convert.py
```

### 3. Run the Application

```bash
streamlit run src/app/streamlit_app.py
```

### 4. Usage

1. Upload a PDF resume
2. Paste the job description
3. Click **Analyze Resume**
4. View the ATS compliance report

### Optional: Configure Cloud Fallback

```bash
copy .env.example .env
```

Add your Groq API key in `.env`:
```
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## Inference Pipeline

The system uses a two-tier inference chain:

| Priority | Backend | Model | Requirements |
|----------|---------|-------|--------------|
| 1 (Primary) | GGUF (llama-cpp) | Phi-3-mini Q8_0 | ~3.8 GB RAM, CPU-only |
| 2 (Fallback) | Groq API | Llama 3.1 8B | GROQ_API_KEY |

The application runs fully offline when the local GGUF model is available.

---

## Fine-Tuning (Optional)

### Google Colab (Recommended)

1. Upload `colab/` folder to Google Drive
2. Open `ats_fine_tuning_pipeline.ipynb` in Colab
3. Set runtime to T4 GPU
4. Run all cells
5. Download trained adapter and run `merge_and_convert.py`

### Local (Requires GPU)

```bash
python scripts/train_model.py
python scripts/merge_and_convert.py
```

---

## References

- Phi-3 Model: [microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- llama.cpp: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
