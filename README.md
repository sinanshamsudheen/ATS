# ATS Resume Compliance Checker

A web-based tool aimed at helping job seekers optimize their resumes for Applicant Tracking Systems (ATS). This application analyzes PDF resumes against job descriptions to provide compatibility scores, detect missing keywords, and offer actionable improvement suggestions using semantic analysis and (optional) generative AI.

## Features

- **Resume Parsing**: Extracts text and segments resumes into standard sections (Experience, Education, Skills, etc.).
- **Smart Analysis**:
    - **Keyword Matching**: Identifies missing critical skills using semantic similarity (Embeddings).
    - **Formatting Checks**: Flags ATS-unfriendly elements like tables, images, and encryption.
    - **Scoring**: Calculates an overall match score based on content relevance and formatting.
- **Interactive Dashboard**: User-friendly Streamlit interface for instant feedback.

## Tech Stack

- **Python 3.9+**
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF
- **NLP/ML**: 
    - `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
    - `scikit-learn` for cosine similarity
    - (Coming in Phase 2) Fine-tuned LLM (Phi-3-mini) for rewrite suggestions

## Installation

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository_url>
   cd ATS
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   conda create -n ats_env python=3.10
   conda activate ats_env
   # OR with venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

2. **Open the browser**
   The app usually runs at `http://localhost:8501`.

3. **Analyze a Resume**
   - Upload your PDF Resume.
   - Paste the Job Description text.
   - Click "Analyze Resume" to see the score and report.

## Project Structure

```
ATS/
├── plan.md              # 3-Phase Implementation Plan
├── PRD.md               # Product Requirements Document
├── requirements.txt
├── src/
│   ├── app/             # Streamlit Application
│   ├── analysis/        # Core Logic (Embeddings, Similarity, Formatting)
│   ├── preprocessing/   # PDF Extraction and Parsing
│   └── utils/
├── data/
└── tests/
```

## Contributing
Please follow the `plan.md` phases for contribution. Currently in **Phase 2** (Generative AI Integration).

## License
MIT
