# ATS Resume Compliance Checker & Optimizer

A web-based AI-powered tool that helps job seekers optimize their resumes for Applicant Tracking Systems (ATS). This application analyzes PDF resumes against job descriptions to provide compatibility scores, detect missing keywords, and offer **actionable AI-powered rewrite suggestions** using semantic analysis and generative AI.

**🎯 Current Version:** 2.0.0 - **Phase 2 Complete** (LLM-Powered Optimizer)

## ✨ Features

### Phase 1: Core Analysis (✅ Complete)
- **Resume Parsing**: Extracts text and segments resumes into standard sections (Experience, Education, Skills, etc.)
- **Smart Analysis**:
    - **Keyword Matching**: Identifies missing critical skills using semantic similarity (Embeddings)
    - **Formatting Checks**: Flags ATS-unfriendly elements like tables, images, and encryption
    - **Scoring**: Calculates an overall match score based on content relevance and formatting
- **Interactive Dashboard**: User-friendly Streamlit interface for instant feedback

### Phase 2: AI-Powered Optimization (✅ Complete)
- **🤖 LLM-Powered Rewrites**: Uses Phi-3-mini to generate improved bullet points
- **Quality Analysis**: Scores each bullet point (0-100) based on:
  - Action verb strength
  - Quantifiable metrics
  - Specificity vs. genericity
  - Optimal length
- **Side-by-Side Comparison**: See original vs. improved versions with detailed analysis
- **Smart Detection**: Automatically identifies weak bullet points that need improvement
- **Batch Processing**: Analyzes all resume sections efficiently

## 🛠 Tech Stack

- **Python 3.9+**
- **Frontend**: Streamlit with custom CSS
- **PDF Processing**: PyMuPDF (fitz)
- **NLP/ML**: 
    - `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
    - `scikit-learn` for cosine similarity
    - **Phi-3-mini** (microsoft/Phi-3-mini-4k-instruct) for rewrite generation
    - `transformers` + `bitsandbytes` for 4-bit quantization
- **Analysis Engine**: Custom heuristics + LLM prompting

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended (16GB+ for GPU inference)
- Optional: CUDA-compatible GPU for faster LLM inference

### Setup Steps

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository_url>
   cd ATS
   ```

2. **Create a virtual environment** (Recommended)
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   .\venv\Scripts\Activate.ps1
   
   # Activate on Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** First run will download ~4GB of models:
   - Phi-3-mini (~2.3GB after quantization)
   - all-MiniLM-L6-v2 (~80MB)

## 🚀 Usage

### Running the Application

1. **Activate your virtual environment** (if not already active)
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

3. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

### Analyzing a Resume

1. **Upload Resume**: Click "Choose a PDF file" and select your resume
2. **Paste Job Description**: Copy the full JD text into the text area
3. **Configure Options** (in sidebar):
   - Toggle "Enable AI Rewrites" for LLM-powered suggestions
   - Toggle "Only Rewrite Weak Bullets" to focus on weak points
4. **Click "Analyze & Optimize Resume"**
5. **View Results**:
   - **AI Rewrites Tab**: See bullet-by-bullet improvements
   - **Keywords Tab**: Check missing and matched keywords
   - **Formatting Tab**: Review ATS compatibility issues
   - **Raw Data Tab**: Inspect parsed resume structure

### First Run Note
⏳ **Model loading takes 30-60 seconds on first run.** Subsequent analyses are faster due to caching.

## 📂 Project Structure

```
ATS/
├── plan.md                          # 3-Phase Implementation Plan
├── PRD.md                           # Product Requirements Document  
├── progress.md                      # Phase 2 Implementation Progress
├── requirements.txt                 # Python Dependencies
├── README.md                        # This file
│
├── src/
│   ├── config.py                    # Central Configuration
│   │
│   ├── preprocessing/               # PDF & Text Processing
│   │   ├── pdf_extractor.py         # PyMuPDF-based extraction
│   │   └── resume_parser.py         # Section & bullet parsing
│   │
│   ├── analysis/                    # Core Analysis Modules
│   │   ├── embeddings.py            # Sentence transformer wrapper
│   │   ├── similarity.py            # Keyword matching & scoring
│   │   └── formatting_check.py      # ATS formatting validation
│   │
│   ├── model/                       # 🆕 LLM & Rewrite Engine
│   │   ├── llm_inference.py         # Phi-3-mini wrapper (4-bit)
│   │   └── rewrite_engine.py        # Quality analysis + rewrites
│   │
│   └── app/                         # Streamlit Application
│       ├── streamlit_app.py         # Main app (Phase 2 enhanced)
│       └── components/              # 🆕 UI Components
│           └── results_display.py   # Enhanced results visualization
│
├── data/
│   ├── README.md                    # Data structure documentation
│   ├── processed/                   # Training data samples
│   │   └── sample_training_data.jsonl
│   └── raw/                         # Raw data (user-provided)
│
├── models/
│   └── fine-tuned/                  # Future: LoRA adapters
│
└── tests/                           # Unit tests (TODO)
```

## 🎯 How It Works

### Analysis Pipeline

```
1. PDF Upload → Extract Text (PyMuPDF)
2. Parse Resume → Section Detection + Bullet Extraction
3. Format Check → ATS Compatibility Scan
4. Similarity Analysis → Keyword Matching (Embeddings)
5. 🆕 LLM Analysis → Quality Scoring + Rewrites
6. Results Display → Interactive Dashboard
```

### Quality Scoring Heuristics

Each bullet point is scored 0-100 based on:
- ✅ **Action Verb**: Starts with strong verb (led, built, achieved)
- ✅ **Metrics**: Contains numbers, percentages, or quantifiable data
- ✅ **Length**: 5-40 words (optimal range)
- ✅ **Specificity**: Avoids generic phrases ("responsible for", "helped with")

### LLM Rewrite Process

1. **Detection**: Identify bullets scoring <70
2. **Contextualization**: Pass bullet + JD to Phi-3-mini
3. **Generation**: LLM suggests improved version with reasoning
4. **Display**: Show side-by-side comparison with copy button

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# LLM Settings
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LLM_TEMPERATURE = 0.7          # Lower = more deterministic
USE_QUANTIZATION = True        # 4-bit for memory efficiency

# Quality Thresholds
WEAK_BULLET_MIN_WORDS = 5
WEAK_BULLET_MAX_WORDS = 40
SIMILARITY_THRESHOLD = 0.3     # Keyword matching sensitivity
```

## 🧪 Testing Phase 2

To validate the LLM integration:

```bash
# Test LLM loading (interactive Python)
python
>>> from src.model.llm_inference import get_llm
>>> llm = get_llm()  # Should load without errors
>>> result = llm.analyze_bullet("Worked on projects")
>>> print(result)
```

## 📊 Performance

### Expected Inference Times
- **Embedding Generation**: 0.5-2 seconds
- **Formatting Check**: 0.1-0.5 seconds
- **LLM Rewrites** (5 bullets):
  - CPU (4-bit): 20-40 seconds
  - GPU (4-bit): 5-15 seconds

### Memory Requirements
- **Without LLM**: ~500MB
- **With LLM (4-bit)**: ~3-4GB
- **With LLM (16-bit)**: ~8-10GB

## 🚧 Roadmap

### Phase 3: Production Polish (Next)
- [ ] Comprehensive testing suite
- [ ] Performance optimization (async processing)
- [ ] PDF export of optimized resume
- [ ] Enhanced error handling
- [ ] Deployment configuration (Docker)

### Future Enhancements
- [ ] Fine-tune Phi-3-mini on production data
- [ ] Industry-specific prompts
- [ ] Multi-language support
- [ ] Browser extension
- [ ] API endpoints

## 🤝 Contributing

Please follow the `plan.md` phases for contribution.  
**Current Status:** Phase 2 Complete, Phase 3 Ready to Start

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow existing code style
4. Add tests for new features
5. Submit a pull request

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Models Used**:
  - [microsoft/Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Frameworks**: Streamlit, Hugging Face Transformers
- **Course**: AMT302 - Concepts in Natural Language Processing

---

**Built with ❤️ for job seekers everywhere**
