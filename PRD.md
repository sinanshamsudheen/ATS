# Product Requirements Document (PRD)
## ATS Resume Compliance Checker

**Version:** 1.0  
**Last Updated:** January 12, 2026  
**Project Course:** AMT302 – Concepts in Natural Language Processing  
**Institution:** Sree Chitra Thirunal College of Engineering, Thiruvananthapuram

---

## 1. Executive Summary

### 1.1 Problem Statement
Applicant Tracking Systems (ATS) automatically reject a significant portion of resumes due to:
- Poor formatting (tables, images, special characters)
- Missing keywords and skills from job descriptions
- Weak or generic bullet points
- Structural issues that prevent proper parsing

This creates a major barrier for job seekers, particularly fresh graduates in competitive markets like India, who often lack awareness of ATS requirements.

### 1.2 Solution Overview
A web-based ATS compliance checker that analyzes PDF resumes against job descriptions, providing:
- ATS compatibility score (0-100)
- Missing skills detection
- Weak bullet point identification
- Actionable rewrite suggestions using a fine-tuned language model

---

## 2. Project Objectives

### 2.1 Primary Goals
1. Build a functional web application for resume analysis
2. Achieve accurate ATS score calculation (0-100 scale)
3. Identify missing keywords through semantic similarity
4. Generate actionable, improved bullet point suggestions
5. Flag ATS-unfriendly formatting issues

### 2.2 Success Criteria
- Application successfully processes PDF resumes and job descriptions
- Score generation completes within 30 seconds per resume
- User-friendly dashboard displays all insights clearly
- Fine-tuned model produces relevant, structured outputs
- System correctly identifies at least 80% of critical missing keywords

---

## 3. System Architecture

### 3.1 High-Level Architecture
```
[User Interface (Streamlit)]
         ↓
[PDF Upload Handler]
         ↓
[Text Extraction Module (PyMuPDF)]
         ↓
[Resume Parser & Segmentation]
         ↓
[Analysis Pipeline]
    ├─ Semantic Embedding (all-MiniLM-L6-v2)
    ├─ Similarity Calculation (Cosine)
    ├─ Formatting Checker (Rule-based)
    └─ LLM Analysis (Fine-tuned Phi-3-mini)
         ↓
[Results Aggregator]
         ↓
[Dashboard Renderer]
```

### 3.2 Core Components

#### 3.2.1 PDF Processing Module
**Responsibility:** Extract and clean text from PDF resumes

**Requirements:**
- Use PyMuPDF (fitz) as primary extraction library
- Fallback to pypdf if PyMuPDF fails
- Handle multi-page resumes
- Preserve section structure
- Remove excessive whitespace while maintaining bullet points

**Input:** PDF file (max 5MB)  
**Output:** Structured text with identified sections

**Implementation Notes:**
```python
# Expected section detection
STANDARD_SECTIONS = [
    "contact", "summary", "experience", 
    "education", "skills", "projects", 
    "certifications", "achievements"
]
```

#### 3.2.2 Resume Segmentation Module
**Responsibility:** Parse extracted text into structured sections and bullet points

**Requirements:**
- Detect section headers using regex and common patterns
- Extract bullet points (•, -, *, numbered lists)
- Identify contact information
- Parse dates and durations
- Handle various resume formats

**Output Structure:**
```python
{
    "contact": {...},
    "sections": [
        {
            "title": "Experience",
            "bullets": ["...", "..."],
            "metadata": {...}
        }
    ]
}
```

#### 3.2.3 Semantic Embedding Module
**Responsibility:** Generate vector embeddings for similarity analysis

**Requirements:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Generate embeddings for:
  - Each resume bullet point
  - Each sentence in job description
  - Skills list from job description
- Cache embeddings to avoid recomputation

**Technical Specs:**
- Embedding dimension: 384
- Batch processing for efficiency
- GPU acceleration if available (fallback to CPU)

#### 3.2.4 Similarity Analysis Module
**Responsibility:** Calculate coverage and identify gaps

**Requirements:**
- Compute cosine similarity between resume and job description embeddings
- Identify missing keywords (similarity threshold < 0.3)
- Calculate section-wise coverage scores
- Rank bullet points by relevance to job description

**Key Metrics:**
```python
{
    "overall_coverage": 0.0-1.0,
    "keyword_matches": [...],
    "missing_skills": [...],
    "weak_bullets": [...]  # Low similarity scores
}
```

#### 3.2.5 Formatting Checker Module
**Responsibility:** Detect ATS-unfriendly elements

**Rules to Implement:**
- ❌ Tables detected → Flag as "Tables may not parse correctly"
- ❌ Images/graphics → Flag as "Images will be ignored by ATS"
- ❌ Headers/footers → Flag as "Content may be missed"
- ❌ Text boxes → Flag as "Text boxes not supported"
- ❌ Special characters (e.g., §, †, ‡) → Flag specific characters
- ❌ Columns → Flag as "Multi-column layouts may confuse ATS"
- ❌ Unusual fonts → Flag if font metadata is unusual
- ✅ Standard bullet points (•, -, *)
- ✅ Clear section headers
- ✅ Standard fonts (Arial, Calibri, Times New Roman)

**Output:**
```python
{
    "formatting_issues": [
        {"type": "table", "severity": "high", "message": "..."},
        ...
    ],
    "formatting_score": 0-100
}
```

#### 3.2.6 Fine-Tuned LLM Module
**Responsibility:** Generate structured analysis and suggestions

**Model Selection:**
- Primary: Phi-3-mini (3.8B parameters)
- Alternative: Llama-3.2-3B or similar small LLM
- Fine-tuning method: LoRA (Low-Rank Adaptation)

**Fine-Tuning Requirements:**

**Dataset:**
- Source 1: "Resume Parsing & Summarizer Dataset" (Kaggle)
- Source 2: "resume-ats-score-v1-en" (Hugging Face)
- Minimum 500 resume-job description pairs
- Each example should include:
  - Resume text
  - Job description
  - ATS score (ground truth or expert-labeled)
  - Missing keywords
  - Weak bullets with improvements

**Training Configuration:**
```python
lora_config = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}

training_args = {
    "num_epochs": 3-5,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 4
}
```

**Prompt Template:**
```
You are an ATS resume optimization expert. Analyze the following resume against the job description.

Resume:
{resume_text}

Job Description:
{job_description}

Similarity Metrics:
{similarity_data}

Provide a structured analysis in JSON format with:
1. ats_score (0-100)
2. missing_keywords (list of important skills/terms from JD not in resume)
3. weak_bullets (list of bullets with low impact)
4. improved_bullets (rewritten versions with stronger action verbs and quantification)

Output:
```

**Expected Output Format:**
```json
{
    "ats_score": 72,
    "score_breakdown": {
        "keyword_match": 65,
        "formatting": 85,
        "experience_relevance": 70
    },
    "missing_keywords": [
        "Python", "Machine Learning", "AWS"
    ],
    "weak_bullets": [
        {
            "original": "Worked on projects",
            "reason": "Vague, no metrics, weak verb"
        }
    ],
    "improved_bullets": [
        {
            "original": "Worked on projects",
            "improved": "Led development of 3 machine learning projects, improving model accuracy by 25%",
            "improvements": ["action verb", "quantification", "specificity"]
        }
    ],
    "recommendations": [
        "Add specific technologies used",
        "Quantify achievements with metrics"
    ]
}
```

---

## 4. User Interface Requirements

### 4.1 Technology Stack
- **Framework:** Streamlit (Python-based)
- **Alternative:** React frontend + FastAPI backend (if team prefers)

### 4.2 Page Structure

#### 4.2.1 Main Dashboard Layout
```
┌─────────────────────────────────────────────┐
│  ATS Resume Compliance Checker              │
│  [Logo] Optimize Your Resume for ATS        │
├─────────────────────────────────────────────┤
│                                             │
│  Step 1: Upload Resume                      │
│  ┌─────────────────────────────────┐       │
│  │  [Drag & Drop PDF]              │       │
│  │  or click to browse             │       │
│  └─────────────────────────────────┘       │
│                                             │
│  Step 2: Paste Job Description              │
│  ┌─────────────────────────────────┐       │
│  │                                 │       │
│  │  (Multi-line text area)         │       │
│  │                                 │       │
│  └─────────────────────────────────┘       │
│                                             │
│         [Analyze Resume Button]             │
│                                             │
└─────────────────────────────────────────────┘
```

#### 4.2.2 Results Dashboard
```
┌─────────────────────────────────────────────┐
│  Analysis Results                           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  ATS SCORE: 72/100                  │   │
│  │  ████████████░░░░░░                 │   │
│  │                                     │   │
│  │  Keyword Match:    65/100 ████░     │   │
│  │  Formatting:       85/100 ████████  │   │
│  │  Experience Match: 70/100 ██████░   │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ⚠️ Formatting Issues (2)                   │
│  • Tables detected - may not parse well     │
│  • Special characters found: §, †           │
│                                             │
│  📋 Missing Keywords (5)                    │
│  • Python  • Machine Learning               │
│  • AWS  • Docker  • CI/CD                   │
│                                             │
│  ❌ Weak Bullets → ✅ Improved Bullets       │
│  ┌─────────────────────────────────────┐   │
│  │ Original: "Worked on projects"      │   │
│  │ Improved: "Led development of 3     │   │
│  │ ML projects, improving accuracy     │   │
│  │ by 25%"                             │   │
│  │                                     │   │
│  │ [Copy Improved] [Apply All]         │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  💡 Recommendations                         │
│  1. Add quantifiable metrics               │
│  2. Use stronger action verbs              │
│  3. Remove table formatting                │
│                                             │
│  [Download Report] [Analyze Another]        │
│                                             │
└─────────────────────────────────────────────┘
```

### 4.3 UI/UX Requirements
- **Loading State:** Show progress indicator during analysis (estimated 15-30s)
- **Error Handling:** Clear error messages for:
  - Invalid PDF format
  - File too large (>5MB)
  - Corrupted or scanned PDF (no extractable text)
  - Empty job description
- **Responsive Design:** Work on desktop and tablet (mobile optional)
- **Color Coding:**
  - Green (80-100): Excellent
  - Yellow (60-79): Good, needs improvement
  - Red (0-59): Poor, major changes needed
- **Interactivity:**
  - Collapsible sections for detailed analysis
  - Copy button for improved bullets
  - Downloadable PDF/Word report

---

## 5. Technical Specifications

### 5.1 Development Environment
- **Python Version:** 3.9+
- **GPU:** Google Colab with T4 GPU (for training)
- **RAM Requirements:** Minimum 8GB for inference, 16GB+ for training
- **Storage:** ~5GB for models and datasets

### 5.2 Dependencies
```txt
# Core ML/NLP
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
peft>=0.7.0  # For LoRA
accelerate>=0.24.0

# PDF Processing
PyMuPDF>=1.23.0
pypdf>=3.17.0

# Data & Metrics
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Web Interface
streamlit>=1.28.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
```

### 5.3 Model Files & Artifacts
```
models/
├── embeddings/
│   └── all-MiniLM-L6-v2/  # Downloaded from HF
├── fine-tuned/
│   ├── phi-3-mini-lora/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   └── tokenizer/
│   └── training_logs/
└── cache/
```

### 5.4 Project Structure
```
ats-resume-checker/
├── README.md
├── PRD.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned training data
│   └── examples/               # Sample resumes & JDs
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration constants
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py   # PDF → text
│   │   └── resume_parser.py   # Text → structured data
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Semantic embedding generation
│   │   ├── similarity.py      # Cosine similarity calculations
│   │   └── formatting_check.py # Rule-based formatting analysis
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── llm_inference.py   # Load & run fine-tuned model
│   │   └── prompt_templates.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── validators.py
│
├── training/
│   ├── prepare_dataset.py      # Dataset preprocessing for training
│   ├── train_lora.py           # LoRA fine-tuning script
│   └── evaluate_model.py       # Model evaluation
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── components/
│   │   ├── upload_section.py
│   │   ├── results_display.py
│   │   └── score_visualizer.py
│   └── static/
│       ├── styles.css
│       └── logo.png
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_parser.py
│   └── test_similarity.py
│
└── docs/
    ├── API.md
    ├── TRAINING.md
    └── USAGE.md
```

---

## 6. Data Requirements

### 6.1 Training Dataset

**Primary Sources:**
1. **Resume Parsing & Summarizer Dataset** (Kaggle)
   - Link: Search "Resume Parsing Kaggle"
   - Expected content: Parsed resume sections, skills, experience

2. **resume-ats-score-v1-en** (Hugging Face)
   - Expected content: Resume texts with ATS scores

**Dataset Preparation Steps:**
1. Download and combine datasets
2. Clean and normalize text (remove special characters, normalize spacing)
3. Create synthetic job descriptions if not present:
   - Extract skills from resumes
   - Generate matching JDs using GPT/Claude
4. Label data:
   - Calculate baseline ATS scores using similarity metrics
   - Manually review and adjust 10-20% of samples
   - Identify weak bullets and create improvements
5. Split: 80% train, 10% validation, 10% test
6. Format as JSON:
```json
{
    "resume": "...",
    "job_description": "...",
    "ats_score": 75,
    "missing_keywords": ["Python", "AWS"],
    "weak_bullets": [...],
    "improved_bullets": [...]
}
```

**Minimum Dataset Size:**
- Training: 400 examples
- Validation: 50 examples
- Test: 50 examples

### 6.2 Test Data
- Collect 10-15 real anonymized resumes from team members, peers
- Source 10-15 real job descriptions from LinkedIn, Indeed
- Create ground truth labels manually

---

## 7. Algorithm Details

### 7.1 ATS Score Calculation

**Components (Weighted Average):**
```python
ats_score = (
    keyword_match_score * 0.40 +
    formatting_score * 0.25 +
    experience_relevance_score * 0.20 +
    bullet_quality_score * 0.15
)
```

**Keyword Match Score (0-100):**
```python
# Extract key skills/requirements from JD
jd_keywords = extract_keywords(job_description)

# Calculate coverage
matched = count_matched_keywords(resume, jd_keywords)
keyword_match_score = (matched / len(jd_keywords)) * 100
```

**Formatting Score (0-100):**
```python
score = 100
if has_tables: score -= 20
if has_images: score -= 15
if has_text_boxes: score -= 15
if has_columns: score -= 10
if has_special_chars: score -= 10
if has_headers_footers: score -= 10
return max(0, score)
```

**Experience Relevance Score (0-100):**
```python
# Average cosine similarity between resume bullets and JD
similarities = [cosine_sim(bullet, jd) for bullet in resume_bullets]
experience_relevance_score = mean(similarities) * 100
```

**Bullet Quality Score (0-100):**
```python
# Check for action verbs, metrics, specificity
quality_scores = [score_bullet(bullet) for bullet in bullets]
bullet_quality_score = mean(quality_scores) * 100
```

### 7.2 Missing Keyword Detection
```python
def identify_missing_keywords(resume_embedding, jd_keywords):
    missing = []
    for keyword in jd_keywords:
        keyword_embedding = embed(keyword)
        max_similarity = max([
            cosine_similarity(keyword_embedding, resume_chunk_embedding)
            for resume_chunk_embedding in resume_embeddings
        ])
        if max_similarity < THRESHOLD:  # 0.3
            missing.append(keyword)
    return missing
```

### 7.3 Weak Bullet Detection
```python
def identify_weak_bullets(bullets, job_description):
    weak = []
    jd_embedding = embed(job_description)
    
    for bullet in bullets:
        bullet_embedding = embed(bullet)
        similarity = cosine_similarity(bullet_embedding, jd_embedding)
        
        quality_score = assess_bullet_quality(bullet)
        
        if similarity < 0.4 or quality_score < 50:
            weak.append({
                "text": bullet,
                "similarity": similarity,
                "quality_score": quality_score,
                "reasons": get_weakness_reasons(bullet, quality_score)
            })
    
    return weak

def assess_bullet_quality(bullet):
    score = 0
    
    # Has action verb
    if starts_with_action_verb(bullet): score += 30
    
    # Has metrics/numbers
    if contains_numbers(bullet): score += 30
    
    # Length appropriate (10-25 words)
    word_count = len(bullet.split())
    if 10 <= word_count <= 25: score += 20
    
    # Specific (not vague)
    if not is_vague(bullet): score += 20
    
    return score
```

---

## 8. Training Process

### 8.1 Fine-Tuning Pipeline

**Step 1: Environment Setup**
```bash
# In Google Colab or local GPU machine
pip install -q -U transformers peft accelerate
```

**Step 2: Load Base Model**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "microsoft/Phi-3-mini-4k-instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Step 3: Configure LoRA**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(base_model, lora_config)
```

**Step 4: Prepare Dataset**
```python
# Format: instruction-following format
def format_training_example(example):
    instruction = f"""Analyze this resume against the job description.

Resume:
{example['resume']}

Job Description:
{example['job_description']}

Provide ATS analysis in JSON format."""

    output = json.dumps({
        "ats_score": example['ats_score'],
        "missing_keywords": example['missing_keywords'],
        "weak_bullets": example['weak_bullets'],
        "improved_bullets": example['improved_bullets']
    })
    
    return {
        "text": f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
    }
```

**Step 5: Train**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./phi3-ats-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

**Step 6: Save Model**
```python
model.save_pretrained("./models/fine-tuned/phi-3-mini-lora")
tokenizer.save_pretrained("./models/fine-tuned/phi-3-mini-lora")
```

### 8.2 Evaluation Metrics

**Quantitative Metrics:**
1. **Score Accuracy:** Mean Absolute Error between predicted and ground truth ATS scores
   - Target: MAE < 10 points
2. **Keyword Precision/Recall:** How accurately missing keywords are identified
   - Target: F1 > 0.75
3. **BLEU/ROUGE Score:** For bullet point improvements
   - Target: ROUGE-L > 0.6

**Qualitative Evaluation:**
- Manual review of 20 test cases
- Check if improvements are actionable
- Verify recommendations make sense

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure
- [ ] Implement PDF extraction module
- [ ] Implement resume parser
- [ ] Build basic Streamlit UI (upload + display text)
- [ ] Test with sample resumes

**Deliverable:** Working PDF upload and text extraction

### Phase 2: Analysis Pipeline (Week 3-4)
- [ ] Implement semantic embedding generation
- [ ] Build similarity calculation module
- [ ] Implement formatting checker
- [ ] Calculate basic ATS scores (without LLM)
- [ ] Display results in UI

**Deliverable:** Working ATS score without suggestions

### Phase 3: Model Training (Week 5-6)
- [ ] Collect and prepare training dataset
- [ ] Fine-tune Phi-3-mini with LoRA
- [ ] Evaluate model performance
- [ ] Iterate on prompt engineering
- [ ] Save model checkpoints

**Deliverable:** Fine-tuned model weights

### Phase 4: Integration (Week 7)
- [ ] Integrate fine-tuned model into pipeline
- [ ] Generate bullet point improvements
- [ ] Complete dashboard UI
- [ ] Add export functionality
- [ ] Implement error handling

**Deliverable:** Fully functional application

### Phase 5: Testing & Documentation (Week 8)
- [ ] Test with real resumes
- [ ] Fix bugs and edge cases
- [ ] Write technical documentation
- [ ] Create demo video
- [ ] Prepare final presentation

**Deliverable:** Final submission package

---

## 10. Testing Requirements

### 10.1 Unit Tests
```python
# test_pdf_extractor.py
def test_extract_text_from_valid_pdf():
    text = extract_text("sample_resume.pdf")
    assert len(text) > 0
    assert "Experience" in text

# test_parser.py
def test_section_detection():
    sections = parse_resume(sample_text)
    assert "experience" in sections
    assert len(sections["experience"]["bullets"]) > 0

# test_similarity.py
def test_cosine_similarity():
    sim = calculate_similarity("Python developer", "Python programming")
    assert sim > 0.7
```

### 10.2 Integration Tests
- Upload PDF → Extract → Parse → Analyze → Display results
- Test with various resume formats (single-column, two-column, different fonts)
- Test with long job descriptions (>1000 words)
- Test with resumes in different experience levels (entry, mid, senior)

### 10.3 Edge Cases
- Scanned PDF (no extractable text) → Show error
- Resume with no bullet points → Analyze paragraphs
- Empty job description → Prompt user
- Very short resume (<100 words) → Warn user
- Resume with only contact info → Handle gracefully

---

## 11. Deployment Considerations

### 11.1 Hosting Options
**For Course Project:**
- Streamlit Community Cloud (Free, easy deployment)
- Google Colab (For demo, not production)
- Local deployment with instructions

**For Production (Future):**
- AWS EC2 with GPU (for model inference)
- Dockerized application
- CDN for model weights

### 11.2 Performance Optimization
- Cache embedding model in memory
- Batch process multiple bullets together
- Use quantized model (8-bit) for faster inference
- Implement timeout for long-running requests (>60s)

### 11.3 Security & Privacy
- Don't store uploaded resumes permanently
- Clear temporary files after analysis
- Add disclaimer about data privacy
- No PII logging

---

## 12. Limitations & Future Enhancements

### 12.1 Current Limitations
1. **PDF Only:** Does not support Word documents
2. **English Only:** No multi-language support
3. **Single Resume:** No batch processing
4. **No ATS-Specific Rules:** Different ATS systems have different requirements
5. **Limited Context:** Model has 4k token context limit

### 12.2 Future Enhancements
- Support for .docx files
- Batch resume analysis
- A/B testing different resume versions
- Integration with job boards (LinkedIn, Indeed)
- Browser extension for one-click analysis
- ATS-specific optimization (Taleo, Workday, Greenhouse)
- Multi-language support
- Resume builder with real-time ATS scoring

---

## 13. Success Metrics

### 13.1 Technical Metrics
- ✅ ATS score accuracy: MAE < 10 points
- ✅ Processing time: < 30 seconds per resume
- ✅ Keyword detection F1: > 0.75
- ✅ Application uptime: > 95%

### 13.2 User Experience Metrics
- ✅ Clear, actionable feedback provided
- ✅ UI intuitive enough for first-time users
- ✅ Results easy to understand without domain knowledge

### 13.3 Academic Metrics
- ✅ Demonstrates understanding of NLP concepts
- ✅ Proper implementation of fine-tuning techniques
- ✅ Comprehensive documentation
- ✅ Successful demo presentation

---

## 14. Documentation Deliverables

### 14.1 Technical Documentation
1. **README.md**
   - Project overview
   - Installation instructions
   - Usage guide
   - Example outputs

2. **TRAINING.md**
   - Dataset preparation steps
   - Training procedure
   - Hyperparameter choices
   - Evaluation results

3. **API.md**
   - Module documentation
   - Function signatures
   - Example code snippets

### 14.2 User Documentation
1. **USAGE_GUIDE.md**
   - How to prepare resume for upload
   - How to interpret results
   - How to apply suggestions

2. **FAQ.md**
   - Common questions
   - Troubleshooting

### 14.3 Academic Report
- System architecture diagram
- Dataset statistics
- Training curves
- Evaluation metrics
- Sample outputs
- Limitations and discussion
- References

---

## 15. Timeline & Milestones

| Week | Tasks | Milestone |
|------|-------|-----------|
| 1-2 | Setup + PDF extraction + parsing | ✓ Text extraction working |
| 3-4 | Similarity analysis + formatting checker | ✓ Basic ATS score |
| 5-6 | Dataset prep + model fine-tuning | ✓ Model trained |
| 7 | Integration + UI completion | ✓ Full app working |
| 8 | Testing + documentation + demo | ✓ Final submission |

---
