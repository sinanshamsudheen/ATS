# Phase 2 Quick Start Guide

## Getting Started with LLM-Powered Resume Optimization

This guide will help you quickly set up and test the Phase 2 LLM integration.

---

## 1. Environment Setup

### Activate Virtual Environment
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Or Command Prompt
.\venv\Scripts\activate.bat
```

### Install New Dependencies
```bash
pip install -r requirements.txt
```

**What will be installed:**
- `transformers` - Hugging Face library for LLMs
- `accelerate` - Efficient model loading
- `bitsandbytes` - 4-bit quantization support
- `einops` - Tensor operations
- `sentencepiece` - Tokenization
- `protobuf` - Protocol buffers

**Download Size:** ~100MB packages + ~2.3GB model on first run

---

## 2. Quick Test: LLM Loading

Test if Phi-3-mini loads correctly:

```python
# Open Python REPL
python

# Test LLM import and loading
>>> from src.model.llm_inference import get_llm
>>> llm = get_llm()
# This will download Phi-3-mini on first run (~2.3GB)
# Subsequent loads will be from cache

>>> # Test a simple bullet rewrite
>>> result = llm.analyze_bullet("Worked on various projects")
>>> print(result)
```

**Expected Output:**
```python
{
    'original': 'Worked on various projects',
    'analysis': 'Weak action verb, lacks specificity and metrics',
    'improved': 'Led 3 cross-functional projects, delivering...',
    'success': True
}
```

---

## 3. Run the Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

**First Launch:**
1. Navigate to `http://localhost:8501`
2. You'll see the enhanced UI with Phase 2 features
3. Upload a sample resume PDF
4. Paste a job description
5. Enable "AI Rewrites" in the sidebar
6. Click "Analyze & Optimize Resume"

**What to Expect:**
- Progress bar showing analysis stages
- Model loading message (first time: 30-60s)
- Results with 4 tabs:
  - 🎯 **AI Rewrites** - Bullet-by-bullet improvements
  - 🔑 **Keywords** - Missing/matched keywords
  - 📋 **Formatting** - ATS compatibility
  - 📊 **Raw Data** - JSON structure

---

## 4. Understanding the Results

### Metrics Overview
- **Overall Match**: Semantic similarity (resume vs JD)
- **Keyword Coverage**: % of JD keywords found in resume
- **Format Score**: ATS compatibility (0-100)
- **Bullet Quality**: Average quality of all bullets (0-100)

### Quality Scores per Bullet
- **80-100**: Excellent (green)
- **60-79**: Good (yellow)
- **0-59**: Needs improvement (red)

### Analysis Components
Each bullet shows:
- Original text
- Improved version  
- Quality score
- Issues detected (e.g., "No metrics", "Weak action verb")
- AI feedback/reasoning

---

## 5. Performance Optimization

### For Faster Inference

**Option 1: Disable LLM for Quick Tests**
- Uncheck "Enable AI Rewrites" in sidebar
- Still get keyword matching and formatting analysis

**Option 2: GPU Acceleration** (if available)
The code automatically uses GPU if CUDA is available:
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True if GPU available
```

**Option 3: Reduce Bullets Analyzed**
- Check "Only Rewrite Weak Bullets" in sidebar
- Only analyzes bullets scoring <70

---

## 6. Configuration Customization

Edit `src/config.py` to tune behavior:

```python
# LLM Settings
LLM_TEMPERATURE = 0.7  # Lower = more conservative, Higher = more creative
LLM_MAX_LENGTH = 2048  # Context window

# Quality Thresholds
WEAK_BULLET_MIN_WORDS = 5   # Min words for good bullet
WEAK_BULLET_MAX_WORDS = 40  # Max words before "too long"
SIMILARITY_THRESHOLD = 0.3  # Keyword matching sensitivity

# Quantization
USE_QUANTIZATION = True  # Set False to use full precision (needs 8GB+ RAM)
QUANTIZATION_BITS = 4    # Change to 8 for better quality (slower)
```

---

## 7. Testing Sample Data

Use the provided sample training data to understand expected outputs:

```python
import json

# Load sample weak-strong pairs
with open('data/processed/sample_training_data.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# Print first example
print(samples[0])
```

These samples show the **ideal transformation** from weak to strong bullets.

---

## 8. Troubleshooting

### Issue: "Model download failed"
**Solution:** Check internet connection. Phi-3-mini downloads from HuggingFace Hub.

### Issue: "Out of memory"
**Solutions:**
1. Ensure `USE_QUANTIZATION = True` in config
2. Close other applications
3. Reduce batch size (edit `rewrite_engine.py`)

### Issue: "Import errors"
**Solution:** 
```bash
pip install --upgrade transformers accelerate bitsandbytes
```

### Issue: "Slow inference on CPU"
**Expected:** CPU inference takes 20-40s for 5 bullets with 4-bit quantization.
**Solutions:**
- Use GPU if available
- Enable "Only Rewrite Weak Bullets"
- Reduce number of bullets

### Issue: "LLM output not formatted correctly"
The model *sometimes* doesn't follow the ANALYSIS/IMPROVED format perfectly.  
This is expected with few-shot prompting. Fine-tuning would improve consistency.

---

## 9. Next Steps

### Test with Real Resume
1. Export your resume as PDF (ensure ATS-friendly format)
2. Find a real job description
3. Run full analysis
4. Compare original vs improved bullets
5. Iterate and refine

### Provide Feedback
Note any issues for Phase 3 improvements:
- Which suggestions were helpful?
- Which were off-target?
- Any crashes or errors?

### Prepare for Phase 3
Phase 3 focuses on:
- Comprehensive testing
- Performance optimization
- PDF export functionality
- Deployment configuration

---

## 10. Command Cheat Sheet

```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Install/update dependencies
pip install -r requirements.txt

# Run app
streamlit run src/app/streamlit_app.py

# Test LLM in Python
python
>>> from src.model.llm_inference import get_llm
>>> llm = get_llm()
>>> llm.analyze_bullet("Test bullet")

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# View logs
# Streamlit logs appear in terminal

# Stop app
# Ctrl+C in terminal
```

---

## Support

For issues or questions:
1. Check `progress.md` for known issues
2. Review error messages in terminal
3. Check HuggingFace Hub status (for model downloads)
4. Verify all dependencies installed: `pip list`

---

**Happy Optimizing! 🚀**
