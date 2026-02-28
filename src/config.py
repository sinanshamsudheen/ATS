import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FINETUNED_DIR = MODELS_DIR / "fine-tuned"
HF_CACHE_DIR = MODELS_DIR / "huggingface_cache"

# Set HuggingFace cache to project directory (for sentence-transformers)
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR)

# Create cache directory if it doesn't exist
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# App Settings
APP_TITLE = "ATS Resume Compliance Checker"
APP_VERSION = "2.0.0"  # Phase 2: LLM-powered optimizer

# Analysis Config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.3

# OpenAI API Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable or .env file
OPENAI_MODEL = "gpt-4o-mini"  # Lightweight, fast, cost-effective model
LLM_MAX_TOKENS = 300  # Max tokens for bullet rewrite responses
LLM_TEMPERATURE = 0.7  # Creativity level (0.0 = deterministic, 1.0 = creative)

# Rewrite Engine Config
WEAK_BULLET_MIN_WORDS = 5  # Bullets shorter than this are flagged as weak
WEAK_BULLET_MAX_WORDS = 40  # Bullets longer than this may need splitting
MIN_ACTION_VERBS = [
    # Leadership & Management
    "managed", "led", "directed", "supervised", "coordinated", "orchestrated", "guided",
    # Creation & Development
    "developed", "created", "built", "designed", "engineered", "architected", "established",
    "implemented", "deployed", "launched", "initiated", "founded", "formulated", "constructed",
    # Improvement & Optimization
    "improved", "optimized", "enhanced", "streamlined", "accelerated", "increased", "reduced",
    "minimized", "maximized", "boosted", "upgraded", "refined", "strengthened",
    # Analysis & Research
    "analyzed", "researched", "investigated", "evaluated", "assessed", "identified", "diagnosed",
    # Achievement & Results
    "achieved", "delivered", "generated", "produced", "exceeded", "outperformed", "secured",
    # Innovation & Strategy
    "pioneered", "spearheaded", "transformed", "revolutionized", "innovated", "strategized"
]
