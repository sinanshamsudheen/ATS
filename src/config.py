import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FINETUNED_DIR = MODELS_DIR / "fine-tuned"
HF_CACHE_DIR = MODELS_DIR / "huggingface_cache"

os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------
APP_TITLE = "ATS Resume Compliance Checker"
APP_VERSION = "3.0.0"

# ---------------------------------------------------------------------------
# Embedding / similarity
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# ATS formatting rules (used by scoring.formatting_check)
# ---------------------------------------------------------------------------
ATS_FORMAT_RULES = {
    "max_pages": 2,
    "max_images": 3,
    "min_text_length": 200,
}

# ---------------------------------------------------------------------------
# Bullet rewriter
# ---------------------------------------------------------------------------
WEAK_BULLET_MIN_WORDS = 5
WEAK_BULLET_MAX_WORDS = 40
ACTION_VERBS = [
    # Leadership
    "managed", "led", "directed", "supervised", "coordinated", "orchestrated", "guided",
    # Creation
    "developed", "created", "built", "designed", "engineered", "architected", "established",
    "implemented", "deployed", "launched", "initiated", "founded", "formulated", "constructed",
    # Improvement
    "improved", "optimized", "enhanced", "streamlined", "accelerated", "increased", "reduced",
    "minimized", "maximized", "boosted", "upgraded", "refined", "strengthened",
    # Analysis
    "analyzed", "researched", "investigated", "evaluated", "assessed", "identified", "diagnosed",
    # Achievement
    "achieved", "delivered", "generated", "produced", "exceeded", "outperformed", "secured",
    # Innovation
    "pioneered", "spearheaded", "transformed", "revolutionized", "innovated", "strategized",
]

# ---------------------------------------------------------------------------
# Fine-tuned Phi-3 LoRA model
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LORA_ADAPTER_PATH = str(BASE_DIR / "colab" / "ats_phi_lora")
