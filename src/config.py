import os
import warnings
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FINETUNED_DIR = MODELS_DIR / "fine-tuned"
HF_CACHE_DIR = MODELS_DIR / "huggingface_cache"

os.environ['HF_HOME'] = str(HF_CACHE_DIR)
# TRANSFORMERS_CACHE is deprecated since transformers v5 — HF_HOME is the replacement
os.environ.pop('TRANSFORMERS_CACHE', None)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Suppress TensorFlow / oneDNN noise (this project doesn't use TF directly)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')       # hide C++ INFO/WARNING logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')      # disable oneDNN custom ops
os.environ.setdefault('TF_KERAS_LEGACY_DEPRECATION', '0') # suppress tf_keras warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='google')
warnings.filterwarnings('ignore', message='Using `TRANSFORMERS_CACHE`')

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

# OpenAI fallback (used if SentenceTransformer fails)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

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
# Fine-tuned Phi-3 LoRA model (local)
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LORA_ADAPTER_PATH = str(BASE_DIR / "ats_phi_lora")
GGUF_MODEL_PATH = str(MODELS_DIR / "phi3-ats-q8_0.gguf")

# ---------------------------------------------------------------------------
# Groq LLM (cloud - fast inference)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Llama 3.1 8B - fast inference
