"""Sentence embedding generator with multiple fallback options."""

import logging
import os
import time
from typing import Union, List

import numpy as np

from ..config import EMBEDDING_MODEL, HF_CACHE_DIR, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubles each retry


class EmbeddingGenerator:
    """
    Singleton wrapper for text embeddings.
    
    Tries backends in order:
    1. SentenceTransformer (local, free)
    2. HuggingFace Transformers direct (local, free - fallback)
    3. OpenAI API (if key provided)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._backend = None
        self._model = None
        self._tokenizer = None
        self._openai_client = None
        
        # Try backends in order
        if self._try_sentence_transformer():
            self._backend = "sentence_transformers"
        elif self._try_huggingface_direct():
            self._backend = "huggingface"
        elif self._try_openai():
            self._backend = "openai"
        else:
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers/transformers or set OPENAI_API_KEY."
            )
        
        self._initialized = True
        logger.info(f"Embedding backend initialized: {self._backend}")

    def _try_sentence_transformer(self) -> bool:
        """Attempt to load SentenceTransformer model."""
        try:
            # Suppress codecarbon warnings
            os.environ["CODECARBON_LOG_LEVEL"] = "error"
            
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=str(HF_CACHE_DIR),
            )
            logger.info("SentenceTransformer loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable: {e}")
            return False

    def _try_huggingface_direct(self) -> bool:
        """Attempt to load model directly via HuggingFace transformers."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Use the full model name
            model_name = EMBEDDING_MODEL
            logger.info(f"Loading HuggingFace model directly: {model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(HF_CACHE_DIR),
            )
            self._model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(HF_CACHE_DIR),
            )
            self._model.eval()
            logger.info("HuggingFace model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"HuggingFace direct loading failed: {e}")
            return False

    def _try_openai(self) -> bool:
        """Attempt to initialize OpenAI client."""
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found — fallback unavailable")
            return False
        
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized (model: {OPENAI_EMBEDDING_MODEL})")
            return True
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            return False

    @property
    def backend(self) -> str:
        """Return the active embedding backend name."""
        return self._backend

    def generate(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for one or more texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self._backend == "sentence_transformers":
            return self._model.encode(texts, show_progress_bar=False)
        elif self._backend == "huggingface":
            return self._generate_huggingface(texts)
        elif self._backend == "openai":
            return self._generate_openai(texts)
        else:
            raise RuntimeError("No embedding backend available")

    def _generate_huggingface(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace transformers directly."""
        import torch
        
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**encoded)
            # Mean pooling over token embeddings
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        
        return embeddings.numpy()

    def _generate_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via OpenAI API with retry logic for rate limits."""
        from openai import RateLimitError, APIError
        
        delay = RETRY_DELAY
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._openai_client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=texts,
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)
            
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            
            except APIError as e:
                last_error = e
                logger.warning(f"OpenAI API error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
        
        raise RuntimeError(f"OpenAI embedding failed after {MAX_RETRIES} retries: {last_error}")
