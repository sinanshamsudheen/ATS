"""Sentence embedding generator using SentenceTransformers."""

import logging
from typing import Union, List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL, HF_CACHE_DIR

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Singleton wrapper around SentenceTransformer for text embeddings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=str(HF_CACHE_DIR),
            )
            self._initialized = True
            logger.info("Embedding model loaded successfully")

    def generate(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for one or more texts."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False)
