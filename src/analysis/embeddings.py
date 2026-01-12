from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
from ..config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Singleton-style class to load the model once and generate embeddings.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            try:
                self._model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def generate(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for a text or list of texts.
        """
        try:
            return self._model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
