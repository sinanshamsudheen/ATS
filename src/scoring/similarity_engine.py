"""Semantic similarity analysis between resume and job description."""

import logging
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingGenerator
from ..config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Computes keyword-level and overall semantic similarity."""

    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def analyze_keywords(self, resume_text: str, jd_keywords: List[str]) -> Dict[str, Any]:
        """
        Determine which JD keywords are semantically present in the resume.

        Returns dict with 'missing', 'matched', and 'score' (% matched).
        """
        if not jd_keywords:
            return {"missing": [], "matched": [], "score": 0.0}

        resume_chunks = [
            line.strip() for line in resume_text.split('\n') if len(line.strip()) > 10
        ]
        if not resume_chunks:
            return {"missing": jd_keywords, "matched": [], "score": 0.0}

        resume_embeddings = self.embedder.generate(resume_chunks)
        keyword_embeddings = self.embedder.generate(jd_keywords)

        sim_matrix = cosine_similarity(keyword_embeddings, resume_embeddings)
        max_sims = np.max(sim_matrix, axis=1)

        missing: List[str] = []
        matched: List[str] = []

        for i, keyword in enumerate(jd_keywords):
            if max_sims[i] < SIMILARITY_THRESHOLD:
                missing.append(keyword)
            else:
                matched.append(keyword)

        match_score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 0.0

        return {
            "missing": missing,
            "matched": matched,
            "score": round(match_score, 1),
        }

    def calculate_overall_match(self, resume_text: str, jd_text: str) -> float:
        """Compute overall semantic similarity (0–100) between resume and JD."""
        resume_emb = self.embedder.generate(resume_text)
        jd_emb = self.embedder.generate(jd_text)
        score = cosine_similarity(
            resume_emb.reshape(1, -1), jd_emb.reshape(1, -1)
        )[0][0]
        return round(score * 100, 1)
