from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
from .embeddings import EmbeddingGenerator
import logging
from ..config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def analyze_keywords(self, resume_text: str, jd_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze which JD keywords are missing from the resume using semantic similarity.
        """
        if not jd_keywords:
            return {"missing": [], "matched": [], "score": 0.0}

        # Embed resume (chunks or full text?)
        # For efficiency and accuracy, we should maybe split resume into chunks (sentences/bullets)
        # But simpler approach: Embed full resume vs keywords? No, that's bad.
        # Better: Embed all chunks of resume, then for each keyword, find max similarity.
        
        # Split resume into chunks (simple approach: lines/sentences)
        resume_chunks = [line.strip() for line in resume_text.split('\n') if len(line.strip()) > 10]
        if not resume_chunks:
             return {"missing": jd_keywords, "matched": [], "score": 0.0}
             
        resume_embeddings = self.embedder.generate(resume_chunks)
        keyword_embeddings = self.embedder.generate(jd_keywords)
        
        # Calculate similarity matrix: (n_keywords, n_resume_chunks)
        sim_matrix = cosine_similarity(keyword_embeddings, resume_embeddings)
        
        # For each keyword, find max similarity across all resume chunks
        max_sims = np.max(sim_matrix, axis=1)
        
        missing = []
        matched = []
        
        for i, keyword in enumerate(jd_keywords):
            score = max_sims[i]
            if score < SIMILARITY_THRESHOLD:
                missing.append(keyword)
            else:
                matched.append(keyword)
        
        # Calculate score (percentage of matched keywords)
        match_score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 0.0
        
        return {
            "missing": missing,
            "matched": matched,
            "score": round(match_score, 1)
        }

    def calculate_overall_match(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate overall semantic similarity between full resume and full JD.
        """
        # Truncate if too long for model (transformer limit usually 512 tokens, but SentenceTransformer handles truncation)
        resume_emb = self.embedder.generate(resume_text)
        jd_emb = self.embedder.generate(jd_text)
        
        # Reshape for sklearn
        score = cosine_similarity(resume_emb.reshape(1, -1), jd_emb.reshape(1, -1))[0][0]
        return round(score * 100, 1)
