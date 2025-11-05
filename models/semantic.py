"""
This module leverages BERT - sentence-transformers/bert-base-nli-mean-tokens
to compute cosine similarity.
"""

from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

class SemanticAnalyzer:
    """
    A class to calculate semantic similarity between two texts using a SentenceTransformer model.
    """

    def __init__(self):
        self.model = None  # Lazy load model

    def _load_model(self):
        """Load model only when first used to avoid slow container boot"""
        if self.model is None:
            print("⚙️ Loading SentenceTransformer model...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
            print("✅ Model loaded successfully!")

    def calculate_similarity(self, text1: str, text2: str) -> dict:
        """
        Calculates the cosine similarity between two texts.
        """
        self._load_model()  # ensures model loads only when needed

        embeddings = self.model.encode([text1, text2])

        cosine_sim = cosine_similarity(
            [embeddings[0]], 
            [embeddings[1]]
        )[0][0]

        return {
            "cosine_similarity": float(cosine_sim),
        }
