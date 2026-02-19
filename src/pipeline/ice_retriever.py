"""Retrieve the most similar in-context examples for a given question.

Uses pre-computed sentence embeddings (all-mpnet-base-v2) and cosine similarity
to select the K most relevant training examples per test question.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class ICERetriever:
    def __init__(self, ice_file: str, embeddings_file: str, k: int = 6):
        with open(ice_file) as f:
            self.ices = json.load(f)
        self.embeddings = np.load(embeddings_file)  # (N, D), already normalized
        self.k = k
        self.model = SentenceTransformer("all-mpnet-base-v2")
        print(f"ICERetriever: loaded {len(self.ices)} ICEs, k={k}")

    def retrieve(self, question: str) -> list:
        """Return the k most similar ICEs for the given question."""
        q_emb = self.model.encode([question], normalize_embeddings=True)  # (1, D)
        similarities = q_emb @ self.embeddings.T  # (1, N)
        top_k_indices = np.argsort(similarities[0])[::-1][:self.k]
        return [self.ices[i] for i in top_k_indices]
