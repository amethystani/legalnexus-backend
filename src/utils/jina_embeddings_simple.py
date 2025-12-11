"""
Simple Jina Embeddings using SentenceTransformer

Bypasses the custom Transformers loading that's causing issues.
"""
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class JinaEmbeddingsSimple:
    def __init__(self, model_path: str = "jinaai/jina-embeddings-v3"):
        """
        Initialize Jina Embeddings using SentenceTransformer.
        
        Args:
            model_path: Model name or path
        """
        print(f"Loading {model_path} with SentenceTransformer...")
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        print(f"✓ Loaded {model_path}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
