"""
Jina Embeddings Wrapper for LangChain

Wraps jinaai/jina-embeddings-v3 using Hugging Face Transformers.
Implements the LangChain Embeddings interface.
"""

from typing import List
from langchain_core.embeddings import Embeddings
from transformers import AutoModel
import torch
import numpy as np

class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/jina-embeddings-v3", task: str = "retrieval.passage"):
        """
        Initialize Jina Embeddings.
        
        Args:
            model_name: Hugging Face model ID
            task: Task adapter to use ('retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching')
        """
        self.model_name = model_name
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        print(f"✓ Loaded {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Jina model handles batching internally or we can loop
        # For safety with large batches, let's process in chunks
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                embeddings = self.model.encode(batch, task=self.task)
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                print(f"Error embedding batch: {e}")
                # Fallback: embed one by one
                for text in batch:
                    try:
                        emb = self.model.encode([text], task=self.task)
                        all_embeddings.extend(emb.tolist())
                    except:
                        all_embeddings.append([0.0] * 1024) # Zero vector fallback
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        # Use retrieval.query task for queries if the main task is retrieval.passage
        query_task = "retrieval.query" if self.task == "retrieval.passage" else self.task
        embedding = self.model.encode([text], task=query_task)
        return embedding[0].tolist()
