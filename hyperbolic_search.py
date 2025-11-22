"""
Hyperbolic Search Engine

Uses Poincaré distance instead of cosine similarity for retrieval.
Integrates trained HGCN embeddings into the search pipeline.
"""

import torch
import numpy as np
import pickle
import os
from geoopt import PoincareBall
from typing import List, Tuple


class HyperbolicSearchEngine:
    """
    Search engine using hyperbolic embeddings.
    
    Replaces cosine similarity with Poincaré distance.
    """
    
    def __init__(self, embeddings_path='models/hgcn_embeddings.pkl', c=1.0):
        """
        Initialize hyperbolic search engine.
        
        Args:
            embeddings_path: Path to pre-computed hyperbolic embeddings
            c: Curvature of Poincaré Ball
        """
        self.manifold = PoincareBall(c=c)
        self.c = c
        
        # Load pre-computed embeddings
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"✓ Loaded {len(self.embeddings)} hyperbolic embeddings")
        else:
            print(f"⚠️  Embeddings not found at {embeddings_path}")
            self.embeddings = {}
    
    def project_query_to_hyperbolic(self, query_embedding: np.ndarray) -> torch.Tensor:
        """
        Project Euclidean query embedding to Poincaré Ball.
        
        Args:
            query_embedding: Euclidean embedding (e.g., from Gemini)
        
        Returns:
            Hyperbolic embedding in Poincaré Ball
        """
        # Convert to tensor
        query_tensor = torch.from_numpy(query_embedding).float()
        
        # Project to hyperbolic space using exponential map at origin
        query_hyp = self.manifold.expmap0(query_tensor)
        
        return query_hyp
    
    def search(self, query_embedding: np.ndarray, candidate_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search using Poincaré distance.
        
        Args:
            query_embedding: Query embedding (Euclidean, e.g., from Gemini)
            candidate_ids: List of candidate case IDs to rank
            top_k: Number of results to return
        
        Returns:
            List of (case_id, distance) tuples, sorted by distance (lower is better)
        """
        # Project query to hyperbolic space
        query_hyp = self.project_query_to_hyperbolic(query_embedding)
        
        results = []
        
        for case_id in candidate_ids:
            if case_id in self.embeddings:
                case_emb = torch.from_numpy(self.embeddings[case_id]).float()
                
                # Calculate Poincaré distance
                dist = self.manifold.dist(query_hyp, case_emb).item()
                
                results.append((case_id, dist))
            else:
                # Case not in hyperbolic embeddings, assign high distance
                results.append((case_id, 10.0))
        
        # Sort by distance (lower = more relevant)
        results.sort(key=lambda x: x[1])
        
        return results[:top_k]
    
    def distance_to_similarity(self, distance: float, max_dist: float = 5.0) -> float:
        """
        Convert hyperbolic distance to similarity score (0-1).
        
        Args:
            distance: Poincaré distance
            max_dist: Maximum expected distance for normalization
        
        Returns:
            Similarity score (1 = most similar, 0 = least similar)
        """
        # Inverse distance with normalization
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def get_hierarchy_info(self, case_id: str) -> dict:
        """
        Get hierarchy information for a case.
        
        Args:
            case_id: Case ID
        
        Returns:
            Dict with radius and inferred court level
        """
        if case_id not in self.embeddings:
            return {'radius': None, 'court_level': 'Unknown'}
        
        emb = torch.from_numpy(self.embeddings[case_id]).float()
        radius = torch.norm(emb).item()
        
        # Infer court level from radius
        # Lower radius → higher court
        if radius < 0.3:
            court_level = 'Supreme Court (inferred)'
        elif radius < 0.6:
            court_level = 'High Court (inferred)'
        else:
            court_level = 'Lower Court (inferred)'
        
        return {
            'radius': radius,
            'court_level': court_level
        }


def integrate_hyperbolic_search(hybrid_system, hyperbolic_engine):
    """
    Add hyperbolic search as a component in the hybrid search pipeline.
    
    This function patches the NovelHybridSearchSystem to include hyperbolic distance.
    """
    hybrid_system.hyperbolic_engine = hyperbolic_engine
    
    # Store original hybrid_search method
    original_hybrid_search = hybrid_system.hybrid_search
    
    def new_hybrid_search(query: str, top_k: int = 5, weights: dict = None):
        """
        Enhanced hybrid search with hyperbolic component.
        
        weights: {
            'semantic': 0.25,
            'citation': 0.0,
            'text': 0.25,
            'hyperbolic': 0.5  # NEW
        }
        """
        if weights is None:
            weights = {
                'semantic': 0.25,
                'citation': 0.0,
                'text': 0.25,
                'hyperbolic': 0.5  # Give priority to hierarchical structure
            }
        
        # Get query embedding
        query_embedding = hybrid_system.embeddings_model.embed_query(query)
        
        # Retrieve candidates using existing methods
        # (This will use semantic + text similarity)
        semantic_results = hybrid_system.semantic_search(query, top_k=20)
        text_results = hybrid_system.text_pattern_search(query, top_k=20)
        
        # Combine candidate IDs
        candidate_ids = list(set(
            [doc.metadata.get('id') for doc, _ in semantic_results if doc.metadata.get('id')] +
            [doc.metadata.get('id') for doc, _ in text_results if doc.metadata.get('id')]
        ))
        
        # Hyperbolic search on candidates
        hyp_results = hybrid_system.hyperbolic_engine.search(
            query_embedding, candidate_ids, top_k=len(candidate_ids)
        )
        
        # Create score dictionaries
        semantic_scores = {doc.metadata.get('id'): score for doc, score in semantic_results}
        text_scores = {doc.metadata.get('id'): score for doc, score in text_results}
        hyp_scores = {case_id: hybrid_system.hyperbolic_engine.distance_to_similarity(dist) 
                     for case_id, dist in hyp_results}
        
        # Combined scoring
        final_scores = {}
        for case_id in candidate_ids:
            score = (
                weights.get('semantic', 0.25) * semantic_scores.get(case_id, 0) +
                weights.get('text', 0.25) * text_scores.get(case_id, 0) +
                weights.get('hyperbolic', 0.5) * hyp_scores.get(case_id, 0)
            )
            final_scores[case_id] = score
        
        # Sort and return top K
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Convert back to document format
        id_to_doc = {doc.metadata.get('id'): doc for doc, _ in semantic_results + text_results}
        
        final_results = []
        for case_id, score in ranked:
            if case_id in id_to_doc:
                final_results.append((id_to_doc[case_id], score))
        
        return final_results
    
    # Replace method
    hybrid_system.hybrid_search_hyperbolic = new_hybrid_search
    
    print("✓ Hyperbolic search integrated into hybrid system")
    print("  Use system.hybrid_search_hyperbolic(query) for hyperbolic-enhanced search")


if __name__ == "__main__":
    # Test hyperbolic search
    engine = HyperbolicSearchEngine()
    
    # Example: Test with random query
    if engine.embeddings:
        print("\nTesting hyperbolic search...")
        query_emb = np.random.randn(768).astype(np.float32) * 0.01
        candidate_ids = list(engine.embeddings.keys())[:10]
        
        results = engine.search(query_emb, candidate_ids, top_k=5)
        
        print(f"\nTop 5 results:")
        for i, (case_id, dist) in enumerate(results, 1):
            hierarchy = engine.get_hierarchy_info(case_id)
            print(f"  {i}. {case_id}")
            print(f"     Distance: {dist:.4f}, Radius: {hierarchy['radius']:.4f}")
            print(f"     Inferred level: {hierarchy['court_level']}")
