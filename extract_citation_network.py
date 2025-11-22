"""
Extract Citation Network from Neo4j

Builds adjacency matrix and feature matrix for HGCN training.
"""

from neo4j import GraphDatabase
import numpy as np
import torch
import pickle
import os
from dotenv import load_dotenv

load_dotenv()


class CitationNetworkExtractor:
    """Extract citation graph from Neo4j for HGCN training"""
    
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def extract_citations(self):
        """
        Extract all citation edges from Neo4j.
        
        Returns:
            edges: List of (source_id, target_id) tuples
            case_ids: List of unique case IDs
        """
        with self.driver.session() as session:
            # Query for citation edges
            result = session.run("""
                MATCH (citing:Case)-[:CITES]->(cited:Case)
                RETURN citing.id AS source, cited.id AS target
            """)
            
            edges = [(record['source'], record['target']) for record in result]
            
            if not edges:
                print("⚠️  No citation edges found in Neo4j")
                print("Attempting to extract from Toulmin graph...")
                
                # Fallback: Use Toulmin argument structure
                result = session.run("""
                    MATCH (d:Data)-[:SUPPORTS]->(c:Claim)
                    RETURN d.case_id AS source, c.case_id AS target
                    UNION
                    MATCH (w:Warrant)-[:WARRANTS]->(c:Claim)
                    RETURN w.case_id AS source, c.case_id AS target
                """)
                
                edges = [(record['source'], record['target']) 
                        for record in result if record['source'] and record['target']]
            
            # Get unique case IDs
            case_ids = list(set([src for src, _ in edges] + [tgt for _, tgt in edges]))
            
            return edges, case_ids
    
    def build_adjacency_matrix(self, edges, case_ids):
        """
        Build adjacency matrix from edge list.
        
        Args:
            edges: List of (source_id, target_id)
            case_ids: List of all unique case IDs
        
        Returns:
            adj: Sparse adjacency matrix (N x N)
            id_to_idx: Mapping from case_id to matrix index
        """
        # Create ID to index mapping
        id_to_idx = {case_id: idx for idx, case_id in enumerate(case_ids)}
        n = len(case_ids)
        
        # Build adjacency matrix
        adj = np.zeros((n, n))
        
        for source, target in edges:
            if source in id_to_idx and target in id_to_idx:
                src_idx = id_to_idx[source]
                tgt_idx = id_to_idx[target]
                adj[src_idx, tgt_idx] = 1
                adj[tgt_idx, src_idx] = 1  # Undirected
        
        return adj, id_to_idx
    
    def extract_metadata(self, case_ids):
        """
        Extract metadata (court level) for validation.
        
        Returns:
            metadata: Dict mapping case_id to {'court': str, 'date': int}
        """
        with self.driver.session() as session:
            metadata = {}
            
            for case_id in case_ids:
                # Try to extract court level from case ID
                court_level = self._infer_court_level(case_id)
                metadata[case_id] = {'court': court_level, 'date': None}
            
            return metadata
    
    def _infer_court_level(self, case_id):
        """Infer court level from case ID string"""
        case_id_lower = case_id.lower()
        
        if 'supreme' in case_id_lower or 'sc_' in case_id_lower:
            return 'Supreme Court'
        elif 'high' in case_id_lower or '_hc' in case_id_lower:
            return 'High Court'
        else:
            return 'Lower Court'
    
    def close(self):
        self.driver.close()


def sample_negative_edges(positive_edges, num_nodes, num_negatives_per_positive=5):
    """
    Sample negative edges for contrastive learning.
    
    Args:
        positive_edges: List of (source, target) positive edges
        num_nodes: Total number of nodes
        num_negatives_per_positive: Ratio of negative to positive samples
    
    Returns:
        negative_edges: numpy array of (source, target) for non-existent edges
    """
    positive_set = set([(src, tgt) for src, tgt in positive_edges])
    
    negative_edges = []
    num_negatives = len(positive_edges) * num_negatives_per_positive
    
    while len(negative_edges) < num_negatives:
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        
        if src != tgt and (src, tgt) not in positive_set:
            negative_edges.append((src, tgt))
    
    return np.array(negative_edges)


def load_case_features(case_ids, embeddings_cache_path='data/case_embeddings_cache.pkl'):
    """
    Load Gemini embeddings for cases as initial features.
    
    Args:
        case_ids: List of case IDs
        embeddings_cache_path: Path to cached embeddings
    
    Returns:
        features: numpy array (num_cases x embedding_dim)
    """
    # Load cached embeddings
    if os.path.exists(embeddings_cache_path):
        with open(embeddings_cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        embeddings = cache.get('embeddings', [])
        docs = cache.get('docs', [])
        
        # Create mapping from case_id to embedding
        id_to_emb = {}
        for doc, emb in zip(docs, embeddings):
            case_id = doc.metadata.get('id')
            if case_id:
                id_to_emb[case_id] = emb
        
        # Build feature matrix
        features = []
        for case_id in case_ids:
            if case_id in id_to_emb:
                features.append(id_to_emb[case_id])
            else:
                # Random initialization for missing embeddings
                features.append(np.random.randn(768) * 0.01)
        
        features = np.array(features, dtype=np.float32)
        print(f"✓ Loaded features for {len(case_ids)} cases (dim={features.shape[1]})")
        
        return features
    else:
        print(f"⚠️  Embeddings cache not found at {embeddings_cache_path}")
        print("Using random initialization")
        return np.random.randn(len(case_ids), 768).astype(np.float32) * 0.01


if __name__ == "__main__":
    print("="*80)
    print("CITATION NETWORK EXTRACTION")
    print("="*80)
    
    extractor = CitationNetworkExtractor()
    
    # Extract citations
    print("\n1. Extracting citation edges from Neo4j...")
    edges, case_ids = extractor.extract_citations()
    print(f"   ✓ Found {len(edges)} citation edges")
    print(f"   ✓ Unique cases: {len(case_ids)}")
    
    # Build adjacency matrix
    print("\n2. Building adjacency matrix...")
    adj, id_to_idx = extractor.build_adjacency_matrix(edges, case_ids)
    print(f"   ✓ Adjacency matrix: {adj.shape}")
    print(f"   ✓ Density: {np.sum(adj) / (adj.shape[0] * adj.shape[1]):.4f}")
    
    # Load features
    print("\n3. Loading case features...")
    features = load_case_features(case_ids)
    print(f"   ✓ Feature matrix: {features.shape}")
    
    # Extract metadata
    print("\n4. Extracting metadata...")
    metadata = extractor.extract_metadata(case_ids)
    court_counts = {}
    for case_id, meta in metadata.items():
        court = meta['court']
        court_counts[court] = court_counts.get(court, 0) + 1
    
    print("   Court distribution:")
    for court, count in sorted(court_counts.items()):
        print(f"     - {court}: {count}")
    
    # Save data
    print("\n5. Saving processed data...")
    data = {
        'edges': edges,
        'case_ids': case_ids,
        'adj': adj,
        'id_to_idx': id_to_idx,
        'features': features,
        'metadata': metadata
    }
    
    with open('data/citation_network.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"   ✓ Saved to data/citation_network.pkl")
    
    extractor.close()
    print("\n✅ Citation network extraction complete!")
