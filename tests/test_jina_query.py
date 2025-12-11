"""
Test Jina Embeddings Model on Query

Uses the locally saved Jina model to embed queries and find similar cases.
"""
import pickle
import numpy as np
from jina_embeddings_simple import JinaEmbeddingsSimple

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("="*80)
    print("JINA EMBEDDINGS MODEL TEST")
    print("="*80)
    
    # 1. Load the local Jina model
    print("\n1. Loading Jina model from local directory...")
    try:
        jina_model = JinaEmbeddingsSimple(model_path="models/jina-embeddings-v3")
        print("✓ Successfully loaded Jina model from models/jina-embeddings-v3")
    except Exception as e:
        print(f"❌ Failed to load local model: {e}")
        print("\nTrying to load from HuggingFace...")
        try:
            jina_model = JinaEmbeddingsSimple(model_path="jinaai/jina-embeddings-v3")
            print("✓ Successfully loaded Jina model from HuggingFace")
        except Exception as e2:
            print(f"❌ Failed to load from HuggingFace: {e2}")
            return
    
    # 2. Load cached case embeddings
    print("\n2. Loading case embeddings cache...")
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            case_embeddings = pickle.load(f)
        print(f"✓ Loaded {len(case_embeddings)} case embeddings")
        
        # Show some sample case IDs
        sample_ids = list(case_embeddings.keys())[:5]
        print(f"\nSample case IDs: {sample_ids}")
        
        # Check embedding dimension
        first_emb = list(case_embeddings.values())[0]
        print(f"Embedding dimension: {len(first_emb)}")
    except Exception as e:
        print(f"❌ Failed to load case embeddings: {e}")
        return
    
    # 3. Test query
    test_query = "negligence duty of care breach damages"
    print(f"\n3. Testing query: '{test_query}'")
    
    try:
        query_embedding = jina_model.embed_query(test_query)
        query_emb_array = np.array(query_embedding)
        print(f"✓ Query embedding shape: {query_emb_array.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(query_emb_array):.4f}")
    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return
    
    # 4. Find most similar cases
    print("\n4. Finding most similar cases...")
    similarities = {}
    
    for case_id, case_emb in case_embeddings.items():
        case_emb_array = np.array(case_emb)
        sim = cosine_similarity(query_emb_array, case_emb_array)
        similarities[case_id] = sim
    
    # Sort by similarity
    ranked_cases = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 5. Display top 10 results
    print("\n" + "="*80)
    print(f"TOP 10 MOST SIMILAR CASES FOR: '{test_query}'")
    print("="*80)
    print(f"{'Rank':<6} {'Case ID':<30} {'Similarity':<15}")
    print("-"*80)
    
    for i, (case_id, sim) in enumerate(ranked_cases[:10], 1):
        print(f"{i:<6} {str(case_id):<30} {sim:<15.6f}")
    
    print("="*80)
    
    # 6. Statistics
    print("\n" + "="*80)
    print("SIMILARITY STATISTICS")
    print("="*80)
    all_sims = list(similarities.values())
    print(f"Mean similarity:   {np.mean(all_sims):.6f}")
    print(f"Max similarity:    {np.max(all_sims):.6f}")
    print(f"Min similarity:    {np.min(all_sims):.6f}")
    print(f"Std deviation:     {np.std(all_sims):.6f}")
    print("="*80)
    
    # 7. Test multiple queries
    print("\n" + "="*80)
    print("TESTING MULTIPLE QUERIES")
    print("="*80)
    
    test_queries = [
        "contract breach damages",
        "criminal law murder intent",
        "property rights ownership",
        "constitutional rights amendment"
    ]
    
    for query in test_queries:
        query_emb = np.array(jina_model.embed_query(query))
        
        # Find top result
        top_sim = -1
        top_case = None
        for case_id, case_emb in case_embeddings.items():
            sim = cosine_similarity(query_emb, np.array(case_emb))
            if sim > top_sim:
                top_sim = sim
                top_case = case_id
        
        print(f"\nQuery: '{query}'")
        print(f"  → Top result: {top_case} (similarity: {top_sim:.6f})")
    
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
