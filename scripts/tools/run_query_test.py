"""
Test Hyperbolic Search with Real Query

Uses Jina embeddings for query encoding and hyperbolic search.
"""
import sys
import numpy as np
import pickle
from jina_embeddings_simple import JinaEmbeddingsSimple
from hyperbolic_search import HyperbolicSearchEngine

def main():
    print("="*80)
    print("HYPERBOLIC SEARCH TEST")
    print("="*80)

    # 1. Initialize Jina Embeddings Model (for encoding queries)
    print("\n1. Initializing Jina Embeddings Model...")
    try:
        jina_model = JinaEmbeddingsSimple(model_path="jinaai/jina-embeddings-v3")
    except Exception as e:
        print(f"❌ Failed to load Jina model: {e}")
        print("\nTrying with local path...")
        try:
            jina_model = JinaEmbeddingsSimple(model_path="models/jina-embeddings-v3")
        except Exception as e2:
            print(f"❌ Also failed with local path: {e2}")
            return

    # 2. Load Jina embeddings cache (to verify we have embeddings for cases)
    print("\n2. Loading Jina embeddings cache...")
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            jina_cache = pickle.load(f)
        print(f"   ✓ Loaded {len(jina_cache)} Jina case embeddings")
    except Exception as e:
        print(f"⚠️  Could not load Jina cache: {e}")
        jina_cache = {}

    # 3. Initialize Hyperbolic Search Engine
    print("\n3. Initializing Hyperbolic Search Engine...")
    engine = HyperbolicSearchEngine(embeddings_path='models/hgcn_embeddings.pkl')
    
    if not engine.embeddings:
        print("❌ No hyperbolic embeddings found. Cannot search.")
        return

    # 4. Define Query
    query = "negligence duty of care breach damages"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    print(f"\n4. Processing Query: '{query}'")

    # 5. Embed Query
    print("   Encoding query with Jina...")
    try:
        query_emb = np.array(jina_model.embed_query(query))
        print(f"   ✓ Query embedding shape: {query_emb.shape}")
    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return
    
    # 6. Search
    print("\n5. Performing hyperbolic search...")
    candidate_ids = list(engine.embeddings.keys())
    print(f"   Searching through {len(candidate_ids)} candidates...")
    
    results = engine.search(query_emb, candidate_ids, top_k=10)
    
    # 7. Display Results
    print(f"\n{'='*80}")
    print(f"TOP 10 RESULTS FOR: '{query}'")
    print('='*80)
    print(f"{'Rank':<6} {'Case ID':<20} {'Distance':<12} {'Similarity':<12} {'Radius':<10} {'Court Level'}")
    print('-'*80)
    
    for i, (case_id, dist) in enumerate(results, 1):
        hierarchy = engine.get_hierarchy_info(case_id)
        similarity = engine.distance_to_similarity(dist)
        radius = hierarchy.get('radius', 0)
        court_level = hierarchy['court_level'].replace(' (inferred)', '')
        print(f"{i:<6} {str(case_id):<20} {dist:<12.4f} {similarity:<12.4f} {radius:<10.4f} {court_level}")
        
    print('='*80)
    
    # 8. Show hierarchy statistics
    print(f"\n{'='*80}")
    print("HIERARCHY ANALYSIS OF RESULTS")
    print('='*80)
    radii = [engine.get_hierarchy_info(case_id)['radius'] for case_id, _ in results]
    valid_radii = [r for r in radii if r is not None]
    
    if valid_radii:
        print(f"Average radius:    {np.mean(valid_radii):.4f}")
        print(f"Min radius:        {np.min(valid_radii):.4f} (highest in hierarchy)")
        print(f"Max radius:        {np.max(valid_radii):.4f} (lowest in hierarchy)")
        print(f"Std deviation:     {np.std(valid_radii):.4f}")
        
        # Count by court level
        court_counts = {}
        for case_id, _ in results:
            court = engine.get_hierarchy_info(case_id)['court_level']
            court_counts[court] = court_counts.get(court, 0) + 1
        
        print(f"\nCourt Level Distribution:")
        for court, count in sorted(court_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {court}: {count}")
        
        print(f"\n💡 Interpretation: Lower radius values indicate higher position in legal hierarchy")
    print('='*80)

if __name__ == "__main__":
    main()
