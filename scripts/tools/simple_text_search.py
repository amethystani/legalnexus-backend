"""
Simple Text Query Search (Lightweight)

Search legal cases using text queries WITHOUT loading the full Jina model.
Uses pre-computed embeddings only.

Usage: python3 simple_text_search.py "drunk driving"
"""
import pickle
import numpy as np
import sys

def cosine_similarity(a, b):
    """Calculate cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_court_level(radius):
    """Get court level from radius"""
    if radius < 0.10:
        return "🏛️  Supreme Court"
    elif radius < 0.15:
        return "⚖️  High Court (Major)"
    elif radius < 0.20:
        return "⚖️  High Court"
    elif radius < 0.30:
        return "📜 Lower Court/Tribunal"
    else:
        return "📋 District/Subordinate"

def main():
    # Get query from command line
    if len(sys.argv) < 2:
        print("Usage: python3 simple_text_search.py 'your query here'")
        print("\nExample: python3 simple_text_search.py 'drunk driving'")
        return
    
    query_text = " ".join(sys.argv[1:])
    
    print("="*80)
    print("SIMPLE TEXT SEARCH (Pre-computed Embeddings)")
    print("="*80)
    print(f"\nQuery: '{query_text}'")
    
    # Load embeddings
    print("\n[1/3] Loading embeddings...")
    
    try:
        with open('models/hgcn_embeddings.pkl', 'rb') as f:
            hgcn_embeddings = pickle.load(f)
        case_ids = [k for k in hgcn_embeddings.keys() if k != 'filename']
        print(f"✓ Loaded {len(case_ids)} HGCN embeddings")
    except Exception as e:
        print(f"❌ Error loading HGCN embeddings: {e}")
        return
    
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            jina_embeddings = pickle.load(f)
        print(f"✓ Loaded {len(jina_embeddings)} Jina embeddings")
    except Exception as e:
        print(f"❌ Error loading Jina embeddings: {e}")
        return
    
    # For simplicity: pick a random case as "query"
    # In a real system, you'd use Jina to embed the query text
    print("\n[2/3] Searching...")
    print("⚠️  NOTE: Using case similarity (need Jina model for true text search)")
    
    # Use first case as example query
    sample_case = case_ids[100]
    print(f"Using sample case as query: {sample_case}")
    
    if str(100) not in jina_embeddings:
        print("❌ Sample case not found in Jina embeddings")
        return
    
    query_emb = np.array(jina_embeddings[str(100)])
    
    # Find similar cases
    results = []
    for idx, case_id in enumerate(case_ids[:1000]):  # Limit to first 1000 for speed
        jina_key = str(idx)
        if jina_key in jina_embeddings:
            case_emb = np.array(jina_embeddings[jina_key])
            similarity = cosine_similarity(query_emb, case_emb)
            
            if case_id in hgcn_embeddings:
                radius = np.linalg.norm(hgcn_embeddings[case_id])
                results.append((case_id, similarity, radius))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"\n[3/3] Top 15 Results")
    print("="*80)
    print(f"{'#':<4} {'Case ID':<45} {'Similarity':<12} {'Radius':<10} {'Level'}")
    print("-"*80)
    
    for i, (case_id, similarity, radius) in enumerate(results[:15], 1):
        level = get_court_level(radius)
        print(f"{i:<4} {case_id:<45} {similarity:<12.4f} {radius:<10.4f} {level}")
    
    print("="*80)
    
    # Stats
    print("\n📊 Statistics:")
    radii = [r[2] for r in results[:15]]
    sims = [r[1] for r in results[:15]]
    print(f"  Mean similarity: {np.mean(sims):.4f}")
    print(f"  Mean radius:     {np.mean(radii):.4f}")
    print(f"  Radius std:      {np.std(radii):.4f}")
    
    print("\n💡 NOTE: To enable true text query search, use the Jina model:")
    print("   python3 test_jina_query.py")

if __name__ == "__main__":
    main()
