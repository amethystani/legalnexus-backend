"""
Test HGCN Hyperbolic Embeddings on Query

Uses the trained hyperbolic embeddings (models/hgcn_embeddings.pkl) to find similar cases.
"""
import pickle
import numpy as np

def poincare_distance(x, y, c=1.0):
    """
    Calculate Poincaré distance between two points in hyperbolic space.
    
    Args:
        x, y: Points in Poincaré ball (numpy arrays)
        c: Curvature parameter (default: 1.0)
    
    Returns:
        Hyperbolic distance
    """
    sqrt_c = np.sqrt(c)
    x = np.array(x)
    y = np.array(y)
    
    diff_norm_sq = np.sum((x - y) ** 2)
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    
    numerator = 2 * diff_norm_sq
    denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    
    # Avoid numerical issues
    if denominator <= 0:
        return float('inf')
    
    return (1.0 / sqrt_c) * np.arccosh(1 + c * numerator / denominator)

def euclidean_distance(x, y):
    """Simple Euclidean distance."""
    return np.linalg.norm(np.array(x) - np.array(y))

def cosine_similarity(x, y):
    """Cosine similarity between two vectors."""
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def main():
    print("="*80)
    print("HYPERBOLIC GNN MODEL TEST (HGCN Embeddings)")
    print("="*80)
    
    # 1. Load hyperbolic embeddings
    print("\n1. Loading hyperbolic embeddings (models/hgcn_embeddings.pkl)...")
    try:
        with open('models/hgcn_embeddings.pkl', 'rb') as f:
            hgcn_embeddings = pickle.load(f)
        
        print(f"✓ Loaded {len(hgcn_embeddings)} hyperbolic embeddings")
        
        # Show structure
        sample_keys = list(hgcn_embeddings.keys())[:5]
        print(f"\nSample keys: {sample_keys}")
        
        # Get embedding dimension
        first_emb = list(hgcn_embeddings.values())[0]
        print(f"Embedding dimension: {len(first_emb)}")
        print(f"Embedding type: {type(first_emb)}")
        
    except Exception as e:
        print(f"❌ Failed to load hyperbolic embeddings: {e}")
        return
    
    # 2. Get list of case IDs (exclude 'filename' if it exists)
    case_ids = [k for k in hgcn_embeddings.keys() if k != 'filename']
    print(f"\n2. Found {len(case_ids)} cases with hyperbolic embeddings")
    
    # 3. Pick a test case to use as query
    # Select a case as our "query" (avoiding 'filename')
    query_case_id = case_ids[42] if len(case_ids) > 42 else case_ids[0]
    print(f"\n3. Using case '{query_case_id}' as query")
    
    query_hgcn_emb = np.array(hgcn_embeddings[query_case_id])
    print(f"   Query HGCN embedding shape: {query_hgcn_emb.shape}")
    print(f"   Query HGCN norm (radius): {np.linalg.norm(query_hgcn_emb):.4f}")
    
    # 4. Find similar cases using Poincaré distance
    print("\n5. Finding similar cases using Poincaré distance...")
    
    poincare_distances = {}
    euclidean_distances = {}
    
    for case_id, case_emb in hgcn_embeddings.items():
        if case_id == query_case_id:
            continue  # Skip the query itself
        
        case_emb_array = np.array(case_emb)
        
        # Calculate both distances
        p_dist = poincare_distance(query_hgcn_emb, case_emb_array)
        e_dist = euclidean_distance(query_hgcn_emb, case_emb_array)
        
        poincare_distances[case_id] = p_dist
        euclidean_distances[case_id] = e_dist
    
    # Rank by Poincaré distance (lower is more similar)
    ranked_poincare = sorted(poincare_distances.items(), key=lambda x: x[1])
    ranked_euclidean = sorted(euclidean_distances.items(), key=lambda x: x[1])
    
    # 5. Display results
    print("\n" + "="*80)
    print(f"TOP 10 SIMILAR CASES (Poincaré Distance)")
    print(f"Query Case: {query_case_id}")
    print("="*80)
    print(f"{'Rank':<6} {'Case ID':<30} {'Poincaré Dist':<15} {'Radius':<12}")
    print("-"*80)
    
    for i, (case_id, dist) in enumerate(ranked_poincare[:10], 1):
        radius = np.linalg.norm(hgcn_embeddings[case_id])
        print(f"{i:<6} {str(case_id):<30} {dist:<15.6f} {radius:<12.4f}")
    
    print("="*80)
    
    # 6. Compare with Euclidean distance
    print("\n" + "="*80)
    print(f"TOP 10 SIMILAR CASES (Euclidean Distance)")
    print(f"Query Case: {query_case_id}")
    print("="*80)
    print(f"{'Rank':<6} {'Case ID':<30} {'Euclidean Dist':<15} {'Radius':<12}")
    print("-"*80)
    
    for i, (case_id, dist) in enumerate(ranked_euclidean[:10], 1):
        radius = np.linalg.norm(hgcn_embeddings[case_id])
        print(f"{i:<6} {str(case_id):<30} {dist:<15.6f} {radius:<12.4f}")
    
    print("="*80)
    
    # 7. Hierarchy analysis
    print("\n" + "="*80)
    print("HIERARCHY ANALYSIS")
    print("="*80)
    
    # Analyze radii distribution
    all_radii = [np.linalg.norm(emb) for emb in hgcn_embeddings.values()]
    
    print(f"Total cases: {len(all_radii)}")
    print(f"Mean radius: {np.mean(all_radii):.4f}")
    print(f"Std radius:  {np.std(all_radii):.4f}")
    print(f"Min radius:  {np.min(all_radii):.4f} (highest hierarchy)")
    print(f"Max radius:  {np.max(all_radii):.4f} (lowest hierarchy)")
    
    # Categorize by radius
    top_10_radii = [np.linalg.norm(hgcn_embeddings[cid]) for cid, _ in ranked_poincare[:10]]
    print(f"\nTop 10 results mean radius: {np.mean(top_10_radii):.4f}")
    print(f"Query radius: {np.linalg.norm(query_hgcn_emb):.4f}")
    
    print("\n💡 Interpretation:")
    print("   - Lower radius = higher in legal hierarchy (e.g., Supreme Court)")
    print("   - Poincaré distance captures hierarchical relationships")
    print("   - Similar radius values suggest similar court levels")
    print("="*80)
    
    # 8. Test with multiple queries
    print("\n" + "="*80)
    print("TESTING WITH MULTIPLE CASES AS QUERIES")
    print("="*80)
    
    test_case_ids = case_ids[:5]
    
    for test_id in test_case_ids:
        test_emb = np.array(hgcn_embeddings[test_id])
        test_radius = np.linalg.norm(test_emb)
        
        # Find top match
        min_dist = float('inf')
        top_match = None
        
        for case_id, case_emb in hgcn_embeddings.items():
            if case_id == test_id:
                continue
            dist = poincare_distance(test_emb, np.array(case_emb))
            if dist < min_dist:
                min_dist = dist
                top_match = case_id
        
        top_radius = np.linalg.norm(hgcn_embeddings[top_match])
        
        print(f"\nQuery: {test_id} (radius: {test_radius:.4f})")
        print(f"  → Top match: {top_match} (dist: {min_dist:.4f}, radius: {top_radius:.4f})")
    
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
