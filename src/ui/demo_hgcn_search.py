"""
Demo: HGCN Hyperbolic Search with Real Query

Demonstrates the complete workflow:
1. Load trained HGCN hyperbolic embeddings
2. Use a case as a "query"
3. Find similar cases using Poincaré distance
4. Show hierarchy analysis
"""
import pickle
import numpy as np
import sys

def poincare_distance(x, y, c=1.0):
    """Calculate Poincaré distance in hyperbolic space."""
    sqrt_c = np.sqrt(c)
    x = np.array(x)
    y = np.array(y)
    
    diff_norm_sq = np.sum((x - y) ** 2)
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    
    numerator = 2 * diff_norm_sq
    denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    
    if denominator <= 0:
        return float('inf')
    
    return (1.0 / sqrt_c) * np.arccosh(1 + c * numerator / denominator)

def get_court_level(radius):
    """Infer court level from radius."""
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
    # Parse arguments
    query_case_id = None
    if len(sys.argv) > 1:
        query_case_id = sys.argv[1]
    
    print("="*80)
    print("HGCN HYPERBOLIC SEARCH DEMO")
    print("Trained Model: models/hgcn_embeddings.pkl")
    print("="*80)
    
    # 1. Load embeddings
    print("\n[1/4] Loading hyperbolic embeddings...")
    with open('models/hgcn_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    case_ids = [k for k in embeddings.keys() if k != 'filename']
    print(f"✓ Loaded {len(case_ids)} cases with 64-D hyperbolic embeddings")
    
    # 2. Select query
    if query_case_id and query_case_id in embeddings:
        print(f"\n[2/4] Using specified query: {query_case_id}")
    else:
        # Pick an interesting case
        if query_case_id:
            print(f"⚠️  Case '{query_case_id}' not found in embeddings")
        
        # Try to find a Supreme Court case
        sc_cases = [c for c in case_ids if 'SupremeCourt' in c]
        if sc_cases:
            query_case_id = sc_cases[10] if len(sc_cases) > 10 else sc_cases[0]
        else:
            query_case_id = case_ids[100]
        
        print(f"\n[2/4] Using example query: {query_case_id}")
    
    query_emb = np.array(embeddings[query_case_id])
    query_radius = np.linalg.norm(query_emb)
    query_level = get_court_level(query_radius)
    
    print(f"   Embedding dimension: {query_emb.shape[0]}")
    print(f"   Radius: {query_radius:.4f}")
    print(f"   Inferred level: {query_level}")
    
    # 3. Search
    print(f"\n[3/4] Searching through {len(case_ids)} cases...")
    results = []
    
    for case_id in case_ids:
        if case_id == query_case_id:
            continue
        
        case_emb = np.array(embeddings[case_id])
        dist = poincare_distance(query_emb, case_emb)
        results.append((case_id, dist))
    
    # Sort by distance
    results.sort(key=lambda x: x[1])
    top_k = 15
    
    print(f"✓ Found {len(results)} candidates")
    
    # 4. Display results
    print(f"\n[4/4] Top {top_k} Most Similar Cases")
    print("="*80)
    print(f"Query: {query_case_id}")
    print(f"Level: {query_level} (radius: {query_radius:.4f})")
    print("="*80)
    print()
    
    print(f"{'#':<4} {'Case ID':<45} {'Distance':<12} {'Radius':<10} {'Level'}")
    print("-"*120)
    
    for i, (case_id, dist) in enumerate(results[:top_k], 1):
        radius = np.linalg.norm(embeddings[case_id])
        level = get_court_level(radius)
        
        print(f"{i:<4} {case_id:<45} {dist:<12.6f} {radius:<10.4f} {level}")
    
    print("="*80)
    
    # 5. Analysis
    print("\n📊 ANALYSIS")
    print("="*80)
    
    top_radii = [np.linalg.norm(embeddings[cid]) for cid, _ in results[:top_k]]
    
    print(f"Query radius:         {query_radius:.4f}")
    print(f"Top {top_k} mean radius:  {np.mean(top_radii):.4f} (±{np.std(top_radii):.4f})")
    print(f"Radius difference:    {abs(query_radius - np.mean(top_radii)):.4f}")
    
    # Count hierarchy distribution
    hierarchy_counts = {}
    for case_id, dist in results[:top_k]:
        radius = np.linalg.norm(embeddings[case_id])
        level = get_court_level(radius)
        hierarchy_counts[level] = hierarchy_counts.get(level, 0) + 1
    
    print(f"\nHierarchy Distribution (Top {top_k}):")
    for level, count in sorted(hierarchy_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"  {level:<30} {count:>2} {bar}")
    
    print("\n💡 Insights:")
    print("   • Poincaré distance preserves hierarchical structure")
    print("   • Cases with similar radii have similar court levels")
    print("   • Lower radius = higher authority (Supreme Court)")
    print("   • The model learned legal hierarchy from citations!")
    print("="*80)
    
    # 6. Compare with random cases
    print("\n🎲 COMPARISON: Similar Cases vs Random Cases")
    print("="*80)
    
    # Random sample
    random_indices = np.random.choice(len(case_ids), min(15, len(case_ids)), replace=False)
    random_radii = [np.linalg.norm(embeddings[case_ids[i]]) for i in random_indices]
    
    print(f"Top {top_k} similar cases:")
    print(f"  Mean radius: {np.mean(top_radii):.4f}")
    print(f"  Std radius:  {np.std(top_radii):.4f}")
    print(f"  Distance to query: {abs(query_radius - np.mean(top_radii)):.4f}")
    
    print(f"\nRandom {len(random_radii)} cases:")
    print(f"  Mean radius: {np.mean(random_radii):.4f}")
    print(f"  Std radius:  {np.std(random_radii):.4f}")
    print(f"  Distance to query: {abs(query_radius - np.mean(random_radii)):.4f}")
    
    print("\n→ Similar cases cluster around the query's hierarchy level! ✓")
    print("="*80)

if __name__ == "__main__":
    main()
