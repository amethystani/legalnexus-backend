"""
Comprehensive Evaluation of HGCN Hyperbolic Embeddings

Tests the actual model performance and generates real metrics for the research paper.
"""
import pickle
import numpy as np
from collections import defaultdict
import time
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys

def print_progress(current, total, prefix="Progress", suffix="", bar_length=50):
    """Print a progress bar."""
    percent = float(current) / total
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write(f'\r{prefix}: [{hashes}{spaces}] {int(round(percent * 100))}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()  # New line when complete

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

def euclidean_distance(x, y):
    """Euclidean distance."""
    return np.linalg.norm(np.array(x) - np.array(y))

def get_court_level(case_id):
    """Extract court level from case ID."""
    case_id_lower = case_id.lower()
    if 'supremecourt' in case_id_lower:
        return 'Supreme Court'
    elif 'hc' in case_id_lower or 'high' in case_id_lower:
        return 'High Court'
    elif 'district' in case_id_lower or 'subordinate' in case_id_lower:
        return 'Lower Court'
    else:
        return 'Other'

def get_court_level_by_radius(radius):
    """Infer court level from radius."""
    if radius < 0.10:
        return 'Supreme Court'
    elif radius < 0.15:
        return 'High Court (Major)'
    elif radius < 0.20:
        return 'High Court'
    elif radius < 0.30:
        return 'Lower Court/Tribunal'
    else:
        return 'District/Subordinate'

def precision_at_k(retrieved, relevant, k):
    """Compute Precision@k."""
    if len(retrieved) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([r for r in retrieved_k if r in relevant_set]) / min(k, len(retrieved))

def recall_at_k(retrieved, relevant, k):
    """Compute Recall@k."""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([r for r in retrieved_k if r in relevant_set]) / len(relevant)

def mean_average_precision(retrieved, relevant):
    """Compute Mean Average Precision (MAP)."""
    if len(relevant) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    precisions = []
    num_relevant = 0
    
    for i, item in enumerate(retrieved, 1):
        if item in relevant_set:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant)

def ndcg_at_k(retrieved, relevant, k):
    """Compute Normalized Discounted Cumulative Gain@k."""
    if len(relevant) == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    # DCG
    dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(retrieved_k) if item in relevant_set])
    
    # IDCG (ideal DCG)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(relevant)))])
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_retrieval(embeddings, query_ids, method='hyperbolic', k=10):
    """Evaluate retrieval performance."""
    case_ids = [k for k in embeddings.keys() if k != 'filename']
    
    results = {
        'precision@5': [],
        'precision@10': [],
        'recall@5': [],
        'recall@10': [],
        'map': [],
        'ndcg@5': [],
        'ndcg@10': [],
        'query_times': [],
        'relevance_set_sizes': []  # Track relevance set sizes
    }
    
    total_queries = len(query_ids)
    print(f"    Processing {total_queries} queries...")
    
    for idx, query_id in enumerate(query_ids):
        print_progress(idx + 1, total_queries, 
                      prefix=f"    {method.capitalize()} evaluation", 
                      suffix=f"Query {idx+1}/{total_queries}")
        if query_id not in embeddings:
            continue
        
        start_time = time.time()
        
        query_emb = np.array(embeddings[query_id])
        query_radius = np.linalg.norm(query_emb)
        query_court = get_court_level(query_id)
        
        # Define "relevant" more strictly:
        # 1. Same court level AND similar radius (tighter threshold)
        # 2. This simulates finding similar cases at the same hierarchy level
        relevant = []
        for case_id in case_ids:
            if case_id == query_id:
                continue
            case_court = get_court_level(case_id)
            case_radius = np.linalg.norm(embeddings[case_id])
            
            # Stricter: same court level AND very similar radius (within 0.02)
            # This creates a more realistic relevance set
            if case_court == query_court and abs(case_radius - query_radius) < 0.02:
                relevant.append(case_id)
        
        # If too few relevant cases, relax slightly but still require same court
        if len(relevant) < 10:
            relevant = []
            for case_id in case_ids:
                if case_id == query_id:
                    continue
                case_court = get_court_level(case_id)
                case_radius = np.linalg.norm(embeddings[case_id])
                if case_court == query_court and abs(case_radius - query_radius) < 0.03:
                    relevant.append(case_id)
        
        # Retrieve using specified method
        distances = []
        total_cases = len(case_ids) - 1  # Exclude query itself
        computed = 0
        
        for case_id in case_ids:
            if case_id == query_id:
                continue
            
            case_emb = np.array(embeddings[case_id])
            
            if method == 'hyperbolic':
                dist = poincare_distance(query_emb, case_emb)
            elif method == 'euclidean':
                dist = euclidean_distance(query_emb, case_emb)
            elif method == 'cosine':
                dist = 1 - cosine(query_emb, case_emb)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            distances.append((case_id, dist))
            computed += 1
            
            # Update progress every 1000 cases or at the end
            if computed % 1000 == 0 or computed == total_cases:
                print_progress(computed, total_cases, 
                              prefix=f"      Computing distances", 
                              suffix=f"{computed}/{total_cases} cases")
        
        # Sort by distance (lower is better for hyperbolic/euclidean, higher for cosine)
        if method == 'cosine':
            distances.sort(key=lambda x: x[1], reverse=True)
        else:
            distances.sort(key=lambda x: x[1])
        
        retrieved = [case_id for case_id, _ in distances]
        
        # Track relevance set size
        results['relevance_set_sizes'].append(len(relevant))
        
        # Compute metrics
        results['precision@5'].append(precision_at_k(retrieved, relevant, 5))
        results['precision@10'].append(precision_at_k(retrieved, relevant, 10))
        results['recall@5'].append(recall_at_k(retrieved, relevant, 5))
        results['recall@10'].append(recall_at_k(retrieved, relevant, 10))
        results['map'].append(mean_average_precision(retrieved, relevant))
        results['ndcg@5'].append(ndcg_at_k(retrieved, relevant, 5))
        results['ndcg@10'].append(ndcg_at_k(retrieved, relevant, 10))
        results['query_times'].append(time.time() - start_time)
    
    # Compute averages
    avg_results = {}
    for key, values in results.items():
        if values:
            avg_results[key] = np.mean(values)
            avg_results[f'{key}_std'] = np.std(values)
    
    return avg_results, results

def analyze_hierarchy(embeddings):
    """Analyze how well the model preserves hierarchy."""
    case_ids = [k for k in embeddings.keys() if k != 'filename']
    
    # Compute radii
    print("    Computing radii for all cases...")
    radii = {}
    total = len(case_ids)
    for idx, case_id in enumerate(case_ids):
        if idx % 5000 == 0 or idx == total - 1:
            print_progress(idx + 1, total, prefix="    Computing radii", suffix=f"{idx+1}/{total} cases")
        radii[case_id] = np.linalg.norm(embeddings[case_id])
    
    # Group by actual court level
    court_levels = defaultdict(list)
    for case_id in case_ids:
        court_level = get_court_level(case_id)
        court_levels[court_level].append(radii[case_id])
    
    # Statistics by court level
    hierarchy_stats = {}
    for court_level, level_radii in court_levels.items():
        if level_radii:
            hierarchy_stats[court_level] = {
                'count': len(level_radii),
                'mean_radius': np.mean(level_radii),
                'std_radius': np.std(level_radii),
                'min_radius': np.min(level_radii),
                'max_radius': np.max(level_radii)
            }
    
    # Overall statistics
    all_radii = list(radii.values())
    overall_stats = {
        'total_cases': len(case_ids),
        'mean_radius': np.mean(all_radii),
        'std_radius': np.std(all_radii),
        'min_radius': np.min(all_radii),
        'max_radius': np.max(all_radii)
    }
    
    return hierarchy_stats, overall_stats, radii

def main():
    total_start_time = time.time()
    print("="*80)
    print("COMPREHENSIVE HGCN MODEL EVALUATION")
    print("="*80)
    
    # Load embeddings
    print("\n[1/5] Loading HGCN embeddings...")
    print("    Reading model file: models/hgcn_embeddings.pkl")
    try:
        load_start = time.time()
        with open('models/hgcn_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        load_time = time.time() - load_start
        
        case_ids = [k for k in embeddings.keys() if k != 'filename']
        print(f"    ✓ Loaded {len(case_ids)} cases in {load_time:.2f}s")
        
        # Check embedding dimension
        sample_emb = embeddings[case_ids[0]]
        print(f"    ✓ Embedding dimension: {len(sample_emb)}")
        
    except Exception as e:
        print(f"    ❌ Error loading embeddings: {e}")
        return
    
    # Analyze hierarchy
    print("\n[2/5] Analyzing hierarchy preservation...")
    hierarchy_stats, overall_stats, radii = analyze_hierarchy(embeddings)
    
    print("\n📊 Hierarchy Statistics by Court Level:")
    print("-"*80)
    for court_level, stats in sorted(hierarchy_stats.items()):
        print(f"{court_level:30s}: {stats['count']:5d} cases, "
              f"radius: {stats['mean_radius']:.4f} ± {stats['std_radius']:.4f} "
              f"[{stats['min_radius']:.4f}, {stats['max_radius']:.4f}]")
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total cases: {overall_stats['total_cases']}")
    print(f"  Mean radius: {overall_stats['mean_radius']:.4f} ± {overall_stats['std_radius']:.4f}")
    print(f"  Radius range: [{overall_stats['min_radius']:.4f}, {overall_stats['max_radius']:.4f}]")
    
    # Check if hierarchy is actually learned
    sc_radius = hierarchy_stats.get('Supreme Court', {}).get('mean_radius', 0)
    hc_radius = hierarchy_stats.get('High Court', {}).get('mean_radius', 0)
    lc_radius = hierarchy_stats.get('Lower Court', {}).get('mean_radius', 0)
    
    if abs(sc_radius - hc_radius) < 0.01 and abs(hc_radius - lc_radius) < 0.01:
        print("\n⚠️  WARNING: Court levels have similar radii!")
        print("   This suggests the model may not have learned hierarchy properly.")
        print("   Expected: Supreme Court < High Court < Lower Court (by radius)")
        print(f"   Actual: SC={sc_radius:.4f}, HC={hc_radius:.4f}, LC={lc_radius:.4f}")
    
    # Select test queries (diverse sample)
    print("\n[3/5] Selecting test queries...")
    np.random.seed(42)
    test_size = min(100, len(case_ids) // 10)  # 10% sample or max 100
    test_queries = np.random.choice(case_ids, test_size, replace=False).tolist()
    print(f"✓ Selected {len(test_queries)} test queries")
    
    # Check if hyperbolic and euclidean embeddings are too similar
    print("\n[3.5/5] Analyzing embedding similarity...")
    sample_cases = case_ids[:100]  # Sample 100 cases
    hyperbolic_embs = np.array([embeddings[cid] for cid in sample_cases])
    euclidean_norms = np.array([np.linalg.norm(emb) for emb in hyperbolic_embs])
    
    # Compare distance rankings for a sample query
    test_query_id = test_queries[0] if test_queries else case_ids[0]
    query_emb = np.array(embeddings[test_query_id])
    
    print(f"    Testing with query: {test_query_id}")
    print(f"    Query radius: {np.linalg.norm(query_emb):.4f}")
    
    # Get top 20 by both methods
    hgcn_distances = []
    euclidean_distances = []
    
    for case_id in case_ids[:1000]:  # Sample 1000 for speed
        if case_id == test_query_id:
            continue
        case_emb = np.array(embeddings[case_id])
        hgcn_distances.append((case_id, poincare_distance(query_emb, case_emb)))
        euclidean_distances.append((case_id, euclidean_distance(query_emb, case_emb)))
    
    hgcn_distances.sort(key=lambda x: x[1])
    euclidean_distances.sort(key=lambda x: x[1])
    
    hgcn_top20 = [cid for cid, _ in hgcn_distances[:20]]
    euclidean_top20 = [cid for cid, _ in euclidean_distances[:20]]
    
    # Calculate overlap
    overlap = len(set(hgcn_top20) & set(euclidean_top20))
    print(f"    Top-20 overlap: {overlap}/20 ({overlap/20*100:.1f}%)")
    
    if overlap > 18:
        print("    ⚠️  WARNING: Rankings are nearly identical!")
        print("    This suggests embeddings may not preserve hyperbolic structure.")
    
    # Evaluate different methods
    print("\n[4/5] Evaluating retrieval performance...")
    methods = ['hyperbolic', 'euclidean', 'cosine']
    method_results = {}
    
    for method_idx, method in enumerate(methods, 1):
        print(f"\n  [{method_idx}/{len(methods)}] Testing {method} method...")
        start_time = time.time()
        avg_results, detailed_results = evaluate_retrieval(embeddings, test_queries, method=method, k=10)
        elapsed = time.time() - start_time
        method_results[method] = avg_results
        method_results[f'{method}_detailed'] = detailed_results
        
        # Show relevance set size info (only for first method to avoid repetition)
        if method_idx == 1 and 'relevance_set_sizes' in detailed_results:
            avg_rel_size = np.mean(detailed_results['relevance_set_sizes'])
            print(f"    Average relevance set size: {avg_rel_size:.1f} cases")
            if avg_rel_size > 1000:
                print(f"    ⚠️  Relevance sets are very large - may inflate precision")
            elif avg_rel_size < 10:
                print(f"    ⚠️  Relevance sets are very small - may deflate recall")
        
        print(f"  ✓ {method.capitalize()} evaluation completed in {elapsed:.1f}s")
    
    # Display results
    print("\n[5/5] Performance Comparison")
    print("="*80)
    print(f"{'Method':<15} {'P@5':<8} {'P@10':<8} {'R@5':<8} {'R@10':<8} {'MAP':<8} {'NDCG@10':<8} {'Time(s)':<8}")
    print("-"*80)
    
    for method in methods:
        results = method_results[method]
        print(f"{method:<15} "
              f"{results.get('precision@5', 0):.3f}   "
              f"{results.get('precision@10', 0):.3f}   "
              f"{results.get('recall@5', 0):.3f}   "
              f"{results.get('recall@10', 0):.3f}   "
              f"{results.get('map', 0):.3f}   "
              f"{results.get('ndcg@10', 0):.3f}   "
              f"{results.get('query_times', 0):.3f}")
    
    print("="*80)
    
    # Save results to JSON
    print("\n💾 Saving results to JSON...")
    output = {
        'dataset_stats': {
            'total_cases': overall_stats['total_cases'],
            'embedding_dim': len(sample_emb),
            'mean_radius': float(overall_stats['mean_radius']),
            'std_radius': float(overall_stats['std_radius']),
            'min_radius': float(overall_stats['min_radius']),
            'max_radius': float(overall_stats['max_radius'])
        },
        'hierarchy_stats': {k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32, float)) else vv 
                                 for kk, vv in v.items()} 
                            for k, v in hierarchy_stats.items()},
        'retrieval_performance': {method: {k: float(v) if isinstance(v, (np.float64, np.float32, float)) else v 
                                           for k, v in results.items()} 
                                  for method, results in method_results.items() 
                                  if not method.endswith('_detailed')}
    }
    
    with open('hgcn_evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Results saved to hgcn_evaluation_results.json")
    print("="*80)
    
    # Print summary for paper
    print("\n" + "="*80)
    print("SUMMARY FOR RESEARCH PAPER")
    print("="*80)
    hgcn_results = method_results['hyperbolic']
    euclidean_results = method_results['euclidean']
    
    print(f"\nHyperbolic (HGCN) Performance:")
    print(f"  Precision@5:  {hgcn_results.get('precision@5', 0):.3f}")
    print(f"  Recall@10:    {hgcn_results.get('recall@10', 0):.3f}")
    print(f"  MAP:          {hgcn_results.get('map', 0):.3f}")
    print(f"  NDCG@10:      {hgcn_results.get('ndcg@10', 0):.3f}")
    
    print(f"\nEuclidean Baseline Performance:")
    print(f"  Precision@5:  {euclidean_results.get('precision@5', 0):.3f}")
    print(f"  Recall@10:    {euclidean_results.get('recall@10', 0):.3f}")
    print(f"  MAP:          {euclidean_results.get('map', 0):.3f}")
    print(f"  NDCG@10:      {euclidean_results.get('ndcg@10', 0):.3f}")
    
    improvement_p5 = ((hgcn_results.get('precision@5', 0) - euclidean_results.get('precision@5', 0)) / 
                      max(euclidean_results.get('precision@5', 0.001), 0.001)) * 100
    improvement_r10 = ((hgcn_results.get('recall@10', 0) - euclidean_results.get('recall@10', 0)) / 
                       max(euclidean_results.get('recall@10', 0.001), 0.001)) * 100
    
    print(f"\nImprovement over Euclidean:")
    print(f"  Precision@5:  +{improvement_p5:.1f}%")
    print(f"  Recall@10:    +{improvement_r10:.1f}%")
    
    # Diagnostic: Why are results similar?
    if abs(hgcn_results.get('precision@5', 0) - euclidean_results.get('precision@5', 0)) < 0.01:
        print(f"\n⚠️  DIAGNOSTIC: Hyperbolic and Euclidean results are nearly identical.")
        print(f"   Possible reasons:")
        print(f"   1. Embeddings may not preserve hyperbolic structure")
        print(f"   2. Distance metrics produce similar rankings")
        print(f"   3. Relevance definition may be too loose (check relevance set sizes)")
        print(f"   4. Model may need retraining with better hyperparameters")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n⏱️  Total evaluation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("="*80)
    print("\n✅ Evaluation complete! Check hgcn_evaluation_results.json for detailed results.")

if __name__ == "__main__":
    main()

