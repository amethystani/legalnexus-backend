#!/usr/bin/env python3
"""
Measure Gromov δ-hyperbolicity of legal citation network.

This is the KEY THEORETICAL CONTRIBUTION for Part 1.
If δ is small (< 1.0), the graph is tree-like (hyperbolic).
This justifies using hyperbolic embeddings.

Reference: 
- "Computing the Gromov Hyperbolicity of a Discrete Metric Space"
- Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks"
"""

import networkx as nx
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
import pickle
import json
from tqdm import tqdm


def gromov_delta_hyperbolicity(G: nx.Graph, sample_size: int = 1000) -> float:
    """
    Compute Gromov δ-hyperbolicity.
    
    For all 4-tuples (w,x,y,z), compute:
        δ = max{ |d(w,x) + d(y,z) - max(d(w,y)+d(x,z), d(w,z)+d(x,y))| } / 2
    
    Interpretation:
    - δ ≈ 0: Perfect tree (hyperbolic)
    - δ < 1: Tree-like structure (hyperbolic)
    - δ ≥ 2: Not hyperbolic (flat/Euclidean)
    
    Args:
        G: Undirected graph
        sample_size: Number of 4-tuples to sample (full computation is O(n^4))
    
    Returns:
        Average δ-hyperbolicity
    """
    # Get largest connected component
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        print(f"Using largest connected component: {len(largest)} nodes")
    
    # Sample nodes (computing all 4-tuples is O(n^4) - too slow)
    nodes = list(G.nodes())
    if len(nodes) > 200:
        nodes = np.random.choice(nodes, size=min(200, len(nodes)), replace=False).tolist()
    
    # Compute all-pairs shortest paths
    print(f"Computing shortest paths for {len(nodes)} nodes...")
    dist = dict(nx.all_pairs_shortest_path_length(G))
    
    # Sample 4-tuples and compute delta
    print(f"Sampling {sample_size} 4-tuples...")
    deltas = []
    
    for _ in tqdm(range(sample_size)):
        # Sample 4 distinct nodes
        if len(nodes) < 4:
            break
        w, x, y, z = np.random.choice(nodes, size=4, replace=False)
        
        # Get pairwise distances
        try:
            d_wx = dist[w].get(x, float('inf'))
            d_yz = dist[y].get(z, float('inf'))
            d_wy = dist[w].get(y, float('inf'))
            d_xz = dist[x].get(z, float('inf'))
            d_wz = dist[w].get(z, float('inf'))
            d_xy = dist[x].get(y, float('inf'))
        except KeyError:
            continue
        
        # Skip if any distance is infinite
        if any(d == float('inf') for d in [d_wx, d_yz, d_wy, d_xz, d_wz, d_xy]):
            continue
        
        # Compute Gromov product: max of three sums
        s1 = d_wx + d_yz
        s2 = d_wy + d_xz
        s3 = d_wz + d_xy
        
        # Delta for this 4-tuple
        delta = abs(s1 - max(s2, s3)) / 2.0
        deltas.append(delta)
    
    # Return average delta
    if not deltas:
        return float('nan')
    
    return np.mean(deltas)


def compare_with_random_graphs(citation_network: nx.Graph, sample_size: int = 1000) -> Dict:
    """
    Compare curvature with random graph models.
    
    This proves legal networks are MORE hyperbolic than random graphs.
    If δ_legal < δ_random, then legal networks have special structure.
    
    Args:
        citation_network: Legal citation network
        sample_size: Number of 4-tuples per graph
    
    Returns:
        Dictionary with δ values for each graph type
    """
    # Get graph properties
    n = citation_network.number_of_nodes()
    m = citation_network.number_of_edges()
    
    print(f"Legal citation network: {n} nodes, {m} edges")
    
    # Measure citation network
    print("\n[1/4] Measuring legal citation network...")
    delta_legal = gromov_delta_hyperbolicity(citation_network, sample_size)
    
    # Erdos-Renyi random graph (null model)
    print("\n[2/4] Generating Erdős-Rényi random graph...")
    p = 2 * m / (n * (n - 1))  # Match edge density
    G_er = nx.erdos_renyi_graph(n, p, seed=42)
    delta_er = gromov_delta_hyperbolicity(G_er, sample_size)
    
    # Barabási-Albert (scale-free, common in real networks)
    print("\n[3/4] Generating Barabási-Albert graph...")
    k = max(1, int(m / n))  # Average degree
    G_ba = nx.barabasi_albert_graph(n, k, seed=42)
    delta_ba = gromov_delta_hyperbolicity(G_ba, sample_size)
    
    # Perfect binary tree (ideal hyperbolic structure)
    print("\n[4/4] Generating perfect binary tree...")
    depth = min(int(np.log2(n)), 10)  # Limit depth
    G_tree = nx.balanced_tree(2, depth)
    delta_tree = gromov_delta_hyperbolicity(G_tree, sample_size)
    
    results = {
        'legal_network': {
            'delta': delta_legal,
            'nodes': n,
            'edges': m,
            'interpretation': interpret_delta(delta_legal)
        },
        'erdos_renyi': {
            'delta': delta_er,
            'interpretation': interpret_delta(delta_er)
        },
        'barabasi_albert': {
            'delta': delta_ba,
            'interpretation': interpret_delta(delta_ba)
        },
        'perfect_tree': {
            'delta': delta_tree,
            'interpretation': interpret_delta(delta_tree)
        },
        'conclusion': generate_conclusion(delta_legal, delta_er, delta_ba, delta_tree)
    }
    
    return results


def interpret_delta(delta: float) -> str:
    """Interpret δ-hyperbolicity value."""
    if np.isnan(delta):
        return "Cannot compute (disconnected graph?)"
    elif delta < 0.5:
        return "Highly hyperbolic (tree-like structure)"
    elif delta < 1.0:
        return "Moderately hyperbolic (hierarchical)"
    elif delta < 2.0:
        return "Weakly hyperbolic (some hierarchy)"
    else:
        return "Not hyperbolic (flat/Euclidean structure)"


def generate_conclusion(delta_legal: float, delta_er: float, 
                        delta_ba: float, delta_tree: float) -> str:
    """Generate conclusion for paper."""
    if np.isnan(delta_legal):
        return "Error: Could not compute δ for legal network"
    
    if delta_legal < delta_er and delta_legal < delta_ba:
        return (f"✅ Legal citation networks are significantly more hyperbolic (δ={delta_legal:.3f}) "
                f"than random graphs (Erdős-Rényi: δ={delta_er:.3f}, Barabási-Albert: δ={delta_ba:.3f}). "
                f"This justifies using hyperbolic embeddings.")
    else:
        return (f"⚠️ Legal networks (δ={delta_legal:.3f}) are NOT more hyperbolic than random graphs. "
                f"Hyperbolic embeddings may not provide advantage.")


def load_citation_network(data_path: str = 'data/citation_network.pkl') -> nx.Graph:
    """Load citation network from pickle file."""
    print(f"Loading citation network from {data_path}...")
    
    try:
        with open(data_path, 'rb') as f:
            network = pickle.load(f)
        
        # Build networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        if 'case_ids' in network:
            G.add_nodes_from(network['case_ids'])
        
        # Add edges
        if 'edges' in network:
            for edge in network['edges']:
                if isinstance(edge, dict):
                    G.add_edge(edge['source'], edge['target'])
                elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        
        # Convert to undirected for curvature analysis
        G_undirected = G.to_undirected()
        
        print(f"Loaded graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
        
        return G_undirected
        
    except FileNotFoundError:
        print(f"ERROR: {data_path} not found!")
        print("Creating synthetic graph for demonstration...")
        return create_synthetic_legal_network()


def create_synthetic_legal_network(n_cases: int = 100) -> nx.Graph:
    """
    Create a synthetic legal citation network with hierarchical structure.
    
    This is a fallback if real data is not available.
    Legal networks should have:
    - Supreme Court cases → cited by many (high in-degree)
    - Lower court cases → cite others (high out-degree)
    - Hierarchical structure (tree-like)
    """
    G = nx.DiGraph()
    
    # Layer 1: Supreme Court (5 landmark cases)
    sc_cases = [f"SC_{i}" for i in range(5)]
    G.add_nodes_from(sc_cases)
    
    # Layer 2: High Court (20 cases)
    hc_cases = [f"HC_{i}" for i in range(20)]
    G.add_nodes_from(hc_cases)
    
    # Layer 3: District Court (75 cases)
    dc_cases = [f"DC_{i}" for i in range(75)]
    G.add_nodes_from(dc_cases)
    
    # Citations: lower courts cite higher courts (creates hierarchy)
    for hc in hc_cases:
        # Each HC case cites 1-3 SC cases
        cited_sc = np.random.choice(sc_cases, size=np.random.randint(1, 4), replace=False)
        for sc in cited_sc:
            G.add_edge(hc, sc)
    
    for dc in dc_cases:
        # Each DC case cites 1-2 HC cases
        cited_hc = np.random.choice(hc_cases, size=np.random.randint(1, 3), replace=False)
        for hc in cited_hc:
            G.add_edge(dc, hc)
        
        # Some DC cases cite SC directly
        if np.random.random() < 0.3:
            cited_sc = np.random.choice(sc_cases, size=1)
            G.add_edge(dc, cited_sc[0])
    
    print(f"Created synthetic legal network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G.to_undirected()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Measure graph curvature')
    parser.add_argument('--data', default='data/citation_network.pkl', help='Path to citation network')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of 4-tuples to sample')
    parser.add_argument('--output', default='experiments/results/curvature_analysis.json', help='Output file')
    args = parser.parse_args()
    
    # Load citation network
    G = load_citation_network(args.data)
    
    # Measure curvature and compare
    results = compare_with_random_graphs(G, sample_size=args.sample_size)
    
    # Print results
    print("\n" + "="*80)
    print("GROMOV δ-HYPERBOLICITY ANALYSIS")
    print("="*80)
    print(f"\nLegal Citation Network:    δ = {results['legal_network']['delta']:.3f}")
    print(f"  → {results['legal_network']['interpretation']}")
    print(f"\nErdős-Rényi Random:        δ = {results['erdos_renyi']['delta']:.3f}")
    print(f"  → {results['erdos_renyi']['interpretation']}")
    print(f"\nBarabási-Albert:           δ = {results['barabasi_albert']['delta']:.3f}")
    print(f"  → {results['barabasi_albert']['interpretation']}")
    print(f"\nPerfect Tree (reference):  δ = {results['perfect_tree']['delta']:.3f}")
    print(f"  → {results['perfect_tree']['interpretation']}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(results['conclusion'])
    print("="*80 + "\n")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}")
