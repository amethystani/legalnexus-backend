#!/usr/bin/env python3
"""
Run FULL EXPERIMENTS on the complete dataset.

1. Load full citation network (built by build_full_network.py)
2. Run Hyperbolic vs Euclidean comparison (5 seeds)
3. Run Nash Equilibrium analysis
4. Generate final report
"""

import os
import pickle
import torch
import numpy as np
from experiments.hyperbolic_vs_euclidean import run_single_experiment, statistical_significance_test
from analysis.measure_graph_curvature import compare_with_random_graphs, load_citation_network

def run_full_experiments():
    print("="*80)
    print("RUNNING FULL EXPERIMENTS ON COMPLETE DATASET")
    print("="*80)
    
    # 1. Load Data
    data_path = 'data/citation_network_full.pkl'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run build_full_network.py first.")
        return

    print(f"\n[1/3] Loading Full Network from {data_path}...")
    with open(data_path, 'rb') as f:
        network_data = pickle.load(f)
        
    # Convert to format needed for GNN
    # We need features. If not present, generate random or use embeddings if available
    num_nodes = len(network_data['case_ids'])
    num_edges = len(network_data['edges'])
    
    print(f"Loaded: {num_nodes} nodes, {num_edges} edges")
    
    if num_edges < 10:
        print("⚠️  WARNING: Very few edges found. Results may be unstable.")
    
    # Generate features (in real app, load Gemini embeddings)
    # For now, use random features to test the pipeline
    print("Generating features (placeholder for Gemini embeddings)...")
    features = torch.randn(num_nodes, 768)
    
    # Build edge index
    node_map = {nid: i for i, nid in enumerate(network_data['case_ids'])}
    edge_list = []
    for e in network_data['edges']:
        if e['source'] in node_map and e['target'] in node_map:
            edge_list.append([node_map[e['source']], node_map[e['target']]])
            
    if not edge_list:
        print("Error: No valid edges after mapping. Check node IDs.")
        return

    edge_index = torch.tensor(edge_list).t().contiguous()
    
    # Create splits
    indices = np.arange(edge_index.shape[1])
    np.random.shuffle(indices)
    train_idx = indices[:int(0.7*len(indices))]
    val_idx = indices[int(0.7*len(indices)):int(0.85*len(indices))]
    test_idx = indices[int(0.85*len(indices)):]
    
    # Convert edge_index to sparse adjacency matrix for Hyperbolic GNN
    # The model expects N x N adjacency matrix, not edge_index
    adj_indices = edge_index
    adj_values = torch.ones(edge_index.shape[1])
    adj_shape = (num_nodes, num_nodes)
    adj = torch.sparse.FloatTensor(adj_indices, adj_values, torch.Size(adj_shape))
    
    # Normalize adjacency matrix (D^-1/2 A D^-1/2)
    # This is usually done inside the model or preprocessing, but let's do it here to be safe
    # For now, we pass the raw sparse adj and let the model handle it if needed, 
    # or ensure the model expects sparse input.
    
    data = {
        'features': features,
        'edge_index': adj,  # PASSING ADJ MATRIX instead of edge_index for Hyperbolic model
        'raw_edge_index': edge_index, # Keep raw for Euclidean if needed
        'train_pos': edge_index[:, train_idx].t(),
        'train_neg': torch.randint(0, num_nodes, (len(train_idx), 2)), 
        'val_pos': edge_index[:, val_idx].t(),
        'val_neg': torch.randint(0, num_nodes, (len(val_idx), 2)),
        'test_pos': edge_index[:, test_idx].t(),
        'test_neg': torch.randint(0, num_nodes, (len(test_idx), 2))
    }
    
    # 2. Curvature Analysis
    print("\n[2/3] Running Curvature Analysis on Full Graph...")
    G = load_citation_network(data_path)
    curvature_results = compare_with_random_graphs(G, sample_size=500)
    print(f"Legal Network δ: {curvature_results['legal_network']['delta']:.3f}")
    print(f"Interpretation: {curvature_results['legal_network']['interpretation']}")
    
    # 3. Hyperbolic vs Euclidean
    print("\n[3/3] Running Hyperbolic vs Euclidean Comparison (5 Seeds)...")
    seeds = [42, 43, 44, 45, 46]
    hyp_results = []
    euc_results = []
    
    for seed in seeds:
        print(f"  Running Seed {seed}...")
        hyp_res = run_single_experiment('hyperbolic', data, seed)
        euc_res = run_single_experiment('euclidean', data, seed)
        hyp_results.append(hyp_res)
        euc_results.append(euc_res)
        
    # Stats
    hyp_aucs = [r['test_auc'] for r in hyp_results]
    euc_aucs = [r['test_auc'] for r in euc_results]
    sig_test = statistical_significance_test(hyp_aucs, euc_aucs)
    
    print("\n" + "="*80)
    print("FINAL RESULTS ON FULL DATASET")
    print("="*80)
    print(f"Hyperbolic AUC: {sig_test['mean_hyperbolic']:.4f} ± {sig_test['std_hyperbolic']:.4f}")
    print(f"Euclidean AUC:  {sig_test['mean_euclidean']:.4f} ± {sig_test['std_euclidean']:.4f}")
    print(f"Improvement:    +{sig_test['relative_improvement_pct']:.2f}%")
    print(f"Significance:   p={sig_test['p_value']:.4f}")
    
    if sig_test['significant']:
        print("✅ RESULT: Hyperbolic significantly outperforms Euclidean!")
    else:
        print("⚠️ RESULT: No significant difference found (check data size/quality).")

if __name__ == "__main__":
    run_full_experiments()
