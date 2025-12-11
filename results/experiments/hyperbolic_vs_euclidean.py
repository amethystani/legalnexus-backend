#!/usr/bin/env python3
"""
Hyperbolic vs Euclidean GNN Comparison Experiment

This is the KEY experiment to justify using hyperbolic embeddings.

If hyperbolic doesn't win, the paper has no novelty.
"""

import torch
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import json
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from hyperbolic_gnn import LegalHyperbolicModel, hyperbolic_contrastive_loss, normalize_adjacency
from baselines.euclidean_gnn import EuclideanGCN, euclidean_contrastive_loss
from geoopt import PoincareBall
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp


def load_citation_data():
    """Load citation network and case embeddings"""
    print("Loading citation network...")
    
    # Load embeddings
    import pickle
    with open('case_embeddings_gemini.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    case_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[cid] for cid in case_ids])
    
    print(f"Loaded {len(case_ids)} cases with {embeddings.shape[1]}-dim embeddings")
    
    # Create synthetic citation edges (in absence of real ones)
    # TODO: Replace with actual extracted citations
    num_cases = len(case_ids)
    num_edges = min(num_cases * 3, 200)  # ~3 citations per case
    
    edges = []
    for _ in range(num_edges):
        src = np.random.randint(0, num_cases)
        tgt = np.random.randint(0, num_cases)
        if src != tgt:
            edges.append([src, tgt])
    
    edges = np.array(edges).T  # Shape: (2, num_edges)
    
    print(f"Created graph: {num_cases} nodes, {edges.shape[1]} edges")
    
    return case_ids, embeddings, edges


def create_train_test_split(edges, test_ratio=0.2):
    """Split edges into train/test"""
    num_edges = edges.shape[1]
    num_test = int(num_edges * test_ratio)
    
    indices = np.random.permutation(num_edges)
    test_idx = indices[:num_test]
    train_idx = indices[num_test:]
    
    train_edges = edges[:, train_idx]
    test_edges = edges[:, test_idx]
    
    return train_edges, test_edges


def generate_negative_samples(num_nodes, positive_edges, num_negatives):
    """Generate negative edges (non-citations)"""
    positive_set = set(map(tuple, positive_edges.T))
    
    negatives = []
    while len(negatives) < num_negatives:
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        
        if src != tgt and (src, tgt) not in positive_set:
            negatives.append([src, tgt])
    
    return np.array(negatives).T


def train_and_evaluate_model(model_type, x, edge_index, positive_edges, test_edges, 
                              num_nodes, epochs=50):
    """
    Train and evaluate either Hyperbolic or Euclidean GNN.
    
    Args:
        model_type: 'hyperbolic' or 'euclidean'
        x: Node features (N x input_dim)
        edge_index: Graph structure (2 x E)
        positive_edges: Training citation edges
        test_edges: Test citation edges
        num_nodes: Number of nodes
        epochs: Training epochs
    
    Returns:
        ROC-AUC score on link prediction
    """
    
    input_dim = x.shape[1]
    
    # Create model
    if model_type == 'hyperbolic':
        model = LegalHyperbolicModel(input_dim=input_dim, hidden_dim=128, output_dim=64, c=1.0)
        loss_fn = hyperbolic_contrastive_loss
        manifold = PoincareBall(c=1.0)
        print(f"\n{'='*80}")
        print(f"TRAINING HYPERBOLIC GNN")
        print(f"{'='*80}")
    else:
        model = EuclideanGCN(input_dim=input_dim, hidden_dim=128, output_dim=64)
        loss_fn = euclidean_contrastive_loss
        manifold = None
        print(f"\n{'='*80}")
        print(f"TRAINING EUCLIDEAN GNN BASELINE")
        print(f"{'='*80}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Generate negative samples
    num_negatives = positive_edges.shape[1]
    negative_edges = generate_negative_samples(num_nodes, positive_edges, num_negatives)
    
    # Convert to torch
    x_torch = torch.FloatTensor(x)
    edge_index_torch = torch.LongTensor(edge_index)
    pos_edges_torch = torch.LongTensor(positive_edges.T)
    neg_edges_torch = torch.LongTensor(negative_edges.T)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'hyperbolic':
            # Normalize adjacency for hyperbolic GCN
            adj_normalized = normalize_adjacency(sp.csr_matrix(
                (np.ones(edge_index.shape[1]), edge_index),
                shape=(num_nodes, num_nodes)
            ))
            embeddings = model(x_torch, adj_normalized)
        else:
            # Euclidean uses edge_index directly
            embeddings = model(x_torch, edge_index_torch)
        
        # Compute loss
        if model_type == 'hyperbolic':
            loss, pos_loss, neg_loss = loss_fn(
                embeddings, pos_edges_torch, neg_edges_torch, manifold
            )
        else:
            loss, pos_loss, neg_loss = loss_fn(
                embeddings, pos_edges_torch, neg_edges_torch
            )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Pos: {pos_loss:.4f} | Neg: {neg_loss:.4f}")
    
    # Evaluation on test edges
    model.eval()
    
    with torch.no_grad():
        if model_type == 'hyperbolic':
            embeddings = model(x_torch, adj_normalized).detach()
        else:
            embeddings = model(x_torch, edge_index_torch).detach()
        
        # Test positive edges
        test_pos_edges = torch.LongTensor(test_edges.T)
        test_neg_edges = torch.LongTensor(
            generate_negative_samples(num_nodes, test_edges, test_edges.shape[1]).T
        )
        
        # Compute distances
        if model_type == 'hyperbolic':
            # Hyperbolic distance
            pos_src = embeddings[test_pos_edges[:, 0]]
            pos_tgt = embeddings[test_pos_edges[:, 1]]
            pos_dist = manifold.dist(pos_src, pos_tgt).cpu().numpy()
            
            neg_src = embeddings[test_neg_edges[:, 0]]
            neg_tgt = embeddings[test_neg_edges[:, 1]]
            neg_dist = manifold.dist(neg_src, neg_tgt).cpu().numpy()
        else:
            # Euclidean distance
            pos_src = embeddings[test_pos_edges[:, 0]]
            pos_tgt = embeddings[test_pos_edges[:, 1]]
            pos_dist = torch.norm(pos_src - pos_tgt, dim=1).cpu().numpy()
            
            neg_src = embeddings[test_neg_edges[:, 0]]
            neg_tgt = embeddings[test_neg_edges[:, 1]]
            neg_dist = torch.norm(neg_src - neg_tgt, dim=1).cpu().numpy()
        
        # Convert distances to probabilities (closer = higher prob)
        pos_probs = np.exp(-pos_dist)
        neg_probs = np.exp(-neg_dist)
        
        # Combine and compute AUC
        all_probs = np.concatenate([pos_probs, neg_probs])
        all_labels = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
        
        auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_type.upper()} GNN")
    print(f"{'='*80}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Link Prediction AUC: {auc:.4f}")
    print(f"{'='*80}\n")
    
    return auc, losses


def main():
    """Run full experiment comparing hyperbolic vs euclidean"""
    
    print("="*80)
    print("HYPERBOLIC VS EUCLIDEAN GNN EXPERIMENT")
    print("="*80)
    print("This experiment compares:")
    print("  1. Hyperbolic GNN (your claimed novelty)")
    print("  2. Euclidean GNN (standard baseline)")
    print("\nIf hyperbolic doesn't outperform euclidean, there's no novelty.")
    print("="*80 + "\n")
    
    # Load data
    case_ids, embeddings, edges = load_citation_data()
    
    # Split data
    train_edges, test_edges = create_train_test_split(edges, test_ratio=0.2)
    
    print(f"Data split:")
    print(f"  Train edges: {train_edges.shape[1]}")
    print(f"  Test edges: {test_edges.shape[1]}")
    
    num_nodes = len(case_ids)
    
    # Train Euclidean baseline
    euclidean_auc, euclidean_losses = train_and_evaluate_model(
        'euclidean', embeddings, edges, train_edges, test_edges, num_nodes, epochs=50
    )
    
    # Train Hyperbolic model
    hyperbolic_auc, hyperbolic_losses = train_and_evaluate_model(
        'hyperbolic', embeddings, edges, train_edges, test_edges, num_nodes, epochs=50
    )
    
    # Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Euclidean GNN AUC:  {euclidean_auc:.4f}")
    print(f"Hyperbolic GNN AUC: {hyperbolic_auc:.4f}")
    
    improvement = (hyperbolic_auc - euclidean_auc) / euclidean_auc * 100
    
    if hyperbolic_auc > euclidean_auc:
        print(f"\n✓ Hyperbolic WINS by {improvement:.1f}%")
        print("  → Justifies use of hyperbolic embeddings")
    else:
        print(f"\n✗ Euclidean WINS (hyperbolic is {abs(improvement):.1f}% worse)")
        print("  → No justification for hyperbolic embeddings")
        print("  → Cannot claim novelty!")
    
    print("="*80)
    
    # Save results
    results = {
        'euclidean_auc': float(euclidean_auc),
        'hyperbolic_auc': float(hyperbolic_auc),
        'improvement_percent': float(improvement),
        'winner': 'hyperbolic' if hyperbolic_auc > euclidean_auc else 'euclidean',
        'euclidean_losses': [float(x) for x in euclidean_losses],
        'hyperbolic_losses': [float(x) for x in hyperbolic_losses]
    }
    
    output_file = 'experiments/results/hyperbolic_vs_euclidean.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
