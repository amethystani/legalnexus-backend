"""
Euclidean Graph Convolutional Network Baseline

This is the FAIR BASELINE for comparing against hyperbolic GNN.
Uses standard GCN in Euclidean space instead of hyperbolic.

This comparison is ESSENTIAL for proving hyperbolic embeddings add value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


class EuclideanGCN(nn.Module):
    """
    Standard 2-layer GCN in Euclidean space.
    
    This is the DIRECT comparison to hyperbolic_gnn.py.
    Architecture is IDENTICAL except for the embedding space.
    
    If hyperbolic GNN doesn't outperform this, there's no novelty.
    """
    
    def __init__(self, input_dim=1024, hidden_dim=128, output_dim=64, dropout=0.5):
        super(EuclideanGCN, self).__init__()
        
        self.dropout = dropout
        
        # Layer 1: Input → Hidden
        self.gc1 = GCNConv(input_dim, hidden_dim)
        
        # Layer 2: Hidden → Output
        self.gc2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        """
        Forward pass through 2-layer GCN.
        
        Args:
            x: Input features (N x input_dim)
            edge_index: Edge connectivity (2 x E)
        
        Returns:
            Euclidean embeddings (N x output_dim)
        """
        # Layer 1
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Layer 2
        x = self.gc2(x, edge_index)
        
        return x


class EuclideanLinkPredictor(nn.Module):
    """
    Link prediction decoder using Euclidean distance.
    
    Predicts citation probability based on Euclidean distance between embeddings.
    """
    
    def __init__(self):
        super(EuclideanLinkPredictor, self).__init__()
        # No learnable parameters - just distance computation
    
    def forward(self, z, edge_index):
        """
        Predict link probability.
        
        Args:
            z: Node embeddings (N x dim)
            edge_index: Candidate edges (2 x E)
        
        Returns:
            Link probabilities (E,)
        """
        # Get source and target embeddings
        src = z[edge_index[0]]
        tgt = z[edge_index[1]]
        
        # Compute Euclidean distance
        dist = torch.norm(src - tgt, dim=1)
        
        # Convert distance to similarity (closer = higher probability)
        # Using exponential decay: p = exp(-dist)
        probs = torch.exp(-dist)
        
        return probs


def euclidean_contrastive_loss(embeddings, positive_edges, negative_edges, margin=1.0):
    """
    Contrastive loss in Euclidean space.
    
    Objective:
    - Minimize distance between cited cases (positive edges)
    - Maximize distance between non-cited cases (negative edges)
    
    Args:
        embeddings: Euclidean embeddings (N x dim)
        positive_edges: Citation edges (num_pos x 2)
        negative_edges: Non-citation edges (num_neg x 2)
        margin: Margin for negative samples
    
    Returns:
        Scalar loss
    """
    # Positive pairs (should be close)
    pos_src = embeddings[positive_edges[:, 0]]
    pos_tgt = embeddings[positive_edges[:, 1]]
    pos_dist = torch.norm(pos_src - pos_tgt, dim=1)
    
    # Negative pairs (should be far)
    neg_src = embeddings[negative_edges[:, 0]]
    neg_tgt = embeddings[negative_edges[:, 1]]
    neg_dist = torch.norm(neg_src - neg_tgt, dim=1)
    
    # Loss: minimize positive distance, maximize negative distance with margin
    pos_loss = torch.mean(pos_dist ** 2)
    neg_loss = torch.mean(torch.clamp(margin - neg_dist, min=0) ** 2)
    
    loss = pos_loss + neg_loss
    
    return loss, pos_loss.item(), neg_loss.item()


def train_euclidean_model(model, optimizer, data, positive_edges, negative_edges, epochs=100):
    """
    Training loop for Euclidean GCN.
    
    This should be IDENTICAL to hyperbolic training for fair comparison.
    """
    model.train()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data.x, data.edge_index)
        
        # Compute loss
        loss, pos_loss, neg_loss = euclidean_contrastive_loss(
            embeddings, 
            positive_edges, 
            negative_edges,
            margin=1.0
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Pos: {pos_loss:.4f} | Neg: {neg_loss:.4f}")
    
    return losses


def evaluate_link_prediction(model, data, test_edges, test_labels):
    """
    Evaluate link prediction performance.
    
    Returns:
        ROC-AUC score
    """
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        
        # Compute distances for test edges
        src = embeddings[test_edges[:, 0]]
        tgt = embeddings[test_edges[:, 1]]
        
        dist = torch.norm(src - tgt, dim=1).cpu().numpy()
        
        # Convert distance to probability (closer = higher prob)
        probs = np.exp(-dist)
        
        # Compute AUC
        auc = roc_auc_score(test_labels, probs)
    
    return auc


if __name__ == '__main__':
    """
    Test Euclidean GCN on synthetic data.
    
    This should be run ALONGSIDE hyperbolic_gnn.py for comparison.
    """
    
    print("="*80)
    print("EUCLIDEAN GCN BASELINE TEST")
    print("="*80)
    
    # Create synthetic data
    num_nodes = 100
    input_dim = 768  # Match Gemini embeddings
    
    # Random features
    x = torch.randn(num_nodes, input_dim)
    
    # Random edges (citation network)
    num_edges = 200
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create model
    model = EuclideanGCN(input_dim=input_dim, hidden_dim=128, output_dim=64)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)
    
    embeddings = model(data.x, data.edge_index)
    
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding space: Euclidean R^{embeddings.shape[1]}")
    
    print("\n✓ Euclidean GCN baseline ready for comparison")
    print("="*80)
