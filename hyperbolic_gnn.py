"""
Hyperbolic Graph Neural Networks for Legal Citation Networks

Implements Graph Convolution on the Poincaré Ball to capture
the hierarchical structure of legal precedents:
- Supreme Court cases → Center of ball (radius ~0)
- District Court cases → Edge of ball (radius ~1)

Mathematical Foundation:
- Poincaré Ball Model: (𝔻ⁿ, gₓ) with curvature c
- Möbius operations for aggregation
- Exponential/Logarithmic maps for projection

References:
- Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"
- Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter, PoincareBall
from geoopt.manifolds.stereographic.math import mobius_add, logmap0, expmap0
import numpy as np


class HyperbolicGraphConv(nn.Module):
    """
    Graph Convolutional Layer in Hyperbolic Space (Poincaré Ball).
    
    Workflow:
    1. Input features → Tangent space at origin (logmap0)
    2. Linear transformation in tangent space
    3. Aggregate neighbors using adjacency matrix
    4. Project result → Poincaré Ball (expmap0)
    
    This avoids numerical instability by operating in tangent space
    for linear operations and aggregation.
    """
    
    def __init__(self, in_features, out_features, c=1.0, use_bias=True):
        super(HyperbolicGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c  # Curvature of the Poincaré Ball
        
        self.ball = PoincareBall(c=c)
        
        # Learnable parameters (defined in Euclidean space)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass of hyperbolic graph convolution.
        
        Args:
            x: Input features (N x in_features), can be Euclidean or Hyperbolic
            adj: Sparse adjacency matrix (N x N)
        
        Returns:
            Output features in Poincaré Ball (N x out_features)
        """
        # 1. Project to tangent space at origin (if input is hyperbolic)
        # For first layer, input is Euclidean, so this is identity
        x_tangent = self.ball.logmap0(x) if x.dtype == torch.float32 else x
        
        # 2. Linear transformation in tangent space
        support = torch.mm(x_tangent, self.weight)
        
        if self.bias is not None:
            support = support + self.bias
        
        # 3. Aggregate neighbors (standard GCN aggregation in tangent space)
        # adj is assumed to be normalized: D^(-1/2) A D^(-1/2)
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        
        # 4. Project aggregated features to Poincaré Ball
        output_hyp = self.ball.expmap0(output)
        
        return output_hyp


class LegalHyperbolicModel(nn.Module):
    """
    2-Layer Hyperbolic GCN for Legal Citation Networks.
    
    Architecture:
    - Input: Euclidean features (e.g., Gemini embeddings, dim=768)
    - Layer 1: Euclidean → Hyperbolic (hidden_dim)
    - Non-linearity in tangent space
    - Layer 2: Hyperbolic → Hyperbolic (output_dim)
    - Output: Hyperbolic embeddings in Poincaré Ball
    
    The learned embeddings will have:
    - Supreme Court cases near center (low radius)
    - Lower court cases near edge (high radius)
    """
    
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=64, c=1.0, dropout=0.5):
        super(LegalHyperbolicModel, self).__init__()
        
        self.c = c
        self.ball = PoincareBall(c=c)
        self.dropout = dropout
        
        # Layer 1: Euclidean → Hyperbolic
        self.gc1 = HyperbolicGraphConv(input_dim, hidden_dim, c=c)
        
        # Layer 2: Hyperbolic → Hyperbolic
        self.gc2 = HyperbolicGraphConv(hidden_dim, output_dim, c=c)
    
    def forward(self, x, adj):
        """
        Forward pass through 2-layer HGCN.
        
        Args:
            x: Input features (N x input_dim), Euclidean
            adj: Normalized adjacency matrix (N x N)
        
        Returns:
            Hyperbolic embeddings (N x output_dim) in Poincaré Ball
        """
        # Layer 1: Input (Euclidean) → Hyperbolic
        x = self.gc1(x, adj)
        
        # Non-linearity: Apply ReLU in tangent space
        x = self.ball.logmap0(x)  # Hyperbolic → Tangent
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ball.expmap0(x)  # Tangent → Hyperbolic
        
        # Layer 2: Hyperbolic → Hyperbolic
        x = self.gc2(x, adj)
        
        return x  # Returns hyperbolic embeddings
    
    def get_radius(self, embeddings):
        """
        Calculate radius (distance from origin) for each embedding.
        Used to verify hierarchy: Supreme Court → low radius, District → high radius
        
        Args:
            embeddings: Hyperbolic embeddings (N x dim)
        
        Returns:
            Radius for each node (N,)
        """
        return torch.norm(embeddings, dim=1)


class FermiDiracDecoder(nn.Module):
    """
    Fermi-Dirac decoder for link prediction in hyperbolic space.
    
    Predicts probability of citation link based on hyperbolic distance.
    This is the standard decoder for hyperbolic embeddings.
    
    Formula:
        p(link | d) = 1 / (exp((d - r) / t) + 1)
    
    where:
        d = hyperbolic distance
        r = radius parameter (decision boundary)
        t = temperature (sharpness of boundary)
    """
    
    def __init__(self, r=2.0, t=1.0):
        super(FermiDiracDecoder, self).__init__()
        self.r = nn.Parameter(torch.Tensor([r]))
        self.t = nn.Parameter(torch.Tensor([t]))
    
    def forward(self, dist):
        """
        Args:
            dist: Hyperbolic distances (batch_size,)
        
        Returns:
            Link probabilities (batch_size,)
        """
        probs = 1.0 / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


def hyperbolic_contrastive_loss(embeddings, positive_edges, negative_edges, manifold, margin=5.0):
    """
    Contrastive loss in hyperbolic space for citation network training.
    
    Objective:
    - Minimize distance between cited cases (positive edges)
    - Maximize distance between non-cited cases (negative edges)
    
    Args:
        embeddings: Hyperbolic embeddings (N x dim)
        positive_edges: Citation edges (num_pos x 2) [source, target]
        negative_edges: Non-citation edges (num_neg x 2)
        manifold: PoincareBall manifold
        margin: Margin for negative samples
    
    Returns:
        Scalar loss
    """
    # Positive pairs (should be close)
    pos_src = embeddings[positive_edges[:, 0]]
    pos_tgt = embeddings[positive_edges[:, 1]]
    pos_dist = manifold.dist(pos_src, pos_tgt)
    
    # Negative pairs (should be far)
    neg_src = embeddings[negative_edges[:, 0]]
    neg_tgt = embeddings[negative_edges[:, 1]]
    neg_dist = manifold.dist(neg_src, neg_tgt)
    
    # Loss: minimize positive distance, maximize negative distance with margin
    pos_loss = torch.mean(pos_dist ** 2)
    neg_loss = torch.mean(torch.clamp(margin - neg_dist ** 2, min=0))
    
    loss = pos_loss + neg_loss
    
    return loss, pos_loss.item(), neg_loss.item()


def normalize_adjacency(adj):
    """
    Normalize adjacency matrix for GCN: D^(-1/2) A D^(-1/2)
    
    Args:
        adj: Adjacency matrix (N x N), can be dense or scipy sparse
    
    Returns:
        Normalized adjacency matrix (torch sparse tensor)
    """
    import scipy.sparse as sp
    
    # Convert to scipy sparse if needed
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    
    # Compute D^(-1/2)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # D^(-1/2) A D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    # Convert to torch sparse tensor
    adj_normalized = adj_normalized.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    shape = adj_normalized.shape
    
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))


# Helper functions for Poincaré operations (if geoopt versions are unstable)

def poincare_distance(u, v, c=1.0, eps=1e-5):
    """
    Compute Poincaré distance between two points.
    
    Formula:
        d(u,v) = arcosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
    
    Args:
        u, v: Points in Poincaré Ball (batch_size x dim)
        c: Curvature
        eps: Small constant for numerical stability
    
    Returns:
        Distances (batch_size,)
    """
    sqrt_c = c ** 0.5
    
    # Compute squared norms
    u_norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True)
    v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
    
    # Ensure norms are < 1 (inside the ball)
    u_norm_sq = torch.clamp(u_norm_sq, max=1 - eps)
    v_norm_sq = torch.clamp(v_norm_sq, max=1 - eps)
    
    # Compute numerator and denominator
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)
    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq.squeeze()) * (1 - v_norm_sq.squeeze())
    
    # Distance formula
    arg = 1 + numerator / (denominator + eps)
    arg = torch.clamp(arg, min=1 + eps)  # Ensure arg >= 1 for arcosh
    
    dist = (1 / sqrt_c) * torch.acosh(arg)
    
    return dist
