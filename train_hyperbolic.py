"""
Training Pipeline for Hyperbolic GNN on Legal Citation Network

Trains HGCN model to learn embeddings where:
- Supreme Court cases → near center (low radius)
- Lower court cases → near edge (high radius)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
import os
from geoopt import PoincareBall
from geoopt.optim import RiemannianSGD, RiemannianAdam

from hyperbolic_gnn import (
    LegalHyperbolicModel,
    hyperbolic_contrastive_loss,
    normalize_adjacency
)
from extract_citation_network import sample_negative_edges


def train_epoch(model, optimizer, features, adj, positive_edges, negative_edges, manifold):
    """Single training epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    embeddings = model(features, adj)
    
    # Calculate loss
    loss, pos_loss, neg_loss = hyperbolic_contrastive_loss(
        embeddings, positive_edges, negative_edges, manifold
    )
    
    # Backward pass
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item(), pos_loss, neg_loss


def evaluate(model, features, adj, positive_edges, negative_edges, manifold):
    """Evaluate model on validation set"""
    model.eval()
    
    with torch.no_grad():
        embeddings = model(features, adj)
        loss, pos_loss, neg_loss = hyperbolic_contrastive_loss(
            embeddings, positive_edges, negative_edges, manifold
        )
    
    return loss.item(), pos_loss, neg_loss


def train_hyperbolic_model(
    data_path='data/citation_network.pkl',
    output_dir='models',
    epochs=200,
    hidden_dim=128,
    output_dim=64,
    learning_rate=0.01,
    weight_decay=0.0005,
    dropout=0.5,
    train_split=0.8,
    patience=20
):
    """
    Train hyperbolic GNN on citation network.
    
    Args:
        data_path: Path to preprocessed citation network
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        learning_rate: Learning rate
        weight_decay: L2 regularization
        dropout: Dropout rate
        train_split: Train/val split ratio
        patience: Early stopping patience
    """
    print("="*80)
    print("HYPERBOLIC GNN TRAINING")
    print("="*80)
    
    # Load data
    print("\n1. Loading citation network...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    edges = data['edges']
    case_ids = data['case_ids']
    adj = data['adj']
    features = data['features']
    id_to_idx = data['id_to_idx']
    metadata = data['metadata']
    
    print(f"   ✓ Loaded {len(case_ids)} cases, {len(edges)} edges")
    print(f"   ✓ Feature dim: {features.shape[1]}")
    
    # Normalize adjacency matrix
    print("\n2. Preparing graph...")
    adj_normalized = normalize_adjacency(adj)
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    
    # Split edges into train/val
    np.random.shuffle(edges)
    split_idx = int(len(edges) * train_split)
    train_edges = edges[:split_idx]
    val_edges = edges[split_idx:]
    
    # Convert edge lists to index pairs
    train_edge_indices = [(id_to_idx[src], id_to_idx[tgt]) 
                          for src, tgt in train_edges 
                          if src in id_to_idx and tgt in id_to_idx]
    val_edge_indices = [(id_to_idx[src], id_to_idx[tgt]) 
                        for src, tgt in val_edges 
                        if src in id_to_idx and tgt in id_to_idx]
    
    train_pos_edges = torch.LongTensor(train_edge_indices)
    val_pos_edges = torch.LongTensor(val_edge_indices)
    
    # Sample negative edges
    train_neg_edges = torch.LongTensor(
        sample_negative_edges(train_edge_indices, len(case_ids), num_negatives_per_positive=5)
    )
    val_neg_edges = torch.LongTensor(
        sample_negative_edges(val_edge_indices, len(case_ids), num_negatives_per_positive=5)
    )
    
    print(f"   ✓ Train edges: {len(train_pos_edges)} pos, {len(train_neg_edges)} neg")
    print(f"   ✓ Val edges: {len(val_pos_edges)} pos, {len(val_neg_edges)} neg")
    
    # Initialize model
    print("\n3. Initializing model...")
    model = LegalHyperbolicModel(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout
    )
    
    manifold = PoincareBall(c=1.0)
    
    # Optimizer (Riemannian Adam for hyperbolic parameters)
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print("\n4. Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss, train_pos, train_neg = train_epoch(
            model, optimizer, features_tensor, adj_normalized,
            train_pos_edges, train_neg_edges, manifold
        )
        
        # Validate
        val_loss, val_pos, val_neg = evaluate(
            model, features_tensor, adj_normalized,
            val_pos_edges, val_neg_edges, manifold
        )
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} (pos: {train_pos:.4f}, neg: {train_neg:.4f}) | "
                  f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'case_ids': case_ids,
                'id_to_idx': id_to_idx,
                'metadata': metadata
            }, os.path.join(output_dir, 'hgcn_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n   Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n   ✓ Best validation loss: {best_val_loss:.4f}")
    
    # Generate final embeddings
    print("\n5. Generating embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model(features_tensor, adj_normalized)
    
    # Save embeddings
    embeddings_dict = {
        case_id: embeddings[id_to_idx[case_id]].numpy()
        for case_id in case_ids
    }
    
    with open(os.path.join(output_dir, 'hgcn_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"   ✓ Saved embeddings to {output_dir}/hgcn_embeddings.pkl")
    
    # Analyze hierarchy
    print("\n6. Analyzing learned hierarchy...")
    radii = torch.norm(embeddings, dim=1).numpy()
    
    court_radii = {'Supreme Court': [], 'High Court': [], 'Lower Court': []}
    for case_id, radius in zip(case_ids, radii):
        court = metadata[case_id]['court']
        court_radii[court].append(radius)
    
    print("\n   Radius by court level:")
    for court in ['Supreme Court', 'High Court', 'Lower Court']:
        if court_radii[court]:
            mean_radius = np.mean(court_radii[court])
            std_radius = np.std(court_radii[court])
            print(f"     {court:15s}: {mean_radius:.4f} ± {std_radius:.4f}")
    
    print("\n✅ Training complete!")
    
    return model, embeddings_dict


if __name__ == "__main__":
    model, embeddings = train_hyperbolic_model(
        epochs=200,
        hidden_dim=128,
        output_dim=64,
        learning_rate=0.01,
        patience=20
    )
