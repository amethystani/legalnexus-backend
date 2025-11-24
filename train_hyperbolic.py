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
        embeddings, positive_edges, negative_edges, manifold, margin=2.0
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
            embeddings, positive_edges, negative_edges, manifold, margin=2.0
        )
    
    return loss.item(), pos_loss, neg_loss


def train_hyperbolic_model(
    data_path='data/citation_network.pkl',
    output_dir='models',
    epochs=5,
    hidden_dim=128,  # Reduced from 128 for speed
    output_dim=64,  # Reduced from 64 for speed
    learning_rate=0.001,
    weight_decay=0.0005,
    dropout=0.5,  # Reduced from 0.5
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
    
    # Extract data (new format)
    case_ids = data['case_ids']
    adj = data['adj']
    metadata = data['metadata']
    features = data['embeddings']  # Use embeddings as features
    
    # Build id_to_idx mapping
    id_to_idx = {cid: idx for idx, cid in enumerate(case_ids)}
    
    # Extract edges from adjacency matrix
    edges = []
    adj_coo = adj.tocoo()
    for i, j in zip(adj_coo.row, adj_coo.col):
        if i < j:  # Only add once (undirected)
            edges.append((case_ids[i], case_ids[j]))
    
    print(f"   ✓ Loaded {len(case_ids)} cases, {len(edges)} edges")
    print(f"   ✓ Feature dim: {features.shape[1]}")
    
    # Optimize for maximum performance (16 threads)
    device = torch.device("cpu")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
    print("🚀 Using CPU with 16 threads (maximum performance)")
    
    # Normalize adjacency matrix
    print("\n2. Preparing graph...")
    adj_normalized = normalize_adjacency(adj)
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Convert adjacency to sparse tensor
    if not torch.is_tensor(adj_normalized):
        adj_coo = adj_normalized.tocoo()
        indices = torch.LongTensor([adj_coo.row, adj_coo.col])
        values = torch.FloatTensor(adj_coo.data)
        shape = torch.Size(adj_coo.shape)
        adj_tensor = torch.sparse.FloatTensor(indices, values, shape).to(device)
    else:
        adj_tensor = adj_normalized.to(device)
    
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
    
    train_pos_edges = torch.LongTensor(train_edge_indices).to(device)
    val_pos_edges = torch.LongTensor(val_edge_indices).to(device)
    
    # Sample negative edges (will be moved to device in loop)
    # ...
    
    print(f"   ✓ Train edges: {len(train_pos_edges)} pos")
    print(f"   ✓ Val edges: {len(val_pos_edges)} pos")
    
    # Initialize model
    print("\n3. Initializing model...")
    model = LegalHyperbolicModel(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    manifold = PoincareBall(c=1.0)
    
    # Optimizer (Riemannian Adam for hyperbolic parameters)
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print("\n4. Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    from tqdm import tqdm
    import time
    
    start_time = time.time()
    pbar = tqdm(range(epochs), desc="Training Epochs")
    
    for epoch in pbar:
        epoch_start = time.time()
        
        # Sample negative edges for this epoch (dynamic sampling)
        train_neg_indices = sample_negative_edges(train_edge_indices, len(case_ids), num_negatives_per_positive=5)
        train_neg_edges = torch.LongTensor(train_neg_indices).to(device)
        
        val_neg_indices = sample_negative_edges(val_edge_indices, len(case_ids), num_negatives_per_positive=5)
        val_neg_edges = torch.LongTensor(val_neg_indices).to(device)
        
        # Train
        train_loss, train_pos, train_neg = train_epoch(
            model, optimizer, features_tensor, adj_tensor,
            train_pos_edges, train_neg_edges, manifold
        )
        
        # Validate
        val_loss, val_pos, val_neg = evaluate(
            model, features_tensor, adj_tensor,
            val_pos_edges, val_neg_edges, manifold
        )
        
        scheduler.step(val_loss)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
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
    
    total_time = time.time() - start_time
    print(f"\n   ✓ Training took {total_time:.1f}s ({total_time/epochs:.2f}s/epoch)")    
    print(f"\n   ✓ Best validation loss: {best_val_loss:.4f}")
    
    # Generate final embeddings
    print("\n5. Generating embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model(features_tensor, adj_tensor)
        embeddings = embeddings.cpu()  # Move back to CPU for saving
    
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
    
    from collections import defaultdict
    court_radii = defaultdict(list)
    for case_id, radius in zip(case_ids, radii):
        court = metadata[case_id]['court']
        court_radii[court].append(radius)
    
    print("\n   Radius by court level:")
    # Sort by mean radius to show hierarchy
    sorted_courts = sorted(court_radii.keys(), key=lambda c: np.mean(court_radii[c]))
    
    for court in sorted_courts:
        if court_radii[court]:
            mean_radius = np.mean(court_radii[court])
            std_radius = np.std(court_radii[court])
            count = len(court_radii[court])
            print(f"     {court:15s}: {mean_radius:.4f} ± {std_radius:.4f} (n={count})")
    
    print("\n✅ Training complete!")
    
    return model, embeddings_dict


if __name__ == "__main__":
    model, embeddings = train_hyperbolic_model(
        epochs=5,
        hidden_dim=128,
        output_dim=64,
        learning_rate=0.01,
        patience=20
    )
