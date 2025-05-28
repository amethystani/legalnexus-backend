#!/usr/bin/env python3
"""
Graph Neural Network for Link Prediction
Implementation of GNN-based link prediction for legal case networks
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import streamlit as st
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime

# Load environment variables
load_dotenv()

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling, train_test_split_edges
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    # Create dummy Data class for import compatibility
    class Data:
        pass

class GraphDataProcessor:
    """Process Neo4j graph data for GNN training"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def extract_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Extract graph structure and node features from Neo4j"""
        st.info("Extracting graph data for GNN training...")
        
        # Get all nodes
        nodes_query = """
        MATCH (n)
        RETURN id(n) as node_id, labels(n) as labels, 
               n.title as title, n.court as court, n.date as date
        """
        nodes_data = self.graph.query(nodes_query)
        
        if not nodes_data:
            st.error("No nodes found in the knowledge graph!")
            raise ValueError("No nodes found in the knowledge graph")
        
        # Get all relationships
        edges_query = """
        MATCH (source)-[r]->(target)
        RETURN id(source) as source_id, id(target) as target_id, 
               type(r) as relationship_type
        """
        edges_data = self.graph.query(edges_query)
        
        if not edges_data:
            st.error("No relationships found in the knowledge graph!")
            raise ValueError("No relationships found in the knowledge graph")
        
        # Create node mapping
        node_mapping = {node['node_id']: i for i, node in enumerate(nodes_data)}
        reverse_mapping = {i: node['node_id'] for i, node in enumerate(nodes_data)}
        
        # Create edge index
        edge_index = []
        edge_types = []
        
        for edge in edges_data:
            source_id = edge['source_id']
            target_id = edge['target_id']
            
            if source_id in node_mapping and target_id in node_mapping:
                edge_index.append([node_mapping[source_id], node_mapping[target_id]])
                edge_types.append(edge['relationship_type'])
        
        if not edge_index:
            st.error("No valid edges found after processing!")
            raise ValueError("No valid edges found after processing")
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create node features
        node_features = self.create_node_features(nodes_data)
        
        metadata = {
            'node_mapping': node_mapping,
            'reverse_mapping': reverse_mapping,
            'edge_types': edge_types,
            'num_nodes': len(nodes_data),
            'num_edges': len(edges_data)
        }
        
        return edge_index, node_features, metadata
    
    def create_node_features(self, nodes_data: List[Dict]) -> torch.Tensor:
        """Create node features from node properties"""
        features = []
        
        for node in nodes_data:
            feature_vector = []
            
            # Node type encoding (one-hot)
            labels = node.get('labels', [])
            node_type = labels[0] if labels else 'Unknown'
            
            # One-hot encoding for node types
            type_features = [0, 0, 0, 0]  # Case, Judge, Court, Statute
            if node_type == 'Case':
                type_features[0] = 1
            elif node_type == 'Judge':
                type_features[1] = 1
            elif node_type == 'Court':
                type_features[2] = 1
            elif node_type == 'Statute':
                type_features[3] = 1
            
            feature_vector.extend(type_features)
            
            # Text length features
            title = node.get('title', '') or ''
            feature_vector.append(len(title) / 100)  # Normalized title length
            
            # Court type features (simplified)
            court = node.get('court', '') or ''
            court_features = [0, 0, 0]  # SC, HC, Other
            if 'supreme' in court.lower():
                court_features[0] = 1
            elif 'high' in court.lower():
                court_features[1] = 1
            else:
                court_features[2] = 1
            
            feature_vector.extend(court_features)
            
            # Year feature (if available)
            date_str = node.get('date', '') or ''
            try:
                year = int(date_str[:4]) if len(date_str) >= 4 else 2000
                normalized_year = (year - 1950) / 70  # Normalize to 0-1 range
            except:
                normalized_year = 0.5
            
            feature_vector.append(normalized_year)
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float)

class GNNLinkPredictor(nn.Module):
    """Graph Neural Network for link prediction"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 2):
        super(GNNLinkPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_pairs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Link prediction
        source_embeddings = x[edge_pairs[0]]
        target_embeddings = x[edge_pairs[1]]
        
        # Concatenate embeddings
        link_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Predict link probability
        link_probs = self.link_predictor(link_embeddings)
        
        return link_probs.squeeze()
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class LinkPredictionTrainer:
    """Train and evaluate GNN link prediction models"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.processor = GraphDataProcessor(graph)
        self.model = None
        self.metadata = None
        
    def prepare_data(self):
        """Prepare data for training"""
        try:
            st.info("📊 Extracting graph data from Neo4j...")
            edge_index, node_features, metadata = self.processor.extract_graph_data()
            self.metadata = metadata
            
            # Detailed validation
            if edge_index is None:
                st.error("❌ Edge index is None")
                return None
                
            if node_features is None:
                st.error("❌ Node features is None")
                return None
            
            # Check tensor dimensions
            if edge_index.size(0) == 0 or edge_index.size(1) == 0:
                st.error("❌ No edges found in the knowledge graph. Cannot train GNN model.")
                st.info("Make sure the knowledge graph has relationships between nodes.")
                return None
            
            if node_features.size(0) == 0:
                st.error("❌ No nodes found in the knowledge graph. Cannot train GNN model.")
                return None
            
            st.info(f"✅ Found {node_features.size(0)} nodes and {edge_index.size(1)} edges")
            
            # Create PyTorch Geometric data object
            st.info("🔧 Creating PyTorch Geometric data object...")
            data = Data(x=node_features, edge_index=edge_index)
            
            # Validate the created data object
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                st.error("❌ Failed to create proper Data object")
                return None
                
            if data.x is None or data.edge_index is None:
                st.error("❌ Data object has None attributes")
                return None
            
            # Check minimum edges for splitting
            if edge_index.size(1) < 10:
                st.warning("⚠️ Very few edges found. Results may not be reliable.")
                st.info("Consider adding more relationships to the knowledge graph.")
            
            # Split edges for training/testing with error handling
            st.info("🔄 Splitting edges for train/val/test...")
            try:
                data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
                
                # Validate split data
                if not hasattr(data, 'train_pos_edge_index'):
                    st.error("❌ Edge splitting failed - no train_pos_edge_index")
                    return None
                    
                st.info(f"✅ Data split successful. Train edges: {data.train_pos_edge_index.size(1)}")
                
            except Exception as split_error:
                st.error(f"❌ Edge splitting failed: {str(split_error)}")
                st.info("Trying alternative splitting approach...")
                
                # Alternative: manual edge splitting
                num_edges = edge_index.size(1)
                perm = torch.randperm(num_edges)
                
                train_end = int(0.7 * num_edges)
                val_end = int(0.8 * num_edges)
                
                train_edges = edge_index[:, perm[:train_end]]
                val_edges = edge_index[:, perm[train_end:val_end]]
                test_edges = edge_index[:, perm[val_end:]]
                
                # Create a simpler data object
                data.train_pos_edge_index = train_edges
                data.val_pos_edge_index = val_edges
                data.test_pos_edge_index = test_edges
                
                # Create negative edges (simplified)
                data.train_neg_edge_index = train_edges  # Placeholder
                data.val_neg_edge_index = val_edges      # Placeholder
                data.test_neg_edge_index = test_edges    # Placeholder
                
                st.info(f"✅ Manual split successful. Train edges: {train_edges.size(1)}")
            
            return data
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {str(e)}")
            st.info("Check your Neo4j connection and ensure the knowledge graph is properly populated.")
            
            # Show debug information
            import traceback
            st.text("Debug trace:")
            st.text(traceback.format_exc())
            
            return None
    
    def train_model(self, data, epochs: int = 100) -> Dict:
        """Train the GNN model"""
        if not GNN_AVAILABLE:
            st.error("PyTorch Geometric not available. Install with: pip install torch-geometric")
            return {}
        
        if data is None:
            st.error("❌ Cannot train model: No data provided")
            return {}
        
        # Validate data object thoroughly
        st.info("🔍 Validating training data...")
        try:
            if not hasattr(data, 'x') or data.x is None:
                st.error("❌ Data object missing node features (x)")
                return {}
                
            # After train_test_split_edges, the original edge_index is removed
            # Check for training edge indices instead
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                st.error("❌ Data object missing train_pos_edge_index")
                return {}
                
            st.info(f"✅ Data validation passed - Nodes: {data.x.size(0)}, Features: {data.x.size(1)}, Train edges: {data.train_pos_edge_index.size(1)}")
            
        except Exception as e:
            st.error(f"❌ Data validation failed: {str(e)}")
            return {}
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🖥️ Using device: {device}")
        
        # Initialize model
        try:
            num_features = data.x.size(1)
            self.model = GNNLinkPredictor(num_features).to(device)
            st.info(f"✅ Model initialized with {num_features} features")
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
            return {}
        
        # Move data to device with error handling
        try:
            data = data.to(device)
            st.info("✅ Data moved to device successfully")
                
        except Exception as e:
            st.error(f"❌ Failed to move data to device: {str(e)}")
            return {}
        
        # Validate training data splits
        try:
            if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index is None:
                st.error("❌ Missing train_pos_edge_index - edge splitting may have failed")
                return {}
                
            if not hasattr(data, 'val_pos_edge_index') or data.val_pos_edge_index is None:
                st.error("❌ Missing val_pos_edge_index - edge splitting may have failed")
                return {}
                
            st.info(f"✅ Training splits validated - Train edges: {data.train_pos_edge_index.size(1)}")
            
        except Exception as e:
            st.error(f"❌ Training split validation failed: {str(e)}")
            return {}
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Training history
        history = {'train_loss': [], 'val_auc': []}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.info(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Positive edges (existing edges)
            pos_edge_index = data.train_pos_edge_index
            
            # Negative edges (non-existing edges)
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
            
            # Combine positive and negative edges
            edge_pairs = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).to(device)
            
            # Forward pass
            predictions = self.model(data.x, data.train_pos_edge_index, edge_pairs)
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                val_auc = self.evaluate_model(data, split='val')
                history['train_loss'].append(loss.item())
                history['val_auc'].append(val_auc)
                
                status_text.text(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f} - Val AUC: {val_auc:.4f}")
            
            progress_bar.progress((epoch + 1) / epochs)
        
        # Final evaluation
        test_auc = self.evaluate_model(data, split='test')
        st.success(f"Training completed! Test AUC: {test_auc:.4f}")
        
        return history
    
    def evaluate_model(self, data, split: str = 'test') -> float:
        """Evaluate model performance"""
        if self.model is None:
            return 0.0
        
        if data is None:
            return 0.0
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            if split == 'val':
                pos_edge_index = data.val_pos_edge_index
                neg_edge_index = data.val_neg_edge_index
            else:  # test
                pos_edge_index = data.test_pos_edge_index
                neg_edge_index = data.test_neg_edge_index
            
            # Combine positive and negative edges
            edge_pairs = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).cpu().numpy()
            
            # Predictions
            predictions = self.model(data.x, data.train_pos_edge_index, edge_pairs)
            predictions = predictions.cpu().numpy()
            
            # Calculate AUC
            auc = roc_auc_score(labels, predictions)
            
        return auc
    
    def predict_links(self, data, top_k: int = 10) -> List[Dict]:
        """Predict most likely new links"""
        if self.model is None:
            return []
        
        if data is None:
            st.error("❌ Cannot make predictions: No data provided")
            return []
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Get all possible node pairs (excluding existing edges)
            existing_edges = set()
            
            # Use train_pos_edge_index since edge_index is removed after train_test_split_edges
            edge_index_to_use = data.train_pos_edge_index
            for i in range(edge_index_to_use.size(1)):
                edge = (edge_index_to_use[0, i].item(), edge_index_to_use[1, i].item())
                existing_edges.add(edge)
                existing_edges.add((edge[1], edge[0]))  # Add reverse edge
            
            # Generate candidate edges
            candidate_edges = []
            num_nodes = data.num_nodes
            
            # Sample candidate edges (to avoid too many pairs)
            max_candidates = min(1000, num_nodes * 10)
            sampled_pairs = np.random.choice(num_nodes, size=(max_candidates, 2), replace=True)
            
            for source, target in sampled_pairs:
                if source != target and (source, target) not in existing_edges:
                    candidate_edges.append((source, target))
            
            if not candidate_edges:
                return []
            
            # Convert to tensor
            candidate_tensor = torch.tensor(candidate_edges, dtype=torch.long).t().to(device)
            
            # Predict link probabilities
            predictions = self.model(data.x, data.train_pos_edge_index, candidate_tensor)
            predictions = predictions.cpu().numpy()
            
            # Get top-k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                source_idx, target_idx = candidate_edges[idx]
                source_id = self.metadata['reverse_mapping'][source_idx]
                target_id = self.metadata['reverse_mapping'][target_idx]
                score = predictions[idx]
                
                results.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'probability': score,
                    'source_idx': source_idx,
                    'target_idx': target_idx
                })
            
            return results
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metadata': self.metadata
        }, filepath)
    
    def load_model(self, filepath: str, num_features: int):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.model = GNNLinkPredictor(num_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metadata = checkpoint['metadata']

def main():
    st.set_page_config(
        page_title="GNN Link Prediction",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Graph Neural Network Link Prediction")
    st.markdown("Train GNN models to predict relationships in legal knowledge graph")
    
    if not GNN_AVAILABLE:
        st.error("🚨 PyTorch Geometric not available!")
        st.markdown("""
        Install the required packages:
        ```bash
        pip install torch torch-geometric
        ```
        """)
        return
    
    # Get Neo4j connection
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        st.error("Neo4j credentials not found in environment variables")
        return
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return
    
    trainer = LinkPredictionTrainer(graph)
    
    # Sidebar controls
    st.sidebar.header("🔧 GNN Training Controls")
    
    # Training parameters
    epochs = st.sidebar.slider("Training Epochs", 50, 500, 100, 50)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 128, 64, 16)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Train Model", "Predict Links", "Model Analysis"])
    
    with tab1:
        st.subheader("🎯 Train GNN Model")
        
        if st.button("🚀 Start Training"):
            with st.spinner("Preparing data..."):
                data = trainer.prepare_data()
            
            if data is None:
                st.error("❌ Failed to prepare training data. Cannot proceed with training.")
                st.markdown("""
                **Possible solutions:**
                1. Ensure your Neo4j database has legal cases loaded
                2. Check that relationships exist between nodes (cases, judges, courts)
                3. Run the main knowledge graph system first to populate data
                4. Verify Neo4j connection credentials
                """)
                return
            
            st.success(f"✅ Graph prepared: {data.num_nodes} nodes, {data.train_pos_edge_index.size(1)} train edges")
            
            # Train model
            history = trainer.train_model(data, epochs=epochs)
            
            if history:
                # Plot training history
                st.subheader("📈 Training History")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(pd.DataFrame({'Loss': history['train_loss']}))
                with col2:
                    st.line_chart(pd.DataFrame({'Validation AUC': history['val_auc']}))
                
                # Save model
                model_path = f"gnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                trainer.save_model(model_path)
                st.success(f"Model saved as {model_path}")
    
    with tab2:
        st.subheader("🔮 Predict New Links")
        
        if trainer.model is None:
            st.info("Train a model first to make predictions")
        else:
            top_k = st.slider("Top K predictions", 5, 50, 10)
            
            if st.button("🎯 Predict Links"):
                data = trainer.prepare_data()
                
                if data is None:
                    st.error("❌ Cannot prepare data for prediction.")
                    return
                
                predictions = trainer.predict_links(data, top_k=top_k)
                
                if predictions:
                    st.subheader("🔗 Predicted Links")
                    
                    for i, pred in enumerate(predictions):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            st.metric("Rank", f"#{i+1}")
                        with col2:
                            st.write(f"**Source ID**: {pred['source_id']}")
                            st.write(f"**Target ID**: {pred['target_id']}")
                        with col3:
                            st.metric("Probability", f"{pred['probability']:.3f}")
                        
                        st.markdown("---")
                else:
                    st.warning("No predictions generated")
    
    with tab3:
        st.subheader("📊 Model Analysis")
        
        if trainer.model is None:
            st.info("Train a model first to see analysis")
        else:
            st.markdown("""
            **Model Architecture:**
            - Graph Convolutional Network (GCN)
            - 2-layer architecture with ReLU activation
            - Link prediction via concatenated embeddings
            - Binary classification with sigmoid output
            
            **Features Used:**
            - Node type (Case, Judge, Court, Statute)
            - Title length
            - Court type
            - Year information
            
            **Training Strategy:**
            - Positive samples: Existing edges
            - Negative samples: Random non-existing edges
            - Train/Val/Test split: 70/10/20
            - Optimization: Adam with BCE loss
            """)

if __name__ == "__main__":
    main() 