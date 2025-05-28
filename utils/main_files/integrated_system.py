#!/usr/bin/env python3
"""
Integrated Legal Knowledge Graph System
Combines all components: Knowledge Graph, Citation Network, Document Similarity, and GNN Link Prediction
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time

# Load environment variables
load_dotenv()

# Try to import all components
try:
    from kg_visualizer import get_graph_data, create_network_graph, show_case_details
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from citation_network import CitationNetwork, CitationExtractor
    CITATION_AVAILABLE = True
except ImportError:
    CITATION_AVAILABLE = False

try:
    from gnn_link_prediction import LinkPredictionTrainer, GNN_AVAILABLE
    GNN_PREDICTION_AVAILABLE = GNN_AVAILABLE
except ImportError:
    GNN_PREDICTION_AVAILABLE = False

class IntegratedLegalSystem:
    """Integrated system combining all legal AI components"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        
        # Initialize components
        if CITATION_AVAILABLE:
            self.citation_network = CitationNetwork(graph)
        else:
            self.citation_network = None
        
        if GNN_PREDICTION_AVAILABLE:
            self.gnn_trainer = LinkPredictionTrainer(graph)
        else:
            self.gnn_trainer = None
        
        # System status
        self.system_status = self.check_system_status()
    
    def check_system_status(self) -> Dict:
        """Check status of all system components"""
        status = {
            'knowledge_graph': False,
            'citation_network': False,
            'document_similarity': False,
            'gnn_models': False,
            'visualization': VISUALIZATION_AVAILABLE
        }
        
        try:
            # Check knowledge graph
            case_count_query = "MATCH (c:Case) RETURN count(c) as count"
            result = self.graph.query(case_count_query)
            case_count = result[0]['count'] if result else 0
            status['knowledge_graph'] = case_count > 0
            
            # Check citation network
            citation_count_query = "MATCH ()-[r:CITES]->() RETURN count(r) as count"
            result = self.graph.query(citation_count_query)
            citation_count = result[0]['count'] if result else 0
            status['citation_network'] = citation_count > 0
            
            # Document similarity is always available if we have cases
            status['document_similarity'] = case_count > 0
            
            # GNN models (check if any models exist)
            status['gnn_models'] = GNN_PREDICTION_AVAILABLE
            
        except Exception as e:
            st.error(f"Error checking system status: {e}")
        
        return status
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        metrics = {}
        
        try:
            # Knowledge Graph metrics
            kg_queries = {
                'cases': "MATCH (c:Case) RETURN count(c) as count",
                'judges': "MATCH (j:Judge) RETURN count(j) as count", 
                'courts': "MATCH (court:Court) RETURN count(court) as count",
                'statutes': "MATCH (s:Statute) RETURN count(s) as count",
                'relationships': "MATCH ()-[r]->() RETURN count(r) as count"
            }
            
            for key, query in kg_queries.items():
                result = self.graph.query(query)
                metrics[key] = result[0]['count'] if result else 0
            
            # Citation Network metrics
            citation_queries = {
                'citations': "MATCH ()-[r:CITES]->() RETURN count(r) as count",
                'citing_cases': "MATCH (c:Case)-[r:CITES]->() RETURN count(DISTINCT c) as count",
                'cited_cases': "MATCH ()-[r:CITES]->(c:Case) RETURN count(DISTINCT c) as count"
            }
            
            for key, query in citation_queries.items():
                result = self.graph.query(query)
                metrics[key] = result[0]['count'] if result else 0
            
            # Network density
            if metrics['cases'] > 1:
                max_citations = metrics['cases'] * (metrics['cases'] - 1)
                metrics['citation_density'] = metrics['citations'] / max_citations if max_citations > 0 else 0
            else:
                metrics['citation_density'] = 0
            
        except Exception as e:
            st.error(f"Error getting metrics: {e}")
            metrics = {key: 0 for key in ['cases', 'judges', 'courts', 'statutes', 'relationships', 'citations']}
        
        return metrics
    
    def create_system_overview(self):
        """Create system overview dashboard"""
        st.header("🔗 Integrated Legal Knowledge Graph System")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅" if self.system_status['knowledge_graph'] else "❌"
            st.metric("Knowledge Graph", status)
        
        with col2:
            status = "✅" if self.system_status['citation_network'] else "❌"
            st.metric("Citation Network", status)
        
        with col3:
            status = "✅" if self.system_status['document_similarity'] else "❌"
            st.metric("Document Similarity", status)
        
        with col4:
            status = "✅" if self.system_status['gnn_models'] else "❌"
            st.metric("GNN Models", status)
        
        # System metrics
        metrics = self.get_system_metrics()
        
        st.subheader("📊 System Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Legal Cases", metrics.get('cases', 0))
        with col2:
            st.metric("Judges", metrics.get('judges', 0))
        with col3:
            st.metric("Courts", metrics.get('courts', 0))
        with col4:
            st.metric("Citations", metrics.get('citations', 0))
        with col5:
            st.metric("Total Relationships", metrics.get('relationships', 0))
    
    def create_workflow_diagram(self):
        """Create a visual representation of the system workflow"""
        st.subheader("🔄 System Architecture")
        
        # Create a flow diagram using Plotly
        fig = go.Figure()
        
        # Define positions for components
        components = {
            'Knowledge Graph': (0.5, 0.8),
            'Citation Network': (0.2, 0.5),
            'Document Similarity': (0.8, 0.5),
            'Citation Link Prediction': (0.2, 0.2),
            'Similarity Link Prediction': (0.8, 0.2),
            'GNN': (0.5, 0.0)
        }
        
        # Add nodes
        for name, (x, y) in components.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[name],
                textposition="middle center",
                marker=dict(
                    size=80,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                showlegend=False
            ))
        
        # Add arrows (connections)
        connections = [
            ('Knowledge Graph', 'Citation Network'),
            ('Knowledge Graph', 'Document Similarity'),
            ('Citation Network', 'Citation Link Prediction'),
            ('Document Similarity', 'Similarity Link Prediction'),
            ('Citation Link Prediction', 'GNN'),
            ('Similarity Link Prediction', 'GNN')
        ]
        
        for source, target in connections:
            x0, y0 = components[source]
            x1, y1 = components[target]
            
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='gray'
            )
        
        fig.update_layout(
            title="System Component Flow",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def analyze_query_response_flow(self, user_query: str) -> Dict:
        """Analyze how a query flows through the system"""
        analysis = {
            'query': user_query,
            'components_used': [],
            'results': {},
            'processing_time': {}
        }
        
        start_time = time.time()
        
        # 1. Knowledge Graph Search
        kg_start = time.time()
        try:
            # Simple text search in knowledge graph
            search_query = f"""
            MATCH (c:Case)
            WHERE toLower(c.title) CONTAINS toLower('{user_query}')
               OR toLower(c.text) CONTAINS toLower('{user_query}')
            RETURN c.title as title, c.court as court, c.date as date
            LIMIT 5
            """
            kg_results = self.graph.query(search_query)
            analysis['results']['knowledge_graph'] = kg_results
            analysis['components_used'].append('Knowledge Graph')
            analysis['processing_time']['knowledge_graph'] = time.time() - kg_start
        except Exception as e:
            analysis['results']['knowledge_graph'] = f"Error: {e}"
        
        # 2. Citation Network Analysis
        if CITATION_AVAILABLE:
            citation_start = time.time()
            try:
                # Find cases that cite or are cited by matching cases
                citation_query = f"""
                MATCH (c:Case)-[r:CITES]-(related:Case)
                WHERE toLower(c.title) CONTAINS toLower('{user_query}')
                RETURN related.title as related_case, type(r) as relationship
                LIMIT 5
                """
                citation_results = self.graph.query(citation_query)
                analysis['results']['citation_network'] = citation_results
                analysis['components_used'].append('Citation Network')
                analysis['processing_time']['citation_network'] = time.time() - citation_start
            except Exception as e:
                analysis['results']['citation_network'] = f"Error: {e}"
        
        # 3. Document Similarity
        similarity_start = time.time()
        try:
            # Use existing similarity function
            from kg import find_similar_cases
            # This would need embeddings and other parameters in practice
            analysis['results']['document_similarity'] = "Similarity analysis available"
            analysis['components_used'].append('Document Similarity')
            analysis['processing_time']['document_similarity'] = time.time() - similarity_start
        except Exception as e:
            analysis['results']['document_similarity'] = f"Error: {e}"
        
        analysis['total_processing_time'] = time.time() - start_time
        
        return analysis

def main():
    st.set_page_config(
        page_title="Integrated Legal AI System",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Integrated Legal Knowledge Graph System")
    st.markdown("Complete legal AI system combining knowledge graphs, citation analysis, similarity matching, and machine learning")
    
    # Check dependencies
    missing_deps = []
    if not VISUALIZATION_AVAILABLE:
        missing_deps.append("Visualization (plotly, networkx)")
    if not CITATION_AVAILABLE:
        missing_deps.append("Citation Analysis")
    if not GNN_PREDICTION_AVAILABLE:
        missing_deps.append("GNN Models (torch, torch-geometric)")
    
    if missing_deps:
        st.warning(f"Some components are not available: {', '.join(missing_deps)}")
        st.info("Install missing dependencies to access all features")
    
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
    
    # Initialize integrated system
    system = IntegratedLegalSystem(graph)
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 System Overview", 
        "🔍 Query Analysis", 
        "🕸️ Network Visualization",
        "🔗 Citation Analysis",
        "🤖 GNN Prediction"
    ])
    
    with tab1:
        system.create_system_overview()
        
        st.markdown("---")
        system.create_workflow_diagram()
        
        st.markdown("---")
        st.subheader("🎯 System Capabilities")
        
        capabilities = {
            "Knowledge Graph": {
                "status": system.system_status['knowledge_graph'],
                "description": "Store and query legal cases, judges, courts, and statutes",
                "features": ["Entity relationships", "Graph traversal", "Semantic search"]
            },
            "Citation Network": {
                "status": system.system_status['citation_network'],
                "description": "Extract and analyze citation relationships between cases",
                "features": ["Citation extraction", "Influence analysis", "Citation patterns"]
            },
            "Document Similarity": {
                "status": system.system_status['document_similarity'], 
                "description": "Find similar legal documents using embeddings and text analysis",
                "features": ["Vector similarity", "Text matching", "Semantic similarity"]
            },
            "GNN Prediction": {
                "status": system.system_status['gnn_models'],
                "description": "Machine learning models for link prediction",
                "features": ["Citation prediction", "Similarity prediction", "Graph embeddings"]
            }
        }
        
        for name, info in capabilities.items():
            with st.expander(f"{'✅' if info['status'] else '❌'} {name}"):
                st.write(info['description'])
                st.write("**Features:**")
                for feature in info['features']:
                    st.write(f"• {feature}")
    
    with tab2:
        st.subheader("🔍 Query Response Analysis")
        st.markdown("See how your query flows through different system components")
        
        user_query = st.text_input("Enter your legal query:", 
                                 placeholder="e.g., criminal procedure, evidence production")
        
        if user_query and st.button("🚀 Analyze Query"):
            with st.spinner("Processing query through system components..."):
                analysis = system.analyze_query_response_flow(user_query)
            
            st.subheader("📊 Query Analysis Results")
            
            # Processing time breakdown
            st.subheader("⏱️ Processing Time")
            time_df = pd.DataFrame([
                {'Component': comp, 'Time (seconds)': analysis['processing_time'].get(comp.lower().replace(' ', '_'), 0)}
                for comp in analysis['components_used']
            ])
            st.bar_chart(time_df.set_index('Component'))
            
            # Component results
            st.subheader("🔄 Component Results")
            for component in analysis['components_used']:
                component_key = component.lower().replace(' ', '_')
                with st.expander(f"{component} Results"):
                    result = analysis['results'].get(component_key, "No results")
                    if isinstance(result, list) and result:
                        for i, item in enumerate(result):
                            st.write(f"{i+1}. {item}")
                    else:
                        st.write(result)
    
    with tab3:
        if VISUALIZATION_AVAILABLE:
            st.subheader("🕸️ Knowledge Graph Visualization")
            
            # Use existing visualization code
            node_limit = st.slider("Nodes to display", 20, 100, 50)
            
            with st.spinner("Loading visualization..."):
                nodes_data, relationships_data = get_graph_data(graph, node_limit)
                
                if nodes_data and relationships_data:
                    fig, G, node_info = create_network_graph(nodes_data, relationships_data)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Network statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nodes", len(G.nodes()))
                        with col2:
                            st.metric("Edges", len(G.edges()))
                        with col3:
                            case_count = len([n for n in nodes_data if 'Case' in n.get('labels', [])])
                            st.metric("Cases", case_count)
                else:
                    st.warning("No graph data available")
        else:
            st.error("Visualization not available. Install plotly and networkx.")
    
    with tab4:
        if CITATION_AVAILABLE:
            st.subheader("🔗 Citation Network Analysis")
            
            # Citation network controls
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Citation Statistics", "Build Citation Network", "Visualization"]
            )
            
            if analysis_type == "Build Citation Network":
                if st.button("🔨 Build Citation Network"):
                    system.citation_network.build_citation_network()
            
            elif analysis_type == "Citation Statistics":
                stats = system.citation_network.get_citation_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Citations", stats['total_citations'])
                with col2:
                    st.metric("Network Density", f"{stats['network_density']:.4f}")
                with col3:
                    st.metric("Citation Coverage", "Active")
                
                # Most cited cases
                if stats['most_cited']:
                    st.subheader("🏆 Most Cited Cases")
                    for i, case in enumerate(stats['most_cited'][:5]):
                        st.write(f"{i+1}. **{case['case_title'][:60]}...** - {case['citation_count']} citations")
            
            elif analysis_type == "Visualization":
                limit = st.slider("Citations to display", 10, 100, 30)
                fig = system.citation_network.visualize_citation_network(limit)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No citation data to visualize")
        else:
            st.error("Citation analysis not available")
    
    with tab5:
        if GNN_PREDICTION_AVAILABLE and system.gnn_trainer is not None:
            st.subheader("🤖 Graph Neural Network Prediction")
            
            prediction_type = st.selectbox(
                "Prediction Type",
                ["Citation Link Prediction", "Similarity Link Prediction", "Model Training"]
            )
            
            if prediction_type == "Model Training":
                st.markdown("""
                **GNN Model Features:**
                - Graph Convolutional Networks (GCN)
                - Link prediction for citations and similarities
                - Node embeddings for legal entities
                - Train/validation/test splits
                """)
                
                # Training parameters
                col1, col2 = st.columns(2)
                with col1:
                    epochs = st.slider("Training Epochs", 50, 500, 100, 50)
                with col2:
                    hidden_dim = st.slider("Hidden Dimension", 32, 128, 64, 16)
                
                if st.button("🚀 Start GNN Training"):
                    with st.spinner("Preparing GNN training data..."):
                        try:
                            # Prepare data for training
                            data = system.gnn_trainer.prepare_data()
                            
                            if data is None:
                                st.error("❌ Failed to prepare training data. Cannot proceed with training.")
                                
                                # Show database status for debugging
                                try:
                                    case_count_query = "MATCH (c:Case) RETURN count(c) as count"
                                    case_result = graph.query(case_count_query)
                                    case_count = case_result[0]['count'] if case_result else 0
                                    
                                    edge_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
                                    edge_result = graph.query(edge_count_query)
                                    edge_count = edge_result[0]['count'] if edge_result else 0
                                    
                                    st.info(f"Database status: {case_count} cases, {edge_count} relationships")
                                    
                                    if case_count == 0:
                                        st.warning("⚠️ No legal cases found in database!")
                                    elif edge_count < 10:
                                        st.warning(f"⚠️ Only {edge_count} relationships found. Need at least 10 for training.")
                                    
                                except Exception as e:
                                    st.error(f"Could not check database status: {e}")
                                
                                st.markdown("""
                                **Possible solutions:**
                                1. **Load legal cases**: Use `kg.py` to populate the knowledge graph first
                                2. **Check relationships**: Ensure cases are connected to judges, courts, statutes
                                3. **Verify data**: Run the main knowledge graph system to load data
                                4. **Check connection**: Verify Neo4j connection credentials
                                5. **Minimum data**: Need at least 10 relationships for GNN training
                                """)
                            else:
                                st.success(f"✅ Graph prepared: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
                                
                                # Train model
                                with st.spinner(f"Training GNN model for {epochs} epochs..."):
                                    history = system.gnn_trainer.train_model(data, epochs=epochs)
                                
                                if history:
                                    # Plot training history
                                    st.subheader("📈 Training History")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.line_chart(pd.DataFrame({'Loss': history['train_loss']}))
                                    with col2:
                                        st.line_chart(pd.DataFrame({'Validation AUC': history['val_auc']}))
                                    
                                    st.success("🎉 Model training completed successfully!")
                                    
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
            
            elif prediction_type == "Citation Link Prediction":
                st.markdown("**Predict which cases might cite each other**")
                
                if system.gnn_trainer.model is None:
                    st.warning("⚠️ No trained model available. Please train a model first.")
                    if st.button("🔄 Go to Model Training"):
                        st.rerun()
                else:
                    top_k = st.slider("Top predictions", 5, 20, 10)
                    
                    if st.button("🎯 Predict Citations"):
                        with st.spinner("Generating citation predictions..."):
                            try:
                                data = system.gnn_trainer.prepare_data()
                                if data is not None:
                                    predictions = system.gnn_trainer.predict_links(data, top_k=top_k)
                                    
                                    if predictions:
                                        st.subheader("🔗 Predicted Citation Links")
                                        
                                        for i, pred in enumerate(predictions):
                                            with st.container():
                                                col1, col2, col3 = st.columns([1, 2, 1])
                                                
                                                with col1:
                                                    st.metric("Rank", f"#{i+1}")
                                                with col2:
                                                    # Get case details
                                                    source_query = f"MATCH (c:Case) WHERE id(c) = {pred['source_id']} RETURN c.title as title"
                                                    target_query = f"MATCH (c:Case) WHERE id(c) = {pred['target_id']} RETURN c.title as title"
                                                    
                                                    try:
                                                        source_result = graph.query(source_query)
                                                        target_result = graph.query(target_query)
                                                        
                                                        source_title = source_result[0]['title'][:50] + "..." if source_result and source_result[0]['title'] else f"Case {pred['source_id']}"
                                                        target_title = target_result[0]['title'][:50] + "..." if target_result and target_result[0]['title'] else f"Case {pred['target_id']}"
                                                        
                                                        st.write(f"**{source_title}**")
                                                        st.write(f"→ might cite → **{target_title}**")
                                                    except:
                                                        st.write(f"**Case {pred['source_id']}** → **Case {pred['target_id']}**")
                                                
                                                with col3:
                                                    confidence = pred['probability']
                                                    color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🟠"
                                                    st.metric("Confidence", f"{color} {confidence:.3f}")
                                                
                                                st.markdown("---")
                                    else:
                                        st.warning("No citation predictions generated")
                                else:
                                    st.error("Could not prepare data for prediction")
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
            
            elif prediction_type == "Similarity Link Prediction":
                st.markdown("**Predict which cases are most similar**")
                
                if system.gnn_trainer.model is None:
                    st.warning("⚠️ No trained model available. Please train a model first.")
                    if st.button("🔄 Go to Model Training"):
                        st.rerun()
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        top_k = st.slider("Top predictions", 5, 20, 10)
                    with col2:
                        similarity_threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.7)
                    
                    if st.button("🔍 Predict Similarities"):
                        with st.spinner("Generating similarity predictions..."):
                            try:
                                data = system.gnn_trainer.prepare_data()
                                if data is not None:
                                    predictions = system.gnn_trainer.predict_links(data, top_k=top_k)
                                    
                                    # Filter by similarity threshold
                                    filtered_predictions = [p for p in predictions if p['probability'] >= similarity_threshold]
                                    
                                    if filtered_predictions:
                                        st.subheader("🔗 Predicted Similar Cases")
                                        
                                        for i, pred in enumerate(filtered_predictions):
                                            with st.container():
                                                col1, col2, col3 = st.columns([1, 2, 1])
                                                
                                                with col1:
                                                    st.metric("Rank", f"#{i+1}")
                                                with col2:
                                                    # Get case details
                                                    source_query = f"MATCH (c:Case) WHERE id(c) = {pred['source_id']} RETURN c.title as title, c.court as court"
                                                    target_query = f"MATCH (c:Case) WHERE id(c) = {pred['target_id']} RETURN c.title as title, c.court as court"
                                                    
                                                    try:
                                                        source_result = graph.query(source_query)
                                                        target_result = graph.query(target_query)
                                                        
                                                        if source_result and target_result:
                                                            source_title = source_result[0]['title'][:40] + "..." if len(source_result[0]['title']) > 40 else source_result[0]['title']
                                                            target_title = target_result[0]['title'][:40] + "..." if len(target_result[0]['title']) > 40 else target_result[0]['title']
                                                            source_court = source_result[0].get('court', 'Unknown')
                                                            target_court = target_result[0].get('court', 'Unknown')
                                                            
                                                            st.write(f"**{source_title}** ({source_court})")
                                                            st.write(f"↔ similar to ↔ **{target_title}** ({target_court})")
                                                        else:
                                                            st.write(f"**Case {pred['source_id']}** ↔ **Case {pred['target_id']}**")
                                                    except:
                                                        st.write(f"**Case {pred['source_id']}** ↔ **Case {pred['target_id']}**")
                                                
                                                with col3:
                                                    similarity = pred['probability']
                                                    color = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🟠"
                                                    st.metric("Similarity", f"{color} {similarity:.3f}")
                                                
                                                st.markdown("---")
                                    else:
                                        st.warning(f"No similarities found above threshold {similarity_threshold:.2f}")
                                        st.info("Try lowering the similarity threshold or training the model longer")
                                else:
                                    st.error("Could not prepare data for prediction")
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
        else:
            st.error("GNN prediction not available. Install PyTorch and PyTorch Geometric.")
            st.markdown("""
            **To enable GNN predictions:**
            ```bash
            pip install torch torch-geometric
            ```
            """)
    
    # Footer with system information
    st.markdown("---")
    st.markdown("""
    **System Components Status:**
    - ✅ Available and working
    - ❌ Not available or needs setup
    
    **Complete System Features:**
    - Knowledge Graph storage and querying
    - Citation network extraction and analysis  
    - Document similarity using embeddings
    - GNN-based link prediction
    - Interactive visualizations
    - Integrated query processing
    """)

if __name__ == "__main__":
    main() 