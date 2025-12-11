#!/usr/bin/env python3
"""
Test script for knowledge graph visualization
"""

import os
import sys
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Add the Backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_visualization_dependencies():
    """Test if visualization dependencies are available"""
    print("🧪 Testing Visualization Dependencies...")
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly available")
    except ImportError:
        print("❌ Plotly not available - install with: pip install plotly")
        return False
    
    try:
        import networkx as nx
        print("✅ NetworkX available")
    except ImportError:
        print("❌ NetworkX not available - install with: pip install networkx")
        return False
    
    try:
        from kg_visualizer import (
            get_graph_data, 
            create_network_graph, 
            get_case_connections
        )
        print("✅ Visualization modules available")
    except ImportError as e:
        print(f"❌ Visualization modules not available: {e}")
        return False
    
    return True

def test_neo4j_connection():
    """Test Neo4j connection for visualization"""
    print("\n🔌 Testing Neo4j Connection...")
    
    # Load environment variables
    load_dotenv()
    
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        print("❌ Neo4j credentials not found in environment variables")
        return None
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        print("✅ Neo4j connection successful")
        return graph
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return None

def test_graph_data_retrieval(graph):
    """Test retrieving graph data for visualization"""
    print("\n📊 Testing Graph Data Retrieval...")
    
    try:
        from kg_visualizer import get_graph_data
        
        # Test with small limit
        nodes_data, relationships_data = get_graph_data(graph, limit=10)
        
        print(f"✅ Retrieved {len(nodes_data)} nodes")
        print(f"✅ Retrieved {len(relationships_data)} relationships")
        
        if nodes_data:
            # Show sample node
            sample_node = nodes_data[0]
            print(f"📋 Sample node: {sample_node}")
            
        if relationships_data:
            # Show sample relationship
            sample_rel = relationships_data[0]
            print(f"🔗 Sample relationship: {sample_rel}")
            
        return nodes_data, relationships_data
        
    except Exception as e:
        print(f"❌ Error retrieving graph data: {e}")
        return [], []

def test_network_creation(nodes_data, relationships_data):
    """Test creating network graph"""
    print("\n🕸️ Testing Network Graph Creation...")
    
    if not nodes_data or not relationships_data:
        print("⚠️ No data available for network creation")
        return False
    
    try:
        from kg_visualizer import create_network_graph
        
        # Create network graph
        fig, G, node_info = create_network_graph(nodes_data, relationships_data)
        
        if fig and G and node_info:
            print(f"✅ Network graph created successfully")
            print(f"📊 Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
            print(f"📋 Node info available for {len(node_info)} nodes")
            
            # Show some network statistics
            if len(G.nodes()) > 0:
                import networkx as nx
                centrality = nx.degree_centrality(G)
                max_centrality = max(centrality.values()) if centrality else 0
                print(f"🏆 Maximum centrality: {max_centrality:.3f}")
                
            return True
        else:
            print("❌ Network graph creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error creating network graph: {e}")
        return False

def main():
    print("🚀 Knowledge Graph Visualization Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_visualization_dependencies()
    if not deps_ok:
        print("\n❌ Dependency test FAILED")
        print("Please install required packages:")
        print("pip install plotly networkx")
        return
    
    # Test Neo4j connection
    graph = test_neo4j_connection()
    if not graph:
        print("\n❌ Neo4j connection test FAILED")
        return
    
    # Test data retrieval
    nodes_data, relationships_data = test_graph_data_retrieval(graph)
    
    # Test network creation
    network_ok = test_network_creation(nodes_data, relationships_data)
    
    print("\n" + "=" * 50)
    if deps_ok and graph and network_ok:
        print("✅ All visualization tests PASSED!")
        print("🎯 You can now use the knowledge graph visualization.")
        print("\nTo use:")
        print("1. Run: streamlit run kg.py")
        print("2. Go to the '🕸️ Knowledge Graph Visualization' tab")
        print("3. Or run standalone: streamlit run kg_visualizer.py")
    else:
        print("❌ Some visualization tests FAILED")
        print("Please check the errors above and fix them.")

if __name__ == "__main__":
    main() 