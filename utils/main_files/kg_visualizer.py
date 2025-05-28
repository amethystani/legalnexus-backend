#!/usr/bin/env python3
"""
Knowledge Graph Visualizer
Interactive network visualization of legal cases and their connections
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

def get_neo4j_connection():
    """Get Neo4j connection"""
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        st.error("Neo4j credentials not found in environment variables")
        return None
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

def get_graph_data(graph, limit=50):
    """Get nodes and relationships from Neo4j"""
    try:
        # Get nodes with their properties
        nodes_query = f"""
        MATCH (n)
        RETURN id(n) as node_id, labels(n) as labels, n.title as title, 
               n.name as name, n.court as court, n.date as date
        LIMIT {limit}
        """
        
        # Get relationships
        relationships_query = f"""
        MATCH (source)-[r]->(target)
        RETURN id(source) as source_id, id(target) as target_id, 
               type(r) as relationship_type, labels(source) as source_labels,
               labels(target) as target_labels
        LIMIT {limit * 2}
        """
        
        nodes_data = graph.query(nodes_query)
        relationships_data = graph.query(relationships_query)
        
        return nodes_data, relationships_data
        
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return [], []

def create_network_graph(nodes_data, relationships_data, selected_case_id=None):
    """Create interactive network graph using Plotly"""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    node_info = {}
    for node in nodes_data:
        node_id = node['node_id']
        labels = node['labels'][0] if node['labels'] else 'Unknown'
        title = node.get('title') or node.get('name') or f'Node {node_id}'
        
        # Truncate long titles
        display_title = title[:30] + "..." if title and len(title) > 30 else (title or f'Node {node_id}')
        
        G.add_node(node_id)
        node_info[node_id] = {
            'label': labels,
            'title': title,
            'display_title': display_title,
            'court': node.get('court') or '',
            'date': node.get('date') or ''
        }
    
    # Add edges
    for rel in relationships_data:
        source_id = rel['source_id']
        target_id = rel['target_id']
        if source_id in node_info and target_id in node_info:
            G.add_edge(source_id, target_id, relationship=rel['relationship_type'])
    
    if len(G.nodes()) == 0:
        st.warning("No data found in the knowledge graph")
        return None
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node traces by type
    node_traces = {}
    colors = {
        'Case': '#ff6b6b',      # Red
        'Judge': '#4ecdc4',     # Teal  
        'Court': '#45b7d1',     # Blue
        'Statute': '#96ceb4',   # Green
        'Unknown': '#feca57'    # Yellow
    }
    
    for node_id in G.nodes():
        label = node_info[node_id]['label']
        if label not in node_traces:
            node_traces[label] = {
                'x': [], 'y': [], 'text': [], 'ids': [],
                'customdata': [], 'hovertext': []
            }
        
        x, y = pos[node_id]
        node_traces[label]['x'].append(x)
        node_traces[label]['y'].append(y)
        node_traces[label]['text'].append(node_info[node_id]['display_title'])
        node_traces[label]['ids'].append(node_id)
        
        # Enhanced hover info
        hover_text = f"""
        <b>{node_info[node_id]['title']}</b><br>
        Type: {label}<br>
        Court: {node_info[node_id]['court']}<br>
        Date: {node_info[node_id]['date']}<br>
        Node ID: {node_id}
        """
        node_traces[label]['hovertext'].append(hover_text)
        node_traces[label]['customdata'].append({
            'id': node_id,
            'title': node_info[node_id]['title'],
            'type': label
        })
    
    # Create edge traces
    edge_x, edge_y = [], []
    edge_info = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Store edge info for hover
        relationship_type = edge[2].get('relationship', 'CONNECTED')
        source_title = node_info[edge[0]]['display_title']
        target_title = node_info[edge[1]]['display_title']
        edge_info.append(f"{source_title} --{relationship_type}--> {target_title}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Relationships'
    ))
    
    # Add nodes by type
    for label, trace_data in node_traces.items():
        # Highlight selected case
        node_color = colors.get(label, '#feca57')
        node_size = []
        node_colors = []
        
        for node_id in trace_data['ids']:
            if selected_case_id and node_id == selected_case_id:
                node_size.append(25)  # Larger for selected
                node_colors.append('#ff1744')  # Bright red for selected
            elif selected_case_id and node_id in G.neighbors(selected_case_id):
                node_size.append(20)  # Medium for connected
                node_colors.append('#ff9800')  # Orange for connected
            else:
                node_size.append(15)  # Normal size
                node_colors.append(node_color)  # Normal color
        
        fig.add_trace(go.Scatter(
            x=trace_data['x'], y=trace_data['y'],
            mode='markers+text',
            text=trace_data['text'],
            textposition="middle center",
            textfont=dict(size=8),
            hovertext=trace_data['hovertext'],
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            customdata=trace_data['customdata'],
            name=label
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Legal Knowledge Graph Network",
            x=0.5,
            font=dict(size=20)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Drag to pan, scroll to zoom. Click nodes to explore connections.",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="#888", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=700
    )
    
    return fig, G, node_info

def get_case_connections(graph, case_id):
    """Get detailed connections for a specific case"""
    try:
        connections_query = f"""
        MATCH (c:Case)-[r]-(connected)
        WHERE id(c) = {case_id}
        RETURN type(r) as relationship_type, 
               labels(connected) as connected_labels,
               connected.title as connected_title,
               connected.name as connected_name,
               id(connected) as connected_id
        """
        
        connections = graph.query(connections_query)
        return connections
        
    except Exception as e:
        st.error(f"Error getting case connections: {e}")
        return []

def show_case_details(graph, case_id, node_info):
    """Show detailed information about a selected case"""
    if case_id not in node_info:
        return
    
    case_info = node_info[case_id]
    
    st.subheader(f"📋 Case Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Title:** {case_info['title']}")
        st.write(f"**Type:** {case_info['label']}")
    with col2:
        st.write(f"**Court:** {case_info['court']}")
        st.write(f"**Date:** {case_info['date']}")
    
    # Get connections
    connections = get_case_connections(graph, case_id)
    
    if connections:
        st.subheader("🔗 Connected Entities")
        
        # Group connections by type
        connection_groups = {}
        for conn in connections:
            rel_type = conn['relationship_type']
            if rel_type not in connection_groups:
                connection_groups[rel_type] = []
            
            connected_name = conn.get('connected_title') or conn.get('connected_name', 'Unknown')
            connected_label = conn['connected_labels'][0] if conn['connected_labels'] else 'Unknown'
            
            connection_groups[rel_type].append({
                'name': connected_name,
                'type': connected_label,
                'id': conn['connected_id']
            })
        
        for rel_type, entities in connection_groups.items():
            with st.expander(f"{rel_type} ({len(entities)} connections)"):
                for entity in entities:
                    icon = {'Judge': '👨‍⚖️', 'Court': '🏛️', 'Statute': '📜', 'Case': '📋'}.get(entity['type'], '🔸')
                    st.write(f"{icon} {entity['name']} ({entity['type']})")

def main():
    st.set_page_config(
        page_title="Knowledge Graph Visualizer",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Legal Knowledge Graph Visualizer")
    st.markdown("Explore the connections between legal cases, judges, courts, and statutes")
    
    # Sidebar controls
    st.sidebar.header("🔧 Visualization Controls")
    
    # Get Neo4j connection
    graph = get_neo4j_connection()
    if not graph:
        st.error("Cannot connect to Neo4j database")
        return
    
    # Limit control
    node_limit = st.sidebar.slider("Number of nodes to display", 20, 200, 50, 10)
    
    # Get graph data
    with st.spinner("Loading knowledge graph data..."):
        nodes_data, relationships_data = get_graph_data(graph, node_limit)
    
    if not nodes_data:
        st.warning("No data found in the knowledge graph")
        return
    
    # Initialize session state for selected case
    if 'selected_case_id' not in st.session_state:
        st.session_state.selected_case_id = None
    
    # Case selection
    st.sidebar.subheader("📋 Select a Case")
    
    # Get all cases for selection
    cases = [node for node in nodes_data if 'Case' in node.get('labels', [])]
    
    if cases:
        case_options = {f"{case['title'][:50]}...": case['node_id'] 
                       for case in cases if case.get('title')}
        
        selected_case_title = st.sidebar.selectbox(
            "Choose a case to highlight:",
            ["None"] + list(case_options.keys())
        )
        
        if selected_case_title != "None":
            st.session_state.selected_case_id = case_options[selected_case_title]
        else:
            st.session_state.selected_case_id = None
    
    # Create and display the network graph
    fig, G, node_info = create_network_graph(
        nodes_data, 
        relationships_data, 
        st.session_state.selected_case_id
    )
    
    if fig:
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Show graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", len(G.nodes()))
        with col2:
            st.metric("Total Connections", len(G.edges()))
        with col3:
            case_count = len([n for n in nodes_data if 'Case' in n.get('labels', [])])
            st.metric("Legal Cases", case_count)
        with col4:
            judge_count = len([n for n in nodes_data if 'Judge' in n.get('labels', [])])
            st.metric("Judges", judge_count)
        
        # Show case details if one is selected
        if st.session_state.selected_case_id:
            show_case_details(graph, st.session_state.selected_case_id, node_info)
    
    # Network analysis section
    st.subheader("📊 Network Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Most Connected Nodes")
        if G and len(G.nodes()) > 0:
            # Calculate degree centrality
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (node_id, centrality_score) in enumerate(top_nodes):
                if node_id in node_info:
                    name = node_info[node_id]['title']
                    node_type = node_info[node_id]['label']
                    connections = len(list(G.neighbors(node_id)))
                    st.write(f"{i+1}. **{name[:40]}...** ({node_type}) - {connections} connections")
    
    with col2:
        st.subheader("🔗 Relationship Types")
        if relationships_data:
            # Count relationship types
            rel_counts = {}
            for rel in relationships_data:
                rel_type = rel['relationship_type']
                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
            
            for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{rel_type}**: {count}")

if __name__ == "__main__":
    main() 