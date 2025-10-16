#!/usr/bin/env python3
"""
Create a comprehensive knowledge graph for the entire dataset (96,000+ records)
and save it as a visualization image.
"""

import os
import sys
import json
import time
from typing import List, Dict
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

# Import our data loading functions
from kg import load_legal_data, create_legal_knowledge_graph
from utils.main_files.csv_data_loader import load_all_csv_data

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization dependencies not available: {e}")
    VISUALIZATION_AVAILABLE = False

# Load environment variables
load_dotenv()
# Also load from .env.neo4j if it exists
if os.path.exists('.env.neo4j'):
    load_dotenv('.env.neo4j')

# Set Google Gemini API key
GOOGLE_API_KEY = "AIzaSyCE64GFYnFZnZktAATpIx0zTp3HpUAUSbA"
genai.configure(api_key=GOOGLE_API_KEY)

def connect_to_neo4j():
    """Connect to Neo4j database"""
    # Use the correct Neo4j URI from .env.neo4j
    neo4j_url = "neo4j+s://01cbe45f.databases.neo4j.io"
    neo4j_username = "neo4j"
    neo4j_password = "wNtwNLOZzK713gOX1aMHt-k5FejnliJUwiJRPFeSfzY"
    
    print(f"Connecting to Neo4j: {neo4j_url}")
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        print("✅ Connected to Neo4j database")
        return graph
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return None

def load_complete_dataset():
    """Load the complete dataset including CSV data"""
    print("🔄 Loading complete dataset...")
    
    # Load all legal data including CSV datasets
    print("Loading JSON files and CSV datasets...")
    all_docs = load_legal_data(data_path="data", include_csv=True, max_csv_cases=None)  # Load all CSV cases
    
    print(f"✅ Loaded {len(all_docs)} legal documents")
    return all_docs

def create_comprehensive_knowledge_graph(graph, docs, embeddings):
    """Create knowledge graph from the complete dataset"""
    print("🔄 Creating comprehensive knowledge graph...")
    
    # Clear existing data
    print("Clearing existing graph data...")
    cypher = "MATCH (n) DETACH DELETE n;"
    graph.query(cypher)
    
    # Process documents in batches to avoid memory issues
    batch_size = 1000
    total_docs = len(docs)
    processed = 0
    
    print(f"Processing {total_docs} documents in batches of {batch_size}...")
    
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_docs = docs[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: documents {batch_start+1}-{batch_end}")
        
        # Process each document in the batch
        for i, doc in enumerate(batch_docs):
            try:
                # Create Case node
                case_props = {
                    'title': doc.metadata.get('title', 'Unknown Case'),
                    'court': doc.metadata.get('court', 'Unknown Court'),
                    'date': doc.metadata.get('judgment_date', 'Unknown Date'),
                    'source': doc.metadata.get('source', ''),
                    'id': doc.metadata.get('id', f'case_{batch_start + i}'),
                    'text': doc.page_content[:5000]  # Limit text to avoid memory issues
                }
                
                # Create the Case node
                cypher = """
                MERGE (c:Case {id: $id})
                SET c.title = $title,
                    c.court = $court,
                    c.date = $date,
                    c.source = $source,
                    c.text = $text
                RETURN c
                """
                graph.query(cypher, params=case_props)
                
                # Add Judge nodes and relationships
                if 'judges' in doc.metadata:
                    judges = doc.metadata['judges']
                    if isinstance(judges, str):
                        judges = [j.strip() for j in judges.split(',')]
                    elif isinstance(judges, list):
                        judges = [j.strip() for j in judges if j]
                    
                    for judge in judges:
                        if judge and len(judge) > 2:
                            try:
                                # Create judge node
                                judge_cypher = "MERGE (j:Judge {name: $name}) RETURN j"
                                graph.query(judge_cypher, params={'name': judge})
                                
                                # Create relationship
                                rel_cypher = """
                                MATCH (j:Judge {name: $name})
                                MATCH (c:Case {id: $case_id})
                                MERGE (j)-[:JUDGED]->(c)
                                """
                                graph.query(rel_cypher, params={'name': judge, 'case_id': case_props['id']})
                            except Exception as e:
                                continue  # Skip problematic judges
                
                # Add Court node and relationship
                if 'court' in doc.metadata and doc.metadata['court']:
                    try:
                        court_cypher = "MERGE (court:Court {name: $name}) RETURN court"
                        graph.query(court_cypher, params={'name': doc.metadata['court']})
                        
                        rel_cypher = """
                        MATCH (court:Court {name: $name})
                        MATCH (c:Case {id: $case_id})
                        MERGE (c)-[:HEARD_BY]->(court)
                        """
                        graph.query(rel_cypher, params={'name': doc.metadata['court'], 'case_id': case_props['id']})
                    except Exception as e:
                        continue
                
                # Add Statute nodes and relationships
                if 'statutes' in doc.metadata and doc.metadata['statutes']:
                    for statute in doc.metadata['statutes']:
                        if statute and len(statute) > 3:
                            try:
                                statute_cypher = "MERGE (s:Statute {name: $name}) RETURN s"
                                graph.query(statute_cypher, params={'name': statute})
                                
                                rel_cypher = """
                                MATCH (s:Statute {name: $name})
                                MATCH (c:Case {id: $case_id})
                                MERGE (c)-[:REFERENCES]->(s)
                                """
                                graph.query(rel_cypher, params={'name': statute, 'case_id': case_props['id']})
                            except Exception as e:
                                continue
                
                processed += 1
                
                # Progress update
                if processed % 1000 == 0:
                    print(f"Processed {processed}/{total_docs} documents ({processed/total_docs*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing document {batch_start + i}: {e}")
                continue
    
    print(f"✅ Knowledge graph created with {processed} processed documents")
    return True

def get_graph_statistics(graph):
    """Get comprehensive statistics about the knowledge graph"""
    print("📊 Gathering graph statistics...")
    
    stats = {}
    
    # Count nodes by type
    node_types = ['Case', 'Judge', 'Court', 'Statute']
    for node_type in node_types:
        try:
            query = f"MATCH (n:{node_type}) RETURN count(n) as count"
            result = graph.query(query)
            stats[node_type] = result[0].get('count', 0) if result else 0
        except Exception as e:
            stats[node_type] = 0
    
    # Count relationships by type
    try:
        query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC"
        result = graph.query(query)
        stats['relationships'] = {row['rel_type']: row['count'] for row in result} if result else {}
    except Exception as e:
        stats['relationships'] = {}
    
    # Get top courts
    try:
        query = """
        MATCH (court:Court)<-[:HEARD_BY]-(c:Case)
        RETURN court.name as court_name, count(c) as case_count
        ORDER BY case_count DESC
        LIMIT 10
        """
        result = graph.query(query)
        stats['top_courts'] = [(row['court_name'], row['case_count']) for row in result] if result else []
    except Exception as e:
        stats['top_courts'] = []
    
    # Get top judges
    try:
        query = """
        MATCH (judge:Judge)-[:JUDGED]->(c:Case)
        RETURN judge.name as judge_name, count(c) as case_count
        ORDER BY case_count DESC
        LIMIT 10
        """
        result = graph.query(query)
        stats['top_judges'] = [(row['judge_name'], row['case_count']) for row in result] if result else []
    except Exception as e:
        stats['top_judges'] = []
    
    return stats

def create_visualization(stats, output_dir="docs/graphs"):
    """Create comprehensive visualization of the knowledge graph"""
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization dependencies not available")
        return None
    
    print("🎨 Creating knowledge graph visualization...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Node Distribution', 'Relationship Distribution', 
                       'Top Courts by Case Count', 'Top Judges by Case Count'),
        specs=[[{"type": "pie"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Node distribution pie chart
    node_labels = list(stats.keys())
    node_values = [stats[label] for label in node_labels if label != 'relationships' and label != 'top_courts' and label != 'top_judges']
    node_labels = [label for label in node_labels if label != 'relationships' and label != 'top_courts' and label != 'top_judges']
    
    fig.add_trace(
        go.Pie(labels=node_labels, values=node_values, name="Nodes"),
        row=1, col=1
    )
    
    # 2. Relationship distribution pie chart
    if stats['relationships']:
        rel_labels = list(stats['relationships'].keys())
        rel_values = list(stats['relationships'].values())
        
        fig.add_trace(
            go.Pie(labels=rel_labels, values=rel_values, name="Relationships"),
            row=1, col=2
        )
    
    # 3. Top courts bar chart
    if stats['top_courts']:
        court_names = [court[0] for court in stats['top_courts'][:10]]
        court_counts = [court[1] for court in stats['top_courts'][:10]]
        
        fig.add_trace(
            go.Bar(x=court_names, y=court_counts, name="Courts"),
            row=2, col=1
        )
    
    # 4. Top judges bar chart
    if stats['top_judges']:
        judge_names = [judge[0] for judge in stats['top_judges'][:10]]
        judge_counts = [judge[1] for judge in stats['top_judges'][:10]]
        
        fig.add_trace(
            go.Bar(x=judge_names, y=judge_counts, name="Judges"),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Legal Knowledge Graph - Complete Dataset Analysis",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    # Save as HTML
    html_path = os.path.join(output_dir, "complete_knowledge_graph_analysis.html")
    fig.write_html(html_path)
    print(f"✅ HTML visualization saved: {html_path}")
    
    # Save as PNG
    png_path = os.path.join(output_dir, "complete_knowledge_graph_analysis.png")
    fig.write_image(png_path, width=1200, height=800, scale=2)
    print(f"✅ PNG visualization saved: {png_path}")
    
    return png_path

def create_network_visualization(graph, output_dir="docs/graphs", max_nodes=500):
    """Create a network visualization of the knowledge graph"""
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization dependencies not available")
        return None
    
    print("🕸️ Creating network visualization...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Get sample of nodes and relationships for visualization
    print(f"Loading up to {max_nodes} nodes for network visualization...")
    
    # Get cases with their relationships
    query = f"""
    MATCH (c:Case)
    OPTIONAL MATCH (c)-[:HEARD_BY]->(court:Court)
    OPTIONAL MATCH (judge:Judge)-[:JUDGED]->(c)
    OPTIONAL MATCH (c)-[:REFERENCES]->(statute:Statute)
    RETURN c.id as case_id, c.title as case_title, c.court as court_name,
           collect(DISTINCT judge.name) as judges,
           collect(DISTINCT statute.name) as statutes
    LIMIT {max_nodes}
    """
    
    result = graph.query(query)
    
    if not result:
        print("❌ No data found for network visualization")
        return None
    
    # Add nodes and edges
    for row in result:
        case_id = row['case_id']
        case_title = row['case_title'][:50] + "..." if len(row['case_title']) > 50 else row['case_title']
        
        # Add case node
        G.add_node(case_id, label=case_title, type='Case', size=10)
        
        # Add court relationship
        if row['court_name']:
            court_name = row['court_name']
            G.add_node(court_name, label=court_name, type='Court', size=8)
            G.add_edge(case_id, court_name, relationship='HEARD_BY')
        
        # Add judge relationships
        for judge in row['judges']:
            if judge:
                G.add_node(judge, label=judge, type='Judge', size=6)
                G.add_edge(judge, case_id, relationship='JUDGED')
        
        # Add statute relationships
        for statute in row['statutes']:
            if statute:
                G.add_node(statute, label=statute[:30] + "..." if len(statute) > 30 else statute, 
                          type='Statute', size=4)
                G.add_edge(case_id, statute, relationship='REFERENCES')
    
    # Create plotly network visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare data for plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # Create node traces by type
    node_traces = {}
    colors = {'Case': 'red', 'Court': 'blue', 'Judge': 'green', 'Statute': 'orange'}
    
    for node_type in ['Case', 'Court', 'Judge', 'Statute']:
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            if G.nodes[node]['type'] == node_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node]['label'])
                node_info.append(f"Type: {node_type}<br>Name: {G.nodes[node]['label']}")
        
        if node_x:  # Only create trace if there are nodes of this type
            node_traces[node_type] = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=node_info,
                marker=dict(size=G.nodes[node]['size'], color=colors[node_type]),
                name=node_type
            )
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + list(node_traces.values()),
                   layout=go.Layout(
                       title='Legal Knowledge Graph Network',
                       titlefont_size=16,
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Network visualization of legal cases, courts, judges, and statutes",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="black", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    # Save network visualization
    network_html_path = os.path.join(output_dir, "complete_knowledge_graph_network.html")
    fig.write_html(network_html_path)
    print(f"✅ Network HTML saved: {network_html_path}")
    
    network_png_path = os.path.join(output_dir, "complete_knowledge_graph_network.png")
    fig.write_image(network_png_path, width=1200, height=800, scale=2)
    print(f"✅ Network PNG saved: {network_png_path}")
    
    return network_png_path

def save_graph_summary(stats, output_dir="docs/graphs"):
    """Save a text summary of the knowledge graph"""
    print("📝 Saving graph summary...")
    
    summary_path = os.path.join(output_dir, "complete_knowledge_graph_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LEGAL KNOWLEDGE GRAPH - COMPLETE DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("NODE STATISTICS:\n")
        f.write("-" * 20 + "\n")
        total_nodes = 0
        for node_type, count in stats.items():
            if node_type not in ['relationships', 'top_courts', 'top_judges']:
                f.write(f"{node_type}: {count:,}\n")
                total_nodes += count
        
        f.write(f"\nTotal Nodes: {total_nodes:,}\n\n")
        
        f.write("RELATIONSHIP STATISTICS:\n")
        f.write("-" * 25 + "\n")
        total_relationships = 0
        if 'relationships' in stats:
            for rel_type, count in stats['relationships'].items():
                f.write(f"{rel_type}: {count:,}\n")
                total_relationships += count
        
        f.write(f"\nTotal Relationships: {total_relationships:,}\n\n")
        
        f.write("TOP COURTS BY CASE COUNT:\n")
        f.write("-" * 30 + "\n")
        if 'top_courts' in stats:
            for i, (court, count) in enumerate(stats['top_courts'][:10], 1):
                f.write(f"{i:2d}. {court}: {count:,} cases\n")
        
        f.write("\nTOP JUDGES BY CASE COUNT:\n")
        f.write("-" * 30 + "\n")
        if 'top_judges' in stats:
            for i, (judge, count) in enumerate(stats['top_judges'][:10], 1):
                f.write(f"{i:2d}. {judge}: {count:,} cases\n")
        
        f.write(f"\nDATASET COVERAGE:\n")
        f.write("-" * 20 + "\n")
        f.write("Binary Classification Dataset: 43,009 cases\n")
        f.write("Ternary Classification Dataset: 52,900 cases\n")
        f.write("Total CSV Records: 95,909 cases\n")
        f.write("Additional JSON Cases: Variable\n")
    
    print(f"✅ Summary saved: {summary_path}")
    return summary_path

def main():
    """Main function to create complete knowledge graph and visualizations"""
    print("🚀 Starting Complete Knowledge Graph Creation")
    print("=" * 60)
    
    # Connect to Neo4j
    graph = connect_to_neo4j()
    if not graph:
        return
    
    # Initialize embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-001",
            task_type="retrieval_document",
            title="Legal case document"
        )
        print("✅ Google Gemini embeddings initialized")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return
    
    # Load complete dataset
    docs = load_complete_dataset()
    if not docs:
        print("❌ No documents loaded")
        return
    
    # Create knowledge graph
    success = create_comprehensive_knowledge_graph(graph, docs, embeddings)
    if not success:
        print("❌ Failed to create knowledge graph")
        return
    
    # Get statistics
    stats = get_graph_statistics(graph)
    
    # Print statistics
    print("\n📊 KNOWLEDGE GRAPH STATISTICS")
    print("=" * 40)
    for node_type, count in stats.items():
        if node_type not in ['relationships', 'top_courts', 'top_judges']:
            print(f"{node_type}: {count:,}")
    
    if 'relationships' in stats:
        print("\nRelationships:")
        for rel_type, count in stats['relationships'].items():
            print(f"  {rel_type}: {count:,}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Statistical visualization
    stats_image = create_visualization(stats)
    
    # Network visualization
    network_image = create_network_visualization(graph)
    
    # Save summary
    summary_file = save_graph_summary(stats)
    
    print("\n✅ COMPLETE KNOWLEDGE GRAPH CREATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Statistics visualization: {stats_image}")
    print(f"🕸️ Network visualization: {network_image}")
    print(f"📝 Summary report: {summary_file}")
    print(f"📈 Total nodes processed: {sum([stats[k] for k in stats.keys() if k not in ['relationships', 'top_courts', 'top_judges']]):,}")
    
    if 'relationships' in stats:
        print(f"🔗 Total relationships: {sum(stats['relationships'].values()):,}")

if __name__ == "__main__":
    main()
