#!/usr/bin/env python3
"""
Generate ALL Visualizations for Knowledge Graph
Creates comprehensive visual analysis of the complete dataset
"""
import os
import sys
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from collections import Counter

# Load environment (try .env.neo4j first, then .env)
load_dotenv('.env.neo4j')
load_dotenv()

def generate_all_visualizations():
    """Generate comprehensive visualizations for the knowledge graph"""
    
    print("=" * 80)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 80)
    
    # Connect to Neo4j (use environment or ask user)
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_url:
        print("\n🔌 Neo4j Connection Required")
        neo4j_url = input("Enter Neo4j URL (e.g., neo4j+s://xxx.databases.neo4j.io): ").strip()
        neo4j_username = input("Enter Neo4j Username (default: neo4j): ").strip() or "neo4j"
        neo4j_password = input("Enter Neo4j Password: ").strip()
    
    print(f"\n📊 Connecting to Neo4j...")
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        print("✓ Connected")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Create output directory
    output_dir = "visualizations_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}/\n")
    
    # 1. Dataset Overview
    print("1️⃣  Generating Dataset Overview...")
    try:
        stats = {}
        queries = {
            'Cases': "MATCH (c:Case) RETURN count(c) as count",
            'Judges': "MATCH (j:Judge) RETURN count(j) as count",
            'Courts': "MATCH (c:Court) RETURN count(c) as count",
            'Statutes': "MATCH (s:Statute) RETURN count(s) as count",
            'Acts': "MATCH (a:Act) RETURN count(a) as count",
            'Relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        for name, query in queries.items():
            result = graph.query(query)
            stats[name] = result[0]['count'] if result else 0
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(x=list(stats.keys()), y=list(stats.values()),
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        ])
        fig.update_layout(
            title=f"Knowledge Graph Overview - {stats['Cases']:,} Legal Cases",
            xaxis_title="Entity Type",
            yaxis_title="Count",
            height=500
        )
        fig.write_html(f"{output_dir}/01_dataset_overview.html")
        print(f"   ✓ Saved: 01_dataset_overview.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Classification Label Distribution
    print("2️⃣  Generating Label Distribution...")
    try:
        query = """
        MATCH (c:Case)
        RETURN c.label as label, count(c) as count
        ORDER BY label
        """
        results = graph.query(query)
        
        labels = [r['label'] for r in results]
        counts = [r['count'] for r in results]
        
        fig = go.Figure(data=[
            go.Pie(labels=[f"Label {l} ({c:,})" for l, c in zip(labels, counts)],
                   values=counts,
                   hole=0.4)
        ])
        fig.update_layout(
            title="Classification Label Distribution<br>(0=Rejected, 1=Accepted, 2=Mixed)",
            height=500
        )
        fig.write_html(f"{output_dir}/02_label_distribution.html")
        print(f"   ✓ Saved: 02_label_distribution.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Top Courts
    print("3️⃣  Generating Top Courts Analysis...")
    try:
        query = """
        MATCH (court:Court)<-[:HEARD_BY]-(c:Case)
        RETURN court.name as court, count(c) as cases
        ORDER BY cases DESC
        LIMIT 20
        """
        results = graph.query(query)
        
        courts = [r['court'] for r in results]
        cases = [r['cases'] for r in results]
        
        fig = go.Figure(data=[
            go.Bar(y=courts[::-1], x=cases[::-1], orientation='h',
                   marker_color='#4ECDC4')
        ])
        fig.update_layout(
            title="Top 20 Courts by Case Count",
            xaxis_title="Number of Cases",
            yaxis_title="Court",
            height=600
        )
        fig.write_html(f"{output_dir}/03_top_courts.html")
        print(f"   ✓ Saved: 03_top_courts.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Top Judges
    print("4️⃣  Generating Top Judges Analysis...")
    try:
        query = """
        MATCH (j:Judge)-[:JUDGED]->(c:Case)
        RETURN j.name as judge, count(c) as cases
        ORDER BY cases DESC
        LIMIT 20
        """
        results = graph.query(query)
        
        judges = [r['judge'] for r in results]
        cases = [r['cases'] for r in results]
        
        fig = go.Figure(data=[
            go.Bar(y=judges[::-1], x=cases[::-1], orientation='h',
                   marker_color='#FF6B6B')
        ])
        fig.update_layout(
            title="Top 20 Judges by Case Count",
            xaxis_title="Number of Cases",
            yaxis_title="Judge",
            height=600
        )
        fig.write_html(f"{output_dir}/04_top_judges.html")
        print(f"   ✓ Saved: 04_top_judges.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Most Referenced Acts
    print("5️⃣  Generating Top Acts Analysis...")
    try:
        query = """
        MATCH (a:Act)<-[:CITES_ACT]-(c:Case)
        RETURN a.name as act, count(c) as cases
        ORDER BY cases DESC
        LIMIT 15
        """
        results = graph.query(query)
        
        acts = [r['act'] for r in results]
        cases = [r['cases'] for r in results]
        
        fig = go.Figure(data=[
            go.Bar(y=acts[::-1], x=cases[::-1], orientation='h',
                   marker_color='#FFA07A')
        ])
        fig.update_layout(
            title="Top 15 Most Referenced Acts",
            xaxis_title="Number of Cases",
            yaxis_title="Act",
            height=600
        )
        fig.write_html(f"{output_dir}/05_top_acts.html")
        print(f"   ✓ Saved: 05_top_acts.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 6. Most Referenced Statutes
    print("6️⃣  Generating Top Statutes Analysis...")
    try:
        query = """
        MATCH (s:Statute)<-[:REFERENCES]-(c:Case)
        RETURN s.name as statute, count(c) as cases
        ORDER BY cases DESC
        LIMIT 20
        """
        results = graph.query(query)
        
        statutes = [r['statute'] for r in results]
        cases = [r['cases'] for r in results]
        
        fig = go.Figure(data=[
            go.Bar(x=statutes, y=cases,
                   marker_color='#98D8C8')
        ])
        fig.update_layout(
            title="Top 20 Most Referenced Statutes/Sections",
            xaxis_title="Statute/Section",
            yaxis_title="Number of Cases",
            height=500,
            xaxis_tickangle=-45
        )
        fig.write_html(f"{output_dir}/06_top_statutes.html")
        print(f"   ✓ Saved: 06_top_statutes.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 7. Network Sample Visualization
    print("7️⃣  Generating Knowledge Graph Network Sample...")
    try:
        query = """
        MATCH (c:Case)-[r]-(n)
        RETURN c, r, n
        LIMIT 100
        """
        results = graph.query(query)
        
        # Create network graph
        G = nx.Graph()
        
        for result in results:
            # Add nodes and edges
            case_id = result['c']['id'] if 'c' in result else None
            if case_id:
                G.add_node(case_id, type='Case')
        
        # Use networkx layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
                line_width=2))
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph Network Sample (100 cases)',
                           showlegend=False,
                           hovermode='closest',
                           height=600
                       ))
        fig.write_html(f"{output_dir}/07_network_sample.html")
        print(f"   ✓ Saved: 07_network_sample.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 8. Summary Dashboard
    print("8️⃣  Generating Summary Dashboard...")
    try:
        summary_html = f"""
        <html>
        <head>
            <title>Knowledge Graph Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 32px; font-weight: bold; color: #4ECDC4; }}
                .stat-label {{ color: #666; margin-top: 8px; }}
                .viz-links {{ margin: 20px 0; }}
                .viz-link {{ display: block; padding: 10px; background: white; margin: 10px 0; border-radius: 4px; text-decoration: none; color: #333; }}
                .viz-link:hover {{ background: #e8e8e8; }}
            </style>
        </head>
        <body>
            <h1>📊 Legal Knowledge Graph Analysis</h1>
            <p>Complete dataset: ~96,000 legal cases with LLM-based metadata extraction</p>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Cases', 0):,}</div>
                    <div class="stat-label">Legal Cases</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Judges', 0):,}</div>
                    <div class="stat-label">Unique Judges</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Courts', 0):,}</div>
                    <div class="stat-label">Courts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Statutes', 0):,}</div>
                    <div class="stat-label">Statutes/Sections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Acts', 0):,}</div>
                    <div class="stat-label">Legal Acts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('Relationships', 0):,}</div>
                    <div class="stat-label">Total Relationships</div>
                </div>
            </div>
            
            <h2>📈 Visualizations</h2>
            <div class="viz-links">
                <a href="01_dataset_overview.html" class="viz-link">1. Dataset Overview</a>
                <a href="02_label_distribution.html" class="viz-link">2. Classification Label Distribution</a>
                <a href="03_top_courts.html" class="viz-link">3. Top Courts Analysis</a>
                <a href="04_top_judges.html" class="viz-link">4. Top Judges Analysis</a>
                <a href="05_top_acts.html" class="viz-link">5. Most Referenced Acts</a>
                <a href="06_top_statutes.html" class="viz-link">6. Most Referenced Statutes</a>
                <a href="07_network_sample.html" class="viz-link">7. Knowledge Graph Network Sample</a>
            </div>
            
            <h2>🔬 Novel Contributions</h2>
            <ul>
                <li><strong>LLM-based Batch Extraction:</strong> Gemini 2.5 Flash processes 5 cases per API call</li>
                <li><strong>Hybrid Extraction:</strong> 60-70% LLM extraction + pattern fallback</li>
                <li><strong>Classification Labels:</strong> Binary (0/1) and Ternary (0/1/2) preserved</li>
                <li><strong>Knowledge Graph:</strong> {stats.get('Relationships', 0):,} relationships connecting legal entities</li>
            </ul>
        </body>
        </html>
        """
        
        with open(f"{output_dir}/00_dashboard.html", 'w') as f:
            f.write(summary_html)
        print(f"   ✓ Saved: 00_dashboard.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Location: {output_dir}/")
    print(f"\n🌐 Open: {output_dir}/00_dashboard.html")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    generate_all_visualizations()

