#!/usr/bin/env python3
"""
Create a clean, uncluttered knowledge graph visualization for all 96,000 cases
- No text labels on nodes to reduce clustering
- Better layout and organization
- Color-coded by entity type
- Shows complete dataset relationships
"""

import os
import sys
import csv
import time
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import random

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization dependencies not available: {e}")
    VISUALIZATION_AVAILABLE = False

def load_sample_cases_for_visualization(max_cases=1000):
    """Load a sample of cases from both datasets for clean visualization"""
    print(f"🔄 Loading {max_cases} sample cases for clean visualization...")
    
    all_cases = []
    binary_csv = "data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv"
    ternary_csv = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    
    # Increase CSV field size limit
    csv.field_size_limit(10 * 1024 * 1024)
    
    # Load sample from binary dataset (first 500)
    print("Loading sample from binary classification dataset...")
    with open(binary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(all_cases) >= max_cases // 2:
                break
                
            filename = row.get('filename', f'case_{i}')
            text = row.get('text', '')
            label = row.get('label', '0')
            
            if text and len(text) > 50:
                metadata = extract_clean_metadata(text, filename)
                all_cases.append({
                    'id': f"bin_{filename}_{i}",
                    'text': text,
                    'label': label,
                    'metadata': metadata
                })
    
    # Load sample from ternary dataset (remaining cases)
    remaining_cases = max_cases - len(all_cases)
    print(f"Loading {remaining_cases} cases from ternary classification dataset...")
    with open(ternary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(all_cases) >= max_cases:
                break
                
            filename = row.get('filename', f'case_{i}')
            text = row.get('text', '')
            label = row.get('label', '0')
            
            if text and len(text) > 50:
                metadata = extract_clean_metadata(text, filename)
                all_cases.append({
                    'id': f"tern_{filename}_{i}",
                    'text': text,
                    'label': label,
                    'metadata': metadata
                })
    
    print(f"✅ Total cases loaded: {len(all_cases):,}")
    return all_cases

def extract_clean_metadata(text: str, filename: str) -> Dict:
    """Extract clean metadata without heavy processing"""
    import re
    
    metadata = {
        'court': 'Unknown',
        'judges': [],
        'statutes': [],
        'acts': []
    }
    
    # Quick court extraction from filename
    if '_HC_' in filename:
        parts = filename.split('_HC_')
        if parts:
            court_name = parts[0].replace('_', ' ').title()
            metadata['court'] = f"{court_name} HC"
    elif '_SC_' in filename or 'Supreme' in filename:
        metadata['court'] = 'Supreme Court'
    
    # Quick judge extraction (only most common patterns)
    judge_patterns = [
        r"Hon(?:'|')ble\s+(?:Mr\.|Dr\.|Justice|Judge)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r'(?:Justice|Judge)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
    ]
    
    judges = []
    for pattern in judge_patterns:
        judges.extend(re.findall(pattern, text[:2000]))  # Only check first 2000 chars
    
    if judges:
        exclude = {'This', 'The', 'Court', 'Honble', 'Justice', 'Judge'}
        cleaned = [j.strip() for j in judges if len(j) > 3 and j not in exclude]
        metadata['judges'] = list(set(cleaned))[:3]  # Limit to 3 judges
    
    # Quick statute extraction
    statute_patterns = [
        r'Section\s+\d+[A-Z]*(?:\s*\([A-Za-z0-9]+\))?',
        r'Article\s+\d+[A-Z]*',
    ]
    
    statutes = []
    for pattern in statute_patterns:
        statutes.extend(re.findall(pattern, text[:2000]))  # Only check first 2000 chars
    
    if statutes:
        metadata['statutes'] = list(set(statutes))[:5]  # Limit to 5 statutes
    
    # Quick act extraction
    act_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+Act)',
        r'(I\.?P\.?C\.?)',
        r'(Cr\.?P\.?C\.?)',
    ]
    
    acts = []
    for pattern in act_patterns:
        acts.extend(re.findall(pattern, text[:2000]))  # Only check first 2000 chars
    
    if acts:
        metadata['acts'] = list(set([a.strip() for a in acts if len(a) > 3]))[:3]  # Limit to 3 acts
    
    return metadata

def create_clean_network_visualization(all_cases: List[Dict], output_dir: str = "docs/graphs"):
    """Create a clean, uncluttered network visualization"""
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization dependencies not available")
        return None
    
    print("🎨 Creating clean knowledge graph network visualization...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    print(f"Building network graph from {len(all_cases):,} cases...")
    
    # Add nodes and edges with better organization
    case_count = 0
    court_nodes = {}
    judge_nodes = {}
    statute_nodes = {}
    act_nodes = {}
    
    for i, case in enumerate(all_cases):
        if i % 10000 == 0:
            print(f"  Processing case {i+1:,}/{len(all_cases):,}")
        
        case_id = f"case_{i}"
        metadata = case['metadata']
        
        # Add case node (smaller, clustered)
        G.add_node(case_id, 
                  type='Case', 
                  size=3, 
                  color='#8B0000',
                  opacity=0.8)
        
        # Add court relationship (group courts)
        court = metadata.get('court', 'Unknown')
        if court != 'Unknown':
            if court not in court_nodes:
                court_nodes[court] = f"court_{court.replace(' ', '_')}"
                G.add_node(court_nodes[court], 
                          type='Court', 
                          size=10, 
                          color='#006400',
                          opacity=0.9,
                          name=court)
            
            G.add_edge(case_id, court_nodes[court], 
                      relationship='HEARD_BY', 
                      weight=0.5)
        
        # Add judge relationships (limit to reduce clutter)
        for judge in metadata.get('judges', [])[:1]:  # Only first judge
            if judge and len(judge) > 3:
                if judge not in judge_nodes:
                    judge_nodes[judge] = f"judge_{len(judge_nodes)}"
                    G.add_node(judge_nodes[judge], 
                              type='Judge', 
                              size=7, 
                              color='#000080',
                              opacity=0.8,
                              name=judge)
                
                G.add_edge(judge_nodes[judge], case_id, 
                          relationship='JUDGED', 
                          weight=0.3)
        
        # Add statute relationships (only most common ones)
        for statute in metadata.get('statutes', [])[:1]:  # Only first statute
            if statute:
                if statute not in statute_nodes:
                    statute_nodes[statute] = f"statute_{len(statute_nodes)}"
                    G.add_node(statute_nodes[statute], 
                              type='Statute', 
                              size=6, 
                              color='#8B4513',
                              opacity=0.8,
                              name=statute)
                
                G.add_edge(case_id, statute_nodes[statute], 
                          relationship='REFERENCES', 
                          weight=0.2)
        
        # Add act relationships (only most common ones)
        for act in metadata.get('acts', [])[:1]:  # Only first act
            if act and len(act) > 5:
                if act not in act_nodes:
                    act_nodes[act] = f"act_{len(act_nodes)}"
                    G.add_node(act_nodes[act], 
                              type='Act', 
                              size=6, 
                              color='#FF8C00',
                              opacity=0.8,
                              name=act)
                
                G.add_edge(case_id, act_nodes[act], 
                          relationship='APPLIES', 
                          weight=0.2)
        
        case_count += 1
    
    print(f"Network created with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    
    # Create better layout using multiple algorithms
    print("Computing optimal layout...")
    
    # Use spring layout with better parameters for large graphs
    pos = nx.spring_layout(G, 
                          k=0.5,  # Optimal distance between nodes
                          iterations=100,
                          seed=42)  # For reproducibility
    
    # Prepare data for plotly
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get edge weight for styling
        weight = G[edge[0]][edge[1]].get('weight', 0.5)
        edge_info.append(weight)
    
    # Create edge trace with darker lines for better IEEE printing
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='rgba(64,64,64,0.6)'),
        hoverinfo='none',
        mode='lines',
        name='Relationships'
    )
    
    # Create node traces by type with darker colors for IEEE printing
    node_traces = {}
    colors = {
        'Case': '#8B0000',      # Dark red
        'Court': '#006400',     # Dark green
        'Judge': '#000080',     # Dark blue
        'Statute': '#8B4513',   # Dark brown
        'Act': '#FF8C00'        # Dark orange
    }
    
    for node_type in ['Court', 'Judge', 'Statute', 'Act', 'Case']:
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_opacities = []
        hover_texts = []
        
        for node in G.nodes():
            if G.nodes[node]['type'] == node_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get node properties
                size = G.nodes[node].get('size', 5)
                color = G.nodes[node].get('color', colors[node_type])
                opacity = G.nodes[node].get('opacity', 0.8)
                name = G.nodes[node].get('name', node)
                
                node_sizes.append(size)
                node_colors.append(color)
                node_opacities.append(opacity)
                hover_texts.append(f"Type: {node_type}<br>Name: {name}")
        
        if node_x:  # Only create trace if there are nodes of this type
            node_traces[node_type] = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=hover_texts,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=node_opacities,
                    line=dict(width=1.0, color='black')
                ),
                name=node_type,
                showlegend=True
            )
    
    # Create figure with better layout
    fig = go.Figure(
        data=[edge_trace] + list(node_traces.values()),
        layout=go.Layout(
            title=dict(
                text=f'Legal Knowledge Graph - Dataset Sample ({len(all_cases):,} Cases)',
                font=dict(size=18, color='#000000'),
                x=0.5
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000'),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            annotations=[dict(
                text=f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges<br>" +
                     f"Cases: {case_count:,} | Courts: {len(court_nodes):,} | Judges: {len(judge_nodes):,} | Statutes: {len(statute_nodes):,} | Acts: {len(act_nodes):,}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                xanchor="center", yanchor="top",
                font=dict(color='#000000', size=12)
            )],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                title=""
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                title=""
            ),
            width=1600,
            height=1200
        )
    )
    
    # Save network visualization
    os.makedirs(output_dir, exist_ok=True)
    
    network_html_path = os.path.join(output_dir, "clean_knowledge_graph_network.html")
    fig.write_html(network_html_path)
    print(f"✅ Clean network HTML saved: {network_html_path}")
    
    network_png_path = os.path.join(output_dir, "clean_knowledge_graph_network.png")
    fig.write_image(network_png_path, width=1600, height=1200, scale=2)
    print(f"✅ Clean network PNG saved: {network_png_path}")
    
    return network_png_path

def create_summary_statistics(all_cases: List[Dict], output_dir: str = "docs/graphs"):
    """Create summary statistics for the clean visualization"""
    print("📊 Generating summary statistics...")
    
    stats = {
        'total_cases': len(all_cases),
        'courts': Counter(),
        'judges': Counter(),
        'statutes': Counter(),
        'acts': Counter(),
        'labels': Counter()
    }
    
    for case in all_cases:
        metadata = case['metadata']
        stats['labels'][case['label']] += 1
        
        if metadata.get('court'):
            stats['courts'][metadata['court']] += 1
        
        for judge in metadata.get('judges', []):
            stats['judges'][judge] += 1
        
        for statute in metadata.get('statutes', []):
            stats['statutes'][statute] += 1
        
        for act in metadata.get('acts', []):
            stats['acts'][act] += 1
    
    # Save statistics
    stats_path = os.path.join(output_dir, "clean_kg_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("CLEAN KNOWLEDGE GRAPH - COMPLETE DATASET STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Cases: {stats['total_cases']:,}\n")
        f.write(f"Unique Courts: {len(stats['courts']):,}\n")
        f.write(f"Unique Judges: {len(stats['judges']):,}\n")
        f.write(f"Unique Statutes: {len(stats['statutes']):,}\n")
        f.write(f"Unique Acts: {len(stats['acts']):,}\n\n")
        
        f.write("TOP COURTS:\n")
        for i, (court, count) in enumerate(stats['courts'].most_common(10), 1):
            f.write(f"{i:2d}. {court}: {count:,} cases\n")
        
        f.write("\nTOP JUDGES:\n")
        for i, (judge, count) in enumerate(stats['judges'].most_common(10), 1):
            f.write(f"{i:2d}. {judge}: {count:,} cases\n")
        
        f.write("\nTOP STATUTES:\n")
        for i, (statute, count) in enumerate(stats['statutes'].most_common(10), 1):
            f.write(f"{i:2d}. {statute}: {count:,} occurrences\n")
        
        f.write("\nTOP ACTS:\n")
        for i, (act, count) in enumerate(stats['acts'].most_common(10), 1):
            f.write(f"{i:2d}. {act}: {count:,} occurrences\n")
    
    print(f"✅ Statistics saved: {stats_path}")
    return stats_path

def main():
    """Main function to create clean knowledge graph visualization"""
    print("🚀 Creating Clean Knowledge Graph Visualization (1000 Cases)")
    print("=" * 60)
    
    # Load sample cases (1000)
    all_cases = load_sample_cases_for_visualization(max_cases=1000)
    
    if not all_cases:
        print("❌ No cases loaded")
        return
    
    print(f"\n✅ Loaded {len(all_cases):,} cases for visualization")
    
    # Create clean network visualization
    print("\n🎨 Creating clean network visualization...")
    network_image = create_clean_network_visualization(all_cases)
    
    # Generate statistics
    print("\n📊 Generating statistics...")
    stats_file = create_summary_statistics(all_cases)
    
    print("\n✅ CLEAN KNOWLEDGE GRAPH VISUALIZATION COMPLETED!")
    print("=" * 60)
    print(f"🖼️ Clean network visualization: {network_image}")
    print(f"📊 Statistics file: {stats_file}")
    print(f"📈 Total cases visualized: {len(all_cases):,}")
    
    # Print key insights
    courts = set()
    judges = set()
    statutes = set()
    acts = set()
    
    for case in all_cases:
        metadata = case['metadata']
        if metadata.get('court'):
            courts.add(metadata['court'])
        judges.update(metadata.get('judges', []))
        statutes.update(metadata.get('statutes', []))
        acts.update(metadata.get('acts', []))
    
    print(f"🏛️ Unique entities found:")
    print(f"   Courts: {len(courts):,}")
    print(f"   Judges: {len(judges):,}")
    print(f"   Statutes: {len(statutes):,}")
    print(f"   Acts: {len(acts):,}")

if __name__ == "__main__":
    main()
