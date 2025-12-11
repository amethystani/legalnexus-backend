#!/usr/bin/env python3
"""
Create a standalone visualization of the complete dataset knowledge graph
This script analyzes the CSV data and creates comprehensive visualizations
without requiring Neo4j connection.
"""

import os
import sys
import json
import csv
import time
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter, defaultdict

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization dependencies not available: {e}")
    VISUALIZATION_AVAILABLE = False

def load_csv_data_analysis(csv_file_path: str, max_rows: int = None) -> Tuple[List[Dict], Dict]:
    """
    Load and analyze CSV data for visualization
    Returns: (cases_data, statistics)
    """
    print(f"📊 Analyzing data from: {csv_file_path}")
    
    cases_data = []
    stats = {
        'total_cases': 0,
        'courts': Counter(),
        'judges': Counter(),
        'statutes': Counter(),
        'acts': Counter(),
        'text_lengths': [],
        'labels': Counter()
    }
    
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        return cases_data, stats
    
    try:
        # Increase CSV field size limit to handle large legal case texts
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit
        
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                
                # Extract basic data
                filename = row.get('filename', f'case_{i}')
                text = row.get('text', '')
                label = row.get('label', '0')
                
                if not text or len(text) < 50:
                    continue
                
                # Extract metadata using patterns
                metadata = extract_metadata_patterns(text, filename)
                
                case_data = {
                    'id': f"{filename}_{i}",
                    'filename': filename,
                    'text': text,
                    'label': label,
                    'metadata': metadata
                }
                
                cases_data.append(case_data)
                
                # Update statistics
                stats['total_cases'] += 1
                stats['text_lengths'].append(len(text))
                stats['labels'][label] += 1
                
                # Count courts
                if metadata.get('court'):
                    stats['courts'][metadata['court']] += 1
                
                # Count judges
                for judge in metadata.get('judges', []):
                    stats['judges'][judge] += 1
                
                # Count statutes
                for statute in metadata.get('statutes', []):
                    stats['statutes'][statute] += 1
                
                # Count acts
                for act in metadata.get('acts', []):
                    stats['acts'][act] += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} cases...")
        
        print(f"✅ Loaded {len(cases_data)} cases from {os.path.basename(csv_file_path)}")
        return cases_data, stats
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return cases_data, stats

def extract_metadata_patterns(text: str, filename: str) -> Dict:
    """Extract metadata using pattern matching"""
    import re
    
    metadata = {
        'court': 'Unknown Court',
        'judges': [],
        'statutes': [],
        'acts': [],
        'date': 'Unknown Date'
    }
    
    # Extract court from filename or text
    if '_HC_' in filename:
        parts = filename.split('_HC_')
        if parts:
            court_name = parts[0].replace('_', ' ')
            metadata['court'] = f"{court_name} High Court"
    elif '_SC_' in filename or 'Supreme' in filename:
        metadata['court'] = 'Supreme Court of India'
    
    # Judge patterns
    judge_patterns = [
        r"Hon(?:'|')ble\s+(?:Mr\.|Dr\.|Justice|Judge)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r'(?:Justice|Judge)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'Coram\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    ]
    
    judges = []
    for pattern in judge_patterns:
        judges.extend(re.findall(pattern, text))
    
    if judges:
        exclude = {'This', 'The', 'Court', 'Honble', 'Justice', 'Judge'}
        cleaned = [j.strip() for j in judges if len(j) > 3 and j not in exclude]
        metadata['judges'] = list(set(cleaned))[:5]
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metadata['date'] = matches[0]
            break
    
    # Statutes
    statute_patterns = [
        r'Section\s+\d+[A-Z]*(?:\s*\([A-Za-z0-9]+\))?',
        r'Article\s+\d+[A-Z]*',
    ]
    
    statutes = []
    for pattern in statute_patterns:
        statutes.extend(re.findall(pattern, text))
    
    if statutes:
        metadata['statutes'] = list(set(statutes))[:15]
    
    # Acts
    act_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s+Act\s*(?:\d{4})?)',
        r'(I\.?P\.?C\.?)',
        r'(Cr\.?P\.?C\.?)',
    ]
    
    acts = []
    for pattern in act_patterns:
        acts.extend(re.findall(pattern, text))
    
    if acts:
        metadata['acts'] = list(set([a.strip() for a in acts if len(a) > 3]))[:10]
    
    return metadata

def create_comprehensive_visualization(binary_stats: Dict, ternary_stats: Dict, output_dir: str = "docs/graphs"):
    """Create comprehensive visualization of the complete dataset"""
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization dependencies not available")
        return None
    
    print("🎨 Creating comprehensive knowledge graph visualization...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine statistics
    total_cases = binary_stats['total_cases'] + ternary_stats['total_cases']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Dataset Distribution', 'Court Distribution (Top 10)',
            'Judge Distribution (Top 10)', 'Statute Distribution (Top 10)',
            'Text Length Distribution', 'Label Distribution'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "pie"}]
        ]
    )
    
    # 1. Dataset distribution pie chart
    fig.add_trace(
        go.Pie(
            labels=['Binary Classification', 'Ternary Classification'],
            values=[binary_stats['total_cases'], ternary_stats['total_cases']],
            name="Dataset Split"
        ),
        row=1, col=1
    )
    
    # 2. Court distribution (combine both datasets)
    all_courts = Counter(binary_stats['courts']) + Counter(ternary_stats['courts'])
    top_courts = all_courts.most_common(10)
    
    if top_courts:
        court_names = [court[0] for court in top_courts]
        court_counts = [court[1] for court in top_courts]
        
        fig.add_trace(
            go.Bar(x=court_names, y=court_counts, name="Courts"),
            row=1, col=2
        )
    
    # 3. Judge distribution (combine both datasets)
    all_judges = Counter(binary_stats['judges']) + Counter(ternary_stats['judges'])
    top_judges = all_judges.most_common(10)
    
    if top_judges:
        judge_names = [judge[0] for judge in top_judges]
        judge_counts = [judge[1] for judge in top_judges]
        
        fig.add_trace(
            go.Bar(x=judge_names, y=judge_counts, name="Judges"),
            row=2, col=1
        )
    
    # 4. Statute distribution (combine both datasets)
    all_statutes = Counter(binary_stats['statutes']) + Counter(ternary_stats['statutes'])
    top_statutes = all_statutes.most_common(10)
    
    if top_statutes:
        statute_names = [statute[0] for statute in top_statutes]
        statute_counts = [statute[1] for statute in top_statutes]
        
        fig.add_trace(
            go.Bar(x=statute_names, y=statute_counts, name="Statutes"),
            row=2, col=2
        )
    
    # 5. Text length distribution
    all_text_lengths = binary_stats['text_lengths'] + ternary_stats['text_lengths']
    
    fig.add_trace(
        go.Histogram(x=all_text_lengths, nbinsx=50, name="Text Lengths"),
        row=3, col=1
    )
    
    # 6. Label distribution (combine both datasets)
    all_labels = Counter(binary_stats['labels']) + Counter(ternary_stats['labels'])
    
    if all_labels:
        label_names = [f"Label {label}" for label in all_labels.keys()]
        label_counts = list(all_labels.values())
        
        fig.add_trace(
            go.Pie(labels=label_names, values=label_counts, name="Labels"),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Legal Knowledge Graph - Complete Dataset Analysis ({total_cases:,} cases)",
        title_x=0.5,
        height=1200,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Courts", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Judges", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Statutes", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Text Length (characters)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    
    # Save as HTML
    html_path = os.path.join(output_dir, "complete_knowledge_graph_analysis.html")
    fig.write_html(html_path)
    print(f"✅ HTML visualization saved: {html_path}")
    
    # Save as PNG
    png_path = os.path.join(output_dir, "complete_knowledge_graph_analysis.png")
    fig.write_image(png_path, width=1400, height=1200, scale=2)
    print(f"✅ PNG visualization saved: {png_path}")
    
    return png_path

def create_network_visualization_sample(binary_cases: List[Dict], ternary_cases: List[Dict], 
                                      output_dir: str = "docs/graphs", sample_size: int = 1000):
    """Create a network visualization from a sample of cases"""
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization dependencies not available")
        return None
    
    print("🕸️ Creating network visualization from sample data...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Combine and sample cases
    all_cases = binary_cases + ternary_cases
    sample_cases = all_cases[:sample_size]  # Take first sample_size cases
    
    print(f"Creating network from {len(sample_cases)} sample cases...")
    
    # Add nodes and edges
    for case in sample_cases:
        case_id = case['id']
        case_title = case['filename'][:30] + "..." if len(case['filename']) > 30 else case['filename']
        metadata = case['metadata']
        
        # Add case node
        G.add_node(case_id, label=case_title, type='Case', size=10)
        
        # Add court relationship
        if metadata.get('court') and metadata['court'] != 'Unknown Court':
            court_name = metadata['court']
            G.add_node(court_name, label=court_name, type='Court', size=8)
            G.add_edge(case_id, court_name, relationship='HEARD_BY')
        
        # Add judge relationships
        for judge in metadata.get('judges', []):
            if judge:
                G.add_node(judge, label=judge, type='Judge', size=6)
                G.add_edge(judge, case_id, relationship='JUDGED')
        
        # Add statute relationships (limit to avoid overcrowding)
        for statute in metadata.get('statutes', [])[:3]:  # Limit to 3 statutes per case
            if statute:
                statute_label = statute[:20] + "..." if len(statute) > 20 else statute
                G.add_node(statute, label=statute_label, type='Statute', size=4)
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
                       title=dict(text=f'Legal Knowledge Graph Network (Sample of {len(sample_cases)} cases)', font=dict(size=16)),
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

def save_comprehensive_summary(binary_stats: Dict, ternary_stats: Dict, output_dir: str = "docs/graphs"):
    """Save a comprehensive text summary"""
    print("📝 Saving comprehensive summary...")
    
    summary_path = os.path.join(output_dir, "complete_dataset_summary.txt")
    
    total_cases = binary_stats['total_cases'] + ternary_stats['total_cases']
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LEGAL KNOWLEDGE GRAPH - COMPLETE DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Binary Classification Dataset: {binary_stats['total_cases']:,} cases\n")
        f.write(f"Ternary Classification Dataset: {ternary_stats['total_cases']:,} cases\n")
        f.write(f"Total Cases: {total_cases:,}\n\n")
        
        # Combined statistics
        all_courts = Counter(binary_stats['courts']) + Counter(ternary_stats['courts'])
        all_judges = Counter(binary_stats['judges']) + Counter(ternary_stats['judges'])
        all_statutes = Counter(binary_stats['statutes']) + Counter(ternary_stats['statutes'])
        all_acts = Counter(binary_stats['acts']) + Counter(ternary_stats['acts'])
        all_labels = Counter(binary_stats['labels']) + Counter(ternary_stats['labels'])
        
        f.write("ENTITY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Unique Courts: {len(all_courts)}\n")
        f.write(f"Unique Judges: {len(all_judges)}\n")
        f.write(f"Unique Statutes: {len(all_statutes)}\n")
        f.write(f"Unique Acts: {len(all_acts)}\n")
        f.write(f"Classification Labels: {len(all_labels)}\n\n")
        
        f.write("TOP COURTS BY CASE COUNT:\n")
        f.write("-" * 30 + "\n")
        for i, (court, count) in enumerate(all_courts.most_common(15), 1):
            f.write(f"{i:2d}. {court}: {count:,} cases\n")
        
        f.write("\nTOP JUDGES BY CASE COUNT:\n")
        f.write("-" * 30 + "\n")
        for i, (judge, count) in enumerate(all_judges.most_common(15), 1):
            f.write(f"{i:2d}. {judge}: {count:,} cases\n")
        
        f.write("\nTOP STATUTES BY FREQUENCY:\n")
        f.write("-" * 30 + "\n")
        for i, (statute, count) in enumerate(all_statutes.most_common(15), 1):
            f.write(f"{i:2d}. {statute}: {count:,} occurrences\n")
        
        f.write("\nTOP ACTS BY FREQUENCY:\n")
        f.write("-" * 30 + "\n")
        for i, (act, count) in enumerate(all_acts.most_common(10), 1):
            f.write(f"{i:2d}. {act}: {count:,} occurrences\n")
        
        f.write("\nLABEL DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        for label, count in sorted(all_labels.items()):
            percentage = (count / total_cases) * 100
            f.write(f"Label {label}: {count:,} cases ({percentage:.1f}%)\n")
        
        # Text length statistics
        all_text_lengths = binary_stats['text_lengths'] + ternary_stats['text_lengths']
        if all_text_lengths:
            f.write(f"\nTEXT LENGTH STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Average length: {sum(all_text_lengths) / len(all_text_lengths):.0f} characters\n")
            f.write(f"Median length: {sorted(all_text_lengths)[len(all_text_lengths)//2]:.0f} characters\n")
            f.write(f"Min length: {min(all_text_lengths):.0f} characters\n")
            f.write(f"Max length: {max(all_text_lengths):.0f} characters\n")
    
    print(f"✅ Summary saved: {summary_path}")
    return summary_path

def main():
    """Main function to create complete dataset visualization"""
    print("🚀 Starting Complete Dataset Knowledge Graph Visualization")
    print("=" * 70)
    
    # File paths
    binary_csv = "data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv"
    ternary_csv = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    
    # Load and analyze binary dataset
    print("\n📊 Analyzing Binary Classification Dataset...")
    binary_cases, binary_stats = load_csv_data_analysis(binary_csv)
    
    # Load and analyze ternary dataset
    print("\n📊 Analyzing Ternary Classification Dataset...")
    ternary_cases, ternary_stats = load_csv_data_analysis(ternary_csv)
    
    total_cases = binary_stats['total_cases'] + ternary_stats['total_cases']
    print(f"\n✅ Total cases analyzed: {total_cases:,}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Statistical visualization
    stats_image = create_comprehensive_visualization(binary_stats, ternary_stats)
    
    # Network visualization (sample)
    network_image = create_network_visualization_sample(binary_cases, ternary_cases)
    
    # Save summary
    summary_file = save_comprehensive_summary(binary_stats, ternary_stats)
    
    print("\n✅ COMPLETE DATASET VISUALIZATION CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Statistical analysis: {stats_image}")
    print(f"🕸️ Network visualization: {network_image}")
    print(f"📝 Summary report: {summary_file}")
    print(f"📈 Total cases analyzed: {total_cases:,}")
    
    # Print key statistics
    all_courts = Counter(binary_stats['courts']) + Counter(ternary_stats['courts'])
    all_judges = Counter(binary_stats['judges']) + Counter(ternary_stats['judges'])
    
    print(f"🏛️ Unique courts found: {len(all_courts)}")
    print(f"⚖️ Unique judges found: {len(all_judges)}")
    
    if all_courts:
        print(f"🏆 Top court: {all_courts.most_common(1)[0][0]} ({all_courts.most_common(1)[0][1]:,} cases)")
    
    if all_judges:
        print(f"👨‍⚖️ Most active judge: {all_judges.most_common(1)[0][0]} ({all_judges.most_common(1)[0][1]:,} cases)")

if __name__ == "__main__":
    main()
