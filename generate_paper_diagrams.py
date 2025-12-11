#!/usr/bin/env python3
"""
Generate professional research diagrams for LegalNexus paper.

Diagrams:
1. Hyperbolic Hierarchy (Poincaré Disk)
2. Curvature Comparison (Bar Chart)
3. Nash Equilibrium Convergence (Line Plot)
4. Multi-Agent Interaction (Conceptual Graph)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from matplotlib.cm import get_cmap

def setup_plot_style():
    """Set up professional plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

def plot_hyperbolic_hierarchy():
    """
    Visualize legal hierarchy in Poincaré disk.
    SC in center, HC in middle, DC near boundary.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw Poincaré disk boundary
    disk = patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(disk)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Generate synthetic nodes
    np.random.seed(42)
    
    # 1. Supreme Court (Center)
    n_sc = 5
    r_sc = np.random.uniform(0.0, 0.2, n_sc)
    theta_sc = np.random.uniform(0, 2*np.pi, n_sc)
    x_sc = r_sc * np.cos(theta_sc)
    y_sc = r_sc * np.sin(theta_sc)
    
    # 2. High Courts (Middle)
    n_hc = 15
    r_hc = np.random.uniform(0.4, 0.6, n_hc)
    theta_hc = np.random.uniform(0, 2*np.pi, n_hc)
    x_hc = r_hc * np.cos(theta_hc)
    y_hc = r_hc * np.sin(theta_hc)
    
    # 3. District Courts (Edge)
    n_dc = 40
    r_dc = np.random.uniform(0.8, 0.95, n_dc)
    theta_dc = np.random.uniform(0, 2*np.pi, n_dc)
    x_dc = r_dc * np.cos(theta_dc)
    y_dc = r_dc * np.sin(theta_dc)
    
    # Draw edges (geodesics approximated as straight lines for visualization simplicity)
    # Connect HC to SC
    for i in range(n_hc):
        # Connect to nearest SC
        dists = np.sqrt((x_sc - x_hc[i])**2 + (y_sc - y_hc[i])**2)
        nearest = np.argmin(dists)
        ax.plot([x_hc[i], x_sc[nearest]], [y_hc[i], y_sc[nearest]], 
                color='gray', alpha=0.3, linewidth=0.5)
        
    # Connect DC to HC
    for i in range(n_dc):
        # Connect to nearest HC
        dists = np.sqrt((x_hc - x_dc[i])**2 + (y_hc - y_dc[i])**2)
        nearest = np.argmin(dists)
        ax.plot([x_dc[i], x_hc[nearest]], [y_dc[i], y_hc[nearest]], 
                color='gray', alpha=0.2, linewidth=0.5)

    # Plot nodes
    ax.scatter(x_sc, y_sc, c='#d62728', s=150, label='Supreme Court', zorder=3, edgecolors='white')
    ax.scatter(x_hc, y_hc, c='#1f77b4', s=80, label='High Court', zorder=2, edgecolors='white')
    ax.scatter(x_dc, y_dc, c='#2ca02c', s=30, label='District Court', zorder=1, alpha=0.7)
    
    plt.title("Legal Citation Hierarchy in Hyperbolic Space", pad=20)
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('diagram_hyperbolic_hierarchy.png', bbox_inches='tight')
    print("Generated diagram_hyperbolic_hierarchy.png")

def plot_curvature_comparison():
    """Compare δ-hyperbolicity across graph types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    graphs = ['Perfect Tree', 'Legal Network\n(Ours)', 'Barabási-Albert', 'Erdős-Rényi']
    deltas = [0.0, 0.335, 1.523, 2.145]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    
    bars = ax.bar(graphs, deltas, color=colors, alpha=0.8, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'δ = {height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Add threshold line
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.text(3.8, 1.05, 'Hyperbolic Threshold (δ < 1)', color='gray', ha='right')
    
    ax.set_ylabel('Gromov δ-Hyperbolicity (Lower is More Tree-like)')
    ax.set_title('Curvature Analysis: Legal Networks are Highly Hyperbolic')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    ax.annotate('6.4x more hyperbolic\nthan random graphs', 
                xy=(1, 0.335), xytext=(1.5, 1.0),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('diagram_curvature_comparison.png')
    print("Generated diagram_curvature_comparison.png")

def plot_nash_convergence():
    """Visualize convergence of multi-agent payoffs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = [1, 2, 3, 4]
    linker = [0.70, 0.75, 0.80, 0.80]
    interpreter = [1.00, 1.00, 1.00, 1.00]
    conflict = [0.50, 0.75, 1.00, 1.00]
    total = [0.733, 0.833, 0.933, 0.933]
    
    ax.plot(iterations, linker, 'o-', label='Linker (Precision)', color='#1f77b4', linewidth=2)
    ax.plot(iterations, interpreter, 's-', label='Interpreter (Accuracy)', color='#ff7f0e', linewidth=2)
    ax.plot(iterations, conflict, '^-', label='Conflict (Consistency)', color='#2ca02c', linewidth=2)
    ax.plot(iterations, total, 'D--', label='Total System Payoff', color='#d62728', linewidth=3)
    
    ax.set_xlabel('Debate Iteration')
    ax.set_ylabel('Agent Payoff (Utility)')
    ax.set_title('Nash Equilibrium Convergence: Multi-Agent System')
    ax.set_xticks(iterations)
    ax.set_ylim(0.4, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight convergence
    ax.axvline(x=3, color='gray', linestyle=':', alpha=0.5)
    ax.text(3.1, 0.5, 'Equilibrium Reached', color='gray')
    
    plt.tight_layout()
    plt.savefig('diagram_nash_convergence.png')
    print("Generated diagram_nash_convergence.png")

def plot_agent_interaction():
    """Conceptual diagram of agent interaction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    G = nx.DiGraph()
    G.add_edge("Linker Agent\n(Proposer)", "Knowledge Graph", label="Proposes\nCitations")
    G.add_edge("Interpreter Agent\n(Analyst)", "Knowledge Graph", label="Classifies\nEdges")
    G.add_edge("Conflict Agent\n(Critic)", "Knowledge Graph", label="Detects\nCycles")
    G.add_edge("Knowledge Graph", "Linker Agent\n(Proposer)", label="Feedback\n(Precision)")
    G.add_edge("Knowledge Graph", "Interpreter Agent\n(Analyst)", label="Feedback\n(Consistency)")
    G.add_edge("Knowledge Graph", "Conflict Agent\n(Critic)", label="Feedback\n(Logic)")
    
    pos = {
        "Knowledge Graph": (0, 0),
        "Linker Agent\n(Proposer)": (0, 1),
        "Interpreter Agent\n(Analyst)": (-0.866, -0.5),
        "Conflict Agent\n(Critic)": (0.866, -0.5)
    }
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6000, node_color='#e6f2ff', edgecolors='#1f77b4')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                           arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    # Edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Game-Theoretic Multi-Agent Interaction", pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('diagram_agent_interaction.png')
    print("Generated diagram_agent_interaction.png")

if __name__ == "__main__":
    setup_plot_style()
    plot_hyperbolic_hierarchy()
    plot_curvature_comparison()
    plot_nash_convergence()
    plot_agent_interaction()
