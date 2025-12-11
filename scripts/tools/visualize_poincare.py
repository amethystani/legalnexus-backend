"""
Visualize Poincaré Disk Embeddings

Creates 2D projection of hyperbolic embeddings to verify hierarchy:
- Supreme Court cases should cluster near center
- Lower court cases should be near edge
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import torch
from sklearn.manifold import TSNE
from hyperbolic_gnn import LegalHyperbolicModel
from geoopt import PoincareBall


def load_model_and_embeddings(model_path='models/hgcn_best.pt'):
    """Load trained model and generate 2D embeddings"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    case_ids = checkpoint['case_ids']
    metadata = checkpoint['metadata']
    
    # Load embeddings
    with open('models/hgcn_embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    # Convert to matrix
    embeddings = np.array([embeddings_dict[cid] for cid in case_ids])
    
    return case_ids, embeddings, metadata


def visualize_poincare_3d(case_ids, embeddings, metadata, output_path='poincare_3d_visualization.png'):
    """
    Plot 3D Poincaré ball with color-coded court levels.
    Uses first 3 dimensions of embeddings directly.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use first 3 dimensions
    embeddings_3d = embeddings[:, :3]
    
    # Normalize to fit in unit ball
    max_norm = np.max(np.linalg.norm(embeddings_3d, axis=1))
    if max_norm > 0:
        embeddings_3d = embeddings_3d / max_norm * 0.95
    
    # Color map for courts
    court_colors = {
        'Supreme Court': '#e74c3c',  # Red
        'High Court': '#3498db',      # Blue
        'Lower Court': '#2ecc71'      # Green
    }
    
    # Plot cases
    for i, case_id in enumerate(case_ids):
        x, y, z = embeddings_3d[i]
        court = metadata[case_id]['court']
        color = court_colors.get(court, '#95a5a6')
        
        ax.scatter(x, y, z, c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Draw unit sphere (boundary of Poincaré ball)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=court_colors['Supreme Court'], 
                  markersize=10, label='Supreme Court'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=court_colors['High Court'], 
                  markersize=10, label='High Court'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=court_colors['Lower Court'], 
                  markersize=10, label='Lower Court')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Poincaré Ball: Legal Citation Hierarchy\n(Center = Supreme Court, Edge = Lower Courts)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 3D visualization to {output_path}")
    
    return fig


def visualize_poincare_disk(case_ids, embeddings_2d, metadata, output_path='poincare_visualization.png'):
    """
    Plot 2D Poincaré disk with color-coded court levels.
    
    Expected Result:
    - Supreme Court (red) near center
    - High Courts (blue) middle region
    - Lower Courts (green) near edge
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw unit circle (boundary of Poincaré disk)
    circle = patches.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    # Color map for courts
    court_colors = {
        'Supreme Court': '#e74c3c',  # Red
        'High Court': '#3498db',      # Blue
        'Lower Court': '#2ecc71'      # Green
    }
    
    # Plot cases
    for i, case_id in enumerate(case_ids):
        x, y = embeddings_2d[i]
        court = metadata[case_id]['court']
        color = court_colors.get(court, '#95a5a6')  # Gray for unknown
        
        ax.scatter(x, y, c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Annotate with case ID (abbreviated)
        ax.annotate(case_id[:15], (x, y), fontsize=8, ha='center', va='bottom')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=court_colors['Supreme Court'], label='Supreme Court'),
        patches.Patch(facecolor=court_colors['High Court'], label='High Court'),
        patches.Patch(facecolor=court_colors['Lower Court'], label='Lower Court')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Draw concentric circles to show hierarchy layers
    for radius in [0.3, 0.6, 0.9]:
        circle_layer = patches.Circle((0, 0), radius, fill=False, 
                                     edgecolor='gray', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.add_patch(circle_layer)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Poincaré Disk: Legal Citation Hierarchy\n(Center = Supreme Court, Edge = Lower Courts)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    return fig


def analyze_hierarchy(case_ids, embeddings, metadata):
    """
    Statistical analysis of learned hierarchy.
    Verify that radius correlates with court level.
    """
    print("\n" + "="*80)
    print("HIERARCHY ANALYSIS")
    print("="*80)
    
    # Calculate radii
    radii = np.linalg.norm(embeddings, axis=1)
    
    # Group by court level
    court_radii = {}
    for i, case_id in enumerate(case_ids):
        court = metadata[case_id]['court']
        if court not in court_radii:
            court_radii[court] = []
        court_radii[court].append(radii[i])
    
    # Print statistics
    print("\nRadius Distribution by Court Level:")
    print("-" * 60)
    for court in ['Supreme Court', 'High Court', 'Lower Court']:
        if court in court_radii and court_radii[court]:
            mean_r = np.mean(court_radii[court])
            std_r = np.std(court_radii[court])
            min_r = np.min(court_radii[court])
            max_r = np.max(court_radii[court])
            
            print(f"{court:20s}: μ={mean_r:.4f}, σ={std_r:.4f}, range=[{min_r:.4f}, {max_r:.4f}]")
    
    # Verify hierarchy
    print("\nHierarchy Verification:")
    print("-" * 60)
    
    if 'Supreme Court' in court_radii and 'High Court' in court_radii:
        sc_mean = np.mean(court_radii['Supreme Court'])
        hc_mean = np.mean(court_radii['High Court'])
        
        if sc_mean < hc_mean:
            print("✓ Supreme Court < High Court (Hierarchy preserved!)")
        else:
            print("✗ Supreme Court >= High Court (Hierarchy NOT preserved)")
    
    if 'High Court' in court_radii and 'Lower Court' in court_radii:
        hc_mean = np.mean(court_radii['High Court'])
        lc_mean = np.mean(court_radii['Lower Court'])
        
        if hc_mean < lc_mean:
            print("✓ High Court < Lower Court (Hierarchy preserved!)")
        else:
            print("✗ High Court >= Lower Court (Hierarchy NOT preserved)")


if __name__ == "__main__":
    print("="*80)
    print("POINCARÉ VISUALIZATION (2D & 3D)")
    print("="*80)
    
    # Load model and embeddings
    print("\n1. Loading embeddings...")
    case_ids, embeddings, metadata = load_model_and_embeddings()
    print(f"   ✓ Loaded {len(case_ids)} cases")
    
    # Sample if too large
    MAX_POINTS = 5000
    if len(case_ids) > MAX_POINTS:
        print(f"   ⚠️ Dataset too large ({len(case_ids)}), sampling {MAX_POINTS} for visualization...")
        indices = np.random.choice(len(case_ids), MAX_POINTS, replace=False)
        case_ids = [case_ids[i] for i in indices]
        embeddings = embeddings[indices]
        # metadata is a dict, no need to slice

    
    # Analyze hierarchy
    analyze_hierarchy(case_ids, embeddings, metadata)
    
    # 3D Visualization (primary)
    print("\n2. Creating 3D Poincaré ball visualization...")
    fig_3d = visualize_poincare_3d(case_ids, embeddings, metadata, 'poincare_3d_visualization.png')
    
    # 2D Visualization (for comparison)
    print("\n3. Creating 2D Poincaré disk visualization...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    max_norm = np.max(np.linalg.norm(embeddings_2d, axis=1))
    if max_norm > 0:
        embeddings_2d = embeddings_2d / max_norm * 0.95
    fig_2d = visualize_poincare_disk(case_ids, embeddings_2d, metadata, 'poincare_2d_visualization.png')
    
    print("\n✅ Visualizations complete!")
    print(f"   - 3D: poincare_3d_visualization.png")
    print(f"   - 2D: poincare_2d_visualization.png")
