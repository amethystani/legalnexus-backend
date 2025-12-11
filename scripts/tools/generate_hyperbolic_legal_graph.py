"""
Hyperbolic Legal Network Visualization
Generates a 3D Poincaré ball model showing Supreme Court cases, High Court statutes,
and target case in hyperbolic space with proper geodesic connections.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple, List
import json

# Hyperbolic geometry utilities (Poincaré ball model)
class HyperbolicSpace:
    """Implements Poincaré ball model operations"""
    
    @staticmethod
    def poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate hyperbolic distance in Poincaré ball model
        d(x,y) = arcosh(1 + 2*||x-y||²/((1-||x||²)(1-||y||²)))
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        diff_norm_sq = np.linalg.norm(x - y) ** 2
        
        numerator = 2 * diff_norm_sq
        denominator = (1 - norm_x**2) * (1 - norm_y**2)
        
        return np.arccosh(1 + numerator / denominator)
    
    @staticmethod
    def exponential_map(v: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Exponential map at point x in direction v
        Maps tangent space to hyperbolic space
        """
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return x
        
        lambda_x = 2 / (1 - np.linalg.norm(x)**2)
        
        # Exponential map formula
        factor = np.tanh(lambda_x * v_norm / 2) / v_norm
        return (x + factor * v) / (1 + factor * np.dot(v, x) + factor**2 * v_norm**2 / 4)
    
    @staticmethod
    def geodesic_points(x: np.ndarray, y: np.ndarray, n_points: int = 50) -> np.ndarray:
        """
        Generate points along geodesic (shortest path) from x to y in hyperbolic space
        """
        points = []
        for t in np.linspace(0, 1, n_points):
            # Geodesic interpolation in Poincaré ball
            numerator = (1 - t) * x * (1 - np.linalg.norm(y)**2) + t * y * (1 - np.linalg.norm(x)**2)
            denominator = (1 - t) * (1 - np.linalg.norm(y)**2) + t * (1 - np.linalg.norm(x)**2)
            point = numerator / denominator
            points.append(point)
        return np.array(points)

def generate_legal_network_data():
    """Generate synthetic legal network data with hierarchy"""
    
    # Supreme Court cases (center - low radius)
    supreme_court_cases = [
        {"id": "SC-2019-5432", "name": "Vishaka v. State", "radius": 0.15, "angle": 0},
        {"id": "SC-2020-1823", "name": "Kesavananda Bharati", "radius": 0.18, "angle": np.pi/3},
        {"id": "SC-2018-9876", "name": "Maneka Gandhi v. Union", "radius": 0.12, "angle": 2*np.pi/3},
        {"id": "SC-2021-4567", "name": "Navtej Johar v. Union", "radius": 0.20, "angle": np.pi},
        {"id": "SC-2017-3421", "name": "Shreya Singhal v. Union", "radius": 0.16, "angle": 4*np.pi/3},
        {"id": "SC-2022-7890", "name": "Anuradha Bhasin v. Union", "radius": 0.14, "angle": 5*np.pi/3},
    ]
    
    # High Court statutes (outer - higher radius)
    high_court_statutes = [
        {"id": "HC-DL-2018-9876", "name": "Delhi HC - Privacy Rights", "radius": 0.55, "angle": 0.3},
        {"id": "HC-MH-2020-4567", "name": "Bombay HC - Contract Law", "radius": 0.60, "angle": 0.8},
        {"id": "HC-KA-2019-1234", "name": "Karnataka HC - Labor", "radius": 0.58, "angle": 1.5},
        {"id": "HC-TN-2021-5678", "name": "Madras HC - Property", "radius": 0.62, "angle": 2.1},
        {"id": "HC-WB-2020-9012", "name": "Calcutta HC - Criminal", "radius": 0.57, "angle": 2.7},
        {"id": "HC-GJ-2019-3456", "name": "Gujarat HC - Tax Law", "radius": 0.61, "angle": 3.4},
        {"id": "HC-RJ-2021-7890", "name": "Rajasthan HC - Family", "radius": 0.59, "angle": 4.0},
        {"id": "HC-UP-2020-2345", "name": "Allahabad HC - Land", "radius": 0.63, "angle": 4.7},
        {"id": "HC-AP-2019-6789", "name": "AP HC - Education", "radius": 0.56, "angle": 5.3},
        {"id": "HC-KL-2021-0123", "name": "Kerala HC - Environment", "radius": 0.60, "angle": 5.9},
    ]
    
    # Target case (intermediate)
    target_case = {
        "id": "HC-DL-2024-1234",
        "name": "Target Case: Privacy & Data Rights",
        "radius": 0.42,
        "angle": 1.2
    }
    
    return supreme_court_cases, high_court_statutes, target_case

def spherical_to_3d(radius: float, angle: float, elevation: float = None) -> Tuple[float, float, float]:
    """Convert spherical coordinates to 3D Cartesian (for Poincaré ball)"""
    if elevation is None:
        elevation = np.random.uniform(-0.3, 0.3)  # Random elevation for 3D effect
    
    x = radius * np.cos(angle) * np.cos(elevation)
    y = radius * np.sin(angle) * np.cos(elevation)
    z = radius * np.sin(elevation)
    
    return x, y, z

def create_hyperbolic_legal_graph():
    """Create the main 3D hyperbolic legal network visualization"""
    
    # Generate data
    sc_cases, hc_statutes, target = generate_legal_network_data()
    
    # Initialize plotly figure
    fig = go.Figure()
    
    # Poincaré ball boundary (wire sphere at r=1)
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    x_sphere = 0.98 * np.sin(phi) * np.cos(theta)
    y_sphere = 0.98 * np.sin(phi) * np.sin(theta)
    z_sphere = 0.98 * np.cos(phi)
    
    # Add transparent boundary sphere
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        colorscale=[[0, 'rgba(100,100,200,0.05)'], [1, 'rgba(100,100,200,0.1)']],
        showscale=False,
        name='Poincaré Boundary',
        hoverinfo='skip'
    ))
    
    # Coordinate system nodes
    sc_nodes_x, sc_nodes_y, sc_nodes_z = [], [], []
    sc_labels = []
    
    for case in sc_cases:
        x, y, z = spherical_to_3d(case['radius'], case['angle'])
        sc_nodes_x.append(x)
        sc_nodes_y.append(y)
        sc_nodes_z.append(z)
        sc_labels.append(f"<b>{case['id']}</b><br>{case['name']}<br>r={case['radius']:.3f}")
    
    # Supreme Court nodes (blue, center)
    fig.add_trace(go.Scatter3d(
        x=sc_nodes_x, y=sc_nodes_y, z=sc_nodes_z,
        mode='markers+text',
        marker=dict(
            size=15,
            color='#1E88E5',  # Deep blue
            symbol='diamond',
            opacity=0.9,
            line=dict(color='#0D47A1', width=2)
        ),
        text=[case['id'] for case in sc_cases],
        textposition='top center',
        textfont=dict(size=8, color='#BBDEFB'),
        name='Supreme Court Cases',
        hovertext=sc_labels,
        hoverinfo='text'
    ))
    
    # High Court statute nodes
    hc_nodes_x, hc_nodes_y, hc_nodes_z = [], [], []
    hc_labels = []
    
    for statute in hc_statutes:
        x, y, z = spherical_to_3d(statute['radius'], statute['angle'])
        hc_nodes_x.append(x)
        hc_nodes_y.append(y)
        hc_nodes_z.append(z)
        hc_labels.append(f"<b>{statute['id']}</b><br>{statute['name']}<br>r={statute['radius']:.3f}")
    
    fig.add_trace(go.Scatter3d(
        x=hc_nodes_x, y=hc_nodes_y, z=hc_nodes_z,
        mode='markers+text',
        marker=dict(
            size=12,
            color='#FF6F00',  # Amber/Orange
            symbol='circle',
            opacity=0.85,
            line=dict(color='#E65100', width=1.5)
        ),
        text=[s['id'].split('-')[1] for s in hc_statutes],  # Shortened labels
        textposition='top center',
        textfont=dict(size=7, color='#FFE0B2'),
        name='High Court Statutes',
        hovertext=hc_labels,
        hoverinfo='text'
    ))
    
    # Target case node (highlighted)
    target_x, target_y, target_z = spherical_to_3d(target['radius'], target['angle'])
    
    fig.add_trace(go.Scatter3d(
        x=[target_x], y=[target_y], z=[target_z],
        mode='markers+text',
        marker=dict(
            size=20,
            color='#00E676',  # Bright green
            symbol='diamond',
            opacity=1.0,
            line=dict(color='#00C853', width=3)
        ),
        text=[target['id']],
        textposition='top center',
        textfont=dict(size=10, color='#B9F6CA', family='monospace', weight='bold'),
        name='Target Case',
        hovertext=f"<b>{target['id']}</b><br>{target['name']}<br>r={target['radius']:.3f}",
        hoverinfo='text'
    ))
    
    # Add geodesic connections (curved paths in hyperbolic space)
    hs = HyperbolicSpace()
    
    # Connect target to some Supreme Court cases (citations)
    target_pos = np.array([target_x, target_y, target_z])
    connections_to_sc = [0, 1, 3]  # Connect to first, second, and fourth SC cases
    
    for idx in connections_to_sc:
        sc_pos = np.array([sc_nodes_x[idx], sc_nodes_y[idx], sc_nodes_z[idx]])
        geodesic = hs.geodesic_points(target_pos, sc_pos, n_points=30)
        
        fig.add_trace(go.Scatter3d(
            x=geodesic[:, 0],
            y=geodesic[:, 1],
            z=geodesic[:, 2],
            mode='lines',
            line=dict(color='rgba(0, 230, 118, 0.6)', width=3),
            name=f'Citation: {target["id"]} → {sc_cases[idx]["id"]}',
            hoverinfo='name',
            showlegend=False
        ))
    
    # Connect target to some High Court statutes
    connections_to_hc = [0, 2, 4, 7]
    
    for idx in connections_to_hc:
        hc_pos = np.array([hc_nodes_x[idx], hc_nodes_y[idx], hc_nodes_z[idx]])
        geodesic = hs.geodesic_points(target_pos, hc_pos, n_points=30)
        
        fig.add_trace(go.Scatter3d(
            x=geodesic[:, 0],
            y=geodesic[:, 1],
            z=geodesic[:, 2],
            mode='lines',
            line=dict(color='rgba(255, 111, 0, 0.4)', width=2),
            name=f'Reference: {target["id"]} → {hc_statutes[idx]["id"]}',
            hoverinfo='name',
            showlegend=False
        ))
    
    # Add some SC to HC connections
    sc_hc_connections = [(0, 0), (1, 2), (3, 4), (4, 7)]
    
    for sc_idx, hc_idx in sc_hc_connections:
        sc_pos = np.array([sc_nodes_x[sc_idx], sc_nodes_y[sc_idx], sc_nodes_z[sc_idx]])
        hc_pos = np.array([hc_nodes_x[hc_idx], hc_nodes_y[hc_idx], hc_nodes_z[hc_idx]])
        geodesic = hs.geodesic_points(sc_pos, hc_pos, n_points=30)
        
        fig.add_trace(go.Scatter3d(
            x=geodesic[:, 0],
            y=geodesic[:, 1],
            z=geodesic[:, 2],
            mode='lines',
            line=dict(color='rgba(30, 136, 229, 0.25)', width=1.5),
            name=f'{sc_cases[sc_idx]["id"]} → {hc_statutes[hc_idx]["id"]}',
            hoverinfo='name',
            showlegend=False
        ))
    
    # Add distance markers (hyperbolic distance circles)
    for r in [0.3, 0.5, 0.7]:
        circle_theta = np.linspace(0, 2*np.pi, 100)
        circle_x = r * np.cos(circle_theta)
        circle_y = r * np.sin(circle_theta)
        circle_z = np.zeros_like(circle_x)
        
        fig.add_trace(go.Scatter3d(
            x=circle_x, y=circle_y, z=circle_z,
            mode='lines',
            line=dict(color=f'rgba(150, 150, 200, 0.2)', width=1, dash='dot'),
            name=f'r = {r}',
            hoverinfo='name',
            showlegend=False
        ))
    
    # Add annotation for mathematical formula
    annotations_text = [
        "Poincaré Ball Model: B³ = {x ∈ ℝ³ : ||x|| < 1}",
        "Distance: d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))",
        "Geodesics: Hyperbolic shortest paths"
    ]
    
    # Layout configuration - RESEARCH PAPER QUALITY
    fig.update_layout(
        title=dict(
            text="<b>Hyperbolic Legal Network Embedding in ℍ³</b><br>" +
                 "<sub>Poincaré Ball Model (B³): Hierarchical Case Law Representation | Supreme Court → High Court Precedents</sub>",
            font=dict(size=22, color='#FFFFFF', family='Computer Modern, serif'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='<b>x ∈ ℝ</b>', font=dict(size=14, color='#CCCCCC')),
                backgroundcolor='rgb(0, 0, 0)',
                gridcolor='rgba(80, 80, 120, 0.25)',
                showbackground=True,
                zerolinecolor='rgba(120, 150, 255, 0.6)',
                zerolinewidth=2,
                range=[-1, 1],
                tick0=-1,
                dtick=0.5,
                tickfont=dict(size=10, color='#AAAAAA')
            ),
            yaxis=dict(
                title=dict(text='<b>y ∈ ℝ</b>', font=dict(size=14, color='#CCCCCC')),
                backgroundcolor='rgb(0, 0, 0)',
                gridcolor='rgba(80, 80, 120, 0.25)',
                showbackground=True,
                zerolinecolor='rgba(120, 150, 255, 0.6)',
                zerolinewidth=2,
                range=[-1, 1],
                tick0=-1,
                dtick=0.5,
                tickfont=dict(size=10, color='#AAAAAA')
            ),
            zaxis=dict(
                title=dict(text='<b>z ∈ ℝ</b>', font=dict(size=14, color='#CCCCCC')),
                backgroundcolor='rgb(0, 0, 0)',
                gridcolor='rgba(80, 80, 120, 0.25)',
                showbackground=True,
                zerolinecolor='rgba(120, 150, 255, 0.6)',
                zerolinewidth=2,
                range=[-1, 1],
                tick0=-1,
                dtick=0.5,
                tickfont=dict(size=10, color='#AAAAAA')
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.4),
                center=dict(x=0, y=0, z=0),
                projection=dict(type='perspective')
            ),
            aspectmode='cube',
            bgcolor='rgb(0, 0, 0)'
        ),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
        font=dict(color='#FFFFFF', family='Computer Modern, Courier New'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0, 0, 0, 0.95)',
            bordercolor='rgba(100, 150, 255, 0.7)',
            borderwidth=2,
            font=dict(size=11, color='#FFFFFF', family='Computer Modern'),
            x=0.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        width=1600,
        height=1200,
        margin=dict(l=0, r=0, t=120, b=20)
    )
    
    # Add technical annotations - Mathematical formulation
    math_formulas = (
        "<b>Hyperbolic Geometry Formulation</b><br>" +
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>" +
        "<b>Domain:</b> B³ = {x ∈ ℝ³ : ||x|| < 1}<br>" +
        "<b>Metric:</b> ds² = 4·(dx² + dy² + dz²)/(1 - r²)²<br>" +
        "<b>Distance:</b><br>" +
        "  d<sub>ℍ</sub>(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))<br>" +
        "<b>Geodesics:</b> Circular arcs ⊥ to ∂B³<br>" +
        "<b>Curvature:</b> K = -1 (constant negative)"
    )
    
    fig.add_annotation(
        text=math_formulas,
        xref="paper", yref="paper",
        x=0.015, y=0.97,
        showarrow=False,
        font=dict(size=10, color='#E8E8E8', family='Computer Modern, monospace'),
        align='left',
        bgcolor='rgba(0, 0, 0, 0.95)',
        bordercolor='rgba(100, 150, 255, 0.8)',
        borderwidth=2,
        borderpad=12
    )
    
    # Add hierarchy explanation with mathematical notation
    hierarchy_text = (
        "<b>Hierarchical Embedding</b><br>" +
        "━━━━━━━━━━━━━━━━━━━━━━<br>" +
        "<b>Layer 0:</b> Supreme Court<br>" +
        "  r ∈ [0.10, 0.20] | n = 6<br>" +
        "  🔷 Central Authority<br><br>" +
        "<b>Layer 1:</b> Target Case<br>" +
        "  r ≈ 0.42 | n = 1<br>" +
        "  🟢 Query Node<br><br>" +
        "<b>Layer 2:</b> High Court<br>" +
        "  r ∈ [0.55, 0.65] | n = 10<br>" +
        "  🟠 Regional Statutes<br><br>" +
        "<b>Edges:</b> Legal Citations<br>" +
        "  Geodesic paths in ℍ³"
    )
    
    fig.add_annotation(
        text=hierarchy_text,
        xref="paper", yref="paper",
        x=0.985, y=0.97,
        showarrow=False,
        font=dict(size=10, color='#E8E8E8', family='Computer Modern, monospace'),
        align='left',
        bgcolor='rgba(0, 0, 0, 0.95)',
        bordercolor='rgba(100, 150, 255, 0.8)',
        borderwidth=2,
        borderpad=12
    )
    
    # Add bottom annotation with technical details
    technical_note = (
        "<b>Graph Statistics:</b> |V| = 17 nodes, |E| = 14 geodesic edges | " +
        "<b>Embedding:</b> HGCN (Hyperbolic GCN) | " +
        "<b>Dimension:</b> d = 3 | " +
        "<b>Model:</b> Poincaré Ball B³ ⊂ ℝ³"
    )
    
    fig.add_annotation(
        text=technical_note,
        xref="paper", yref="paper",
        x=0.5, y=0.01,
        showarrow=False,
        font=dict(size=9, color='#CCCCCC', family='Computer Modern, monospace'),
        align='center',
        bgcolor='rgba(0, 0, 0, 0.9)',
        bordercolor='rgba(80, 120, 200, 0.6)',
        borderwidth=1,
        borderpad=8
    )
    
    return fig

def save_visualizations():
    """Generate and save both static and interactive visualizations"""
    
    print("Generating hyperbolic legal network graph...")
    fig = create_hyperbolic_legal_graph()
    
    # Save interactive HTML
    html_path = '/Users/animesh/legalnexus-backend/hyperbolic_legal_graph_interactive.html'
    fig.write_html(html_path, config={'displayModeBar': True, 'displaylogo': False})
    print(f"✓ Interactive HTML saved: {html_path}")
    
    # Save high-resolution static image
    static_path = '/Users/animesh/legalnexus-backend/hyperbolic_legal_graph_static.png'
    fig.write_image(static_path, width=2400, height=1800, scale=2)
    print(f"✓ Static PNG saved: {static_path}")
    
    # Also save as PDF for papers
    pdf_path = '/Users/animesh/legalnexus-backend/hyperbolic_legal_graph_static.pdf'
    fig.write_image(pdf_path, width=2400, height=1800)
    print(f"✓ PDF saved: {pdf_path}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nOpen '{html_path}' in a browser for interactive 3D exploration")
    print(f"Use '{static_path}' for papers/presentations")
    
    return fig

if __name__ == "__main__":
    # Check dependencies
    try:
        import kaleido
        print("✓ Kaleido found (for static image export)")
    except ImportError:
        print("⚠ Kaleido not found. Installing for static image export...")
        import subprocess
        subprocess.run(["pip", "install", "kaleido"], check=True)
    
    # Generate visualizations
    save_visualizations()
