#!/usr/bin/env python3
"""
Generate GNN Integration Diagram
Shows how GNN connects to Knowledge Graph and Vector Index system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

def create_gnn_integration_diagram():
    """Generate complete GNN integration flow diagram"""
    fig, ax = plt.subplots(figsize=(18, 22), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Title
    ax.text(9, 21.5, 'GNN Integration with Knowledge Graph System', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(9, 21, 'Complete Pipeline Flow - LegalNexus Architecture', 
            fontsize=12, ha='center', style='italic', color='#666')
    
    # Color scheme
    COLOR_KG = '#FFE6E6'      # Light red - Knowledge Graph
    COLOR_DATA = '#E6F3FF'    # Light blue - Data layer
    COLOR_GNN = '#FFF4E6'     # Light orange - GNN
    COLOR_PREDICT = '#E6FFE6' # Light green - Predictions
    COLOR_SEARCH = '#F0E6FF'  # Light purple - Search
    
    y_pos = 19.5
    
    # ========================================================================
    # PHASE 1: KNOWLEDGE GRAPH CREATION
    # ========================================================================
    
    # Main box
    create_box(ax, 0.5, y_pos, 17, 3.5, COLOR_KG, 
               'PHASE 1: KNOWLEDGE GRAPH CREATION (kg.py)', 11)
    
    # Step boxes inside
    create_small_box(ax, 1, y_pos+2.8, 3.5, 0.5, '#FFD6D6', '1. Load Legal Documents')
    create_small_box(ax, 5, y_pos+2.8, 3.5, 0.5, '#FFD6D6', '2. Create Nodes in Neo4j')
    create_small_box(ax, 9, y_pos+2.8, 3.5, 0.5, '#FFD6D6', '3. Create Relationships')
    create_small_box(ax, 13, y_pos+2.8, 3.5, 0.5, '#FFD6D6', '4. Generate Embeddings')
    
    # Details
    details_y = y_pos + 2.2
    ax.text(2.75, details_y, '• JSON files\n• CSV data', fontsize=7, ha='center')
    ax.text(6.75, details_y, '• Case nodes\n• Judge nodes\n• Court nodes\n• Statute nodes', fontsize=7, ha='center')
    ax.text(10.75, details_y, '• JUDGED\n• HEARD_BY\n• REFERENCES', fontsize=7, ha='center')
    ax.text(14.75, details_y, '• Gemini API\n• 768-dim vectors\n• Store embeddings', fontsize=7, ha='center')
    
    # Vector Index creation
    create_small_box(ax, 6, y_pos+0.3, 6, 0.6, '#FFD6D6', '5. Create Vector Index')
    ax.text(9, y_pos+0.65, 'vector_index (semantic) + entity_index (keyword)', 
            fontsize=7, ha='center', style='italic')
    
    # Arrow down
    create_arrow(ax, 9, y_pos, 9, y_pos-0.3)
    
    y_pos -= 1
    
    # Neo4j data layer
    create_box(ax, 3, y_pos, 12, 1, COLOR_DATA, 'DATA IN NEO4J GRAPH', 10)
    ax.text(9, y_pos+0.5, 'Nodes: Case, Judge, Court, Statute | Relationships: JUDGED, HEARD_BY, REFERENCES | Indexes: vector_index, entity_index', 
            fontsize=7, ha='center')
    
    create_arrow(ax, 9, y_pos, 9, y_pos-0.5)
    
    # ========================================================================
    # PHASE 2: GNN EXTRACTS DATA
    # ========================================================================
    
    y_pos -= 1.2
    
    create_box(ax, 0.5, y_pos-3.5, 17, 3.8, COLOR_GNN, 
               'PHASE 2: GNN EXTRACTS DATA FROM GRAPH', 11)
    
    # GraphDataProcessor
    ax.text(9, y_pos-0.8, 'GraphDataProcessor.extract_graph_data()', 
            fontsize=9, ha='center', fontweight='bold')
    
    # Query boxes
    create_small_box(ax, 1, y_pos-1.5, 7.5, 1.2, '#FFE6CC', 'QUERY 1: Get all nodes')
    ax.text(4.75, y_pos-1.8, 'MATCH (n)\nRETURN id(n), labels(n),\n       n.title, n.court, n.date', 
            fontsize=7, ha='center', family='monospace')
    
    create_small_box(ax, 9.5, y_pos-1.5, 7.5, 1.2, '#FFE6CC', 'QUERY 2: Get all relationships')
    ax.text(13.25, y_pos-1.8, 'MATCH (source)-[r]->(target)\nRETURN id(source), id(target),\n       type(r)', 
            fontsize=7, ha='center', family='monospace')
    
    # Results
    ax.text(4.75, y_pos-2.9, 'Returns: All Case, Judge,\nCourt, Statute nodes', 
            fontsize=7, ha='center', style='italic', color='#666')
    ax.text(13.25, y_pos-2.9, 'Returns: All JUDGED, HEARD_BY,\nREFERENCES relationships', 
            fontsize=7, ha='center', style='italic', color='#666')
    
    # Creates section
    ax.text(9, y_pos-3.4, 'Creates: edge_index [[0,1,2,...], [3,4,5,...]] | node_features (9-dim) | metadata (mappings)', 
            fontsize=7, ha='center', fontweight='bold')
    
    create_arrow(ax, 9, y_pos-3.5, 9, y_pos-4)
    
    # ========================================================================
    # PHASE 3: GNN TRAINING
    # ========================================================================
    
    y_pos -= 4.5
    
    create_box(ax, 0.5, y_pos-3, 17, 3.3, '#E6F2FF', 
               'PHASE 3: GNN TRAINING (gnn_link_prediction.py)', 11)
    
    # Input section
    ax.text(2.5, y_pos-0.5, 'Input from Graph:', fontsize=8, fontweight='bold')
    ax.text(2.5, y_pos-0.8, '• Nodes: Cases, Judges, Courts, Statutes', fontsize=7)
    ax.text(2.5, y_pos-1.05, '• Features: 9-dim vectors', fontsize=7)
    ax.text(2.5, y_pos-1.3, '• Edges: Existing relationships', fontsize=7)
    
    # GNN Model architecture (vertical flow)
    model_x = 7
    create_small_box(ax, model_x, y_pos-0.5, 5, 0.4, '#CCE5FF', 'Feature Engineering (9-dim)')
    create_arrow(ax, model_x+2.5, y_pos-0.5, model_x+2.5, y_pos-0.7, label='')
    
    create_small_box(ax, model_x, y_pos-1, 5, 0.4, '#B3D9FF', 'GCN Layer 1 (9→64)')
    create_arrow(ax, model_x+2.5, y_pos-1, model_x+2.5, y_pos-1.2)
    
    create_small_box(ax, model_x, y_pos-1.5, 5, 0.4, '#B3D9FF', 'ReLU + Dropout (0.5)')
    create_arrow(ax, model_x+2.5, y_pos-1.5, model_x+2.5, y_pos-1.7)
    
    create_small_box(ax, model_x, y_pos-2, 5, 0.4, '#B3D9FF', 'GCN Layer 2 (64→64)')
    create_arrow(ax, model_x+2.5, y_pos-2, model_x+2.5, y_pos-2.2)
    
    create_small_box(ax, model_x, y_pos-2.5, 5, 0.4, '#99CCFF', 'Link Prediction Head')
    
    # Training details
    ax.text(14.5, y_pos-0.5, 'Training:', fontsize=8, fontweight='bold')
    ax.text(14.5, y_pos-0.8, '• Positive: Existing edges', fontsize=7)
    ax.text(14.5, y_pos-1.05, '• Negative: Random non-edges', fontsize=7)
    ax.text(14.5, y_pos-1.3, '• Loss: Binary Cross Entropy', fontsize=7)
    ax.text(14.5, y_pos-1.55, '• Optimizer: Adam', fontsize=7)
    ax.text(14.5, y_pos-1.8, '• ROC-AUC: 0.78-0.85', fontsize=7, fontweight='bold', color='#0066CC')
    
    create_arrow(ax, 9, y_pos-3, 9, y_pos-3.5)
    
    # ========================================================================
    # PHASE 4: GNN PREDICTIONS
    # ========================================================================
    
    y_pos -= 4
    
    create_box(ax, 0.5, y_pos-2, 17, 2.3, COLOR_PREDICT, 
               'PHASE 4: GNN PREDICTIONS (predict_links)', 11)
    
    ax.text(9, y_pos-0.5, 'GNN predicts missing relationships:', 
            fontsize=9, ha='center', fontweight='bold')
    
    # Prediction examples
    create_small_box(ax, 2, y_pos-1.2, 5, 0.5, '#D4EDDA', 'Case_123 -[CITES]→ Case_456')
    ax.text(4.5, y_pos-1.6, '89% probability', fontsize=7, ha='center', style='italic')
    
    create_small_box(ax, 7.5, y_pos-1.2, 5, 0.5, '#D4EDDA', 'Case_789 -[REFERENCES]→ Statute_X')
    ax.text(10, y_pos-1.6, '92% probability', fontsize=7, ha='center', style='italic')
    
    create_small_box(ax, 13, y_pos-1.2, 4, 0.5, '#D4EDDA', 'Judge_Y -[JUDGED]→ Case_321')
    ax.text(15, y_pos-1.6, '85% probability', fontsize=7, ha='center', style='italic')
    
    # Usage
    ax.text(9, y_pos-1.9, 'Usage: Enrich knowledge graph | Recommend cases | Validate missing citations', 
            fontsize=7, ha='center', style='italic')
    
    create_arrow(ax, 9, y_pos-2, 9, y_pos-2.5)
    
    # ========================================================================
    # PHASE 5: INTEGRATION WITH SEARCH
    # ========================================================================
    
    y_pos -= 3
    
    create_box(ax, 0.5, y_pos-3.5, 17, 3.8, COLOR_SEARCH, 
               'PHASE 5: INTEGRATION WITH SEARCH SYSTEM', 11)
    
    # User query
    create_small_box(ax, 6, y_pos-0.5, 6, 0.5, '#E6D4F5', 'User Query: "Section 65B electronic evidence"')
    create_arrow(ax, 9, y_pos-0.5, 9, y_pos-0.8)
    
    # Step 1: Vector search
    create_small_box(ax, 1, y_pos-1.5, 7, 1.2, '#F0E6FF', 'Step 1: Vector Index Search')
    ax.text(4.5, y_pos-1.7, '• Uses Gemini embeddings (768-dim)', fontsize=7, ha='center')
    ax.text(4.5, y_pos-1.95, '• Finds semantically similar cases', fontsize=7, ha='center')
    ax.text(4.5, y_pos-2.2, '• Returns: Anvar P.V. case (89% match)', fontsize=7, ha='center', fontweight='bold')
    
    # Step 2: GNN enhancement
    create_small_box(ax, 10, y_pos-1.5, 7, 1.2, '#F0E6FF', 'Step 2: GNN Enhancement')
    ax.text(13.5, y_pos-1.7, '• GNN suggests related cases', fontsize=7, ha='center')
    ax.text(13.5, y_pos-1.95, '• Cases that cite Anvar P.V.', fontsize=7, ha='center')
    ax.text(13.5, y_pos-2.2, '• Similar cases from same court', fontsize=7, ha='center')
    
    # Combined result
    create_small_box(ax, 4, y_pos-3.2, 10, 0.6, '#D6C6E6', 'Combined Result')
    ax.text(9, y_pos-2.85, 'Direct matches (Vector) + Related cases (GNN) + Graph relationships', 
            fontsize=7, ha='center', fontweight='bold')
    
    # Legend
    legend_y = 0.8
    ax.text(9, legend_y, 'Pipeline Legend', fontsize=10, ha='center', fontweight='bold')
    
    legend_items = [
        (1, legend_y-0.3, COLOR_KG, 'Knowledge Graph Creation'),
        (5, legend_y-0.3, COLOR_GNN, 'GNN Data Extraction'),
        (9, legend_y-0.3, '#E6F2FF', 'GNN Training'),
        (13, legend_y-0.3, COLOR_PREDICT, 'GNN Predictions'),
        (5, legend_y-0.7, COLOR_SEARCH, 'Search Integration'),
        (9, legend_y-0.7, COLOR_DATA, 'Data Storage Layer')
    ]
    
    for x, y, color, label in legend_items:
        rect = Rectangle((x-0.7, y-0.1), 0.4, 0.2, facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(rect)
        ax.text(x-0.2, y, label, fontsize=7, va='center')
    
    plt.tight_layout()
    plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/gnn_integration_complete.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Generated: docs/graphs/gnn_integration_complete.png")
    plt.close()

def create_box(ax, x, y, width, height, color, title, title_size=10):
    """Create a phase box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='#2d3436',
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.2, title, 
            fontsize=title_size, fontweight='bold', ha='center', va='top')

def create_small_box(ax, x, y, width, height, color, text):
    """Create a small component box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor='#555',
        linewidth=1.5,
        alpha=0.85
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            fontsize=8, ha='center', va='center', fontweight='bold')

def create_arrow(ax, x1, y1, x2, y2, label=None):
    """Create an arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color='#2d3436',
        linewidth=2.5,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=7, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

if __name__ == "__main__":
    create_gnn_integration_diagram()
    print("\n📊 GNN Integration Diagram Created Successfully!")
    print("   File: docs/graphs/gnn_integration_complete.png")
    print("   Resolution: 300 DPI (high quality)")


