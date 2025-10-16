#!/usr/bin/env python3
"""
Generate LegalNexus Pipeline Diagram
Creates a detailed visual diagram of the complete system pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure with high resolution
fig, ax = plt.subplots(figsize=(20, 24), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 24)
ax.axis('off')

# Color scheme
COLOR_INGESTION = '#FF6B6B'      # Red
COLOR_PROCESSING = '#4ECDC4'     # Teal
COLOR_EXTRACTION = '#45B7D1'     # Blue
COLOR_GRAPH = '#96CEB4'          # Green
COLOR_PARALLEL = '#FFEAA7'       # Yellow
COLOR_INDEX = '#DDA15E'          # Orange
COLOR_LEARNING = '#BC6C25'       # Brown
COLOR_RETRIEVAL = '#6C5CE7'      # Purple
COLOR_RESPONSE = '#A29BFE'       # Light Purple
COLOR_VIZ = '#FD79A8'            # Pink

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, title=None, details=None):
    """Create a fancy box with title and details"""
    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='#2d3436',
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    
    # Title
    if title:
        ax.text(x + width/2, y + height - 0.15, title,
                fontsize=13, fontweight='bold', ha='center', va='top',
                color='#2d3436')
    
    # Main text
    if text:
        y_offset = y + height - 0.4 if title else y + height/2
        ax.text(x + width/2, y_offset, text,
                fontsize=10, ha='center', va='top',
                color='#2d3436', style='italic')
    
    # Details (bullet points)
    if details:
        detail_y = y + height - 0.7 if title else y + height - 0.3
        for i, detail in enumerate(details):
            ax.text(x + 0.15, detail_y - i*0.25, f"• {detail}",
                    fontsize=8, ha='left', va='top',
                    color='#2d3436')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label=None, style='->', dashed=False):
    """Create an arrow with optional label"""
    linestyle = '--' if dashed else '-'
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='#2d3436',
        linewidth=2,
        mutation_scale=25,
        linestyle=linestyle
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                color='#2d3436')

# Title
ax.text(10, 23.5, 'LegalNexus System Pipeline Architecture',
        fontsize=22, fontweight='bold', ha='center', va='top',
        color='#2d3436')
ax.text(10, 23, 'Novel Algorithms for Legal Case Similarity & Knowledge Graph',
        fontsize=12, ha='center', va='top', color='#636e72', style='italic')

# Layer 1: DATA INGESTION (top)
y_pos = 21.5
create_box(ax, 3, y_pos, 6, 1.2, None, COLOR_INGESTION,
           title="DATA INGESTION LAYER",
           details=["Multi-format support", "JSON + CSV parsers"])

# Data sources
create_box(ax, 1, y_pos-1.5, 3.5, 0.8, "JSON Files\n(Scraped Cases)", COLOR_INGESTION)
create_box(ax, 5.5, y_pos-1.5, 3.5, 0.8, "CSV Files\n(Binary/Ternary)", COLOR_INGESTION)

# Arrows from sources
create_arrow(ax, 2.75, y_pos-0.7, 2.75, y_pos, label="load_legal_data()")
create_arrow(ax, 7.25, y_pos-0.7, 7.25, y_pos, label="load_all_csv_data()")
create_arrow(ax, 4.5, y_pos-1.5, 6, y_pos-0.3)
create_arrow(ax, 5.5, y_pos-1.5, 6, y_pos-0.3)

# Layer 2: DOCUMENT PROCESSING
y_pos = 19
create_box(ax, 3, y_pos, 6, 1.5, None, COLOR_PROCESSING,
           title="DOCUMENT PROCESSING",
           details=["RecursiveCharacterTextSplitter",
                    "chunk_size=300, overlap=30",
                    "Metadata extraction"])
create_arrow(ax, 6, y_pos+1.5, 6, y_pos+2)

# Layer 3: ENTITY EXTRACTION
y_pos = 17
create_box(ax, 3, y_pos, 6, 1.5, None, COLOR_EXTRACTION,
           title="ENTITY EXTRACTION LAYER",
           details=["Extract Judges from metadata",
                    "Court hierarchy (SC/HC/District)",
                    "Statutes & legal provisions",
                    "Case metadata (title, date, id)"])
create_arrow(ax, 6, y_pos+1.5, 6, y_pos+2)

# Layer 4: KNOWLEDGE GRAPH CONSTRUCTION
y_pos = 14.8
create_box(ax, 2.5, y_pos, 7, 1.8, None, COLOR_GRAPH,
           title="KNOWLEDGE GRAPH CONSTRUCTION",
           details=["Create Case nodes (id, title, court, date, text)",
                    "JUDGED: Judge → Case",
                    "HEARD_BY: Case → Court",
                    "REFERENCES: Case → Statute"])
create_arrow(ax, 6, y_pos+1.8, 6, y_pos+2.2)

# Layer 5: PARALLEL PROCESSING (Citation + Embeddings)
y_pos = 12
# Citation branch (left)
create_box(ax, 0.5, y_pos, 4.5, 2, None, COLOR_PARALLEL,
           title="CITATION EXTRACTION",
           details=["CitationExtractor patterns:",
                    "• AIR (AIR 1950 SC 124)",
                    "• SCC ((1950) 1 SCC 124)",
                    "• High Court formats",
                    "• Case name patterns",
                    "Build CITES relationships"])

# Embedding branch (right)
create_box(ax, 5.5, y_pos, 4.5, 2, None, COLOR_PARALLEL,
           title="EMBEDDING GENERATION",
           details=["Gemini API:",
                    "models/embedding-001",
                    "768-dimensional vectors",
                    "Batch processing (5/batch)",
                    "Retry mechanism (3x)",
                    "Cache: case_embeddings.pkl"])

# Arrows from graph to parallel
create_arrow(ax, 4.5, y_pos+2, 2.75, y_pos+2.2)
create_arrow(ax, 7.5, y_pos+2, 7.75, y_pos+2.2)

# Layer 6: INDEXING
y_pos = 9.5
create_box(ax, 2, y_pos, 6, 1.8, None, COLOR_INDEX,
           title="INDEXING LAYER",
           details=["Neo4j Vector Index (vector_index)",
                    "Neo4j Keyword Index (entity_index)",
                    "Hybrid Search (vector + keyword)",
                    "Fallback: Text-based similarity"])

# Arrows from parallel to indexing
create_arrow(ax, 2.75, y_pos+1.8, 4, y_pos+2.3)
create_arrow(ax, 7.75, y_pos+1.8, 6, y_pos+2.3)

# Layer 7: ADVANCED LEARNING (GNN)
y_pos = 7
create_box(ax, 1.5, y_pos, 7, 2, None, COLOR_LEARNING,
           title="ADVANCED LEARNING LAYER (Optional)",
           details=["GNN Link Prediction:",
                    "• GraphDataProcessor",
                    "• create_node_features() → 9-dim",
                    "• GNNLinkPredictor (GCN)",
                    "• Train/Val/Test: 70/10/20",
                    "• ROC-AUC: 0.78-0.85",
                    "Output: Predicted relationships"])
create_arrow(ax, 5, y_pos+2, 5, y_pos+2.5)

# Layer 8: MULTI-MODAL RETRIEVAL
y_pos = 3.8
# Main retrieval box
create_box(ax, 1, y_pos, 8, 2.5, None, COLOR_RETRIEVAL,
           title="MULTI-MODAL QUERY & RETRIEVAL")

# Query routing
ax.text(5, y_pos+2, "Query Input → Routing Logic",
        fontsize=9, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Three retrieval methods
create_box(ax, 1.3, y_pos+0.3, 2.2, 1.2, "Vector Search\n(Gemini)", COLOR_RETRIEVAL)
create_box(ax, 3.8, y_pos+0.3, 2.2, 1.2, "Cypher Query\n(GraphQA)", COLOR_RETRIEVAL)
create_box(ax, 6.3, y_pos+0.3, 2.2, 1.2, "Text Similarity\n(Fallback)", COLOR_RETRIEVAL)

# Ranking box
create_box(ax, 3.5, y_pos-0.5, 3, 0.6, "Ranking & Scoring", COLOR_RETRIEVAL)

# Arrows
create_arrow(ax, 5, y_pos+1.8, 2.4, y_pos+1.5, style='->')
create_arrow(ax, 5, y_pos+1.8, 4.9, y_pos+1.5, style='->')
create_arrow(ax, 5, y_pos+1.8, 7.4, y_pos+1.5, style='->')
create_arrow(ax, 2.4, y_pos+0.3, 4.5, y_pos+0.1, style='->')
create_arrow(ax, 4.9, y_pos+0.3, 5, y_pos+0.1, style='->')
create_arrow(ax, 7.4, y_pos+0.3, 5.5, y_pos+0.1, style='->')

# Arrow from GNN to Retrieval
create_arrow(ax, 5, y_pos+2.5, 5, y_pos+3.2)

# Layer 9: RESPONSE GENERATION
y_pos = 1.5
create_box(ax, 2, y_pos, 6, 1.8, None, COLOR_RESPONSE,
           title="RESPONSE GENERATION",
           details=["format_case_results()",
                    "display_case_results()",
                    "Gemini Flash LLM analysis",
                    "Legal reasoning extraction",
                    "Precedent analysis"])
create_arrow(ax, 5, y_pos+1.8, 5, y_pos+2.3)

# Layer 10: VISUALIZATION
y_pos = 0
create_box(ax, 2.5, y_pos, 5, 1.2, None, COLOR_VIZ,
           title="VISUALIZATION LAYER",
           details=["Knowledge Graph (Plotly+NetworkX)",
                    "Citation Network, Analytics Dashboard"])
create_arrow(ax, 5, y_pos+1.2, 5, y_pos+1.5)

# RIGHT SIDE: Novel Algorithm Callouts
x_right = 11

# Callout 1: Hybrid Text Similarity
create_box(ax, x_right, 18, 8, 1.3, None, '#FFF3CD',
           title="🔍 Novel: Hybrid Text Similarity",
           details=["Location: kg.py::compute_text_similarity()",
                    "Triggers: API errors, offline mode",
                    "Benefits: 100% uptime, O(n) speed"])

# Callout 2: GNN Link Prediction
create_box(ax, x_right, 16, 8, 1.3, None, '#D1ECF1',
           title="🤖 Novel: GNN Link Prediction",
           details=["Location: gnn_link_prediction.py",
                    "Stage: Post-graph construction",
                    "Output: Predicted relationships"])

# Callout 3: Citation Extraction
create_box(ax, x_right, 14, 8, 1.3, None, '#D4EDDA',
           title="📚 Novel: Citation Network",
           details=["Location: citation_network.py",
                    "Patterns: 7 Indian formats (AIR, SCC)",
                    "Output: CITES relationships"])

# Callout 4: Feature Engineering
create_box(ax, x_right, 12, 8, 1.3, None, '#F8D7DA',
           title="🏗️ Novel: Feature Engineering",
           details=["Location: create_node_features()",
                    "Features: Court hierarchy, temporal",
                    "Dimensions: 9-feature vector/node"])

# Callout 5: Multi-Modal KG
create_box(ax, x_right, 10, 8, 1.3, None, '#E7E9EB',
           title="🕸️ Novel: Multi-Modal KG",
           details=["4 entity types: Case/Judge/Court/Statute",
                    "4 relationships: JUDGED/HEARD_BY/",
                    "REFERENCES/CITES"])

# Connecting lines from callouts to pipeline
create_arrow(ax, x_right, 18.6, 9, 14.2, dashed=True)  # Similarity to Retrieval
create_arrow(ax, x_right, 16.6, 8.5, 8, dashed=True)   # GNN to Learning
create_arrow(ax, x_right, 14.6, 5, 13, dashed=True)    # Citation to Parallel
create_arrow(ax, x_right, 12.6, 8.5, 8, dashed=True)   # Features to Learning
create_arrow(ax, x_right, 10.6, 9.5, 15.5, dashed=True) # KG to Graph Construction

# Legend (bottom right)
legend_elements = [
    mpatches.Patch(color=COLOR_INGESTION, label='Data Ingestion', alpha=0.8),
    mpatches.Patch(color=COLOR_PROCESSING, label='Processing', alpha=0.8),
    mpatches.Patch(color=COLOR_EXTRACTION, label='Entity Extraction', alpha=0.8),
    mpatches.Patch(color=COLOR_GRAPH, label='Graph Construction', alpha=0.8),
    mpatches.Patch(color=COLOR_PARALLEL, label='Parallel Processing', alpha=0.8),
    mpatches.Patch(color=COLOR_INDEX, label='Indexing', alpha=0.8),
    mpatches.Patch(color=COLOR_LEARNING, label='ML/GNN Learning', alpha=0.8),
    mpatches.Patch(color=COLOR_RETRIEVAL, label='Query & Retrieval', alpha=0.8),
    mpatches.Patch(color=COLOR_RESPONSE, label='Response Generation', alpha=0.8),
    mpatches.Patch(color=COLOR_VIZ, label='Visualization', alpha=0.8),
]

ax.legend(handles=legend_elements, 
          loc='lower right', 
          fontsize=8, 
          title='Pipeline Layers',
          title_fontsize=10,
          framealpha=0.9,
          bbox_to_anchor=(0.98, 0.02))

# Add file reference footer
ax.text(10, -0.5, 'Implementation: kg.py, gnn_link_prediction.py, citation_network.py, csv_data_loader.py',
        fontsize=8, ha='center', va='top', color='#636e72', style='italic')

plt.tight_layout()

# Save the diagram
output_path = '/Users/animesh/legalnexus-backend/docs/graphs/pipeline_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Pipeline diagram saved to: {output_path}")

# Also save as high-res PDF
output_pdf = '/Users/animesh/legalnexus-backend/docs/graphs/pipeline_architecture.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"✅ Pipeline diagram (PDF) saved to: {output_pdf}")

plt.close()

print("\n📊 Diagram generation complete!")
print(f"   PNG: {output_path}")
print(f"   PDF: {output_pdf}")

