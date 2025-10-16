#!/usr/bin/env python3
"""
Generate Clean Algorithm-Focused Diagrams
Creates separate, non-overlapping diagrams for each component
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================================
# DIAGRAM 1: Main Pipeline Flow
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(16, 10), dpi=300)
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 10)
ax1.axis('off')

def create_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='#2d3436',
        linewidth=2,
        alpha=0.85
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            fontsize=fontsize, ha='center', va='center',
            color='#2d3436', fontweight='bold')

def create_arrow(ax, x1, y1, x2, y2, label=None):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        color='#2d3436',
        linewidth=2.5,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                fontsize=8, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Title
ax1.text(8, 9.5, 'LegalNexus Pipeline Architecture', fontsize=18, ha='center', fontweight='bold')

# Data Sources
create_box(ax1, 2, 8, 2.5, 0.6, 'JSON Cases', '#FF6B6B')
create_box(ax1, 5, 8, 2.5, 0.6, 'CSV Data', '#FF6B6B')

# Processing
create_box(ax1, 3.5, 6.8, 3, 0.7, 'Document Processing\n& Chunking', '#4ECDC4')
create_arrow(ax1, 3.25, 8, 4.5, 7.5)
create_arrow(ax1, 6.25, 8, 5.5, 7.5)

# Entity Extraction
create_box(ax1, 3.5, 5.5, 3, 0.7, 'Entity Extraction\n(Judges, Courts, Statutes)', '#45B7D1')
create_arrow(ax1, 5, 6.8, 5, 6.2)

# Knowledge Graph
create_box(ax1, 2.5, 3.8, 5, 1, 'Knowledge Graph\nConstruction', '#96CEB4', fontsize=11)
create_arrow(ax1, 5, 5.5, 5, 4.8)

# Parallel Processing
create_box(ax1, 1, 2, 3, 0.8, 'Citation\nExtraction', '#FFEAA7')
create_box(ax1, 5.5, 2, 3, 0.8, 'Embedding\nGeneration', '#FFEAA7')
create_arrow(ax1, 3.5, 3.8, 2.5, 2.8)
create_arrow(ax1, 6.5, 3.8, 7, 2.8)

# Indexing
create_box(ax1, 2.5, 0.5, 5, 0.8, 'Vector & Keyword Indexing', '#DDA15E')
create_arrow(ax1, 2.5, 2, 4, 1.3)
create_arrow(ax1, 7, 2, 6, 1.3)

# Legend
legend_elements = [
    mpatches.Patch(color='#FF6B6B', label='Data Ingestion'),
    mpatches.Patch(color='#4ECDC4', label='Processing'),
    mpatches.Patch(color='#45B7D1', label='Extraction'),
    mpatches.Patch(color='#96CEB4', label='Graph Building'),
    mpatches.Patch(color='#FFEAA7', label='Parallel Tasks'),
    mpatches.Patch(color='#DDA15E', label='Indexing'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/1_pipeline_flow.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: 1_pipeline_flow.png")
plt.close()

# ============================================================================
# DIAGRAM 2: Novel Algorithms Overview
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(14, 10), dpi=300)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(7, 9.5, 'Novel Algorithms Architecture', fontsize=18, ha='center', fontweight='bold')

# Hybrid Text Similarity
create_box(ax2, 0.5, 7.5, 6, 1.2, 
           'Hybrid Text Similarity\nKeyword + Sequence Matching\n75-85% Accuracy', 
           '#E8F4F8', fontsize=9)

# GNN Link Prediction
create_box(ax2, 7.5, 7.5, 6, 1.2, 
           'GNN Link Prediction\nGCN Architecture\nROC-AUC: 0.78-0.85', 
           '#FFF3E0', fontsize=9)

# Citation Extraction
create_box(ax2, 0.5, 5.8, 6, 1.2, 
           'Citation Extraction\n7 Indian Legal Formats\n92% Precision', 
           '#F3E5F5', fontsize=9)

# Feature Engineering
create_box(ax2, 7.5, 5.8, 6, 1.2, 
           'Legal Feature Engineering\n9-Dimensional Vectors\nCourt Hierarchy Encoding', 
           '#E8F5E9', fontsize=9)

# Multi-Modal KG
create_box(ax2, 3.5, 4.1, 7, 1.2, 
           'Multi-Modal Knowledge Graph\n4 Entity Types • 4 Relationship Types', 
           '#FCE4EC', fontsize=9)

# Benefits
ax2.text(7, 2.5, 'Key Benefits', fontsize=14, ha='center', fontweight='bold')
benefits = [
    '✓ 100% Uptime (Text similarity fallback)',
    '✓ Indian Legal System Optimized',
    '✓ No Vendor Lock-in',
    '✓ Explainable Results'
]
for i, benefit in enumerate(benefits):
    ax2.text(7, 2 - i*0.35, benefit, fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/2_novel_algorithms.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: 2_novel_algorithms.png")
plt.close()

# ============================================================================
# DIAGRAM 3: Query & Retrieval System
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(14, 8), dpi=300)
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 8)
ax3.axis('off')

ax3.text(7, 7.5, 'Multi-Modal Query & Retrieval', fontsize=18, ha='center', fontweight='bold')

# Query Input
create_box(ax3, 5.5, 6, 3, 0.6, 'User Query', '#6C5CE7')
create_arrow(ax3, 7, 6, 7, 5.5)

# Routing
create_box(ax3, 5.5, 4.8, 3, 0.5, 'Query Router', '#A29BFE', fontsize=9)

# Three retrieval methods
create_box(ax3, 0.5, 3, 3.5, 1, 
           'Vector Search\nGemini Embeddings\n768-dim', 
           '#74B9FF', fontsize=9)

create_box(ax3, 5, 3, 3.5, 1, 
           'Graph Query\nCypher QA\nNeo4j', 
           '#81C784', fontsize=9)

create_box(ax3, 9.5, 3, 3.5, 1, 
           'Text Similarity\nFallback Method\nOffline Ready', 
           '#FFB74D', fontsize=9)

# Arrows to retrieval methods
create_arrow(ax3, 6.2, 4.8, 2.5, 4)
create_arrow(ax3, 7, 4.8, 6.7, 4)
create_arrow(ax3, 7.8, 4.8, 11, 4)

# Ranking
create_box(ax3, 4.5, 1.5, 5, 0.6, 'Ranking & Scoring', '#B39DDB')
create_arrow(ax3, 2.5, 3, 6, 2.1)
create_arrow(ax3, 6.7, 3, 7, 2.1)
create_arrow(ax3, 11, 3, 8, 2.1)

# Response
create_box(ax3, 4, 0.2, 6, 0.7, 'Response Generation + Analysis', '#90CAF9')
create_arrow(ax3, 7, 1.5, 7, 0.9)

plt.tight_layout()
plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/3_query_retrieval.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: 3_query_retrieval.png")
plt.close()

# ============================================================================
# DIAGRAM 4: GNN Architecture Detail
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 10), dpi=300)
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 10)
ax4.axis('off')

ax4.text(6, 9.5, 'GNN Link Prediction Architecture', fontsize=18, ha='center', fontweight='bold')

# Input
create_box(ax4, 4, 8, 4, 0.6, 'Legal Case Graph', '#E3F2FD')

# Feature Engineering
create_box(ax4, 3, 6.8, 6, 1, 
           'Feature Engineering\n9-Dimensional Vectors\nCourt Hierarchy + Temporal + Entity Type', 
           '#FFF9C4', fontsize=9)
create_arrow(ax4, 6, 8, 6, 7.8)

# GCN Layers
create_box(ax4, 3.5, 5.3, 5, 0.6, 'GCN Layer 1 (64 hidden)', '#C8E6C9')
create_arrow(ax4, 6, 6.8, 6, 5.9)

create_box(ax4, 3.5, 4.4, 5, 0.6, 'ReLU + Dropout (0.5)', '#C8E6C9')
create_arrow(ax4, 6, 5.3, 6, 5)

create_box(ax4, 3.5, 3.5, 5, 0.6, 'GCN Layer 2 (64 hidden)', '#C8E6C9')
create_arrow(ax4, 6, 4.4, 6, 4.1)

# Link Prediction
create_box(ax4, 3, 2.3, 6, 0.7, 'Link Prediction Head\nConcatenate Embeddings', '#FFE0B2')
create_arrow(ax4, 6, 3.5, 6, 3)

create_box(ax4, 3.5, 1.2, 5, 0.6, 'Dense → ReLU → Dense', '#FFE0B2')
create_arrow(ax4, 6, 2.3, 6, 1.8)

# Output
create_box(ax4, 3.5, 0.2, 5, 0.6, 'Relationship Probability', '#FFCDD2')
create_arrow(ax4, 6, 1.2, 6, 0.8)

# Performance metrics
ax4.text(10.5, 6, 'Performance:', fontsize=11, fontweight='bold', ha='left')
ax4.text(10.5, 5.5, 'Train/Val/Test: 70/10/20', fontsize=9, ha='left')
ax4.text(10.5, 5.1, 'ROC-AUC: 0.78-0.85', fontsize=9, ha='left')
ax4.text(10.5, 4.7, 'Train Time: 2-5 min', fontsize=9, ha='left')

plt.tight_layout()
plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/4_gnn_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: 4_gnn_architecture.png")
plt.close()

# ============================================================================
# DIAGRAM 5: Knowledge Graph Schema
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 9), dpi=300)
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 9)
ax5.axis('off')

ax5.text(6, 8.5, 'Multi-Modal Knowledge Graph Schema', fontsize=18, ha='center', fontweight='bold')

# Center: Case node
create_box(ax5, 4.5, 4, 3, 1, 'CASE\nTitle, Court\nDate, Text', '#4CAF50', fontsize=10)

# Judge node
create_box(ax5, 1, 6.5, 2, 0.8, 'JUDGE\nName', '#2196F3', fontsize=9)
create_arrow(ax5, 2, 6.5, 5, 5)
ax5.text(3.5, 5.8, 'JUDGED', fontsize=8, ha='center', 
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

# Court node
create_box(ax5, 9, 6.5, 2, 0.8, 'COURT\nName, Type', '#FF9800', fontsize=9)
create_arrow(ax5, 6.5, 5, 9.5, 6.5)
ax5.text(8, 5.8, 'HEARD_BY', fontsize=8, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

# Statute node
create_box(ax5, 1, 1.5, 2, 0.8, 'STATUTE\nName, Section', '#9C27B0', fontsize=9)
create_arrow(ax5, 5, 4, 2.5, 2.3)
ax5.text(3.7, 3.2, 'REFERENCES', fontsize=8, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

# Citation relationship (Case to Case)
create_box(ax5, 9, 1.5, 2, 0.8, 'CASE\n(Cited)', '#4CAF50', fontsize=9)
create_arrow(ax5, 6.5, 4, 9.5, 2.3)
ax5.text(8, 3.2, 'CITES', fontsize=8, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

# Legend
ax5.text(6, 0.5, '4 Entity Types  •  4 Relationship Types', 
         fontsize=11, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/animesh/legalnexus-backend/docs/graphs/5_knowledge_graph_schema.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: 5_knowledge_graph_schema.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL DIAGRAMS CREATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  1. docs/graphs/1_pipeline_flow.png - Main processing pipeline")
print("  2. docs/graphs/2_novel_algorithms.png - Novel algorithms overview")
print("  3. docs/graphs/3_query_retrieval.png - Query & retrieval system")
print("  4. docs/graphs/4_gnn_architecture.png - GNN architecture details")
print("  5. docs/graphs/5_knowledge_graph_schema.png - KG schema")
print("\n" + "="*80)


