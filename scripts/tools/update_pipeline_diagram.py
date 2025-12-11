#!/usr/bin/env python3
"""
Update Pipeline Diagram with GNN Link Prediction
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def generate_pipeline_diagram():
    """Generate detailed pipeline flow diagram with GNN stage"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(5, 13.5, 'LegalNexus Processing Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    stages = [
        {
            'y': 12, 'title': 'Stage 1: Data Ingestion',
            'items': ['Load JSON/PDF documents', 'Validate schema', 'Extract metadata']
        },
        {
            'y': 10.2, 'title': 'Stage 2: Text Processing',
            'items': ['Recursive chunking (300/30)', 'Entity extraction', 'Clean & normalize']
        },
        {
            'y': 8.4, 'title': 'Stage 3: Embedding Generation',
            'items': ['Gemini API call', 'Vector generation (768D)', 'Cache embeddings']
        },
        {
            'y': 6.6, 'title': 'Stage 4: Graph Creation',
            'items': ['Create Case nodes', 'Link entities', 'Build relationships']
        },
        {
            'y': 4.8, 'title': 'Stage 4.5: GNN Link Prediction',
            'items': ['9-dim feature engineering', 'GCN training (ROC-AUC: 0.78-0.85)', 'Predict missing relationships']
        },
        {
            'y': 3, 'title': 'Stage 5: Index Creation',
            'items': ['Vector index setup', 'Keyword index', 'Hybrid search config']
        },
        {
            'y': 1.2, 'title': 'Stage 6: Query & Retrieval',
            'items': ['Semantic search', 'Cypher queries', 'Similarity ranking']
        },
        {
            'y': -0.6, 'title': 'Stage 7: Response Generation',
            'items': ['LLM analysis', 'Result formatting', 'Visualization']
        }
    ]
    
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#BC6C25', '#98D8C8', '#C7CEEA', '#B8E6B8']
    
    for i, stage in enumerate(stages):
        # Draw stage box
        box = FancyBboxPatch((1, stage['y']), 8, 1.4, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#333', facecolor=colors_list[i], 
                            linewidth=2, alpha=0.7)
        ax.add_patch(box)
        
        # Title
        ax.text(5, stage['y'] + 1.15, stage['title'], 
                fontsize=11, fontweight='bold', ha='center')
        
        # Items
        y_offset = stage['y'] + 0.85
        for item in stage['items']:
            ax.text(5, y_offset, f"• {item}", fontsize=9, ha='center')
            y_offset -= 0.25
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((5, stage['y']), (5, stage['y'] - 0.4),
                                   arrowstyle='->', mutation_scale=20, 
                                   lw=2, color='#333')
            ax.add_patch(arrow)
    
    # Add timing annotations
    timings = ['~2-5s', '~1-3s', '~3-10s', '~2-5s', '~2-5min', '~1-2s', '~1-5s', '~2-8s']
    for i, (stage, timing) in enumerate(zip(stages, timings)):
        ax.text(9.5, stage['y'] + 0.7, timing, 
                fontsize=8, style='italic', ha='left', color='#666')
    
    plt.tight_layout()
    plt.savefig('docs/graphs/2_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Updated: docs/graphs/2_pipeline_diagram.png")
    plt.close()

if __name__ == "__main__":
    generate_pipeline_diagram()
    print("\n✨ Pipeline diagram updated with GNN Link Prediction stage!")

