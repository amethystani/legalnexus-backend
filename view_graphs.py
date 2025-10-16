"""
Simple viewer to display all generated documentation graphs
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_all_graphs():
    """Display all generated graphs in a grid"""
    graph_dir = 'docs/graphs'
    
    # Get all PNG files
    graphs = sorted([f for f in os.listdir(graph_dir) if f.endswith('.png')])
    
    if not graphs:
        print("No graphs found in docs/graphs/")
        return
    
    # Create figure with subplots
    n_graphs = len(graphs)
    cols = 2
    rows = (n_graphs + 1) // 2
    
    fig = plt.figure(figsize=(20, rows * 8))
    fig.suptitle('LegalNexus Documentation Graphs', fontsize=24, fontweight='bold')
    
    for i, graph_file in enumerate(graphs, 1):
        ax = plt.subplot(rows, cols, i)
        
        # Load and display image
        img_path = os.path.join(graph_dir, graph_file)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Add title
        title = graph_file.replace('.png', '').replace('_', ' ').title()
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Displayed {n_graphs} graphs")
    print(f"✓ Location: {graph_dir}/")
    print("\nGraph files:")
    for graph in graphs:
        file_size = os.path.getsize(os.path.join(graph_dir, graph)) / 1024
        print(f"  - {graph} ({file_size:.1f} KB)")


def view_single_graph(graph_number):
    """Display a single graph by number (1-8)"""
    graph_dir = 'docs/graphs'
    graph_file = f"{graph_number}_*.png"
    
    import glob
    matching = glob.glob(os.path.join(graph_dir, f"{graph_number}_*.png"))
    
    if not matching:
        print(f"Graph {graph_number} not found")
        return
    
    img_path = matching[0]
    img = mpimg.imread(img_path)
    
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(img_path).replace('.png', '').replace('_', ' ').title(),
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Displayed: {os.path.basename(img_path)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            graph_num = int(sys.argv[1])
            view_single_graph(graph_num)
        except ValueError:
            print("Usage: python view_graphs.py [graph_number]")
            print("Example: python view_graphs.py 1")
    else:
        view_all_graphs()


