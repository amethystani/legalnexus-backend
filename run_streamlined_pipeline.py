"""
Streamlined Legal Nexus Pipeline

Skips expensive multi-agent graph construction.
Focuses on: Fast Embeddings → HGCN Training → Search
"""

import subprocess
import sys

def run(cmd, desc):
    print(f"\n{'='*80}")
    print(f"STEP: {desc}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        sys.exit(1)
    print(f"✅ Completed: {desc}")

def main():
    print("STREAMLINED LEGAL NEXUS PIPELINE")
    print("="*80)
    
    # Step 1: Generate Embeddings (Fast!)
    run(
        "python generate_embeddings_streamlined.py",
        "Generate Jina Embeddings for All Cases"
    )
    
    # Step 2: Load Basic Cases to Neo4j (Optional, quick)
    run(
        "python load_cases_to_neo4j.py",
        "Load Case Nodes to Neo4j (No Multi-Agent)"
    )
    
    # Step 3: Extract/Create Citation Network
    run(
        "python extract_citation_network.py",
        "Extract Citation Network from Neo4j/Toulmin"
    )
    
    # Step 4: Train HGCN
    run(
        "python train_hyperbolic.py",
        "Train Hyperbolic GNN"
    )
    
    # Step 5: Visualize
    run(
        "python visualize_poincare.py",
        "Generate Visualizations"
    )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext: Restart Streamlit to see results")

if __name__ == "__main__":
    main()
