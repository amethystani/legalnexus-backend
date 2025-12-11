"""
Run Complete Legal Nexus Pipeline

Orchestrates the entire data processing workflow:
1. Load Data & Generate Embeddings (Jina v3)
2. Build Multi-Agent Knowledge Graph (Neo4j)
3. Extract Citation Network
4. Train Hyperbolic GNN
"""

import subprocess
import sys
import time

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"🚀 Running: {command}\n")
    
    start_time = time.time()
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        elapsed = time.time() - start_time
        print(f"\n✅ Step completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"\n❌ Error running {command}: {e}")
        sys.exit(1)

def main():
    print("STARTING LEGAL NEXUS PIPELINE")
    
    # 1. Generate Embeddings
    # This loads CSVs, generates Jina embeddings, and saves cache
    run_command("python generate_embeddings.py", "Generate Embeddings (Jina v3)")
    
    # 2. Build Knowledge Graph
    # This reads CSVs, runs Multi-Agent Swarm, and populates Neo4j
    run_command("python build_knowledge_graph.py", "Build Multi-Agent Knowledge Graph (Neo4j)")
    
    # 3. Extract Citation Network
    # This reads from Neo4j and prepares data for HGCN
    run_command("python extract_citation_network.py", "Extract Citation Network for HGCN")
    
    # 4. Train Hyperbolic GNN
    # This trains the model using the extracted network and embeddings
    run_command("python train_hyperbolic.py", "Train Hyperbolic GNN")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print("You can now restart the Streamlit app to see the results.")

if __name__ == "__main__":
    main()
