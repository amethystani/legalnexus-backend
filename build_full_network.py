#!/usr/bin/env python3
"""
Build full citation network from all available cases.

Steps:
1. Load all cases from data directory
2. Run Multi-Agent Swarm (Nash Mode) on each case
3. Extract citations and build graph
4. Save to data/citation_network_full.pkl
"""

import os
import json
import pickle
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Set
import glob

# Import our system
from multi_agent_swarm import MultiAgentSwarm, Citation

def load_all_cases(data_dir: str = "data/legal_cases") -> List[Dict]:
    """Load all case JSON files."""
    cases = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Creating {data_dir} and generating synthetic cases for testing...")
        os.makedirs(data_dir, exist_ok=True)
        # Generate some synthetic cases if none exist
        from create_synthetic_graph import create_synthetic_data
        create_synthetic_data(num_cases=50, output_dir=data_dir)
    
    files = glob.glob(f"{data_dir}/*.json")
    
    if not files:
        print(f"No cases found in {data_dir}. Generating synthetic cases...")
        from create_synthetic_graph import create_synthetic_data
        create_synthetic_data(num_cases=50, output_dir=data_dir)
        files = glob.glob(f"{data_dir}/*.json")
        
    print(f"Found {len(files)} case files.")
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                case = json.load(f)
                cases.append(case)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    return cases

def build_citation_network():
    """Run extraction pipeline on all cases."""
    print("="*80)
    print("BUILDING FULL CITATION NETWORK")
    print("="*80)
    
    # 1. Load cases
    cases = load_all_cases()
    if not cases:
        print("No cases found! Please add case JSONs to data/legal_cases/")
        return
    
    # 2. Initialize Swarm
    print("\nInitializing Multi-Agent Swarm (Nash Mode)...")
    swarm = MultiAgentSwarm()
    
    # 3. Process all cases
    all_citations = []
    all_case_ids = {c.get('id', f"CASE_{i}") for i, c in enumerate(cases)}
    
    print(f"\nProcessing {len(cases)} cases...")
    for i, case in enumerate(tqdm(cases)):
        case_id = case.get('id', f"CASE_{i}")
        case_text = case.get('text', '') or case.get('content', '')
        
        if not case_text:
            continue
            
        # Run Nash Equilibrium extraction
        # We use a simplified version for speed if needed, but here we go full Nash
        try:
            if swarm.use_nash:
                result = swarm.process_case_with_nash_equilibrium(
                    case_text, 
                    case_id, 
                    all_case_ids
                )
                citations = result['citations']
            else:
                citations, _ = swarm.process_case_with_debate(
                    case_text, 
                    case_id, 
                    all_case_ids
                )
            
            all_citations.extend(citations)
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            continue
            
    # 4. Build Graph
    print(f"\nExtracted {len(all_citations)} total citations.")
    
    G = nx.DiGraph()
    G.add_nodes_from(all_case_ids)
    
    edges = []
    for cit in all_citations:
        G.add_edge(cit.source_id, cit.target_id, type=cit.edge_type.value)
        edges.append({
            'source': cit.source_id,
            'target': cit.target_id,
            'type': cit.edge_type.value,
            'confidence': cit.confidence
        })
        
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 5. Save
    output_path = 'data/citation_network_full.pkl'
    data = {
        'case_ids': list(all_case_ids),
        'edges': edges,
        'stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G)
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved full network to {output_path}")
    return output_path

if __name__ == "__main__":
    build_citation_network()
