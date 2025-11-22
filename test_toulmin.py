"""
Test script for Toulmin Argumentation Mining

Demonstrates argument-based retrieval vs semantic similarity.
"""

import os
import sys
from hybrid_case_search import NovelHybridSearchSystem
from toulmin_extractor import ToulminExtractor
from argument_chain_traversal import ArgumentGraph


def test_toulmin_extraction():
    """Test Toulmin extraction on sample cases"""
    print("="*80)
    print("TOULMIN ARGUMENTATION MINING - TEST")
    print("="*80)
    
    # Initialize system
    print("\n1. Initializing system...")
    system = NovelHybridSearchSystem()
    
    # Extract argument structures
    print("\n2. Extracting Toulmin structures from cases...")
    extractor = ToulminExtractor(system.llm)
    
    # Prepare case data: (case_id, case_text)
    case_samples = [
        (doc.metadata.get('id'), doc.page_content)
        for doc in system.cases_data[:10]  # First 10 cases for demo
    ]
    
    structures = extractor.extract_batch(case_samples, max_cases=10)
    
    # Build argument graph
    print("\n3. Building Argument Graph...")
    arg_graph = ArgumentGraph()
    
    for case_id, structure in structures.items():
        # Find corresponding document
        case_doc = next((doc for doc in system.cases_data 
                        if doc.metadata.get('id') == case_id), None)
        if case_doc:
            arg_graph.add_case(case_id, structure, case_doc)
    
    stats = arg_graph.get_stats()
    print(f"\n  ✓ Graph Stats:")
    print(f"     - Cases: {stats['num_cases']}")
    print(f"     - Total Nodes: {stats['total_nodes']}")
    print(f"     - Claims: {stats['claims']}")
    print(f"     - Warrants: {stats['warrants']}")
    
    # Test argument chain retrieval
    print("\n4. Testing Argument Chain Retrieval...")
    test_queries = [
        "The defendant is liable for damages",
        "Application for bail should be granted",
        "The case should be dismissed"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = arg_graph.find_argument_chain(query, max_depth=3)
        
        if results:
            print(f"  Top results:")
            for case_id, strength in results[:3]:
                print(f"    - {case_id}: chain strength = {strength:.2f}")
                
                # Show argument explanation
                explanation = arg_graph.explain_chain(case_id)
                print(f"\n{explanation}")
        else:
            print("  No argument chains found.")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_toulmin_extraction()
