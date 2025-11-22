"""
Comprehensive Test: All Three Research Features

Tests:
1. Toulmin Argumentation Mining
2. Temporal Precedent Decay
3. Counterfactual "What-If" Analysis
"""

import os
import sys
from hybrid_case_search import NovelHybridSearchSystem
from toulmin_extractor import ToulminExtractor
from argument_chain_traversal import ArgumentGraph
from temporal_scorer import TemporalScorer, extract_year_from_text
from counterfactual_engine import CounterfactualEngine


def demonstrate_all_features():
    """Comprehensive demonstration of all three research features"""
    
    print("\n" + "="*80)
    print("RESEARCH-GRADE LEGAL AI: THREE NOVEL FEATURES")
    print("="*80)
    
    # Initialize
    print("\n📦 Initializing system...")
    system = NovelHybridSearchSystem()
    
    print(f"✓ Loaded {len(system.cases_data)} cases")
    
    # ============================================================================
    # FEATURE 1: TOULMIN ARGUMENTATION
    # ============================================================================
    print("\n" + "="*80)
    print("FEATURE 1: TOULMIN ARGUMENTATION MINING")
    print("="*80)
    print("\n Description: Extract logical argument structures, not just keywords.\n")
    
    extractor = ToulminExtractor(system.llm)
    arg_graph = ArgumentGraph()
    
    #  Extract from 5 cases
    print("Extracting argument structures...")
    case_samples = [
        (doc.metadata.get('id'), doc.page_content)
        for doc in system.cases_data[:5]
    ]
    
    structures = extractor.extract_batch(case_samples, max_cases=5)
    
    for case_id, structure in structures.items():
        case_doc = next((doc for doc in system.cases_data 
                        if doc.metadata.get('id') == case_id), None)
        if case_doc:
            arg_graph.add_case(case_id, structure, case_doc)
    
    stats = arg_graph.get_stats()
    print(f"\n✓ Argument Graph Built:")
    print(f"   - Cases: {stats['num_cases']}")
    print(f"   - Claims: {stats['claims']}")
    print(f"   - Warrants: {stats['warrants']}")
    
    # Test query
    test_query = "Application for bail should be granted"
    print(f"\n🔍 Query: '{test_query}'")
    results = arg_graph.find_argument_chain(test_query, max_depth=3)
    
    if results:
        print(f"✓ Found {len(results)} argument chains")
        case_id, strength = results[0]
        print(f"\n Top result: {case_id} (strength: {strength:.2f})")
        print(arg_graph.explain_chain(case_id)[:300] + "...")
    
    # ============================================================================
    # FEATURE 2: TEMPORAL PRECEDENT DECAY
    # ============================================================================
    print("\n" + "="*80)
    print("FEATURE 2: TEMPORAL PRECEDENT DECAY")
    print("="*80)
    print("\nDescription: Old cases get lower scores unless frequently cited.\n")
    
    temporal_scorer = TemporalScorer()
    
    # Extract temporal metadata
    print("Extracting temporal metadata...")
    for doc in system.cases_data[:20]:
        case_id = doc.metadata.get('id')
        temporal_scorer.add_case(case_id, doc.page_content)
    
    stats = temporal_scorer.get_stats()
    if stats:
        print(f"\n✓ Temporal Analysis:")
        print(f"   - Total cases: {stats['total_cases']}")
        print(f"   - Date range: {stats['earliest_year']} - {stats['latest_year']}")
        print(f"   - Average age: {stats['avg_age']:.1f} years")
    
    # Show scores
    print("\n📊 Temporal Scores (sample):")
    for doc in system.cases_data[:5]:
        case_id = doc.metadata.get('id')
        year = temporal_scorer.case_dates.get(case_id)
        score = temporal_scorer.score(case_id)
        print(f"   - {case_id}: Year {year}, Score {score:.3f}")
    
    # ============================================================================
    # FEATURE 3: COUNTERFACTUAL "WHAT-IF" ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("FEATURE 3: COUNTERFACTUAL \"WHAT-IF\" ENGINE")
    print("="*80)
    print("\nDescription: Identify which facts are 'pivot points' that change outcomes.\n")
    
    counterfactual = CounterfactualEngine(system.llm, system)
    
    test_query = "I hit a pedestrian but it was dark"
    
    print(f"Testing query: '{test_query}'")
    pivot_analysis = counterfactual.identify_pivot_points(test_query, top_k=3)
    
    if pivot_analysis:
        print(f"\n✓ Analyzed {len(pivot_analysis)} facts")
        
        # Show pivot points
        pivots = [p for p in pivot_analysis if p['is_pivot']]
        if pivots:
            print(f"\n⚠️ Found {len(pivots)} PIVOT POINTS:")
            for pivot in pivots:
                print(f"   - '{pivot['fact']}' → '{pivot['negation']}'")
                print(f"     Impact: {pivot['impact_score']:.0%}")
        
        # Generate recommendations
        recommendations = counterfactual.generate_recommendations(pivot_analysis)
        print(recommendations)
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("SUMMARY: RESEARCH CONTRIBUTIONS")
    print("="*80)
    
    print("""
1. TOULMIN ARGUMENTATION MINING
   → Moves beyond keyword matching to logical argument chains
   → Research Paper: "Neuro-Symbolic Extraction of Toulmin Structures"
   
2. TEMPORAL PRECEDENT DECAY
   → Addresses concept drift in legal corpus
   → Research Paper: "Modeling Precedent Decay using Temporal GNNs"
   
3. COUNTERFACTUAL "WHAT-IF" ENGINE
   → Provides interpretability via fact perturbation
   → Research Paper: "Identifying Legal Pivot Points via Counterfactual RAG"
    """)
    
    print("="*80)


if __name__ == "__main__":
    demonstrate_all_features()
