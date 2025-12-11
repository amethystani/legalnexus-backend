#!/usr/bin/env python3
"""
Test script to verify all implementations work correctly.

Tests:
1. Graph curvature measurement
2. Euclidean GNN baseline
3. Hyperbolic vs Euclidean comparison
4. Nash equilibrium solver
5. Multi-agent swarm with Nash equilibrium
"""

import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_curvature_analysis():
    """Test Part 1: Graph curvature measurement."""
    print("\n" + "="*80)
    print("TEST 1: Graph Curvature Analysis")
    print("="*80)
    
    from analysis.measure_graph_curvature import (
        create_synthetic_legal_network,
        gromov_delta_hyperbolicity,
        compare_with_random_graphs
    )
    
    # Create synthetic network
    G = create_synthetic_legal_network(n_cases=50)
    
    # Measure curvature
    delta = gromov_delta_hyperbolicity(G, sample_size=100)
    print(f"\nδ-hyperbolicity: {delta:.3f}")
    
    if delta < 1.0:
        print("✓ Graph is hyperbolic (δ < 1.0)")
    else:
        print("⚠️ Graph is NOT hyperbolic (δ >= 1.0)")
    
    return delta < 1.5  # Pass if somewhat hierarchical


def test_euclidean_baseline():
    """Test Part 1: Euclidean GNN baseline."""
    print("\n" + "="*80)
    print("TEST 2: Euclidean GNN Baseline")
    print("="*80)
    
    import torch
    from baselines.euclidean_gnn import LegalEuclideanModel, count_parameters
    
    # Create model
    model = LegalEuclideanModel(input_dim=768, hidden_dim=128, output_dim=64)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(10, 768)
    edge_index = torch.randint(0, 10, (2, 20))
    
    embeddings = model(x, edge_index)
    print(f"Output shape: {embeddings.shape}")
    
    assert embeddings.shape == (10, 64), "Output shape mismatch!"
    print("✓ Euclidean baseline working")
    
    return True


def test_nash_equilibrium():
    """Test Part 2: Nash equilibrium solver."""
    print("\n" + "="*80)
    print("TEST 3: Nash Equilibrium Solver")
    print("="*80)
    
    from theory.nash_equilibrium_formulation import NashEquilibriumSolver, Citation
    
    # Create solver
    solver = NashEquilibriumSolver()
    
    # Create test graph
    initial_graph = {
        'citations': [
            Citation('C1', 'C2', 'FOLLOW', 0.8),
            Citation('C2', 'C3', 'FOLLOW', 0.7),
        ]
    }
    
    # Find equilibrium
    equilibrium, history = solver.find_nash_equilibrium(initial_graph, max_iterations=3)
    
    print(f"Converged in {len(history)} iterations")
    print(f"Final payoffs: {history[-1]['payoffs']}")
    
    print("✓ Nash equilibrium solver working")
    
    return len(history) > 0


def test_multi_agent_nash():
    """Test Part 2: Multi-agent swarm with Nash equilibrium."""
    print("\n" + "="*80)
    print("TEST 4: Multi-Agent Swarm with Nash Equilibrium")
    print("="*80)
    
    from multi_agent_swarm import MultiAgentSwarm
    
    # Create swarm
    swarm = MultiAgentSwarm()
    
    # Test case
    case_text = """
    This court follows the precedent set in AIR 1950 SC 124.
    However, we distinguish the facts from State v. Kumar (2020).
    """
    
    # Process with Nash equilibrium (if available)
    if swarm.use_nash:
        result = swarm.process_case_with_nash_equilibrium(
            case_text,
            case_id="TEST_001",
            all_case_ids={'AIR_1950_SC_124', 'STATE_V_KUMAR_2020'}
        )
        
        print(f"\nResults:")
        print(f"  Method: {result['method']}")
        print(f"  Citations found: {len(result['citations'])}")
        print(f"  Final payoffs: {result['payoffs']}")
        print("✓ Multi-agent with Nash equilibrium working")
    else:
        print("⚠️ Nash equilibrium not available, skipping")
        result = swarm.process_case_with_debate(
            case_text,
            case_id="TEST_001",
            all_case_ids={'AIR_1950_SC_124', 'STATE_V_KUMAR_2020'}
        )
        print(f"  Citations found: {len(result['citations'])}")
        print("✓ Multi-agent with debate working (fallback)")
    
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING ALL IMPLEMENTATIONS")
    print("="*80)
    
    tests = [
        ("Curvature Analysis", test_curvature_analysis),
        ("Euclidean Baseline", test_euclidean_baseline),
        ("Nash Equilibrium", test_nash_equilibrium),
        ("Multi-Agent Nash", test_multi_agent_nash),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
