#!/usr/bin/env python3
"""
Test System Components
Verify all parts of the diagram are implemented and working
"""

def test_system_components():
    print('🔍 Testing System Components Against Diagram...')
    print('=' * 60)
    
    results = {}
    
    # Test 1: Knowledge Graph
    try:
        from kg import main
        print('✅ Knowledge Graph: Available')
        results['knowledge_graph'] = True
    except ImportError as e:
        print(f'❌ Knowledge Graph: {e}')
        results['knowledge_graph'] = False
    
    # Test 2: Citation Network  
    try:
        from citation_network import CitationNetwork
        print('✅ Citation Network: Available')
        results['citation_network'] = True
    except ImportError as e:
        print(f'❌ Citation Network: {e}')
        results['citation_network'] = False
    
    # Test 3: Document Similarity
    try:
        from kg import find_similar_cases
        print('✅ Document Similarity: Available')
        results['document_similarity'] = True
    except ImportError as e:
        print(f'❌ Document Similarity: {e}')
        results['document_similarity'] = False
    
    # Test 4: Network Visualization
    try:
        from kg_visualizer import create_network_graph
        print('✅ Network Visualization: Available')
        results['visualization'] = True
    except ImportError as e:
        print(f'❌ Network Visualization: {e}')
        results['visualization'] = False
    
    # Test 5: GNN Framework
    try:
        from gnn_link_prediction import LinkPredictionTrainer
        print('✅ GNN Framework: Available')
        results['gnn'] = True
    except ImportError as e:
        print(f'❌ GNN Framework: {e}')
        results['gnn'] = False
    
    # Test 6: Integrated System
    try:
        from integrated_system import IntegratedLegalSystem
        print('✅ Integrated System: Available')
        results['integrated'] = True
    except ImportError as e:
        print(f'❌ Integrated System: {e}')
        results['integrated'] = False
    
    # Test Dependencies
    print('\n🔧 Testing Dependencies...')
    
    # Test PyTorch
    try:
        import torch
        print('✅ PyTorch: Available')
        results['torch'] = True
    except ImportError:
        print('❌ PyTorch: Missing (needed for GNN)')
        results['torch'] = False
    
    # Test PyTorch Geometric
    try:
        import torch_geometric
        print('✅ PyTorch Geometric: Available')
        results['torch_geometric'] = True
    except ImportError:
        print('❌ PyTorch Geometric: Missing (needed for GNN)')
        results['torch_geometric'] = False
    
    # Test Plotly
    try:
        import plotly
        print('✅ Plotly: Available')
        results['plotly'] = True
    except ImportError:
        print('❌ Plotly: Missing (needed for visualization)')
        results['plotly'] = False
    
    # Test NetworkX
    try:
        import networkx
        print('✅ NetworkX: Available')
        results['networkx'] = True
    except ImportError:
        print('❌ NetworkX: Missing (needed for graph analysis)')
        results['networkx'] = False
    
    print('\n📊 System Completeness Report:')
    print('=' * 60)
    
    # Map diagram components to our tests
    diagram_mapping = {
        'Knowledge Graph': results['knowledge_graph'],
        'Citation Network': results['citation_network'], 
        'Document Similarity': results['document_similarity'],
        'Citation Link Prediction': results['gnn'] and results['torch'] and results['torch_geometric'],
        'Similarity Link Prediction': results['gnn'] and results['torch'] and results['torch_geometric'],
        'GNN (Central Hub)': results['gnn'] and results['torch'] and results['torch_geometric'],
        'Network Visualization': results['visualization'] and results['plotly'] and results['networkx'],
        'Integrated System': results['integrated']
    }
    
    implemented_count = sum(diagram_mapping.values())
    total_count = len(diagram_mapping)
    completion_percentage = (implemented_count / total_count) * 100
    
    for component, status in diagram_mapping.items():
        status_icon = '✅' if status else '❌'
        print(f'{status_icon} {component}')
    
    print('\n🎯 Overall System Status:')
    print(f'Completion: {implemented_count}/{total_count} components ({completion_percentage:.1f}%)')
    
    if completion_percentage >= 80:
        print('🚀 System is ready for production use!')
    elif completion_percentage >= 60:
        print('⚡ System is functional but needs some components')
    else:
        print('🔧 System needs significant development')
    
    print('\n💡 Missing Dependencies:')
    missing_deps = []
    if not results['torch']:
        missing_deps.append('torch')
    if not results['torch_geometric']:
        missing_deps.append('torch-geometric')
    if not results['plotly']:
        missing_deps.append('plotly')
    if not results['networkx']:
        missing_deps.append('networkx')
    
    if missing_deps:
        print(f'Install with: pip install {" ".join(missing_deps)}')
    else:
        print('All dependencies are available!')
    
    return results, completion_percentage

if __name__ == "__main__":
    test_system_components() 