#!/usr/bin/env python3
"""
Test Integrated System Components
Quick test to verify all components load correctly
"""

def test_integrated_system():
    print('🔍 Testing Integrated System Components...')
    print('=' * 60)
    
    try:
        from integrated_system import IntegratedLegalSystem
        print('✅ IntegratedLegalSystem class imported successfully')
        
        # Test component availability
        from integrated_system import VISUALIZATION_AVAILABLE, CITATION_AVAILABLE, GNN_PREDICTION_AVAILABLE
        
        print(f'📊 Visualization Available: {VISUALIZATION_AVAILABLE}')
        print(f'🔗 Citation Analysis Available: {CITATION_AVAILABLE}')
        print(f'🤖 GNN Prediction Available: {GNN_PREDICTION_AVAILABLE}')
        
        if VISUALIZATION_AVAILABLE and CITATION_AVAILABLE and GNN_PREDICTION_AVAILABLE:
            print('🎉 All components are available!')
            return True
        else:
            missing = []
            if not VISUALIZATION_AVAILABLE: missing.append('Visualization')
            if not CITATION_AVAILABLE: missing.append('Citation Analysis')  
            if not GNN_PREDICTION_AVAILABLE: missing.append('GNN Prediction')
            print(f'⚠️ Missing components: {", ".join(missing)}')
            return False
            
    except Exception as e:
        print(f'❌ Error testing integrated system: {e}')
        return False

if __name__ == "__main__":
    success = test_integrated_system()
    if success:
        print('\n🚀 Integrated system is ready to run!')
        print('Use: streamlit run integrated_system.py')
    else:
        print('\n🔧 Please install missing dependencies') 