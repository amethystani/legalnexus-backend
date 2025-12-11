#!/usr/bin/env python3
"""
Test GNN Training Pipeline
Test the complete GNN training process to verify it works
"""

import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from gnn_link_prediction import LinkPredictionTrainer

def test_gnn_training():
    print('Testing GNN Training Pipeline...')
    print('=' * 60)
    
    load_dotenv()
    
    try:
        # Connect to Neo4j
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        print('Connected to Neo4j')
        
        # Initialize trainer
        trainer = LinkPredictionTrainer(graph)
        print('Trainer initialized')
        
        # Prepare data
        print('\n Preparing data...')
        data = trainer.prepare_data()
        
        if data is None:
            print(' Data preparation failed')
            return False
        
        print(f'Data prepared: {data}')
        
        # Test training with minimal epochs
        print('\n Testing training (5 epochs)...')
        
        # Mock streamlit functions for testing
        class MockStreamlit:
            @staticmethod
            def info(msg):
                print(f'INFO: {msg}')
            @staticmethod
            def error(msg):
                print(f'ERROR: {msg}')
            @staticmethod
            def success(msg):
                print(f'SUCCESS: {msg}')
            @staticmethod
            def progress(val):
                return MockProgress()
            @staticmethod
            def empty():
                return MockEmpty()
        
        class MockProgress:
            def progress(self, val):
                print(f'Progress: {val:.1%}')
        
        class MockEmpty:
            def text(self, msg):
                print(f'STATUS: {msg}')
        
        # Replace streamlit temporarily
        import gnn_link_prediction
        original_st = gnn_link_prediction.st
        gnn_link_prediction.st = MockStreamlit()
        
        try:
            history = trainer.train_model(data, epochs=5)
            
            if history:
                print(' Training completed successfully!')
                print(f'Training loss history: {history.get("train_loss", [])}')
                return True
            else:
                print(' Training failed - no history returned')
                return False
                
        finally:
            # Restore streamlit
            gnn_link_prediction.st = original_st
        
    except Exception as e:
        print(f' Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gnn_training()
    if success:
        print('\n GNN training test passed!')
    else:
        print('\n GNN training test failed!') 