import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from kg import load_legal_data, load_test_cases

class MockGraph:
    """Mock Neo4j graph for testing"""
    
    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        self.queries = []
    
    def query(self, cypher, params=None):
        """Mock query execution"""
        self.queries.append({"cypher": cypher, "params": params})
        
        # Return different mock responses based on the query
        if "MATCH (n) RETURN count(n)" in cypher:
            return [{"node_count": len(self.nodes)}]
        elif "MATCH (c:Case) RETURN count(c)" in cypher:
            return [{"case_count": len([n for n in self.nodes.values() if n.get("label") == "Case"])}]
        elif "MATCH (c:Case)" in cypher and "RETURN c.title" in cypher:
            return [{"title": n.get("title", "Unknown"), "court": n.get("court", "Unknown")} 
                   for n in self.nodes.values() if n.get("label") == "Case"][:5]
        elif "MATCH (j:Judge)" in cypher:
            judges = [n for n in self.nodes.values() if n.get("label") == "Judge"]
            return [{"judge": j.get("name", "Unknown"), "case_count": 1, "cases": ["Some Case"]} 
                   for j in judges][:5]
        elif "MATCH (s:Statute)" in cypher:
            statutes = [n for n in self.nodes.values() if n.get("label") == "Statute"]
            return [{"statute": s.get("name", "Unknown"), "case_count": 1, "cases": ["Some Case"]} 
                   for s in statutes][:5]
        elif "DETACH DELETE" in cypher:
            # Clear the graph
            self.nodes = {}
            self.relationships = {}
            return []
        
        # Default return empty list for other queries
        return []
    
    def add_node(self, label, **properties):
        """Add a node to the mock graph"""
        node_id = len(self.nodes) + 1
        node = {"id": node_id, "label": label, **properties}
        self.nodes[node_id] = node
        return node_id
    
    def add_relationship(self, from_id, to_id, rel_type, **properties):
        """Add a relationship to the mock graph"""
        rel_id = len(self.relationships) + 1
        rel = {
            "id": rel_id,
            "from": from_id,
            "to": to_id,
            "type": rel_type,
            **properties
        }
        self.relationships[rel_id] = rel
        return rel_id
    
    def clear(self):
        """Clear the graph"""
        self.nodes = {}
        self.relationships = {}
        self.queries = []

class TestMockKnowledgeGraph(unittest.TestCase):
    """Test suite using a mock Knowledge Graph"""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once"""
        print("Setting up mock test environment...")
        
        # Create mock graph
        cls.graph = MockGraph()
    
    def setUp(self):
        """Set up before each test"""
        # Clear the graph before each test to avoid interference
        self.graph.clear()
    
    def test_load_test_cases(self):
        """Test loading test cases"""
        print("\n--- Testing: Loading test cases with mock graph ---")
        test_docs = load_test_cases()
        self.assertIsNotNone(test_docs, "Test cases should not be None")
        self.assertTrue(len(test_docs) > 0, "Should find at least one test case")
        
        # Print found test cases
        for doc in test_docs:
            print(f"Found test case: {doc.metadata.get('title', 'Untitled')}")
    
    def test_populate_mock_graph(self):
        """Test populating a mock graph with test cases"""
        print("\n--- Testing: Populating mock graph ---")
        # Ensure graph is clear
        self.graph.clear()
        
        test_docs = load_test_cases()
        self.assertTrue(len(test_docs) > 0, "No test cases found")
        
        # Add nodes and relationships to mock graph
        case_count = 0
        judge_count = 0
        statute_count = 0
        
        for doc in test_docs:
            # Create Case node
            case_id = self.graph.add_node(
                "Case",
                title=doc.metadata.get('title', 'Unknown Case'),
                court=doc.metadata.get('court', 'Unknown Court'),
                date=doc.metadata.get('judgment_date', 'Unknown Date'),
                id=doc.metadata.get('id', f"test_{case_count}"),
                text=doc.page_content[:1000]  # Add first 1000 chars as text
            )
            case_count += 1
            print(f"Added Case node: {doc.metadata.get('title', 'Unknown Case')}")
            
            # Add Judge nodes
            if 'judges' in doc.metadata:
                judges = doc.metadata['judges']
                if isinstance(judges, str):
                    judges = [j.strip() for j in judges.split(',')]
                elif not isinstance(judges, list):
                    judges = []
                    
                for judge in judges:
                    if judge:
                        # Add Judge node
                        judge_id = self.graph.add_node("Judge", name=judge)
                        judge_count += 1
                        
                        # Add relationship
                        self.graph.add_relationship(judge_id, case_id, "JUDGED")
                        print(f"  - Added Judge: {judge}")
            
            # Add Statute nodes
            if 'statutes' in doc.metadata and isinstance(doc.metadata['statutes'], list):
                for statute in doc.metadata['statutes']:
                    if statute:
                        # Add Statute node
                        statute_id = self.graph.add_node("Statute", name=statute)
                        statute_count += 1
                        
                        # Add relationship
                        self.graph.add_relationship(case_id, statute_id, "REFERENCES")
                        print(f"  - Added Statute: {statute}")
        
        # Check node counts
        print(f"Added {case_count} Case nodes")
        print(f"Added {judge_count} Judge nodes")
        print(f"Added {statute_count} Statute nodes")
        
        case_nodes = len([n for n in self.graph.nodes.values() if n.get("label") == "Case"])
        print(f"Found {case_nodes} Case nodes in graph")
        self.assertEqual(case_nodes, case_count, "Case count mismatch")
        
        # Test a simple query
        result = self.graph.query("MATCH (c:Case) RETURN count(c) as case_count")
        self.assertEqual(result[0]["case_count"], case_count, 
                        f"Expected {case_count} cases, got {result[0]['case_count']}")
    
    def test_mock_search_queries(self):
        """Test mock search queries"""
        print("\n--- Testing: Mock search queries ---")
        
        # Clear the graph first
        self.graph.clear()
        self.test_populate_mock_graph()
        
        # Test case search query
        print("Testing case search...")
        case_query = "MATCH (c:Case) WHERE c.title CONTAINS 'evidence' RETURN c.title AS title, c.court AS court LIMIT 5"
        case_results = self.graph.query(case_query)
        print(f"Found {len(case_results)} cases")
        for i, result in enumerate(case_results):
            print(f"{i+1}. {result.get('title', 'Unknown')} - {result.get('court', 'Unknown')}")
        
        # Test judge search query
        print("\nTesting judge search...")
        judge_query = "MATCH (j:Judge)-[:JUDGED]->(c:Case) RETURN j.name AS judge, count(c) AS case_count LIMIT 5"
        judge_results = self.graph.query(judge_query)
        print(f"Found {len(judge_results)} judges")
        for i, result in enumerate(judge_results):
            print(f"{i+1}. {result.get('judge', 'Unknown')} - {result.get('case_count')} cases")
        
        # Test statute search query
        print("\nTesting statute search...")
        statute_query = "MATCH (c:Case)-[:REFERENCES]->(s:Statute) RETURN s.name AS statute, count(c) AS case_count LIMIT 5"
        statute_results = self.graph.query(statute_query)
        print(f"Found {len(statute_results)} statutes")
        for i, result in enumerate(statute_results):
            print(f"{i+1}. {result.get('statute', 'Unknown')} - {result.get('case_count')} cases")

def run_tests():
    """Run mock knowledge graph tests"""
    print("=" * 50)
    print("Running Mock Knowledge Graph Tests")
    print("=" * 50)
    
    # Create a test loader and discover tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMockKnowledgeGraph)
    
    # Run the tests
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    return len(result.errors) == 0 and len(result.failures) == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 