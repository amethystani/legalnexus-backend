import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
import unittest

# Import functionality from kg.py
from kg import load_legal_data, load_test_cases, create_legal_knowledge_graph, find_similar_cases, simple_text_search

class TestKnowledgeGraph(unittest.TestCase):
    """Test suite for the Knowledge Graph functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        print("Setting up test environment...")
        
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials
        cls.neo4j_url = os.getenv("NEO4J_URI")
        cls.neo4j_username = os.getenv("NEO4J_USERNAME") 
        cls.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Check if OpenAI API key is set
        cls.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not cls.openai_api_key:
            print("WARNING: OPENAI_API_KEY not set. Some tests will be skipped.")
        
        # Initialize graph connection if credentials are available
        if cls.neo4j_url and cls.neo4j_username and cls.neo4j_password:
            try:
                cls.graph = Neo4jGraph(
                    url=cls.neo4j_url,
                    username=cls.neo4j_username,
                    password=cls.neo4j_password
                )
                print(f"Connected to Neo4j database: {cls.neo4j_url}")
                
                # Initialize NLP components if API key is available
                if cls.openai_api_key:
                    cls.embeddings = OpenAIEmbeddings()
                    cls.llm = ChatOpenAI(model_name="gpt-4o")
            except Exception as e:
                print(f"Error connecting to Neo4j: {e}")
                cls.graph = None
        else:
            print("Neo4j credentials not found. Database tests will be skipped.")
            cls.graph = None
    
    def setUp(self):
        """Set up before each test"""
        pass
    
    def test_load_test_cases(self):
        """Test if test cases can be loaded correctly"""
        print("\n--- Testing: Loading test cases ---")
        test_docs = load_test_cases()
        self.assertIsNotNone(test_docs, "Test cases should not be None")
        self.assertTrue(len(test_docs) > 0, "Should find at least one test case")
        
        # Verify content of test documents
        for doc in test_docs:
            print(f"Found test case: {doc.metadata.get('title', 'Untitled')}")
            self.assertTrue(len(doc.page_content) > 0, "Document content should not be empty")
            self.assertTrue('id' in doc.metadata, "Document should have ID in metadata")
    
    def test_load_legal_data(self):
        """Test if legal data can be loaded correctly"""
        print("\n--- Testing: Loading legal data ---")
        legal_docs = load_legal_data()
        self.assertIsNotNone(legal_docs, "Legal documents should not be None")
        print(f"Found {len(legal_docs)} legal documents")
        
        # Print details of first few documents for debugging
        for i, doc in enumerate(legal_docs[:3]):
            if i < len(legal_docs):
                print(f"Document {i+1}: {doc.metadata.get('title', 'Untitled')}")
    
    @unittest.skipIf(os.getenv("NEO4J_URI") is None, "Neo4j credentials not set")
    def test_clear_graph(self):
        """Test clearing the graph database"""
        print("\n--- Testing: Clearing graph database ---")
        if self.graph:
            # Clear the database
            cypher = "MATCH (n) DETACH DELETE n;"
            self.graph.query(cypher)
            
            # Verify it's empty
            count_query = "MATCH (n) RETURN count(n) as node_count"
            result = self.graph.query(count_query)
            self.assertEqual(result[0]['node_count'], 0, "Database should be empty after clear")
            print("Graph database cleared successfully")
    
    @unittest.skipIf(os.getenv("NEO4J_URI") is None or os.getenv("OPENAI_API_KEY") is None, 
                     "Neo4j or OpenAI credentials not set")
    def test_create_knowledge_graph(self):
        """Test creating knowledge graph from test cases"""
        print("\n--- Testing: Creating knowledge graph ---")
        if not self.graph:
            self.skipTest("Graph connection not available")
            
        # First clear the database
        self.test_clear_graph()
        
        # Load test cases
        test_docs = load_test_cases()
        self.assertTrue(len(test_docs) > 0, "No test cases found to create graph")
        
        # Define a mock progress display function for testing
        class MockProgress:
            def progress(self, value):
                print(f"Progress: {value}%")
                
            def text(self, value):
                print(value)
                
            def empty(self):
                return self
                
        mock_st = MockProgress()
        
        try:
            # Create graph just with test documents (using shorter process for testing)
            print(f"Creating knowledge graph with {len(test_docs)} test documents...")
            
            # Create a simplified graph without using the transformer to save API calls
            for i, doc in enumerate(test_docs):
                # Create Case node with basic properties
                case_props = {
                    'title': doc.metadata.get('title', 'Unknown Case'),
                    'court': doc.metadata.get('court', 'Unknown Court'),
                    'date': doc.metadata.get('judgment_date', 'Unknown Date'),
                    'source': doc.metadata.get('source', ''),
                    'id': doc.metadata.get('id', f"test_{i}"),
                    'text': doc.page_content[:1000]  # Add first 1000 chars as text
                }
                
                # Create Case node
                cypher = """
                MERGE (c:Case {id: $id})
                SET c.title = $title,
                    c.court = $court,
                    c.date = $date,
                    c.source = $source,
                    c.text = $text
                RETURN c
                """
                self.graph.query(cypher, params=case_props)
                print(f"Created case node: {case_props['title']}")
                
                # Add judges if available
                if 'judges' in doc.metadata:
                    judges = doc.metadata['judges']
                    if isinstance(judges, str):
                        judges = [j.strip() for j in judges.split(',')]
                    elif not isinstance(judges, list):
                        judges = []
                        
                    for judge in judges:
                        if judge:
                            # Create judge node
                            judge_cypher = """
                            MERGE (j:Judge {name: $name})
                            RETURN j
                            """
                            self.graph.query(judge_cypher, params={'name': judge})
                            
                            # Create relationship
                            rel_cypher = """
                            MATCH (j:Judge {name: $name})
                            MATCH (c:Case {id: $case_id})
                            MERGE (j)-[:JUDGED]->(c)
                            """
                            self.graph.query(rel_cypher, params={'name': judge, 'case_id': case_props['id']})
                            print(f"  - Added judge: {judge}")
                
                # Add statutes if available
                if 'statutes' in doc.metadata and isinstance(doc.metadata['statutes'], list):
                    for statute in doc.metadata['statutes']:
                        if statute:
                            # Create statute node
                            statute_cypher = """
                            MERGE (s:Statute {name: $name})
                            RETURN s
                            """
                            self.graph.query(statute_cypher, params={'name': statute})
                            
                            # Create relationship
                            rel_cypher = """
                            MATCH (s:Statute {name: $name})
                            MATCH (c:Case {id: $case_id})
                            MERGE (c)-[:REFERENCES]->(s)
                            """
                            self.graph.query(rel_cypher, params={'name': statute, 'case_id': case_props['id']})
                            print(f"  - Added statute: {statute}")
            
            # Verify nodes were created
            count_query = "MATCH (n) RETURN count(n) as node_count"
            result = self.graph.query(count_query)
            self.assertTrue(result[0]['node_count'] > 0, "No nodes created in graph")
            print(f"Created {result[0]['node_count']} nodes in total")
            
            # Check case count
            case_query = "MATCH (c:Case) RETURN count(c) as case_count"
            result = self.graph.query(case_query)
            self.assertEqual(result[0]['case_count'], len(test_docs), 
                            f"Expected {len(test_docs)} case nodes, found {result[0]['case_count']}")
            print(f"Created {result[0]['case_count']} case nodes")
            
            return True
        except Exception as e:
            print(f"Error creating knowledge graph: {e}")
            self.fail(f"Knowledge graph creation failed: {e}")
            return False
    
    @unittest.skipIf(os.getenv("NEO4J_URI") is None, "Neo4j credentials not set")
    def test_simple_search(self):
        """Test simple text search functionality"""
        print("\n--- Testing: Simple text search ---")
        if not self.graph:
            self.skipTest("Graph connection not available")
        
        # Make sure we have data
        count_query = "MATCH (c:Case) RETURN count(c) as case_count"
        result = self.graph.query(count_query)
        if result[0]['case_count'] == 0:
            self.test_create_knowledge_graph()
        
        # Test search for "evidence"
        print("Testing search for 'evidence'...")
        results = simple_text_search(self.graph, "evidence")
        self.assertIsNotNone(results, "Search results should not be None")
        self.assertTrue(len(results) > 0, "Should find at least one result for 'evidence'")
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('title', 'Untitled')}")
        
        # Test search for "CD evidence"
        print("\nTesting search for 'CD evidence'...")
        results = simple_text_search(self.graph, "CD evidence")
        self.assertIsNotNone(results, "Search results should not be None")
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('title', 'Untitled')}")
    
    @unittest.skipIf(os.getenv("NEO4J_URI") is None or os.getenv("OPENAI_API_KEY") is None, 
                     "Neo4j or OpenAI credentials not set")
    def test_similar_cases(self):
        """Test finding similar cases"""
        print("\n--- Testing: Finding similar cases ---")
        if not self.graph or not self.openai_api_key:
            self.skipTest("Graph connection or OpenAI API key not available")
        
        # Make sure we have data
        count_query = "MATCH (c:Case) RETURN count(c) as case_count"
        result = self.graph.query(count_query)
        if result[0]['case_count'] == 0:
            self.test_create_knowledge_graph()
        
        # Test query
        test_query = """
        I want to know if a CD recording is admissible as evidence in a criminal trial.
        Can audio recordings on CD be presented in court without certification?
        What does the law say about digital evidence like CDs?
        """
        
        print("Testing similarity search with query about CD evidence...")
        similar_cases = find_similar_cases(
            graph=self.graph,
            case_text=test_query,
            llm=self.llm,
            embeddings=self.embeddings,
            neo4j_url=self.neo4j_url,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password
        )
        
        self.assertIsNotNone(similar_cases, "Similar cases should not be None")
        
        print(f"Found {len(similar_cases)} similar cases:")
        for i, doc in enumerate(similar_cases):
            print(f"{i+1}. {doc.metadata.get('title', 'Untitled')}")
            print(f"   Court: {doc.metadata.get('court', 'Unknown')}")
            print(f"   Date: {doc.metadata.get('date', 'Unknown')}")
            excerpt = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"   Excerpt: {excerpt}")
    
    @unittest.skipIf(os.getenv("NEO4J_URI") is None, "Neo4j credentials not set")
    def test_cypher_queries(self):
        """Test various Cypher queries on the graph"""
        print("\n--- Testing: Cypher queries ---")
        if not self.graph:
            self.skipTest("Graph connection not available")
        
        # Make sure we have data
        count_query = "MATCH (c:Case) RETURN count(c) as case_count"
        result = self.graph.query(count_query)
        if result[0]['case_count'] == 0:
            self.test_create_knowledge_graph()
        
        # Test query 1: Find cases related to evidence laws
        query1 = """
        MATCH (c:Case)
        WHERE c.title CONTAINS 'evidence' OR c.text CONTAINS 'evidence'
        RETURN c.title AS title, c.court AS court, c.date AS date
        LIMIT 5
        """
        print("Query 1: Find cases related to evidence laws")
        results = self.graph.query(query1)
        self.assertIsNotNone(results, "Query results should not be None")
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('title', 'Untitled')} - {result.get('court', 'Unknown')}")
        
        # Test query 2: Find judges and their cases
        query2 = """
        MATCH (j:Judge)-[:JUDGED]->(c:Case)
        RETURN j.name AS judge, collect(c.title) AS cases, count(c) AS case_count
        ORDER BY case_count DESC
        LIMIT 5
        """
        print("\nQuery 2: Find judges and their cases")
        results = self.graph.query(query2)
        self.assertIsNotNone(results, "Query results should not be None")
        
        print(f"Found {len(results)} judges:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('judge', 'Unknown')} - {result.get('case_count', 0)} cases")
        
        # Test query 3: Find statutes and related cases
        query3 = """
        MATCH (c:Case)-[:REFERENCES]->(s:Statute)
        RETURN s.name AS statute, collect(c.title) AS cases, count(c) AS case_count
        ORDER BY case_count DESC
        LIMIT 5
        """
        print("\nQuery 3: Find statutes and related cases")
        results = self.graph.query(query3)
        self.assertIsNotNone(results, "Query results should not be None")
        
        print(f"Found {len(results)} statutes:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('statute', 'Unknown')} - {result.get('case_count', 0)} cases")

def run_tests():
    print("=" * 50)
    print("Running Knowledge Graph Tests")
    print("=" * 50)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests() 