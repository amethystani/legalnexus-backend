#!/usr/bin/env python3
import os
import json
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.schema import Document
from langchain.vectorstores import Neo4jVector

def load_test_cases(test_cases_dir="data/test_cases"):
    """Load test case files from the directory"""
    # Adjust the path if running from Backend directory
    if os.path.basename(os.getcwd()) == "Backend":
        test_cases_dir = os.path.join("..", "data", "test_cases")
        
    print(f"Looking for test cases in: {os.path.abspath(test_cases_dir)}")
    
    if not os.path.exists(test_cases_dir):
        print(f"Test cases directory {test_cases_dir} not found!")
        return []
    
    # Get all JSON files in the test cases directory
    test_case_files = glob.glob(os.path.join(test_cases_dir, "*.json"))
    print(f"Found {len(test_case_files)} test case files")
    
    if not test_case_files:
        print("No test case files found!")
        return []
    
    # Load each test case file
    test_docs = []
    for file_path in test_case_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"Loading test case: {file_path}")
                data = json.load(f)
                
                # Extract content and metadata
                content = data.get('content', '')
                if not content:
                    print(f"Warning: No content found in {file_path}")
                    continue
                    
                metadata = {
                    'source': data.get('source', 'test_case'),
                    'title': data.get('title', os.path.basename(file_path)),
                    'court': data.get('court', 'Unknown Court'),
                    'judgment_date': data.get('judgment_date', 'Unknown Date'),
                    'id': data.get('id', f"test_{len(test_docs)}"),
                }
                
                # Add judges if available
                if 'entities' in data and 'judges' in data['entities']:
                    metadata['judges'] = data['entities']['judges']
                
                # Add more metadata from entity fields
                if 'entities' in data:
                    for entity_type, entities in data['entities'].items():
                        if entities:
                            metadata[entity_type] = entities
                
                # Create Document
                doc = Document(page_content=content, metadata=metadata)
                test_docs.append(doc)
                print(f"Successfully loaded test case: {metadata.get('title', 'Unnamed')}")
        except Exception as e:
            print(f"Error loading test case {file_path}: {str(e)}")
    
    print(f"Successfully loaded {len(test_docs)} test cases")
    return test_docs

def add_test_cases_to_neo4j(neo4j_url, neo4j_username, neo4j_password, embeddings):
    """Add test cases to Neo4j with proper embeddings"""
    print("\n=== Adding Test Cases to Neo4j Database ===")
    
    # Load test cases
    docs = load_test_cases()
    if not docs:
        print("No test cases found. Exiting.")
        return False
    
    # Connect to Neo4j graph
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        print(f"Connected to Neo4j database: {neo4j_url}")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return False
    
    # Add each test case as a Case node
    for doc in docs:
        try:
            # Create a Case node with properties
            case_props = {
                'id': doc.metadata.get('id', ''),
                'title': doc.metadata.get('title', 'Unknown Case'),
                'court': doc.metadata.get('court', 'Unknown Court'),
                'date': doc.metadata.get('judgment_date', 'Unknown Date'),
                'source': doc.metadata.get('source', 'test_case'),
                'text': doc.page_content
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
            
            result = graph.query(cypher, params=case_props)
            print(f"Added Case node: {case_props['title']}")
            
            # Add Judge relationships if available
            if 'judges' in doc.metadata:
                judges = doc.metadata['judges']
                if isinstance(judges, str):
                    judges = [j.strip() for j in judges.split(',')]
                
                for judge in judges:
                    if judge:
                        # Create Judge node
                        judge_cypher = """
                        MERGE (j:Judge {name: $name})
                        RETURN j
                        """
                        graph.query(judge_cypher, params={'name': judge})
                        
                        # Create relationship
                        rel_cypher = """
                        MATCH (j:Judge {name: $name})
                        MATCH (c:Case {id: $case_id})
                        MERGE (j)-[:JUDGED]->(c)
                        """
                        graph.query(rel_cypher, params={'name': judge, 'case_id': doc.metadata.get('id', '')})
            
            # Add statute relationships if available
            if 'statutes' in doc.metadata:
                statutes = doc.metadata['statutes']
                if isinstance(statutes, str):
                    statutes = [s.strip() for s in statutes.split(',')]
                
                for statute in statutes:
                    if statute:
                        # Create Statute node
                        statute_cypher = """
                        MERGE (s:Statute {name: $name})
                        RETURN s
                        """
                        graph.query(statute_cypher, params={'name': statute})
                        
                        # Create relationship
                        rel_cypher = """
                        MATCH (s:Statute {name: $name})
                        MATCH (c:Case {id: $case_id})
                        MERGE (c)-[:REFERENCES]->(s)
                        """
                        graph.query(rel_cypher, params={'name': statute, 'case_id': doc.metadata.get('id', '')})
                
        except Exception as e:
            print(f"Error adding Case node {doc.metadata.get('title')}: {e}")
    
    # Create vector embeddings
    print("\n=== Creating Vector Embeddings for Test Cases ===")
    try:
        # Create vector store with embeddings
        vector_store = Neo4jVector.from_documents(
            docs,
            embeddings,
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            index_name="vector_index",
            node_label="Case",
            text_node_property="text",
            embedding_node_property="embedding"
        )
        print("Successfully added vector embeddings to test cases")
        return True
    except Exception as e:
        print(f"Error creating vector embeddings: {e}")
        return False

def main():
    """Main function to load test cases into Neo4j"""
    print("=" * 50)
    print("Loading Test Cases into Neo4j")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        print("Error: Neo4j connection details not found in environment variables.")
        print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
        return
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Add test cases to Neo4j
    success = add_test_cases_to_neo4j(
        neo4j_url=neo4j_url,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        embeddings=embeddings
    )
    
    if success:
        print("\n=== Success! ===")
        print("Test cases have been loaded into Neo4j with vector embeddings.")
        print("You can now run the test_similar_cases.py script to test similarity search.")
    else:
        print("\n=== Error! ===")
        print("Failed to load test cases into Neo4j.")

if __name__ == "__main__":
    main() 