import os
import json
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
import streamlit as st
import tempfile
from neo4j import GraphDatabase
import time

# Import graph visualization components
try:
    import plotly.graph_objects as go
    import networkx as nx
    from utils.main_files.kg_visualizer import (
        get_graph_data, 
        create_network_graph, 
        get_case_connections, 
        show_case_details
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Visualization import error: {e}")

# Set Google Gemini API key
GOOGLE_API_KEY = "AIzaSyCE64GFYnFZnZktAATpIx0zTp3HpUAUSbA"
# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

def load_legal_data(data_path="data", include_csv=True, max_csv_cases=100):
    """Load all the legal data from JSON files and CSV datasets"""
    # Adjust the path if running from Backend directory
    if os.path.basename(os.getcwd()) == "Backend":
        data_path = os.path.join("..", data_path)
        
    all_docs = []
    
    # Load JSON files
    json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)
    
    # Log the number of files found to help with debugging
    print(f"Found {len(json_files)} JSON files in {data_path}")
    for file_path in json_files:
        print(f"  - {file_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract the relevant fields from the JSON structure
                content = data.get('content', '')
                if not content:
                    continue
                    
                # Create metadata from the available fields
                metadata = {
                    'source': data.get('source', ''),
                    'title': data.get('title', ''),
                    'court': data.get('court', ''),
                    'judgment_date': data.get('judgment_date', ''),
                    'id': data.get('id', ''),
                }
                
                # Add judges if available
                if 'judges' in data.get('metadata', {}):
                    metadata['judges'] = data['metadata']['judges']
                
                # Add entities if available
                if 'entities' in data:
                    for entity_type, entities in data['entities'].items():
                        if entities:
                            metadata[f'{entity_type}'] = entities
                
                # Create a Document object for this legal case
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
                print(f"Successfully loaded: {metadata.get('title', 'Unnamed case')}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Load CSV data if requested
    if include_csv:
        try:
            from utils.main_files.csv_data_loader import load_all_csv_data
            print(f"\n=== Loading CSV Classification Datasets ===")
            csv_docs = load_all_csv_data(data_path, max_cases_per_file=max_csv_cases)
            all_docs.extend(csv_docs)
            print(f"Added {len(csv_docs)} cases from CSV datasets")
        except ImportError:
            print("CSV loader not available, skipping CSV data")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
    
    print(f"\n=== Total: {len(all_docs)} legal documents loaded ===")
    return all_docs

def create_legal_knowledge_graph(graph, docs, llm, embeddings, neo4j_url, neo4j_username, neo4j_password):
    """Create a knowledge graph from legal documents"""
    # Show progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Initialize index variable
    index = None
    
    # Split documents into smaller chunks for processing
    progress_text.text("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = text_splitter.split_documents(docs)
    progress_bar.progress(10)
    
    # Clear the graph database
    progress_text.text("Clearing existing graph database...")
    cypher = """
      MATCH (n)
      DETACH DELETE n;
    """
    graph.query(cypher)
    progress_bar.progress(20)
    
    # Add specific legal entity nodes and relationships
    progress_text.text("Creating case nodes and relationships...")
    total_docs = len(docs)
    
    for i, doc in enumerate(docs):
        # Update progress
        progress_percent = 20 + int((i / total_docs) * 30)
        progress_bar.progress(progress_percent)
        progress_text.text(f"Processing document {i+1}/{total_docs}: {doc.metadata.get('title', 'Unknown Case')}")
        
        # Create a Case node
        case_props = {
            'title': doc.metadata.get('title', 'Unknown Case'),
            'court': doc.metadata.get('court', 'Unknown Court'),
            'date': doc.metadata.get('judgment_date', 'Unknown Date'),
            'source': doc.metadata.get('source', ''),
            'id': doc.metadata.get('id', '')
        }
        
        # Create the Case node with error handling
        try:
            # Ensure we have a valid ID
            if not case_props['id']:
                case_props['id'] = f"case_{i}"
                
            cypher = """
            MERGE (c:Case {id: $id})
            SET c.title = $title,
                c.court = $court,
                c.date = $date,
                c.source = $source,
                c.text = $text
            RETURN c
            """
            # Add the text content to the node properties
            case_props['text'] = doc.page_content
            graph.query(cypher, params=case_props)
        except Exception as e:
            progress_text.text(f"Error creating case node for {case_props['title']}: {str(e)}")
            # Skip this document as we can't create relationships without the case node
            continue
        
        # Add Judge nodes and relationships
        if 'judges' in doc.metadata:
            judges = doc.metadata['judges']
            if isinstance(judges, str):
                judges = [j.strip() for j in judges.split(',')]
            
            for judge in judges:
                if judge:
                    try:
                        # First create/merge the judge node
                        judge_cypher = """
                        MERGE (j:Judge {name: $name})
                        RETURN j
                        """
                        graph.query(judge_cypher, params={'name': judge})
                        
                        # Then create the relationship in a separate query
                        rel_cypher = """
                        MATCH (j:Judge {name: $name})
                        MATCH (c:Case {id: $case_id})
                        MERGE (j)-[:JUDGED]->(c)
                        """
                        graph.query(rel_cypher, params={'name': judge, 'case_id': doc.metadata.get('id', '')})
                    except Exception as e:
                        # Log the error but continue processing
                        print(f"Error processing judge {judge} for case {doc.metadata.get('id', '')}: {str(e)}")
                        continue
        
        # Add Court node and relationship
        if 'court' in doc.metadata and doc.metadata['court']:
            try:
                # First create/merge the court node
                court_cypher = """
                MERGE (court:Court {name: $name})
                RETURN court
                """
                graph.query(court_cypher, params={'name': doc.metadata['court']})
                
                # Then create the relationship in a separate query
                rel_cypher = """
                MATCH (court:Court {name: $name})
                MATCH (c:Case {id: $case_id})
                MERGE (c)-[:HEARD_BY]->(court)
                """
                graph.query(rel_cypher, params={'name': doc.metadata['court'], 'case_id': doc.metadata.get('id', '')})
            except Exception as e:
                # Log the error but continue processing
                print(f"Error processing court {doc.metadata['court']} for case {doc.metadata.get('id', '')}: {str(e)}")
        
        # Add Statute nodes and relationships
        if 'statutes' in doc.metadata and doc.metadata['statutes']:
            for statute in doc.metadata['statutes']:
                if statute:
                    try:
                        # Use a different approach that won't fail if the case node doesn't exist
                        # First create/merge the statute node
                        statute_cypher = """
                        MERGE (s:Statute {name: $name})
                        RETURN s
                        """
                        graph.query(statute_cypher, params={'name': statute})
                        
                        # Then create the relationship in a separate query
                        # This checks if both nodes exist before creating the relationship
                        rel_cypher = """
                        MATCH (s:Statute {name: $name})
                        MATCH (c:Case {id: $case_id})
                        MERGE (c)-[:REFERENCES]->(s)
                        """
                        graph.query(rel_cypher, params={'name': statute, 'case_id': doc.metadata.get('id', '')})
                    except Exception as e:
                        # Log the error but continue processing
                        print(f"Error processing statute {statute} for case {doc.metadata.get('id', '')}: {str(e)}")
                        continue
    
    # Skip the graph transformer as it's taking too much time
    progress_text.text("Skipping graph transformer for faster processing...")
    progress_bar.progress(80)
    
    # Create vector index for similarity search or use existing one
    progress_text.text("Setting up vector index for similarity search...")
    progress_bar.progress(90)
    
    try:
        # Check if index exists with better compatibility across Neo4j versions
        index_exists = False
        try:
            # Try to connect to existing index first
            index = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database="neo4j",
                node_label="Case",
                text_node_properties=["id", "text"], 
                embedding_node_property="embedding", 
                index_name="vector_index", 
                keyword_index_name="entity_index", 
                search_type="hybrid" 
            )
            progress_text.text("Successfully connected to existing vector index.")
            index_exists = True
        except Exception as e:
            progress_text.text(f"No existing vector index found: {str(e)}")
            index_exists = False
        
        if not index_exists:
            # Try to create a new index with error handling
            progress_text.text("Creating new vector index...")
            try:
                # First, try to drop any incomplete indexes
                driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
                with driver.session() as session:
                    try:
                        # Try newer syntax first
                        session.run("CALL db.index.vector.dropIfExists('vector_index')")
                    except Exception:
                        try:
                            # Try alternative drop methods
                            session.run("DROP INDEX vector_index IF EXISTS")
                        except Exception:
                            # If dropping fails, continue anyway
                            pass
                driver.close()
                
                # Create the vector index
                index = Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    database="neo4j",
                    node_label="Case",
                    text_node_properties=["id", "text"], 
                    embedding_node_property="embedding", 
                    index_name="vector_index", 
                    keyword_index_name="entity_index", 
                    search_type="hybrid" 
                )
                progress_text.text("New vector index created successfully.")
                
            except Exception as create_error:
                progress_text.text(f"Vector index creation failed: {str(create_error)}")
                progress_text.text("Vector search may not be available, but text search will work.")
                index = None
                
        progress_bar.progress(100)
        progress_text.text("Knowledge graph processing completed!")
        
    except Exception as e:
        progress_text.text(f"Error with vector search setup: {str(e)}")
        progress_text.text("System will use text-based search as fallback.")
        index = None
    
    return index

def find_similar_cases(graph, case_text, llm, embeddings, neo4j_url, neo4j_username, neo4j_password):
    """Find cases similar to the input text using Gemini embeddings with text similarity fallback"""
    # Show progress status
    status = st.empty()
    status.info("Searching for similar cases...")
    
    # Create a temporary document from the input text
    query_doc = Document(page_content=case_text)
    
    # PRIMARY METHOD: Vector search with Gemini embeddings
    try:
        status.info("Using Gemini vector search...")
        index = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database="neo4j",
            node_label="Case",
            text_node_properties=["id", "text"], 
            embedding_node_property="embedding", 
            index_name="vector_index", 
            keyword_index_name="entity_index", 
            search_type="hybrid" 
        )
        
        # Search for similar cases with scores
        similar_docs_with_scores = index.similarity_search_with_score(query_doc.page_content, k=3)
        
        # If we got results, return them with scores
        if similar_docs_with_scores:
            status.success("Found similar cases using Gemini vector search!")
            docs = []
            scores = []
            for doc, score in similar_docs_with_scores:
                docs.append(doc)
                scores.append(score)
            return docs, scores
        
        # If no results but no error, fall through to fallback
        status.info("Vector search returned no results, trying text similarity fallback...")
            
    except Exception as e:
        status.warning(f"Gemini vector search not available: {str(e)}")
    
    # FALLBACK METHOD: Text similarity search
    try:
        status.info("Using text-based similarity search...")
        # Get all cases from database
        query = """
        MATCH (c:Case)
        RETURN c.id AS id, c.title AS title, c.court AS court, c.date AS date, c.text AS text
        LIMIT 50
        """
        results = graph.query(query)
        
        if results:
            # Compute text similarity for each case
            similarities = []
            for result in results:
                content = result.get("text", "")
                if content:
                    similarity = compute_text_similarity(case_text, content)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "title": result.get("title", "Unknown"),
                            "court": result.get("court", "Unknown"),
                            "date": result.get("date", "Unknown"),
                            "id": result.get("id", "")
                        }
                    )
                    similarities.append((doc, similarity))
            
            # Sort by similarity and return top 3 with scores
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if similarities:
                status.success("Found similar cases using text similarity!")
                docs = []
                scores = []
                for doc, score in similarities[:3]:
                    docs.append(doc)
                    scores.append(score)
                return docs, scores
                
    except Exception as e:
        status.warning(f"Error with text similarity: {str(e)}")
    
    # If all methods fail, return empty results
    status.error("No similar cases found. Please try a different query.")
    return [], []

# Add helper function for text similarity from case_similarity_cli.py
def compute_text_similarity(query_text, document_text):
    """Compute simple text similarity when embeddings are not available"""
    import re
    import difflib
    
    # Convert to lowercase for better matching
    query_lower = query_text.lower()
    doc_lower = document_text.lower()
    
    # Extract key words from query (remove common stop words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'has', 'have', 'had', 'been', 'of', 'for', 'by', 'with', 'to', 'in', 
                 'on', 'at', 'from', 'as', 'it', 'its'}
    
    # Extract words from query and document
    query_words = set([word.strip('.,;:()[]{}""\'') for word in query_lower.split() 
                     if word.strip('.,;:()[]{}""\'') and word.strip('.,;:()[]{}""\'') not in stop_words])
    
    # Count matches
    match_count = sum(1 for word in query_words if word in doc_lower)
    
    # Calculate similarity score
    # Base score from word matching
    base_score = match_count / max(len(query_words), 1)
    
    # Add bonus from sequence matching using difflib
    similarity_bonus = difflib.SequenceMatcher(None, query_lower, doc_lower[:min(len(doc_lower), 1000)]).ratio() * 0.2
    
    # Final score combines direct word matches and sequence similarity
    return min(base_score + similarity_bonus, 1.0)  # Cap at 1.0

# Add helper function for cosine similarity with embeddings
def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    import numpy as np
    # Ensure embeddings are numpy arrays with matching dimensions
    # Gemini embeddings are 768-dimensional
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # If embeddings have different dimensions, attempt to handle
    if embedding1.shape != embedding2.shape:
        # Pad or truncate to match dimensions
        max_dim = max(len(embedding1), len(embedding2))
        min_dim = min(len(embedding1), len(embedding2))
        
        # Only use dimensions both have in common
        embedding1 = embedding1[:min_dim] if len(embedding1) > min_dim else embedding1
        embedding2 = embedding2[:min_dim] if len(embedding2) > min_dim else embedding2
    
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Handle zero vectors to prevent division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

def load_test_cases():
    """Explicitly load test cases from the test_cases directory"""
    # Adjust the path if running from Backend directory
    test_cases_dir = os.path.join("data", "test_cases")
    if os.path.basename(os.getcwd()) == "Backend":
        test_cases_dir = os.path.join("..", "data", "test_cases")
        
    print(f"Looking for test cases in: {os.path.abspath(test_cases_dir)}")
    
    if os.path.exists(test_cases_dir):
        print(f"Loading test cases from {test_cases_dir}")
        test_case_files = glob.glob(os.path.join(test_cases_dir, "*.json"))
        
        if not test_case_files:
            print("No test case files found!")
            return []
        
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
    else:
        print(f"Test cases directory {test_cases_dir} not found!")
        # Attempt to create the directory
        try:
            os.makedirs(test_cases_dir, exist_ok=True)
            print(f"Created test cases directory: {test_cases_dir}")
        except Exception as e:
            print(f"Error creating test cases directory: {str(e)}")
        return []

def check_knowledge_graph_exists(graph):
    """Check if knowledge graph already has legal cases loaded"""
    try:
        # Check if we have any Case nodes in the graph
        count_query = "MATCH (c:Case) RETURN count(c) as case_count"
        result = graph.query(count_query)
        case_count = result[0].get('case_count', 0) if result else 0
        
        # For vector index check, use a simpler approach that works across Neo4j versions
        # Try different methods to check for vector indexes
        try:
            # Method 1: Try the newer syntax first
            index_query = "CALL db.indexes() YIELD name, type WHERE type = 'VECTOR' AND name = 'vector_index' RETURN count(*) as index_count"
            index_result = graph.query(index_query)
            index_count = index_result[0].get('index_count', 0) if index_result else 0
        except Exception:
            try:
                # Method 2: Try older syntax
                index_query = "CALL db.indexes() YIELD name, type WHERE type = 'VECTOR' RETURN count(*) as index_count"
                index_result = graph.query(index_query)
                index_count = index_result[0].get('index_count', 0) if index_result else 0
            except Exception:
                try:
                    # Method 3: Try APOC if available
                    index_query = "CALL apoc.meta.schema() YIELD value RETURN value"
                    graph.query(index_query)
                    index_count = 1  # Assume index exists if we can query schema
                except Exception:
                    # Method 4: Just check if we have meaningful data and assume index is working
                    # if cases exist and the vector operations work during runtime
                    index_count = 1 if case_count > 0 else 0
        
        # Consider the graph ready if we have cases (vector index is optional for basic functionality)
        return case_count > 0
    except Exception as e:
        print(f"Error checking knowledge graph: {e}")
        return False

def get_knowledge_graph_stats(graph):
    """Get statistics about the existing knowledge graph"""
    try:
        stats = {}
        
        # Count cases
        case_query = "MATCH (c:Case) RETURN count(c) as case_count"
        case_result = graph.query(case_query)
        stats['cases'] = case_result[0].get('case_count', 0) if case_result else 0
        
        # Count judges
        judge_query = "MATCH (j:Judge) RETURN count(j) as judge_count"
        judge_result = graph.query(judge_query)
        stats['judges'] = judge_result[0].get('judge_count', 0) if judge_result else 0
        
        # Count courts
        court_query = "MATCH (court:Court) RETURN count(court) as court_count"
        court_result = graph.query(court_query)
        stats['courts'] = court_result[0].get('court_count', 0) if court_result else 0
        
        # Check vector index with fallback methods
        try:
            # Try newer syntax first
            index_query = "CALL db.indexes() YIELD name, type WHERE type = 'VECTOR' RETURN count(*) as vector_count"
            index_result = graph.query(index_query)
            stats['vector_indexes'] = index_result[0].get('vector_count', 0) if index_result else 0
        except Exception:
            try:
                # Try older syntax
                index_query = "CALL db.indexes() YIELD name, type RETURN count(*) as total_indexes"
                index_result = graph.query(index_query)
                stats['vector_indexes'] = index_result[0].get('total_indexes', 0) if index_result else 0
            except Exception:
                # If we can't check indexes, assume they exist if we have cases
                stats['vector_indexes'] = 1 if stats['cases'] > 0 else 0
        
        return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {'cases': 0, 'judges': 0, 'courts': 0, 'vector_indexes': 0}

def setup_qa_chain(graph, llm):
    """Set up QA chain for an existing knowledge graph"""
    try:
        # Retrieve the graph schema
        schema = graph.get_schema
        
        # Set up the QA chain with a simpler prompt to reduce token usage
        template = """
        Generate a Cypher query to search legal cases based on the following question.
        
        SCHEMA: {schema}
        
        TIPS:
        - Focus on Case nodes with properties: id, title, court, date, text
        - Use text matching with CONTAINS on Case.text and Case.title
        - Keep queries simple, direct text matching is best
        
        QUESTION: {question}
        
        CYPHER QUERY:
        """
        
        question_prompt = PromptTemplate(
            template=template, 
            input_variables=["schema", "question"] 
        )
        
        # Create a QA chain
        qa = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=question_prompt,
            verbose=True,
            return_direct=True,
            top_k=5,
            allow_dangerous_requests=True
        )
        
        return qa
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def main():
    graph = None
    st.set_page_config(
        layout="wide",
        page_title="Legal Knowledge Graph",
        page_icon=":scale:"
    )

    st.title("Legal Knowledge Graph: Find Similar Cases")

    # Load environment variables
    load_dotenv()

    # Set Google Gemini API key
    if 'GOOGLE_API_KEY' not in st.session_state:
        # Use the defined API key
        google_api_key = GOOGLE_API_KEY
        if google_api_key:
            st.session_state['GOOGLE_API_KEY'] = google_api_key
            # Initialize Gemini embeddings and LLM
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    google_api_key=google_api_key,
                    model="models/embedding-001",
                    task_type="retrieval_document",
                    title="Legal case document"
                )
                
                # Use lightweight Gemini 2.5 Flash Preview model to avoid quota issues
                llm = ChatGoogleGenerativeAI(
                    google_api_key=google_api_key,
                    model="models/gemini-flash-latest",
                    temperature=0.1,
                    convert_system_message_to_human=True,
                    max_output_tokens=2048,
                    top_k=32,
                    top_p=0.95
                )
                st.session_state['embeddings'] = embeddings
                st.session_state['llm'] = llm
                st.sidebar.success("Gemini API configured successfully.")
            except Exception as e:
                st.sidebar.error(f"Error initializing Gemini API: {str(e)}")
                st.sidebar.info("Please check your API key or try again later.")
        else:
            st.sidebar.subheader("Google Gemini API Key")
            google_api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type='password')
            if google_api_key:
                st.session_state['GOOGLE_API_KEY'] = google_api_key
                try:
                    # Initialize Gemini embeddings and LLM
                    embeddings = GoogleGenerativeAIEmbeddings(
                        google_api_key=google_api_key,
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        title="Legal case document"
                    )
                    
                    # Use lightweight Gemini 2.5 Flash Preview model to avoid quota issues
                    llm = ChatGoogleGenerativeAI(
                        google_api_key=google_api_key,
                        model="models/gemini-flash-latest",
                        temperature=0.1,
                        convert_system_message_to_human=True,
                        max_output_tokens=2048,
                        top_k=32,
                        top_p=0.95
                    )
                    st.session_state['embeddings'] = embeddings
                    st.session_state['llm'] = llm
                    st.sidebar.success("Google Gemini API Key set successfully.")
                except Exception as e:
                    st.sidebar.error(f"Error initializing Gemini API: {str(e)}")
                    st.sidebar.info("Please check your API key or try again later.")
    else:
        embeddings = st.session_state['embeddings']
        llm = st.session_state['llm']
        
    # Initialize variables
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    graph = None

    # Set Neo4j connection details
    if 'neo4j_connected' not in st.session_state:
        # Check if credentials are in environment variables
        if neo4j_url and neo4j_username and neo4j_password:
            try:
                graph = Neo4jGraph(
                    url=neo4j_url, 
                    username=neo4j_username, 
                    password=neo4j_password
                )
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                # Store connection parameters for later use
                st.session_state['neo4j_url'] = neo4j_url
                st.session_state['neo4j_username'] = neo4j_username
                st.session_state['neo4j_password'] = neo4j_password
                st.sidebar.success("Connected to Neo4j database using environment variables.")
            except Exception as e:
                st.error(f"Failed to connect to Neo4j with environment variables: {e}")
                
                # Fall back to manual entry
                st.sidebar.subheader("Connect to Neo4j Database")
                neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
                neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
                neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
                connect_button = st.sidebar.button("Connect")
                if connect_button and neo4j_password:
                    try:
                        graph = Neo4jGraph(
                            url=neo4j_url, 
                            username=neo4j_username, 
                            password=neo4j_password
                        )
                        st.session_state['graph'] = graph
                        st.session_state['neo4j_connected'] = True
                        # Store connection parameters for later use
                        st.session_state['neo4j_url'] = neo4j_url
                        st.session_state['neo4j_username'] = neo4j_username
                        st.session_state['neo4j_password'] = neo4j_password
                        st.sidebar.success("Connected to Neo4j database.")
                    except Exception as e:
                        st.error(f"Failed to connect to Neo4j: {e}")
        else:
            # Manual Neo4j connection entry
            st.sidebar.subheader("Connect to Neo4j Database")
            neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
            neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
            neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
            connect_button = st.sidebar.button("Connect")
            if connect_button and neo4j_password:
                try:
                    graph = Neo4jGraph(
                        url=neo4j_url, 
                        username=neo4j_username, 
                        password=neo4j_password
                    )
                    st.session_state['graph'] = graph
                    st.session_state['neo4j_connected'] = True
                    # Store connection parameters for later use
                    st.session_state['neo4j_url'] = neo4j_url
                    st.session_state['neo4j_username'] = neo4j_username
                    st.session_state['neo4j_password'] = neo4j_password
                    st.sidebar.success("Connected to Neo4j database.")
                except Exception as e:
                    st.error(f"Failed to connect to Neo4j: {e}")
    else:
        graph = st.session_state['graph']
        neo4j_url = st.session_state['neo4j_url']
        neo4j_username = st.session_state['neo4j_username']
        neo4j_password = st.session_state['neo4j_password']

    # Ensure that the Neo4j connection is established before proceeding
    if graph is not None:
        # Check if knowledge graph already exists to avoid reprocessing
        if check_knowledge_graph_exists(graph):
            st.success("📊 Knowledge graph already exists! Loading existing data...")
            
            # Display existing graph statistics
            stats = get_knowledge_graph_stats(graph)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Legal Cases", stats.get('cases', 0))
            with col2:
                st.metric("Judges", stats.get('judges', 0))
            with col3:
                st.metric("Courts", stats.get('courts', 0))
            with col4:
                st.metric("Vector Indexes", stats.get('vector_indexes', 0))
            
            # Set up QA chain for existing graph
            if 'qa' not in st.session_state:
                with st.spinner("Setting up QA system..."):
                    qa = setup_qa_chain(graph, llm)
                    if qa:
                        st.session_state['qa'] = qa
                        st.session_state['kg_loaded'] = True  # Mark as loaded
                        st.success("✅ QA system ready!")
                    else:
                        st.error("❌ Failed to set up QA system")
            else:
                st.success("✅ System ready! Knowledge graph and QA are loaded.")
                # Mark as already loaded in session state
                st.session_state['kg_loaded'] = True
            
            # Add option to rebuild knowledge graph
            with st.expander("🔄 Advanced Options"):
                st.warning("⚠️ These operations will clear existing data")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear & Rebuild Knowledge Graph"):
                        # Clear existing data
                        clear_query = "MATCH (n) DETACH DELETE n"
                        graph.query(clear_query)
                        # Clear session state
                        for key in ['qa', 'kg_loaded']:
                            st.session_state.pop(key, None)
                        st.success("Knowledge graph cleared! Please refresh to rebuild.")
                        st.stop()
                with col2:
                    if st.button("🔄 Refresh Page"):
                        st.rerun()
        
        elif 'qa' not in st.session_state:
            # Knowledge graph doesn't exist, create it
            st.info("🚀 Setting up knowledge graph for the first time...")
            
            # Options for loading data
            data_options = ["Use scraped legal data", "Upload PDF"]
            data_choice = st.radio("Choose data source:", data_options)
            
            if data_choice == "Use scraped legal data":
                with st.spinner("Loading legal cases from data directory..."):
                    # Load all legal data from data directory
                    legal_docs = load_legal_data()
                    
                    if not legal_docs:
                        st.error("No legal data found in the data directory. Make sure the scraper has saved data.")
                    else:
                        st.success(f"Loaded {len(legal_docs)} legal cases from the data directory.")
                        
                        # Create knowledge graph from legal documents
                        with st.spinner("Creating knowledge graph from legal cases..."):
                            # First explicitly load test cases
                            test_docs = load_test_cases()
                            if test_docs:
                                legal_docs.extend(test_docs)
                                st.success(f"Added {len(test_docs)} test cases to the dataset")
                            
                            create_legal_knowledge_graph(
                                graph=graph, 
                                docs=legal_docs, 
                                llm=llm, 
                                embeddings=embeddings,
                                neo4j_url=neo4j_url,
                                neo4j_username=neo4j_username,
                                neo4j_password=neo4j_password
                            )
                            
                        st.success("Knowledge graph created successfully!")
                        
                        # Retrieve the graph schema
                        schema = graph.get_schema
                        
                        # Set up the QA chain with a simpler prompt to reduce token usage
                        template = """
                        Generate a Cypher query to search legal cases based on the following question.
                        
                        SCHEMA: {schema}
                        
                        TIPS:
                        - Focus on Case nodes with properties: id, title, court, date, text
                        - Use text matching with CONTAINS on Case.text and Case.title
                        - Keep queries simple, direct text matching is best
                        
                        QUESTION: {question}
                        
                        CYPHER QUERY:
                        """
                        
                        question_prompt = PromptTemplate(
                            template=template, 
                            input_variables=["schema", "question"] 
                        )
                        
                        # Create a QA chain with LLM caching and simplified response formatting
                        try:
                            qa = GraphCypherQAChain.from_llm(
                                llm=llm,
                                graph=graph,
                                cypher_prompt=question_prompt,
                                verbose=True,
                                return_direct=True,  # Return direct results without additional LLM processing
                                top_k=5,  # Limit number of results
                                allow_dangerous_requests=True
                            )
                            st.session_state['qa'] = qa
                            st.session_state['kg_loaded'] = True  # Mark as loaded after creation
                        except Exception as e:
                            st.error(f"Error initializing QA chain: {str(e)}")
                            st.warning("QA functionality unavailable. Please check your Gemini API key and Neo4j connection.")
            
            elif data_choice == "Upload PDF":
                # Original PDF upload functionality
                uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")
                
                if uploaded_file is not None:
                    with st.spinner("Processing the PDF..."):
                        # Save uploaded file to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        
                        # Load and split the PDF
                        loader = PyPDFLoader(tmp_file_path)
                        pages = loader.load_and_split()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                        docs = text_splitter.split_documents(pages)
                        
                        lc_docs = []
                        for doc in docs:
                            lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
                            metadata={'source': uploaded_file.name}))
                        
                        # Clear the graph database
                        cypher = """
                          MATCH (n)
                          DETACH DELETE n;
                        """
                        graph.query(cypher)
                        
                        # Transform documents into graph documents
                        transformer = LLMGraphTransformer(
                            llm=llm,
                            node_properties=False, 
                            relationship_properties=False
                        ) 
                        
                        graph_documents = transformer.convert_to_graph_documents(lc_docs)
                        graph.add_graph_documents(graph_documents, include_source=True)
                        
                        # Use the stored connection parameters
                        index = Neo4jVector.from_existing_graph(
                            embedding=embeddings,
                            url=neo4j_url,
                            username=neo4j_username,
                            password=neo4j_password,
                            database="neo4j",
                            node_label="Case",  # Changed from "Legal Case" to match the rest of the system
                            text_node_properties=["id", "text"], 
                            embedding_node_property="embedding", 
                            index_name="vector_index", 
                            keyword_index_name="entity_index", 
                            search_type="hybrid" 
                        )
                        
                        st.success(f"{uploaded_file.name} preparation is complete.")
                        
                        # Retrieve the graph schema
                        schema = graph.get_schema
                        
                        # Set up the QA chain with a simpler prompt to reduce token usage
                        template = """
                        Generate a Cypher query to search legal cases based on the following question.
                        
                        SCHEMA: {schema}
                        
                        TIPS:
                        - Focus on Case nodes with properties: id, title, court, date, text
                        - Use text matching with CONTAINS on Case.text and Case.title
                        - Keep queries simple, direct text matching is best
                        
                        QUESTION: {question}
                        
                        CYPHER QUERY:
                        """
                        
                        question_prompt = PromptTemplate(
                            template=template, 
                            input_variables=["schema", "question"] 
                        )
                        
                        # Create a QA chain with LLM caching and simplified response formatting
                        try:
                            qa = GraphCypherQAChain.from_llm(
                                llm=llm,
                                graph=graph,
                                cypher_prompt=question_prompt,
                                verbose=True,
                                return_direct=True,  # Return direct results without additional LLM processing
                                top_k=5,  # Limit number of results
                                allow_dangerous_requests=True
                            )
                            st.session_state['qa'] = qa
                            st.session_state['kg_loaded'] = True  # Mark as loaded after creation
                        except Exception as e:
                            st.error(f"Error initializing QA chain: {str(e)}")
                            st.warning("QA functionality unavailable. Please check your Gemini API key and Neo4j connection.")
        
        # Display query interface once the knowledge graph is set up
        if 'qa' in st.session_state or 'kg_loaded' in st.session_state:
            # Add tabs for different search options
            tab1, tab2, tab3, tab4 = st.tabs(["Ask Questions", "Find Similar Cases", "Simple Search", "🕸️ Knowledge Graph Visualization"])
            
            with tab1:
                st.subheader("Ask a Question About Your Legal Situation")
                
                if 'qa' not in st.session_state:
                    st.warning("⚠️ QA functionality is not available. Please ensure:")
                    st.markdown("""
                    - Google Gemini API key is valid
                    - Neo4j database is connected
                    - Legal documents are loaded
                    """)
                    st.info("Try using the 'Find Similar Cases' or 'Simple Search' tabs instead.")
                else:
                    with st.form(key='question_form'):
                        question = st.text_input("Describe your legal situation or ask about similar cases:")
                        use_analysis = st.checkbox("Generate detailed legal analysis (uses more API quota)", value=False)
                        submit_button = st.form_submit_button(label='Search for Similar Cases')
                        
                    if submit_button and question:
                        with st.spinner("Searching for relevant legal cases..."):
                            try:
                                # Process the query through the QA chain
                                res = st.session_state['qa'].invoke({"query": question})
                                
                                # Display the results
                                st.subheader("Relevant Legal Cases")
                                
                                # Check if result is a list of raw results from Neo4j
                                if 'result' in res and isinstance(res['result'], list):
                                    # Format and display the results properly
                                    formatted_cases = format_case_results(res['result'])
                                    display_case_results(formatted_cases, show_similarity=False)
                                else:
                                    # Fallback to plain text display
                                    st.write(res['result'])
                                
                                # Show the Cypher query if available
                                if 'intermediate_steps' in res and 'cypher' in res['intermediate_steps']:
                                    with st.expander("View Search Query"):
                                        st.code(res['intermediate_steps']['cypher'], language='cypher')
                                
                                # Only do analysis if checkbox is checked to save quota
                                if use_analysis:
                                    # Additional analysis with more concise prompt
                                    prompt = f"""
                                    Based on the user's question: "{question}"
                                    
                                    And the search results: 
                                    {res['result']}
                                    
                                    Provide a brief legal analysis focusing on:
                                    1. Key legal principles that apply
                                    2. Important considerations for this case
                                    
                                    Keep it under 200 words.
                                    """
                                    
                                    with st.spinner("Generating legal analysis..."):
                                        try:
                                            analysis = st.session_state['llm'].invoke(prompt)
                                            
                                            st.subheader("Legal Analysis")
                                            st.write(analysis.content)
                                        except Exception as e:
                                            st.warning(f"Could not generate analysis due to API limits: {str(e)}")
                                            # Provide a simple response without using LLM
                                            st.info("Please review the legal cases above for relevant information to your situation. For detailed legal advice, consult a qualified legal professional.")
                            except Exception as e:
                                st.error(f"Error searching cases: {str(e)}")
                                st.info("Please try a different search query or check connection to the database.")
            
            with tab2:
                st.subheader("Find Cases Similar to Your Case")
                with st.form(key='similarity_form'):
                    case_text = st.text_area("Enter case details, judgment, or legal situation:", height=300)
                    similarity_button = st.form_submit_button(label='Find Similar Cases')
                
                if similarity_button and case_text:
                    # Display a warning if the text is too short
                    if len(case_text.split()) < 20:
                        st.warning("Please provide more detailed information (at least 20 words) for better results.")
                    
                    with st.spinner("Searching for similar legal cases..."):
                        # Find similar cases based on input text
                        similar_cases, scores = find_similar_cases(
                            graph=graph,
                            case_text=case_text,
                            llm=llm,
                            embeddings=embeddings,
                            neo4j_url=neo4j_url,
                            neo4j_username=neo4j_username,
                            neo4j_password=neo4j_password
                        )
                        
                        # Display the results
                        st.subheader("Similar Legal Cases")
                        
                        if not similar_cases:
                            st.info("No exact matches found in the database. This could be because:")
                            st.markdown("""
                            - The case database is still being populated with legal cases
                            - Your query might need more specific legal terminology
                            - The system is still learning to make connections between cases
                            
                            **Suggestions:**
                            1. Try adding more specific legal details (statutes, sections, precedents)
                            2. Include case names if you know them
                            3. Use legal terminology from your jurisdiction
                            4. Try the 'Simple Search' tab with keywords
                            """)
                            
                            # Show test database status
                            st.subheader("Database Status")
                            try:
                                # Count cases in the database
                                count_query = "MATCH (c:Case) RETURN count(c) as case_count"
                                count_result = graph.query(count_query)
                                case_count = count_result[0].get('case_count', 0) if count_result else 0
                                
                                st.write(f"Currently indexed cases in database: {case_count}")
                                
                                # Show some sample cases that are available
                                if case_count > 0:
                                    st.write("Here are some cases currently available in the system:")
                                    sample_query = "MATCH (c:Case) RETURN c.title AS title, c.court AS court LIMIT 5"
                                    sample_results = graph.query(sample_query)
                                    
                                    for i, result in enumerate(sample_results):
                                        st.write(f"{i+1}. {result.get('title', 'Untitled')} ({result.get('court', 'Unknown Court')})")
                            except Exception as e:
                                st.error(f"Could not retrieve database status: {str(e)}")
                            
                            # If we have user input but no similar cases, offer to extract key legal points
                            if len(case_text.split()) >= 20:
                                st.subheader("Legal Analysis of Your Case")
                                with st.spinner("Analyzing your case details..."):
                                    analysis_prompt = f"""
                                    # Legal Case Analysis
                                    
                                    You are a legal expert analyzing a case description.
                                    
                                    ## Case Details
                                    ```
                                    {case_text[:2000]}
                                    ```
                                    
                                    ## Required Analysis
                                    1. Identify the key legal issues presented in this case
                                    2. Extract any relevant legal provisions or statutes that might apply
                                    3. Suggest what precedents might be relevant for this type of case
                                    4. Recommend search terms the user could try for finding similar cases
                                    
                                    Present your analysis in clear, concise language for a non-legal expert.
                                    """
                                    
                                    analysis = st.session_state['llm'].invoke(analysis_prompt)
                                    st.write(analysis.content)
                                    
                                    # Suggest specific search terms
                                    st.subheader("Suggested Search Terms")
                                    terms_prompt = f"""
                                    Based on the case details below, provide 5-10 specific search terms or phrases 
                                    that could help find similar cases. Return ONLY the terms separated by commas, 
                                    nothing else.
                                    
                                    Case details:
                                    {case_text[:1000]}
                                    """
                                    
                                    terms_response = st.session_state['llm'].invoke(terms_prompt)
                                    search_terms = [term.strip() for term in terms_response.content.strip().split(',')]
                                    
                                    # Display clickable search terms
                                    st.write("Try searching with these terms:")
                                    cols = st.columns(min(3, len(search_terms)))
                                    for i, term in enumerate(search_terms[:9]):  # Limit to 9 terms
                                        col_index = i % 3
                                        with cols[col_index]:
                                            st.button(term, key=f"term_{i}", help=f"Click to search for '{term}'", use_container_width=True)
                        
                        else:
                            # Format cases with similarity scores
                            formatted_cases = format_case_results(similar_cases, scores)
                            display_case_results(formatted_cases, show_similarity=True)
                            
                            # Comparative analysis with Gemini
                            case_summaries = "\n\n".join([
                                f"Case {i+1}: {case['title']}\nCourt: {case['court']}\nSimilarity: {case.get('similarity_score', 0):.2%}\nExcerpt: {case['text'][:300]+'...' if len(case['text']) > 300 else case['text']}"
                                for i, case in enumerate(formatted_cases)
                            ])
                            
                            comparison_prompt = f"""
                            # Legal Case Comparison Analysis
                            
                            ## User's Case
                            ```
                            {case_text[:1500]}
                            ```
                            
                            ## Similar Cases Found in Database
                            ```
                            {case_summaries}
                            ```
                            
                            ## Required Analysis
                            Provide a structured analysis with the following sections:
                            
                            ### 1. Key Similarities
                            - Identify common legal principles, doctrines, or statutory interpretations
                            - Note similar fact patterns or contexts
                            - Highlight matching legal reasoning
                            
                            ### 2. Significant Distinctions
                            - Point out material differences that could affect legal outcomes
                            - Identify any conflicting interpretations of law
                            - Note jurisdictional or temporal differences that matter
                            
                            ### 3. Potential Precedential Value
                            - Explain how the similar cases might influence the legal outcome of the user's situation
                            - Assess the strength of the precedents in relation to the user's case
                            - Indicate any evolution in legal interpretation over time
                            
                            Present your analysis in clear, concise language accessible to a non-legal expert.
                            Focus on substantive legal analysis rather than superficial similarities.
                            """
                            
                            with st.spinner("Analyzing case similarities and differences..."):
                                comparison = st.session_state['llm'].invoke(comparison_prompt)
                                
                                st.subheader("Comparative Legal Analysis")
                                st.write(comparison.content)
            
            with tab3:
                st.subheader("Simple Text Search")
                st.write("Search for cases without using AI - uses basic text matching")
                
                with st.form(key='simple_search_form'):
                    search_query = st.text_input("Enter search terms (e.g., 'evidence production', 'CrPC 173'):")
                    search_button = st.form_submit_button(label='Search Cases')
                
                if search_button and search_query:
                    with st.spinner("Searching for cases..."):
                        # Perform simple text search
                        search_results = simple_text_search(
                            graph=graph,
                            query_text=search_query
                        )
                        
                        # Display results
                        st.subheader("Search Results")
                        
                        if not search_results:
                            st.info("No matching cases found. Try different search terms.")
                        else:
                            # Format and display the results
                            formatted_cases = format_case_results(search_results)
                            display_case_results(formatted_cases, show_similarity=False)

            with tab4:
                st.subheader("🕸️ Knowledge Graph Visualization")
                
                if not VISUALIZATION_AVAILABLE:
                    st.error("📦 Visualization dependencies not available")
                    st.markdown("""
                    Please install the required packages:
                    ```bash
                    pip install plotly networkx
                    ```
                    """)
                    return
                
                st.markdown("Explore the connections between legal cases, judges, courts, and statutes")
                
                # Visualization controls
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.subheader("🔧 Controls")
                    
                    # Node limit control
                    node_limit = st.slider("Nodes to display", 20, 100, 50, 10)
                    
                    # Case selection for highlighting
                    st.subheader("📋 Highlight Case")
                    
                    # Get cases for selection
                    try:
                        cases_query = "MATCH (c:Case) RETURN id(c) as node_id, c.title as title LIMIT 20"
                        available_cases = graph.query(cases_query)
                        
                        if available_cases:
                            case_options = {f"{case['title'][:40]}...": case['node_id'] 
                                          for case in available_cases if case.get('title')}
                            
                            selected_case_title = st.selectbox(
                                "Choose a case:",
                                ["None"] + list(case_options.keys()),
                                key="graph_case_selector"
                            )
                            
                            selected_case_id = None
                            if selected_case_title != "None":
                                selected_case_id = case_options[selected_case_title]
                        else:
                            st.info("No cases found for selection")
                            selected_case_id = None
                            
                    except Exception as e:
                        st.error(f"Error loading cases: {e}")
                        selected_case_id = None
                
                with col2:
                    # Get and display graph data
                    with st.spinner("Loading knowledge graph data..."):
                        try:
                            nodes_data, relationships_data = get_graph_data(graph, node_limit)
                            
                            if nodes_data and relationships_data:
                                # Create network graph
                                fig, G, node_info = create_network_graph(
                                    nodes_data, 
                                    relationships_data, 
                                    selected_case_id
                                )
                                
                                if fig:
                                     # Display the interactive graph
                                     st.plotly_chart(fig, use_container_width=True)
                                     
                                     # Show network statistics
                                     stat_col1, stat_col2, stat_col3 = st.columns(3)
                                     with stat_col1:
                                         st.metric("Nodes", len(G.nodes()))
                                     with stat_col2:
                                         st.metric("Connections", len(G.edges()))
                                     with stat_col3:
                                         case_count = len([n for n in nodes_data if 'Case' in n.get('labels', [])])
                                         st.metric("Cases", case_count)
                                     
                                     # Store variables in session state for use outside this block
                                     st.session_state['graph_G'] = G
                                     st.session_state['graph_node_info'] = node_info
                                     st.session_state['graph_relationships_data'] = relationships_data
                                else:
                                    st.warning("No graph data to display")
                            else:
                                st.warning("No data found in knowledge graph")
                                
                        except Exception as e:
                            st.error(f"Error creating visualization: {e}")
                
                # Show case details if one is selected  
                if selected_case_id and 'graph_node_info' in st.session_state:
                    st.markdown("---")
                    show_case_details(graph, selected_case_id, st.session_state['graph_node_info'])
                
                # Network analysis
                if 'graph_G' in st.session_state and st.session_state['graph_G'] and len(st.session_state['graph_G'].nodes()) > 0:
                    G = st.session_state['graph_G']
                    node_info = st.session_state['graph_node_info']
                    relationships_data = st.session_state['graph_relationships_data']
                    st.markdown("---")
                    st.subheader("📊 Network Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.subheader("🏆 Most Connected")
                        centrality = nx.degree_centrality(G)
                        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        for i, (node_id, score) in enumerate(top_nodes):
                            if node_id in node_info:
                                name = node_info[node_id]['title'][:30] + "..."
                                node_type = node_info[node_id]['label']
                                connections = len(list(G.neighbors(node_id)))
                                st.write(f"{i+1}. **{name}** ({node_type}) - {connections} connections")
                    
                    with analysis_col2:
                        st.subheader("🔗 Relationships")
                        if relationships_data:
                            rel_counts = {}
                            for rel in relationships_data:
                                rel_type = rel['relationship_type']
                                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
                            
                            for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"**{rel_type}**: {count}")
                
                # Instructions
                st.markdown("---")
                st.info("""
                **How to use:**
                - 🔍 **Zoom**: Scroll to zoom in/out
                - 🖱️ **Pan**: Drag to move around
                - 🎯 **Highlight**: Select a case from the dropdown to highlight its connections
                - 📊 **Analyze**: View network statistics and most connected entities
                
                **Color coding:**
                - 🔴 **Red**: Legal Cases
                - 🟢 **Teal**: Judges  
                - 🔵 **Blue**: Courts
                - 🟢 **Green**: Statutes
                """)
    else:
        st.warning("Please connect to the Neo4j database to proceed.")

# Add this new function after the compute_cosine_similarity function (around line 760)
def format_case_results(results, similarity_scores=None):
    """Format case results in a user-friendly way with optional similarity scores"""
    formatted_results = []
    
    # Handle different types of results
    if isinstance(results, list) and len(results) > 0:
        # Check if results are raw dictionaries from Neo4j
        if isinstance(results[0], dict):
            for i, result in enumerate(results):
                formatted_case = {
                    'title': result.get('c.title', result.get('title', 'Untitled Case')),
                    'court': result.get('c.court', result.get('court', 'Unknown Court')),
                    'date': result.get('c.date', result.get('date', 'Unknown Date')),
                    'id': result.get('c.id', result.get('id', f'case_{i}')),
                    'text': result.get('c.text', result.get('text', 'No content available')),
                    'similarity_score': similarity_scores[i] if similarity_scores and i < len(similarity_scores) else None
                }
                formatted_results.append(formatted_case)
        # Handle Document objects
        elif hasattr(results[0], 'metadata'):
            for i, doc in enumerate(results):
                formatted_case = {
                    'title': doc.metadata.get('title', 'Untitled Case'),
                    'court': doc.metadata.get('court', 'Unknown Court'),
                    'date': doc.metadata.get('date', 'Unknown Date'),
                    'id': doc.metadata.get('id', f'case_{i}'),
                    'text': doc.page_content,
                    'similarity_score': similarity_scores[i] if similarity_scores and i < len(similarity_scores) else None
                }
                formatted_results.append(formatted_case)
    
    return formatted_results

def display_case_results(cases, show_similarity=False):
    """Display formatted case results in Streamlit with better UI"""
    if not cases:
        st.info("No cases found matching your query.")
        return
    
    st.markdown("---")
    
    for i, case in enumerate(cases):
        # Create a container for each case
        with st.container():
            # Display case header with title and similarity score
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {i+1}. {case['title']}")
            with col2:
                if show_similarity and case.get('similarity_score') is not None:
                    score = case['similarity_score']
                    # Color code based on similarity
                    if score > 0.8:
                        color = "🟢"
                    elif score > 0.6:
                        color = "🟡"
                    else:
                        color = "🟠"
                    st.metric("Similarity", f"{color} {score:.2%}")
            
            # Display case metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Court:** {case['court']}")
            with col2:
                st.markdown(f"**Date:** {case['date']}")
            with col3:
                st.markdown(f"**Case ID:** {case['id']}")
            
            # Display case excerpt in an expander
            with st.expander("View Case Details"):
                # Limit text length for display
                text = case['text']
                if len(text) > 1500:
                    text = text[:1500] + "..."
                st.markdown("**Case Summary:**")
                st.text(text)
            
            st.markdown("---")

def simple_text_search(graph, query_text):
    """Perform simple text search in the graph database"""
    # Extract basic keywords from query (remove common stop words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'has', 'have', 'had', 'been', 'of', 'for', 'by', 'with', 'to', 'in', 
                 'on', 'at', 'from', 'as', 'it', 'its'}
    
    # Get meaningful words from query
    keywords = [word.strip('.,;:()[]{}""\'').lower() 
               for word in query_text.split() 
               if word.strip('.,;:()[]{}""\'').lower() not in stop_words and len(word) > 2]
    
    conditions = []
    for keyword in keywords[:5]:  # Limit to first 5 keywords
        if keyword:
            keyword = keyword.replace("'", "\\'")
            conditions.append(f"toLower(c.text) CONTAINS toLower('{keyword}')")
    
    if conditions:
        where_clause = " OR ".join(conditions)
        cypher = f"""
        MATCH (c:Case)
        WHERE {where_clause}
        RETURN c.id as id, c.title as title, c.court as court, c.date as date, c.text as text
        LIMIT 5
        """
    else:
        cypher = """
        MATCH (c:Case)
        RETURN c.id as id, c.title as title, c.court as court, c.date as date, c.text as text
        LIMIT 5
        """
    
    results = graph.query(cypher)
    return results

if __name__ == "__main__":
    main()


