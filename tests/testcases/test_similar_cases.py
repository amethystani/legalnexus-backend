#!/usr/bin/env python3
import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.schema import Document

class TestSimilarCaseFunctionality:
    """Test the similar case functionality of the Legal Knowledge Graph"""

    def __init__(self):
        """Initialize the test environment"""
        print("=" * 50)
        print("Testing Similar Case Functionality")
        print("=" * 50)
        
        print("Setting up test environment...")
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j connection details
        self.neo4j_url = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_url, self.neo4j_username, self.neo4j_password]):
            print("Error: Neo4j connection details not found in environment variables.")
            print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
            sys.exit(1)
        
        # Connect to Neo4j
        try:
            self.graph = Neo4jGraph(
                url=self.neo4j_url, 
                username=self.neo4j_username, 
                password=self.neo4j_password
            )
            print(f"Connected to Neo4j database: {self.neo4j_url}")
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            sys.exit(1)
        
        # Initialize OpenAI components
        try:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                print("Error: OPENAI_API_KEY not found in environment variables.")
                sys.exit(1)
                
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-4o")
            print("Initialized OpenAI components")
        except Exception as e:
            print(f"Error initializing OpenAI components: {e}")
            sys.exit(1)
            
        # Verify that cases exist in the database
        self.verify_test_cases_loaded()

    def verify_test_cases_loaded(self):
        """Verify that test cases are loaded in the database"""
        try:
            # Count the number of case nodes in the database
            count_query = "MATCH (c:Case) RETURN count(c) as case_count"
            count_result = self.graph.query(count_query)
            case_count = count_result[0].get('case_count', 0) if count_result else 0
            
            if case_count == 0:
                print("Warning: No cases found in the database. Tests may fail.")
                print("Please run the data loading process to populate the database before testing.")
            else:
                print(f"Found {case_count} cases in the database")
                
            # Show a sample of cases
            if case_count > 0:
                sample_query = "MATCH (c:Case) RETURN c.title AS title LIMIT 5"
                sample_results = self.graph.query(sample_query)
                if sample_results:
                    print("Sample cases in database:")
                    for i, result in enumerate(sample_results):
                        print(f"  {i+1}. {result.get('title', 'Untitled')}")
        except Exception as e:
            print(f"Warning: Could not verify test cases: {e}")

    def test_electronic_evidence_scenario(self):
        """Test finding similar cases for electronic evidence scenario"""
        print("\n=== Testing Electronic Evidence Scenario ===")
        
        query = """
        I need to submit some WhatsApp chat screenshots as evidence in a criminal case.
        The defense is arguing that these electronic records are inadmissible without proper
        certification. Do I need a certificate under Section 65B of the Evidence Act?
        What's the legal position on this type of digital evidence?
        """
        print("Query:")
        print(query)
        
        print("\nSearching for similar cases...")
        
        # Import simple_text_search from kg module
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from kg import find_similar_cases, simple_text_search
            
            # First try vector search
            similar_docs = find_similar_cases(
                graph=self.graph,
                case_text=query,
                llm=self.llm,
                embeddings=self.embeddings,
                neo4j_url=self.neo4j_url,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password
            )
            
            print(f"\nFound {len(similar_docs)} similar cases:")
            
            if not similar_docs:
                print("\nNo similar cases found with vector search. Trying simple text search...")
                # Extract key terms from the query for text search
                key_terms = self.extract_key_terms(query)
                
                # Use simple text search with the key terms
                search_query = " ".join(key_terms)
                results = simple_text_search(self.graph, search_query)
                
                print(f"Found {len(results)} results with text search:")
                for i, result in enumerate(results):
                    print(f"{i+1}. {result.get('title', 'Unknown')} - {result.get('court', 'Unknown Court')}")
            else:
                # Show the similar cases
                for i, doc in enumerate(similar_docs):
                    print(f"{i+1}. {doc.metadata.get('title', 'Unknown')} - {doc.metadata.get('court', 'Unknown Court')}")
            
            return similar_docs
        except Exception as e:
            print(f"Error during electronic evidence test: {e}")
            return []

    def test_pension_rights_scenario(self):
        """Test finding similar cases for pension rights scenario"""
        print("\n=== Testing Pension Rights Scenario ===")
        
        query = """
        I worked for a government bank for 25 years, and they're now denying me some of my
        pension benefits because they claim I didn't complete certain formalities before retiring.
        Can they legally withhold part of my pension? I've heard pension is a fundamental right.
        """
        print("Query:")
        print(query)
        
        print("\nSearching for similar cases...")
        
        # Import simple_text_search from kg module
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from kg import find_similar_cases, simple_text_search
            
            # First try vector search
            similar_docs = find_similar_cases(
                graph=self.graph,
                case_text=query,
                llm=self.llm,
                embeddings=self.embeddings,
                neo4j_url=self.neo4j_url,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password
            )
            
            print(f"\nFound {len(similar_docs)} similar cases:")
            
            if not similar_docs:
                print("\nNo similar cases found with vector search. Trying simple text search...")
                # Extract key terms from the query for text search
                key_terms = self.extract_key_terms(query)
                
                # Use simple text search with the key terms
                search_query = " ".join(key_terms)
                results = simple_text_search(self.graph, search_query)
                
                print(f"Found {len(results)} results with text search:")
                for i, result in enumerate(results):
                    print(f"{i+1}. {result.get('title', 'Unknown')} - {result.get('court', 'Unknown Court')}")
            else:
                # Show the similar cases
                for i, doc in enumerate(similar_docs):
                    print(f"{i+1}. {doc.metadata.get('title', 'Unknown')} - {doc.metadata.get('court', 'Unknown Court')}")
            
            return similar_docs
        except Exception as e:
            print(f"Error during pension rights test: {e}")
            return []

    def test_custom_scenario(self, query):
        """Test finding similar cases for a custom scenario"""
        print("\n=== Testing Custom Scenario ===")
        
        print("Query:")
        print(query)
        
        print("\nSearching for similar cases...")
        
        # Import simple_text_search from kg module
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from kg import find_similar_cases, simple_text_search
            
            # First try vector search
            similar_docs = find_similar_cases(
                graph=self.graph,
                case_text=query,
                llm=self.llm,
                embeddings=self.embeddings,
                neo4j_url=self.neo4j_url,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password
            )
            
            print(f"\nFound {len(similar_docs)} similar cases:")
            
            if not similar_docs:
                print("\nNo similar cases found with vector search. Trying simple text search...")
                # Extract key terms from the query for text search
                key_terms = self.extract_key_terms(query)
                
                # Use simple text search with the key terms
                search_query = " ".join(key_terms)
                results = simple_text_search(self.graph, search_query)
                
                print(f"Found {len(results)} results with text search:")
                for i, result in enumerate(results):
                    print(f"{i+1}. {result.get('title', 'Unknown')} - {result.get('court', 'Unknown Court')}")
            else:
                # Show the similar cases
                for i, doc in enumerate(similar_docs):
                    print(f"{i+1}. {doc.metadata.get('title', 'Unknown')} - {doc.metadata.get('court', 'Unknown Court')}")
            
            return similar_docs
        except Exception as e:
            print(f"Error during custom scenario test: {e}")
            return []

    def extract_key_terms(self, text):
        """Extract key terms from text for search"""
        try:
            # Use a simple approach - split text into words and filter out common words
            words = text.lower().split()
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                         'your', 'yours', 'their', 'they', 'this', 'that', 'these', 'those', 
                         'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                         'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
                         'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 
                         'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                         'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                         'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 
                         'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
            
            # Filter out stopwords and short words
            key_terms = [word for word in words if word not in stopwords and len(word) > 2]
            
            # Focus on longer words (potential legal terms)
            key_terms.sort(key=len, reverse=True)
            
            # Keep specific legal terms even if they're short
            legal_terms = ["law", "act", "ipc", "cpc", "crpc", "case", "suit"]
            for term in legal_terms:
                if term in words and term not in key_terms:
                    key_terms.append(term)
            
            # Return the most relevant terms (up to 10)
            return key_terms[:10]
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return ["evidence", "legal", "case", "court"]  # Default fallback

def main():
    """Run the test script"""
    test = TestSimilarCaseFunctionality()
    
    # Run predefined test scenarios
    test.test_electronic_evidence_scenario()
    test.test_pension_rights_scenario()
    
    # Offer to run custom scenario
    print("\nNow you can test your own scenario:\n")
    user_query = input("Enter your legal situation or question (or press Enter to skip): ")
    if user_query.strip():
        test.test_custom_scenario(user_query)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 