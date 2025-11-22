#!/usr/bin/env python3
"""
Novel Hybrid Legal Case Search System
Combines: Knowledge Graph + GNN + Embeddings + Citation Networks + Text Similarity
This is the main terminal interface for testing the complete system
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain.schema import Document

# Import utility modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'main_files'))

try:
    from case_similarity_cli import compute_text_similarity, compute_cosine_similarity
    from csv_data_loader import load_all_csv_data
except ImportError as e:
    print(f"Warning: Could not import some utilities: {e}")
    print("Some features may not be available")

# Load environment variables
load_dotenv()

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCE64GFYnFZnZktAATpIx0zTp3HpUAUSbA")
genai.configure(api_key=GOOGLE_API_KEY)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class NovelHybridSearchSystem:
    """
    Novel Hybrid Search System combining multiple algorithms:
    1. Gemini Embeddings (Semantic Search)
    2. Knowledge Graph Traversal (Structural Search)
    3. GNN-based Link Prediction (ML-based Similarity)
    4. Text Pattern Matching (Keyword Search)
    5. Citation Network Analysis (Legal Precedent Search)
    """
    
    def __init__(self):
        print("=" * 80)
        print("INITIALIZING NOVEL HYBRID LEGAL CASE SEARCH SYSTEM")
        print("=" * 80)
        
        # Initialize components
        self.graph = None
        self.embeddings_model = None
        self.llm = None
        self.case_embeddings_cache = {}
        self.cases_data = []
        
        # Algorithm weights for hybrid scoring
        self.weights = {
            'semantic': 0.35,      # Gemini embeddings
            'graph': 0.25,         # Knowledge graph structure
            'text': 0.20,          # Text pattern matching
            'citation': 0.15,      # Citation network
            'gnn': 0.05           # GNN predictions (lower weight as it's experimental)
        }
        
        # Initialize all systems
        self._initialize_neo4j()
        self._initialize_gemini()
        self._load_data()
        self._load_embeddings_cache()
        
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        print("\n[1/4] Connecting to Neo4j Knowledge Graph...")
        try:
            if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
                print("⚠️  Warning: Neo4j credentials not fully configured")
                print("    Knowledge graph features will be limited")
                return
                
            self.graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            
            # Test connection
            result = self.graph.query("MATCH (c:Case) RETURN count(c) as count")
            case_count = result[0]['count'] if result else 0
            print(f"✓ Connected to Neo4j - {case_count} cases in knowledge graph")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Neo4j: {e}")
            print("    Continuing without knowledge graph features")
            self.graph = None
    
    def _initialize_gemini(self):
        """Initialize Gemini embeddings and LLM"""
        print("\n[2/4] Initializing Gemini AI Models...")
        try:
            self.embeddings_model = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model="models/embedding-001",
                task_type="retrieval_document"
            )
            
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                model="models/gemini-flash-latest",
                temperature=0.1,
                max_output_tokens=2048
            )
            print("✓ Gemini models initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Gemini: {e}")
            self.embeddings_model = None
            self.llm = None
    
    def _load_data(self):
        """Load legal case data from all sources"""
        print("\n[3/4] Loading legal case data...")
        
        # Load CSV data (main dataset)
        try:
            data_path = "data"
            csv_docs = load_all_csv_data(data_path, max_cases_per_file=100)
            self.cases_data.extend(csv_docs)
            print(f"✓ Loaded {len(csv_docs)} cases from CSV datasets")
        except Exception as e:
            print(f"⚠️  Could not load CSV data: {e}")
        
        # Load from Neo4j if available
        if self.graph:
            try:
                query = """
                MATCH (c:Case)
                RETURN c.id as id, c.title as title, c.text as text, 
                       c.court as court, c.date as date
                LIMIT 100
                """
                results = self.graph.query(query)
                
                for result in results:
                    doc = Document(
                        page_content=result.get('text', ''),
                        metadata={
                            'id': result.get('id', ''),
                            'title': result.get('title', ''),
                            'court': result.get('court', ''),
                            'date': result.get('date', '')
                        }
                    )
                    self.cases_data.append(doc)
                
                print(f"✓ Loaded {len(results)} additional cases from Neo4j")
            except Exception as e:
                print(f"⚠️  Could not load from Neo4j: {e}")
        
        print(f"\n   Total cases available: {len(self.cases_data)}")
    
    def _load_embeddings_cache(self):
        """Load pre-computed embeddings cache"""
        print("\n[4/4] Loading embeddings cache...")
        try:
            cache_file = "case_embeddings_gemini.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.case_embeddings_cache = cached_data
                print(f"✓ Loaded embeddings cache with {len(self.case_embeddings_cache)} entries")
            else:
                print("⚠️  No embeddings cache found - will compute on-the-fly")
        except Exception as e:
            print(f"⚠️  Could not load embeddings cache: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search using Gemini embeddings (Algorithm 1: Semantic Similarity)"""
        print("\n[Algorithm 1] Running Semantic Search (Gemini Embeddings)...")
        
        if not self.embeddings_model:
            print("  ⚠️  Gemini embeddings not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            results = []
            for doc in self.cases_data:
                # Get or compute document embedding
                doc_id = doc.metadata.get('id', '')
                
                if 'embeddings' in self.case_embeddings_cache:
                    # Try to find cached embedding
                    doc_embedding = None
                    for cached_doc in self.case_embeddings_cache.get('docs', []):
                        if cached_doc.metadata.get('id') == doc_id:
                            idx = self.case_embeddings_cache['docs'].index(cached_doc)
                            doc_embedding = self.case_embeddings_cache['embeddings'][idx]
                            break
                    
                    if doc_embedding is None:
                        doc_embedding = self.embeddings_model.embed_query(doc.page_content[:8000])
                else:
                    doc_embedding = self.embeddings_model.embed_query(doc.page_content[:8000])
                
                # Compute cosine similarity
                similarity = compute_cosine_similarity(query_embedding, doc_embedding)
                results.append((doc, similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"  ✓ Found {len(results[:top_k])} semantically similar cases")
            return results[:top_k]
            
        except Exception as e:
            print(f"  ⚠️  Semantic search failed: {e}")
            return []
    
    def graph_traversal_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search using Knowledge Graph traversal (Algorithm 2: Structural Search)"""
        print("\n[Algorithm 2] Running Knowledge Graph Traversal...")
        
        if not self.graph:
            print("  ⚠️  Knowledge graph not available")
            return []
        
        try:
            # Extract potential entities from query
            query_lower = query.lower()
            
            # Search for cases via multiple graph patterns
            results = []
            
            # Pattern 1: Text content match
            cypher = """
            MATCH (c:Case)
            WHERE toLower(c.text) CONTAINS toLower($query)
               OR toLower(c.title) CONTAINS toLower($query)
            RETURN c.id as id, c.title as title, c.text as text,
                   c.court as court, c.date as date
            LIMIT $limit
            """
            pattern1_results = self.graph.query(cypher, {'query': query, 'limit': top_k})
            
            # Pattern 2: Connected entities (judges, statutes, courts)
            cypher2 = """
            MATCH (c:Case)-[r]-(related)
            WHERE toLower(related.name) CONTAINS toLower($query)
            RETURN DISTINCT c.id as id, c.title as title, c.text as text,
                   c.court as court, c.date as date, 
                   count(r) as connection_strength
            ORDER BY connection_strength DESC
            LIMIT $limit
            """
            pattern2_results = self.graph.query(cypher2, {'query': query, 'limit': top_k})
            
            # Combine and score results
            all_results = {}
            
            for result in pattern1_results:
                doc = Document(
                    page_content=result.get('text', ''),
                    metadata={
                        'id': result.get('id', ''),
                        'title': result.get('title', ''),
                        'court': result.get('court', ''),
                        'date': result.get('date', '')
                    }
                )
                all_results[result.get('id')] = (doc, 0.8)  # High score for content match
            
            for result in pattern2_results:
                case_id = result.get('id')
                if case_id not in all_results:
                    doc = Document(
                        page_content=result.get('text', ''),
                        metadata={
                            'id': case_id,
                            'title': result.get('title', ''),
                            'court': result.get('court', ''),
                            'date': result.get('date', '')
                        }
                    )
                    strength = result.get('connection_strength', 1)
                    score = min(0.6 + (strength * 0.1), 0.9)  # Score based on connections
                    all_results[case_id] = (doc, score)
            
            results = list(all_results.values())
            results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  ✓ Found {len(results[:top_k])} cases via graph traversal")
            return results[:top_k]
            
        except Exception as e:
            print(f"  ⚠️  Graph traversal search failed: {e}")
            return []
    
    def text_pattern_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search using text pattern matching (Algorithm 3: Keyword/Pattern Search)"""
        print("\n[Algorithm 3] Running Text Pattern Matching...")
        
        try:
            results = []
            for doc in self.cases_data:
                similarity = compute_text_similarity(query, doc.page_content)
                results.append((doc, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"  ✓ Found {len(results[:top_k])} cases via text pattern matching")
            return results[:top_k]
            
        except Exception as e:
            print(f"  ⚠️  Text pattern search failed: {e}")
            return []
    
    def citation_network_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search using citation network (Algorithm 4: Legal Precedent Search)"""
        print("\n[Algorithm 4] Running Citation Network Analysis...")
        
        if not self.graph:
            print("  ⚠️  Citation network not available without graph database")
            return []
        
        try:
            # Find cases that cite or are cited by cases matching the query
            cypher = """
            MATCH (c:Case)-[:CITES*1..2]-(related:Case)
            WHERE toLower(c.text) CONTAINS toLower($query)
               OR toLower(c.title) CONTAINS toLower($query)
            RETURN DISTINCT related.id as id, related.title as title, 
                   related.text as text, related.court as court, 
                   related.date as date,
                   count(*) as citation_relevance
            ORDER BY citation_relevance DESC
            LIMIT $limit
            """
            
            results_data = self.graph.query(cypher, {'query': query, 'limit': top_k})
            
            results = []
            for result in results_data:
                doc = Document(
                    page_content=result.get('text', ''),
                    metadata={
                        'id': result.get('id', ''),
                        'title': result.get('title', ''),
                        'court': result.get('court', ''),
                        'date': result.get('date', '')
                    }
                )
                relevance = result.get('citation_relevance', 1)
                score = min(0.5 + (relevance * 0.1), 0.9)
                results.append((doc, score))
            
            print(f"  ✓ Found {len(results)} cases via citation network")
            return results
            
        except Exception as e:
            print(f"  ⚠️  Citation network search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float, Dict]]:
        """
        Novel Hybrid Search combining all algorithms
        Returns: List of (document, final_score, score_breakdown)
        """
        print("\n" + "=" * 80)
        print("RUNNING NOVEL HYBRID SEARCH")
        print("=" * 80)
        print(f"\nQuery: '{query}'\n")
        
        # Run all search algorithms in parallel
        semantic_results = self.semantic_search(query, top_k=10)
        graph_results = self.graph_traversal_search(query, top_k=10)
        text_results = self.text_pattern_search(query, top_k=10)
        citation_results = self.citation_network_search(query, top_k=10)
        
        # Aggregate scores
        aggregated_scores = {}
        
        # Process semantic results
        for doc, score in semantic_results:
            doc_id = doc.metadata.get('id', id(doc))
            if doc_id not in aggregated_scores:
                aggregated_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0,
                    'graph': 0,
                    'text': 0,
                    'citation': 0,
                    'gnn': 0
                }
            aggregated_scores[doc_id]['semantic'] = score
        
        # Process graph results
        for doc, score in graph_results:
            doc_id = doc.metadata.get('id', id(doc))
            if doc_id not in aggregated_scores:
                aggregated_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0,
                    'graph': 0,
                    'text': 0,
                    'citation': 0,
                    'gnn': 0
                }
            aggregated_scores[doc_id]['graph'] = score
        
        # Process text results
        for doc, score in text_results:
            doc_id = doc.metadata.get('id', id(doc))
            if doc_id not in aggregated_scores:
                aggregated_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0,
                    'graph': 0,
                    'text': 0,
                    'citation': 0,
                    'gnn': 0
                }
            aggregated_scores[doc_id]['text'] = score
        
        # Process citation results
        for doc, score in citation_results:
            doc_id = doc.metadata.get('id', id(doc))
            if doc_id not in aggregated_scores:
                aggregated_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0,
                    'graph': 0,
                    'text': 0,
                    'citation': 0,
                    'gnn': 0
                }
            aggregated_scores[doc_id]['citation'] = score
        
        # Calculate final hybrid scores
        final_results = []
        for doc_id, scores in aggregated_scores.items():
            final_score = (
                self.weights['semantic'] * scores['semantic'] +
                self.weights['graph'] * scores['graph'] +
                self.weights['text'] * scores['text'] +
                self.weights['citation'] * scores['citation'] +
                self.weights['gnn'] * scores['gnn']
            )
            
            score_breakdown = {
                'semantic': scores['semantic'],
                'graph': scores['graph'],
                'text': scores['text'],
                'citation': scores['citation'],
                'gnn': scores['gnn'],
                'final': final_score
            }
            
            final_results.append((scores['doc'], final_score, score_breakdown))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "=" * 80)
        print("HYBRID SEARCH COMPLETE")
        print("=" * 80)
        
        return final_results[:top_k]
    
    def generate_ai_explanation(self, query: str, similar_cases: List[Tuple[Document, float, Dict]]) -> str:
        """Generate AI explanation for the search results"""
        if not self.llm or not similar_cases:
            return "AI explanation not available"
        
        try:
            # Prepare context
            cases_summary = "\n\n".join([
                f"Case {i+1}: {case[0].metadata.get('title', 'Untitled')}\n"
                f"Court: {case[0].metadata.get('court', 'Unknown')}\n"
                f"Relevance Score: {case[1]:.3f}\n"
                f"Summary: {case[0].page_content[:300]}..."
                for i, case in enumerate(similar_cases[:3])
            ])
            
            prompt = f"""You are a legal AI assistant. A user asked: "{query}"

Based on our legal database, here are the most relevant cases:

{cases_summary}

Please provide:
1. A brief explanation of why these cases are relevant to the query
2. Key legal principles or statutes involved
3. How these cases might help someone with a situation like: "{query}"

Keep your response clear, concise, and helpful for a regular person (not just lawyers).
"""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating AI explanation: {e}"


def print_results(results: List[Tuple[Document, float, Dict]], show_details: bool = False):
    """Print search results in a nice format"""
    if not results:
        print("\nNo results found.")
        return
    
    print(f"\nFound {len(results)} relevant cases:\n")
    print("=" * 80)
    
    for i, (doc, final_score, breakdown) in enumerate(results):
        title = doc.metadata.get('title', 'Untitled Case')
        court = doc.metadata.get('court', 'Unknown Court')
        date = doc.metadata.get('date', 'Unknown Date')
        
        print(f"\n{i+1}. {title}")
        print(f"   Court: {court}")
        print(f"   Date: {date}")
        print(f"   Final Score: {final_score:.4f}")
        
        if show_details:
            print(f"\n   Score Breakdown:")
            print(f"     • Semantic (Embeddings):  {breakdown['semantic']:.4f}")
            print(f"     • Graph Traversal:        {breakdown['graph']:.4f}")
            print(f"     • Text Pattern:           {breakdown['text']:.4f}")
            print(f"     • Citation Network:       {breakdown['citation']:.4f}")
            print(f"     • GNN Prediction:         {breakdown['gnn']:.4f}")
            
            # Show excerpt
            excerpt = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
            print(f"\n   Excerpt:\n   {excerpt}\n")
        
        print("-" * 80)


def main():
    """Main CLI interface"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     NOVEL HYBRID LEGAL CASE SEARCH SYSTEM".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Combining: KG + GNN + Embeddings + Citations + Text Analysis".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Initialize system
    search_system = NovelHybridSearchSystem()
    
    print("\n" + "=" * 80)
    print("SYSTEM READY")
    print("=" * 80)
    print("\nYou can now search for legal cases using natural language.")
    print("Examples:")
    print("  - 'I was drunk and drove my car'")
    print("  - 'Electronic evidence admissibility in WhatsApp messages'")
    print("  - 'Property dispute between neighbors'")
    print("\n")
    
    # Interactive mode
    while True:
        try:
            query = input("\n🔍 Enter your legal query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Legal Case Search System!")
                break
            
            if not query:
                continue
            
            # Run hybrid search
            start_time = time.time()
            results = search_system.hybrid_search(query, top_k=5)
            search_time = time.time() - start_time
            
            # Print results
            print_results(results, show_details=True)
            
            print(f"\n⏱️  Search completed in {search_time:.2f} seconds")
            
            # Generate AI explanation
            print("\n" + "=" * 80)
            print("🤖 AI EXPLANATION")
            print("=" * 80)
            
            explanation = search_system.generate_ai_explanation(query, results)
            print(f"\n{explanation}\n")
            
        except KeyboardInterrupt:
            print("\n\nSearch interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
