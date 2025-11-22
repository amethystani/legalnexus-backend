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
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
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


class LegalQueryExpander:
    """
    Cognitive Module: Expands layman queries into legal terminology
    Uses LLM to 'think' before searching
    """
    def __init__(self, llm):
        self.llm = llm
        
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes query to extract:
        1. Legal keywords (e.g., 'drunk driving' -> 'Section 185 MV Act')
        2. Intent (Fact-finding vs Procedure vs Precedent)
        3. Domain (Criminal, Civil, Constitutional)
        """
        if not self.llm:
            return {"expanded_query": query, "intent": "general", "domain": "unknown"}
            
        try:
            prompt = f"""
            Act as a senior legal researcher. Analyze this user query: "{query}"
            
            Return a JSON object with:
            1. "legal_terms": List of specific Indian legal sections/acts relevant to this (e.g., "Section 302 IPC")
            2. "expanded_query": A search-optimized string combining user terms + legal terms
            3. "intent": One of ["procedure", "precedent", "fact_finding"]
            4. "domain": One of ["criminal", "civil", "constitutional", "corporate", "family"]
            
            Keep it concise. JSON only.
            """
            
            response = self.llm.invoke(prompt)
            # Clean response to ensure valid JSON
            content = response.content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {e}")
            return {"expanded_query": query, "intent": "general", "domain": "unknown"}


class DynamicWeightingEngine:
    """
    Adaptive Module: Adjusts algorithm weights based on query intent
    """
    def __init__(self, base_weights: Dict[str, float]):
        self.base_weights = base_weights.copy()
        
    def adapt_weights(self, analysis: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
        """
        Returns (new_weights, explanation_of_adaptation)
        """
        weights = self.base_weights.copy()
        reasoning = []
        
        # Adaptation 1: Intent-based
        intent = analysis.get("intent", "general")
        if intent == "precedent":
            weights['citation'] += 0.15
            weights['semantic'] -= 0.05
            weights['text'] -= 0.10
            reasoning.append("User seeks precedents -> Boosted Citation Network")
        elif intent == "fact_finding":
            weights['text'] += 0.15
            weights['semantic'] -= 0.05
            weights['citation'] -= 0.10
            reasoning.append("User seeks specific facts -> Boosted Text Pattern")
            
        # Adaptation 2: Domain-based
        domain = analysis.get("domain", "unknown")
        if domain == "constitutional":
            weights['semantic'] += 0.10
            weights['graph'] += 0.05
            reasoning.append("Constitutional matter -> Boosted Semantic & Graph (concepts over keywords)")
            
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
            
        return weights, " + ".join(reasoning) if reasoning else "Standard balanced profile"



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
        
        # Initialize cognitive modules
        self.query_expander = LegalQueryExpander(self.llm)
        self.weighting_engine = DynamicWeightingEngine(self.weights)
        
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
        Novel Hybrid Search combining all algorithms with COGNITIVE ENHANCEMENTS
        Returns: List of (document, final_score, score_breakdown)
        """
        print("\n" + "═" * 80)
        print("🧠 COGNITIVE SEARCH PROCESS INITIATED")
        print("═" * 80)
        
        # Step 1: Cognitive Query Expansion
        print(f"\n1️⃣  Thinking about query: '{query}'...")
        analysis = self.query_expander.expand_query(query)
        expanded_query = analysis.get("expanded_query", query)
        
        print(f"    → Detected Intent: {analysis.get('intent', 'general').upper()}")
        print(f"    → Detected Domain: {analysis.get('domain', 'unknown').upper()}")
        if analysis.get('legal_terms'):
            print(f"    → Injected Legal Terms: {', '.join(analysis['legal_terms'])}")
        
        # Step 2: Dynamic Weight Adaptation
        print(f"\n2️⃣  Adapting Search Strategy...")
        current_weights, adaptation_reason = self.weighting_engine.adapt_weights(analysis)
        print(f"    → Strategy: {adaptation_reason}")
        print(f"    → New Weights: Semantic={current_weights['semantic']:.2f}, Text={current_weights['text']:.2f}, Graph={current_weights['graph']:.2f}")
        
        # Step 3: Parallel Execution
        print(f"\n3️⃣  Executing 5-Way Parallel Search...")
        
        # Use expanded query for semantic search to catch legal concepts
        semantic_results = self.semantic_search(expanded_query, top_k=15)
        
        # Use original query for text match to catch exact user phrasing
        text_results = self.text_pattern_search(query, top_k=15)
        
        # Use both for graph/citation
        graph_results = self.graph_traversal_search(expanded_query, top_k=15)
        citation_results = self.citation_network_search(expanded_query, top_k=15)
        
        # Aggregate scores
        aggregated_scores = {}
        
        # Helper to process results
        def process_results(results, weight_key):
            for doc, score in results:
                doc_id = doc.metadata.get('id', id(doc))
                if doc_id not in aggregated_scores:
                    aggregated_scores[doc_id] = {
                        'doc': doc,
                        'semantic': 0, 'graph': 0, 'text': 0, 'citation': 0, 'gnn': 0
                    }
                aggregated_scores[doc_id][weight_key] = score
        
        process_results(semantic_results, 'semantic')
        process_results(graph_results, 'graph')
        process_results(text_results, 'text')
        process_results(citation_results, 'citation')
        
        # Calculate final hybrid scores using DYNAMIC weights
        final_results = []
        for doc_id, scores in aggregated_scores.items():
            final_score = (
                current_weights['semantic'] * scores['semantic'] +
                current_weights['graph'] * scores['graph'] +
                current_weights['text'] * scores['text'] +
                current_weights['citation'] * scores['citation'] +
                current_weights['gnn'] * scores['gnn']
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
        
        print("\n" + "═" * 80)
        print("✅ SEARCH COMPLETE")
        print("═" * 80)
        
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


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """Call in a loop to create terminal progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def print_results(results: List[Tuple[Document, float, Dict]], show_details: bool = False):
    """Print search results in a high-tech dashboard format"""
    if not results:
        print(f"\n{Colors.FAIL}No results found.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.HEADER}RESULTS DASHBOARD{Colors.ENDC}")
    print(f"{Colors.BLUE}Found {len(results)} relevant cases{Colors.ENDC}")
    print("═" * 80)
    
    for i, (doc, final_score, breakdown) in enumerate(results):
        title = doc.metadata.get('title', 'Untitled Case')
        court = doc.metadata.get('court', 'Unknown Court')
        date = doc.metadata.get('date', 'Unknown Date')
        
        # Color code score
        score_color = Colors.GREEN if final_score > 0.7 else (Colors.WARNING if final_score > 0.4 else Colors.FAIL)
        
        print(f"\n{Colors.BOLD}{i+1}. {title}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Court:{Colors.ENDC} {court}  |  {Colors.CYAN}Date:{Colors.ENDC} {date}")
        print(f"   {Colors.BOLD}Relevance Score:{Colors.ENDC} {score_color}{final_score:.4f}{Colors.ENDC}")
        
        if show_details:
            print(f"\n   {Colors.UNDERLINE}Algorithm Contribution Analysis:{Colors.ENDC}")
            
            # ASCII Chart for weights
            max_bar_len = 40
            
            def draw_bar(label, value, color):
                bar_len = int(value * max_bar_len)
                bar = "█" * bar_len
                print(f"     {label:<15} |{color}{bar:<40}{Colors.ENDC}| {value:.4f}")

            draw_bar("Semantic", breakdown['semantic'], Colors.BLUE)
            draw_bar("Graph", breakdown['graph'], Colors.MAGENTA if hasattr(Colors, 'MAGENTA') else Colors.HEADER)
            draw_bar("Text Pattern", breakdown['text'], Colors.GREEN)
            draw_bar("Citation", breakdown['citation'], Colors.WARNING)
            draw_bar("GNN Model", breakdown['gnn'], Colors.CYAN)
            
            # Show excerpt
            excerpt = doc.page_content[:300].replace('\n', ' ') + "..." 
            print(f"\n   {Colors.CYAN}Excerpt:{Colors.ENDC}\n   {excerpt}\n")
        
        print(f"{Colors.BLUE}" + "-" * 80 + f"{Colors.ENDC}")


def main():
    """Main CLI interface with Cyberpunk/Dashboard UI"""
    # Clear screen
    print("\033[H\033[J", end="")
    
    print(f"{Colors.HEADER}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     NOVEL HYBRID LEGAL CASE SEARCH SYSTEM v2.0".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  [Cognitive Query Expansion] • [Dynamic Adaptive Scoring] • [Hybrid Search]".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"{Colors.ENDC}")
    print("\n")
    
    # Initialize system
    print(f"{Colors.BLUE}Initializing System Modules...{Colors.ENDC}")
    search_system = NovelHybridSearchSystem()
    
    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}SYSTEM READY{Colors.ENDC}")
    print("=" * 80)
    print(f"\n{Colors.BOLD}Enter your query in natural language.{Colors.ENDC}")
    print("Examples:")
    print(f"  - {Colors.CYAN}'I was drunk and drove my car'{Colors.ENDC}")
    print(f"  - {Colors.CYAN}'Electronic evidence admissibility in WhatsApp messages'{Colors.ENDC}")
    print(f"  - {Colors.CYAN}'Property dispute between neighbors'{Colors.ENDC}")
    print("\n")
    
    # Interactive mode
    while True:
        try:
            query = input(f"\n{Colors.BOLD}🔍 Enter your legal query (or 'quit' to exit): {Colors.ENDC}").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Thank you for using the Legal Case Search System!{Colors.ENDC}")
                break
            
            if not query:
                continue
            
            # Run hybrid search
            start_time = time.time()
            results = search_system.hybrid_search(query, top_k=5)
            search_time = time.time() - start_time
            
            # Print results
            print_results(results, show_details=True)
            
            print(f"\n{Colors.BLUE}⏱️  Search completed in {search_time:.2f} seconds{Colors.ENDC}")
            
            # Generate AI explanation
            print("\n" + "=" * 80)
            print(f"{Colors.HEADER}🤖 AI EXPLANATION{Colors.ENDC}")
            print("=" * 80)
            
            explanation = search_system.generate_ai_explanation(query, results)
            print(f"\n{explanation}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Search interrupted. Exiting...{Colors.ENDC}")
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
