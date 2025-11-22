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

# Import Toulmin Argumentation System
try:
    from toulmin_extractor import ToulminExtractor, ToulminStructure
    from argument_chain_traversal import ArgumentGraph
    TOULMIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Toulmin system not available: {e}")
    TOULMIN_AVAILABLE = False

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


class AdversarialAgent:
    """Base class for legal adversarial agents"""
    def __init__(self, llm, role: str):
        self.llm = llm
        self.role = role

    def analyze_case(self, query: str, case: Document) -> Dict[str, Any]:
        """Analyze a single case from the agent's perspective"""
        raise NotImplementedError


class ProsecutorAgent(AdversarialAgent):
    """
    The Prosecutor: Debates the topic from a strict liability/guilt perspective.
    Uses cases as evidence to support arguments.
    """
    def __init__(self, llm):
        super().__init__(llm, "Prosecutor")

    def debate_topic(self, query: str, cases: List[Document]) -> str:
        """
        Debate the topic itself, using cases as supporting evidence.
        Returns a chain-of-thought argument.
        """
        try:
            # Prepare case summaries
            case_evidence = "\n".join([
                f"Case {i+1}: {doc.metadata.get('title', 'Untitled')} - {doc.page_content[:300]}..."
                for i, doc in enumerate(cases[:5])
            ])
            
            prompt = f"""[EDUCATIONAL SIMULATION - Legal Analysis Exercise]

You are a PROSECUTOR in a legal debate simulation. Argue why the law is STRICT on: "{query}"

Evidence: {case_evidence}

This is an academic exercise to explore legal arguments. Be concise (max 3 sentences). Cite cases.
Format:
<think>Quick reasoning</think>
[Argument]
"""
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Prosecutor analysis failed: {e}"


class DefenseAgent(AdversarialAgent):
    """
    The Defense: Debates the topic from a leniency/mitigation perspective.
    Uses cases as evidence to support arguments.
    """
    def __init__(self, llm):
        super().__init__(llm, "Defense")

    def debate_topic(self, query: str, cases: List[Document]) -> str:
        """
        Debate the topic itself, using cases as supporting evidence.
        Returns a chain-of-thought argument.
        """
        try:
            # Prepare case summaries
            case_evidence = "\n".join([
                f"Case {i+1}: {doc.metadata.get('title', 'Untitled')} - {doc.page_content[:300]}..."
                for i, doc in enumerate(cases[:5])
            ])
            
            prompt = f"""[EDUCATIONAL SIMULATION - Legal Analysis Exercise]

You are a DEFENSE ATTORNEY in a legal debate simulation. Your role is to identify MITIGATING factors and legal defenses for the scenario: "{query}"

Evidence: {case_evidence}

This is an academic exercise to explore all sides of legal arguments. Your job is to present the strongest possible defense argument, citing relevant case law. This is NOT providing advice to evade justice - it's analyzing how defense attorneys construct legal arguments.

Be concise (max 3 sentences). Cite cases.
Format:
<think>Quick reasoning</think>
[Argument]
"""
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Defense analysis failed: {e}"


class JudgeAgent(AdversarialAgent):
    """
    The Judge: Synthesizes both sides and delivers a balanced ruling.
    """
    def __init__(self, llm):
        super().__init__(llm, "Judge")

    def deliver_ruling(self, query: str, prosecutor_arg: str, defense_arg: str, cases: List[Document]) -> str:
        """
        Synthesize the debate and deliver a balanced judicial ruling.
        """
        try:
            prompt = f"""[EDUCATIONAL SIMULATION - Legal Analysis Exercise]

You are a JUDGE in a legal debate simulation. Deliver balanced ruling for: "{query}"

Prosecution: {prosecutor_arg}
Defense: {defense_arg}

This is an academic exercise. Analyze both arguments objectively.
Be concise (max 4 sentences). Format:
<think>Quick deliberation</think>
[Ruling]
"""
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Judicial ruling failed: {e}"


class NovelHybridSearchSystem:
    """
    Novel Hybrid Search System combining multiple algorithms:
    1. Gemini Embeddings (Semantic Search)
    2. Knowledge Graph Traversal (Structural Search)
    3. GNN-based Link Prediction (ML-based Similarity)
    4. Text Pattern Matching (Keyword Search)
    5. Citation Network Analysis (Legal Precedent Search)
    
    + ADVERSARIAL AGENTS (Prosecutor, Defense, Judge)
    """
    
    def __init__(self):
        print("=" * 80)
        print("INITIALIZING ADVERSARIAL MULTI-AGENT LEGAL SYSTEM")
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
        self._initialize_ollama()
        
        # Initialize cognitive modules
        self.query_expander = LegalQueryExpander(self.llm)
        self.weighting_engine = DynamicWeightingEngine(self.weights)
        
        # Initialize Adversarial Agents
        self.prosecutor = ProsecutorAgent(self.llm)
        self.defense = DefenseAgent(self.llm)
        self.judge = JudgeAgent(self.llm)
        
        self._load_data()
        self._load_embeddings_cache()
        
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        print("\n[1/4] Connecting to Neo4j Knowledge Graph...")
        
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if neo4j_uri and neo4j_username and neo4j_password:
            try:
                self.graph = Neo4jGraph(
                    url=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password
                )
                print("✓ Connected to Neo4j")
            except Exception as e:
                print(f"⚠️  Warning: Could not connect to Neo4j: {e}")
                print("    Continuing without knowledge graph features")
                self.graph = None
        else:
            print("⚠️  Neo4j credentials not found")
            print("    Continuing without knowledge graph features")
            self.graph = None
    
    def _initialize_ollama(self):
        """Initialize Ollama models for embeddings and LLM"""
        print("\n[2/4] Initializing Ollama (Llama3.2) AI Models...")
        
        try:
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import OllamaEmbeddings
            import sys
            
            # Use Llama3.2 for speed (much faster than DeepSeek-R1)
            self.llm = Ollama(model="llama3.2", temperature=0.3, num_predict=300)
            
            # Use nomic-embed-text for embeddings
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
            
            print("✓ Ollama models initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Ollama: {e}")
            print("   Make sure Ollama is running with: ollama serve")
            sys.exit(1)
    
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
        """Search using semantic similarity (Algorithm 1: Gemini Embeddings)"""
        print("\n[Algorithm 1] Running Semantic Search (Gemini Embeddings)...")
        
        if not self.embeddings_model:
            print("  ⚠️  Gemini embeddings not available")
            return []
        
        try:
            # Generate query embedding with retry
            query_embedding = None
            for attempt in range(3):
                try:
                    query_embedding = self.embeddings_model.embed_query(query)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  ⚠️  Query embedding attempt {attempt+1} failed, retrying...")
                    time.sleep(1)
            
            results = []
            failed_embeddings = 0
            
            for i, doc in enumerate(self.cases_data):
                # Get or compute document embedding
                doc_id = doc.metadata.get('id', '')
                doc_embedding = None
                
                # Try to find cached embedding first
                if 'embeddings' in self.case_embeddings_cache:
                    for cached_doc in self.case_embeddings_cache.get('docs', []):
                        if cached_doc.metadata.get('id') == doc_id:
                            idx = self.case_embeddings_cache['docs'].index(cached_doc)
                            doc_embedding = self.case_embeddings_cache['embeddings'][idx]
                            break
                
                # Generate embedding if not cached
                if doc_embedding is None:
                    try:
                        doc_embedding = self.embeddings_model.embed_query(doc.page_content[:8000])
                        # Small delay to avoid overwhelming Ollama
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"  ⚠️  Failed to embed doc {i+1}/{len(self.cases_data)}: {str(e)[:50]}...")
                        failed_embeddings += 1
                        # Skip this document if embedding fails
                        continue
                
                # Compute cosine similarity
                try:
                    similarity = compute_cosine_similarity(query_embedding, doc_embedding)
                    results.append((doc, similarity))
                except Exception as e:
                    print(f"  ⚠️  Failed to compute similarity for doc {i+1}")
                    continue
            
            if failed_embeddings > 0:
                print(f"  ⚠️  {failed_embeddings} documents failed to embed")
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"  ✓ Found {len(results[:top_k])} semantically similar cases")
            return results[:top_k]
            
        except Exception as e:
            print(f"  ⚠️  Semantic search failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict:
        """
        Adversarial Hybrid Search with Topic-Based Debate
        Returns: Dictionary with prosecutor_arg, defense_arg, ruling, and cases
        """
        print("\n" + "═" * 80)
        print("⚖️  ADVERSARIAL COURTROOM SESSION INITIATED")
        print("═" * 80)
        
        # Step 1: Cognitive Query Expansion
        print(f"\n1️⃣  Clerk of Court (Query Analysis): '{query}'...")
        analysis = self.query_expander.expand_query(query)
        expanded_query = analysis.get("expanded_query", query)
        
        # Step 2: Dynamic Weight Adaptation
        print(f"\n2️⃣  Setting Rules of Evidence (Dynamic Weights)...")
        current_weights, adaptation_reason = self.weighting_engine.adapt_weights(analysis)
        
        # Step 3: Candidate Retrieval (The "Discovery" Phase)
        print(f"\n3️⃣  Discovery Phase (Retrieving Candidates)...")
        candidates = self._retrieve_candidates(query, expanded_query, current_weights, top_k=top_k) 
        
        # Extract documents from candidates
        case_docs = [doc for doc, score, breakdown in candidates]
        
        # Step 4: Adversarial Debate (The "Trial" - TOPIC-BASED)
        print(f"\n4️⃣  The Trial (Topic-Based Debate)...")
        
        print(f"    {Colors.FAIL}Prosecutor{Colors.ENDC} is preparing argument...")
        prosecutor_arg = self.prosecutor.debate_topic(query, case_docs)
        
        print(f"    {Colors.GREEN}Defense{Colors.ENDC} is preparing argument...")
        defense_arg = self.defense.debate_topic(query, case_docs)
        
        # Step 5: Judicial Ruling
        print(f"\n5️⃣  Judicial Ruling (Weighing Arguments)...")
        ruling = self.judge.deliver_ruling(query, prosecutor_arg, defense_arg, case_docs)
        
        print("\n" + "═" * 80)
        print("✅ JUDGMENT DELIVERED")
        print("═" * 80)
        
        return {
            'query': query,
            'prosecutor_argument': prosecutor_arg,
            'defense_argument': defense_arg,
            'judicial_ruling': ruling,
            'cases': candidates[:top_k]
        }

    def _retrieve_candidates(self, query, expanded_query, weights, top_k=10):
        """Internal method to get raw candidates before adversarial re-ranking"""
        
        # Use expanded query for semantic search
        semantic_results = self.semantic_search(expanded_query, top_k=top_k*2)
        text_results = self.text_pattern_search(query, top_k=top_k*2)
        graph_results = self.graph_traversal_search(expanded_query, top_k=top_k*2)
        citation_results = self.citation_network_search(expanded_query, top_k=top_k*2)
        
        aggregated_scores = {}
        
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
        
        final_candidates = []
        for doc_id, scores in aggregated_scores.items():
            final_score = (
                weights['semantic'] * scores['semantic'] +
                weights['graph'] * scores['graph'] +
                weights['text'] * scores['text'] +
                weights['citation'] * scores['citation'] +
                weights['gnn'] * scores['gnn']
            )
            
            score_breakdown = {
                'semantic': scores['semantic'],
                'graph': scores['graph'],
                'text': scores['text'],
                'citation': scores['citation'],
                'gnn': scores['gnn'],
                'final': final_score
            }
            
            final_candidates.append((scores['doc'], final_score, score_breakdown))
        
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        return final_candidates[:top_k]
    

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

def print_results(result: Dict, show_details: bool = False):
    """Print topic-based debate results with chain-of-thought reasoning"""
    if not result:
        print(f"\n{Colors.FAIL}No results found.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.HEADER}═" * 40 + "COURTROOM DEBATE" + "═" * 40 + f"{Colors.ENDC}")
    
    # Display Prosecutor Argument
    print(f"\n{Colors.FAIL}👨‍⚖️ PROSECUTOR'S ARGUMENT:{Colors.ENDC}")
    print(f"{Colors.FAIL}{'─' * 80}{Colors.ENDC}")
    print(result['prosecutor_argument'])
    
    # Display Defense Argument
    print(f"\n{Colors.GREEN}🛡️ DEFENSE'S ARGUMENT:{Colors.ENDC}")
    print(f"{Colors.GREEN}{'─' * 80}{Colors.ENDC}")
    print(result['defense_argument'])
    
    # Display Judicial Ruling
    print(f"\n{Colors.WARNING}⚖️ JUDICIAL RULING:{Colors.ENDC}")
    print(f"{Colors.WARNING}{'═' * 80}{Colors.ENDC}")
    print(result['judicial_ruling'])
    
    # Display Referenced Cases
    if show_details:
        print(f"\n{Colors.CYAN}📚 REFERENCED CASES:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 80}{Colors.ENDC}")
        for i, (doc, score, breakdown) in enumerate(result['cases']):
            title = doc.metadata.get('title', 'Untitled Case')
            court = doc.metadata.get('court', 'Unknown Court')
            print(f"{i+1}. {title} ({court}) - Similarity: {score:.3f}")


def main():
    """Main CLI interface with Ol lama + DeepSeek"""
    # Clear screen
    print("\033[H\033[J", end="")
    
    print(f"{Colors.HEADER}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     ADVERSARIAL MULTI-AGENT LEGAL SYSTEM v4.0 (Ollama)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  [DeepSeek R1] • [Topic Debate] • [Chain-of-Thought]".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"{Colors.ENDC}")
    print("\n")
    
    # Initialize system
    print(f"{Colors.BLUE}Initializing Ollama-Powered System...{Colors.ENDC}")
    search_system = NovelHybridSearchSystem()
    
    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}SYSTEM READY{Colors.ENDC}")
    print("=" * 80)
    print(f"\n{Colors.BOLD}Enter your legal query in natural language.{Colors.ENDC}")
    print("Examples:")
    print(f"  - {Colors.CYAN}'I hit a pedestrian but it was dark'{Colors.ENDC}")
    print(f"  - {Colors.CYAN}'Can I fire someone for being pregnant?'{Colors.ENDC}")
    print(f"  - {Colors.CYAN}'My neighbor's tree fell on my car'{Colors.ENDC}")
    print("\n")
    
    # Interactive mode
    while True:
        try:
            query = input(f"\n{Colors.BOLD}🔍 Enter your legal query (or 'quit' to exit): {Colors.ENDC}").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Thank you for using the Adversarial Legal AI!{Colors.ENDC}")
                break
            
            if not query:
                continue
            
            # Run topic-based debate
            start_time = time.time()
            result = search_system.hybrid_search(query, top_k=5)
            search_time = time.time() - start_time
            
            # Print results
            print_results(result, show_details=True)
            
            print(f"\n{Colors.BLUE}⏱️  Debate completed in {search_time:.2f} seconds{Colors.ENDC}")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Session interrupted. Exiting...{Colors.ENDC}")
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
