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
    The Prosecutor: Looks for strict liability, guilt, and aggravating factors.
    Argues WHY this case supports a conviction/liability.
    """
    def __init__(self, llm):
        super().__init__(llm, "Prosecutor")

    def analyze_case(self, query: str, case: Document) -> Dict[str, Any]:
        try:
            prompt = f"""
            ROLE: Aggressive Prosecutor
            TASK: Analyze this legal case precedent to see if it supports a STRICT/GUILTY outcome for the query.
            
            QUERY: "{query}"
            CASE PRECEDENT: "{case.page_content[:1000]}"...
            
            OUTPUT JSON ONLY:
            {{
                "relevance_score": <float 0-1, high if it supports prosecution/strictness>,
                "argument": "<1 sentence argument why this case supports strict liability/guilt>"
            }}
            """
            response = self.llm.invoke(prompt)
            content = response.content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
        except:
            return {"relevance_score": 0.0, "argument": "Analysis failed"}


class DefenseAgent(AdversarialAgent):
    """
    The Defense: Looks for exceptions, loopholes, and mitigating factors.
    Argues WHY this case supports acquittal/leniency.
    """
    def __init__(self, llm):
        super().__init__(llm, "Defense")

    def analyze_case(self, query: str, case: Document) -> Dict[str, Any]:
        try:
            prompt = f"""
            ROLE: Strategic Defense Attorney
            TASK: Analyze this legal case precedent to see if it supports a LENIENT/NOT GUILTY outcome for the query.
            
            QUERY: "{query}"
            CASE PRECEDENT: "{case.page_content[:1000]}"...
            
            OUTPUT JSON ONLY:
            {{
                "relevance_score": <float 0-1, high if it supports defense/leniency>,
                "argument": "<1 sentence argument why this case supports leniency/exceptions>"
            }}
            """
            response = self.llm.invoke(prompt)
            content = response.content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
        except:
            return {"relevance_score": 0.0, "argument": "Analysis failed"}


class JudgeAgent(AdversarialAgent):
    """
    The Judge: Weighs arguments and re-ranks.
    """
    def __init__(self, llm):
        super().__init__(llm, "Judge")

    def synthesize_and_rank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Candidates list contains: {'doc': doc, 'prosecutor_score': ..., 'defense_score': ..., 'p_arg': ..., 'd_arg': ...}
        """
        ranked_results = []
        for cand in candidates:
            # Simple synthesis for speed (can be LLM based for full novelty)
            # Judge values high conflict (relevant to both) or high specificity
            
            # If both sides find it relevant, it's a CRITICAL precedent
            conflict_score = (cand['prosecutor_score'] + cand['defense_score']) / 2
            
            # Judge's final relevance score
            final_score = cand['base_score'] * 0.6 + conflict_score * 0.4
            
            cand['final_score'] = final_score
            ranked_results.append(cand)
            
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked_results


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
        self._initialize_gemini()
        
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
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Adversarial Hybrid Search
        Returns: List of result dictionaries with agent arguments
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
        candidates = self._retrieve_candidates(query, expanded_query, current_weights, top_k=8) # Get more for reranking
        
        # Step 4: Adversarial Debate (The "Trial")
        print(f"\n4️⃣  The Trial (Adversarial Analysis)...")
        processed_candidates = []
        
        print(f"    {Colors.FAIL}Prosecutor{Colors.ENDC} and {Colors.GREEN}Defense{Colors.ENDC} are analyzing {len(candidates)} cases...")
        
        # Parallel analysis (simulated loop here)
        for i, (doc, base_score, breakdown) in enumerate(candidates):
            print(f"    → Analyzing Case {i+1}...", end="\r")
            
            # Prosecutor Analysis
            p_analysis = self.prosecutor.analyze_case(query, doc)
            
            # Defense Analysis
            d_analysis = self.defense.analyze_case(query, doc)
            
            processed_candidates.append({
                'doc': doc,
                'base_score': base_score,
                'breakdown': breakdown,
                'prosecutor_score': p_analysis.get('relevance_score', 0),
                'prosecutor_arg': p_analysis.get('argument', ''),
                'defense_score': d_analysis.get('relevance_score', 0),
                'defense_arg': d_analysis.get('argument', '')
            })
            
        print(f"    ✓ Analysis Complete.                               ")

        # Step 5: Judicial Ruling
        print(f"\n5️⃣  Judicial Ruling (Final Ranking)...")
        final_results = self.judge.synthesize_and_rank(query, processed_candidates)
        
        print("\n" + "═" * 80)
        print("✅ JUDGMENT DELIVERED")
        print("═" * 80)
        
        return final_results[:top_k]

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
    
    def generate_ai_explanation(self, query: str, results: List[Dict]) -> str:
        """Generate AI explanation based on the adversarial debate"""
        if not self.llm or not results:
            return "AI explanation not available"
        
        try:
            # Prepare context from the debate
            cases_summary = "\n\n".join([
                f"Case {i+1}: {case['doc'].metadata.get('title', 'Untitled')}\n"
                f"Prosecutor Argues: {case['prosecutor_arg']}\n"
                f"Defense Argues: {case['defense_arg']}\n"
                f"Summary: {case['doc'].page_content[:200]}..."
                for i, case in enumerate(results[:3])
            ])
            
            prompt = f"""You are a Chief Justice. A user asked: "{query}"

Review the arguments from your Prosecutor and Defense agents on these top cases:

{cases_summary}

Provide a "Judicial Summary":
1. What is the balanced legal view?
2. Which side (Prosecution/Strictness or Defense/Leniency) has stronger precedents here?
3. Practical takeaway for the user.

Keep it authoritative but clear.
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

def print_results(results: List[Dict], show_details: bool = False):
    """Print search results in a high-tech dashboard format with Adversarial Arguments"""
    if not results:
        print(f"\n{Colors.FAIL}No results found.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.HEADER}JUDICIAL RESULTS DASHBOARD{Colors.ENDC}")
    print(f"{Colors.BLUE}Found {len(results)} relevant cases after adversarial review{Colors.ENDC}")
    print("═" * 80)
    
    for i, res in enumerate(results):
        doc = res['doc']
        title = doc.metadata.get('title', 'Untitled Case')
        court = doc.metadata.get('court', 'Unknown Court')
        final_score = res['final_score']
        
        # Color code score
        score_color = Colors.GREEN if final_score > 0.7 else (Colors.WARNING if final_score > 0.4 else Colors.FAIL)
        
        print(f"\n{Colors.BOLD}{i+1}. {title}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Court:{Colors.ENDC} {court}")
        print(f"   {Colors.BOLD}Judicial Relevance:{Colors.ENDC} {score_color}{final_score:.4f}{Colors.ENDC}")
        
        # ADVERSARIAL ARGUMENTS DISPLAY
        print(f"\n   {Colors.FAIL}Prosecutor's Take:{Colors.ENDC} {res['prosecutor_arg']}")
        print(f"   {Colors.GREEN}Defense's Take:   {Colors.ENDC} {res['defense_arg']}")
        
        if show_details:
            print(f"\n   {Colors.UNDERLINE}Technical Analysis:{Colors.ENDC}")
            print(f"     • Base Similarity:    {res['base_score']:.4f}")
            print(f"     • Prosecutor Score:   {res['prosecutor_score']:.4f}")
            print(f"     • Defense Score:      {res['defense_score']:.4f}")
            
            # Show excerpt
            excerpt = doc.page_content[:200].replace('\n', ' ') + "..." 
            print(f"\n   {Colors.CYAN}Excerpt:{Colors.ENDC}\n   {excerpt}\n")
        
        print(f"{Colors.BLUE}" + "-" * 80 + f"{Colors.ENDC}")


def main():
    """Main CLI interface with Cyberpunk/Dashboard UI"""
    # Clear screen
    print("\033[H\033[J", end="")
    
    print(f"{Colors.HEADER}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     ADVERSARIAL MULTI-AGENT LEGAL SYSTEM v3.0".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  [Prosecutor Agent] • [Defense Agent] • [Judge Agent]".center(78) + "║")
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
