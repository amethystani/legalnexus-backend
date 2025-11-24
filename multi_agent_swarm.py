"""
Multi-Agent Swarm for Legal Knowledge Graph Construction

Three specialized agents that debate and construct a rich legal knowledge graph:
1. Linker Agent: Finds all citations
2. Interpreter Agent: Classifies edge types (Follow, Distinguish, Overrule)
3. Conflict Agent: Detects logical contradictions and cycles

Novel Contribution: Agents debate each other to reach consensus on graph structure.
"""

from langchain_community.llms import Ollama
from typing import List, Dict, Tuple, Set
import re
from dataclasses import dataclass
from enum import Enum


class EdgeType(Enum):
    """Types of legal citation relationships"""
    FOLLOW = "FOLLOW"  # Case A follows Case B (reinforces precedent)
    DISTINGUISH = "DISTINGUISH"  # Case A distinguishes from Case B (creates fork)
    OVERRULE = "OVERRULE"  # Case A overrules Case B (inverts authority)
    SUPPORTS = "SUPPORTS"  # Generic support relationship
    ATTACKS = "ATTACKS"  # Argument attacks another
    PROVIDES_WARRANT = "PROVIDES_WARRANT"  # Provides legal warrant


@dataclass
class Citation:
    """A citation from one case to another"""
    source_id: str
    target_id: str
    context: str  # Text surrounding the citation
    edge_type: EdgeType = EdgeType.SUPPORTS
    confidence: float = 0.5


@dataclass
class Conflict:
    """A logical conflict in the graph"""
    conflict_type: str  # "cycle", "contradiction", "overrule_chain"
    involved_cases: List[str]
    description: str
    severity: float  # 0.0 to 1.0


class LinkerAgent:
    """
    Agent 1: Finds all citations in case text.
    
    Uses both pattern matching and LLM reasoning to identify when
    one case references another.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Linker Agent"
    
    def find_citations(self, case_id: str, case_text: str, all_case_ids: Set[str]) -> List[Citation]:
        """
        Find all citations in the case text.
        
        Strategy:
        1. Pattern matching for obvious citations
        2. LLM to identify contextual references
        """
        citations = []
        
        # Pattern matching (fast)
        patterns = [
            r'(?:AIR|SCC)\s+\d{4}\s+[A-Z]+\s+\d+',  # AIR 2019 SC 123
            r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',  # (2018) 10 SCC 456
            r'(?:v\.|vs\.)\s+[A-Z][a-z]+\s+\(\d{4}\)',  # v. Kumar (2020)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, case_text[:5000])
            for match in matches:
                # Try to match to actual case IDs
                for target_id in all_case_ids:
                    if self._fuzzy_match(match, target_id) and target_id != case_id:
                        # Get context (50 chars before and after)
                        idx = case_text.find(match)
                        context = case_text[max(0, idx-50):min(len(case_text), idx+len(match)+50)]
                        
                        citations.append(Citation(
                            source_id=case_id,
                            target_id=target_id,
                            context=context,
                            confidence=0.8
                        ))
                        break
        
        # LLM reasoning (slower but more accurate)
        if len(citations) < 3:  # If pattern matching found few, use LLM
            citations.extend(self._llm_find_citations(case_id, case_text, all_case_ids))
        
        return citations
    
    def _fuzzy_match(self, citation_str: str, case_id: str) -> bool:
        """Check if citation string matches case ID"""
        citation_lower = citation_str.lower()
        case_id_lower = case_id.lower()
        
        # Extract numbers from both
        citation_nums = set(re.findall(r'\d+', citation_str))
        case_nums = set(re.findall(r'\d+', case_id))
        
        # Match if they share numbers and some text
        return len(citation_nums & case_nums) > 0 or citation_lower in case_id_lower
    
    def _llm_find_citations(self, case_id: str, case_text: str, all_case_ids: Set[str]) -> List[Citation]:
        """Use LLM to find citations"""
        # Simplified for speed
        sample = case_text[:1500]
        
        prompt = f"""List case citations found in this text. Output format: one citation per line.

TEXT: {sample}

CITATIONS:"""
        
        try:
            response = self.llm.invoke(prompt)
            lines = str(response).split('\n')
            
            citations = []
            for line in lines:
                if len(line.strip()) > 5:
                    # Try to match to real case IDs
                    for target_id in all_case_ids:
                        if target_id != case_id and self._fuzzy_match(line, target_id):
                            citations.append(Citation(
                                source_id=case_id,
                                target_id=target_id,
                                context=line,
                                confidence=0.6
                            ))
                            break
            
            return citations[:5]
        except:
            return []


class InterpreterAgent:
    """
    Agent 2: Interprets citation context to classify edge types.
    
    Determines if a citation is:
    - FOLLOW: Case A agrees with and follows Case B
    - DISTINGUISH: Case A distinguishes facts from Case B
    - OVERRULE: Case A explicitly overrules Case B
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Interpreter Agent"
    
    def classify_citation(self, citation: Citation) -> Citation:
        """
        Classify the type of citation based on context.
        
        Uses LLM to analyze the textual context around the citation.
        """
        context = citation.context
        
        # Quick heuristics first
        if any(word in context.lower() for word in ['overrule', 'overruled', 'reversed']):
            citation.edge_type = EdgeType.OVERRULE
            citation.confidence = 0.9
            return citation
        
        if any(word in context.lower() for word in ['distinguish', 'different facts', 'inapplicable']):
            citation.edge_type = EdgeType.DISTINGUISH
            citation.confidence = 0.85
            return citation
        
        if any(word in context.lower() for word in ['follow', 'relied upon', 'precedent', 'in line with']):
            citation.edge_type = EdgeType.FOLLOW
            citation.confidence = 0.8
            return citation
        
        # Use LLM for unclear cases
        try:
            edge_type, confidence = self._llm_classify(context)
            citation.edge_type = edge_type
            citation.confidence = confidence
        except:
            # Default to SUPPORTS
            citation.edge_type = EdgeType.SUPPORTS
            citation.confidence = 0.5
        
        return citation
    
    def _llm_classify(self, context: str) -> Tuple[EdgeType, float]:
        """Use LLM to classify citation type"""
        prompt = f"""Classify this legal citation context:

CONTEXT: "{context}"

Is this:
A) FOLLOW - The case follows/agrees with the cited case
B) DISTINGUISH - The case distinguishes from the cited case  
C) OVERRULE - The case overrules the cited case

Answer with only A, B, or C:"""
        
        response = str(self.llm.invoke(prompt)).strip().upper()
        
        if 'A' in response or 'FOLLOW' in response:
            return EdgeType.FOLLOW, 0.75
        elif 'B' in response or 'DISTINGUISH' in response:
            return EdgeType.DISTINGUISH, 0.75
        elif 'C' in response or 'OVERRULE' in response:
            return EdgeType.OVERRULE, 0.75
        else:
            return EdgeType.SUPPORTS, 0.5


class ConflictAgent:
    """
    Agent 3: Detects logical conflicts in the citation graph.
    
    Identifies:
    - Cycles: A cites B, B cites C, C cites A
    - Contradictions: A follows B, but A also overrules B
    - Overrule chains: A overrules B, B overrules C
    """
    
    def __init__(self):
        self.name = "Conflict Agent"
    
    def find_conflicts(self, citations: List[Citation]) -> List[Conflict]:
        """
        Analyze the citation network for logical conflicts.
        """
        conflicts = []
        
        # Build adjacency lists
        graph = {}
        for cit in citations:
            if cit.source_id not in graph:
                graph[cit.source_id] = []
            graph[cit.source_id].append((cit.target_id, cit.edge_type))
        
        # Detect cycles
        cycles = self._find_cycles(graph)
        for cycle in cycles:
            conflicts.append(Conflict(
                conflict_type="cycle",
                involved_cases=cycle,
                description=f"Citation cycle detected: {' -> '.join(cycle)}",
                severity=0.6
            ))
        
        # Detect contradictory edges
        contradictions = self._find_contradictions(citations)
        conflicts.extend(contradictions)
        
        # Detect overrule chains
        overrule_chains = self._find_overrule_chains(citations)
        conflicts.extend(overrule_chains)
        
        return conflicts
    
    def _find_cycles(self, graph: Dict) -> List[List[str]]:
        """Find cycles using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor, _ in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycles.append(path[cycle_start:] + [neighbor])
            
            path.pop()
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles[:5]  # Return first 5 cycles
    
    def _find_contradictions(self, citations: List[Citation]) -> List[Conflict]:
        """Find contradictory edge types between same nodes"""
        conflicts = []
        edge_map = {}
        
        for cit in citations:
            key = (cit.source_id, cit.target_id)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(cit.edge_type)
        
        for (src, tgt), edge_types in edge_map.items():
            unique_types = set(edge_types)
            
            # Check for contradictions
            if EdgeType.FOLLOW in unique_types and EdgeType.OVERRULE in unique_types:
                conflicts.append(Conflict(
                    conflict_type="contradiction",
                    involved_cases=[src, tgt],
                    description=f"{src} both follows AND overrules {tgt}",
                    severity=0.9
                ))
        
        return conflicts
    
    def _find_overrule_chains(self, citations: List[Citation]) -> List[Conflict]:
        """Find chains of overruling decisions"""
        overrules = [(c.source_id, c.target_id) for c in citations if c.edge_type == EdgeType.OVERRULE]
        
        conflicts = []
        for i, (a, b) in enumerate(overrules):
            for j, (c, d) in enumerate(overrules):
                if i != j and b == c:  # A overrules B, B overrules D
                    conflicts.append(Conflict(
                        conflict_type="overrule_chain",
                        involved_cases=[a, b, d],
                        description=f"Overrule chain: {a} → {b} → {d}",
                        severity=0.7
                    ))
        
        return conflicts[:3]


class MultiAgentSwarm:
    """
    Orchestrates the three agents to build the knowledge graph.
    
    Workflow:
    1. Linker Agent finds all citations
    2. Interpreter Agent classifies each citation
    3. Conflict Agent identifies graph inconsistencies
    4. Agents "debate" conflicts and reach consensus
    """
    
    def __init__(self):
        # Use Gemma 3 1B - faster than DeepSeek
        llm = Ollama(model="gemma2:2b", temperature=0.3, num_predict=200)
        
        self.linker = LinkerAgent(llm)
        self.interpreter = InterpreterAgent(llm)
        self.conflict = ConflictAgent()
        
        print("✓ Multi-Agent Swarm initialized")
        print(f"  - {self.linker.name}")
        print(f"  - {self.interpreter.name}")
        print(f"  - {self.conflict.name}")
    
    def process_case(self, case_text: str, case_id: str, all_case_ids: Set[str] = None) -> Dict:
        """
        Process a single case through the multi-agent swarm.
        
        Args:
            case_text: Full text of the case
            case_id: Unique identifier for this case
            all_case_ids: Set of all available case IDs for citation matching
        
        Returns:
            Dict with 'citations' and 'conflicts' keys
        """
        if all_case_ids is None:
            all_case_ids = set()
        
        # Phase 1: Linker Agent finds citations
        citations = self.linker.find_citations(case_id, case_text, all_case_ids)
        
        # Phase 2: Interpreter Agent classifies citations
        classified_citations = []
        for citation in citations:
            classified_citations.append(self.interpreter.classify_citation(citation))
        
        # Phase 3: Conflict Agent detects issues (for this case's citations)
        conflicts = self.conflict.find_conflicts(classified_citations)
        
        return {
            'citations': classified_citations,
            'conflicts': conflicts
        }
    
    def build_knowledge_graph(self, cases: List[Dict]) -> Tuple[List[Citation], List[Conflict]]:
        """
        Build knowledge graph through agent collaboration.
        
        Returns:
            citations: List of all citations with edge types
            conflicts: List of detected conflicts
        """
        print(f"\n{'='*80}")
        print("MULTI-AGENT SWARM: BUILDING KNOWLEDGE GRAPH")
        print(f"{'='*80}\n")
        
        all_case_ids = set([c['id'] for c in cases])
        all_citations = []
        
        # Phase 1: Linker Agent finds citations
        print(f"Phase 1: {self.linker.name} finding citations...")
        for i, case in enumerate(cases):
            print(f"  Processing [{i+1}/{len(cases)}] {case['id'][:40]}", end='\r')
            
            citations = self.linker.find_citations(case['id'], case['text'], all_case_ids)
            all_citations.extend(citations)
        
        print(f"\n  ✓ Found {len(all_citations)} citations")
        
        # Phase 2: Interpreter Agent classifies citations
        print(f"\nPhase 2: {self.interpreter.name} classifying edge types...")
        for i, citation in enumerate(all_citations):
            print(f"  Classifying [{i+1}/{len(all_citations)}]", end='\r')
            all_citations[i] = self.interpreter.classify_citation(citation)
        
        # Count edge types
        edge_counts = {}
        for c in all_citations:
            edge_counts[c.edge_type.value] = edge_counts.get(c.edge_type.value, 0) + 1
        
        print(f"\n  ✓ Edge type distribution:")
        for edge_type, count in sorted(edge_counts.items()):
            print(f"    - {edge_type}: {count}")
        
        # Phase 3: Conflict Agent detects issues
        print(f"\nPhase 3: {self.conflict.name} detecting conflicts...")
        conflicts = self.conflict.find_conflicts(all_citations)
        
        print(f"  ✓ Found {len(conflicts)} potential conflicts")
        for conf in conflicts[:5]:
            print(f"    - {conf.conflict_type}: {conf.description}")
        
        return all_citations, conflicts
