"""
Toulmin Argument Chain Traversal

Implements graph-based reasoning over legal arguments.
Instead of finding "similar" cases, finds "logically supporting" argument chains.
"""

import networkx as nx
from typing import List, Dict, Tuple, Set
from toulmin_extractor import ToulminStructure


class ArgumentGraph:
    """In-memory graph of Toulmin argument structures"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.case_structures = {}  # case_id -> ToulminStructure
        
    def add_case(self, case_id: str, structure: ToulminStructure, case_doc):
        """Add a case's argument structure to the graph"""
        self.case_structures[case_id] = structure
        
        # Add nodes
        claim_id = f"{case_id}_claim"
        self.graph.add_node(claim_id, type='claim', text=structure.claim, case_id=case_id)
        
        # Add data nodes and edges
        for i, data_point in enumerate(structure.data):
            data_id = f"{case_id}_data_{i}"
            self.graph.add_node(data_id, type='data', text=data_point, case_id=case_id)
            self.graph.add_edge(data_id, claim_id, relation='SUPPORTS')
        
        # Add warrant node
        if structure.warrant:
            warrant_id = f"{case_id}_warrant"
            self.graph.add_node(warrant_id, type='warrant', text=structure.warrant, case_id=case_id)
            self.graph.add_edge(warrant_id, claim_id, relation='WARRANTS')
        
        # Add backing nodes
        for i, backing in enumerate(structure.backing):
            backing_id = f"{case_id}_backing_{i}"
            self.graph.add_node(backing_id, type='backing', text=backing, case_id=case_id)
            if structure.warrant:
                self.graph.add_edge(backing_id, warrant_id, relation='SUPPORTS_WARRANT')
    
    def find_argument_chain(self, query_claim: str, max_depth: int = 3) -> List[Tuple[str, float]]:
        """
        Find cases with argument chains that support the query claim.
        
        Novel approach: Instead of semantic similarity, traverse logical support chains.
        Returns: List of (case_id, chain_strength) ordered by strength.
        """
        # Find claim nodes semantically similar to query
        claim_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'claim']
        
        if not claim_nodes:
            return []
        
        # Simplified scoring: count supporting path length
        results = {}
        
        for claim_node in claim_nodes:
            case_id = self.graph.nodes[claim_node].get('case_id')
            claim_text = self.graph.nodes[claim_node].get('text', '')
            
            # Simple text matching (in production, use embeddings)
            # Count how many words from query appear in claim
            query_words = set(query_claim.lower().split())
            claim_words = set(claim_text.lower().split())
            overlap = len(query_words & claim_words)
            
            if overlap > 2:  # Minimum threshold
                # Calculate chain strength: count supporting nodes
                supporting_data = list(self.graph.predecessors(claim_node))
                chain_strength = len(supporting_data) * 0.3 + overlap * 0.7
                
                if case_id not in results or results[case_id] < chain_strength:
                    results[case_id] = chain_strength
        
        # Sort by chain strength
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def explain_chain(self, case_id: str) -> str:
        """Generate human-readable explanation of argument chain"""
        if case_id not in self.case_structures:
            return "Case not found in argument graph."
        
        structure = self.case_structures[case_id]
        
        explanation = f"**Argument Chain for {case_id}:**\n\n"
        explanation += f"📌 **Claim:** {structure.claim}\n\n"
        
        if structure.data:
            explanation += f"📊 **Supporting Facts:**\n"
            for i, data in enumerate(structure.data, 1):
                explanation += f"  {i}. {data}\n"
            explanation += "\n"
        
        if structure.warrant:
            explanation += f"⚖️ **Legal Warrant:** {structure.warrant}\n\n"
        
        if structure.backing:
            explanation += f"📚 **Authority:**\n"
            for i, backing in enumerate(structure.backing, 1):
                explanation += f"  {i}. {backing}\n"
            explanation += "\n"
        
        if structure.rebuttal:
            explanation += f"🛡️ **Rebuttals Considered:** {structure.rebuttal}\n\n"
        
        return explanation
    
    def get_stats(self) -> Dict:
        """Return graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'num_cases': len(self.case_structures),
            'claims': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'claim']),
            'warrants': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'warrant'])
        }
