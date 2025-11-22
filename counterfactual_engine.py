"""
Counterfactual "What-If" Engine

Identifies legal pivot points by perturbing facts and measuring impact on retrieval results.
Answers: "What fact do I need to change to win this case?"
"""

import re
from typing import List, Dict, Tuple, Set


class FactExtractor:
    """Extracts legal facts from queries using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_facts(self, query: str) -> List[Dict[str, str]]:
        """
        Extract factual assertions from query.
        Returns: [{'fact': str, 'type': str, 'negation': str}, ...]
        """
        prompt = f"""[FACT EXTRACTION - Legal Analysis]

Extract the KEY FACTS from this legal query. For each fact, provide:
1. The fact itself
2. The type (temporal, action, condition, etc.)
3. A plausible negation/alternative

QUERY: "{query}"

OUTPUT FORMAT (JSON array):
[
    {{"fact": "it was dark", "type": "condition", "negation": "it was daylight"}},
    {{"fact": "hit a pedestrian", "type": "action", "negation": "avoided the pedestrian"}},
    {{"fact": "was drunk", "type": "condition", "negation": "was sober"}}
]

Extract 3-5 key facts. Be concise.
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            # Find JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                return facts
            else:
                return []
                
        except Exception as e:
            print(f"  ⚠️  Fact extraction failed: {str(e)[:50]}")
            return []


class ShadowAgent:
    """Runs parallel searches with perturbed facts"""
    
    def __init__(self, search_system):
        self.system = search_system
    
    def create_counterfactual (self, original_query: str, fact: Dict) -> str:
        """Create a counterfactual query by negating one fact"""
        # Simple replacement
        original_fact = fact.get('fact', '')
        negation = fact.get('negation', '')
        
        if original_fact and negation:
            counterfactual = original_query.replace(original_fact, negation)
            return counterfactual
        return original_query
    
    def run_parallel_search(self, original_query: str, counterfactual_query: str, top_k: int = 3) -> Tuple[List, List]:
        """
        Run search on both original and counterfactual queries.
        Returns: (original_results, counterfactual_results)
        """
        # For now, use simple semantic search
        # In full implementation, use hybrid_search
        
        orig_results = self.system.semantic_search(original_query, top_k=top_k)
        counter_results = self.system.semantic_search(counterfactual_query, top_k=top_k)
        
        return orig_results, counter_results


class CounterfactualEngine:
    """Main counterfactual analysis engine"""
    
    def __init__(self, llm, search_system):
        self.llm = llm
        self.system = search_system
        self.fact_extractor = FactExtractor(llm)
        self.shadow_agent = ShadowAgent(search_system)
    
    def identify_pivot_points(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Identify which facts are "pivot points" - changing them changes outcomes.
        
        Returns: [
            {
                'fact': str,
                'type': str,
                'is_pivot': bool,
                'impact_score': float,
                'original_cases': List[str],
                'counterfactual_cases': List[str],
                'explanation': str
            }
        ]
        """
        print("\n🔍 COUNTERFACTUAL ANALYSIS")
        print("="*80)
        
        # Step 1: Extract facts
        print(f"\n1. Extracting facts from query: '{query}'")
        facts = self.fact_extractor.extract_facts(query)
        
        if not facts:
            print("  ⚠️  Could not extract facts")
            return []
        
        print(f"  ✓ Extracted {len(facts)} facts:")
        for i, fact in enumerate(facts, 1):
            print(f"     {i}. {fact.get('fact')} → {fact.get('negation')}")
        
        # Step 2: Run counterfactual searches
        print("\n2. Running counterfactual searches...")
        pivot_analysis = []
        
        for fact in facts:
            counterfactual_query = self.shadow_agent.create_counterfactual(query, fact)
            
            print(f"\n  Testing: {fact.get('fact')}")
            print(f"    Original:       '{query}'")
            print(f"    Counterfactual: '{counterfactual_query}'")
            
            # Run  parallel searches
            orig_results, counter_results = self.shadow_agent.run_parallel_search(
                query, counterfactual_query, top_k=top_k
            )
            
            # Calculate impact: how much did results change?
            orig_ids = {doc.metadata.get('id') for doc, score in orig_results}
            counter_ids = {doc.metadata.get('id') for doc, score in counter_results}
            
            # Jaccard distance: 1 - (intersection / union)
            if orig_ids or counter_ids:
                intersection = len(orig_ids & counter_ids)
                union = len(orig_ids | counter_ids)
                impact_score = 1.0 - (intersection / union) if union > 0 else 0.0
            else:
                impact_score = 0.0
            
            is_pivot = impact_score > 0.5  # Threshold: >50% change
            
            # Generate explanation
            if is_pivot:
                explanation = f"⚠️ PIVOT POINT: Changing '{fact.get('fact')}' to '{fact.get('negation')}' dramatically changes case retrieval ({impact_score:.0%} different results)."
            else:
                explanation = f"ℹ️ Stable fact: Changing '{fact.get('fact')}' has minimal impact ({impact_score:.0%} change)."
            
            pivot_analysis.append({
                'fact': fact.get('fact'),
                'type': fact.get('type'),
                'negation': fact.get('negation'),
                'is_pivot': is_pivot,
                'impact_score': impact_score,
                'original_cases': [doc.metadata.get('id') for doc, score in orig_results],
                'counterfactual_cases': [doc.metadata.get('id') for doc, score in counter_results],
                'explanation': explanation
            })
            
            print(f"    {explanation}")
        
        # Sort by impact
        pivot_analysis.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return pivot_analysis
    
    def generate_recommendations(self, pivot_analysis: List[Dict]) -> str:
        """Generate legal strategy recommendations based on pivot points"""
        if not pivot_analysis:
            return "No pivot points identified."
        
        recommendations = "\n💡 LEGAL STRATEGY RECOMMENDATIONS:\n"
        recommendations += "="*80 + "\n"
        
        pivots = [p for p in pivot_analysis if p['is_pivot']]
        
        if pivots:
            recommendations += "\n🎯 Focus your legal argument on these CRITICAL facts:\n\n"
            for i, pivot in enumerate(pivots, 1):
                recommendations += f"  {i}. '{pivot['fact']}' (Impact: {pivot['impact_score']:.0%})\n"
                recommendations += f"     - If you can change this to '{pivot['negation']}', case precedents shift significantly.\n"
                recommendations += f"     - Original cases: {', '.join(pivot['original_cases'][:3])}\n"
                recommendations += f"     - Counterfactual cases: {', '.join(pivot['counterfactual_cases'][:3])}\n\n"
        else:
            recommendations += "\nℹ️ No critical pivot points found. The outcome is relatively stable across fact variations.\n"
        
        return recommendations
