"""
Toulmin Argumentation Structure Extractor

Extracts legal argument components using LLM:
- Claim: The legal conclusion/assertion
- Data: The facts supporting the claim
- Warrant: The legal rule/precedent connecting data to claim
- Backing: Authority supporting the warrant
- Rebuttal: Counter-arguments or exceptions
- Qualifier: Degree of certainty (e.g., "usually", "presumably")
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToulminStructure:
    """Represents a legal argument in Toulmin model"""
    claim: str
    data: List[str]
    warrant: str
    backing: List[str]
    rebuttal: Optional[str]
    qualifier: Optional[str]
    confidence: float  # 0-1 score from LLM
    

class ToulminExtractor:
    """Extracts Toulmin argument structures from legal text using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def extract_structure(self, case_text: str, case_id: str = "") -> Optional[ToulminStructure]:
        """
        Extract Toulmin argument structure from case text.
        Uses LLM with chain-of-thought prompting.
        """
        # Limit text length for LLM
        text_sample = case_text[:3000]
        
        prompt = f"""[LEGAL ARGUMENT ANALYSIS - Toulmin Model]

Analyze this legal case and extract the argument structure using the Toulmin Model.

CASE TEXT:
{text_sample}

TASK:
Identify the following components:
1. CLAIM: What is the main legal conclusion/ruling?
2. DATA: What are the key facts that support this claim?
3. WARRANT: What legal rule/principle connects the facts to the claim?
4. BACKING: What authority (statute, precedent) supports the warrant?
5. REBUTTAL: What counter-arguments or exceptions were considered?
6. QUALIFIER: What is the degree of certainty? (e.g., "always", "usually", "presumed")

OUTPUT FORMAT (JSON):
{{
    "claim": "The defendant is liable for...",
    "data": ["Fact 1", "Fact 2"],
    "warrant": "According to Section X...",
    "backing": ["Case precedent Y", "Statute Z"],
    "rebuttal": "Defense argued...",
    "qualifier": "unless exceptional circumstances",
    "confidence": 0.85
}}

Be precise and extract actual phrases from the text where possible.
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            import json
            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return ToulminStructure(
                    claim=data.get('claim', ''),
                    data=data.get('data', []),
                    warrant=data.get('warrant', ''),
                    backing=data.get('backing', []),
                    rebuttal=data.get('rebuttal'),
                    qualifier=data.get('qualifier'),
                    confidence=data.get('confidence', 0.5)
                )
            else:
                print(f"  ⚠️  Could not parse JSON from LLM response for {case_id}")
                return None
                
        except Exception as e:
            print(f"  ⚠️  Toulmin extraction failed for {case_id}: {str(e)[:50]}")
            return None
    
    def extract_batch(self, cases: List[tuple], max_cases: int = 50) -> Dict[str, ToulminStructure]:
        """
        Extract structures from multiple cases.
        Returns: {case_id: ToulminStructure}
        """
        results = {}
        
        for i, (case_id, case_text) in enumerate(cases[:max_cases]):
            print(f"  Extracting {i+1}/{min(len(cases), max_cases)}: {case_id}...", end='\r')
            
            structure = self.extract_structure(case_text, case_id)
            if structure and structure.confidence > 0.3:  # Quality threshold
                results[case_id] = structure
                
        print(f"  ✓ Extracted {len(results)} valid argument structures")
        return results
