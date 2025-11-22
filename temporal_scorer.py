"""
Temporal Scoring for Legal Cases

Implements precedent decay and concept drift measurement.
Addresses the problem that old cases might be semantically similar but legally obsolete.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np


def extract_year_from_text(text: str) -> Optional[int]:
    """Extract judgment year from case text"""
    # Common patterns: "2023", "Date: 23rd Nov., 2016", "Dated 18th October, 2019"
    patterns = [
        r'Date.*?(\d{4})',
        r'Dated.*?(\d{4})',
        r'Judgment.*?(\d{4})',
        r'\b(19\d{2}|20\d{2})\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:1000])  # Search in first 1000 chars
        if match:
            year = int(match.group(1))
            if 1947 <= year <= 2025:  # Valid range for Indian law
                return year
    return None


def calculate_temporal_score(
    case_date: Optional[int],
    citation_history: List[int],
    current_year: int = 2025
) -> float:
    """
    Calculate temporal relevance score with precedent decay.
    
    Novelty: Cases that are old BUT cited recently (Resurrected) get high scores.
    Cases that are old and NOT cited recently get penalized (Decay).
    
    Args:
        case_date: Year of judgment
        citation_history: List of years when this case was cited
        current_year: Current year for calculations
    
    Returns:
        Temporal score (0-1)
    """
    if case_date is None:
        return 0.5  # Neutral score if date unknown
    
    age = current_year - case_date
    
    if age < 0 or age > 100:  # Sanity check
        return 0.1
    
    # Base decay: older cases get lower scores
    # Use log decay to not over-penalize moderately old cases
    if age == 0:
        base_decay = 1.0
    else:
        base_decay = 1.0 / np.log(age + 2)  # +2 to avoid log(1)=0
    
    # Recency bias: if cited recently, boost score
    if citation_history and len(citation_history) > 0:
        # Weight citations by recency
        recency_weights = []
        for cite_year in citation_history:
            cite_age = current_year - cite_year
            if cite_age >= 0:
                # More recent citations get exponentially higher weight
                weight = 1.0 / (cite_age + 1)
                recency_weights.append(weight)
        
        if recency_weights:
            recency_bias = sum(recency_weights) / len(citation_history)
        else:
            recency_bias = 0.0
    else:
        recency_bias = 0.0
    
    # Combine: base decay modulated by citation recency
    if recency_bias > 0:
        # "Resurrection" effect: old cases cited recently get boosted
        temporal_score = min(base_decay * (1 + recency_bias), 1.0)
    else:
        # Pure decay: no recent citations
        temporal_score = base_decay * 0.7  # Penalty for not being cited
    
    return temporal_score


class TemporalScorer:
    """Manages temporal scoring for a corpus of cases"""
    
    def __init__(self):
        self.case_dates = {}  # case_id -> year
        self.citation_graph = {}  # case_id -> [years when cited]
        
    def add_case(self, case_id: str, case_text: str):
        """Extract and store temporal metadata"""
        year = extract_year_from_text(case_text)
        if year:
            self.case_dates[case_id] = year
    
    def add_citation(self, citing_case_id: str, cited_case_id: str, cite_year: int):
        """Record a citation event"""
        if cited_case_id not in self.citation_graph:
            self.citation_graph[cited_case_id] = []
        self.citation_graph[cited_case_id].append(cite_year)
    
    def score(self, case_id: str, current_year: int = 2025) -> float:
        """Get temporal score for a case"""
        case_date = self.case_dates.get(case_id)
        citation_history = self.citation_graph.get(case_id, [])
        
        return calculate_temporal_score(case_date, citation_history, current_year)
    
    def get_stats(self) -> Dict:
        """Return corpus statistics"""
        if not self.case_dates:
            return {}
        
        years = list(self.case_dates.values())
        return {
            'total_cases': len(self.case_dates),
            'earliest_year': min(years),
            'latest_year': max(years),
            'avg_age': 2025 - np.mean(years),
            'cases_with_citations': len(self.citation_graph)
        }
