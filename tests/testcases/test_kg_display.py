#!/usr/bin/env python3
"""
Test script to verify the knowledge graph display improvements
"""

import streamlit as st
from kg import format_case_results, display_case_results

# Test data that mimics what the Neo4j query would return
test_results = [
    {
        "c.id": "pension_rights_2022",
        "c.title": "D.S. Nakara & Others v. Union of India (Pension Rights Case)",
        "c.court": "Supreme Court of India",
        "c.date": "2022-07-12",
        "c.text": "This appeal concerns the fundamental rights of retired government employees..."
    },
    {
        "c.id": "www_latestlaws_com_2025_may_2025-latest-caselaw-613-sc",
        "c.title": "The Reserve Bank of India Vs. M.T. Mani and Anr.",
        "c.court": "Kerala High Court",
        "c.date": "May 23, 2025",
        "c.text": "In this Appeal, challenge has been raised by the Reserve Bank of India..."
    }
]

# Test similarity scores
test_scores = [0.85, 0.72]

def main():
    st.set_page_config(page_title="Test KG Display", layout="wide")
    st.title("Test Knowledge Graph Display Improvements")
    
    # Test case results formatting
    st.header("Test Case Results Formatting")
    
    # Format the test results
    formatted_cases = format_case_results(test_results, test_scores)
    
    # Display without similarity scores
    st.subheader("Without Similarity Scores")
    display_case_results(formatted_cases, show_similarity=False)
    
    # Display with similarity scores
    st.subheader("With Similarity Scores")
    display_case_results(formatted_cases, show_similarity=True)
    
    # Show raw vs formatted comparison
    with st.expander("Raw Results vs Formatted"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Results")
            st.json(test_results)
        with col2:
            st.subheader("Formatted Results")
            st.json(formatted_cases)

if __name__ == "__main__":
    main() 