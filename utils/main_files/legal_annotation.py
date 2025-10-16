import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Legal Case Annotation Tool", layout="wide")

def save_annotation(case_data, file_path):
    """Save the annotated case data to a JSON file"""
    with open(file_path, 'w') as f:
        json.dump(case_data, f, indent=2)
    return True

def load_case_file(file_path):
    """Load a case file from JSON"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    st.title("Legal Case Annotation Tool")
    
    # Sidebar for file operations
    with st.sidebar:
        st.header("File Operations")
        operation = st.radio("Select Operation", ["Load Existing Case", "Create New Case"])
        
        if operation == "Load Existing Case":
            # List available case files in the project directory
            case_files = [f for f in os.listdir() if f.endswith('.json') and f != 'requirements.json']
            selected_file = st.selectbox("Select a case file", case_files)
            
            if st.button("Load Case"):
                if selected_file:
                    case_data = load_case_file(selected_file)
                    if case_data:
                        st.session_state.case_data = case_data
                        st.session_state.file_path = selected_file
                        st.success(f"Loaded case: {case_data.get('title', 'Unnamed Case')}")
        else:
            # Create new case
            new_case_title = st.text_input("Case Title")
            if st.button("Create New Case"):
                if new_case_title:
                    case_data = {
                        "id": f"{new_case_title.lower().replace(' ', '_')}",
                        "source": "",
                        "title": new_case_title,
                        "court": "",
                        "judgment_date": "",
                        "content": "",
                        "entities": {
                            "cases": [],
                            "statutes": [],
                            "judges": [],
                            "jurisdictions": []
                        },
                        "metadata": {
                            "judges": "",
                            "primary_case_title": "",
                            "primary_case_number": "",
                            "primary_citation": ""
                        },
                        "cited_cases": [],
                        "final_decision": "",
                        "case_type": "",
                        "created_at": datetime.now().isoformat()
                    }
                    file_path = f"{case_data['id']}.json"
                    st.session_state.case_data = case_data
                    st.session_state.file_path = file_path
                    st.success(f"Created new case: {new_case_title}")
    
    # Main annotation area
    if 'case_data' in st.session_state:
        case_data = st.session_state.case_data
        
        # Basic case information
        st.header("Basic Case Information")
        col1, col2 = st.columns(2)
        
        with col1:
            case_data['title'] = st.text_input("Case Title", case_data.get('title', ''))
            case_data['court'] = st.text_input("Court", case_data.get('court', ''))
            case_data['judgment_date'] = st.date_input("Judgment Date", 
                                                     datetime.fromisoformat(case_data.get('judgment_date', datetime.now().isoformat())) 
                                                     if case_data.get('judgment_date') else None)
            
        with col2:
            case_data['source'] = st.text_input("Source", case_data.get('source', ''))
            case_data['case_type'] = st.text_input("Case Type", case_data.get('case_type', ''))
            case_data['final_decision'] = st.text_area("Final Decision", case_data.get('final_decision', ''), height=100)
        
        # Case content
        st.header("Case Content")
        case_data['content'] = st.text_area("Content", case_data.get('content', ''), height=300)
        
        # Entities annotation
        st.header("Entities")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Cited Cases", "Statutes", "Judges", "Jurisdictions", "Metadata"
        ])
        
        with tab1:
            st.subheader("Cited Cases")
            
            # Existing cited cases from entities
            existing_cases = case_data.get('entities', {}).get('cases', [])
            
            # Additional cited cases with more details
            if 'cited_cases' not in case_data:
                case_data['cited_cases'] = []
            
            # Display existing cited cases
            for i, cited_case in enumerate(case_data['cited_cases']):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    case_data['cited_cases'][i]['title'] = st.text_input(
                        f"Case Title #{i+1}", cited_case.get('title', ''))
                with col2:
                    case_data['cited_cases'][i]['citation'] = st.text_input(
                        f"Citation #{i+1}", cited_case.get('citation', ''))
                with col3:
                    case_data['cited_cases'][i]['relevance'] = st.selectbox(
                        f"Relevance #{i+1}", 
                        ["High", "Medium", "Low"],
                        index=["High", "Medium", "Low"].index(cited_case.get('relevance', 'Medium')))
                with col4:
                    if st.button(f"Remove #{i+1}"):
                        case_data['cited_cases'].pop(i)
                        st.experimental_rerun()
            
            # Add new cited case
            st.subheader("Add New Cited Case")
            new_case_col1, new_case_col2, new_case_col3 = st.columns([3, 2, 2])
            
            with new_case_col1:
                new_case_title = st.text_input("New Case Title", "")
            with new_case_col2:
                new_case_citation = st.text_input("New Case Citation", "")
            with new_case_col3:
                new_case_relevance = st.selectbox("Relevance", ["High", "Medium", "Low"], index=1)
            
            if st.button("Add Cited Case"):
                if new_case_title:
                    case_data['cited_cases'].append({
                        "title": new_case_title,
                        "citation": new_case_citation,
                        "relevance": new_case_relevance
                    })
                    
                    # Also add to entities.cases if not already there
                    if new_case_title not in case_data.get('entities', {}).get('cases', []):
                        if 'entities' not in case_data:
                            case_data['entities'] = {}
                        if 'cases' not in case_data['entities']:
                            case_data['entities']['cases'] = []
                        case_data['entities']['cases'].append(new_case_title)
                    
                    st.experimental_rerun()
        
        with tab2:
            st.subheader("Statutes Referenced")
            statutes = st.text_area("Statutes (one per line)", 
                                    "\n".join(case_data.get('entities', {}).get('statutes', [])))
            case_data['entities'] = case_data.get('entities', {})
            case_data['entities']['statutes'] = [s.strip() for s in statutes.split("\n") if s.strip()]
        
        with tab3:
            st.subheader("Judges")
            judges = st.text_area("Judges (one per line)", 
                                 "\n".join(case_data.get('entities', {}).get('judges', [])))
            case_data['entities'] = case_data.get('entities', {})
            case_data['entities']['judges'] = [j.strip() for j in judges.split("\n") if j.strip()]
            
            # Also update metadata.judges
            case_data['metadata'] = case_data.get('metadata', {})
            case_data['metadata']['judges'] = ", ".join(case_data['entities']['judges'])
        
        with tab4:
            st.subheader("Jurisdictions")
            jurisdictions = st.text_area("Jurisdictions (one per line)", 
                                       "\n".join(case_data.get('entities', {}).get('jurisdictions', [])))
            case_data['entities'] = case_data.get('entities', {})
            case_data['entities']['jurisdictions'] = [j.strip() for j in jurisdictions.split("\n") if j.strip()]
        
        with tab5:
            st.subheader("Additional Metadata")
            case_data['metadata'] = case_data.get('metadata', {})
            case_data['metadata']['primary_case_title'] = st.text_input(
                "Primary Case Title", case_data['metadata'].get('primary_case_title', ''))
            case_data['metadata']['primary_case_number'] = st.text_input(
                "Primary Case Number", case_data['metadata'].get('primary_case_number', ''))
            case_data['metadata']['primary_citation'] = st.text_input(
                "Primary Citation", case_data['metadata'].get('primary_citation', ''))
        
        # Save changes
        if st.button("Save Annotations"):
            file_path = st.session_state.file_path
            # Convert date to string if needed
            if isinstance(case_data['judgment_date'], datetime.date):
                case_data['judgment_date'] = case_data['judgment_date'].isoformat()
                
            save_annotation(case_data, file_path)
            st.success(f"Saved annotations to {file_path}")

if __name__ == "__main__":
    main() 