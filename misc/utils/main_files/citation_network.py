#!/usr/bin/env python3
"""
Citation Network Analysis
Extract and analyze citation relationships between legal cases
"""

import re
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import streamlit as st
from typing import List, Dict, Set, Tuple
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

class CitationExtractor:
    """Extract citations from legal case text"""
    
    def __init__(self):
        # Common citation patterns for Indian legal cases
        self.citation_patterns = [
            # AIR patterns: AIR 1950 SC 124
            r'AIR\s+(\d{4})\s+([A-Z]+)\s+(\d+)',
            # SCC patterns: (1950) 1 SCC 124
            r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)',
            # High Court patterns: 2020 SCC OnLine Del 1234
            r'(\d{4})\s+SCC\s+OnLine\s+([A-Za-z]+)\s+(\d+)',
            # General case number patterns: Crl.A. 123/2020
            r'([A-Za-z\.]+)\s*(\d+)/(\d{4})',
            # Supreme Court patterns: Civil Appeal No. 1234 of 2020
            r'Civil\s+Appeal\s+No\.\s*(\d+)\s+of\s+(\d{4})',
            # Writ Petition patterns: W.P.(C) 1234/2020
            r'W\.P\.\([A-Z]\)\s*(\d+)/(\d{4})'
        ]
    
    def extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from case text"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = {
                    'full_text': match.group(0),
                    'groups': match.groups(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                }
                citations.append(citation)
        
        return citations
    
    def extract_case_references(self, text: str) -> Set[str]:
        """Extract referenced case names"""
        # Pattern for case names like "State v. Kumar" or "ABC Ltd. v. XYZ Corp."
        case_name_pattern = r'([A-Z][a-zA-Z\s&\.]+)\s+v[s]?\.\s+([A-Z][a-zA-Z\s&\.]+)'
        
        matches = re.finditer(case_name_pattern, text)
        case_names = set()
        
        for match in matches:
            case_name = match.group(0).strip()
            # Filter out common false positives
            if len(case_name) > 10 and 'versus' not in case_name.lower():
                case_names.add(case_name)
        
        return case_names

class CitationNetwork:
    """Analyze citation networks in legal knowledge graph"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.extractor = CitationExtractor()
    
    def build_citation_network(self):
        """Build citation relationships in the knowledge graph"""
        st.info("Building citation network... This may take a few minutes.")
        
        # Get all cases
        cases_query = """
        MATCH (c:Case)
        RETURN id(c) as case_id, c.title as title, c.text as text
        """
        
        cases = self.graph.query(cases_query)
        total_cases = len(cases)
        
        if total_cases == 0:
            st.warning("No cases found in knowledge graph")
            return
        
        progress_bar = st.progress(0)
        citations_found = 0
        
        for i, case in enumerate(cases):
            case_id = case['case_id']
            case_text = case.get('text', '')
            
            if case_text:
                # Extract citations from case text
                citations = self.extractor.extract_citations(case_text)
                case_references = self.extractor.extract_case_references(case_text)
                
                # Try to match citations with existing cases
                for citation in citations:
                    cited_case = self.find_matching_case(citation)
                    if cited_case:
                        self.create_citation_relationship(case_id, cited_case['case_id'])
                        citations_found += 1
                
                # Try to match case name references
                for case_ref in case_references:
                    cited_case = self.find_case_by_name(case_ref)
                    if cited_case:
                        self.create_citation_relationship(case_id, cited_case['case_id'])
                        citations_found += 1
            
            # Update progress
            progress_bar.progress((i + 1) / total_cases)
        
        st.success(f"Citation network built! Found {citations_found} citations.")
    
    def find_matching_case(self, citation: Dict) -> Dict:
        """Find a case that matches the citation"""
        # This is a simplified matching - in practice, you'd want more sophisticated matching
        citation_text = citation['full_text']
        
        # Search for cases with similar citation patterns in title or text
        search_query = f"""
        MATCH (c:Case)
        WHERE toLower(c.title) CONTAINS toLower('{citation_text}')
           OR toLower(c.text) CONTAINS toLower('{citation_text}')
        RETURN id(c) as case_id, c.title as title
        LIMIT 1
        """
        
        try:
            results = self.graph.query(search_query)
            return results[0] if results else None
        except:
            return None
    
    def find_case_by_name(self, case_name: str) -> Dict:
        """Find a case by its name"""
        # Clean the case name for searching
        clean_name = case_name.replace("'", "\\'")
        
        search_query = f"""
        MATCH (c:Case)
        WHERE toLower(c.title) CONTAINS toLower('{clean_name}')
        RETURN id(c) as case_id, c.title as title
        LIMIT 1
        """
        
        try:
            results = self.graph.query(search_query)
            return results[0] if results else None
        except:
            return None
    
    def create_citation_relationship(self, citing_case_id: int, cited_case_id: int):
        """Create a CITES relationship between cases"""
        if citing_case_id == cited_case_id:
            return  # Don't create self-citations
        
        create_query = f"""
        MATCH (citing:Case), (cited:Case)
        WHERE id(citing) = {citing_case_id} AND id(cited) = {cited_case_id}
        MERGE (citing)-[:CITES]->(cited)
        """
        
        try:
            self.graph.query(create_query)
        except Exception as e:
            print(f"Error creating citation relationship: {e}")
    
    def get_citation_statistics(self) -> Dict:
        """Get citation network statistics"""
        stats = {}
        
        # Total citations
        citation_count_query = """
        MATCH ()-[r:CITES]->()
        RETURN count(r) as citation_count
        """
        result = self.graph.query(citation_count_query)
        stats['total_citations'] = result[0]['citation_count'] if result else 0
        
        # Most cited cases
        most_cited_query = """
        MATCH (cited:Case)<-[r:CITES]-(citing:Case)
        RETURN cited.title as case_title, count(r) as citation_count
        ORDER BY citation_count DESC
        LIMIT 5
        """
        stats['most_cited'] = self.graph.query(most_cited_query)
        
        # Most citing cases
        most_citing_query = """
        MATCH (citing:Case)-[r:CITES]->(cited:Case)
        RETURN citing.title as case_title, count(r) as citations_made
        ORDER BY citations_made DESC
        LIMIT 5
        """
        stats['most_citing'] = self.graph.query(most_citing_query)
        
        # Citation network density
        total_cases_query = "MATCH (c:Case) RETURN count(c) as case_count"
        case_result = self.graph.query(total_cases_query)
        total_cases = case_result[0]['case_count'] if case_result else 0
        
        if total_cases > 1:
            max_possible_citations = total_cases * (total_cases - 1)
            stats['network_density'] = stats['total_citations'] / max_possible_citations if max_possible_citations > 0 else 0
        else:
            stats['network_density'] = 0
        
        return stats
    
    def visualize_citation_network(self, limit: int = 50):
        """Create a visualization of the citation network"""
        # Get citation data
        citation_query = f"""
        MATCH (citing:Case)-[r:CITES]->(cited:Case)
        RETURN id(citing) as citing_id, citing.title as citing_title,
               id(cited) as cited_id, cited.title as cited_title
        LIMIT {limit}
        """
        
        citations = self.graph.query(citation_query)
        
        if not citations:
            st.warning("No citations found in the network")
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()  # Directed graph for citations
        
        # Add nodes and edges
        for citation in citations:
            citing_id = citation['citing_id']
            cited_id = citation['cited_id']
            citing_title = citation['citing_title'][:30] + "..." if len(citation['citing_title']) > 30 else citation['citing_title']
            cited_title = citation['cited_title'][:30] + "..." if len(citation['cited_title']) > 30 else citation['cited_title']
            
            G.add_node(citing_id, title=citing_title, type='citing')
            G.add_node(cited_id, title=cited_title, type='cited')
            G.add_edge(citing_id, cited_id)
        
        if len(G.nodes()) == 0:
            st.warning("No nodes to display")
            return None
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x, node_y, node_text, node_info = [], [], [], []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['title'])
            
            # Color based on citation count
            in_degree = G.in_degree(node)  # Times cited
            out_degree = G.out_degree(node)  # Times citing
            
            if in_degree > out_degree:
                node_colors.append('#ff6b6b')  # Red for highly cited
            elif out_degree > in_degree:
                node_colors.append('#4ecdc4')  # Teal for frequent citers
            else:
                node_colors.append('#45b7d1')  # Blue for balanced
            
            node_info.append(f"Case: {G.nodes[node]['title']}<br>Cited: {in_degree} times<br>Cites: {out_degree} cases")
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges (citations)
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Citations'
        ))
        
        # Add nodes (cases)
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            hoverinfo='text',
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name='Cases'
        ))
        
        fig.update_layout(
            title=dict(
                text="Citation Network - Who Cites Whom",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="Citation Network Analysis",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 Citation Network Analysis")
    st.markdown("Analyze citation relationships between legal cases")
    
    # Get Neo4j connection
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        st.error("Neo4j credentials not found in environment variables")
        return
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return
    
    citation_network = CitationNetwork(graph)
    
    # Sidebar controls
    st.sidebar.header("🔧 Citation Analysis Controls")
    
    # Build citation network
    if st.sidebar.button("🔨 Build Citation Network"):
        citation_network.build_citation_network()
    
    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Citation Statistics", "Citation Network Visualization", "Citation Patterns"]
    )
    
    if analysis_type == "Citation Statistics":
        st.subheader("📊 Citation Statistics")
        
        stats = citation_network.get_citation_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Citations", stats['total_citations'])
        with col2:
            st.metric("Network Density", f"{stats['network_density']:.4f}")
        with col3:
            st.metric("Citation Types", "1 (CITES)")
        
        # Most cited cases
        st.subheader("🏆 Most Cited Cases")
        if stats['most_cited']:
            for i, case in enumerate(stats['most_cited']):
                st.write(f"{i+1}. **{case['case_title'][:60]}...** - {case['citation_count']} citations")
        else:
            st.info("No citation data available")
        
        # Most citing cases
        st.subheader("📝 Most Active Citing Cases")
        if stats['most_citing']:
            for i, case in enumerate(stats['most_citing']):
                st.write(f"{i+1}. **{case['case_title'][:60]}...** - cites {case['citations_made']} cases")
        else:
            st.info("No citation data available")
    
    elif analysis_type == "Citation Network Visualization":
        st.subheader("🕸️ Citation Network Visualization")
        
        limit = st.slider("Number of citations to display", 10, 100, 30)
        
        fig = citation_network.visualize_citation_network(limit)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Color Legend:**
            - 🔴 **Red**: Highly cited cases (more incoming citations)
            - 🟢 **Teal**: Active citing cases (more outgoing citations)  
            - 🔵 **Blue**: Balanced cases (equal in/out citations)
            
            **Arrows show citation direction**: A → B means "A cites B"
            """)
        else:
            st.warning("No citation network to display")
    
    elif analysis_type == "Citation Patterns":
        st.subheader("🔍 Citation Patterns")
        
        st.info("This section analyzes citation patterns and trends")
        
        # Add pattern analysis here
        st.markdown("""
        **Citation Pattern Analysis:**
        - Identify citation clusters
        - Find influential cases
        - Analyze citation evolution over time
        - Detect citation communities
        """)

if __name__ == "__main__":
    main() 