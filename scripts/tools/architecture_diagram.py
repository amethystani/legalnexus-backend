"""
LegalNexus System Architecture Diagram
Professional research-level diagram using Python diagrams library
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.programming.language import Python
from diagrams.onprem.database import Neo4J
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.generic.blank import Blank
import os

# Custom attributes for styling
graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "pad": "0.5",
    "ranksep": "1.0",
    "nodesep": "0.8",
    "splines": "ortho",
    "dpi": "300"
}

node_attr = {
    "fontsize": "12",
    "height": "1.2",
    "width": "2.0"
}

edge_attr = {
    "penwidth": "2.0"
}

with Diagram(
    "LegalNexus System Architecture",
    filename="legalnexus_architecture",
    outformat="png",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr
):
    
    # ============ INPUT LAYER ============
    with Cluster("Data Input Layer"):
        legal_cases = Storage("Legal Cases\n49,633 SC Cases\n(1950-2024)")
    
    # ============ PREPROCESSING LAYER ============
    with Cluster("Preprocessing Pipeline"):
        text_chunk = Server("Text Chunking\n300 chars/chunk\n30 char overlap")
        entity_extract = Server("Entity Extraction\nJudges | Statutes\nCourts | Citations")
    
    # ============ DUAL PROCESSING BRANCHES ============
    with Cluster("Component 1: Semantic Embeddings"):
        jina_embed = Python("Jina v3\nEmbeddings\n768-dim vectors")
        embed_cache = Storage("Embedding Cache\nPKL Storage")
        jina_embed >> Edge(label="Store", color="purple") >> embed_cache
    
    with Cluster("Component 2: Knowledge Graph"):
        graph_builder = Rack("Graph Builder\nNodes + Relationships")
        neo4j_db = Neo4J("Neo4j Database\n50 Cases\n142 Judges\n15 Courts")
        graph_builder >> Edge(label="Persist", color="orange") >> neo4j_db
    
    # ============ CONVERGENCE LAYER ============
    with Cluster("Unified Knowledge Base"):
        kb_layer = SQL("Legal Knowledge Base\nCases + Entities + Embeddings + Graph")
    
    # ============ QUERY PROCESSING ============
    user_query = Users("User Query\nNatural Language\nQuestion")
    
    with Cluster("Query Processing Engine"):
        llm_processor = Python("Gemini 2.5 Flash\nQuery Processor\nIntent Classification")
    
    # ============ MULTI-STRATEGY RETRIEVAL ============
    with Cluster("Component 3: Hybrid Retrieval System"):
        vector_search = Server("Vector Search\nCosine Similarity\nα=0.6")
        keyword_search = Server("Keyword Search\nFull-text Index\nβ=0.3")
        graph_traverse = Server("Graph Traversal\nRelationship-based\nγ=0.1")
        
        fusion_engine = Rack("Hybrid Fusion\nDynamic Weighting\nScore Aggregation")
        
        vector_search >> fusion_engine
        keyword_search >> fusion_engine
        graph_traverse >> fusion_engine
    
    # ============ RANKING & ANALYSIS ============
    with Cluster("Post-Processing"):
        ranker = Server("Ranking & Filtering\nTop-5 Results\nThreshold: 0.70")
        llm_analysis = Python("LLM Analysis\nComparative Reasoning\nLegal Insights")
    
    # ============ OUTPUT ============
    output_results = Storage("Ranked Results\nSimilar Cases + Scores\nAnalysis + Reasoning\n\nP@5: 92% | R@5: 89%")
    
    # ============ DATA FLOW CONNECTIONS ============
    
    # Input to preprocessing
    legal_cases >> Edge(color="blue", style="bold") >> text_chunk
    text_chunk >> Edge(color="green") >> entity_extract
    
    # Preprocessing to dual branches
    entity_extract >> Edge(color="purple", label="Text") >> jina_embed
    entity_extract >> Edge(color="orange", label="Entities") >> graph_builder
    
    # Dual branches to knowledge base
    embed_cache >> Edge(color="purple", style="dashed") >> kb_layer
    neo4j_db >> Edge(color="orange", style="dashed") >> kb_layer
    
    # Query flow
    user_query >> Edge(color="red", style="bold") >> llm_processor
    llm_processor >> Edge(color="gold") >> vector_search
    llm_processor >> Edge(color="gold") >> keyword_search
    llm_processor >> Edge(color="gold") >> graph_traverse
    
    # Knowledge base to retrieval
    kb_layer >> Edge(color="gray", style="dotted", label="Lookup") >> vector_search
    kb_layer >> Edge(color="gray", style="dotted") >> keyword_search
    kb_layer >> Edge(color="gray", style="dotted") >> graph_traverse
    
    # Fusion to ranking
    fusion_engine >> Edge(color="gold", style="bold") >> ranker
    ranker >> Edge(color="red") >> llm_analysis
    llm_analysis >> Edge(color="darkgreen", style="bold", label="Final") >> output_results

print("✅ LegalNexus architecture diagram generated successfully!")
print("📁 Output file: legalnexus_architecture.png")
