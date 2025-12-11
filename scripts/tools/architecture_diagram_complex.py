"""
LegalNexus System Architecture - COMPLEX RESEARCH-LEVEL DIAGRAM
Full detailed architecture with all components, metrics, and data flows
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import Neo4J, MongoDB
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.network import Nginx
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.generic.device import Tablet, Mobile
from diagrams.programming.framework import FastAPI
from diagrams.onprem.queue import Kafka

# Ultra-detailed graph configuration for research papers
graph_attr = {
    "fontsize": "18",
    "bgcolor": "white",
    "pad": "0.8",
    "ranksep": "1.5",
    "nodesep": "1.0",
    "splines": "spline",
    "dpi": "600",  # High-resolution for papers
    "compound": "true",
    "concentrate": "false"
}

node_attr = {
    "fontsize": "11",
    "height": "1.4",
    "width": "2.2",
    "fontname": "Arial"
}

edge_attr = {
    "penwidth": "2.5",
    "fontsize": "10"
}

with Diagram(
    "LegalNexus: Hybrid Multi-Modal Legal Case Retrieval System",
    filename="legalnexus_architecture_complex",
    outformat=["png", "pdf"],  # Generate both PNG and PDF
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr
):
    
    # ============ DATA INPUT & SOURCES ============
    with Cluster("Data Sources & Input Layer"):
        with Cluster("Legal Databases"):
            legal_input = Storage("Supreme Court\nCases Database\n49,633 cases\n(1950-2024)")
            indian_kanoon = Storage("Indian Kanoon\nPublic Repository")
        
        with Cluster("User Interfaces"):
            web_ui = Nginx("Web Interface\nStreamlit/FastAPI")
            api_endpoint = FastAPI("REST API\n/search\n/analyze")
            mobile_ui = Mobile("Mobile App\n(Future)")
    
    # ============ DATA PREPROCESSING PIPELINE ============
    with Cluster("Stage 1: Data Ingestion & Preprocessing (2-5s)"):
        with Cluster("Text Processing"):
            json_parser = Server("JSON Parser\nSchema Validation")
            text_splitter = Server("RecursiveChar\nTextSplitter\n300|30 chars")
            
        with Cluster("Entity Recognition"):
            entity_ner = Python("Entity Extraction\nRegex + NER")
            judge_extractor = Server("Judge Extractor\n142 unique")
            statute_extractor = Server("Statute Extractor\n87 references")
            citation_extractor = Server("Citation Extractor\n234 links")
    
    # ============ COMPONENT 1: SEMANTIC EMBEDDINGS ============
    with Cluster("Component 1: Deep Semantic Embeddings (3-10s/doc)"):
        with Cluster("Embedding Generation"):
            jina_model = Python("Jina AI v3\nText Embeddings\nModel: jina-base")
            gemini_embed = Python("Google Gemini\nembedding-001\n768 dimensions")
            
        with Cluster("Vector Processing"):
            chunk_embed = Server("Chunk-level\nEmbeddings\n(n chunks)")
            doc_aggregation = Server("Document\nAggregation\nMean Pooling")
            l2_normalize = Server("L2 Normalization\nCosine-ready")
        
        with Cluster("Embedding Storage"):
            pkl_cache = Storage("PKL Cache\n~50MB\n100% Hit Rate")
            vector_backup = Storage("Backup Store\nVersioned")
    
    # ============ COMPONENT 2: KNOWLEDGE GRAPH ============
    with Cluster("Component 2: Legal Knowledge Graph (2-5s/case)"):
        with Cluster("Graph Construction"):
            node_creator = Server("Node Creation\nCases|Judges\nCourts|Statutes")
            rel_builder = Server("Relationship\nBuilder\nJUDGED|CITES")
            
        with Cluster("Neo4j Database"):
            neo4j_main = Neo4J("Neo4j v5.x\n50 cases\n~500 nodes\n~1200 rels")
            vector_index = SQL("Vector Index\ncosine, 768-dim\n<100ms query")
            graph_algo = Server("Graph Analytics\nPageRank\nCentrality")
        
        with Cluster("Graph Features"):
            centrality_calc = Server("Centrality Metrics\nDegree|Between\nPageRank")
            community_detect = Server("Community\nDetection\nLouvain Algo")
    
    # ============ CONVERGENCE: UNIFIED KNOWLEDGE BASE ============
    with Cluster("Unified Knowledge Base Layer"):
        kb_fusion = Rack("Knowledge Fusion\nEngine\nMulti-modal Integration")
        metadata_index = SQL("Metadata Index\nCourt|Date|Type")
        relationship_cache = Storage("Relationship\nCache\nFast Access")
    
    # ============ QUERY PROCESSING LAYER ============
    with Cluster("Query Processing & Intent Recognition"):
        query_input = Users("User Query\nNatural Language\n'Find cases on...'\n")
        
        with Cluster("LLM Query Engine"):
            query_expansion = Python("Query Expansion\nGemini 2.5 Flash\nSynonyms|Context")
            intent_classifier = Python("Intent Classification\nSimilarity|Graph|Hybrid")
            cypher_generator = Python("Cypher Generator\nNL→Query Trans")
        
        with Cluster("Query Optimization"):
            query_planner = Server("Query Planner\nCost Estimation")
            cache_checker = Server("Cache Lookup\nRedis/Memcache")
    
    # ============ COMPONENT 3: HYBRID RETRIEVAL SYSTEM ============
    with Cluster("Component 3: Multi-Strategy Hybrid Retrieval"):
        
        with Cluster("Strategy 1: Vector Similarity (α=0.6)"):
            vec_query_embed = Python("Query Embedding\nSame as docs")
            vec_similarity = Server("Cosine Similarity\nTop-K Retrieval")
            vec_scorer = Server("Vector Score\nNormalized [0,1]")
        
        with Cluster("Strategy 2: Keyword Search (β=0.3)"):
            keyword_extract = Server("Keyword Extract\nTF-IDF|Rake")
            fulltext_search = Neo4J("Full-text Index\nNeo4j Search")
            keyword_scorer = Server("Keyword Score\nBM25 Ranking")
        
        with Cluster("Strategy 3: Graph Traversal (γ=0.1)"):
            graph_query = Server("Graph Query\nCypher Exec")
            path_finder = Server("Path Finding\nShortest Paths")
            graph_scorer = Server("Graph Score\nStruct. Similarity")
        
        with Cluster("Additional Strategies"):
            tfidf_retrieval = Server("TF-IDF\nClassical IR")
            pagerank_retrieval = Server("PageRank\nAuthority-based")
        
        with Cluster("Fusion Layer"):
            score_aggregation = Rack("Score Aggregation\nWeighted Sum\nα+β+γ=1.0")
            reciprocal_rank = Server("Reciprocal Rank\nFusion (RRF)")
            diversity_filter = Server("Diversity Filter\nMMR Algorithm")
    
    # ============ POST-PROCESSING & RANKING ============
    with Cluster("Ranking & Re-ranking Pipeline"):
        with Cluster("Initial Ranking"):
            top_k_selector = Server("Top-K Selection\nK=10 initially")
            threshold_filter = Server("Threshold Filter\n≥0.70 cutoff")
        
        with Cluster("LLM Re-ranking"):
            llm_reranker = Python("Gemini Re-ranker\nSemantic Relevance")
            cross_encoder = Python("Cross-Encoder\nQuery-Doc Scoring")
        
        with Cluster("Final Selection"):
            final_ranker = Server("Final Ranking\nCombined Scores")
            top_5_selector = Server("Top-5 Selection\nPresentation")
    
    # ============ COMPONENT 4: LLM ANALYSIS & INSIGHTS ============
    with Cluster("Component 4: Legal Analysis & Synthesis"):
        with Cluster("Comparative Analysis"):
            case_comparator = Python("Case Comparator\nSide-by-side\nGemini 2.5")
            similarity_explainer = Python("Similarity Explain\nReason Generation")
        
        with Cluster("Legal Reasoning"):
            precedent_analyzer = Python("Precedent Analysis\nDoctrine Extract")
            statute_mapper = Python("Statute Mapping\nProvision Links")
            summary_generator = Python("Summary Gen.\nAbstractive")
        
        with Cluster("Insight Generation"):
            trend_analyzer = Server("Trend Analysis\nTemporal Patterns")
            citation_network = Server("Citation Network\nInfluence Graph")
    
    # ============ OUTPUT & PRESENTATION ============
    with Cluster("Output Layer & Presentation"):
        with Cluster("Result Formatting"):
            json_formatter = Server("JSON Formatter\nStructured Output")
            html_renderer = Server("HTML Renderer\nWeb Display")
            
        with Cluster("Visualization"):
            graph_visualizer = Server("Graph Viz\nD3.js/Cytoscape")
            timeline_viz = Server("Timeline View\nChronological")
            
        final_output = Storage("Ranked Results\n━━━━━━━━━━━━━━━\nTop 5 Cases\n+ Similarity Scores\n+ Analysis\n+ Explanations\n━━━━━━━━━━━━━━━\nP@5: 92%\nR@5: 89%\nF1: 0.91\nMAP: 0.91\n")
    
    # ============ MONITORING & ANALYTICS ============
    with Cluster("System Monitoring & Analytics"):
        perf_monitor = Server("Performance\nMonitor\nLatency Tracking")
        quality_metrics = Server("Quality Metrics\nP/R/F1/NDCG")
        error_logger = Server("Error Logger\nException Track")
        analytics_db = MongoDB("Analytics DB\nQuery Logs\nUser Behavior")
    
    # ========================================
    # DATA FLOW CONNECTIONS
    # ========================================
    
    # Input to preprocessing
    legal_input >> Edge(label="Load", color="blue", style="bold") >> json_parser
    indian_kanoon >> Edge(color="blue", style="dashed") >> json_parser
    json_parser >> Edge(label="Validate", color="green") >> text_splitter
    text_splitter >> Edge(label="Chunks", color="green") >> entity_ner
    
    # Entity extraction flow
    entity_ner >> Edge(color="purple") >> judge_extractor
    entity_ner >> Edge(color="purple") >> statute_extractor
    entity_ner >> Edge(color="purple") >> citation_extractor
    
    # Component 1: Embedding flow
    text_splitter >> Edge(label="Text", color="purple", style="bold") >> gemini_embed
    gemini_embed >> Edge(label="768-d", color="purple") >> chunk_embed
    chunk_embed >> Edge(color="purple") >> doc_aggregation
    doc_aggregation >> Edge(color="purple") >> l2_normalize
    l2_normalize >> Edge(label="Cached", color="purple") >> pkl_cache
    pkl_cache >> Edge(color="purple", style="dashed") >> vector_backup
    
    # Component 2: Graph flow
    judge_extractor >> Edge(label="Entities", color="orange") >> node_creator
    statute_extractor >> Edge(color="orange") >> node_creator
    citation_extractor >> Edge(color="orange") >> rel_builder
    node_creator >> Edge(label="Nodes", color="orange", style="bold") >> neo4j_main
    rel_builder >> Edge(label="Edges", color="orange", style="bold") >> neo4j_main
    neo4j_main >> Edge(color="orange") >> vector_index
    neo4j_main >> Edge(color="orange") >> graph_algo
    graph_algo >> Edge(color="orange") >> centrality_calc
    graph_algo >> Edge(color="orange") >> community_detect
    
    # Convergence to KB
    pkl_cache >> Edge(label="Embeddings", color="purple", style="bold") >> kb_fusion
    neo4j_main >> Edge(label="Graph", color="orange", style="bold") >> kb_fusion
    centrality_calc >> Edge(label="Features", color="orange") >> kb_fusion
    kb_fusion >> Edge(color="darkgreen") >> metadata_index
    kb_fusion >> Edge(color="darkgreen") >> relationship_cache
    
    # Query processing
    web_ui >> Edge(label="HTTP", color="red") >> query_input
    api_endpoint >> Edge(color="red") >> query_input
    mobile_ui >> Edge(color="red", style="dashed") >> query_input
    
    query_input >> Edge(label="NL Query", color="red", style="bold") >> query_expansion
    query_expansion >> Edge(color="red") >> intent_classifier
    intent_classifier >> Edge(color="red") >> cypher_generator
    query_expansion >> Edge(color="red") >> cache_checker
    
    # Hybrid retrieval - Strategy 1
    query_expansion >> Edge(label="Q_embed", color="gold", style="bold") >> vec_query_embed
    vec_query_embed >> Edge(color="gold") >> vec_similarity
    kb_fusion >> Edge(label="Lookup", color="gray", style="dotted") >> vec_similarity
    vec_similarity >> Edge(label="Scores", color="gold") >> vec_scorer
    
    # Hybrid retrieval - Strategy 2
    query_expansion >> Edge(label="Keywords", color="gold") >> keyword_extract
    keyword_extract >> Edge(color="gold") >> fulltext_search
    kb_fusion >> Edge(color="gray", style="dotted") >> fulltext_search
    fulltext_search >> Edge(color="gold") >> keyword_scorer
    
    # Hybrid retrieval - Strategy 3
    cypher_generator >> Edge(label="Cypher", color="gold") >> graph_query
    graph_query >> Edge(color="gold") >> path_finder
    kb_fusion >> Edge(color="gray", style="dotted") >> graph_query
    path_finder >> Edge(color="gold") >> graph_scorer
    
    # Additional strategies
    query_expansion >> Edge(color="gold", style="dashed") >> tfidf_retrieval
    graph_algo >> Edge(color="gold", style="dashed") >> pagerank_retrieval
    
    # Fusion
    vec_scorer >> Edge(label="α=0.6", color="gold", style="bold") >> score_aggregation
    keyword_scorer >> Edge(label="β=0.3", color="gold", style="bold") >> score_aggregation
    graph_scorer >> Edge(label="γ=0.1", color="gold", style="bold") >> score_aggregation
    tfidf_retrieval >> Edge(color="gold") >> reciprocal_rank
    pagerank_retrieval >> Edge(color="gold") >> reciprocal_rank
    
    score_aggregation >> Edge(label="Fused", color="gold", style="bold") >> reciprocal_rank
    reciprocal_rank >> Edge(color="gold") >> diversity_filter
    
    # Ranking pipeline
    diversity_filter >> Edge(label="Top-K", color="darkgreen", style="bold") >> top_k_selector
    top_k_selector >> Edge(color="darkgreen") >> threshold_filter
    threshold_filter >> Edge(label="≥0.70", color="darkgreen") >> llm_reranker
    llm_reranker >> Edge(color="darkgreen") >> cross_encoder
    cross_encoder >> Edge(color="darkgreen") >> final_ranker
    final_ranker >> Edge(label="Top-5", color="darkgreen", style="bold") >> top_5_selector
    
    # Analysis
    top_5_selector >> Edge(label="Results", color="red", style="bold") >> case_comparator
    case_comparator >> Edge(color="red") >> similarity_explainer
    case_comparator >> Edge(color="red") >> precedent_analyzer
    precedent_analyzer >> Edge(color="red") >> statute_mapper
    similarity_explainer >> Edge(color="red") >> summary_generator
    
    # Additional insights
    final_ranker >> Edge(color="blue", style="dashed") >> trend_analyzer
    neo4j_main >> Edge(color="blue", style="dashed") >> citation_network
    
    # Output
    summary_generator >> Edge(label="Analysis", color="darkgreen", style="bold") >> json_formatter
    trend_analyzer >> Edge(color="blue") >> json_formatter
    citation_network >> Edge(color="blue") >> graph_visualizer
    json_formatter >> Edge(color="darkgreen") >> html_renderer
    html_renderer >> Edge(label="Display", color="darkgreen", style="bold") >> final_output
    graph_visualizer >> Edge(color="blue") >> final_output
    timeline_viz >> Edge(color="blue") >> final_output
    
    # Monitoring
    query_input >> Edge(color="gray", style="dotted") >> perf_monitor
    final_output >> Edge(color="gray", style="dotted") >> quality_metrics
    llm_reranker >> Edge(color="gray", style="dotted") >> error_logger
    perf_monitor >> Edge(color="gray") >> analytics_db
    quality_metrics >> Edge(color="gray") >> analytics_db

print("✅ Complex LegalNexus architecture diagram generated successfully!")
print("📁 Output files:")
print("   - legalnexus_architecture_complex.png (High-res 600 DPI)")
print("   - legalnexus_architecture_complex.pdf (Vector format for papers)")
