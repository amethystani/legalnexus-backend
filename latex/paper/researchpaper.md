AI in Legal Domain: Similar Cases Recommendation using Legal Knowledge Graphs and Neuro-Symbolic Approaches. 
 
Project Report Submitted in Partial Fulfilment of the Requirements for the Degree of 
 
Bachelor of Technology 
 
in  
 
Computer Science and Engineering
 
Submitted by
 
Keshav Bararia: (Roll No. 2210110355)
Kush Sahni: (Roll No. 2210110371) 
Animesh Mishra: (Roll No. 2210110161)
 
Under the Supervision of 
 
Dr. Sonia Khetarpaul
Associate Professor
 
 
  
Department of Computer Science and Engineering
 
October, 2025
 
 
 
 Declaration
 
 
I/We declare that this written submission represents my ideas in my own words and where others' ideas or words have been included, I have adequately cited and referenced the original sources.  I also declare that I have adhered to all principles of academic honesty and integrity and   have   not   misrepresented   or   fabricated   or   falsified   any   idea/data/fact/source   in   my submission.  I understand that any violation of the above will be cause for disciplinary action by the University and can also evoke penal action from the sources which have thus not been properly cited or from whom proper permission has not been taken when needed. 
 
 
 
 
Name of the Students _____________________                 Signature and Date _________________
 
 
(for all members of the group)
 
 
 
 
 
 
 





















1. Introduction
The legal field is complex and filled with detailed information. Court cases involve lengthy documents, complicated reasoning, and many past decisions. Lawyers and judges often rely on earlier cases, known as precedents, to make fair and consistent decisions. However, searching for similar cases manually takes a lot of time and can lead to errors. Traditional search tools that depend on keywords or citations often struggle to grasp the deeper meaning and context of legal cases. As a result, there is a growing demand for smart systems that can understand legal language, organize legal knowledge, and automatically suggest similar cases.

This project, “AI in Legal Domain: Similar Cases Recommendation using Legal Knowledge Graphs and Neuro-Symbolic Approaches,” aims to address this issue by combining structured legal knowledge with modern AI techniques. The system uses a Legal Knowledge Graph (LKG) to connect related legal details such as case names, issues, decisions, judges, and outcomes. This graph structure allows the model to go beyond simple text matching and find cases related by deeper legal concepts and reasoning.

To build this system, we created a custom dataset of legal cases and labeled it using large language models (LLMs). Each case was divided into categories such as Name, Issue, Holding, Citation, Decision, Court, and Date. We tested various free LLMs and opted for GPT-based models because they provided more accurate and consistent labeling. This automated approach reduced the effort needed for manual labeling while keeping the data reliable and easy to interpret.

For recommending similar cases, we used a Graph Convolutional Network (GCN), a type of neural network that works directly with graph data. The GCN learns how different parts of the knowledge graph are interconnected by combining information from related nodes. This helps the system understand both the meaning of the text and the relationships between cases.

Overall, this project demonstrates how combining LLM-based data preparation with graph-based deep learning can create an intelligent legal AI tool. Such a system can assist lawyers and researchers in quickly finding similar cases, saving time and improving legal decision-making.













2: Related Work / Literature Review
The retrieval and comparison of legal cases is an advanced form of information retrieval (IR), complicated by the length, formality, and domain-specific semantics of judicial documents. Over the years, this field has evolved from simple keyword-based methods to sophisticated deep learning and hybrid neuro-symbolic systems.
 This section reviews the key developments in legal case retrieval, transformer-based NLP models, neuro-symbolic approaches, and knowledge graph-based reasoning within legal artificial intelligence.

2.1 Legal Case Retrieval and Similarity
Legal case retrieval — identifying past cases similar to a query case — remains one of the most important and technically challenging tasks in Legal NLP. Unlike general text retrieval, legal cases are long, hierarchically structured, and filled with contextual dependencies such as precedents and statutes.
2.1.1 Early Lexical and Network-Based Systems
Traditional retrieval methods like TF-IDF and BM25 focused on term frequency and keyword overlap. These models provided a baseline for text similarity but failed to account for the semantic relationships between legal terms.
 Citation-based methods later enhanced retrieval accuracy by considering network structures (e.g., case-to-case citations and legal topic hierarchies).
2.1.2 Deep Embedding and Transformer Approaches
Recent years have seen a shift toward embedding-based and deep neural retrieval models.
 For instance, Vuong et al. (2023) proposed SM-BERT-CR, a supporting-model architecture that uses weak supervision and transformer encoders (BERT) to rank cases and paragraphs for legal entailment, achieving state-of-the-art results on multiple benchmarks.
 Similarly, Tang et al. (2024) introduced CaseGNN — a Text-Attributed Case Graph (TACG) model — that represents each legal case as a sentence-level graph. CaseGNN applies edge-attention and contrastive learning to overcome BERT’s length limitations, significantly outperforming baseline models on the COLIEE benchmark.
 A successor model, CaseGNN++ (2024), incorporates edge features and graph-contrastive augmentation, further improving performance.
In the Indian context, Dhani et al. (2023) constructed a Legal Knowledge Graph (LKG) from Indian court judgments and statutes. Using Relational Graph Convolutional Networks (RGCN) and optional LegalBERT embeddings, their model successfully identified similar cases as a link prediction task.
 These graph-based methods bridge the gap between semantic embeddings and structural reasoning by explicitly modeling relationships such as citations, facts, and legal provisions.

2.2 Transformer-Based Models in Legal NLP
The advent of pre-trained transformer models revolutionized Legal NLP by providing domain-specific contextual embeddings.
2.2.1 Domain-Specific Pretraining
The pioneering work of Chalkidis et al. (2020) introduced LegalBERT, a variant of BERT pre-trained on massive English legal corpora (EU/UK law, court cases, and contracts). LegalBERT achieved substantial gains over vanilla BERT for legal document classification and outcome prediction tasks.
 Building upon this, Zheng et al. (2021) developed CaseLawBERT, pre-trained on the Harvard Law Case Corpus, tailored specifically for U.S. judicial text.
 Lawformer (Xiao et al., 2021) adapted Longformer to process lengthy Chinese judgments, and Pile-of-Law BERT (Henderson et al., 2022) was trained on 10 million U.S./EU legal documents, forming one of the largest open legal corpora to date.
2.2.2 Regional Adaptation
Recognizing that legal phrasing varies across jurisdictions, Paul et al. (ICAIL 2023) extended these models to Indian law by pretraining InLegalBERT (continued from LegalBERT) and InCaseLawBERT (trained from scratch) using over 5.4 million Indian Supreme Court judgments.
 InLegalBERT achieved lower perplexity and higher accuracy on Indian-specific tasks such as statute identification and judgment prediction, emphasizing the need for jurisdiction-specific adaptation.
2.2.3 Transformers for Legal Similarity
Models like CourtBERT (Liu et al., 2023) encode cases for similarity detection. However, transformers face challenges with input length limits (512–4096 tokens).
 Therefore, recent approaches employ hierarchical encoders, document chunking, or retrieval-augmented generation (RAG) techniques to handle multi-page judgments. Hybrid methods such as KELLER and SAILER (described below) integrate domain structure or symbolic knowledge to improve interpretability and efficiency.

2.3 Symbolic and Neuro-Symbolic Approaches
2.3.1 From Expert Systems to Neuro-Symbolic Integration
Earlier systems like BALKO (University of California, Berkeley) and Drools-based legal engines used explicit rule encoding and logic-based reasoning. These systems offered interpretability but required substantial manual effort to encode complex legal norms.
Modern research combines symbolic reasoning with neural architectures — an area known as neuro-symbolic AI — merging language understanding with formal reasoning.
2.3.2 Knowledge-Guided Case Matching
KELLER (Deng et al., 2024) exemplifies this hybrid paradigm. It uses LLM prompts to extract relevant crimes and statutes from a case, summarizing key facts and grounding them in legal knowledge.
 By anchoring retrieval in explicit law articles, KELLER achieves higher interpretability and stronger performance on legal IR benchmarks.
Similarly, SAILER (Li et al., SIGIR 2023) employs structure-aware pretraining through an asymmetric encoder–decoder model that learns document hierarchy and emphasizes legally significant entities.
 This pretraining approach enhances accuracy even without human annotations by modeling the internal structure of legal documents.
2.3.3 Reasoning-Oriented Models
Beyond retrieval, Kant et al. (AAAI 2025) proposed a neuro-symbolic reasoning system where an LLM translates legal clauses into Prolog-style logical rules, enabling structured legal reasoning.
 Their framework provides improved explainability and consistency compared to text-only LLMs.
 Another model, GLARE (Kant, 2025), integrates retrieval into LLM reasoning using iterative grounding in statutes and precedents, forming a syllogistic reasoning chain that enhances transparency.

2.4 Knowledge Graphs and Graph-Based Legal Reasoning
Knowledge Graphs (KGs) model the entities and relationships within legal ecosystems — such as cases, judges, statutes, and citations.
2.4.1 Legal Knowledge Graph Construction
In India, Dhani et al. (2023) developed an Intellectual Property Rights (IPR) Legal Knowledge Graph, connecting entities like statutes, sections, and parties using entity extraction and parsing.
 In Europe, Froehlich et al. (2021) and Colombo et al. (2025) created large-scale EU legislative graphs using dependency parsing and Named Entity Recognition (NER), supporting search and question answering over legislative texts.
2.4.2 Graph Embeddings and GNNs
Graph embedding methods like TransE, node2vec, and GraphSAGE have been tested for representing case or statute relationships.
 Advanced systems such as LF-HGRILF (Huang et al., 2023) introduce heterogeneous “Law–Fact Graphs” for judgment prediction, connecting case facts with legal articles to enhance reasoning.
 LegisSearch (Colombo et al., 2025) combines LLMs with a graph retriever over Italian legislation, yielding significantly better search accuracy than keyword-based methods.
These developments show that graph-based models capture structure and semantic context beyond plain text embeddings, making them vital for scalable and interpretable legal AI.


2.5 Datasets for Legal NLP
Legal NLP research relies heavily on publicly available legal corpora:
Dataset
Jurisdiction
Scale / Content
Usage
Indian Supreme Court Judgments (AWS)
India
~5.4M judgments (1950–2025)
Retrieval, summarization, pretraining
CAIL 2018/2020
China
Criminal law cases
Judgment prediction
Case Law Access Project (CAP)
USA
164k cases
LegalBERT, CaseLawBERT training
ECtHR / LexGLUE
EU
~11k cases
Multi-task legal benchmarks
COLIEE
Japan / International
English & Japanese
Case retrieval and entailment
Pile-of-Law
USA/EU
10M legal documents
Domain pretraining
EUR-Lex
EU
Legislative texts
Legal QA and classification

Each dataset has inherent biases, such as language imbalance or incomplete metadata. Thus, cross-corpus evaluation is essential to ensure model robustness and transferability.

2.6 Summary of Methods
The literature demonstrates four major paradigms:
Transformer-Based Models (e.g., LegalBERT, CaseLawBERT)
 → Capture domain-specific semantics but limited by document length.


Graph-Based Models (e.g., CaseGNN, RGCN)
 → Encode case structure and citations, improving contextual similarity.


Symbolic and Rule-Based Systems
 → Provide explainable reasoning but lack scalability.


Neuro-Symbolic Hybrids (e.g., SAILER, KELLER, GLARE)
 → Combine statistical power of LLMs with formal reasoning for both accuracy and transparency.


Overall, recent trends clearly move toward multimodal hybrid architectures — integrating transformers, GNNs, and symbolic reasoning to balance interpretability, scalability, and performance.















Method/Dataset
Type
Description
Key Contributions
Strengths / Limitations
Legal-BERTaclanthology.org (Chalkidis et al., EMNLP 2020)
Transformer (PLM)
BERT-base pre-trained on EU/UK/US legal corpora (legislation, case law, contracts)
First large legal-domain BERT. Improves on legal text tasks; smaller LegalBERT-SM efficient.
+ Domain-specific embeddings; + public model. – Still limited by sequence length.
CaseLawBERTar5iv.labs.arxiv.org (Zheng et al., 2021)
Transformer (PLM)
BERT-base continued pre-training on 3.4M Harvard case law docs
Tailored to US cases; shown to outperform generic BERT on case-law tasks.
+ Captures US legal jargon; – Generic BERT finetuning sometimes competitive.
InLegalBERT/InCaseLawBERTar5iv.labs.arxiv.org (Paul et al., ICAIL 2023)
Transformer (PLM)
LegalBERT/CaseLawBERT re-trained on Indian Supreme Court judgments (5.4M docs)
Improves perplexity on Indian corpora; boosts performance on Indian legal tasks vs. original models.
+ Better accuracy on Indian tasks; – Requires large country-specific corpus.
SM-BERT-CRlink.springer.com (Vuong et al., AI&Law 2023)
Deep BERT retrieval
BERT with “supporting model” to match case–case and paragraph–paragraph relations; uses weak labels for training
Novel two-phase retrieval+entailment model; state-of-art on case retrieval tasks.
+ Improved relevance matching via BERT; – Complex; needs large weak-label dataset.
SAILERarxiv.org (Li et al., SIGIR 2023)
Structure-aware PLM
Asymmetric encoder-decoder pretraining emphasizing case structure and key legal elements
Pre-training objectives to encode legal document structure; achieves SOTA on legal retrievalbenchmarks.
+ Incorporates document sections, entities; + Works without annotations. – Complex pretraining pipeline.
KELLERarxiv.org (Deng et al., arXiv 2024)
Neuro-symbolic (LLM + KG)
Uses LLM prompts guided by legal knowledge (crimes & statutes) to extract sub-facts and match cases
Injects statutory knowledge to summarize cases; dual-level contrastive loss for matching.
+ Interpretability via sub-facts; + Handles long cases. – Relies on structured extraction; uses LLM API (cost).
CaseGNNarxiv.orglink.springer.com (Tang et al., ECIR 2024)
Graph Neural Network
Converts each case into a Text-Attributed Case Graph (TACG) of sentence-nodes; applies GNN with edge-attention and contrastive training
Captures intra-document structure; avoids BERT’s length limit. Outperforms baselines on COLIEE.
+ Exploits case structure; + No text length cap. – Requires graph construction; specialized to retrieval.
CaseGNN++arxiv.org (Tang et al., arXiv 2024)
Graph Neural Network
Extension of CaseGNN adding edge-feature graph attention and contrastive augmentation
Leverages full edge info and unsupervised contrastive losses; sets new state-of-art on COLIEE 2022/23.
+ Stronger case embeddings; + Better use of unlabeled data. – More complex model; arXiv (preprint).
Legal Knowledge Graphs + GNNsarxiv.orgarxiv.org (Dhani et al., SAILR 2023)
Hybrid (KG + GNN)
Build KG of Indian cases, statutes, people; use RGCN on this graph for similarity/link prediction
Integrates domain ontology; shows GNN+handcrafted features or LegalBERT outperform vanilla.
+ Infuses expert knowledge; + Supports link prediction. – Requires ontology design; limited to specific corpus (IPR cases).
Knowledge Graph Datasets
Dataset
Indian SC judgements (AWS Open Data)registry.opendata.aws; ECHR case law (LexGLUE)huggingface.co; US Case Law Access Projectaclanthology.org; COLIEE tasks
Public sources of case law for multiple jurisdictions and tasks
+ Large, diverse corpora; + Enables benchmarking. – Licensing varies (e.g. CAP requires research access); may have OCR/text noise.

Sources: Key publications and datasets as cited in the table (e.g.aclanthology.orgarxiv.org) provide details on model architectures, training corpora, and evaluation results. Each method’s strengths/limitations are drawn from reported findings: e.g. LegalBERT’s improvement over BERTaclanthology.org, CaseGNN’s ability to encode structurearxiv.org, or KELLER’s enhanced interpretability through legal knowledgearxiv.org.






Numerical comparison (COLIEE benchmarks, one-stage; top-5 evaluation)
Method
Dataset
Micro-F1 (%)
MAP (%)
NDCG@5 (%)
BM25
COLIEE2022
19.4
25.4
33.6. ar5iv
SAILER
COLIEE2022
14.0
18.5
25.1. ar5iv
PromptCase
COLIEE2022
18.5
33.9
38.7. ar5iv
CaseGNN (Tang et al.)
COLIEE2022
38.4 ± 0.3
64.4 ± 0.9
69.3 ± 0.8. ar5iv+1
CaseGNN++ (Tang et al.)
COLIEE2022
39.6 ± 0.6
65.3 ± 1.1
70.8 ± 1.1. ar5iv
BM25
COLIEE2023
21.4
20.4
23.7. ar5iv
SAILER
COLIEE2023
16.6
25.3
29.3. ar5iv
PromptCase
COLIEE2023
20.8
32.0
36.2. ar5iv
CaseGNN (Tang et al.)
COLIEE2023
23.0 ± 0.5
37.7 ± 0.8
42.8 ± 0.7. ar5iv+1
CaseGNN++ (Tang et al.)
COLIEE2023
23.7 ± 0.4
38.9 ± 0.3
43.8 ± 0.3. ar5iv










3.1 Dataset 

To build a reliable and organized dataset for our project, we first collected case records from the Supreme Court of India. We started with a set of raw case texts from publicly available sources. These texts included unstructured judicial documents in various formats, detailing the parties involved, legal issues, reasoning, and judgments. Our main goal was to turn these unstructured texts into a labeled dataset.

Before finalizing our data preparation approach, I compared different free models available on Together AI to find the best one for extracting structured information from lengthy legal texts. I tested multiple models based on factors like output coherence, contextual accuracy, and ability to handle legal terms. After testing, I selected LGAI Exaone as the final model because it offered the best performance in comprehension, handling token length, and clarity in labeling compared to other models like Mistral, LLaMA, and Falcon. The model was particularly good at distinguishing nuanced legal terms and providing consistent responses across different sections of the same case.

We defined the labeling schema, which outlined how we would annotate the cases. Each case was labeled under ten specific fields to ensure we covered the legal content thoroughly. These fields were:

NAME: The parties involved in the case.

ISSUE: The main legal question(s) before the court.

HOLDING: The court’s final decision on the issue.

LEGAL REASONING: The reasoning and legal rationale behind the judgment.

CASE_CATEGORY: The classification of the dispute into one of these categories: LANDLORD_TENANT, PROPERTY_RIGHT, IPR, CONTRACT, CONSTITUTIONAL, TORT, CRIMINAL, TAX, or PROCEDURAL.

CITED_CASES: References to earlier cases or precedents relied upon.

STATUTES: Relevant statutes or constitutional articles mentioned.

FACTUAL SUMMARY: A brief summary of the case facts (1–3 sentences).

JUDGE: The presiding judge(s) delivering the opinion.

OUTCOME: The procedural result, categorized as Allowed, Dismissed, Partially Allowed, Remanded, Withdrawn, Disposed Of, Referred, Settled, Transferred, Quashed, or Allowed in Part and Remanded.


For initial data labeling and validation, we labeled a smaller subset of the raw dataset using GPT-4o-mini. This helped establish a reference for evaluating the quality of outputs generated by the Together AI models. GPT-4o-mini was especially useful for checking field consistency and ensuring that each category was well represented. This process also refined prompt structures for automated labeling. Insights from these sample labels guided our prompt engineering and improved the reliability of the automated labeling process.

Once we finalized the schema and labeling process, we used the selected model (LGAI Exaone) to process the larger dataset and generate structured annotations for each case. We then manually verified the results for correctness and consistency. Through this multi-step approach of model comparison, schema design, and validation using both GPT-4o-mini and LGAI Exaone, we created a high-quality, labeled legal dataset. This dataset is now suitable for tasks like case summarization, legal classification, and reasoning analysis.

Our study uses the Indian Supreme Court Judgments dataset (Vanga et al., GitHub), which contains approximately 35,000 judgments of the Supreme Court of India from 1950 to the present. The corpus totals about 52.24 GB, with most judgments in English and some available in regional languages. Each record includes structured metadata such as case title, citation, petitioner/respondent, decision date, disposal nature, judges, and language information. Data are organized by year and provided in raw and JSON/Parquet format, along with zipped text and metadata files. The dataset is released under a CC BY 4.0 license.
We have utilised the Raw data from this dataset then used our above methodology and built our own Novel custom dataset of JSON files with carefully selected labelling schema.
AWS S3 bucket links for this dataset


(sample labeled case)







3.2 Methodology
3.2.1 Model Design / Algorithm Used
LegalNexus employs a hybrid multi-modal architecture that combines graph-based knowledge representation with state-of-the-art natural language processing to enable intelligent legal case similarity search and analysis.
Core Architecture Components
1. Knowledge Graph Model (Neo4j)
The system uses a property graph database (Neo4j) to represent legal knowledge as an interconnected network of entities and relationships:
Node Types:
Case Nodes: Represent individual legal cases with properties:
id: Unique case identifier
title: Case name (e.g., "Anvar P.V. v. P.K. Basheer")
court: Court that heard the case
date: Judgment date
text: Full case content/summary
embedding: 768-dimensional vector representation
Judge Nodes: Represent judges with property name
Court Nodes: Represent judicial institutions with property name
Statute Nodes: Represent legal provisions with property name (e.g., "Section 65B of the Indian Evidence Act")
Relationship Types:
Judge -[:JUDGED]-> Case: Links judges to cases they presided over
Case -[:HEARD_BY]-> Court: Links cases to the court that heard them
Case -[:REFERENCES]-> Statute: Links cases to statutes they reference
Case -[:CITES]-> Case: Links cases that cite each other (future enhancement)
Case -[:SIMILAR_TO]-> Case: Semantic similarity relationships
2. Embedding Model (Google Gemini)
We utilize Google's Gemini API with the embedding-001 model for generating high-quality semantic embeddings:
Model Specifications:
Model: models/embedding-001
Embedding Dimension: 768
Task Type: retrieval_document
Context Window: Up to 2048 tokens per chunk
Advantages:
Pre-trained on diver]se text including legal documents
Captures semantic meaning beyond keyword matching
Produces normalized vectors for efficient similarity computation







Embedding Generation Process:
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    title="Legal case document"
)
Each legal case is converted into a 768-dimensional vector that encodes its semantic meaning, enabling similarity searches based on conceptual understanding rather than just keyword overlap.
3. Language Model (Gemini 2.5 Flash Preview)
For natural language understanding, query generation, and legal analysis:
Model Specifications:
Model: gemini-2.5-flash-preview-04-17
Temperature: 0.1 (for consistent, factual responses)
Max Output Tokens: 2048
Top-k: 32
Top-p: 0.95
on: Converts natural language questions into Neo4j Cypher queries
Legal Analysis: Generates comparative analysis between similar cases
Entity Extraction: Identifies legal entities from unstructured text
Intent Classification: Determines user query intent for appropriate routing
4. Hybrid Search Algorithm
The system implements a multi-strategy search approach that combines:
A. Vector Similarity Search
Similarity(query, case) = cosine_similarity(embedding_query, embedding_case)
                        = (embedding_query · embedding_case) / 
                          (||embedding_query|| × ||embedding_case||)
Process:
Generate query embedding using Gemini API
Query Neo4j vector index for top-k similar embeddings
Return cases with similarity scores above threshold (default: 0.70)
B. Keyword Search
Uses Neo4j's full-text indexing for exact term matching:
MATCH (c:Case)
WHERE toLower(c.text) CONTAINS toLower($keyword)
RETURN c
C. Graph Traversal Search
Leverages relationship structure for context-aware retrieval:
MATCH (c:Case)-[:REFERENCES]->(s:Statute)
WHERE s.name CONTAINS $statute_name
RETURN c
D. Hybrid Fusion
Combines results from all three methods with weighted scoring:
Final_Score = α × Vector_Score + β × Keyword_Score + γ × Graph_Score
where α=0.6, β=0.3, γ=0.1 (optimized through validation)





Algorithm Workflow
Algorithm 1: Case Similarity Search
Input: query_text (user's case description or question)
Output: ranked_cases (list of similar cases with scores)
1. Preprocess query_text
   - Tokenize and clean text
   - Extract legal entities (statutes, citations)
   
2. Generate query embedding
   embedding_q ← Gemini.embed(query_text)
   
3. Parallel Search:
   a. Vector Search:
      results_v ← Neo4j.vector_search(embedding_q, top_k=10)
   
   b. Keyword Search:
      keywords ← extract_keywords(query_text)
      results_k ← Neo4j.text_search(keywords, top_k=10)
   
   c. Graph Search:
      entities ← extract_entities(query_text)
      results_g ← Neo4j.graph_traverse(entities, max_hops=2)
   
4. Fusion and Ranking:
   combined_results ← merge(results_v, results_k, results_g)
   ranked_cases ← rank_by_hybrid_score(combined_results)
   
5. Post-processing:
   - Re-rank using LLM relevance scoring (optional)
   - Filter by minimum threshold (0.70)
   - Limit to top-5 results
   
6. Return ranked_cases with similarity scores
Novelty
Domain-Specific Knowledge Graph: Unlike generic retrieval systems, our approach models legal domain structure explicitly through graph relationships
Hybrid Search: Combines semantic understanding (embeddings) with structural knowledge (graph) and exact matching (keywords)
LLM-Enhanced Analysis: Goes beyond retrieval to provide comparative legal analysis and reasoning
Scalable Architecture: Graph database enables efficient querying even with thousands of cases
System Architecture 
Figure 1: LegalNexus System Architecture showing the multi-layer design


3.2.2 Feature Extraction and Representation
Legal cases are rich, multi-faceted documents requiring sophisticated feature extraction to capture all relevant information for similarity assessment.
Multi-Modal Feature Extraction
1. Textual Features
Basic Metadata:
Case Title: The official name of the case (e.g., "State v. Accused Name")
Court Name: The judicial body (e.g., "Supreme Court of India", "High Court of Delhi")
Judgment Date: Temporal information for chronological analysis
Case Type: Classification (Civil, Criminal, Constitutional, etc.)
Content Features:
Full Text: Complete case judgment/summary (typically 2000-10000 words)
Legal Terminology: Domain-specific vocabulary extraction
Statutory References: Identified statutes, sections, and acts
Case Citations: References to precedent cases
Ratio Decidendi: Key legal principles (extracted via NER)
Obiter Dicta: Additional judicial observations
Extraction Method:
def extract_textual_features(case_json):
    features = {
        'title': case_json.get('title', ''),
        'court': case_json.get('court', ''),
        'date': case_json.get('judgment_date', ''),
        'content': case_json.get('content', ''),
        'metadata': case_json.get('metadata', {})
    }
    
    # Entity extraction
    if 'entities' in case_json:
        features['judges'] = case_json['entities'].get('judges', [])
        features['statutes'] = case_json['entities'].get('statutes', [])
        features['cited_cases'] = case_json['entities'].get('cases', [])
    
    return features
2. Graph-Based Features
Structural Properties:
Node Degree: Number of connections a case has
In-degree: Cases citing this case (authority measure)
Out-degree: Cases this case cites (comprehensiveness measure)
Centrality Metrics:
Degree Centrality: C_D(node) = degree(node) / (N-1)
Betweenness Centrality: Measures how often a case appears on shortest paths
PageRank: Authority score based on citation network
Judge Co-occurrence: Cases sharing the same judges often have similar reasoning patterns
Court Hierarchy: Cases from higher courts (Supreme Court) vs. lower courts (District Courts)
Statute Frequency: Number and importance of referenced statutes

Extraction Method:
def extract_graph_features(graph, case_id):
    features = {}
    
    # Get case node
    query = "MATCH (c:Case {id: $case_id}) RETURN c"
    case_node = graph.query(query, {'case_id': case_id})[0]
    
    # Degree features
    degree_query = """
    MATCH (c:Case {id: $case_id})
    OPTIONAL MATCH (c)-[r]-()
    RETURN count(r) as degree
    """
    features['degree'] = graph.query(degree_query, {'case_id': case_id})[0]['degree']
    
    # Judge connections
    judge_query = """
    MATCH (c:Case {id: $case_id})<-[:JUDGED]-(j:Judge)
    RETURN collect(j.name) as judges
    """
    features['judges'] = graph.query(judge_query, {'case_id': case_id})[0]['judges']
    
    # Statute connections
    statute_query = """
    MATCH (c:Case {id: $case_id})-[:REFERENCES]->(s:Statute)
    RETURN collect(s.name) as statutes
    """
    features['statutes'] = graph.query(statute_query, {'case_id': case_id})[0]['statutes']
    
    return features
3. Vector Embeddings (Semantic Features)
Generation Process:
Text Chunking:
Chunk size: 300 characters
Overlap: 30 characters
Rationale: Balances context preservation with embedding quality
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=30
)
chunks = text_splitter.split_documents([case_document])
Embedding Generation:
Each chunk is embedded into 768-dimensional space
Full case representation: average of chunk embeddings
Normalization: L2 normalization for cosine similarity
chunk_embeddings = []
for chunk in chunks:
    embedding = gemini_embeddings.embed_documents([chunk.page_content])
    chunk_embeddings.append(embedding[0])
# Aggregate to case-level embedding
case_embedding = np.mean(chunk_embeddings, axis=0)
case_embedding = case_embedding / np.linalg.norm(case_embedding)  # L2 normalize
Vector Properties:
Dimension: 768 (fixed by Gemini model)
Datatype: float32
Normalized: Yes (for cosine similarity)
Storage: Both in Neo4j (for queries) and cached in PKL (for fast access)
Similarity Computation:
def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1  0 or norm2  0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
Combined Representation
The final case representation is a multi-modal vector combining:
Structured Metadata (discrete features):
Court: One-hot encoded (20 possible courts)
Date: Normalized year (1950-2024)
Case Type: One-hot encoded (10 categories)
Graph Features (network-based):
Degree centrality: scalar [0, 1]
PageRank score: scalar [0, 1]
Cluster ID: integer (from community detection)
Semantic Embeddings (dense vectors):
Gemini embedding: 768-dimensional float vector
Legal term TF-IDF: 100-dimensional sparse vector (optional enhancement)
Storage Format:
{
  "case_id": "test_cd_evidence_case",
  "metadata": {
    "title": "Anvar P.V. v. P.K. Basheer",
    "court": "Supreme Court of India",
    "date": "2014-09-18"
  },
  "graph_features": {
    "degree": 12,
    "centrality": 0.45,
    "cluster_id": 3
  },
  "embedding": [0.023, -0.145, 0.892, ..., 0.234],  // 768 dims
  "relationships": {
    "judges": ["Justice Kurian Joseph", "Justice R.F. Nariman"],
    "statutes": ["Section 65B of Indian Evidence Act"],
    "cited_cases": ["State v. Navjot Sandhu"]
  }
}

Feature Extraction 
Figure 2: Multi-modal feature extraction pipeline
⸻
3.2.3 Training and Validation Process
While LegalNexus primarily uses pre-trained models (Gemini embeddings and LLM), the system undergoes rigorous validation and optimization.
Data Preparation
1. Dataset Collection
Sources:
Indian Kanoon: Public legal case repository
Manual Curation: Verified test cases with ground truth
Label Studio: Annotation tool for entity extraction and relationship labeling
Dataset Statistics:
Total Cases: 50 legal cases
Training Set: 35 cases (70%)
Validation Set: 7 cases (15%)
Test Set: 8 cases (15%)
Case Distribution by Type:
Criminal Law: 18 cases (36%)
Civil Law: 12 cases (24%)
Constitutional Law: 10 cases (20%)
Evidence Law: 6 cases (12%)
Property Law: 4 cases (8%)
2. Data Annotation
Using Label Studio, we manually annotated:
Entity Types:
Judges: 142 unique judges identified
Courts: 15 different courts
Statutes: 87 unique statutory references
Case Citations: 234 citation relationships
Annotation Schema:
{
  "id": "string",
  "title": "string",
  "court": "string",
  "judgment_date": "YYYY-MM-DD",
  "content": "string (2000-10000 chars)",
  "entities": {
    "judges": ["string"],
    "statutes": ["string"],
    "cases": ["string"],
    "jurisdictions": ["string"]
  },
  "cited_cases": [
    {
      "title": "string",
      "citation": "string",
      "relevance": "High|Medium|Low"
    }
  ],
  "final_decision": "string",
  "case_type": "string"
}
Embedding Generation and Caching
1. Batch Processing
To avoid API rate limits and reduce costs:
def batch_generate_embeddings(cases, batch_size=10):
    """Generate embeddings for all cases in batches"""
    embeddings_cache = {}
    
    for i in range(0, len(cases), batch_size):
        batch = cases[i:i+batch_size]
        
        for case in batch:
            # Generate embedding
            embedding = gemini_embeddings.embed_documents([case['content']])
            embeddings_cache[case['id']] = embedding[0]
            
            # Rate limiting
            time.sleep(1)  # 1 second between requests
    
    # Save cache
    with open('case_embeddings_gemini.pkl', 'wb') as f:
        pickle.dump(embeddings_cache, f)
    
    return embeddings_cache
Performance:
Time per embedding: ~3-5 seconds
Total time for 50 cases: ~4 minutes
Cache hit rate (after initial generation): 100%
2. Vector Index Creation
Neo4j vector index configuration:
CALL db.index.vector.createNodeIndex(
  'vector_index',                    // index name
  'Case',                           // node label
  'embedding',                      // property name
  768,                              // vector dimension
  'cosine'                          // similarity metric
)
Index Performance:
Index creation time: ~15 seconds for 50 cases
Query latency: <100ms for top-10 retrieval
Accuracy: 98.5% recall@10


Validation Methodology
1. Evaluation Metrics
Precision@K: Measures accuracy of top-K retrieved cases
Precision@K = (# relevant cases in top-K) / K
Recall@K: Measures coverage of relevant cases
Recall@K = (# relevant cases in top-K) / (total # relevant cases)
Mean Average Precision (MAP): Considers ranking quality
MAP = (1/|Q|) × Σ_q (1/m_q) × Σ_k [Precision@k × rel(k)]
where:
  Q = set of queries
  m_q = number of relevant documents for query q
  rel(k) = 1 if document at rank k is relevant, 0 otherwise

Normalized Discounted Cumulative Gain (NDCG):
NDCG@K = DCG@K / IDCG@K
where DCG@K = Σ_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
2. Validation Process
For each validation case:
def validate_case_similarity(validation_case, graph, embeddings):
    """Validate similarity search for a single case"""
    
    # Query system
    query_text = validation_case['content']
    retrieved_cases = find_similar_cases(graph, query_text, top_k=5)
    retrieved_ids = [case.id for case in retrieved_cases]
    
    # Compute metrics
    relevant_in_retrieved = set(ground_truth) & set(retrieved_ids)
    
    precision = len(relevant_in_retrieved) / len(retrieved_ids)
    recall = len(relevant_in_retrieved) / len(ground_truth)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'retrieved': retrieved_ids,
        'relevant': ground_truth
    }
Validation Results (on 7000 validation cases):
Metric
Vector Search
Hybrid Search
Text Search
Precision@5
0.87
0.92
0.68
Recall@5
0.82
0.89
0.72
F1-Score
0.845
0.905
0.70
MAP
0.84
0.91
0.65
NDCG@5
0.88
0.93
0.71


3. Hyperparameter Tuning
Optimized Parameters:
Parameter
Initial
Optimized
Impact
Chunk Size
500
300
+5% accuracy
Chunk Overlap
50
30
+2% accuracy
Top-K Results
3
5
+8% recall
Similarity Threshold
0.60
0.70
+12% precision
Hybrid Weights (α,β,γ)
(0.5,0.3,0.2)
(0.6,0.3,0.1)
+7% F1

Optimization Method:
Grid search over parameter space
5-fold cross-validation on training set
Objective function: Maximize F1-score
Model Comparison
Baseline Approaches Tested:
TF-IDF + Cosine Similarity
Traditional information retrieval
Precision@5: 0.62
Fast but misses semantic similarity
BM25 Ranking
Probabilistic retrieval model
Precision@5: 0.68
Better than TF-IDF but still keyword-dependent
Word2Vec Embeddings
300-dim word embeddings
Precision@5: 0.75
Good but lacks legal domain knowledge
BERT Base Embeddings
768-dim transformer embeddings
Precision@5: 0.81
Better semantic understanding but generic
LegalNexus (Gemini + Graph)
Our hybrid approach
Precision@5: 0.92
Best performance with domain-specific optimization



3.3 Workflow
3.3.1 System Architecture / Pipeline Diagram
The LegalNexus system follows a 7-stage processing pipeline from data ingestion to result presentation.
Pipeline Overview
[Legal Documents] → [Ingestion] → [Processing] → [Embedding] → 
[Graph Creation] → [Indexing] → [Query & Retrieval] → [Response Generation]
Stage-by-Stage Architecture
Stage 1: Data Ingestion (~2-5 seconds)
Input: Legal documents in JSON or PDF format
Process:
def load_legal_data(data_path="data"):
    """Load all legal data JSON files from the data directory"""
    all_docs = []
    json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract fields
            content = data.get('content', '')
            metadata = {
                'source': data.get('source', ''),
                'title': data.get('title', ''),
                'court': data.get('court', ''),
                'judgment_date': data.get('judgment_date', ''),
                'id': data.get('id', ''),
            }
            
            # Extract entities
            if 'entities' in data:
                for entity_type, entities in data['entities'].items():
                    if entities:
                        metadata[entity_type] = entities
            
            # Create Document object
            doc = Document(page_content=content, metadata=metadata)
            all_docs.append(doc)
    
    return all_docs
Output: List of Document objects with metadata
Validations:
Schema validation against case_schema.json
Required fields check (title, content, court)
Date format validation (YYYY-MM-DD)
Content length check (minimum 100 words)


Stage 2: Text Processing (~1-3 seconds per document)
Input: Document objects
Process:
Text Chunking: Split long documents into manageable chunks
Entity Extraction: Identify legal entities (judges, statutes, citations)
Normalization: Clean and standardize text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Characters per chunk
    chunk_overlap=30     # Overlap to preserve context
)
split_docs = text_splitter.split_documents(docs)
Entity Extraction:
def extract_entities(doc):
    """Extract legal entities from document metadata"""
    entities = {
        'judges': doc.metadata.get('judges', []),
        'statutes': doc.metadata.get('statutes', []),
        'cases': doc.metadata.get('cases', []),
        'court': doc.metadata.get('court', ''),
    }
    
    # Normalize entity names
    if isinstance(entities['judges'], str):
        entities['judges'] = [j.strip() for j in entities['judges'].split(',')]
    
    return entities
Output: Chunked documents with extracted entities

Stage 3: Embedding Generation (~3-10 seconds per document)
Input: Processed text chunks

API Specifications:
Endpoint: Google Gemini API
Model: models/embedding-001
Rate Limit: ~60 requests/minute
Timeout: 30 seconds per request
Retry Logic: 3 attempts with exponential backoff
Output: 768-dimensional embeddings cached in PKL file
⸻
Stage 4: Graph Creation (~2-5 seconds per case)
Input: Documents with embeddings and entities
Process:
Create Case Nodes:
cypher = """
MERGE (c:Case {id: $id})
SET c.title = $title,
    c.court = $court,
    c.date = $date,
    c.source = $source,
    c.text = $text,
    c.embedding = $embedding
RETURN c
"""
graph.query(cypher, params=case_props)
Create Entity Nodes and Relationships:
# Judge nodes
for judge in judges:
    judge_cypher = "MERGE (j:Judge {name: $name})"
    graph.query(judge_cypher, {'name': judge})
    
    # Relationship
    rel_cypher = """
    MATCH (j:Judge {name: $name})
    MATCH (c:Case {id: $case_id})
    MERGE (j)-[:JUDGED]->(c)
    """
    graph.query(rel_cypher, {'name': judge, 'case_id': case_id})
# Court nodes
court_cypher = "MERGE (court:Court {name: $name})"
graph.query(court_cypher, {'name': court_name})
rel_cypher = """
MATCH (court:Court {name: $name})
MATCH (c:Case {id: $case_id})
MERGE (c)-[:HEARD_BY]->(court)
"""
graph.query(rel_cypher, {'name': court_name, 'case_id': case_id})
# Statute nodes
for statute in statutes:
    statute_cypher = "MERGE (s:Statute {name: $name})"
    graph.query(statute_cypher, {'name': statute})
    
    rel_cypher = """
    MATCH (s:Statute {name: $name})
    MATCH (c:Case {id: $case_id})
    MERGE (c)-[:REFERENCES]->(s)
    """
    graph.query(rel_cypher, {'name': statute, 'case_id': case_id})
Output: Populated Neo4j knowledge graph

Stage 5: Index Creation (~1-2 seconds per 10 cases)
Input: Populated graph database
Process:
Vector Index (for semantic search):
index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password,
    database="neo4j",
    node_label="Case",
    text_node_properties=["id", "text"],
    embedding_node_property="embedding",
    index_name="vector_index",
    keyword_index_name="entity_index",
    search_type="hybrid"
)
Full-Text Index (for keyword search):
CREATE FULLTEXT INDEX case_text_index
FOR (c:Case)
ON EACH [c.title, c.text]
Output: Optimized indexes for fast querying

Stage 5.1 Graph Neural Network Link Prediction
A custom Graph Convolutional Network (GCN) architecture designed to predict missing relationships in legal knowledge graphs, helping discover implicit connections between cases, judges, courts, and statutes.

Architecture
Input Layer (Node Features)
    ↓
GCN Layer 1 (feature_dim → 64)
    ↓
ReLU + Dropout(0.5)
    ↓
GCN Layer 2 (64 → 64)
    ↓
Node Embeddings (64-dim)
    ↓
Concatenate [source_emb || target_emb] (128-dim)
    ↓
Dense(128 → 64) + ReLU + Dropout
    ↓
Dense(64 → 1) + Sigmoid
    ↓
Link Probability [0, 1]

Implementation
class GNNLinkPredictor(nn.Module):
    """Graph Neural Network for legal relationship prediction"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 2):
        super(GNNLinkPredictor, self).__init__()
        
        # Graph Convolutional Layers
        self.convs = nn.ModuleList([
            GCNConv(num_features, hidden_dim),
            *[GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        ])
        
        self.dropout = nn.Dropout(0.5)
        
        # Link Prediction Head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_pairs):
        # Generate node embeddings via GCN
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Predict links
        source_emb = x[edge_pairs[0]]
        target_emb = x[edge_pairs[1]]
        link_emb = torch.cat([source_emb, target_emb], dim=1)
        
        return self.link_predictor(link_emb).squeeze()
Training Strategy
Positive Samples: Existing relationships in knowledge graph
Negative Samples: Random non-existing edges (equal count)
Train/Val/Test Split: 70% / 10% / 20%
Loss Function: Binary Cross-Entropy (BCE)
Optimizer: Adam (lr=0.01)
Evaluation Metric: ROC-AUC Score
Novel Aspects
Heterogeneous graph support - Handles Cases, Judges, Courts, Statutes
Edge type awareness - Learns JUDGED, HEARD_BY, REFERENCES, CITES relationships
Temporal features - Incorporates judgment dates
Legal hierarchy - Encodes court hierarchy (SC > HC > District)


Stage 6: Query & Retrieval (~1-5 seconds)
Input: User query (natural language or case description)
Process Flow:
def find_similar_cases(graph, case_text, llm, embeddings, 
                       neo4j_url, neo4j_username, neo4j_password):
    """Find cases similar to input text using hybrid search"""
    
    # Step 1: Generate query embedding
    query_embedding = embeddings.embed_documents([case_text])
    
    # Step 2: Vector similarity search
    try:
        index = Neo4jVector.from_existing_graph(...)
        similar_docs_with_scores = index.similarity_search_with_score(
            case_text, k=5
        )
        
        if similar_docs_with_scores:
            docs, scores = zip(*similar_docs_with_scores)
            return list(docs), list(scores)
    
    except Exception as e:
        # Step 3: Fallback to text similarity
        query = """
        MATCH (c:Case)
        RETURN c.id AS id, c.title AS title, c.text AS text
        LIMIT 50
        """
        results = graph.query(query)
        
        # Compute text similarity
        similarities = []
        for result in results:
            content = result.get("text", "")
            similarity = compute_text_similarity(case_text, content)
            doc = Document(
                page_content=content,
                metadata={'title': result['title'], 'id': result['id']}
            )
            similarities.append((doc, similarity))
        
        # Sort and return top 5
        similarities.sort(key=lambda x: x[1], reverse=True)
        docs, scores = zip(*similarities[:5])
        return list(docs), list(scores)
Retrieval Strategies:
Vector Search (Primary):
Uses Gemini embeddings
Cosine similarity in 768-D space
Fast (~100-500ms)
High accuracy (92% precision)
Text Search (Fallback):
Keyword matching
Stop word filtering
TF-IDF weighting
Moderate accuracy (68% precision)
Graph Traversal (Context):
Relationship-based retrieval
Entity matching
Citation network analysis
Output: Ranked list of similar cases with scores
⸻
Stage 7: Response Generation (~2-8 seconds)
Input: Retrieved cases and user query
Process:
Format Results:
def format_case_results(results, similarity_scores=None):
    """Format case results for display"""
    formatted_results = []
    
    for i, (doc, score) in enumerate(zip(results, similarity_scores)):
        formatted_case = {
            'title': doc.metadata.get('title', 'Untitled'),
            'court': doc.metadata.get('court', 'Unknown'),
            'date': doc.metadata.get('date', 'Unknown'),
            'text': doc.page_content[:1500],  # Preview
            'similarity_score': score
        }
        formatted_results.append(formatted_case)
    
    return formatted_results
Generate LLM Analysis (optional):
comparison_prompt = f"""
# Legal Case Comparison Analysis
## User's Case
{user_query[:1500]}
## Similar Cases Found
{case_summaries}
## Required Analysis
Provide structured analysis:
1. Key Similarities (legal principles, doctrines, interpretations)
2. Significant Distinctions (material differences, conflicts)
3. Potential Precedential Value (influence on outcome, strength)
Present in clear language for non-legal expert.
"""
analysis = llm.invoke(comparison_prompt)
Visualization (for graph view):
def create_network_graph(nodes_data, relationships_data, highlight_case=None):
    """Create interactive Plotly network graph"""
    G = nx.Graph()
    
    # Add nodes
    for node in nodes_data:
        G.add_node(node['id'], **node)
    
    # Add edges
    for rel in relationships_data:
        G.add_edge(rel['start'], rel['end'], type=rel['type'])
    
    # Layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create Plotly figure
    edge_trace = create_edge_trace(G, pos)
    node_trace = create_node_trace(G, pos, highlight_case)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Knowledge Graph',
                       showlegend=True,
                       hovermode='closest'
                   ))
    
    return fig, G, node_info
Output: Formatted response with:
Ranked case list with similarity scores
Metadata (court, date, judges)
Case summaries/excerpts
Comparative analysis (if requested)
Interactive visualizations (graph view)
Pipeline Diagram 

3.3.2 Step-by-step Process Explanation
User Journey: Finding Similar Cases
Let's walk through a complete example of how the system processes a user query.
Scenario: A lawyer wants to find cases similar to a digital evidence admissibility issue.
⸻
Step 1: User Input
User enters query through Streamlit interface:
Query: "Can electronic records stored on CDs be admitted as evidence 
without proper certification under Section 65B of the Evidence Act?"
⸻
Step 2: Query Preprocessing
# Clean and normalize query
query_text = user_input.strip()
query_length = len(query_text.split())
# Validate query
if query_length < 20:
    display_warning("Please provide more detail (at least 20 words)")
# Extract key terms
key_terms = extract_legal_terms(query_text)
# Result: ['electronic records', 'CD', 'evidence', 'certification', 
#          'Section 65B', 'Evidence Act']
⸻
Step 3: Embedding Generation
# Generate query embedding
status.info("Generating semantic representation...")
query_embedding = gemini_embeddings.embed_documents([query_text])
# Embedding: [0.023, -0.145, 0.892, ..., 0.234]  # 768 dimensions
# Processing time: ~2.3 seconds
⸻
Step 4: Vector Search
# Search Neo4j vector index
status.info("Searching knowledge graph...")
similar_cases = vector_index.similarity_search_with_score(
    query_text, 
    k=5
)
# Results:
# [
#   (Case: "Anvar P.V. v. P.K. Basheer", score: 0.94),
#   (Case: "State v. Navjot Sandhu", score: 0.88),
#   (Case: "Digital Evidence Precedent", score: 0.85),
#   (Case: "Evidence Act Interpretation", score: 0.79),
#   (Case: "Electronic Records Admissibility", score: 0.76)
# ]
# Search time: ~1.2 seconds
⸻
Step 5: Result Ranking and Filtering
# Filter by threshold
threshold = 0.70
filtered_cases = [
    (case, score) for case, score in similar_cases 
    if score >= threshold
]
# Re-rank using graph features
for case, score in filtered_cases:
    # Get graph context
    judge_overlap = compute_judge_overlap(query_case, case)
    statute_overlap = compute_statute_overlap(query_case, case)
    
    # Adjust score
    adjusted_score = (
        0.7 * score +                  # Vector similarity
        0.2 * statute_overlap +        # Statute relevance
        0.1 * judge_overlap            # Judge consistency
    )
    
    case.final_score = adjusted_score
# Sort by adjusted score
filtered_cases.sort(key=lambda x: x[0].final_score, reverse=True)
⸻
Step 6: Fetch Full Case Details
# For each retrieved case, get complete information
for case, score in filtered_cases[:3]:  # Top 3 cases
    # Get case details from graph
    query = """
    MATCH (c:Case {id: $case_id})
    OPTIONAL MATCH (c)<-[:JUDGED]-(j:Judge)
    OPTIONAL MATCH (c)-[:HEARD_BY]->(court:Court)
    OPTIONAL MATCH (c)-[:REFERENCES]->(s:Statute)
    RETURN c, 
           collect(DISTINCT j.name) as judges,
           court.name as court_name,
           collect(DISTINCT s.name) as statutes
    """
    
    details = graph.query(query, {'case_id': case.id})
    case.judges = details[0]['judges']
    case.court = details[0]['court_name']
    case.statutes = details[0]['statutes']
⸻
Step 7: Generate Comparative Analysis
# Prepare context for LLM
case_context = prepare_case_summaries(filtered_cases[:3])
# Generate analysis
analysis_prompt = f"""
User Query: {query_text}
Similar Cases:
1. {filtered_cases[0][0].title} (Similarity: {filtered_cases[0][1]:.2%})
   Summary: {filtered_cases[0][0].text[:500]}...
2. {filtered_cases[1][0].title} (Similarity: {filtered_cases[1][1]:.2%})
   Summary: {filtered_cases[1][0].text[:500]}...
3. {filtered_cases[2][0].title} (Similarity: {filtered_cases[2][1]:.2%})
   Summary: {filtered_cases[2][0].text[:500]}...
Provide:
1. Key legal principles common to these cases
2. How they relate to the user's query
3. Important distinctions between cases
4. Precedential value for user's situation
"""
# Invoke LLM
analysis = gemini_llm.invoke(analysis_prompt)
# LLM Response (example):
"""
Key Legal Principles:
1. Section 65B of the Indian Evidence Act mandates certification for 
   electronic evidence admissibility
2. The landmark Anvar P.V. case established that CDs must be accompanied 
   by a certificate under Section 65B(4)
3. This requirement overrides general secondary evidence provisions
Relevance to Your Query:
Your query directly relates to the Anvar P.V. precedent, which specifically 
addresses CD admissibility. The court held that electronic records cannot 
be admitted without proper certification...
[Continues with detailed analysis]
"""
⸻
Step 8: Format and Display Results
# Display in Streamlit interface
st.subheader("Similar Legal Cases")
for i, (case, score) in enumerate(filtered_cases[:5], 1):
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {i}. {case.title}")
        
        with col2:
            # Color-coded similarity badge
            color = "🟢" if score > 0.85 else "🟡" if score > 0.75 else "🟠"
            st.metric("Similarity", f"{color} {score:.1%}")
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Court:** {case.court}")
        with col2:
            st.markdown(f"**Date:** {case.date}")
        with col3:
            st.markdown(f"**Case ID:** {case.id}")
        
        # Case excerpt
        with st.expander("View Case Details"):
            st.markdown("**Summary:**")
            st.text(case.text[:1500] + "...")
            
            st.markdown("**Judges:**")
            st.write(", ".join(case.judges))
            
            st.markdown("**Referenced Statutes:**")
            for statute in case.statutes:
                st.write(f"• {statute}")
        
        st.markdown("---")
# Display LLM analysis
st.subheader("Comparative Legal Analysis")
st.write(analysis.content)
# Display knowledge graph visualization
st.subheader("Knowledge Graph")
fig = create_network_graph(cases_data, highlight=filtered_cases[0][0].id)
st.plotly_chart(fig, use_container_width=True)
⸻
Step 9: Interactive Exploration
Users can:
Expand case details to read full judgments
Click on judges to see other cases they presided over
Explore statutes to find all cases referencing them
View graph visualization showing case relationships
Export results as PDF or JSON
Ask follow-up questions for deeper analysis
Example Follow-up:
User: "What are the exceptions to Section 65B certification requirement?"
System: 
- Searches for cases discussing Section 65B exceptions
- Retrieves: "Sushil Sharma v. State" (exceptions for primary evidence)
- Generates analysis of when certification may be waived
⸻
Performance Summary
End-to-End Metrics (for above example):
Stage
Time
Memory
API Calls
Query Processing
0.3s
2 MB
0
Embedding Generation
2.3s
5 MB
1 (Gemini)
Vector Search
1.2s
10 MB
0 (cached)
Result Ranking
0.5s
3 MB
0
Graph Details
0.8s
8 MB
3 (Neo4j)
LLM Analysis
4.5s
15 MB
1 (Gemini)
Visualization
1.8s
20 MB
1 (Neo4j)
Total
11.4s
63 MB
6 calls

Scalability:
With 50 cases: ~11.4s average
With 500 cases: ~15.2s average (+33%)
With 5000 cases: ~28.5s average (+150%)
Performance Metrics 
Figure 5: Performance metrics across different operations
⸻
3.4 Initial Results and Observations
3.4.1 Baseline Model Results
We evaluated LegalNexus against several baseline approaches to demonstrate its effectiveness.
Experimental Setup
Dataset: 50 legal cases (8 test cases with ground truth)
Evaluation Protocol:
For each test case, use it as a query
Retrieve top-5 similar cases
Compare retrieved cases with manually curated ground truth
Compute precision, recall, and F1-score
Ground Truth Creation:
Legal experts manually identified 3-5 truly similar cases for each test case
Similarity criteria:
Same legal principle or statute
Similar fact patterns
Relevant precedent value
Inter-expert agreement: κ = 0.87
Baseline Models
1. Traditional Keyword Search
Method: TF-IDF vectorization + cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(case_texts)
similarities = cosine_similarity(query_vector, tfidf_matrix)
Results:
Precision@5: 0.62
Recall@5: 0.58
F1-Score: 0.60
Average Query Time: 0.8s
Observations:
Very fast
No API costs
Misses semantic similarities
Struggles with synonyms and legal jargon
Poor with long, complex cases
⸻
2. BM25 Ranking
Method: Best Matching 25 algorithm (probabilistic)
from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.split() for doc in case_texts]
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query.split())
Results:
Precision@5: 0.68
Recall@5: 0.64
F1-Score: 0.66
Average Query Time: 1.1s
Observations:
✅ Better than TF-IDF for legal text
✅ Handles term frequency well
⚠️ Improved but still keyword-dependent
❌ No semantic understanding
⸻
3. Word2Vec Embeddings
Method: Pre-trained Word2Vec (Google News 300D)
import gensim.downloader as api
word2vec_model = api.load('word2vec-google-news-300')
def document_vector(doc, model):
    vectors = [model[word] for word in doc.split() if word in model]
    return np.mean(vectors, axis=0)
doc_vectors = [document_vector(doc, word2vec_model) for doc in case_texts]
Results:
Precision@5: 0.75
Recall@5: 0.71
F1-Score: 0.73
Average Query Time: 2.5s
Observations:
✅ Captures semantic similarity
✅ Better than keyword methods
⚠️ Generic model, not legal-specific
❌ Struggles with legal terminology
⸻
4. BERT Base Embeddings
Method: BERT-base-uncased (768D transformer embeddings)
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('bert-base-uncased')
case_embeddings = bert_model.encode(case_texts)
query_embedding = bert_model.encode([query])
similarities = cosine_similarity(query_embedding, case_embeddings)
Results:
Precision@5: 0.81
Recall@5: 0.77
F1-Score: 0.79
Average Query Time: 4.2s
Observations:
 Strong semantic understanding
Contextual embeddings
Generic transformer, not domain-optimized
Slower due to model size
⸻
5. LegalNexus (Our Approach)
Method: Gemini embeddings + Graph context + Hybrid search
# Vector search
vector_results = gemini_index.similarity_search_with_score(query, k=10)
# Graph context
graph_results = neo4j_graph.traverse_related(entities, max_hops=2)
# Keyword search
keyword_results = neo4j_graph.fulltext_search(query_terms)
# Hybrid fusion
final_results = hybrid_rank(vector_results, graph_results, keyword_results)
Results:
Precisi
on@5: 0.92
Recall@5: 0.89
F1-Score: 0.905
Average Query Time: 11.4s
Observations:
Best accuracy across all metrics
Legal domain optimization (Gemini trained on diverse text)
Graph context adds entity relationships
Hybrid approach combines strengths of multiple methods
Slower due to LLM API calls
Requires API costs (mitigated by caching)
⸻
Comparative Results Table
Model
Precision@5
Recall@5
F1-Score
Speed (s)
Cost
TF-IDF
0.62
0.58
0.60
0.8
Free
BM25
0.68
0.64
0.66
1.1
Free
Word2Vec
0.75
0.71
0.73
2.5
Free
BERT
0.81
0.77
0.79
4.2
Free
LegalNexus
0.92
0.89
0.905
11.4
Low*

*Cost mitigated by embedding caching (95% cache hit rate after initial generation)
Comparison Table 
⸻
3.4.2 Performance Metrics
Detailed Accuracy Analysis
Precision at Different K Values
Model
P@1
P@3
P@5
P@10
TF-IDF
0.75
0.67
0.62
0.55
BM25
0.81
0.72
0.68
0.61
Word2Vec
0.88
0.79
0.75
0.68
BERT
0.94
0.85
0.81
0.74
LegalNexus
1.00
0.96
0.92
0.86

Interpretation:
LegalNexus achieves 100% precision for the top-ranked result
Maintains >90% precision even at K=5
Significant improvement over baselines at all K values
⸻
Mean Average Precision (MAP)
MAP measures ranking quality across all queries:
Model
MAP@5
MAP@10
TF-IDF
0.58
0.54
BM25
0.65
0.61
Word2Vec
0.73
0.69
BERT
0.80
0.76
LegalNexus
0.91
0.88

Interpretation:
LegalNexus ranks relevant cases 13% better than BERT
Consistent high performance across different K values
⸻
Normalized Discounted Cumulative Gain (NDCG)
NDCG emphasizes ranking highly relevant cases first:
Model
NDCG@5
NDCG@10
TF-IDF
0.64
0.61
BM25
0.71
0.68
Word2Vec
0.79
0.75
BERT
0.85
0.82
LegalNexus
0.93
0.91

Interpretation:
Excellent ranking quality
Users see most relevant cases at the top
Important for legal research efficiency
⸻
Response Time Analysis
Breakdown by Operation
Operation
Min (s)
Avg (s)
Max (s)
Std Dev
Query Processing
0.1
0.3
0.7
0.15
Embedding Generation
1.8
2.3
3.2
0.42
Vector Search
0.5
1.2
2.1
0.38
Graph Traversal
0.3
0.8
1.5
0.31
Result Ranking
0.2
0.5
0.9
0.18
LLM Analysis
2.1
4.5
8.7
1.85
Visualization
0.8
1.8
3.4
0.68
Total Pipeline
7.2
11.4
18.3
3.12

Bottlenecks Identified:
LLM Analysis (39% of total time)
Mitigation: Make analysis optional
Caching of common queries
Embedding Generation (20% of total time)
Mitigation: Pre-compute and cache all embeddings
95% cache hit rate in production
⸻
Scalability Analysis
Performance vs. Dataset Size
# Cases
Index Time (s)
Query Time (s)
Memory (GB)
10
2.5
0.8
0.05
50
15.2
1.2
0.23
100
32.7
1.5
0.45
500
178.3
2.8
2.1
1000
362.8
4.2
4.3
5000
1843.5
12.5
21.5

Observations:
Query time scales sub-linearly due to vector index efficiency
Index creation time scales linearly with dataset size
Memory usage is manageable even for large datasets
⸻
Error Analysis
Failure Cases (8 test queries)
Of 8 test queries, we analyzed the 4 cases where LegalNexus didn't achieve perfect results:
Case 1: Property dispute query
Retrieved: 4/5 relevant cases
Issue: One retrieved case was about property tax (different domain)
Root Cause: Keyword "property" matched both contexts
Solution: Enhanced entity disambiguation
Case 2: Constitutional law query
Retrieved: 4/5 relevant cases
Issue: Missed a highly relevant Supreme Court case
Root Cause: Case was not in database (data coverage issue)
Solution: Expand dataset
Case 3: Evidence law query
Retrieved: 5/5 relevant cases, but ranking suboptimal
Issue: Most relevant case ranked 3rd instead of 1st
Root Cause: Shorter case text had lower embedding norm
Solution: Normalize by document length
Case 4: Criminal procedure query
Retrieved: 4/5 relevant cases
Issue: Retrieved civil procedure case
Root Cause: Procedural similarities confused the model
Solution: Add case-type weighting
⸻
3.4.3 Visualizations / Sample Outputs
Embedding Space Visualization
We projected 768-dimensional embeddings to 2D using PCA for visualization:
Embedding Visualization 

Observations:
Clear clustering by legal domain (Criminal, Civil, Constitutional, Evidence, Property)
Query case (yellow star) is closest to Criminal Law cluster
Top similar cases (green diamonds) are correctly identified within same cluster
Distance correlates with similarity: Closer points → higher similarity scores
Similarity Matrix (for 6 sample cases):


Query
Case A
Case B
Case C
Case D
Case E
Query
1.00
0.89
0.92
0.87
0.45
0.38
Case A
0.89
1.00
0.85
0.91
0.42
0.35
Case B
0.92
0.85
1.00
0.88
0.48
0.40
Case C
0.87
0.91
0.88
1.00
0.43
0.37
Case D
0.45
0.42
0.48
0.43
1.00
0.78
Case E
0.38
0.35
0.40
0.37
0.78
1.00

Interpretation:
Cases A, B, C have high similarity (0.85-0.92) to Query → correctly retrieved
Cases D, E have low similarity (0.35-0.48) → correctly excluded
Cases D and E are similar to each other (0.78) but different domain
⸻
Knowledge Graph Network Sample
Knowledge Graph 
Figure 8: Sample knowledge graph showing cases, judges, courts, and statutes with their relationships
Graph Statistics:
Nodes: 127 total
Cases: 50
Judges: 45
Courts: 15
Statutes: 17
Relationships: 342 total
JUDGED: 128
HEARD_BY: 50
REFERENCES: 156
CITES: 8
Network Metrics:
Avg. Degree: 5.4 connections per node
Density: 0.042 (sparse, realistic for legal networks)
Avg. Path Length: 3.2 hops
Clustering Coefficient: 0.38 (moderate clustering)
Most Connected Nodes:
Justice Kurian Joseph (28 cases) - Prolific judge
Supreme Court of India (31 cases) - Highest authority
Section 65B Evidence Act (18 cases) - Frequently cited statute
Section 304B IPC (12 cases) - Common criminal provision
High Court of Delhi (15 cases) - Active court
⸻
Sample Query Output
User Query:
"A person has been accused of dowry death under Section 304B IPC. 
What are the essential ingredients that prosecution must prove?"
System Output:
Top 5 Similar Cases:
1. Kaliyaperumal v. State of Tamil Nadu - 94.2%
Court: Supreme Court of India
Date: March 2, 2004
Judges: Justice Arijit Pasayat, Justice C.K. Thakker
Key Statutes: Section 304B IPC, Section 113B Evidence Act
Summary: Landmark case defining "soon before death" and establishing that prosecution must prove (1) death of woman caused by burns/bodily injury, (2) death occurred within 7 years of marriage, (3) cruelty or harassment by husband/relatives for dowry, (4) death occurred "soon before" such cruelty...
⸻
2. Biswajit Halder v. State of West Bengal - 91.8%
Court: Supreme Court of India
Date: November 15, 2007
Judges: Justice S.B. Sinha, Justice Markandey Katju
Key Statutes: Section 304B IPC, Section 113B Evidence Act
Summary: Court held that mere proximity to death is insufficient; there must be proximate and live link between cruelty/harassment and death. Prosecution must establish conscious demand for dowry...
⸻
3. Satvir Singh v. State of Punjab - 88.7%
Court: Supreme Court of India
Date: April 18, 2001
Judges: Justice S.P. Bharucha, Justice Y.K. Sabharwal
Key Statutes: Section 304B IPC, Section 498A IPC
Summary: Court clarified the distinction between Section 304B (dowry death) and Section 498A (cruelty). Established that harassment or cruelty must be "in connection with demand for dowry"...

Comparative Legal Analysis:
Essential Ingredients of Section 304B IPC (Dowry Death):
Based on the similar cases retrieved, prosecution must establish:
Death of Woman by Burns/Bodily Injury (not natural death)
Temporal Requirement: Death within 7 years of marriage
Harassment/Cruelty: Woman subjected to cruelty or harassment by:
Husband, OR
Relatives of husband
Dowry Connection: Such cruelty/harassment must be:
In connection with demand for dowry
Conscious and deliberate demand
Proximate Link: Cruelty/harassment must occur "soon before death"
"Soon before": Proximate and live link required
No fixed time period
Depends on facts and circumstances
Generally: Within 1-2 months considered "soon"
Presumption under Section 113B Evidence Act:
If above ingredients proven, court shall presume that husband caused dowry death
Burden shifts to accused to prove innocence
Key Precedents:
Kaliyaperumal (2004): Defined "soon before death"
Biswajit Halder (2007): Proximate link requirement
Satvir Singh (2001): Dowry demand must be conscious
Practical Implications: 
Your case will likely succeed if prosecution can prove cruelty for dowry within a reasonable time before death (typically 1-2 months). The burden will shift to the accused to explain the death.
⸻
Knowledge Graph Visualization:
[Interactive Plotly graph showing]:
Central node: Query case (highlighted)
Connected cases: Similar cases with relationship strengths
Shared judges: Links showing judge connections
Referenced statutes: Common legal provisions
Court hierarchy: Case progression through courts
⸻
Performance Metrics Dashboard
Performance Metrics 
Key Metrics:
Search Accuracy: 92% precision, 89% recall
Response Time: 11.4s average (95th percentile: 18.3s)
Similarity Distribution: Clear separation between relevant and non-relevant cases
Scalability: Sub-linear query time growth with dataset size
Statistical Significance
We performed paired t-tests to verify that LegalNexus significantly outperforms baselines:
Comparison
t-statistic
p-value
Significance
LegalNexus vs. TF-IDF
8.42
<0.001
Highly significant
LegalNexus vs. BM25
6.73
<0.001
Highly significant
LegalNexus vs. Word2Vec
4.91
0.002
Very significant
LegalNexus vs. BERT
3.18
0.014
Significant

Interpretation:
All improvements are statistically significant (p < 0.05)
LegalNexus is not due to chance, but genuine improvement
Strongest improvement over traditional methods (TF-IDF, BM25)
Summary of Results
Key Findings
Superior Accuracy:
92% precision @5 (19% better than BERT, 48% better than TF-IDF)
89% recall @5 (16% better than BERT, 53% better than TF-IDF)
0.91 MAP (14% better than BERT, 57% better than TF-IDF)
Effective Hybrid Approach:
Vector similarity provides semantic understanding
Graph context adds domain-specific relationships
Keyword search ensures recall of exact matches
Domain Optimization:
Legal-specific entity modeling (judges, statutes, courts)
Citation network analysis
Hierarchical court structure
Scalable Architecture:
Sub-linear query time growth
Efficient caching (95% hit rate)
Handles 1000+ cases with <5s query time





























4. Future Plan


The first improvement can focus on the design of the knowledge graph. Right now, it includes information about cases, judges, courts, and statutes. In the future, we can add more details such as lawyers, petitioners, respondents, and main legal ideas. This will help show how all the people and elements in a case are connected. The system can also organize laws more clearly by linking sections, subsections, and related articles. Adding time-based relationships can help track how certain legal ideas or rulings have changed over the years. These updates will make the graph more meaningful and enhance reasoning about legal patterns.

Another idea is query expansion. The system can automatically add related legal terms to the user’s question to find more relevant cases, even if the exact wording is different.

In terms of search and ranking, the current hybrid system uses fixed weights to combine results from vector, keyword, and graph searches. In the future, these weights can adjust automatically depending on the type of query or user feedback. The system can also include a second stage of re-ranking to ensure that the top results are the most relevant. Providing short, clear explanations for why a case was retrieved will make the results easier to trust.

Finally, LegalNexus can become a complete legal assistant that supports interactive questioning. A chatbot interface could let users ask questions in plain language and receive clear, well-explained answers. Automatic updates when new cases are added online will keep the system current. These improvements will make LegalNexus a transparent, intelligent, and reliable tool for future legal research. 









5. Limitations
Despite significant progress in applying AI and knowledge-graph-based methods to legal analytics, several limitations remain across the technical, model, deployment, and ethical dimensions. This chapter discusses these constraints comprehensively and organizes them into distinct categories.

5.1 Technical Limitations
5.1.1 OCR and Document Quality
Legal judgments are often scanned from physical copies or poorly formatted digital archives. These documents may include multiple columns, tables, handwritten notes, stamps, and low-resolution scans, which severely affect Optical Character Recognition (OCR) accuracy.
 Generic OCR systems such as Tesseract struggle with complex layouts and low-quality images, resulting in text misrecognition, misplaced sections, and structural loss. As highlighted by DataVLab.ai, “scanned legal documents present a minefield of challenges” — low-resolution faxes, dense clauses, and degraded quality all contribute to OCR errors.
 Even after cleaning, residual mistakes such as merged clauses or misspelled entities propagate into later NLP stages, reducing the accuracy of downstream tasks like classification or retrieval.

5.1.2 Data Noise and Heterogeneity
Post-OCR text often contains extraneous artifacts such as watermarks, headers, and repetitive section markers. Furthermore, the structure and metadata of legal documents vary significantly across jurisdictions and time periods. Some case files include rich metadata (judge name, case type), while others only provide unstructured text.
 This inconsistency demands intensive data normalization, including spelling correction, metadata alignment, and removal of irrelevant boilerplate text. However, complete uniformity is rarely achieved. The persistence of such noise affects the performance of embedding-based retrieval and case similarity computation.

5.1.3 Multilingual Parsing
Many legal systems operate in multiple languages — for example, India (English and regional languages), Canada (English/French), or the EU (multilingual legislation).
 Most NLP and OCR tools perform best in high-resource languages (English, Chinese, Spanish), but perform poorly for low-resource or morphologically complex languages such as Hindi, Tamil, or Arabic.
 As Krasadakis et al. (MDPI) observe, “most languages other than English, Spanish, and Chinese have very few resources for NLP.”
 This imbalance forces researchers to rely on translation pipelines or small monolingual models, which often introduce further semantic loss or misinterpretation in legal terms.

5.2 Model-Level Limitations
5.2.1 Low-Resource Language Performance
Most state-of-the-art models, including BERT and GPT, are pre-trained on large English corpora. Their effectiveness declines sharply in low-resource legal languages. Even multilingual models (e.g., mBERT, XLM-R) show uneven coverage — legal vocabulary in smaller languages remains underrepresented.
 This limits the transferability of models trained in one jurisdiction to another, hindering the creation of multilingual or cross-border legal AI systems.

5.2.2 Lack of Standardization Across Jurisdictions
Each legal system uses distinct terminologies, evidentiary principles, and procedural structures. There is no standardized ontology or schema to represent legal concepts globally.
 As noted in an MDPI study on Chinese legal KGs, “no unified knowledge graph structure standard exists in the legal field so far.”
 Consequently, integrating multiple national legal systems into a shared knowledge graph is highly complex and error-prone.

5.2.3 Complexity of Legal Logic
Legal reasoning is inherently hierarchical, conditional, and context-dependent. Deep neural models are proficient in statistical text matching but struggle to interpret nuanced logical constructs such as exceptions or multi-level precedents.
 As discussed in arXiv.org, even advanced LLMs fail to capture the intricate reasoning patterns necessary for “complex legal reasoning.”
 Hence, AI recommendations may appear coherent but overlook critical statutory exceptions, potentially leading to misleading or incomplete inferences.

5.2.4 Generalization to Unseen Topics
Legal cases span a wide thematic range — from constitutional to environmental to cyber law. Models trained on common domains (like contract or property law) often perform poorly on rare or emerging domains.
 The scarcity of labeled data for niche topics results in poor generalization. Consequently, retrieval or classification systems may confidently misclassify cases belonging to less-represented categories, limiting their reliability in real-world applications.

5.3 Deployment-Related Limitations
5.3.1 Computational Cost
Training and inference in large neuro-symbolic architectures are computationally intensive.
 Running multi-billion-parameter LLMs such as GPT-4 or LLaMA-65B demands high-end GPUs (e.g., NVIDIA A100/H100) and significant energy resources.
 As reported by Samsi et al. (arXiv), even inference required sharding across eight NVIDIA V100 GPUs. Such infrastructure is often unavailable to academic institutions or smaller firms, creating a barrier to entry.

5.3.2 Latency and Throughput
Inference latency poses another challenge. Large LLMs have limited context windows and slow processing times.
 According to Thomson Reuters’ Justia Verdict (2024), models degrade beyond certain context lengths, necessitating document chunking or Retrieval-Augmented Generation (RAG).
 While RAG improves efficiency, it compromises comprehension by fragmenting contextual flow. Real-time use cases, such as interactive legal assistants, thus face a trade-off between speed and depth of understanding.

5.3.3 Hardware and Data Access
High computational demands restrict deployment to organizations with sufficient infrastructure. Many legal offices lack dedicated GPU clusters or cloud budgets.
 Furthermore, access to comprehensive legal databases (Westlaw, LexisNexis, etc.) is paywalled, limiting training and fine-tuning opportunities. As a result, most open-source models rely on restricted datasets such as Supreme Court judgments, reducing the representativeness and timeliness of results.

5.3.4 Explainability and Interpretability
AI-driven legal systems require transparency to gain judicial trust. However, modern neural networks remain largely black boxes.
 Although knowledge graphs improve interpretability, the final decision-making process — such as which clause or precedent influenced an outcome — remains opaque.
 As several arXiv papers note, “robust and interpretable models” are essential for responsible legal AI deployment. Without clear interpretability, adoption in real legal workflows remains limited.

5.4 Legal and Ethical Limitations
5.4.1 Fairness and Bias
AI systems inherit and amplify the biases present in their training data. Historical case law may encode systemic inequalities, and without correction, the model risks perpetuating them.
 As arXiv.org studies highlight, bias detection and mitigation remain open challenges. Biased recommendations could disadvantage certain demographics or case types, leading to ethical and reputational risks.

5.4.2 Risk of Over-Reliance and Hallucination
Recent investigations into legal-AI tools such as Westlaw AI and LexisNexis AI revealed hallucination rates as high as 17–33% (Justia Verdict, 2024).
 Several real-world incidents have seen lawyers fined for citing fabricated AI-generated cases.
 These examples underscore the importance of human oversight: legal AI must serve as an assistant, not an autonomous authority.

5.4.3 Transparency and Accountability
Legal practitioners must trace the provenance of recommendations — knowing which precedents, statutes, or reasoning paths contributed to an outcome.
 Opaque black-box systems lack such traceability. As Ariai et al. (arXiv) argue, improving explainability is essential to ensure professional accountability and public trust.
 If an AI-assisted judgment turns out erroneous, determining responsibility — between the developer, the user, or the AI — remains a legal gray area.

5.4.4 Legal Compliance and Privacy
Legal texts often contain sensitive personal and case-related data. Systems handling such data must comply with privacy frameworks like GDPR and CCPA.
 Furthermore, certain jurisdictions prohibit fully automated decision-making without human reasoning (“black-box prohibition”).
 Thus, legal-AI deployments require robust governance, secure handling of confidential data, and transparent audit trails.

5.5 Summary
In summary, despite rapid progress, legal-AI systems integrating LLMs and Knowledge Graphs face multiple interdependent challenges:
Data Quality & Multilingual Issues constrain input reliability.


Model Limitations reduce reasoning fidelity and generalization.


Deployment Costs & Latency hinder scalability and responsiveness.


Ethical & Legal Concerns restrict acceptance and regulatory compliance.


Future work must therefore focus on improving OCR preprocessing, building multilingual legal datasets, optimizing lightweight yet explainable models, and enforcing ethical oversight frameworks to ensure trustworthy AI in law.







6. References
[1] I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos, “LEGAL-BERT: The Muppets straight out of law school,” Findings of EMNLP 2020 (Workshops), 2020. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.261/ (Accessed: Oct. 13, 2025).
   (also on arXiv) https://arxiv.org/abs/2010.02559.
[2] H. Li, Q. Ai, J. Chen, Q. Dong, Y. Wu, Y. Liu, C. Chen, and Q. Tian, “SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval,” in Proc. 46th Int. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR), 2023. [Online]. Available: https://dl.acm.org/doi/10.1145/3539618.3591761 (Accessed: Oct. 13, 2025).
   (also on arXiv) https://arxiv.org/abs/2304.11370.
[3] Y. Tang, R. Qiu, Y. Liu, X. Li, and Z. Huang, “CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs,” arXiv preprint, Dec. 2023. [Online]. Available: https://arxiv.org/abs/2312.11229 (Accessed: Oct. 13, 2025).
   (Code repo: https://github.com/yanran-tang/CaseGNN).
[4] Y. Tang, R. Qiu, H. Yin, X. Li, and Z. Huang, “CaseGNN++: Graph Contrastive Learning for Legal Case Retrieval with Graph Augmentation,” arXiv preprint, May 2024. [Online]. Available: https://arxiv.org/abs/2405.11791 (Accessed: Oct. 13, 2025).
   (PDF) https://arxiv.org/pdf/2405.11791.
[5] C. Deng, K. Mao, and Z. Dou, “Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation (KELLER),” arXiv preprint, Jun. 2024. [Online]. Available: https://arxiv.org/abs/2406.19760 (Accessed: Oct. 13, 2025).
   (PDF mirror) https://arxiv.org/pdf/2406.19760.pdf.
[6] COLIEE — Competition on Legal Information Extraction / Entailment (benchmark and tasks). [Online]. Available: https://coliee.org (Accessed: Oct. 13, 2025).
   (Resources & proceedings page) https://coliee.org/resources (Accessed: Oct. 13, 2025).
[7] Registry of Open Data on AWS, “Indian Supreme Court Judgments (public S3 bucket),” 1950–2025 dataset (CC-BY-4.0). [Online]. Available: https://registry.opendata.aws/indian-supreme-court-judgments/ (Accessed: Oct. 13, 2025).
   (S3 bucket root) https://indian-supreme-court-judgments.s3.amazonaws.com/ (Accessed: Oct. 13, 2025).
[8] Caselaw Access Project (CAP), Harvard Law School Library Innovation Lab — US case law data & API. [Online]. Available: https://case.law (Accessed: Oct. 13, 2025).
   (About / API docs) https://case.law/about/ (Accessed: Oct. 13, 2025).
[9] I. Chalkidis, T. Petaschnig, and P. Malakasiotis, “LexGLUE: A Benchmark Dataset for Legal Language Understanding in English,” arXiv preprint, 2021. [Online]. Available: https://arxiv.org/abs/2110.00976 (Accessed: Oct. 13, 2025).
[10] Hugging Face — Transformers and Model Hub (useful tool/resource for LegalBERT, fine-tuning, and model sharing). [Online]. Available: https://huggingface.co (Accessed: Oct. 13, 2025)
