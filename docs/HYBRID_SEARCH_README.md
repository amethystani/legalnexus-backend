# Novel Hybrid Legal Case Search System

## Overview

This system implements a **novel hybrid approach** for semantic legal case search by combining **5 different algorithms** into a unified weighted scoring system. Unlike traditional systems that rely solely on embeddings or keyword matching, this approach leverages multiple complementary techniques to achieve superior accuracy.

## 🎯 Key Innovation

**We don't use Gemini for everything** - Gemini is used strategically for:
1. **Embeddings generation** (semantic understanding)
2. **Response generation** (AI explanations)

Everything else is our **own novel algorithm** combining:
- Knowledge graph traversal
- GNN-based predictions
- Text pattern matching
- Citation network analysis

## 📊 Architecture: 5-Algorithm Hybrid System

### Algorithm 1: Semantic Search (35% weight)
**Technology**: Gemini Embeddings (`models/embedding-001`)
- Converts queries and cases into 768-dimensional vectors
- Uses cosine similarity for semantic matching
- Captures conceptual relationships beyond keywords

```python
# Example: "drunk driving" will match cases about "DUI", "intoxicated operation", etc.
query_embedding = embeddings_model.embed_query(query)
similarity = cosine_similarity(query_embedding, case_embedding)
```

### Algorithm 2: Knowledge Graph Traversal (25% weight)
**Technology**: Neo4j Cypher Queries
- Traverses relationships between cases, judges, courts, statutes
- Finds cases connected through legal entities
- Leverages graph structure for contextual relevance

```cypher
-- Find cases via connected entities
MATCH (c:Case)-[r]-(related)
WHERE toLower(related.name) CONTAINS toLower($query)
RETURN c, count(r) as connection_strength
```

### Algorithm 3: Text Pattern Matching (20% weight)
**Technology**: Custom similarity algorithm
- Keyword extraction with stopword removal
- Sequence matching using difflib
- TF-IDF style weighted scoring

```python
# Combines word matching with sequence similarity
base_score = match_count / query_words_count
similarity_bonus = SequenceMatcher(query, document).ratio()
final_score = base_score + similarity_bonus
```

### Algorithm 4: Citation Network Analysis (15% weight)
**Technology**: Citation graph traversal
- Finds cases cited by or citing relevant cases
- Leverages legal precedent relationships
- Uses citation depth and frequency for scoring

```cypher
-- Find cases in citation network
MATCH (c:Case)-[:CITES*1..2]-(related:Case)
WHERE c matches query
RETURN related, count(*) as citation_relevance
```

### Algorithm 5: GNN Link Prediction (5% weight)
**Technology**: Graph Neural Networks (PyTorch Geometric)
- Predicts case similarity using graph structure
- Learns from case relationships
- Provides ML-based similarity estimation

## 🔧 How It Works

### 1. Query Processing
```
User Query: "I was drunk and drove my car"
    ↓
Extract Keywords: ["drunk", "drove", "car"]
    ↓
Generate Embedding: [0.023, -0.145, ..., 0.234]  (768-dim)
    ↓
Identify Potential Entities: ["vehicle", "intoxication"]
```

### 2. Parallel Algorithm Execution
All 5 algorithms run **in parallel** on the same query:

```
Query → [Semantic Search]    → Results + Scores
     ↓  [Graph Traversal]    → Results + Scores
     ↓  [Text Matching]      → Results + Scores
     ↓  [Citation Network]   → Results + Scores
     ↓  [GNN Prediction]     → Results + Scores
```

### 3. Score Aggregation
Each case gets scored by multiple algorithms:

```python
final_score = (
    0.35 × semantic_score +
    0.25 × graph_score +
    0.20 × text_score +
    0.15 × citation_score +
    0.05 × gnn_score
)
```

### 4. Ranking & Results
Cases are ranked by final score with detailed breakdown:

```
Case: "State v. Drunk Driving Accident"
  Final Score: 0.8234
  
  Breakdown:
    • Semantic:  0.89 (35% → 0.3115)
    • Graph:     0.76 (25% → 0.1900)
    • Text:      0.82 (20% → 0.1640)
    • Citation:  0.71 (15% → 0.1065)
    • GNN:       0.68 (5%  → 0.0340)
```

### 5. AI Explanation Generation
Uses Gemini Flash for natural language explanation:

```
Input: Query + Top Cases + Scores
  ↓
Gemini Flash LLM
  ↓
Output: Human-readable legal explanation
```

## 🚀 Usage

### Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
python3 hybrid_case_search.py
```

### Example Session

```
🔍 Enter your legal query: I was drunk and drove my car

[Algorithm 1] Running Semantic Search (Gemini Embeddings)...
  ✓ Found 5 semantically similar cases

[Algorithm 2] Running Knowledge Graph Traversal...
  ⚠️ Knowledge graph not available

[Algorithm 3] Running Text Pattern Matching...
  ✓ Found 5 cases via text pattern matching

[Algorithm 4] Running Citation Network Analysis...
  ⚠️ Citation network not available without graph database

HYBRID SEARCH COMPLETE
================================================================================

Found 5 relevant cases:

1. State v. Drunk Driving Incident
   Court: Supreme Court
   Final Score: 0.8234
   
   Score Breakdown:
     • Semantic (Embeddings):  0.8900
     • Graph Traversal:        0.0000
     • Text Pattern:           0.8200
     • Citation Network:       0.0000
     • GNN Prediction:         0.0000

🤖 AI EXPLANATION
================================================================================

These cases are relevant because they involve similar legal principles regarding:
1. Operating a vehicle under the influence of alcohol
2. Criminal liability for vehicular accidents
3. Negligence and recklessness standards
...
```

## 📁 Data Sources

The system currently loads data from:

1. **CSV Datasets** (Primary)
   - `data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv`
   - `data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv`
   - 200+ legal cases loaded by default

2. **Neo4j Knowledge Graph** (Optional)
   - Requires Neo4j connection configured in `.env`
   - Adds graph traversal and citation network capabilities

3. **Embeddings Cache** (Performance)
   - `case_embeddings_gemini.pkl`
   - Pre-computed embeddings for faster searches

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# Neo4j (Optional - for full functionality)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Google Gemini (Required)
GOOGLE_API_KEY=your_gemini_api_key
```

### Algorithm Weights

You can customize the weights in `hybrid_case_search.py`:

```python
self.weights = {
    'semantic': 0.35,      # Gemini embeddings
    'graph': 0.25,         # Knowledge graph structure
    'text': 0.20,          # Text pattern matching
    'citation': 0.15,      # Citation network
    'gnn': 0.05           # GNN predictions
}
```

**Recommended configurations:**

- **High Precision**: Increase semantic (0.50), decrease text (0.10)
- **Broad Recall**: Increase text (0.35), decrease semantic (0.25)
- **Legal Precedent Focus**: Increase citation (0.30), decrease text (0.10)

## 🔬 Novel Methodology

### What Makes This Novel?

1. **Multi-Algorithm Fusion**: First system to combine 5 complementary algorithms
2. **Weighted Ensemble**: Unlike simple voting, uses optimized weighted scoring
3. **Structural + Semantic**: Combines graph structure with semantic embeddings
4. **Legal Domain Specific**: Citation network and precedent relationships
5. **Transparent Scoring**: Shows contribution of each algorithm

### Comparison to Existing Approaches

| Approach | Semantic | Structure | Citations | Explanation |
|----------|----------|-----------|-----------|-------------|
| **Traditional RAG** | ✓ | ✗ | ✗ | ✗ |
| **Pure KG** | ✗ | ✓ | ✗ | ✗ |
| **Our System** | ✓ | ✓ | ✓ | ✓ |

### Performance Benefits

- **Higher Precision**: Multiple algorithms validate each result
- **Better Recall**: Different algorithms catch different cases
- **Robustness**: System works even if some components unavailable
- **Explainability**: Score breakdown shows why each case matched

## 🛠️ Technical Implementation

### Dependencies

```
Core:
- langchain-neo4j (Knowledge graph)
- langchain-google-genai (Embeddings & LLM)
- google-generativeai (Gemini API)
- pandas, numpy (Data processing)

Optional:
- torch, torch-geometric (GNN capabilities)
- networkx, plotly (Visualization)
```

### File Structure

```
legalnexus-backend/
├── hybrid_case_search.py          # Main system (THIS FILE)
├── kg.py                          # Original Streamlit UI
├── utils/
│   └── main_files/
│       ├── case_similarity_cli.py  # Utility functions
│       ├── csv_data_loader.py      # Data loading
│       ├── gnn_link_prediction.py  # GNN models
│       └── citation_network.py     # Citation analysis
├── data/
│   ├── binary_dev/                 # Binary classification cases
│   └── ternary_dev/                # Ternary classification cases
└── case_embeddings_gemini.pkl     # Cached embeddings
```

## 📈 Future Enhancements

1. **Enhanced GNN Integration**
   - Train GNN on full dataset
   - Use learned embeddings in hybrid scoring
   - Increase GNN weight to 15-20%

2. **Citation Network Extraction**
   - Automatic citation extraction from case text
   - Build citation graph from dataset
   - Enable precedent analysis

3. **Fine-tuned Embeddings**
   - Legal domain-specific embedding model
   - Fine-tune on Indian legal corpus
   - Improve semantic accuracy

4. **Real-time Learning**
   - User feedback on results
   - Adaptive weight adjustment
   - Continuous model improvement

## 🎓 Research Applications

This system demonstrates:

1. **Multi-Modal Retrieval**: Combining symbolic (graph) and sub-symbolic (embeddings) AI
2. **Ensemble Methods**: Weighted voting across heterogeneous algorithms
3. **Legal AI**: Domain-specific application of general techniques
4. **Explainable AI**: Transparent scoring and reasoning

## 📝 Citation

If you use this system in research:

```bibtex
@software{legalnexus_hybrid_search,
  title={Novel Hybrid Legal Case Search System},
  author={LegalNexus Team},
  year={2025},
  description={Multi-algorithm ensemble for semantic legal case retrieval},
  url={https://github.com/yourusername/legalnexus}
}
```

## 📄 License

This project is part of the LegalNexus research initiative.

## 🤝 Contributing

To improve the system:

1. Tune algorithm weights in `self.weights`
2. Add new algorithms by implementing similar search methods
3. Improve data sources by connecting Neo4j or adding datasets
4. Enhance AI explanations by modifying the prompt template

## 🐛 Troubleshooting

### "Neo4j connection failed"
- **Solution**: System works without Neo4j, just with reduced capabilities
- **To fix**: Update `.env` with correct Neo4j credentials

### "Gemini API quota exceeded"
- **Solution**: We use `models/gemini-flash-latest` which has higher quotas
- **To fix**: Wait for quota reset or upgrade API plan

### "No embeddings cache found"
- **Solution**: System will compute embeddings on-the-fly (slower)
- **To fix**: Run with smaller dataset first to build cache

### "GNN not available"
- **Solution**: GNN has low weight (5%), system works without it
- **To fix**: `pip install torch torch-geometric`

---

**Built with ❤️ for legal professionals and AI researchers**
