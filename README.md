# LegalNexus Backend

[![CI](https://github.com/amethystani/legalnexus-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/amethystani/legalnexus-backend/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A Comprehensive Legal AI Platform with Hyperbolic Graph Neural Networks**

LegalNexus is a research-grade legal information retrieval system that combines hyperbolic embeddings, multi-agent systems, and graph neural networks to provide state-of-the-art case retrieval and legal reasoning capabilities. This system was developed as part of academic research to address the unique challenges of legal information retrieval.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Results](#key-results)
3. [Project Structure](#project-structure)
4. [Detailed Folder Contents](#detailed-folder-contents)
5. [Running the Evaluation](#running-the-evaluation)
6. [System Architecture](#system-architecture)
7. [Dataset](#dataset)
8. [Algorithm Details](#algorithm-details)
9. [Development Guide](#development-guide)
10. [LaTeX Documents](#latex-documents)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git with Git LFS support (for large files)
- 8GB+ RAM recommended (for loading embeddings)
- Neo4j database (optional, for knowledge graph features)

### Installation

```bash
# Clone the repository
git clone https://github.com/amethystani/legalnexus-backend.git
cd legalnexus-backend

# Install Git LFS (if not already installed)
git lfs install
git lfs pull

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt
```

### Environment Configuration

Create a `config/.env` file with your credentials (copy from `.env.example` if available):

```bash
# Neo4j Configuration
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Google API Key (for Gemini embeddings)
GOOGLE_API_KEY=your_api_key
```

**Important:** Never commit `.env` files to version control. They are gitignored.

### Run Evaluation

```bash
# Activate virtual environment
source venv/bin/activate

# Run the comprehensive evaluation
python src/evaluation/real_evaluation.py
```

This will output metrics for all 6 system contributions and save results to `results/real_evaluation_results.json`.

---

## Key Results

The evaluation validates 6 major system contributions:

| Metric | Result | Description |
|--------|--------|-------------|
| Precision@5 | 0.896 | 89.6% of top-5 retrieved cases are relevant |
| Precision@10 | 0.889 | 89.9% of top-10 retrieved cases are relevant |
| NDCG@10 | 0.893 | Normalized Discounted Cumulative Gain |
| MAP@100 | 0.816 | Mean Average Precision |
| Gromov delta | 0.029 | Hyperbolicity measure (lower = more tree-like) |
| Random baseline delta | 0.404 | For comparison |
| Improvement factor | 13.7x | Our embeddings are 13.7x more tree-like |
| Hierarchy Valid | Yes | Court hierarchy preserved in Poincare space |
| Toulmin Accuracy | 100% | Argumentation extraction success rate |
| Conflict Resolution | 98.3% | Multi-agent citation conflict resolution |
| Resurrection Effect | +62.4% | Old but cited cases score higher |

---

## Project Structure

```
legalnexus-backend/
|
|-- README.md                 # This documentation file
|-- .gitignore               # Git ignore rules
|-- .gitattributes           # Git LFS configuration for large files
|
|-- src/                     # SOURCE CODE - Core implementation
|   |-- core/                # Core algorithms and modules
|   |-- evaluation/          # Evaluation and validation scripts
|   |-- models/              # Model training scripts
|   |-- ui/                  # User interface applications
|   +-- utils/               # Utility functions and helpers
|
|-- scripts/                 # SCRIPTS AND TOOLS
|   |-- setup/               # Setup and installation scripts
|   |-- tools/               # Data processing and generation tools
|   |-- analysis/            # Analysis and inspection scripts
|   +-- baselines/           # Baseline comparison implementations
|
|-- tests/                   # TEST SUITE
|   |-- testcases/           # Comprehensive test case files
|   +-- test_*.py            # Unit and integration tests
|
|-- docs/                    # DOCUMENTATION
|   |-- guides/              # User guides and tutorials
|   |-- reports/             # Validation reports and summaries
|   +-- theory/              # Theoretical background documents
|
|-- latex/                   # LATEX DOCUMENTS
|   |-- paper/               # Research paper (researchpaper.tex)
|   |-- presentation/        # Presentation slides (presentation.tex)
|   |-- collegereport/       # Detailed college report
|   +-- libs/                # LaTeX libraries (pgfplots, tikz, etc.)
|
|-- assets/                  # STATIC ASSETS
|   |-- images/              # PNG/JPG images and diagrams
|   +-- web/                 # HTML interactive visualizations
|
|-- data/                    # DATA FILES
|   |-- case_embeddings_cache.pkl    # 49,634 case embeddings (768-dim)
|   |-- citation_network.pkl         # Citation network graph
|   +-- legal_cases/                 # Individual case JSON files
|
|-- config/                  # CONFIGURATION
|   |-- requirements.txt     # Python dependencies
|   +-- .env.example         # Environment variable template
|
|-- results/                 # RESULTS AND OUTPUTS
|   |-- logs/                # Application and training logs
|   |-- experiments/         # Experiment result files
|   +-- visualizations/      # Generated visualization outputs
|
|-- misc/                    # MISCELLANEOUS
|   |-- models/              # Saved model weights
|   |-- utils/               # Legacy utility files
|   +-- evaluation/          # Legacy evaluation files
|
+-- venv/                    # Python virtual environment (not tracked)
```

---

## Detailed Folder Contents

### src/core/ - Core Algorithms

This folder contains the main algorithmic implementations:

| File | Description | Key Functions |
|------|-------------|---------------|
| `hyperbolic_gnn.py` | Hyperbolic Graph Convolutional Network implementation. Implements message passing in Poincare ball space using Mobius operations. | `HyperbolicGCN`, `mobius_add`, `exp_map`, `log_map` |
| `hyperbolic_search.py` | Search algorithms in hyperbolic space. Computes Poincare distances and finds nearest neighbors. | `poincare_distance`, `hyperbolic_knn` |
| `hybrid_case_search.py` | Hybrid retrieval system combining 5 search algorithms: semantic, structural, citation-weighted, hyperbolic, and GNN-enhanced. | `HybridSearcher`, `search`, `combine_scores` |
| `multi_agent_swarm.py` | Game-theoretic multi-agent system with Linker, Interpreter, and Conflict agents. Uses Nash Equilibrium for knowledge graph consistency. | `MultiAgentSwarm`, `resolve_conflicts`, `compute_nash` |
| `kg.py` | Knowledge Graph construction and Neo4j integration. Handles case nodes, citation edges, and graph queries. | `KnowledgeGraph`, `add_case`, `find_similar`, `get_citations` |
| `temporal_scorer.py` | Temporal scoring with precedent decay and resurrection mechanism. Old cases with recent citations are boosted. | `calculate_temporal_score`, `decay_function`, `resurrection_boost` |
| `toulmin_extractor.py` | Extracts Toulmin argumentation components (Claim, Ground, Warrant, Backing, Rebuttal) from case text using pattern matching. | `ToulminExtractor`, `extract_components` |
| `counterfactual_engine.py` | Counterfactual "What-If" analysis engine. Measures how fact changes affect retrieval outcomes. | `CounterfactualEngine`, `analyze_pivot_points` |
| `argument_chain_traversal.py` | Traverses argument chains in the knowledge graph to find supporting and opposing precedents. | `traverse_chain`, `find_support`, `find_opposition` |

### src/evaluation/ - Evaluation Scripts

| File | Description | Usage |
|------|-------------|-------|
| `real_evaluation.py` | **Main evaluation script**. Validates all 6 system contributions using real data. Computes Precision, NDCG, MAP, Gromov delta, hierarchy validation, and more. | `python src/evaluation/real_evaluation.py` |
| `validate_paper_claims.py` | Validates specific claims made in the research paper against computed metrics. | `python src/evaluation/validate_paper_claims.py` |
| `hybrid_retrieval_eval.py` | Focused evaluation of hybrid retrieval performance with detailed per-algorithm breakdown. | `python src/evaluation/hybrid_retrieval_eval.py` |
| `run_paper_experiments.py` | Runs the complete experiment suite for paper results. | `python src/evaluation/run_paper_experiments.py` |
| `run_full_experiments.py` | Extended experiments with ablation studies. | `python src/evaluation/run_full_experiments.py` |

### src/ui/ - User Interfaces

| File | Description |
|------|-------------|
| `app.py` | Main web application (Flask/FastAPI) serving search API endpoints |
| `hgcn_search_app.py` | Full-featured HGCN search application with visualization |
| `hgcn_search_ui.py` | Streamlit-based HGCN search interface |
| `jina_search_ui.py` | Search interface using Jina embeddings |
| `demo_hgcn_search.py` | Demo application showcasing HGCN search capabilities |

### src/utils/ - Utilities

| File | Description |
|------|-------------|
| `data_loader.py` | Loads case data, embeddings, and citation networks from various sources |
| `jina_embeddings.py` | Generates embeddings using Jina AI embedding models |
| `jina_embeddings_simple.py` | Simplified Jina embedding interface for quick testing |

### scripts/tools/ - Data Processing Tools

| File | Description |
|------|-------------|
| `generate_embeddings*.py` | Various scripts for generating case embeddings (Gemini, Nomic, Jina) |
| `create_*.py` | Knowledge graph and visualization creation tools |
| `build_*.py` | Network and graph building utilities |
| `extract_*.py` | Citation extraction from case text |
| `load_*.py` | Data loading and Neo4j population scripts |
| `visualize_*.py` | Poincare ball and graph visualization generators |

### data/ - Data Files

| File/Folder | Size | Description |
|-------------|------|-------------|
| `case_embeddings_cache.pkl` | ~300MB | Pre-computed 768-dimensional embeddings for 49,634 legal cases using Gemini |
| `citation_network.pkl` | ~155MB | Citation network graph stored as adjacency list (Git LFS) |
| `legal_cases/` | -- | Directory containing individual case JSON files with full text and metadata |
| `CASE_XXXX.json` | ~5KB each | Individual case files with id, title, court, year, topic, and text |

### latex/ - LaTeX Documents

| Folder | Main File | Description |
|--------|-----------|-------------|
| `paper/` | `researchpaper.tex` | Main IEEE-format research paper (~63KB) |
| `presentation/` | `presentation.tex` | Beamer presentation slides (~25KB) |
| `collegereport/` | `collegereport.tex` | Detailed college report (~184KB) |
| `libs/` | Various | pgfplots, tikz libraries for diagrams |

---

## Running the Evaluation

### Main Evaluation Script

The `src/evaluation/real_evaluation.py` script is the primary way to validate the system:

```bash
cd legalnexus-backend
source venv/bin/activate
python src/evaluation/real_evaluation.py
```

### What the Evaluation Does

1. **Loads Data** (49,634 embeddings, 50 case metadata files)

2. **Gromov Delta-Hyperbolicity Analysis**
   - Samples 500 points, computes 2000 quadruples
   - Measures how tree-like the embedding space is
   - Lower delta = more hierarchical structure

3. **Court Hierarchy Validation**
   - Projects embeddings to Poincare ball
   - Verifies Supreme Court < High Court < District Court radii
   - Confirms hierarchical structure emerges without supervision

4. **Temporal Scoring Analysis**
   - Computes temporal scores for cases by age
   - Validates resurrection effect (old but cited cases score higher)

5. **Toulmin Argumentation Extraction**
   - Extracts argument components from case text
   - Validates extraction accuracy

6. **Hybrid Retrieval Evaluation**
   - Builds 4-layer GNN with k=150 neighbors
   - Computes Precision@5, Precision@10, NDCG@10, MAP@100, Recall@10
   - Uses 500 random queries

7. **Multi-Agent Conflict Resolution**
   - Detects citation conflicts between similar cases
   - Validates resolution success rate

### Output Format

```
================================================================================
LEGALNEXUS COMPREHENSIVE PAPER VALIDATION
================================================================================

LOADING DATA
   [OK] 49634 embeddings loaded
   [OK] Embedding dimension: 768
   [OK] 50 case metadata files

GROMOV DELTA-HYPERBOLICITY ANALYSIS
   Computing from 500 samples, 2000 quadruples...
   Gromov delta: 0.0294
   Random baseline delta: 0.4037
   Improvement: 13.74x

COURT HIERARCHY ANALYSIS
   Supreme Court (center):
      Count: 16379
      Radius: 0.5402
   High Court (middle):
      Count: 16379
      Radius: 0.5750
   District Court (outer):
      Count: 16876
      Radius: 0.6189
   Hierarchy preserved: True

... (more sections)

COMPREHENSIVE VALIDATION SUMMARY
+--------------------------------------------+
| Metric                    | Result         |
+---------------------------+----------------+
| Precision@5               | 0.8960         |
| Precision@10              | 0.8888         |
| NDCG@10                   | 0.8927         |
| Recall@10                 | 0.0009         |
| MAP@100                   | 0.8165         |
| Gromov delta              | 0.0294         |
| Hierarchy Valid           | True           |
| Toulmin Accuracy          | 100.0%         |
| Conflict Resolution       | 98.3%          |
| Resurrection Effect       | +62.4%         |
+--------------------------------------------+

[OK] Results saved to: results/real_evaluation_results.json
```

---

## System Architecture

### The 6 Key Contributions

#### 1. Hyperbolic Graph Convolutional Networks (HGCN)

Traditional Euclidean embeddings fail to capture hierarchical structures. Legal citation networks are inherently tree-like (Supreme Court at root, lower courts as leaves). HGCN embeds cases into a Poincare ball where:

- **Radial dimension** encodes authority (Supreme Court near center)
- **Angular dimension** encodes semantic similarity
- Hierarchy emerges naturally without explicit supervision

Key equations:
- Poincare distance: `d(u,v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))`
- Mobius addition for hyperbolic aggregation

#### 2. Game-Theoretic Multi-Agent Swarm

Three specialized agents coordinate using Nash Equilibrium:

- **Linker Agent**: Identifies citation relationships
- **Interpreter Agent**: Extracts legal principles
- **Conflict Agent**: Resolves contradictory precedents

The agents iterate until reaching equilibrium, producing a consistent knowledge graph.

#### 3. Adversarial Hybrid Retrieval System

Combines 5 distinct search algorithms:

1. **Semantic Search**: Cosine similarity on embeddings
2. **Structural Search**: Graph-based similarity
3. **Citation-Weighted Search**: PageRank-style authority
4. **Hyperbolic Search**: Poincare distance
5. **GNN-Enhanced Search**: Message-passing refined embeddings

A Prosecutor-Defense-Judge simulation synthesizes balanced arguments.

#### 4. Toulmin Argumentation Framework

Extracts structured legal arguments:

- **Claim**: The legal position being argued
- **Ground**: Factual basis supporting the claim
- **Warrant**: Legal principle connecting ground to claim
- **Backing**: Precedent or statute supporting the warrant
- **Rebuttal**: Potential counterarguments

#### 5. Temporal Scoring with Resurrection

Legal precedents have temporal dynamics:

- **Decay**: Old cases may be obsolete
- **Resurrection**: Old cases still being cited remain relevant

Formula: `score = decay(age) + resurrection(recent_citations)`

#### 6. Counterfactual "What-If" Engine

Answers questions like:
- "What if this fact were different?"
- "Which facts are pivot points in this case?"

Measures retrieval changes under fact perturbations.

---

## Dataset

### Overview

| Statistic | Value |
|-----------|-------|
| Total cases | 49,634 |
| Embedding dimension | 768 |
| Embedding model | Gemini |
| Legal topics | 4 (Taxation, Constitutional Law, Civil Dispute, Criminal Law) |
| Court levels | 3 (Supreme Court, High Courts, District Courts) |
| Time span | 1950-2024 |

### Case JSON Format

Each case file (`data/legal_cases/CASE_XXXX.json`) contains:

```json
{
  "id": "CASE_0001",
  "title": "State vs. Defendant",
  "court": "Supreme Court of India",
  "year": 2015,
  "citations": ["AIR 2010 SC 1234", "2005 SCC 567"],
  "text": "Topic: Constitutional Law\n\nFull case text..."
}
```

### Embedding Cache Format

The `case_embeddings_cache.pkl` file is a Python dictionary:

```python
{
  "CASE_0000": np.array([...], dtype=float32),  # 768-dim vector
  "CASE_0001": np.array([...], dtype=float32),
  # ... 49,634 entries
}
```

---

## Algorithm Details

### GNN Message Passing

The hybrid retrieval uses a 4-layer GNN with skip connections:

```python
for layer in range(4):
    for each node i:
        # Find k nearest neighbors
        neighbors = top_k_similar(i, k=150)
        
        # Weighted aggregation
        weights = softmax(similarities[neighbors])
        aggregated = sum(weights * embeddings[neighbors])
        
        # Skip connection
        new_embedding[i] = 0.6 * embedding[i] + 0.4 * aggregated
    
    # Normalize
    embeddings = normalize(new_embeddings)
```

### Hybrid Score Computation

```python
def hybrid_score(query, document, weights=[0.25, 0.75]):
    cosine_sim = dot(query_emb, doc_emb)
    gnn_sim = dot(query_gnn_emb, doc_gnn_emb)
    
    # Normalize to [0, 1]
    cosine_sim = (cosine_sim - min) / (max - min)
    gnn_sim = (gnn_sim - min) / (max - min)
    
    return weights[0] * cosine_sim + weights[1] * gnn_sim
```

### Gromov Delta Computation

For 4 random points x, y, z, w:

```python
# Compute all pairwise distances
sums = [
    d(x,y) + d(z,w),
    d(x,z) + d(y,w),
    d(x,w) + d(y,z)
]
sums.sort()

# Delta is half the difference between largest two
delta = (sums[2] - sums[1]) / 2
```

Lower delta indicates more tree-like (hyperbolic) structure.

---

## Development Guide

### Adding New Features

1. **Core algorithms** → `src/core/`
2. **Evaluation scripts** → `src/evaluation/`
3. **UI components** → `src/ui/`
4. **Utility functions** → `src/utils/`
5. **Standalone scripts** → `scripts/tools/`

### Code Style

- Python 3.10+ with type hints
- NumPy for numerical operations
- Pickle for data serialization
- JSON for configuration and results

### Running Tests

```bash
source venv/bin/activate
python -m pytest tests/
```

### Adding New Embeddings

1. Generate embeddings using `scripts/tools/generate_embeddings_*.py`
2. Save as pickle file in `data/`
3. Update data loader in `src/utils/data_loader.py`

---

## LaTeX Documents

### Compile Research Paper

```bash
cd latex/paper
pdflatex researchpaper.tex
bibtex researchpaper
pdflatex researchpaper.tex
pdflatex researchpaper.tex
```

### Compile Presentation

```bash
cd latex/presentation
pdflatex presentation.tex
```

### Compile College Report

```bash
cd latex/collegereport
pdflatex collegereport.tex
biber collegereport
pdflatex collegereport.tex
pdflatex collegereport.tex
```

---

## Troubleshooting

### Common Issues

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`
**Solution:** `pip install scikit-learn`

**Problem:** Out of memory when loading embeddings
**Solution:** The embedding cache is ~300MB. Ensure 8GB+ RAM available.

**Problem:** Git LFS files not downloaded
**Solution:** Run `git lfs pull` after cloning

**Problem:** Neo4j connection failed
**Solution:** Check `config/.env` has correct Neo4j credentials

**Problem:** Evaluation script path errors after reorganization
**Solution:** Run from project root: `python src/evaluation/real_evaluation.py`

### Getting Help

1. Check `docs/guides/` for detailed guides
2. Review `docs/reports/VALIDATION_REPORT.md` for expected results
3. Open an issue on GitHub

---

## License

This project is part of academic research. See individual files for licensing information.

---

## Links

- GitHub Repository: https://github.com/amethystani/legalnexus-backend
