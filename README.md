# LegalNexus Backend

**A Comprehensive Legal AI Platform with Hyperbolic Graph Neural Networks**

LegalNexus is a research-grade legal information retrieval system that combines hyperbolic embeddings, multi-agent systems, and graph neural networks to provide state-of-the-art case retrieval and legal reasoning capabilities.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git LFS (for large files)

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt
```

### Run Evaluation

```bash
# Activate virtual environment
source venv/bin/activate

# Run the comprehensive evaluation
python src/evaluation/real_evaluation.py
```

This will output metrics for all 6 system contributions:
- Precision@5, NDCG@10, MAP, Recall
- Gromov δ-hyperbolicity
- Court hierarchy validation
- Temporal scoring with resurrection effect
- Toulmin argumentation extraction
- Multi-agent conflict resolution

---

## 📊 Key Results

| Metric | Result |
|--------|--------|
| **Precision@5** | 0.896 |
| **NDCG@10** | 0.893 |
| **Gromov δ** | 0.029 (13.7x better than random) |
| **Hierarchy Valid** | ✅ True |
| **Conflict Resolution** | 98.3% |

---

## 📁 Project Structure

```
legalnexus-backend/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── .gitattributes           # Git LFS configuration
│
├── src/                     # 📦 SOURCE CODE
│   ├── core/                # Core algorithms
│   ├── evaluation/          # Evaluation scripts
│   ├── models/              # Model training
│   ├── ui/                  # User interfaces
│   └── utils/               # Utility functions
│
├── scripts/                 # 🛠️ SCRIPTS & TOOLS
│   ├── setup/               # Setup scripts
│   ├── tools/               # Data processing tools
│   ├── analysis/            # Analysis scripts
│   └── baselines/           # Baseline comparisons
│
├── tests/                   # 🧪 TESTS
│   └── testcases/           # Test case files
│
├── docs/                    # 📚 DOCUMENTATION
│   ├── guides/              # User guides
│   ├── reports/             # Reports
│   └── theory/              # Theoretical background
│
├── latex/                   # 📄 LATEX DOCUMENTS
│   ├── paper/               # Research paper
│   ├── presentation/        # Presentation slides
│   ├── collegereport/       # College report
│   └── libs/                # LaTeX libraries
│
├── assets/                  # 🖼️ STATIC ASSETS
│   ├── images/              # Images & diagrams
│   └── web/                 # HTML visualizations
│
├── data/                    # 💾 DATA FILES
├── config/                  # ⚙️ CONFIGURATION
├── results/                 # 📈 RESULTS & LOGS
├── misc/                    # 📦 MISCELLANEOUS
└── venv/                    # Python virtual environment
```

---

## 📦 Detailed Folder Contents

### `src/` - Source Code

#### `src/core/` - Core Algorithms
| File | Description |
|------|-------------|
| `hyperbolic_gnn.py` | Hyperbolic Graph Convolutional Network implementation |
| `hyperbolic_search.py` | Hyperbolic space search algorithms |
| `hybrid_case_search.py` | Hybrid retrieval combining semantic + structural + citation search |
| `multi_agent_swarm.py` | Game-theoretic multi-agent system with Nash Equilibrium |
| `kg.py` | Knowledge Graph construction and querying |
| `temporal_scorer.py` | Temporal scoring with precedent decay & resurrection |
| `toulmin_extractor.py` | Toulmin argumentation framework extraction |
| `counterfactual_engine.py` | Counterfactual "What-If" analysis engine |
| `argument_chain_traversal.py` | Argument chain traversal for legal reasoning |

#### `src/evaluation/` - Evaluation Scripts
| File | Description |
|------|-------------|
| `real_evaluation.py` | **Main evaluation script** - validates all 6 system contributions |
| `validate_paper_claims.py` | Validates claims made in the research paper |
| `hybrid_retrieval_eval.py` | Hybrid retrieval evaluation metrics |
| `run_paper_experiments.py` | Runs experiments for paper results |
| `run_full_experiments.py` | Comprehensive experiment suite |

#### `src/ui/` - User Interfaces
| File | Description |
|------|-------------|
| `app.py` | Main Flask/FastAPI application |
| `hgcn_search_app.py` | HGCN-based search application |
| `hgcn_search_ui.py` | HGCN search user interface |
| `jina_search_ui.py` | Jina embeddings search UI |
| `demo_hgcn_search.py` | Demo application for HGCN search |

#### `src/utils/` - Utilities
| File | Description |
|------|-------------|
| `data_loader.py` | Data loading utilities |
| `jina_embeddings.py` | Jina embedding generation |
| `jina_embeddings_simple.py` | Simplified Jina embeddings |

---

### `scripts/` - Scripts & Tools

#### `scripts/setup/` - Setup Scripts
| File | Description |
|------|-------------|
| `setup_latex.sh` | LaTeX environment setup |
| `install_*.sh` | Various installation scripts |
| `compile_*.sh` | LaTeX compilation scripts |

#### `scripts/tools/` - Data Processing Tools
| File | Description |
|------|-------------|
| `generate_embeddings*.py` | Various embedding generation scripts |
| `create_*.py` | Knowledge graph creation tools |
| `build_*.py` | Network building utilities |
| `extract_*.py` | Citation extraction tools |
| `load_*.py` | Data loading scripts |
| `visualize_*.py` | Visualization generation |

---

### `data/` - Data Files

| File/Folder | Description |
|-------------|-------------|
| `case_embeddings_cache.pkl` | Pre-computed 768-dim embeddings for 49,634 cases |
| `citation_network.pkl` | Citation network graph (Git LFS) |
| `legal_cases/` | Individual case JSON files with metadata |
| `*.pkl` | Various pickle files with cached data |

---

### `latex/` - LaTeX Documents

| Folder | Description |
|--------|-------------|
| `paper/` | `researchpaper.tex` - Main research paper |
| `presentation/` | `presentation.tex` - Presentation slides |
| `collegereport/` | `collegereport.tex` - Detailed college report |
| `libs/` | pgfplots, tikz, and other LaTeX libraries |

---

### `config/` - Configuration

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables |
| `.env.neo4j` | Neo4j database configuration |
| `label_studio_config.xml` | Label Studio configuration |

---

### `results/` - Results & Outputs

| Folder | Description |
|--------|-------------|
| `logs/` | Application logs |
| `experiments/` | Experiment results |
| `visualizations/` | Generated visualizations |
| `*.json` | Evaluation result files |

---

### `tests/` - Test Files

| Folder/File | Description |
|-------------|-------------|
| `testcases/` | Comprehensive test cases |
| `test_*.py` | Unit and integration tests |
| `quick_*.py` | Quick validation tests |

---

### `docs/` - Documentation

| Folder | Description |
|--------|-------------|
| `guides/` | User guides and quickstart docs |
| `reports/` | Validation reports and summaries |
| `theory/` | Theoretical background documents |

---

## 🔬 Running the Evaluation

The main evaluation script `src/evaluation/real_evaluation.py` validates all 6 contributions:

```bash
# From project root
source venv/bin/activate
python src/evaluation/real_evaluation.py
```

### What it evaluates:

1. **Gromov δ-Hyperbolicity** - Measures how tree-like the embedding space is
2. **Court Hierarchy** - Validates Supreme < High < District in Poincaré space
3. **Temporal Scoring** - Tests resurrection effect for old but cited cases
4. **Toulmin Argumentation** - Extracts argument components from case text
5. **Hybrid Retrieval** - Precision@5, NDCG@10, MAP, Recall metrics
6. **Conflict Resolution** - Multi-agent citation conflict resolution

### Output:
```
📊 COMPREHENSIVE VALIDATION SUMMARY
┌────────────────────────────────────────────┐
│ Metric                    │ Result         │
├───────────────────────────┼────────────────┤
│ Precision@5               │ 0.8960         │
│ Precision@10              │ 0.8888         │
│ NDCG@10                   │ 0.8927         │
│ Gromov δ                  │ 0.0294         │
│ Hierarchy Valid           │ True           │
│ Toulmin Accuracy          │ 100.0%         │
│ Conflict Resolution       │ 98.3%          │
└────────────────────────────────────────────┘
```

Results are saved to `results/real_evaluation_results.json`.

---

## 🏗️ System Architecture

### 6 Key Contributions:

1. **Hyperbolic Graph Convolutional Networks (HGCN)**
   - Embeds 49,634 cases into Poincaré ball
   - Court hierarchy emerges naturally in radial dimension

2. **Game-Theoretic Multi-Agent Swarm**
   - Linker, Interpreter, and Conflict agents
   - Nash Equilibrium for consistent knowledge graph

3. **Adversarial Hybrid Retrieval**
   - Combines semantic, structural, and citation-based search
   - Prosecutor-Defense-Judge simulation

4. **Toulmin Argumentation Framework**
   - Extracts Claim, Ground, Warrant, Backing, Rebuttal
   - Enables argument chain traversal

5. **Temporal Scoring**
   - Precedent decay with resurrection mechanism
   - Reduces obsolete case recommendations

6. **Counterfactual "What-If" Engine**
   - Identifies legal pivot points
   - Measures impact of fact perturbations

---

## 📊 Dataset

- **49,634** legal case embeddings
- **768-dimensional** embeddings (Gemini-based)
- **4 legal topics**: Taxation, Constitutional Law, Civil Dispute, Criminal Law
- **Court hierarchy**: Supreme Court, High Courts, District Courts

---

## 🛠️ Development

### Adding New Features

1. Core algorithms go in `src/core/`
2. Evaluation scripts go in `src/evaluation/`
3. UI components go in `src/ui/`
4. Utility functions go in `src/utils/`
5. Standalone scripts go in `scripts/tools/`

### Running Tests

```bash
source venv/bin/activate
python -m pytest tests/
```

---

## 📄 LaTeX Documents

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

---

## 📝 License

This project is part of academic research. See individual files for licensing information.

---

## 👥 Authors

- Animesh Sinha

---

## 🔗 Links

- [GitHub Repository](https://github.com/amethystani/legalnexus-backend)
