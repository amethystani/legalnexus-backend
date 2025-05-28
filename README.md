# Legal Nexus Backend

This directory contains the backend code for the Legal Nexus project, a knowledge graph-based legal case similarity engine powered by AI and graph databases.

##  Quick Start

**Main Application**: The primary working application is `kg.py` which provides a Streamlit-based interface for legal case analysis.

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (see setup section below)

# Run the main application
streamlit run kg.py
```

The application will start on `http://localhost:8501`

## Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud instance)
- OpenAI API key (for embeddings and LLM functionality)
- Google Gemini API key (for embeddings - used in case_embeddings_gemini.pkl)

##  Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Neo4j Database Configuration
NEO4J_URI=neo4j+s://.......
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
# Google Gemini Configuration (for embeddings)
GOOGLE_API_KEY=your_google_gemini_api_key

```

### 3. Database Setup

Ensure your Neo4j database is running and accessible with the credentials provided in your `.env` file.

##  Project Structure

### Core Application
- **`kg.py`** -  **MAIN WORKING APPLICATION** - Streamlit interface for legal case similarity analysis

### Data Files
- **`case_embeddings_gemini.pkl`** - Pre-computed case embeddings using Google Gemini API
  - Contains vectorized representations of legal cases for similarity matching
  - Generated using Gemini embeddings model for enhanced legal text understanding
  - Used by kg.py for fast similarity searches without API calls

### Utils Directory
```
utils/
├── web_scraping/
│   └── web_scraper.py          # 🔧 Tools for legal dataset creation and web scraping
└── main_files/
    ├── case_similarity_cli.py  # ⚠️ Command-line interface (partially working)
    ├── citation_network.py     # ⚠️ Citation analysis (redundant/half-working)
    ├── gnn_link_prediction.py  # ⚠️ Graph Neural Network features (not fully implemented)
    ├── integrated_system.py    # ⚠️ System integration (redundant)
    ├── kg_utils.py             # ⚠️ Knowledge graph utilities (partially working)
    └── kg_visualizer.py        # ⚠️ Graph visualization (half-working)
```

### Test Cases
```
testcases/
├── run_tests.py               # Test runner
├── test_*.py                  # Various test files for different components
└── load_test_cases_to_neo4j.py  # Data loading utilities
```

##  Component Status

###  Working Components
- **`kg.py`** - Fully functional Streamlit application
- **`case_embeddings_gemini.pkl`** - Pre-computed embeddings ready for use
- **`utils/web_scraping/web_scraper.py`** - Functional web scraping for dataset creation

###  Partially Working / Redundant Components
- **`utils/main_files/*`** - These files contain experimental or incomplete implementations:
  - Some functions may work but are not integrated into the main application
  - Consider these as development/research code
  - **Not recommended for production use**

##  Key Features

- **Legal Case Similarity Analysis** - Find similar cases based on content and context
- **Knowledge Graph Integration** - Neo4j-powered graph database for legal relationships
- **AI-Powered Embeddings** - Uses Google Gemini for semantic understanding
- **Interactive Web Interface** - Streamlit-based UI for easy interaction
- **Citation Network Analysis** - Understand case relationships and precedents

##  Running the Application

### Main Application (Recommended)
```bash
streamlit run kg.py
```

### Web Scraping (Dataset Creation)
```bash
cd utils/web_scraping
python web_scraper.py
```

### Running Tests
```bash
cd testcases
python run_tests.py
```

##  About Case Embeddings

The `case_embeddings_gemini.pkl` file contains:
- Pre-computed vector embeddings of legal cases
- Generated using Google's Gemini embedding model
- Optimized for legal text understanding and similarity matching
- Enables fast similarity searches without real-time API calls
- Essential for the performance of kg.py application



