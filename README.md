# Legal Nexus Backend

This directory contains the backend code for the Legal Nexus project, a knowledge graph-based legal case similarity engine.

## Overview

The backend is built using:
- Python with langchain for knowledge graph interactions
- Neo4j for graph database storage
- OpenAI API for embeddings and LLM capabilities

## Setup

1. Install dependencies:
```
pip install -r ../requirements.txt
```

2. Set up environment variables (create a .env file in the Backend directory):
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

## Running the Tests

We have extensive tests to ensure that the knowledge graph functionality works correctly:

```bash
# Run all tests
python run_tests.py

# Run specific tests
python test_kg.py          # Neo4j tests (requires Neo4j connection)
python test_mock_kg.py     # Mock graph tests (no Neo4j needed)
python validate_test_cases.py  # Validate test case files
```

### Test Coverage

The tests cover:

1. **Data Loading**
   - Loading test cases from data/test_cases directory
   - Loading legal data from the data directory and subdirectories

2. **Knowledge Graph Operations**
   - Creating and clearing the graph database
   - Adding nodes (Cases, Judges, Statutes) and relationships
   - Querying the graph for various legal entities

3. **Search Functionality**
   - Simple text search based on legal terminology
   - Similarity search using vector embeddings
   - Fallback mechanisms when exact matches aren't found

4. **Mock Testing**
   - Tests that simulate graph operations without requiring a Neo4j connection
   - Validates the basic logic of the application

## Test Files

- **test_kg.py**: Main test file that requires a Neo4j connection to test real graph operations
- **test_mock_kg.py**: Uses a mock graph implementation to test functionality without Neo4j
- **validate_test_cases.py**: Ensures all test cases have the required fields and structure
- **run_tests.py**: Runs all tests in sequence

## Test Cases

Test cases are located in `data/test_cases/` and include:
- Electronic evidence admissibility cases
- Pension rights cases
- Other legal domains

## Running the Backend App

To start the backend Streamlit app:

```bash
streamlit run kg.py
```

This will start the application on http://localhost:8501 by default.

## Troubleshooting

- If you encounter errors connecting to Neo4j, check your connection strings in .env
- If OpenAI API calls fail, verify your API key and quota
- If test cases aren't loading, ensure the data/test_cases directory exists and has valid JSON files 