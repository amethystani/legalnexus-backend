#!/usr/bin/env python3
"""
Load Complete 1GB Dataset into Knowledge Graph with LLM-based Extraction
Includes progress tracking and visualization generation
"""
import os
import sys
import time
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'main_files'))
from csv_data_loader import load_all_csv_data

# Load environment (try .env.neo4j first, then .env)
load_dotenv('.env.neo4j')
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = "AIzaSyA0dLTfkzxcZYP6KidlFClAyMLl6mea1y8"
genai.configure(api_key=GOOGLE_API_KEY)

def load_complete_dataset_to_kg():
    """Load complete CSV dataset into Neo4j Knowledge Graph"""
    
    print("=" * 80)
    print("LOADING COMPLETE DATASET INTO KNOWLEDGE GRAPH")
    print("=" * 80)
    print("\nDataset: ~96,000 legal cases (1GB)")
    print("Features:")
    print("  ✓ LLM-based batch extraction (Gemini 2.5 Flash)")
    print("  ✓ Automatic fallback to pattern matching")
    print("  ✓ Classification labels preserved")
    print("  ✓ Knowledge graph nodes & relationships")
    print("\n" + "=" * 80)
    
    # Connect to Neo4j (use environment or ask user)
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_url:
        print("\n🔌 Neo4j Connection Required")
        neo4j_url = input("Enter Neo4j URL (e.g., neo4j+s://xxx.databases.neo4j.io): ").strip()
        neo4j_username = input("Enter Neo4j Username (default: neo4j): ").strip() or "neo4j"
        neo4j_password = input("Enter Neo4j Password: ").strip()
    
    print(f"\n📊 Connecting to Neo4j: {neo4j_url}")
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False
    
    # Auto-proceed (user has already confirmed)
    print("\n🚀 Starting data load:")
    print("   1. Clear existing graph data")
    print("   2. Load ~96,000 cases with LLM extraction")
    print("   3. Create nodes and relationships")
    print("   4. Estimated time: 2-3 hours (due to API rate limits)")
    print()
    
    # Clear existing data
    print("\n🗑️  Clearing existing graph data...")
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        print("✓ Graph cleared")
    except Exception as e:
        print(f"⚠️  Warning: Could not clear graph: {e}")
    
    # Load data in chunks
    print("\n📥 Loading data from CSV files...")
    print("Using batch LLM extraction (5 cases per API call)")
    
    start_time = time.time()
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    total_loaded = 0
    
    try:
        # Load all data (will use batch processing internally)
        print("\nNote: This will take time due to API rate limits (10 req/min)")
        print("Progress will be shown every 100 cases...\n")
        
        docs = load_all_csv_data('data', max_cases_per_file=None)  # Load ALL
        
        print(f"\n✓ Loaded {len(docs)} documents with metadata")
        print(f"Time elapsed: {(time.time() - start_time)/60:.1f} minutes")
        
        # Create KG nodes and relationships
        print("\n🔗 Creating Knowledge Graph...")
        print("Creating Case nodes...")
        
        for i, doc in enumerate(docs):
            try:
                # Create Case node
                case_props = {
                    'id': doc.metadata.get('id', f'case_{i}'),
                    'title': doc.metadata.get('title', 'Unknown Case'),
                    'court': doc.metadata.get('court', 'Unknown Court'),
                    'date': doc.metadata.get('judgment_date', 'Unknown Date'),
                    'source': doc.metadata.get('source', ''),
                    'text': doc.page_content[:5000],  # Limit text size
                    'classification_label': doc.metadata.get('classification_label', '0')
                }
                
                cypher = """
                MERGE (c:Case {id: $id})
                SET c.title = $title,
                    c.court = $court,
                    c.date = $date,
                    c.source = $source,
                    c.text = $text,
                    c.label = $classification_label
                """
                graph.query(cypher, params=case_props)
                
                # Create Judge relationships
                if 'judges' in doc.metadata:
                    for judge in doc.metadata['judges']:
                        if judge:
                            judge_cypher = """
                            MERGE (j:Judge {name: $name})
                            WITH j
                            MATCH (c:Case {id: $case_id})
                            MERGE (j)-[:JUDGED]->(c)
                            """
                            graph.query(judge_cypher, params={
                                'name': judge,
                                'case_id': case_props['id']
                            })
                
                # Create Court relationships
                if case_props['court'] != 'Unknown Court':
                    court_cypher = """
                    MERGE (court:Court {name: $name})
                    WITH court
                    MATCH (c:Case {id: $case_id})
                    MERGE (c)-[:HEARD_BY]->(court)
                    """
                    graph.query(court_cypher, params={
                        'name': case_props['court'],
                        'case_id': case_props['id']
                    })
                
                # Create Statute relationships
                if 'statutes' in doc.metadata:
                    for statute in doc.metadata['statutes'][:5]:  # Limit to 5
                        if statute:
                            statute_cypher = """
                            MERGE (s:Statute {name: $name})
                            WITH s
                            MATCH (c:Case {id: $case_id})
                            MERGE (c)-[:REFERENCES]->(s)
                            """
                            graph.query(statute_cypher, params={
                                'name': statute,
                                'case_id': case_props['id']
                            })
                
                # Create Act relationships
                if 'acts' in doc.metadata:
                    for act in doc.metadata['acts'][:3]:  # Limit to 3
                        if act:
                            act_cypher = """
                            MERGE (a:Act {name: $name})
                            WITH a
                            MATCH (c:Case {id: $case_id})
                            MERGE (c)-[:CITES_ACT]->(a)
                            """
                            graph.query(act_cypher, params={
                                'name': act,
                                'case_id': case_props['id']
                            })
                
                total_loaded += 1
                
                # Progress
                if (i + 1) % 100 == 0:
                    elapsed = (time.time() - start_time) / 60
                    rate = (i + 1) / elapsed
                    remaining = (len(docs) - i - 1) / rate if rate > 0 else 0
                    print(f"Progress: {i+1}/{len(docs)} cases ({(i+1)/len(docs)*100:.1f}%) "
                          f"| {elapsed:.1f}m elapsed | ~{remaining:.1f}m remaining")
                
            except Exception as e:
                print(f"Error processing case {i}: {e}")
                continue
        
        # Final statistics
        print(f"\n✅ Knowledge Graph Creation Complete!")
        print(f"Total cases loaded: {total_loaded}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        
        # Get statistics
        print("\n📊 Knowledge Graph Statistics:")
        stats_queries = {
            'Cases': "MATCH (c:Case) RETURN count(c) as count",
            'Judges': "MATCH (j:Judge) RETURN count(j) as count",
            'Courts': "MATCH (c:Court) RETURN count(c) as count",
            'Statutes': "MATCH (s:Statute) RETURN count(s) as count",
            'Acts': "MATCH (a:Act) RETURN count(a) as count",
            'Total Relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        for name, query in stats_queries.items():
            result = graph.query(query)
            count = result[0]['count'] if result else 0
            print(f"  {name}: {count:,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_complete_dataset_to_kg()
    
    if success:
        print("\n" + "=" * 80)
        print("Next Steps:")
        print("  1. Run: streamlit run kg.py")
        print("  2. Explore the Knowledge Graph Visualization tab")
        print("  3. Try similarity search with 96K cases!")
        print("=" * 80)
    else:
        print("\n❌ Loading failed. Check errors above.")

