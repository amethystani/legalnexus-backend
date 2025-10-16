#!/usr/bin/env python3
"""
Quick Start: Load Dataset into Neo4j with Progress Tracking
Simplified version with clear progress output
"""
import os
import sys
import time
from dotenv import load_dotenv

# Load Neo4j credentials
load_dotenv('.env.neo4j')

print("="*80)
print("LEGAL KNOWLEDGE GRAPH - DATASET LOADER")
print("="*80)
print(f"\nStarting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Check environment
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')

if not all([neo4j_uri, neo4j_username, neo4j_password]):
    print("❌ Error: Neo4j credentials not found in .env.neo4j")
    sys.exit(1)

print(f"✓ Neo4j URI: {neo4j_uri}")
print(f"✓ Username: {neo4j_username}")

# Import required modules
print("\n📦 Loading modules...")
try:
    from langchain_neo4j import Neo4jGraph
    from neo4j import GraphDatabase
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'main_files'))
    from csv_data_loader import load_all_csv_data
    print("✓ All modules loaded")
except Exception as e:
    print(f"❌ Error loading modules: {e}")
    sys.exit(1)

# Connect to Neo4j
print("\n🔌 Connecting to Neo4j...")
try:
    graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
    graph.query("RETURN 1")
    print("✓ Connected successfully")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

# Clear existing data
print("\n🗑️  Clearing existing data...")
try:
    graph.query("MATCH (n) DETACH DELETE n")
    print("✓ Database cleared")
except Exception as e:
    print(f"⚠️  Warning: {e}")

# Load CSV data
print("\n📥 Loading CSV data with LLM extraction...")
print("This will take time due to API rate limits (10 req/min)")
print("Processing in batches of 5 cases per API call\n")

start_time = time.time()

try:
    # Load ALL data
    docs = load_all_csv_data('data', max_cases_per_file=None)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Loaded {len(docs)} documents in {elapsed:.1f} minutes")
    
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create knowledge graph
print("\n🔗 Creating Knowledge Graph nodes and relationships...")
print("This may take 30-60 minutes for ~96K cases\n")

created = 0
errors = 0
kg_start = time.time()

for i, doc in enumerate(docs):
    try:
        # Create Case node
        case_props = {
            'id': doc.metadata.get('id', f'case_{i}'),
            'title': doc.metadata.get('title', 'Unknown Case'),
            'court': doc.metadata.get('court', 'Unknown Court'),
            'date': doc.metadata.get('judgment_date', 'Unknown Date'),
            'source': doc.metadata.get('source', ''),
            'text': doc.page_content[:3000],
            'label': doc.metadata.get('classification_label', '0')
        }
        
        graph.query("""
            MERGE (c:Case {id: $id})
            SET c.title = $title, c.court = $court, c.date = $date,
                c.source = $source, c.text = $text, c.label = $label
        """, params=case_props)
        
        # Create Judge relationships
        if 'judges' in doc.metadata:
            for judge in doc.metadata['judges'][:3]:
                if judge:
                    try:
                        graph.query("""
                            MERGE (j:Judge {name: $name})
                            WITH j MATCH (c:Case {id: $case_id})
                            MERGE (j)-[:JUDGED]->(c)
                        """, params={'name': judge, 'case_id': case_props['id']})
                    except:
                        pass
        
        # Create Act relationships  
        if 'acts' in doc.metadata:
            for act in doc.metadata['acts'][:2]:
                if act:
                    try:
                        graph.query("""
                            MERGE (a:Act {name: $name})
                            WITH a MATCH (c:Case {id: $case_id})
                            MERGE (c)-[:CITES_ACT]->(a)
                        """, params={'name': act, 'case_id': case_props['id']})
                    except:
                        pass
        
        created += 1
        
        # Progress update every 1000 cases
        if (i + 1) % 1000 == 0:
            elapsed = (time.time() - kg_start) / 60
            rate = (i + 1) / elapsed
            remaining = (len(docs) - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1:,}/{len(docs):,} ({(i+1)/len(docs)*100:.1f}%) | "
                  f"{elapsed:.1f}m elapsed | ~{remaining:.0f}m remaining | "
                  f"{rate:.1f} cases/min")
            
    except Exception as e:
        errors += 1
        if errors < 10:
            print(f"Error on case {i}: {e}")

# Final stats
total_time = (time.time() - start_time) / 60
print(f"\n{'='*80}")
print("✅ LOADING COMPLETE!")
print(f"{'='*80}")
print(f"Cases loaded: {created:,}/{len(docs):,}")
print(f"Errors: {errors}")
print(f"Total time: {total_time:.1f} minutes")

# Get final counts
print("\n📊 Knowledge Graph Statistics:")
stats = [
    ("Cases", "MATCH (c:Case) RETURN count(c) as count"),
    ("Judges", "MATCH (j:Judge) RETURN count(j) as count"),
    ("Courts", "MATCH (c:Court) RETURN count(c) as count"),
    ("Acts", "MATCH (a:Act) RETURN count(a) as count"),
    ("Relationships", "MATCH ()-[r]->() RETURN count(r) as count")
]

for name, query in stats:
    try:
        result = graph.query(query)
        count = result[0]['count'] if result else 0
        print(f"  {name}: {count:,}")
    except:
        print(f"  {name}: 0")

print(f"\n{'='*80}")
print("Next Steps:")
print("  1. Run: python3 generate_all_visualizations.py")
print("  2. Run: streamlit run kg.py")
print(f"{'='*80}\n")

