
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from data_loader import load_all_cases
from tqdm import tqdm

load_dotenv()

def populate_neo4j():
    print("="*80)
    print("POPULATING NEO4J WITH CASES")
    print("="*80)
    
    # 1. Load cases
    cases = load_all_cases()
    if not cases:
        print("❌ No cases loaded. Check data_loader.py and data paths.")
        return

    # 2. Connect to Neo4j
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # 3. Write to Neo4j
    print(f"\nWriting {len(cases)} cases to Neo4j...")
    
    with driver.session() as session:
        # Clear existing data? Maybe safer to just merge
        # session.run("MATCH (n) DETACH DELETE n") 
        
        # Create constraint for speed
        try:
            session.run("CREATE CONSTRAINT FOR (c:Case) REQUIRE c.id IS UNIQUE")
        except:
            pass # Already exists
            
        batch_size = 1000
        batches = [cases[i:i+batch_size] for i in range(0, len(cases), batch_size)]
        
        for batch in tqdm(batches, desc="Uploading batches"):
            params = []
            for doc in batch:
                params.append({
                    'id': doc.metadata['id'],
                    'court': doc.metadata['court'],
                    'title': doc.metadata['title'],
                    'date': doc.metadata['date'],
                    'text': doc.page_content[:1000] # Truncate text to save space/time
                })
            
            session.run("""
                UNWIND $batch as row
                MERGE (c:Case {id: row.id})
                SET c.court = row.court,
                    c.title = row.title,
                    c.date = row.date,
                    c.text = row.text
            """, batch=params)
            
    driver.close()
    print("\n✅ Population complete!")

if __name__ == "__main__":
    populate_neo4j()
