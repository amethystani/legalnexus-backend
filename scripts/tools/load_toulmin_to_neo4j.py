"""
Load Toulmin Argument Structures into Neo4j

Creates the graph schema:
(Claim)-[:SUPPORTED_BY]->(Data)
(Claim)-[:WARRANTED_BY]->(Warrant)
(Warrant)-[:BACKED_BY]->(Authority)
"""

from neo4j import GraphDatabase
from hybrid_case_search import NovelHybridSearchSystem
from toulmin_extractor import ToulminExtractor
import os
from dotenv import load_dotenv

load_dotenv()

def create_toulmin_schema(session):
    """Create indexes and constraints"""
    print("Creating schema...")
    
    # Create constraints
    queries = [
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT data_id IF NOT EXISTS FOR (d:Data) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT warrant_id IF NOT EXISTS FOR (w:Warrant) REQUIRE w.id IS UNIQUE",
    ]
    
    for query in queries:
        try:
            session.run(query)
        except Exception as e:
            print(f"  Note: {e}")
    
    print("✓ Schema created")


def load_toulmin_to_neo4j(case_limit=20):
    """Load Toulmin structures into Neo4j"""
    
    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"Connecting to Neo4j: {uri}")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Initialize system
    print("Initializing case search system...")
    system = NovelHybridSearchSystem()
    
    print(f"Extracting Toulmin structures from {case_limit} cases...")
    extractor = ToulminExtractor(system.llm)
    
    case_samples = [
        (doc.metadata.get('id'), doc.page_content)
        for doc in system.cases_data[:case_limit]
    ]
    
    structures = extractor.extract_batch(case_samples, max_cases=case_limit)
    
    print(f"\n✓ Extracted {len(structures)} valid structures")
    
    # Load into Neo4j
    print("\nLoading into Neo4j...")
    
    with driver.session() as session:
        # Create schema
        create_toulmin_schema(session)
        
        # Load each structure
        for case_id, structure in structures.items():
            print(f"  Loading {case_id}...")
            
            # Create Claim node
            claim_id = f"{case_id}_claim"
            session.run("""
                MERGE (c:Claim {id: $id})
                SET c.text = $text,
                    c.case_id = $case_id,
                    c.confidence = $confidence
            """, id=claim_id, text=structure.claim, case_id=case_id, confidence=structure.confidence)
            
            # Create Data nodes and relationships
            for i, data_text in enumerate(structure.data):
                data_id = f"{case_id}_data_{i}"
                session.run("""
                    MERGE (d:Data {id: $id})
                    SET d.text = $text, d.case_id = $case_id
                    WITH d
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (d)-[:SUPPORTS]->(c)
                """, id=data_id, text=data_text, case_id=case_id, claim_id=claim_id)
            
            # Create Warrant node
            if structure.warrant:
                warrant_id = f"{case_id}_warrant"
                session.run("""
                    MERGE (w:Warrant {id: $id})
                    SET w.text = $text, w.case_id = $case_id
                    WITH w
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (w)-[:WARRANTS]->(c)
                """, id=warrant_id, text=structure.warrant, case_id=case_id, claim_id=claim_id)
                
                # Create Backing nodes
                for i, backing_text in enumerate(structure.backing):
                    backing_id = f"{case_id}_backing_{i}"
                    session.run("""
                        MERGE (b:Authority {id: $id})
                        SET b.text = $text, b.case_id = $case_id
                        WITH b
                        MATCH (w:Warrant {id: $warrant_id})
                        MERGE (b)-[:BACKS]->(w)
                    """, id=backing_id, text=backing_text, case_id=case_id, warrant_id=warrant_id)
        
        # Get stats
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS type, count(n) AS count
            ORDER BY count DESC
        """)
        
        print("\n✓ Graph loaded successfully!")
        print("\n📊 Graph Statistics:")
        for record in result:
            print(f"   {record['type']}: {record['count']}")
        
        # Get relationship counts
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """)
        
        print("\n🔗 Relationships:")
        for record in result:
            print(f"   {record['rel_type']}: {record['count']}")
    
    driver.close()
    print("\n✅ Complete! Toulmin argument graph is now in Neo4j")


if __name__ == "__main__":
    load_toulmin_to_neo4j(case_limit=20)
