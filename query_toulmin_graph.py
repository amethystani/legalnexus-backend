"""Query the Toulmin argument graph in Neo4j"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    print("=" * 80)
    print("TOULMIN ARGUMENT GRAPH - NEO4J")
    print("=" * 80)
    
    # Get all claims
    print("\n📌 CLAIMS IN DATABASE:")
    result = session.run("""
        MATCH (c:Claim)
        RETURN c.case_id as case_id, c.text as claim, c.confidence as confidence
        ORDER BY c.confidence DESC
    """)
    
    for i, record in enumerate(result, 1):
        print(f"\n{i}. {record['case_id']}")
        print(f"   Claim: {record['claim']}")
        print(f"   Confidence: {record['confidence']:.2f}")
        
        # Get supporting data
        data_result = session.run("""
            MATCH (d:Data)-[:SUPPORTS]->(c:Claim {case_id: $case_id})
            RETURN d.text as data
        """, case_id=record['case_id'])
        
        data_points = [r['data'] for r in data_result]
        if data_points:
            print(f"   Supporting Facts:")
            for j, data in enumerate(data_points, 1):
                print(f"     {j}. {data}")
        
        # Get warrant
        warrant_result = session.run("""
            MATCH (w:Warrant)-[:WARRANTS]->(c:Claim {case_id: $case_id})
            RETURN w.text as warrant
        """, case_id=record['case_id'])
        
        warrant = warrant_result.single()
        if warrant:
            print(f"   Legal Warrant: {warrant['warrant']}")

    # Visualize argument chain example
    print("\n" + "=" * 80)
    print("SAMPLE ARGUMENT CHAIN VISUALIZATION")
    print("=" * 80)
    
    result = session.run("""
        MATCH path = (d:Data)-[:SUPPORTS]->(c:Claim)<-[:WARRANTS]-(w:Warrant)<-[:BACKS]-(a:Authority)
        RETURN d.text as data, c.text as claim, w.text as warrant, a.text as authority
        LIMIT 1
    """)
    
    record = result.single()
    if record:
        print("\n[Fact] → [Claim] ← [Warrant] ← [Authority]\n")
        print(f"📊 Fact: {record['data']}")
        print(f"     ↓")
        print(f"📌 Claim: {record['claim']}")
        print(f"     ↑")
        print(f"⚖️  Warrant: {record['warrant']}")
        print(f"     ↑")
        print(f"📚 Authority: {record['authority']}")

driver.close()
