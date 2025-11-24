
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    print("Checking Node Labels:")
    result = session.run("CALL db.labels()")
    for record in result:
        print(f" - {record['label']}")
        
    print("\nChecking Relationship Types:")
    result = session.run("CALL db.relationshipTypes()")
    for record in result:
        print(f" - {record['relationshipType']}")
        
    print("\nCounting Nodes:")
    result = session.run("MATCH (n) RETURN count(n) as count")
    print(f" - Total nodes: {result.single()['count']}")
    
    print("\nSample Case Node (if any):")
    result = session.run("MATCH (n:Case) RETURN n LIMIT 1")
    record = result.single()
    if record:
        print(record['n'])
    else:
        print("No 'Case' nodes found.")

driver.close()
