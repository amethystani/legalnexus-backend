from neo4j import GraphDatabase

uri = "neo4j+s://0b35ada2.databases.neo4j.io"
username = "neo4j"
password = "SbS7OFW-DQrfNtgv7G0_hqBhDUS28O1UzUIKu5Va6KQ"

print(f"Testing connection to: {uri}")

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("RETURN 1 AS num")
        record = result.single()
        print(f"✅ Connection successful! Result: {record['num']}")
        
        # Check node count
        result = session.run("MATCH (n) RETURN count(n) AS count")
        count = result.single()['count']
        print(f"   Current nodes in database: {count}")
    
    driver.close()
    print("\n✅ Ready to load Toulmin argument graph!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
