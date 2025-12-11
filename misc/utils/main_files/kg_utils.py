#!/usr/bin/env python3
"""
Knowledge Graph Utilities
Manage your legal knowledge graph database
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Add the Backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kg import check_knowledge_graph_exists, get_knowledge_graph_stats

def connect_to_neo4j():
    """Connect to Neo4j database"""
    load_dotenv()
    
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME") 
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        print("❌ Neo4j credentials not found in environment variables")
        print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD")
        return None
    
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username, 
            password=neo4j_password
        )
        return graph
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return None

def status_command():
    """Check knowledge graph status"""
    print("📊 Knowledge Graph Status")
    print("=" * 50)
    
    graph = connect_to_neo4j()
    if not graph:
        return
    
    print("✅ Connected to Neo4j successfully")
    
    # Check if knowledge graph exists
    kg_exists = check_knowledge_graph_exists(graph)
    
    if kg_exists:
        print("✅ Knowledge graph is ready!")
        
        # Get detailed statistics
        stats = get_knowledge_graph_stats(graph)
        
        print("\n📈 Statistics:")
        print(f"  📚 Legal Cases: {stats.get('cases', 0)}")
        print(f"  👨‍⚖️ Judges: {stats.get('judges', 0)}")
        print(f"  🏛️ Courts: {stats.get('courts', 0)}")
        print(f"  🔍 Vector Indexes: {stats.get('vector_indexes', 0)}")
        
        # Show sample cases
        print("\n🔍 Sample Cases:")
        try:
            sample_query = "MATCH (c:Case) RETURN c.title AS title, c.court AS court LIMIT 5"
            sample_results = graph.query(sample_query)
            
            for i, result in enumerate(sample_results):
                title = result.get('title', 'Untitled')[:60] + "..." if len(result.get('title', '')) > 60 else result.get('title', 'Untitled')
                court = result.get('court', 'Unknown Court')
                print(f"  {i+1}. {title} ({court})")
                
            if not sample_results:
                print("  ⚠️ No cases found")
        except Exception as e:
            print(f"  ❌ Error retrieving sample cases: {e}")
            
        print(f"\n🎯 The system is ready! Run 'streamlit run kg.py' to use it.")
    else:
        print("❌ Knowledge graph not found or incomplete")
        print("🔧 Run 'streamlit run kg.py' to create the knowledge graph")

def clear_command():
    """Clear the knowledge graph"""
    print("🗑️ Clear Knowledge Graph")
    print("=" * 50)
    
    # Confirm with user
    confirm = input("⚠️ This will DELETE ALL data in the knowledge graph. Are you sure? (type 'yes' to confirm): ")
    
    if confirm.lower() != 'yes':
        print("❌ Operation cancelled")
        return
    
    graph = connect_to_neo4j()
    if not graph:
        return
    
    try:
        print("🧹 Clearing all nodes and relationships...")
        clear_query = "MATCH (n) DETACH DELETE n"
        graph.query(clear_query)
        
        print("✅ Knowledge graph cleared successfully!")
        print("🔧 Run 'streamlit run kg.py' to rebuild the knowledge graph")
        
    except Exception as e:
        print(f"❌ Error clearing knowledge graph: {e}")

def count_command():
    """Count different types of nodes"""
    print("🔢 Node Counts")
    print("=" * 50)
    
    graph = connect_to_neo4j()
    if not graph:
        return
    
    try:
        # Fallback method that works across Neo4j versions
        print("📊 Node counts by type:")
        
        labels = ['Case', 'Judge', 'Court', 'Statute']
        total = 0
        
        for label in labels:
            try:
                query = f"MATCH (n:{label}) RETURN count(n) as count"
                result = graph.query(query)
                count = result[0].get('count', 0) if result else 0
                total += count
                if count > 0:
                    print(f"  {label}: {count}")
            except Exception as e:
                print(f"  {label}: Error - {e}")
        
        print(f"\nTotal counted nodes: {total}")
            
    except Exception as e:
        print(f"❌ Error counting nodes: {e}")

def relationships_command():
    """Show relationship information"""
    print("🔗 Relationship Information")
    print("=" * 50)
    
    graph = connect_to_neo4j()
    if not graph:
        return
    
    try:
        # Count relationships
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        results = graph.query(rel_query)
        
        print("📊 Relationship counts:")
        total = 0
        for result in results:
            rel_type = result.get('relationship_type', 'Unknown')
            count = result.get('count', 0)
            total += count
            print(f"  {rel_type}: {count}")
        
        print(f"\nTotal relationships: {total}")
        
    except Exception as e:
        print(f"❌ Error retrieving relationships: {e}")

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Utilities")
    parser.add_argument('command', choices=['status', 'clear', 'count', 'relationships'], 
                       help='Command to execute')
    
    args = parser.parse_args()
    
    print("⚖️ Legal Knowledge Graph Utilities")
    print()
    
    if args.command == 'status':
        status_command()
    elif args.command == 'clear':
        clear_command()
    elif args.command == 'count':
        count_command()
    elif args.command == 'relationships':
        relationships_command()

if __name__ == "__main__":
    main() 