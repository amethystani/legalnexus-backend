"""
Neo4j Knowledge Graph Builder with Rich Schema

Nodes: Case, Statute, Argument, Fact
Edges: SUPPORTS, ATTACKS, DISTINGUISHES, PROVIDES_WARRANT, FOLLOW, OVERRULE

Built using multi-agent swarm consensus.
"""

from neo4j import GraphDatabase
from multi_agent_swarm import MultiAgentSwarm, Citation, Conflict, EdgeType
from typing import List, Dict
import os
from dotenv import load_dotenv
from data_loader import load_all_cases

load_dotenv()


class LegalKnowledgeGraphBuilder:
    """
    Builds a rich legal knowledge graph in Neo4j using multi-agent swarm.
    """
    
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print(f"✓ Connected to Neo4j: {uri}")
        self.create_schema()
    
    def create_schema(self):
        """Create indexes and constraints"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT statute_id IF NOT EXISTS FOR (s:Statute) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT argument_id IF NOT EXISTS FOR (a:Argument) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass
            print("✓ Schema created")
            
    def clear_database(self):
        """Clear the entire database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")

    def add_case_node(self, metadata: Dict):
        """Add a single case node"""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Case {id: $id})
                SET c.title = $title,
                    c.court = $court,
                    c.date = $date,
                    c.source = $source,
                    c.indexed_at = datetime()
            """, id=metadata['id'], 
                 title=metadata.get('title', 'Unknown'),
                 court=metadata.get('court', 'Unknown'),
                 date=metadata.get('date', 'Unknown'),
                 source=metadata.get('source', 'Unknown'))

    def process_swarm_result(self, result: Dict, source_case_id: str):
        """Process the result from the Multi-Agent Swarm for a single case"""
        citations = result.get('citations', [])
        conflicts = result.get('conflicts', [])
        
        with self.driver.session() as session:
            # Add Citations
            for cit in citations:
                # Map edge type to Neo4j relationship and weight
                rel_type = cit.edge_type.value
                
                if cit.edge_type == EdgeType.FOLLOW:
                    weight = 1.0
                elif cit.edge_type == EdgeType.OVERRULE:
                    weight = -1.0
                elif cit.edge_type == EdgeType.DISTINGUISH:
                    weight = 0.0
                else:
                    weight = 0.5
                
                # Ensure target node exists (placeholder)
                session.run("MERGE (c:Case {id: $id})", id=cit.target_id)
                
                # Create Edge
                session.run(f"""
                    MATCH (source:Case {{id: $source_id}})
                    MATCH (target:Case {{id: $target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.context = $context,
                        r.confidence = $confidence,
                        r.weight = $weight
                """, source_id=source_case_id, target_id=cit.target_id,
                     context=cit.context, confidence=cit.confidence, weight=weight)
            
            # Add Conflicts
            for conflict in conflicts:
                conflict_id = f"conflict_{conflict.conflict_type}_{hash(tuple(conflict.involved_cases))}"
                
                session.run("""
                    MERGE (conf:Conflict {id: $id})
                    SET conf.type = $type,
                        conf.description = $description,
                        conf.severity = $severity
                """, id=conflict_id, type=conflict.conflict_type,
                     description=conflict.description, severity=conflict.severity)
                
                # Link to involved cases
                for case_id in conflict.involved_cases:
                    session.run("""
                        MATCH (conf:Conflict {id: $conf_id})
                        MATCH (c:Case {id: $case_id})
                        MERGE (conf)-[:INVOLVES]->(c)
                    """, conf_id=conflict_id, case_id=case_id)

    def get_graph_stats(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count")
            edges = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count, avg(r.weight) AS avg_weight")
            
            return {
                'nodes': [(rec['type'], rec['count']) for rec in nodes],
                'edges': [(rec['type'], rec['count'], rec.get('avg_weight')) for rec in edges]
            }
    
    def close(self):
        self.driver.close()


def run_multi_agent_pipeline():
    print("="*80)
    print("MULTI-AGENT LEGAL KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*80)
    
    # 1. Load Cases (Unified Loader)
    print("Loading cases from CSV...")
    cases = load_all_cases()
    print(f"✓ Loaded {len(cases)} total cases")
    
    # DEBUG: Test with small subset first
    TEST_MODE = True  # Set to False for full run
    if TEST_MODE:
        cases = cases[:10]
        print(f"⚠️  DEBUG MODE: Testing with only {len(cases)} cases")
    
    print(f"✓ Processing {len(cases)} cases")
    
    # 2. Initialize Builder
    builder = LegalKnowledgeGraphBuilder()
    
    # 3. Clear Database
    print("\nCleaning existing graph...", flush=True)
    builder.clear_database()
    
    # 4. Initialize Swarm
    print("\nInitializing Multi-Agent Swarm...", flush=True)
    swarm = MultiAgentSwarm()
    print("✓ Swarm initialized", flush=True)
    
    # 5. Process Cases
    print(f"\nStarting processing of {len(cases)} cases...")
    
    # Get all case IDs for citation matching
    all_case_ids = set([doc.metadata['id'] for doc in cases])
    
    # Parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
    import multiprocessing
    
    # Match OLLAMA_NUM_PARALLEL
    max_workers = 8
    print(f"🚀 Using {max_workers} parallel workers", flush=True)
    
    processed_count = 0
    failed_count = 0
    
    def process_single_case(doc):
        import time
        try:
            start = time.time()
            case_id = doc.metadata['id']
            print(f"🔹 Starting {case_id}", flush=True)
            
            text = doc.page_content[:5000]  # Limit text
            
            # Run Swarm
            print(f"  Calling swarm for {case_id}...", flush=True)
            result = swarm.process_case(text, case_id, all_case_ids)
            
            elapsed = time.time() - start
            print(f"✅ Completed {case_id} in {elapsed:.1f}s", flush=True)
            
            return case_id, doc.metadata, result, elapsed
        except Exception as e:
            print(f"\n❌ Error {doc.metadata.get('id', 'unknown')}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_case = {executor.submit(process_single_case, doc): doc for doc in cases}
        
        # Batch results for faster Neo4j writes
        batch = []
        batch_size = 10
        total_time = 0
        
        for future in as_completed(future_to_case):
            try:
                result = future.result(timeout=30)  # 30s timeout per case
                if result:
                    case_id, metadata, swarm_result, elapsed = result
                    batch.append((case_id, metadata, swarm_result))
                    total_time += elapsed
                else:
                    failed_count += 1
            except TimeoutError:
                failed_count += 1
                print(f"\n⏱️  Timeout processing case")
                continue
            except Exception as e:
                failed_count += 1
                print(f"\n❌ Future error: {e}")
                continue
                
            # Write batch to Neo4j
            if len(batch) >= batch_size:
                for case_id, metadata, swarm_result in batch:
                    builder.add_case_node(metadata)
                    builder.process_swarm_result(swarm_result, case_id)
                    
                    processed_count += len(batch)
                    avg_time = total_time / processed_count if processed_count > 0 else 0
                    remaining = len(cases) - processed_count
                    eta_seconds = remaining * avg_time / max_workers
                    eta_hours = eta_seconds / 3600
                    
                    print(f"   ✓ {processed_count}/{len(cases)} | Avg: {avg_time:.1f}s/case | ETA: {eta_hours:.1f}h | Failed: {failed_count}", end='\r')
                    batch = []
        
        # Write remaining batch
        if batch:
            for case_id, metadata, swarm_result in batch:
                builder.add_case_node(metadata)
                builder.process_swarm_result(swarm_result, case_id)
            processed_count += len(batch)
            print(f"   ✓ Processed {processed_count}/{len(cases)} cases")
            
    # 6. Show Stats
    print("\n[Graph Statistics]")
    stats = builder.get_graph_stats()
    
    print("\nNodes:")
    for node_type, count in stats['nodes']:
        print(f"  - {node_type}: {count}")
    
    print("\nEdges:")
    for edge_type, count, avg_weight in stats['edges']:
        weight_str = f" (avg weight: {avg_weight:.2f})" if avg_weight is not None else ""
        print(f"  - {edge_type}: {count}{weight_str}")
            
    builder.close()
    print("\n✅ Knowledge Graph Construction Complete!")

if __name__ == "__main__":
    run_multi_agent_pipeline()
