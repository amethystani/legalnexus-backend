import os
import sys
from hybrid_case_search import NovelHybridSearchSystem

def test_semantic():
    print("Initializing system...")
    system = NovelHybridSearchSystem()
    
    # Test Query
    query = "charge sheet has been filed in the present case"
    print(f"\nTesting Query: '{query}'")
    
    # 1. Check Embedding Generation
    print("\n1. Testing Query Embedding generation...")
    try:
        q_embed = system.embeddings_model.embed_query(query)
        print(f"   Query Embedding Length: {len(q_embed)}")
        print(f"   First 5 values: {q_embed[:5]}")
        if all(v == 0 for v in q_embed):
            print("   ⚠️  WARNING: Query embedding is all zeros!")
    except Exception as e:
        print(f"   ❌ Error generating query embedding: {e}")
        return

    # 2. Check Document Embedding
    print("\n2. Testing Document Embedding (from loaded data)...")
    if not system.cases_data:
        print("   ❌ No cases loaded!")
        return
        
    doc = system.cases_data[0]
    print(f"   First Case: {doc.metadata.get('id')}")
    print(f"   Content start: {doc.page_content[:50]}...")
    
    try:
        d_embed = system.embeddings_model.embed_query(doc.page_content[:1000])
        print(f"   Doc Embedding Length: {len(d_embed)}")
        print(f"   First 5 values: {d_embed[:5]}")
    except Exception as e:
        print(f"   ❌ Error generating doc embedding: {e}")

    # 3. Run Search
    print("\n3. Running Semantic Search...")
    results = system.semantic_search(query, top_k=3)
    for doc, score in results:
        print(f"   - {doc.metadata.get('id')}: Score {score:.4f}")

if __name__ == "__main__":
    test_semantic()
