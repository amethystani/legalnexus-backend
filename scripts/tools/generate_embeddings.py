"""
Generate Embeddings for All Cases

Explicitly generates and saves embeddings for all cases using Ollama (nomic-embed-text).
This ensures embeddings are available for HGCN training and graph construction.
"""

import pickle
import os
import time
from data_loader import load_all_cases
from tqdm import tqdm

def generate_embeddings():
    print("="*80)
    print("GENERATING EMBEDDINGS FOR ALL CASES")
    print("="*80)
    
    # 1. Load Cases
    cases = load_all_cases()
    
    # 2. Initialize Embeddings Model (Jina v3 Local)
    print("\n2. Initializing Embeddings Model (Jina v3 Local)...")
    from jina_embeddings import JinaEmbeddings
    
    # Use 'retrieval.passage' for indexing documents
    embeddings_model = JinaEmbeddings(model_name="models/jina-embeddings-v3", task="retrieval.passage")
    
    # 3. Generate Embeddings
    print("\n3. Generating embeddings...")
    
    cache_path = 'data/case_embeddings_cache.pkl'
    embeddings_cache = {}
    
    # Batch processing for efficiency
    # Increased batch size for faster processing on Mac MPS
    batch_size = 64
    
    # Prepare batches
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    
    count = 0
    for batch in tqdm(batches, desc="Processing Batches", unit="batch"):
        texts = [doc.page_content[:8000] for doc in batch] # Truncate to 8k chars
        ids = [doc.metadata['id'] for doc in batch]
        
        try:
            # Embed batch
            embeddings = embeddings_model.embed_documents(texts)
            
            # Store in cache
            for case_id, emb in zip(ids, embeddings):
                embeddings_cache[case_id] = emb
                count += 1
                
        except Exception as e:
            print(f"   ⚠️ Error processing batch: {e}")
            # Fallback: one by one
            for doc in batch:
                try:
                    emb = embeddings_model.embed_query(doc.page_content[:8000])
                    embeddings_cache[doc.metadata['id']] = emb
                    count += 1
                except Exception as inner_e:
                    print(f"   ❌ Failed to embed case {doc.metadata['id']}: {inner_e}")

    print(f"\n   ✓ Generated {count} embeddings")
    
    # 4. Save Cache
    print("\n4. Saving embeddings cache...")
    os.makedirs('data', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    
    print(f"   ✓ Saved to {cache_path}")
    print(f"✅ Embedding generation complete! Total entries: {len(embeddings_cache)}")

if __name__ == "__main__":
    generate_embeddings()
