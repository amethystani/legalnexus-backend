
import pickle
import os
from tqdm import tqdm
from jina_embeddings import JinaEmbeddings
from data_loader import load_all_cases

def main():
    print("="*80)
    print("GENERATING EMBEDDINGS (NEW)")
    print("="*80)
    
    # 1. Load cases
    print("\n1. Loading cases...")
    cases = load_all_cases()
    print(f"✓ Loaded {len(cases)} cases")
    
    # 2. Initialize Jina model
    print("\n2. Initializing Jina Embeddings...")
    try:
        embeddings_model = JinaEmbeddings(
            model_name="models/jina-embeddings-v3",
            task="retrieval.passage"
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # 3. Generate embeddings
    print("\n3. Generating embeddings...")
    cache = {}
    
    # Process in batches
    batch_size = 8 # Smaller batch to avoid MPS buffer errors
    batches = [cases[i:i+batch_size] for i in range(0, len(cases), batch_size)]
    
    for batch in tqdm(batches, desc="Processing batches"):
        try:
            texts = [c.page_content[:8000] for c in batch]
            ids = [c.metadata['id'] for c in batch]
            
            # Batch embed
            embeddings = embeddings_model.embed_documents(texts)
            
            # Store
            for case_id, emb in zip(ids, embeddings):
                cache[case_id] = emb
                
        except Exception as e:
            print(f"\n⚠️  Batch error: {e}")
            # Fallback: one by one
            for c in batch:
                try:
                    emb = embeddings_model.embed_query(c.page_content[:8000])
                    cache[c.metadata['id']] = emb
                except:
                    pass
    
    # 4. Save
    print(f"\n4. Saving {len(cache)} embeddings...")
    os.makedirs('data', exist_ok=True)
    with open('data/case_embeddings_cache.pkl', 'wb') as f:
        pickle.dump(cache, f)
    
    size_mb = os.path.getsize('data/case_embeddings_cache.pkl') / (1024*1024)
    print(f"✓ Saved to data/case_embeddings_cache.pkl ({size_mb:.1f} MB)")
    print(f"\n✅ Done! Generated {len(cache)} embeddings")

if __name__ == "__main__":
    main()
