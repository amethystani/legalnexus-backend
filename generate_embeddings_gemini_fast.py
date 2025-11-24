"""
Generate embeddings ONLY using Google Gemini (no metadata)
"""
import pickle
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import retry
import os
from datetime import datetime
import time

# Configure Gemini
GOOGLE_API_KEY = "AIzaSyBEogciACgVTm4hoIgGI0RuBVC5lYzjd58"
genai.configure(api_key=GOOGLE_API_KEY)

@retry.Retry(predicate=retry.if_exception_type(Exception))
def generate_batch_embeddings(texts):
    """Generate embeddings for a batch with retry logic"""
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document",
        title="Legal Case"
    )
    return result['embedding']

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"GEMINI EMBEDDING GENERATION (EMBEDDINGS ONLY)")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Process in batches
    BATCH_SIZE = 100  # Max allowed
    cache = {}
    failed = []
    
    print(f"\nProcessing in batches of {BATCH_SIZE}...")
    
    # Create batches
    batches = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    
    for batch_idx, batch_df in enumerate(tqdm(batches, desc="Batches")):
        texts = [str(row['text'])[:9000] for _, row in batch_df.iterrows()]
        
        try:
            embeddings = generate_batch_embeddings(texts)
            
            # Store embeddings only
            for i, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = batch_idx * BATCH_SIZE + i
                cache[str(global_idx)] = embeddings[i]
            
            # Checkpoint every 10 batches (~1000 cases)
            if (batch_idx + 1) % 10 == 0:
                with open('data/embeddings_checkpoint_gemini.pkl', 'wb') as f:
                    pickle.dump(cache, f)
            
            # Rate limit pause
            time.sleep(0.3)
                
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            failed.append((batch_idx, str(e)))
            time.sleep(2)  # Longer pause on error
    
    # Save final
    print(f"\n\nSaving {len(cache)} embeddings...")
    final_path = 'data/case_embeddings_cache.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(cache, f)
    
    size_mb = os.path.getsize(final_path) / (1024*1024)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Embedded: {len(cache)}/{len(df)}")
    print(f"File: {final_path} ({size_mb:.1f} MB)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed:
        print(f"\nFailed batches: {len(failed)}")
    
    print("\n✅ Ready!")
