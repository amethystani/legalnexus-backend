"""
ULTRA-FAST Embedding Generation using Nomic (Parallel, 4 workers for M3)
"""
import pickle
import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_nomic_embedding(text, max_retries=3):
    """Get embedding from Ollama nomic-embed-text with retry"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': 'nomic-embed-text',
                    'prompt': text
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()['embedding']
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
    return None

def get_chunked_embedding(text, chunk_size=2000):
    """
    Split text into chunks, embed each, and average them.
    This preserves full text while avoiding crashes.
    """
    import numpy as np
    
    # Split into chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    if not chunks:
        return None
    
    # Embed each chunk
    chunk_embeddings = []
    for chunk in chunks:
        emb = get_nomic_embedding(chunk)
        if emb:
            chunk_embeddings.append(np.array(emb))
    
    if not chunk_embeddings:
        return None
    
    # Average all chunk embeddings
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    return avg_embedding.tolist()

def process_case(idx_row):
    """Process a single case with chunked embedding"""
    idx, row = idx_row
    text = str(row['text'])  # FULL TEXT, no truncation
    
    emb = get_chunked_embedding(text, chunk_size=2000)
    if emb is not None:
        return (str(idx), emb)
    else:
        return (str(idx), None)

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"ULTRA-FAST NOMIC EMBED (8 Workers, Chunked for Full Text)")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check Ollama
    try:
        test = get_nomic_embedding("test")
        if test:
            print(f"✓ Ollama connected (dim: {len(test)})\n")
        else:
            raise Exception("Failed to get test embedding")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        exit(1)
    
    cache = {}
    failed = []
    
    # Parallel processing with 8 workers (matching OLLAMA_NUM_PARALLEL=8)
    NUM_WORKERS = 8
    print(f"Using {NUM_WORKERS} parallel workers...\n")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_case, (idx, df.iloc[idx])): idx 
                  for idx in range(len(df))}
        
        # Process results as they complete
        with tqdm(total=len(df), desc="Embedding") as pbar:
            for future in as_completed(futures):
                try:
                    idx_str, emb = future.result()
                    if emb is not None:
                        cache[idx_str] = emb
                    else:
                        failed.append(int(idx_str))
                except Exception as e:
                    idx = futures[future]
                    failed.append(idx)
                
                pbar.update(1)
                
                # Checkpoint every 1000
                if len(cache) % 1000 == 0 and len(cache) > 0:
                    with open('data/embeddings_checkpoint_nomic.pkl', 'wb') as f:
                        pickle.dump(cache, f)
    
    # Save final
    print(f"\n\nSaving {len(cache)} embeddings...")
    final_path = 'data/case_embeddings_cache.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(cache, f)
    
    size_mb = os.path.getsize(final_path) / (1024*1024)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Success: {len(cache)}/{len(df)}")
    print(f"Failed: {len(failed)} cases")
    print(f"File: {final_path} ({size_mb:.1f} MB)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n✅ Ready!")
