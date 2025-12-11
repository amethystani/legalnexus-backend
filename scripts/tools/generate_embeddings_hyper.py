"""
HYPER-OPTIMIZED: Parallel chunk processing for maximum speed
Strategy: Process ALL chunks in parallel, then group by case
"""
import pickle
import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def get_embedding(text, retries=2):
    """Get single chunk embedding"""
    for attempt in range(retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': 'nomic-embed-text', 'prompt': text},
                timeout=20  # Reduced from 30
            )
            if response.status_code == 200:
                return response.json()['embedding']
            time.sleep(0.1)  # Reduced from 0.3
        except:
            if attempt < retries - 1:
                time.sleep(0.1)  # Reduced from 0.3
    return None

def process_chunk(chunk_data):
    """Process a single chunk - this runs in parallel"""
    case_idx, chunk_idx, chunk_text = chunk_data
    emb = get_embedding(chunk_text)
    return (case_idx, chunk_idx, emb)

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"HYPER-OPTIMIZED PARALLEL CHUNKING (128 Workers)")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Create ALL chunks upfront
    print("\n[1/3] Creating chunks...")
    all_chunks = []
    CHUNK_SIZE = 2000
    
    for idx in range(len(df)):
        text = str(df.iloc[idx]['text'])
        for chunk_start in range(0, len(text), CHUNK_SIZE):
            chunk_text = text[chunk_start:chunk_start+CHUNK_SIZE].strip()
            if chunk_text:
                all_chunks.append((idx, len(all_chunks), chunk_text))
    
    print(f"   Total chunks to process: {len(all_chunks)}")
    print(f"   Average chunks per case: {len(all_chunks)/len(df):.1f}")
    
    # Step 2: Process all chunks in parallel
    print(f"\n[2/3] Processing chunks with 8 workers...")
    
    chunk_results = {}  # {case_idx: [(chunk_idx, embedding), ...]}
    failed_chunks = []
    
    NUM_WORKERS = 128  # Extreme parallelization for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in all_chunks}
        
        with tqdm(total=len(all_chunks), desc="Chunks") as pbar:
            for future in as_completed(futures):
                try:
                    case_idx, chunk_idx, emb = future.result()
                    if emb:
                        if case_idx not in chunk_results:
                            chunk_results[case_idx] = []
                        chunk_results[case_idx].append((chunk_idx, emb))
                    else:
                        failed_chunks.append(chunk_idx)
                except Exception as e:
                    failed_chunks.append(futures[future][1])
                
                pbar.update(1)
    
    # Step 3: Average chunks to create case embeddings
    print(f"\n[3/3] Averaging chunks into case embeddings...")
    cache = {}
    
    for case_idx in tqdm(range(len(df)), desc="Cases"):
        if case_idx in chunk_results:
            # Sort by chunk_idx and average
            chunks = sorted(chunk_results[case_idx], key=lambda x: x[0])
            embeddings = [np.array(emb) for _, emb in chunks]
            avg_emb = np.mean(embeddings, axis=0)
            cache[str(case_idx)] = avg_emb.tolist()
    
    # Save
    print(f"\nSaving {len(cache)} embeddings...")
    final_path = 'data/case_embeddings_cache.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(cache, f)
    
    size_mb = os.path.getsize(final_path) / (1024*1024)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Success: {len(cache)}/{len(df)} cases")
    print(f"Failed chunks: {len(failed_chunks)}")
    print(f"File: {final_path} ({size_mb:.1f} MB)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Ready!")
