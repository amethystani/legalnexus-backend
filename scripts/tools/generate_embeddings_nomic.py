"""
FAST Embedding Generation using Nomic Embed Text (via Ollama)
"""
import pickle
import pandas as pd
from tqdm import tqdm
import requests
import json
from datetime import datetime
import os
import time

def get_nomic_embedding(text, max_retries=3):
    """Get embedding from Ollama nomic-embed-text with retry"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': 'nomic-embed-text',
                    'prompt': text[:8000]
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['embedding']
            elif attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise Exception(f"Ollama HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise Exception(f"Request failed: {e}")

def get_batch_embeddings(texts):
    """Get embeddings for multiple texts"""
    embeddings = []
    for text in texts:
        emb = get_nomic_embedding(text)
        embeddings.append(emb)
    return embeddings

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"NOMIC EMBED TEXT - FAST GENERATION")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check Ollama connection
    try:
        test_emb = get_nomic_embedding("test")
        print(f"✓ Ollama connected (embedding dim: {len(test_emb)})")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("\nMake sure Ollama is running and nomic-embed-text is pulled:")
        print("  ollama pull nomic-embed-text")
        exit(1)
    
    cache = {}
    failed = []
    
    print(f"\nProcessing {len(df)} cases...")
    
    for idx in tqdm(range(len(df)), desc="Embedding"):
        row = df.iloc[idx]
        text = str(row['text'])[:8000]
        
        try:
            emb = get_nomic_embedding(text)
            cache[str(idx)] = emb
            
            # Checkpoint every 1000
            if (idx + 1) % 1000 == 0:
                with open('data/embeddings_checkpoint_nomic.pkl', 'wb') as f:
                    pickle.dump(cache, f)
                
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            failed.append((idx, str(e)))
    
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
        print(f"\nFailed: {len(failed)} cases")
    
    print("\n✅ Ready!")
