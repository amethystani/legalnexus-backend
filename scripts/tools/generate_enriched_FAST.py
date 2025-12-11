"""
FAST parallel enriched embedding generation (8 workers for M3)
"""
import pickle
import pandas as pd
import re
from tqdm import tqdm
from jina_embeddings import JinaEmbeddings
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_metadata(case_id, text, label):
    """Extract all metadata from case"""
    metadata = {}
    metadata['original_label'] = str(label)
    
    # Court level
    cid_lower = str(case_id).lower()
    if 'sc_' in cid_lower or 'supreme' in cid_lower or 'sci' in cid_lower:
        metadata['court_level'] = 'Supreme Court'
    elif 'hc' in cid_lower or 'high' in cid_lower:
        metadata['court_level'] = 'High Court'
    else:
        metadata['court_level'] = 'Lower Court/Tribunal'
    
    # Court name
    match = re.search(r'^([A-Za-z_]+)_', str(case_id))
    metadata['court_name'] = match.group(1).replace('_', ' ') if match else 'Unknown'
    
    # Year
    year_match = re.search(r'_(\d{4})_', str(case_id))
    metadata['year'] = int(year_match.group(1)) if year_match else None
    
    # Case types
    text_lower = str(text).lower()
    case_types = []
    
    if any(kw in text_lower for kw in ['criminal', 'accused', 'murder', 'theft', 'ipc', 'crpc']):
        case_types.append('Criminal')
    if any(kw in text_lower for kw in ['civil', 'damages', 'contract', 'property', 'suit']):
        case_types.append('Civil')
    if any(kw in text_lower for kw in ['income tax', 'gst', 'sales tax', 'customs', 'excise']):
        case_types.append('Tax')
    if any(kw in text_lower for kw in ['consumer', 'deficiency', 'compensation for service']):
        case_types.append('Consumer')
    if any(kw in text_lower for kw in ['company', 'shareholders', 'directors']):
        case_types.append('Company')
    if any(kw in text_lower for kw in ['constitutional', 'fundamental right', 'writ']):
        case_types.append('Constitutional')
    if any(kw in text_lower for kw in ['labour', 'employee', 'industrial dispute']):
        case_types.append('Labour')
    
    metadata['case_types'] = case_types if case_types else ['General']
    metadata['case_id'] = str(case_id)
    
    return metadata

def process_batch(batch_data):
    """Process a batch of cases (runs in separate process)"""
    batch_idx, batch_rows = batch_data
    
    # Initialize Jina in this worker
    embeddings_model = JinaEmbeddings(
        model_name="models/jina-embeddings-v3",
        task="retrieval.passage"
    )
    
    results = {}
    for idx, row in batch_rows:
        try:
            case_id = row['case_id']
            text = str(row['text'])[:8000]
            label = row['label']
            
            # Generate embedding
            emb = embeddings_model.embed_query(text)
            
            # Extract metadata
            metadata = extract_metadata(case_id, text, label)
            
            # Store
            results[str(idx)] = {
                'embedding': emb,
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue
    
    return results

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"FAST PARALLEL ENRICHED EMBEDDING GENERATION (8 Workers)")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Split into batches for 8 workers
    num_workers = 8
    batch_size = len(df) // num_workers
    
    batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_workers - 1 else len(df)
        
        batch_rows = [(idx, df.iloc[idx]) for idx in range(start_idx, end_idx)]
        batches.append((i, batch_rows))
        print(f"Worker {i+1}: Processing {len(batch_rows)} cases (indices {start_idx}-{end_idx-1})")
    
    print(f"\n[1/2] Starting {num_workers} parallel workers...")
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Batches"
        ))
    
    # Merge all results
    print(f"\n[2/2] Merging results...")
    enriched_cache = {}
    for batch_result in batch_results:
        enriched_cache.update(batch_result)
    
    # Save
    print(f"\nSaving {len(enriched_cache)} embeddings...")
    final_path = 'data/case_embeddings_cache_ENRICHED.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(enriched_cache, f)
    
    size_mb = os.path.getsize(final_path) / (1024*1024)
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total embedded: {len(enriched_cache)}/{len(df)}")
    print(f"File: {final_path} ({size_mb:.1f} MB)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Ready for use!")
