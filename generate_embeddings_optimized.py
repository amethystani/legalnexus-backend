"""
Optimized Embedding Generation (Batched, Single Process for 8GB RAM)
"""
import pickle
import pandas as pd
import re
from tqdm import tqdm
from jina_embeddings import JinaEmbeddings
import numpy as np
import os
from datetime import datetime
import torch

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

if __name__ == '__main__':
    # Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    
    print("="*80)
    print(f"OPTIMIZED EMBEDDING GENERATION (Batched, Single Process)")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize Jina
    print("\n[1/3] Initializing Jina embeddings...")
    embeddings_model = JinaEmbeddings(
        model_name="models/jina-embeddings-v3",
        task="retrieval.passage"
    )
    print("✓ Model loaded")
    
    # Process in batches
    BATCH_SIZE = 4  # Reduced from 32 to fit in 8GB RAM with 8k context
    enriched_cache = {}
    failed = []
    
    print(f"\n[2/3] Processing in batches of {BATCH_SIZE}...")
    
    # Create batches
    batches = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    
    for batch_idx, batch_df in enumerate(tqdm(batches, desc="Batches")):
        texts = [str(row['text'])[:8000] for _, row in batch_df.iterrows()]
        
        try:
            # Generate embeddings for batch
            # JinaEmbeddings.embed_documents handles list of texts
            embeddings = embeddings_model.embed_documents(texts)
            
            # Process results
            for i, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = batch_idx * BATCH_SIZE + i
                case_id = row['case_id']
                text = str(row['text'])[:8000]
                label = row['label']
                
                metadata = extract_metadata(case_id, text, label)
                
                enriched_cache[str(global_idx)] = {
                    'embedding': embeddings[i],
                    'metadata': metadata
                }
            
            # Checkpoint every 50 batches (~1600 cases)
            if (batch_idx + 1) % 50 == 0:
                with open('data/enriched_checkpoint_optimized.pkl', 'wb') as f:
                    pickle.dump(enriched_cache, f)
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Fallback: try individually
            for i, (_, row) in enumerate(batch_df.iterrows()):
                try:
                    global_idx = batch_idx * BATCH_SIZE + i
                    text = str(row['text'])[:8000]
                    emb = embeddings_model.embed_query(text)
                    metadata = extract_metadata(row['case_id'], text, row['label'])
                    enriched_cache[str(global_idx)] = {
                        'embedding': emb,
                        'metadata': metadata
                    }
                except Exception as inner_e:
                    failed.append((global_idx, str(inner_e)))
    
    # Save final
    print(f"\n[3/3] Saving final enriched cache...")
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
    
    if failed:
        print(f"Failed cases: {len(failed)}")
    
    print("\n✅ Ready for use!")
