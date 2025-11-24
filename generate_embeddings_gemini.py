"""
Generate embeddings using Google Gemini (gemini-embedding-001)
"""
import pickle
import pandas as pd
import re
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import retry
import os
from datetime import datetime
import time

# Configure Gemini
GOOGLE_API_KEY = "AIzaSyBEogciACgVTm4hoIgGI0RuBVC5lYzjd58"  # User provided key
genai.configure(api_key=GOOGLE_API_KEY)

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
    print(f"GEMINI EMBEDDING GENERATION")
    print(f"Total cases: {len(df)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Process in batches
    BATCH_SIZE = 50  # Max is 100, use 50 to be safe
    enriched_cache = {}
    failed = []
    
    print(f"\nProcessing in batches of {BATCH_SIZE}...")
    
    # Create batches
    batches = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    
    for batch_idx, batch_df in enumerate(tqdm(batches, desc="Batches")):
        texts = [str(row['text'])[:9000] for _, row in batch_df.iterrows()] # Limit chars
        
        try:
            # Generate embeddings
            embeddings = generate_batch_embeddings(texts)
            
            # Process results
            for i, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = batch_idx * BATCH_SIZE + i
                case_id = row['case_id']
                text = str(row['text'])[:9000]
                label = row['label']
                
                metadata = extract_metadata(case_id, text, label)
                
                enriched_cache[str(global_idx)] = {
                    'embedding': embeddings[i],
                    'metadata': metadata
                }
            
            # Checkpoint every 20 batches (~1000 cases)
            if (batch_idx + 1) % 20 == 0:
                with open('data/enriched_checkpoint_gemini.pkl', 'wb') as f:
                    pickle.dump(enriched_cache, f)
                
            # Rate limit pause (to be safe)
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            failed.append((batch_idx, str(e)))
    
    # Save final
    print(f"\nSaving final enriched cache...")
    final_path = 'data/case_embeddings_cache_GEMINI.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(enriched_cache, f)
    
    size_mb = os.path.getsize(final_path) / (1024*1024)
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total embedded: {len(enriched_cache)}/{len(df)}")
    print(f"File: {final_path} ({size_mb:.1f} MB)")
    
    if failed:
        print(f"Failed batches: {len(failed)}")
    
    print("\n✅ Ready for use!")
