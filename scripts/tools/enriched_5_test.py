"""
Generate embeddings WITH enriched metadata (5 cases test)
"""
import pickle
import pandas as pd
import re
from jina_embeddings import JinaEmbeddings

def extract_metadata(case_id, text, label):
    """Extract all possible metadata"""
    metadata = {}
    
    # 1. Original label
    metadata['original_label'] = str(label)
    
    # 2. Court level
    cid_lower = str(case_id).lower()
    if 'sc_' in cid_lower or 'supreme' in cid_lower or 'sci' in cid_lower:
        metadata['court_level'] = 'Supreme Court'
    elif 'hc' in cid_lower or 'high' in cid_lower:
        metadata['court_level'] = 'High Court'
    else:
        metadata['court_level'] = 'Lower Court/Tribunal'
    
    # 3. Court/Tribunal name
    match = re.search(r'^([A-Za-z_]+)_', str(case_id))
    if match:
        metadata['court_name'] = match.group(1).replace('_', ' ')
    else:
        metadata['court_name'] = 'Unknown'
    
    # 4. Year
    year_match = re.search(r'_(\d{4})_', str(case_id))
    if year_match:
        metadata['year'] = int(year_match.group(1))
    else:
        metadata['year'] = None
    
    # 5. Case type (keyword-based)
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
    
    metadata['case_types'] = case_types if case_types else ['General']
    
    # 6. Case ID
    metadata['case_id'] = str(case_id)
    
    return metadata

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("ENRICHED EMBEDDINGS WITH METADATA (5 cases)")
print("="*80)

# Initialize Jina
print("\n1. Initializing Jina...")
embeddings_model = JinaEmbeddings(
    model_name="models/jina-embeddings-v3",
    task="retrieval.passage"
)

# Generate for 5 cases
print("\n2. Generating embeddings with metadata...")
enriched_cache = {}

for idx in range(5):
    row = df.iloc[idx]
    case_id = row['case_id']
    text = str(row['text'])[:8000]
    label = row['label']
    
    print(f"\n   Case {idx}: {case_id}")
    
    # Generate embedding
    emb = embeddings_model.embed_query(text)
    
    # Extract metadata
    metadata = extract_metadata(case_id, text, label)
    
    # Store together
    enriched_cache[str(idx)] = {
        'embedding': emb,
        'metadata': metadata
    }
    
    print(f"      Court: {metadata['court_level']} - {metadata['court_name']}")
    print(f"      Year: {metadata['year']}")
    print(f"      Types: {', '.join(metadata['case_types'])}")
    print(f"      Label: {metadata['original_label']}")

# Save
print("\n3. Saving enriched cache...")
with open('data/enriched_embeddings_5_TEST.pkl', 'wb') as f:
    pickle.dump(enriched_cache, f)

print("\n✅ Saved to data/enriched_embeddings_5_TEST.pkl")
print("\nStructure:")
print("  {")
print("    '0': {")
print("      'embedding': [1024D vector],")
print("      'metadata': {")
print("        'case_id': '...',")
print("        'court_level': '...',")
print("        'court_name': '...',")
print("        'year': 2019,")
print("        'case_types': ['Civil', 'Tax'],")
print("        'original_label': '1'")
print("      }")
print("    },")
print("    ...")
print("  }")
