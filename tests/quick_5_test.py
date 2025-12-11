"""
Quick test: Generate embeddings for 5 cases only
"""
import pickle
import pandas as pd
from jina_embeddings import JinaEmbeddings
import numpy as np

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("QUICK TEST: 5 CASES ONLY")
print("="*80)

# Initialize Jina
print("\n1. Initializing Jina...")
embeddings_model = JinaEmbeddings(
    model_name="models/jina-embeddings-v3",
    task="retrieval.passage"
)

# Generate for first 5 cases
print("\n2. Generating 5 embeddings...")
cache = {}

for idx in range(5):
    row = df.iloc[idx]
    case_id = row['case_id']
    text = str(row['text'])[:8000]
    
    print(f"   Case {idx}: {case_id[:50]}")
    emb = embeddings_model.embed_query(text)
    cache[str(idx)] = emb

print(f"\n3. Verifying diversity...")
for i in range(4):
    emb_a = np.array(cache[str(i)])
    emb_b = np.array(cache[str(i+1)])
    
    diff = np.linalg.norm(emb_a - emb_b)
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    
    print(f"   Case {i} vs {i+1}: distance={diff:.4f}, similarity={sim:.4f}")

# Save
print(f"\n4. Saving...")
with open('data/case_embeddings_5_TEST.pkl', 'wb') as f:
    pickle.dump(cache, f)

print("\n✅ Saved to data/case_embeddings_5_TEST.pkl")
print("\nCase details:")
for idx in range(5):
    row = df.iloc[idx]
    print(f"\n{idx}. {row['case_id']}")
    print(f"   Text preview: {row['text'][:150]}...")
