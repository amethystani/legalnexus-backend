"""
Generate FULL embedding cache
"""
import pickle
import pandas as pd
from tqdm import tqdm
from jina_embeddings import JinaEmbeddings
import numpy as np

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print(f"GENERATING FULL EMBEDDING CACHE ({len(df)} cases)")
print("="*80)

# Initialize Jina
print("\n1. Initializing Jina embeddings...")
embeddings_model = JinaEmbeddings(
    model_name="models/jina-embeddings-v3",
    task="retrieval.passage"
)

# Generate embeddings with batching
print(f"\n2. Generating {len(df)} embeddings (this will take ~2 hours)...")
cache = {}
batch_size = 1  # Process one at a time to avoid memory issues
failed = []

for idx in tqdm(range(len(df)), desc="Embedding"):
    row = df.iloc[idx]
    text = str(row['text'])[:8000]  # Limit to 8k chars
    
    try:
        emb = embeddings_model.embed_query(text)
        cache[str(idx)] = emb
        
        # Save checkpoint every 1000 cases
        if (idx + 1) % 1000 == 0:
            with open(f'data/case_embeddings_checkpoint_{idx+1}.pkl', 'wb') as f:
                pickle.dump(cache, f)
            print(f"\n   Checkpoint saved at {idx+1} cases")
            
    except Exception as e:
        print(f"\nError embedding case {idx}: {e}")
        failed.append(idx)
        continue

print(f"\n3. Saving final cache ({len(cache)} embeddings)...")
with open('data/case_embeddings_cache.pkl', 'wb') as f:
    pickle.dump(cache, f)

# Verify quality
print("\n4. Verifying quality (spot check)...")
indices_to_check = [0, 100, 1000, 5000]
for i in range(len(indices_to_check)-1):
    idx_a = str(indices_to_check[i])
    idx_b = str(indices_to_check[i+1])
    
    if idx_a in cache and idx_b in cache:
        emb_a = np.array(cache[idx_a])
        emb_b = np.array(cache[idx_b])
        diff = np.linalg.norm(emb_a - emb_b)
        sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        print(f"   Case {idx_a} vs {idx_b}: distance={diff:.4f}, similarity={sim:.4f}")

if failed:
    print(f"\n⚠️  {len(failed)} cases failed to embed")
else:
    print(f"\n✅ All {len(cache)} cases embedded successfully!")

print(f"\n✅ Saved to data/case_embeddings_cache.pkl")
print(f"   Size: {len(cache)} embeddings")
