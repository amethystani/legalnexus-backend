"""
Clean test: 5 embeddings only, then query test
"""
import pickle
import pandas as pd
import requests
import numpy as np

import time

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, retries=5):
    for attempt in range(retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': 'nomic-embed-text', 'prompt': text[:6000]},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()['embedding']
            else:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
    raise Exception(f"Failed after {retries} attempts")

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("CLEAN EMBEDDING TEST (5 Cases)")
print("="*80)

# Step 1: Generate 5 embeddings
print("\n[1/3] Generating 5 embeddings...")
cache = {}
for idx in range(5):
    case_id = df.iloc[idx]['case_id']
    text = str(df.iloc[idx]['text'])[:2000]  # Reduced from 6000 to avoid crashes
    
    print(f"   {idx}. {case_id[:45]}...", end=" ", flush=True)
    emb = get_embedding(text)
    cache[str(idx)] = emb
    print(f"✓ (dim={len(emb)})")

print(f"\n   Total: {len(cache)} embeddings generated")

# Step 2: Verify diversity
print("\n[2/3] Checking diversity...")
for i in range(4):
    emb_a = np.array(cache[str(i)])
    emb_b = np.array(cache[str(i+1)])
    
    diff = np.linalg.norm(emb_a - emb_b)
    sim = cosine_similarity(emb_a, emb_b)
    
    print(f"   Case {i} vs {i+1}: distance={diff:.4f}, similarity={sim:.4f}")

# Step 3: Query test
print("\n[3/3] Query Retrieval Test...")
test_idx = 2
test_case = df.iloc[test_idx]
test_id = test_case['case_id']
test_text = str(test_case['text'])

print(f"\n   Target Case: {test_id}")
print(f"   Preview: {test_text[:150]}...")

# Create query from middle of the case text
query = test_text[100:400]
print(f"\n   Query (from case): '{query[:100]}...'")

print("\n   Embedding query...")
query_emb = get_embedding(query)

print("   Searching in 5 cases...")
results = []
for idx in range(5):
    case_emb = cache[str(idx)]
    sim = cosine_similarity(query_emb, case_emb)
    case_id = df.iloc[idx]['case_id']
    results.append((idx, case_id, sim))

results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'='*80}")
print("SEARCH RESULTS (Ranked by Similarity)")
print(f"{'='*80}\n")

for rank, (idx, case_id, sim) in enumerate(results, 1):
    marker = " 🎯 TARGET FOUND!" if idx == test_idx else ""
    print(f"{rank}. Case {idx}: {case_id[:50]}")
    print(f"   Similarity: {sim:.4f}{marker}\n")

# Verdict
print(f"{'='*80}")
if results[0][0] == test_idx:
    print(f"✅ SUCCESS! Target case ranked #1 with {results[0][2]:.4f} similarity")
    print("   Nomic embeddings are working correctly!")
elif any(idx == test_idx for idx, _, _ in results[:3]):
    rank = next(i+1 for i, (idx, _, _) in enumerate(results) if idx == test_idx)
    print(f"✅ GOOD! Target case in top 3 (rank #{rank})")
    print("   Embeddings are functional.")
else:
    rank = next((i+1 for i, (idx, _, _) in enumerate(results) if idx == test_idx), "not found")
    print(f"⚠️  Target case at rank #{rank}")
    print("   May need different query or more data.")

print(f"{'='*80}\n")
