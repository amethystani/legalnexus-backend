"""
Test Nomic Embedding Quality (5 cases)
"""
import pickle
import pandas as pd
import requests
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_nomic_embedding(text):
    """Get embedding from Ollama"""
    response = requests.post(
        'http://localhost:11434/api/embeddings',
        json={
            'model': 'nomic-embed-text',
            'prompt': text[:6000]
        },
        timeout=60
    )
    if response.status_code == 200:
        return response.json()['embedding']
    else:
        raise Exception(f"Error: {response.text}")

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("NOMIC EMBEDDING QUALITY TEST (5 Cases)")
print("="*80)

# Generate embeddings for first 5 cases
print("\n1. Generating 5 embeddings...")
cache = {}
for idx in range(5):
    row = df.iloc[idx]
    case_id = row['case_id']
    text = str(row['text'])[:6000]
    
    print(f"   Case {idx}: {case_id[:50]}")
    emb = get_nomic_embedding(text)
    cache[str(idx)] = emb

print(f"\n2. Verifying diversity...")
for i in range(4):
    emb_a = np.array(cache[str(i)])
    emb_b = np.array(cache[str(i+1)])
    
    diff = np.linalg.norm(emb_a - emb_b)
    sim = cosine_similarity(emb_a, emb_b)
    
    print(f"   Case {i} vs {i+1}: distance={diff:.4f}, similarity={sim:.4f}")

if all(cosine_similarity(cache[str(i)], cache[str(i+1)]) < 0.99 for i in range(4)):
    print("   ✅ Embeddings are diverse!")
else:
    print("   ❌ Embeddings may be too similar")

# Test query retrieval
print("\n3. Query Test...")
test_idx = 2
test_case = df.iloc[test_idx]
test_text = str(test_case['text'])
test_id = test_case['case_id']

# Extract query from middle of text
query = test_text[100:300]
print(f"\n   Test Case: {test_id}")
print(f"   Query: '{query[:80]}...'")

# Embed query
print("\n   Embedding query...")
query_emb = get_nomic_embedding(query)

# Search
print("   Searching...")
results = []
for idx in range(5):
    case_emb = cache[str(idx)]
    sim = cosine_similarity(query_emb, case_emb)
    results.append((idx, sim))

results.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

for rank, (idx, sim) in enumerate(results, 1):
    case_id = df.iloc[idx]['case_id']
    marker = " 🎯 TARGET" if idx == test_idx else ""
    print(f"{rank}. Case {idx} ({case_id[:40]}): {sim:.4f}{marker}")

if results[0][0] == test_idx:
    print(f"\n✅ SUCCESS! Target case ranked #1 (similarity: {results[0][1]:.4f})")
    print("\nNomic embeddings are working correctly!")
else:
    target_rank = next((i+1 for i, (idx, _) in enumerate(results) if idx == test_idx), None)
    print(f"\n⚠️  Target ranked #{target_rank}")
    print("Embeddings may need tuning.")

print("\n" + "="*60)
