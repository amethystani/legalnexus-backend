"""
Quick quality test - pause main generation temporarily
"""
import pickle
import pandas as pd
import requests
import numpy as np
import time

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_nomic_embedding(text):
    response = requests.post(
        'http://localhost:11434/api/embeddings',
        json={'model': 'nomic-embed-text', 'prompt': text[:6000]},
        timeout=60
    )
    return response.json()['embedding'] if response.status_code == 200 else None

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("QUALITY TEST (5 cases)\n" + "="*60)
print("IMPORTANT: The parallel generation is still running.")
print("This test will take ~30 seconds due to Ollama being busy.\n")

cache = {}
print("Generating 5 embeddings...")
for idx in range(5):
    case_id = df.iloc[idx]['case_id']
    text = str( df.iloc[idx]['text'])[:6000]
    
    print(f"  {idx}: {case_id[:40]}...", end=" ", flush=True)
    
    for attempt in range(3):
        emb = get_nomic_embedding(text)
        if emb:
            cache[str(idx)] = emb
            print("✓")
            break
        time.sleep(2)
    else:
        print("✗ (failed)")

if len(cache) < 5:
    print(f"\n❌ Only got {len(cache)}/5 embeddings. Ollama is too busy.")
    print("Wait for checkpoint or stop parallel generation to test.")
    exit(1)

print(f"\nDiversity check...")
diffs = []
for i in range(4):
    diff = np.linalg.norm(np.array(cache[str(i)]) - np.array(cache[str(i+1)]))
    diffs.append(diff)
    print(f"  {i} vs {i+1}: {diff:.3f}")

avg_diff = np.mean(diffs)
print(f"\n{'✅' if avg_diff > 0.5 else '❌'} Avg distance: {avg_diff:.3f} ({'OK' if avg_diff > 0.5 else 'TOO SIMILAR'})")

# Query test
test_idx = 2
query = str(df.iloc[test_idx]['text'])[100:300]
print(f"\nQuery test (case {test_idx})...")
query_emb = get_nomic_embedding(query)

if query_emb:
    results = [(idx, cosine_similarity(query_emb, cache[str(idx)])) for idx in range(5)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("Results:")
    for rank, (idx, sim) in enumerate(results, 1):
        marker = " 🎯" if idx == test_idx else ""
        print(f"  {rank}. Case {idx}: {sim:.4f}{marker}")
    
    if results[0][0] == test_idx:
        print(f"\n✅ SUCCESS! Quality is good (sim: {results[0][1]:.4f})")
    else:
        print(f"\n⚠️  Target at rank {next((i+1 for i, (idx, _) in enumerate(results) if idx == test_idx), '?')}")
else:
    print("❌ Query embedding failed")

print("\n" + "="*60)
