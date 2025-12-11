"""
Chunked Embedding Strategy: Split long texts, embed each chunk, average embeddings
"""
import pickle
import pandas as pd
import requests
import numpy as np
import time

def get_embedding(text, retries=3):
    """Get single chunk embedding"""
    for attempt in range(retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': 'nomic-embed-text', 'prompt': text},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()['embedding']
            time.sleep(1)
        except:
            if attempt < retries - 1:
                time.sleep(1)
    return None

def get_chunked_embedding(text, chunk_size=2000):
    """
    Split text into chunks, embed each, and average them.
    This preserves full text information while avoiding Ollama crashes.
    """
    # Split into chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
    
    if not chunks:
        return None
    
    # Embed each chunk
    chunk_embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb:
            chunk_embeddings.append(np.array(emb))
    
    if not chunk_embeddings:
        return None
    
    # Average all chunk embeddings
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    return avg_embedding.tolist()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("CHUNKED EMBEDDING TEST (Full Text with 2k chunks)")
print("="*80)

# Test with 5 cases using FULL text
print("\n[1/3] Generating chunked embeddings (full text)...")
cache = {}
for idx in range(5):
    case_id = df.iloc[idx]['case_id']
    full_text = str(df.iloc[idx]['text'])  # FULL TEXT, no truncation
    
    # Calculate chunks
    num_chunks = (len(full_text) + 1999) // 2000
    
    print(f"   {idx}. {case_id[:40]} (len={len(full_text)}, chunks={num_chunks})...", end=" ", flush=True)
    
    emb = get_chunked_embedding(full_text, chunk_size=2000)
    if emb:
        cache[str(idx)] = emb
        print(f"✓")
    else:
        print(f"✗")

print(f"\n   Total: {len(cache)}/5 embeddings")

if len(cache) < 5:
    print("\n❌ Some embeddings failed. Check Ollama.")
    exit(1)

# Diversity check
print("\n[2/3] Checking diversity...")
for i in range(4):
    diff = np.linalg.norm(np.array(cache[str(i)]) - np.array(cache[str(i+1)]))
    sim = cosine_similarity(cache[str(i)], cache[str(i+1)])
    print(f"   Case {i} vs {i+1}: distance={diff:.4f}, similarity={sim:.4f}")

# Query test
print("\n[3/3] Query test...")
test_idx = 2
test_text = str(df.iloc[test_idx]['text'])
query = test_text[100:300]

print(f"   Query: '{query[:80]}...'")
print(f"   Embedding query (chunked)...")

query_emb = get_chunked_embedding(query, chunk_size=2000)

if query_emb:
    results = [(idx, cosine_similarity(query_emb, cache[str(idx)])) for idx in range(5)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    
    for rank, (idx, sim) in enumerate(results, 1):
        case_id = df.iloc[idx]['case_id']
        marker = " 🎯" if idx == test_idx else ""
        print(f"{rank}. Case {idx} ({case_id[:45]}): {sim:.4f}{marker}")
    
    print(f"\n{'='*80}")
    if results[0][0] == test_idx:
        print(f"✅ SUCCESS! Target ranked #1 ({results[0][1]:.4f})")
        print("   Chunked embeddings preserve full text and work perfectly!")
    else:
        rank = next((i+1 for i, (idx, _) in enumerate(results) if idx == test_idx), "?")
        print(f"⚠️  Target at rank #{rank}")
    print(f"{'='*80}\n")
else:
    print("❌ Query embedding failed")
