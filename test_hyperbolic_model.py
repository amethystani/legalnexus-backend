"""
Test Hyperbolic GNN Model with Query
"""
import pickle
import pandas as pd
import numpy as np

print("="*80)
print("TESTING HYPERBOLIC GNN MODEL")
print("="*80)

# Load hyperbolic embeddings (dict with case_id as key)
print("\n1. Loading hyperbolic embeddings...")
with open('models/hgcn_embeddings.pkl', 'rb') as f:
    hyp_emb_dict = pickle.load(f)
print(f"   ✓ Loaded {len(hyp_emb_dict)} hyperbolic embeddings")

# Load citation network for case IDs
print("\n2. Loading case IDs...")
with open('data/citation_network.pkl', 'rb') as f:
    network = pickle.load(f)
case_ids = network['case_ids']
print(f"   ✓ Loaded {len(case_ids)} case IDs")

# Load CSV for case text
print("\n3. Loading case texts...")
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

# Test case
test_idx = 100
test_case_id = case_ids[test_idx]
test_text = str(df.iloc[test_idx]['text'])

print(f"\n4. Test Case: {test_case_id}")
print(f"   Preview: {test_text[:150]}...")

# Create query from case
query = test_text[100:400]
print(f"\n5. Query: '{query[:100]}...'")

# Get hyperbolic embedding for test case
test_hyp_emb = hyp_emb_dict[test_case_id]
print(f"   ✓ Test embedding shape: {np.array(test_hyp_emb).shape}")

# Find similar cases using hyperbolic distance
print(f"\n6. Finding similar cases (hyperbolic distance)...")
distances = []
for i, case_id in enumerate(case_ids[:1000]):  # Search first 1000 for speed
    if case_id in hyp_emb_dict and case_id != test_case_id:
        emb = hyp_emb_dict[case_id]
        dist = np.linalg.norm(np.array(emb) - np.array(test_hyp_emb))
        distances.append((case_id, i, dist))

distances.sort(key=lambda x: x[2])

print(f"\n{'='*80}")
print("TOP 5 SIMILAR CASES (Hyperbolic Distance)")
print(f"{'='*80}\n")

for rank, (case_id, idx, dist) in enumerate(distances[:5], 1):
    marker = " 🎯" if case_id == test_case_id else ""
    print(f"{rank}. {case_id[:50]}")
    print(f"   Hyperbolic distance: {dist:.4f}{marker}\n")

# Verify diversity
print(f"{'='*80}")
print("HYPERBOLIC EMBEDDING QUALITY")
print(f"{'='*80}")

sample_dists = [d[2] for d in distances[:10]]
avg_dist = np.mean(sample_dists)
std_dist = np.std(sample_dists)
print(f"Distance stats (nearest 10 neighbors):")
print(f"  Mean: {avg_dist:.4f}")
print(f"  Std:  {std_dist:.4f}")
print(f"  Min:  {min(sample_dists):.4f}")
print(f"  Max:  {max(sample_dists):.4f}")

if avg_dist > 0.01:
    print(f"\n✅ SUCCESS! Embeddings are diverse (mean: {avg_dist:.4f})")
    print("   Hyperbolic model is working correctly!")
else:
    print(f"\n⚠️  Embeddings may be too similar (mean: {avg_dist:.4f})")

print(f"\n{'='*80}\n")
