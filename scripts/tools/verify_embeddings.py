"""
Verify embeddings quality and prepare for next steps
"""
import pickle
import numpy as np

print("="*80)
print("VERIFYING EMBEDDINGS")
print("="*80)

# Load embeddings
with open('data/case_embeddings_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

print(f"\n✓ Loaded {len(cache)} embeddings")

# Check a few samples
if len(cache) > 0:
    # Get first embedding
    first_key = list(cache.keys())[0]
    first_emb = cache[first_key]
    
    print(f"✓ Embedding dimension: {len(first_emb)}")
    print(f"✓ Embedding type: {type(first_emb)}")
    
    # Check diversity (compare first 5)
    if len(cache) >= 5:
        print(f"\nDiversity check (first 5):")
        for i in range(4):
            emb_a = np.array(cache[str(i)])
            emb_b = np.array(cache[str(i+1)])
            
            distance = np.linalg.norm(emb_a - emb_b)
            similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
            
            print(f"  {i} vs {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")
        
        avg_dist = np.mean([np.linalg.norm(np.array(cache[str(i)]) - np.array(cache[str(i+1)])) for i in range(4)])
        if avg_dist > 1.0:
            print(f"\n✅ Embeddings are diverse! (avg distance: {avg_dist:.4f})")
        else:
            print(f"\n⚠️  Embeddings may be too similar (avg distance: {avg_dist:.4f})")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Create synthetic citation network with new embeddings
2. Train Hyperbolic GNN with valid embeddings
3. Generate visualizations
4. Test query system with hyperbolic embeddings

Ready to proceed!
""")
else:
    print("\n❌ No embeddings found in cache!")
