"""
Create Synthetic Citation Network from Embeddings (MEMORY-OPTIMIZED)

Uses k-nearest neighbors in BATCHES to avoid memory crashes
"""

import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import os
from tqdm import tqdm
import time

def main():
    print("="*80)
    print("CREATING SYNTHETIC CITATION NETWORK (Memory-Optimized for 8GB RAM)")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load embeddings
    print("\n[1/6] Loading embeddings...")
    with open('data/case_embeddings_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    # Sort by integer ID
    int_keys = sorted([k for k in cache.keys() if str(k).isdigit()], key=lambda x: int(x))
    embeddings = np.array([cache[k] for k in int_keys], dtype=np.float32)
    
    print(f"   ✓ Loaded {len(int_keys)} embeddings (768D)")
    print(f"   Memory: ~{embeddings.nbytes / (1024**2):.1f} MB")
    
    # 2. Load CSV
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    print(f"\n[2/6] Loading CSV...")
    import pandas as pd
    df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
    print(f"   ✓ Loaded {len(df)} rows")
    
    # 3. Map metadata
    print("\n[3/6] Mapping metadata...")
    index_to_meta = {}
    real_case_ids = []
    
    for idx in tqdm(range(len(int_keys)), desc="   Mapping"):
        if idx < len(df):
            row = df.iloc[idx]
            real_id = str(row['case_id'])
            
            # Infer court
            court = 'Unknown'
            cid_lower = real_id.lower()
            if 'sc_' in cid_lower or 'supreme' in cid_lower or 'sci' in cid_lower:
                court = 'Supreme Court'
            elif 'hc' in cid_lower or 'high' in cid_lower:
                court = 'High Court'
            else:
                court = 'Lower Court'
                
            index_to_meta[idx] = {'id': real_id, 'court': court}
            real_case_ids.append(real_id)
        else:
            real_id = f"unknown_{idx}"
            index_to_meta[idx] = {'id': real_id, 'court': 'Unknown'}
            real_case_ids.append(real_id)
    
    # 4. Find k-nearest neighbors IN BATCHES
    print(f"\n[4/6] Finding nearest neighbors (k=10) in batches...")
    k = 10
    
    # Fit the model once
    print("   Building KNN index...")
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=4)
    knn.fit(embeddings)
    
    # Process queries in batches to avoid memory overflow
    BATCH_SIZE = 1000  # Process 1000 cases at a time
    n_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"   Processing {len(embeddings)} cases in {n_batches} batches...")
    print(f"   Est. time: ~{n_batches * 2} seconds")
    
    all_indices = []
    all_distances = []
    
    for batch_idx in tqdm(range(n_batches), desc="   Batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(embeddings))
        
        batch_emb = embeddings[start_idx:end_idx]
        distances, indices = knn.kneighbors(batch_emb)
        
        all_distances.append(distances)
        all_indices.append(indices)
    
    # Concatenate all results
    all_distances = np.vstack(all_distances)
    all_indices = np.vstack(all_indices)
    
    # 5. Build edge list
    print(f"\n[5/6] Building edge list...")
    edges = []
    for i in tqdm(range(len(all_indices)), desc="   Edges"):
        src_id = index_to_meta[i]['id']
        for j in all_indices[i][1:]:  # Skip first (self)
            tgt_id = index_to_meta[j]['id']
            edges.append((src_id, tgt_id))
            
    print(f"   ✓ Created {len(edges)} citation edges")
    
    # 6. Build sparse adjacency matrix
    print(f"\n[6/6] Building sparse adjacency matrix...")
    n = len(real_case_ids)
    id_to_idx = {cid: idx for idx, cid in enumerate(real_case_ids)}
    
    adj = sparse.lil_matrix((n, n), dtype=np.float32)
    for src, tgt in tqdm(edges, desc="   Adjacency"):
        s_i = id_to_idx[src]
        t_i = id_to_idx[tgt]
        adj[s_i, t_i] = 1
        adj[t_i, s_i] = 1
        
    adj = adj.tocsr()
    
    # Metadata
    metadata = {info['id']: {'court': info['court'], 'date': None} for info in index_to_meta.values()}
    
    # Save
    print(f"\nSaving citation network...")
    output = {
        'case_ids': real_case_ids,
        'adj': adj,
        'metadata': metadata,
        'embeddings': embeddings
    }
    
    with open('data/citation_network.pkl', 'wb') as f:
        pickle.dump(output, f)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"Total cases: {n}")
    print(f"Total edges: {len(edges)}")
    print(f"Avg degree: {len(edges) / n:.1f}")
    print(f"Time taken: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nSaved to: data/citation_network.pkl")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
