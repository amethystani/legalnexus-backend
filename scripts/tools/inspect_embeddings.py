
import pickle

cache_path = 'data/case_embeddings_cache.pkl'
try:
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    if isinstance(cache, dict) and 'embeddings' not in cache:
        ids = list(cache.keys())
        print(f"Cache is dict. Total IDs: {len(ids)}")
        print(f"First 10 IDs: {ids[:10]}")
        print(f"All numeric? {all(str(i).isdigit() for i in ids)}")
    else:
        print("Cache format is complex/legacy.")
        
except Exception as e:
    print(f"Error: {e}")
