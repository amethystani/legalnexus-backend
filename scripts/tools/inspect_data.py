
import pickle
import random

data_path = 'data/citation_network.pkl'
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    case_ids = data['case_ids']
    metadata = data['metadata']
    
    print(f"Total cases: {len(case_ids)}")
    
    # Sample some cases
    sample_indices = random.sample(range(len(case_ids)), min(10, len(case_ids)))
    print("\nSample cases and their metadata:")
    for idx in sample_indices:
        cid = case_ids[idx]
        meta = metadata.get(cid, 'Missing')
        print(f"ID: {cid} -> Metadata: {meta}")
        
    # Check court distribution in current file
    court_counts = {}
    for cid, meta in metadata.items():
        court = meta.get('court', 'Missing')
        court_counts[court] = court_counts.get(court, 0) + 1
        
    print("\nCurrent Court Distribution in PKL:")
    for court, count in court_counts.items():
        print(f"  {court}: {count}")

except Exception as e:
    print(f"Error reading file: {e}")
