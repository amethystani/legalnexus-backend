
import pickle

data_path = 'data/citation_network.pkl'
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    case_ids = data['case_ids']
    print(f"Total cases in pickle: {len(case_ids)}")
    print(f"First 10 IDs: {case_ids[:10]}")
    print(f"Last 10 IDs: {case_ids[-10:]}")
    
    # Check if they are all integers
    all_ints = all(str(cid).isdigit() for cid in case_ids)
    print(f"All IDs are numeric strings: {all_ints}")
    
    # Check if they are sequential 0..N
    int_ids = [int(cid) for cid in case_ids if str(cid).isdigit()]
    if len(int_ids) == len(case_ids):
        print(f"Min ID: {min(int_ids)}, Max ID: {max(int_ids)}")
        print(f"Unique IDs: {len(set(int_ids))}")
    
except Exception as e:
    print(f"Error: {e}")
