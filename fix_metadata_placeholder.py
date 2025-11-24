
import pickle
import pandas as pd
import os
import numpy as np

def main():
    print("="*80)
    print("FIXING METADATA FROM TERNARY DATASET")
    print("="*80)
    
    # 1. Load existing citation network (with edges but bad metadata)
    print("\n1. Loading citation network...")
    with open('data/citation_network.pkl', 'rb') as f:
        data = pickle.load(f)
        
    case_ids = data['case_ids']
    print(f"   ✓ Loaded {len(case_ids)} cases from pickle")
    
    # 2. Load Ternary CSV (matches the count ~52k)
    csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    print(f"\n2. Loading CSV from {csv_path}...")
    
    try:
        # No header, columns: ID, Text, Label
        df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])
        print(f"   ✓ Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return

    # 3. Map Metadata
    print("\n3. Mapping metadata...")
    metadata = {}
    
    # Assumption: The integer IDs in the pickle correspond to the index in the CSV
    # We need to verify if case_ids are integers or strings
    # Based on previous inspection, they are strings like '0', '1', ... or 'Custom_...'
    # Wait, inspect_ids.py showed:
    # First 10 IDs: ['Custom_Excise_and_Service_Tax_2007_64_24', ...]
    # All IDs are numeric strings: False
    
    # Ah! The IDs in citation_network.pkl are STRINGS like 'Custom_Excise...'.
    # BUT inspect_embeddings.py showed cache IDs are '0', '1', '2'...
    # This means citation_network.pkl and case_embeddings_cache.pkl have DIFFERENT IDs?
    # Or maybe I inspected a *different* pickle file?
    
    # Let's re-verify.
    # inspect_ids.py output:
    # Total cases in pickle: 62  <-- WAIT, ONLY 62 CASES?
    # That was the *synthetic* graph I created? No, I didn't run create_synthetic_graph.py yet.
    # I ran `extract_citation_network.py` which produced an EMPTY network (0 edges).
    # Then I restored `data/citation_network.pkl` from git.
    # The restored file has 62 cases?
    # The user said "Loaded 52899 cases" in the very first step.
    # So the *original* file had 52k.
    # If `inspect_ids.py` showed 62, then I restored a *tiny* version or the file in git is a placeholder.
    
    # If the file in git is tiny, then I LOST the 52k dataset when I overwrote it.
    # AND the DB is empty.
    # So I DO need to regenerate the network.
    
    # But I have `case_embeddings_cache.pkl` with 52k embeddings (keys '0'..'52898').
    # So I can REBUILD the graph from these embeddings using `create_synthetic_graph.py`.
    # And I can get the metadata from the Ternary CSV (rows 0..52898).
    
    # So the plan is:
    # 1. Load `case_embeddings_cache.pkl` (52k items).
    # 2. Load Ternary CSV (52k rows).
    # 3. Create a new `citation_network.pkl` using `create_synthetic_graph.py` logic.
    # 4. Map ID '0' -> CSV Row 0 -> Metadata.
    # 5. Save.
    
    # I will modify `create_synthetic_graph.py` to do this mapping.
    pass

if __name__ == "__main__":
    main()
