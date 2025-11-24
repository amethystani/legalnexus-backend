"""
Inspect actual case data to see what we're working with
"""
import pickle
import pandas as pd

# Load citation network
with open('data/citation_network.pkl', 'rb') as f:
    data = pickle.load(f)

case_ids = data['case_ids']
metadata = data['metadata']

print("="*80)
print("INSPECTING CASE DATA")
print("="*80)
print(f"\nTotal cases: {len(case_ids)}")
print(f"\nFirst 5 case IDs:")
for i, cid in enumerate(case_ids[:5]):
    print(f"{i+1}. {cid}")
    print(f"   Metadata: {metadata.get(cid, {})}")

# Load CSV to see actual text
print("\n\nLoading CSV to see case text...")
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print(f"\nFirst case from CSV:")
print(f"ID: {df.iloc[0]['case_id']}")
print(f"Label: {df.iloc[0]['label']}")
print(f"Text (first 500 chars):")
print(df.iloc[0]['text'][:500])

print(f"\n\nSearching for a specific case with keywords...")
# Find a case with specific keywords
keywords = ['accident', 'negligence', 'compensation', 'motor']
for keyword in keywords:
    matches = df[df['text'].str.contains(keyword, case=False, na=False)]
    if len(matches) > 0:
        print(f"\n✓ Found {len(matches)} cases with '{keyword}'")
        print(f"\nSample case:")
        sample = matches.iloc[0]
        print(f"ID: {sample['case_id']}")
        print(f"Text (first 700 chars):")
        print(sample['text'][:700])
        break
