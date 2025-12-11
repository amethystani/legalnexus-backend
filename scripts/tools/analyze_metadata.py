"""
Analyze what labels/metadata we can extract from the dataset
"""
import pandas as pd
import re
from collections import Counter

# Load CSV
csv_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
df = pd.read_csv(csv_path, header=None, names=['case_id', 'text', 'label'])

print("="*80)
print("CURRENT LABELS & EXTRACTABLE METADATA")
print("="*80)

# 1. Existing label column
print("\n1. EXISTING LABEL COLUMN:")
print(f"   Unique labels: {df['label'].unique()}")
print(f"   Label distribution:")
for label, count in df['label'].value_counts().items():
    print(f"     {label}: {count} cases")

# 2. Court level (already extracted)
print("\n2. COURT LEVEL (from case_id):")
def extract_court(case_id):
    cid = str(case_id).lower()
    if 'sc_' in cid or 'supreme' in cid or 'sci' in cid:
        return 'Supreme Court'
    elif 'hc' in cid or 'high' in cid:
        return 'High Court'
    else:
        return 'Lower Court/Tribunal'

df['court'] = df['case_id'].apply(extract_court)
print(df['court'].value_counts())

# 3. Case type (can be inferred from text keywords)
print("\n3. POTENTIAL CASE TYPES (from text analysis):")
case_types = {
    'Criminal': ['criminal', 'accused', 'murder', 'theft', 'assault', 'ipc', 'crpc'],
    'Civil': ['civil', 'damages', 'contract', 'property', 'suit', 'cpc'],
    'Tax': ['income tax', 'gst', 'sales tax', 'customs', 'excise'],
    'Consumer': ['consumer', 'deficiency', 'compensation for service'],
    'Constitutional': ['constitutional', 'fundamental right', 'article', 'writ'],
    'Company': ['company', 'shareholders', 'directors', 'board'],
    'Family': ['divorce', 'custody', 'maintenance', 'matrimonial'],
    'Labour': ['labour', 'employee', 'industrial dispute', 'termination']
}

# Sample 100 cases
sample = df.sample(min(100, len(df)))
type_counts = Counter()

for _, row in sample.iterrows():
    text_lower = str(row['text']).lower()
    for case_type, keywords in case_types.items():
        if any(kw in text_lower for kw in keywords):
            type_counts[case_type] += 1
            break

print("   (Based on 100 sample cases)")
for case_type, count in type_counts.most_common():
    print(f"     {case_type}: {count} cases")

# 4. Extractable from case_id
print("\n4. TRIBUNAL/COURT NAME (from case_id):")
tribunals = df['case_id'].str.extract(r'([A-Za-z_]+)_')[0].value_counts().head(10)
print("   Top 10 tribunals/courts:")
for tribunal, count in tribunals.items():
    print(f"     {tribunal}: {count}")

# 5. Year (if present in case_id)
print("\n5. YEAR (from case_id):")
years = df['case_id'].str.extract(r'_(\d{4})_')[0]
if years.notna().any():
    year_counts = years.value_counts().head(10)
    print("   Top 10 years:")
    for year, count in year_counts.items():
        print(f"     {year}: {count}")
else:
    print("   No year info in case_id")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("""
We can automatically extract:
✅ Court Level (Supreme/High/Lower) - ALREADY DONE
✅ Tribunal/Court Name (from case_id)
✅ Year (from case_id if available)
✅ Basic Case Type (using keyword matching)

For MORE DETAILED labels (e.g., specific legal topics), you can:
1. Use an LLM (Gemini/GPT) to analyze text and extract:
   - Case type (Criminal/Civil/Tax/etc.)
   - Legal topics (Contract law, Property law, etc.)
   - Parties involved (Individual vs Company, etc.)
   - Outcome (Dismissed, Allowed, etc.)
   
2. This would require an additional processing step after embedding generation.
3. Cost: ~$0.01 per case with GPT-4 or free with local Ollama

Would you like me to create a metadata extraction pipeline using LLM?
""")
