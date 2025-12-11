import pandas as pd

# Read the CSV
df = pd.read_csv('data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv')

# Show the first few cases
print("Sample cases from your dataset:\n")
for i in range(min(3, len(df))):
    case_id = df.iloc[i, 0]
    case_text = df.iloc[i, 1]
    print(f"\n{'='*80}")
    print(f"Case {i+1}: {case_id}")
    print(f"{'='*80}")
    print(case_text[:500])
    print(f"\n\nSuggested test query: (Use keywords from above)")
    # Extract first few words as query
    words = case_text.split()[:10]
    print(f"'{' '.join(words)}'")
