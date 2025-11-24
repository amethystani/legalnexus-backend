"""
Unified Data Loader

Single source of truth for loading legal case data from CSVs.
Used by embedding generation, graph building, and search systems.
"""

import pandas as pd
from langchain.schema import Document
import os

def load_all_cases():
    """
    Load all legal cases from CSV datasets.
    
    Returns:
        List[Document]: List of LangChain Document objects
    """
    cases = []
    
    # Dataset Paths
    binary_path = "data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv"
    ternary_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
    
    print("="*60)
    print("LOADING DATA FROM CSVs")
    print("="*60)
    
    # 1. Load Binary Dataset
    if os.path.exists(binary_path):
        try:
            # CSV has no header: ID, Text, Label
            df = pd.read_csv(binary_path, header=None, names=['case_id', 'text', 'label'])
            print(f"✓ Found Binary Dataset: {len(df)} rows")
            
            for _, row in df.iterrows():
                case_id = str(row['case_id'])
                
                # Infer court from ID
                court = 'Unknown'
                cid_lower = case_id.lower()
                if 'sc_' in cid_lower or 'supreme' in cid_lower or 'sci' in cid_lower:
                    court = 'Supreme Court'
                elif 'hc' in cid_lower or 'high' in cid_lower:
                    court = 'High Court'
                elif 'tribunal' in cid_lower:
                    court = 'Lower Court'
                else:
                    court = 'Lower Court' # Default
                
                # Construct text content
                text = f"Title: {case_id}\n"
                text += f"Court: {court}\n"
                text += f"Text: {row.get('text', '')}"
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "id": case_id,
                        "title": case_id,
                        "court": court,
                        "date": "Unknown", # Date not in CSV
                        "source": "binary_dataset"
                    }
                )
                cases.append(doc)
        except Exception as e:
            print(f"❌ Error loading binary dataset: {e}")
    else:
        print(f"⚠️ Binary dataset not found at {binary_path}")

    # 2. Load Ternary Dataset
    if os.path.exists(ternary_path):
        try:
            # Assuming same format: ID, Text, Label
            df = pd.read_csv(ternary_path, header=None, names=['case_id', 'text', 'label'])
            print(f"✓ Found Ternary Dataset: {len(df)} rows")
            
            for _, row in df.iterrows():
                case_id = str(row['case_id'])
                
                # Avoid duplicates
                if any(c.metadata['id'] == case_id for c in cases):
                    continue
                
                # Infer court
                court = 'Unknown'
                cid_lower = case_id.lower()
                if 'sc_' in cid_lower or 'supreme' in cid_lower or 'sci' in cid_lower:
                    court = 'Supreme Court'
                elif 'hc' in cid_lower or 'high' in cid_lower:
                    court = 'High Court'
                else:
                    court = 'Lower Court'

                text = f"Title: {case_id}\n"
                text += f"Court: {court}\n"
                text += f"Text: {row.get('text', '')}"
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "id": case_id,
                        "title": case_id,
                        "court": court,
                        "date": "Unknown",
                        "source": "ternary_dataset"
                    }
                )
                cases.append(doc)
        except Exception as e:
            print(f"❌ Error loading ternary dataset: {e}")
    else:
        print(f"⚠️ Ternary dataset not found at {ternary_path}")
        
    print(f"✅ Total Unique Cases Loaded: {len(cases)}")
    return cases

if __name__ == "__main__":
    load_all_cases()
