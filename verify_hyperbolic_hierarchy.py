"""
Verify Hyperbolic Hierarchy Hypothesis

Research Question:
Do hyperbolic embeddings naturally capture the legal hierarchy?
Hypothesis: Supreme Court cases should have significantly smaller radii (closer to origin)
than High Court or Tribunal cases.

This script performs a statistical analysis to validate this hypothesis.
"""

import pickle
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os

def verify_hierarchy():
    print("="*80)
    print("HYPERBOLIC HIERARCHY VERIFICATION")
    print("="*80)
    
    # 1. Load Data
    print("\n1. Loading data...")
    
    # Load embeddings
    if not os.path.exists('models/hgcn_embeddings.pkl'):
        print("❌ Error: Embeddings not found. Run train_hyperbolic.py first.")
        return
        
    with open('models/hgcn_embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    # Load metadata (try multiple sources)
    metadata = {}
    if os.path.exists('models/hgcn_best.pt'):
        checkpoint = torch.load('models/hgcn_best.pt', map_location='cpu')
        metadata = checkpoint.get('metadata', {})
    elif os.path.exists('data/citation_network.pkl'):
        with open('data/citation_network.pkl', 'rb') as f:
            data = pickle.load(f)
            metadata = data.get('metadata', {})
            
    if not metadata:
        print("❌ Error: Could not load case metadata.")
        return
        
    print(f"   ✓ Loaded {len(embeddings_dict)} embeddings")
    print(f"   ✓ Loaded metadata for {len(metadata)} cases")
    
    # 2. Calculate Radii
    print("\n2. Calculating hyperbolic radii...")
    
    case_data = []
    
    for case_id, embedding in embeddings_dict.items():
        if case_id in metadata:
            # Calculate radius (L2 norm)
            radius = np.linalg.norm(embedding)
            
            # Get court type
            court = metadata[case_id].get('court', 'Unknown')
            
            # Normalize court names
            court_type = "Other"
            if "Supreme Court" in court or "SC" in court:
                court_type = "Supreme Court"
            elif "High Court" in court or "HC" in court:
                court_type = "High Court"
            elif "Tribunal" in court or "Commission" in court:
                court_type = "Tribunal"
            
            case_data.append({
                'id': case_id,
                'radius': radius,
                'court': court,
                'court_type': court_type
            })
    
    df = pd.DataFrame(case_data)
    
    # 3. Statistical Analysis
    print("\n3. Statistical Analysis by Court Level:")
    
    # Group by court type
    summary = df.groupby('court_type')['radius'].agg(['mean', 'std', 'count'])
    print(summary)
    
    # T-test: Supreme Court vs High Court
    sc_radii = df[df['court_type'] == 'Supreme Court']['radius']
    hc_radii = df[df['court_type'] == 'High Court']['radius']
    
    if len(sc_radii) > 1 and len(hc_radii) > 1:
        t_stat, p_val = stats.ttest_ind(sc_radii, hc_radii, equal_var=False)
        
        print(f"\n   Hypothesis Test (Supreme Court vs High Court):")
        print(f"     Mean SC Radius: {sc_radii.mean():.4f}")
        print(f"     Mean HC Radius: {hc_radii.mean():.4f}")
        print(f"     T-statistic:    {t_stat:.4f}")
        print(f"     P-value:        {p_val:.4e}")
        
        if p_val < 0.05 and t_stat < 0:
            print("\n   ✅ RESULT: SIGNIFICANT. Supreme Court cases are significantly closer to the center.")
            print("      This confirms the hyperbolic hierarchy hypothesis.")
        else:
            print("\n   ⚠️ RESULT: NOT SIGNIFICANT. Hierarchy not clearly captured.")
    else:
        print("\n   ⚠️ Not enough data for T-test.")
        
    # 4. Visualization (ASCII Plot)
    print("\n4. Radius Distribution (ASCII Plot):")
    
    def ascii_hist(data, label):
        if len(data) == 0: return
        hist, bins = np.histogram(data, bins=10, range=(0, 1))
        max_h = max(hist)
        print(f"\n   {label} (n={len(data)}):")
        for h, b in zip(hist, bins):
            bar = '#' * int(20 * h / max_h)
            print(f"     {b:.1f}-{b+0.1:.1f}: {bar} ({h})")
            
    ascii_hist(sc_radii, "Supreme Court")
    ascii_hist(hc_radii, "High Court")
    
    # Save results to file for paper
    with open('hierarchy_analysis_results.txt', 'w') as f:
        f.write("Hyperbolic Hierarchy Analysis\n")
        f.write("=============================\n\n")
        f.write(str(summary))
        f.write(f"\n\nT-test (SC vs HC): t={t_stat:.4f}, p={p_val:.4e}\n")

if __name__ == "__main__":
    verify_hierarchy()
