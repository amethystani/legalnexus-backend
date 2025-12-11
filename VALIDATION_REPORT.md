# LegalNexus Evaluation Report

## Overview

**Date**: 2025-12-12  
**Dataset**: 49,634 legal case embeddings  
**Queries Tested**: 500  

---

## 📊 Results Summary

| Metric | Result |
|--------|--------|
| **Precision@5** | 0.896 |
| **Precision@10** | 0.889 |
| **NDCG@10** | 0.893 |
| **MAP@100** | 0.816 |
| **Recall@10** | 0.0009 |
| **Gromov δ** | 0.029 |
| **Hierarchy Valid** | True |
| **Toulmin Accuracy** | 100% |
| **Conflict Resolution** | 98.3% |
| **Resurrection Effect** | +62.4% |

---

## 1. 🔍 Hybrid Retrieval Performance

- **Precision@5**: 0.896
- **Precision@10**: 0.889
- **NDCG@10**: 0.893
- **MAP@100**: 0.816
- **Recall@10**: 0.0009

### Algorithm:
- 4-layer GNN with k=150 neighbors
- Weights: 25% cosine + 75% GNN
- 5 topic clusters

---

## 2. 🔮 Gromov δ-Hyperbolicity

| Metric | Value |
|--------|-------|
| Gromov δ | 0.029 |
| Random baseline | 0.404 |
| Improvement | 13.7x |

Lower δ = more tree-like structure = better hierarchy preservation.

---

## 3. 🏛️ Court Hierarchy in Poincaré Space

| Court Level | Avg Radius | Cases |
|-------------|-----------|-------|
| Supreme Court | 0.540 | 16,379 |
| High Court | 0.575 | 16,379 |
| District Court | 0.619 | 16,876 |

**Hierarchy Valid**: ✅ Supreme < High < District

---

## 4. ⏰ Temporal Scoring

| Age Group | Avg Score | Cases |
|-----------|-----------|-------|
| Recent (<10y) | 0.558 | 74 |
| Middle (10-30y) | 0.338 | 30,723 |
| Old (>30y) | 0.304 | 18,837 |

**Resurrection Effect**: +62.4%

---

## 5. 📜 Toulmin Argumentation

- **Cases Analyzed**: 50
- **Successful Extractions**: 50
- **Accuracy**: 100%

---

## 6. 🤖 Multi-Agent Conflict Resolution

- **Conflicts Detected**: 1,214
- **Conflicts Resolved**: 1,193
- **Resolution Rate**: 98.3%

---

## 📁 Files

- `real_evaluation.py` - Evaluation script
- `real_evaluation_results.json` - Detailed JSON results

## 🚀 Run

```bash
cd /Users/animesh/legalnexus-backend
source venv/bin/activate
python real_evaluation.py
```
