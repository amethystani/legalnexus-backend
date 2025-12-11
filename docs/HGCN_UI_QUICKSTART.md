# HGCN Hyperbolic Search - Quick Start Guide

## 🎉 Your UI is Now Running!

The HGCN Hyperbolic Search UI is now live at:

**Local URL**: http://localhost:8501

## 🚀 How to Use

### 1. **Search Modes**

The UI offers three ways to search:

#### a) **Search by Case ID**
- Type part of a case ID (e.g., "SupremeCourt_1970")
- Select from matching cases
- Click "🚀 Search Similar Cases"

#### b) **Pick Random Case**
- Click "🎲 Pick Random Case" button
- Automatically selects a random case
- Great for exploring the dataset

#### c) **Browse by Court Level**
- Filter by hierarchy level:
  - Supreme Court (radius < 0.10)
  - High Court (radius 0.10-0.20)
  - Lower Courts (radius 0.20-0.30)
  - District Courts (radius > 0.30)
- Select a case from filtered results

### 2. **View Results**

After clicking search, you'll see:

- **Top N Similar Cases** (configurable in sidebar)
- **Hierarchy Distribution** chart
- **Statistics**: radius, distance, standard deviation
- **Analysis**: Why these results were returned

### 3. **Understand the Results**

Each result shows:
- 🏛️ **Emoji Icon**: Visual indicator of court level
- **Case ID**: Full case identifier
- **Court Level Badge**: Supreme Court, High Court, etc.
- **Distance**: Poincaré distance (lower = more similar)
- **Radius**: Position in hierarchy (lower = higher authority)

## 📊 Features

### Left Panel (Main Search)
- Interactive search with 3 modes
- Real-time case filtering
- Autocomplete for case IDs

### Right Panel (Query Info)
- Query case details
- Hierarchy level indicator
- Radius and embedding info

### Results Section
- Beautiful card-based results
- Hierarchy distribution chart
- Statistics and metrics
- Detailed analysis

### Sidebar
- Model information
- Search settings (adjust top-k)
- Court hierarchy guide
- About section

## 🎨 UI Features

- **Premium Design**: Gradient backgrounds, glassmorphism
- **Interactive**: Hover effects, smooth animations
- **Responsive**: Works on different screen sizes
- **Visual Hierarchy**: Color-coded court levels
- **Real-time Search**: Instant results

## 🎯 Example Queries

Try these interesting cases:

1. **Supreme Court Case**:
   - Search: `SupremeCourt_1970_306`
   - Expected: Other Supreme Court and High Court cases

2. **Random Exploration**:
   - Click "🎲 Pick Random Case"
   - See what similar cases the system finds

3. **Court Level Filtering**:
   - Select "Supreme Court" in browse mode
   - Pick any case to see similar SC cases

## 📈 What to Look For

### Good Results (System Working Well):
✅ Query radius close to result mean radius  
✅ Most results from same or adjacent court levels  
✅ Low radius difference (< 0.05)  
✅ Clustering of similar hierarchy levels  

### Analysis Insights:
- **Lower radius** = Higher court authority
- **Similar radii** = Same court level
- **Poincaré distance** preserves hierarchy
- **Tight clustering** = Strong hierarchy learning

## 🛠️ Settings (Sidebar)

- **Number of results**: 5-50 cases (default: 20)
- Adjust based on how many results you want to see

## 💡 Tips

1. **Try different court levels**: See how hierarchy affects results
2. **Compare radii**: Notice how similar cases have similar radii
3. **Check statistics**: Low std dev = tight clustering
4. **Browse the chart**: Visual hierarchy distribution is insightful

## 🎬 Demo Flow

1. Open http://localhost:8501
2. Select "Pick Random Case"
3. Click "🎲 Pick Random Case" button
4. Click "🚀 Search Similar Cases"
5. Explore the results and statistics!

## ⚙️ To Stop the Server

In your terminal, press:
```
Ctrl + C
```

## 🔄 To Restart

```bash
cd /Users/animesh/legalnexus-backend
source venv/bin/activate
streamlit run hgcn_search_ui.py
```

## 📝 Technical Details

- **Model**: HGCN (Hyperbolic GNN)
- **Cases**: 49,633 legal cases
- **Embedding Dim**: 64
- **Space**: Poincaré Ball
- **Distance**: Poincaré Distance
- **Framework**: Streamlit

## 🌟 Key Advantages

1. **Hierarchy-Aware**: Understands legal precedent structure
2. **Fast**: Efficient 64-D embeddings
3. **Intuitive**: Visual representation of hierarchy
4. **Interactive**: Multiple search modes
5. **Beautiful**: Premium UI design

---

**Enjoy exploring the hyperbolic legal search! ⚖️**
