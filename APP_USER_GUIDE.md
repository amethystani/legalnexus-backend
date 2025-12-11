# ⚖️ LegalNexus Hyperbolic Search - User Guide

## 🚀 Access the App
**URL**: http://localhost:8501

## ✨ Key Features

### 1. **Two Search Modes**
- **Case ID Search (Default)**: 
  - ⚡ **Super Fast** & **Lightweight**
  - Select an existing case to find similar precedents based on hyperbolic geometry.
  - Does **NOT** load the heavy AI model. Safe for all devices.

- **Text Search (Optional)**:
  - 🧠 **AI-Powered**
  - Type queries like *"drunk driving liability"* or *"breach of contract"*.
  - **Requires enabling the toggle in the Sidebar**.
  - ⚠️ **Note**: Loads the Jina AI model (~1GB RAM). Only enable if your system has available memory.

### 2. **Hierarchy Visualization**
- The app visualizes the **Court Hierarchy** of results.
- **Radius Metric**:
  - Lower radius (< 0.10) = **Supreme Court** (Higher Authority)
  - Higher radius (> 0.20) = **Lower Courts**

### 3. **Premium UI**
- Dark mode design for reduced eye strain.
- Interactive result cards with similarity scores.
- Real-time hierarchy distribution charts.

## 🛠️ Troubleshooting

- **"System Slow?"**: Turn off "Enable Text Search" in the sidebar.
- **"No Results?"**: Try a broader query or select a different Case ID.
- **"Error Loading Model?"**: Ensure `sentence-transformers` is installed (`pip install sentence-transformers`).

## 🛑 Stopping the App
Press `Ctrl+C` in the terminal where the app is running.
