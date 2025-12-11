# 🧠 Jina Semantic Search UI - Quick Start

## 🚀 Access the App
**URL**: http://localhost:8501

## ✨ Features
- **Pure Semantic Search**: Uses Jina AI embeddings to find cases with similar *meaning*.
- **Simple Interface**: Just a search bar and results.
- **Memory Optimized**: Designed to run the Jina pipeline with minimal overhead.

## 📝 How to Use
1. **Enter Query**: Type a legal concept like *"medical negligence"* or *"breach of contract"*.
2. **Click Search**: The system encodes your text and finds the top 10 matches.
3. **View Results**: See the Case ID and Similarity Score (0-1).

## ⚠️ Note
- This app loads the **Jina Model (~1GB)**. 
- Initial load takes about 30-60 seconds.
- Subsequent searches are fast (~1-2 seconds).

## 🛑 Stopping the App
Press `Ctrl+C` in the terminal.
