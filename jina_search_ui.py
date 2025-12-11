"""
Jina Semantic Search UI
Run with: streamlit run jina_search_ui.py
"""
import streamlit as st
import pickle
import numpy as np
import time
import sys
import os
import gc
import torch

# Page Config
st.set_page_config(
    page_title="Jina Semantic Search",
    page_icon="🧠",
    layout="centered"
)

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stTextInput > div > div > input { background-color: #262730; color: white; }
    .result-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #4e8cff;
    }
    .similarity-score { color: #4e8cff; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🧠 MODEL LOADING
# -----------------------------------------------------------------------------

@st.cache_resource
def load_jina_resources():
    """Load Jina model and embeddings cache."""
    status = st.empty()
    status.info("⏳ Loading Jina Model... (This may take a minute)")
    
    # 1. Load Model
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from jina_embeddings_simple import JinaEmbeddingsSimple
        
        # Memory optimization
        torch.set_num_threads(2) 
        gc.collect()
        
        model = JinaEmbeddingsSimple(model_path="models/jina-embeddings-v3")
        status.info("⏳ Loading Embeddings Cache...")
    except Exception as e:
        st.error(f"Failed to load Jina model: {e}")
        return None, None

    # 2. Load Embeddings
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            embeddings_cache = pickle.load(f)
        
        # Convert to list for faster iteration if needed, or keep as dict
        # Dict is fine for lookup, but for search we need to iterate
        # Let's pre-process keys to list to ensure order
        case_ids = list(embeddings_cache.keys())
        
        status.success("✅ System Ready")
        time.sleep(1)
        status.empty()
        return model, embeddings_cache
    except Exception as e:
        st.error(f"Failed to load embeddings cache: {e}")
        return None, None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------------------------------------------------------
# 🖥️ MAIN UI
# -----------------------------------------------------------------------------

def main():
    st.title("🧠 Jina Semantic Search")
    st.caption("Search legal cases using natural language queries.")

    # Load Resources
    model, embeddings = load_jina_resources()
    
    if not model or not embeddings:
        st.stop()

    # Try to load Case ID mapping
    real_case_ids = []
    try:
        with open('models/hgcn_embeddings.pkl', 'rb') as f:
            hgcn_data = pickle.load(f)
        real_case_ids = [k for k in hgcn_data.keys() if k != 'filename']
    except:
        pass

    # Search Input
    query = st.text_input("Enter your query:", placeholder="e.g., 'drunk driving accident liability'")
    
    if st.button("Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a query.")
            st.stop()
            
        start_time = time.time()
        
        with st.spinner("Encoding query & searching..."):
            # 1. Embed Query
            query_vec = model.embed_query(query)
            query_vec = np.array(query_vec)
            
            # 2. Calculate Similarities
            scores = []
            
            for key, vec in embeddings.items():
                sim = cosine_similarity(query_vec, np.array(vec))
                scores.append((key, sim))
            
            # Sort
            scores.sort(key=lambda x: x[1], reverse=True)
            top_results = scores[:10]
            
        duration = time.time() - start_time
        st.success(f"Found {len(top_results)} results in {duration:.2f}s")
        
        # Display Results
        for rank, (key, score) in enumerate(top_results, 1):
            # Map to real name if possible
            display_name = key
            if key.isdigit() and real_case_ids and int(key) < len(real_case_ids):
                display_name = real_case_ids[int(key)]
            
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <h3>#{rank} {display_name}</h3>
                        <p>Semantic Match</p>
                    </div>
                    <div class="similarity-score">
                        {score:.4f}
                    </div>
                </div>
                <div style="background:#333; height:5px; width:100%; border-radius:3px; margin-top:10px;">
                    <div style="background:#4e8cff; height:100%; width:{score*100}%; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
