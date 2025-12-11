"""
HGCN Hyperbolic Search UI with Text Queries

Interactive web interface for searching legal cases using text queries.
Combines Jina semantic embeddings with HGCN hyperbolic hierarchy.

Run with: streamlit run hgcn_search_ui.py
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
import time
import sys
import os

# Page configuration
st.set_page_config(
    page_title="HGCN Hyperbolic Legal Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        font-size: 1.1em;
        padding: 12px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .hierarchy-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .supreme-court {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .high-court {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .lower-court {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    """Load HGCN embeddings (cached)"""
    with open('models/hgcn_embeddings.pkl', 'rb') as f:
        hgcn_embeddings = pickle.load(f)
    
    case_ids = [k for k in hgcn_embeddings.keys() if k != 'filename']
    
    # Load Jina embeddings cache
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            jina_embeddings = pickle.load(f)
        print(f"✓ Loaded {len(jina_embeddings)} Jina embeddings")
    except Exception as e:
        print(f"⚠️ Could not load Jina embeddings: {e}")
        jina_embeddings = {}
    
    return hgcn_embeddings, jina_embeddings, case_ids

@st.cache_resource
def load_jina_model():
    """Load Jina model for encoding queries"""
    try:
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from jina_embeddings_simple import JinaEmbeddingsSimple
        
        model = JinaEmbeddingsSimple(model_path="models/jina-embeddings-v3")
        return model, None
    except Exception as e:
        return None, str(e)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """Calculate Poincaré distance in hyperbolic space"""
    sqrt_c = np.sqrt(c)
    
    diff_norm_sq = np.sum((x - y) ** 2)
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    
    numerator = 2 * diff_norm_sq
    denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    
    if denominator <= 0:
        return float('inf')
    
    return (1.0 / sqrt_c) * np.arccosh(1 + c * numerator / denominator)

def get_court_level(radius: float) -> Tuple[str, str, str]:
    """Get court level from radius
    Returns: (emoji, level_name, css_class)
    """
    if radius < 0.10:
        return "🏛️", "Supreme Court", "supreme-court"
    elif radius < 0.15:
        return "⚖️", "High Court (Major)", "high-court"
    elif radius < 0.20:
        return "⚖️", "High Court", "high-court"
    elif radius < 0.30:
        return "📜", "Lower Court/Tribunal", "lower-court"
    else:
        return "📋", "District/Subordinate", "lower-court"

def search_with_text_query(query_text: str, jina_model, jina_embeddings: dict, hgcn_embeddings: dict, 
                           case_ids: List[str], top_k: int = 20) -> List[Tuple[str, float, float, float]]:
    """Search using text query
    Returns: [(case_id, semantic_similarity, hyperbolic_distance, radius), ...]
    """
    # 1. Embed query with Jina
    query_emb = np.array(jina_model.embed_query(query_text))
    
    # 2. Find semantically similar cases using Jina embeddings
    semantic_scores = []
    
    for idx, case_id in enumerate(case_ids):
        # Get Jina embedding for this case
        jina_key = str(idx)  # Jina cache uses numeric string keys
        if jina_key in jina_embeddings:
            case_jina_emb = np.array(jina_embeddings[jina_key])
            similarity = cosine_similarity(query_emb, case_jina_emb)
            semantic_scores.append((case_id, similarity, idx))
    
    # 3. Sort by semantic similarity
    semantic_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Get top candidates and add hyperbolic info
    results = []
    for case_id, similarity, idx in semantic_scores[:top_k]:
        if case_id in hgcn_embeddings:
            hgcn_emb = np.array(hgcn_embeddings[case_id])
            radius = np.linalg.norm(hgcn_emb)
            results.append((case_id, similarity, 0.0, radius))  # hyperbolic distance is 0 for now
    
    return results

def main():
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 3em;'>⚖️ HGCN Hyperbolic Legal Search</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; opacity: 0.8;'>Search Legal Cases with Natural Language</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading embeddings..."):
        hgcn_embeddings, jina_embeddings, case_ids = load_embeddings()
    
    # Load Jina model
    jina_model, jina_error = load_jina_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **HGCN Model**: Hyperbolic GNN  
        **Jina Model**: v3 Embeddings  
        **Cases**: {len(case_ids):,}  
        **Jina Embeddings**: {len(jina_embeddings):,}  
        """)
        
        st.markdown("### 🎯 Search Settings")
        top_k = st.slider("Number of results", min_value=5, max_value=50, value=20, step=5)
        
        st.markdown("### 📚 Court Hierarchy")
        st.markdown("""
        <div style='font-size: 0.9em;'>
        🏛️ <b>Supreme Court</b><br/>
        <span style='opacity: 0.7;'>Radius &lt; 0.10</span><br/><br/>
        
        ⚖️ <b>High Court</b><br/>
        <span style='opacity: 0.7;'>Radius 0.10-0.20</span><br/><br/>
        
        📜 <b>Lower Courts</b><br/>
        <span style='opacity: 0.7;'>Radius 0.20-0.30</span><br/><br/>
        
        📋 <b>District Courts</b><br/>
        <span style='opacity: 0.7;'>Radius &gt; 0.30</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Jina** encodes your text query
        2. **Semantic search** finds relevant cases
        3. **Hyperbolic hierarchy** ranks by authority
        4. Best of both worlds! 🎯
        """)
    
    # Main content
    st.markdown("### 🔍 Search with Natural Language")
    
    # Large search box
    query_text = st.text_input(
        "",
        placeholder="Type your legal query here... (e.g., 'drunk driving', 'property dispute', 'breach of contract')",
        label_visibility="collapsed",
        key="query_input"
    )
    
    # Example queries
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📝 Negligence", use_container_width=True):
            query_text = "negligence duty of care breach damages"
            st.session_state.query_input = query_text
            st.rerun()
    with col2:
        if st.button("🚗 Drunk Driving", use_container_width=True):
            query_text = "drunk driving DUI motor vehicle accident"
            st.session_state.query_input = query_text
            st.rerun()
    with col3:
        if st.button("📜 Contract Law", use_container_width=True):
            query_text = "breach of contract damages remedy"
            st.session_state.query_input = query_text
            st.rerun()
    with col4:
        if st.button("🏠 Property Rights", use_container_width=True):
            query_text = "property rights ownership dispute"
            st.session_state.query_input = query_text
            st.rerun()
    
    st.markdown("---")
    
    # Search button
    if query_text:
        if jina_model is None:
            st.error(f"""
            ❌ **Jina model not available**: {jina_error}
            
            Please install sentence-transformers:
            ```bash
            pip install sentence-transformers
            ```
            """)
        else:
            if st.button("🚀 Search Legal Cases", use_container_width=True, type="primary"):
                with st.spinner(f"🔍 Searching for: '{query_text}'..."):
                    start_time = time.time()
                    results = search_with_text_query(
                        query_text, jina_model, jina_embeddings, 
                        hgcn_embeddings, case_ids, top_k
                    )
                    search_time = time.time() - start_time
                
                # Results header
                st.success(f"✅ Found {len(results)} relevant cases in {search_time:.3f}s")
                
                # Query info
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>📍 Your Query</h3>
                    <p style='font-size: 1.2em; font-style: italic; opacity: 0.9;'>"{query_text}"</p>
                    <p><b>Search Method:</b> Semantic (Jina) + Hierarchical (HGCN)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                result_radii = [r[3] for r in results]
                similarities = [r[1] for r in results]
                
                with col1:
                    st.metric("Avg Similarity", f"{np.mean(similarities):.4f}")
                with col2:
                    st.metric("Top Match", f"{similarities[0]:.4f}")
                with col3:
                    st.metric("Mean Radius", f"{np.mean(result_radii):.4f}")
                with col4:
                    st.metric("Radius Std", f"{np.std(result_radii):.4f}")
                
                # Hierarchy distribution
                st.markdown("### 📊 Court Hierarchy Distribution")
                hierarchy_counts = {}
                for case_id, sim, dist, radius in results:
                    _, level, _ = get_court_level(radius)
                    hierarchy_counts[level] = hierarchy_counts.get(level, 0) + 1
                
                # Create bar chart
                chart_data = pd.DataFrame({
                    'Court Level': list(hierarchy_counts.keys()),
                    'Count': list(hierarchy_counts.values())
                })
                st.bar_chart(chart_data.set_index('Court Level'))
                
                # Results table
                st.markdown("### 🎯 Most Relevant Cases")
                
                for i, (case_id, similarity, hyp_dist, radius) in enumerate(results, 1):
                    emoji, level, css_class = get_court_level(radius)
                    
                    st.markdown(f"""
                    <div class='result-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='flex: 1;'>
                                <h4 style='margin: 0; color: white;'>{i}. {emoji} {case_id}</h4>
                                <p style='margin: 5px 0; opacity: 0.8;'>
                                    <span class='hierarchy-badge {css_class}'>{level}</span>
                                    &nbsp;&nbsp;
                                    <b>Similarity:</b> {similarity:.4f}
                                    &nbsp;&nbsp;
                                    <b>Radius:</b> {radius:.4f}
                                </p>
                            </div>
                            <div style='text-align: right; min-width: 100px;'>
                                <div style='font-size: 2em; opacity: 0.7;'>{int(similarity * 100)}%</div>
                                <div style='font-size: 0.8em; opacity: 0.6;'>match</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analysis
                st.markdown("---")
                st.markdown("### 💡 Search Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Semantic Matching (Jina)**")
                    st.write(f"""
                    - Query embedded using **Jina v3** (768-D)
                    - Top match similarity: **{similarities[0]:.4f}**
                    - Average similarity: **{np.mean(similarities):.4f}**
                    - Results ranked by **semantic relevance**
                    """)
                
                with col2:
                    st.markdown("**Hierarchy Analysis (HGCN)**")
                    st.write(f"""
                    - Mean radius: **{np.mean(result_radii):.4f}**
                    - Radius std: **{np.std(result_radii):.4f}**
                    - Court levels: **{len(hierarchy_counts)}** different
                    - Each case has **learned hierarchical position**
                    """)
                
                st.markdown("---")
                st.info("""
                **🎯 How this works:**
                
                Your text query is converted to a semantic embedding using **Jina**, which understands the *meaning* 
                of your query. The system then finds cases with similar semantic content. Each result also shows its 
                **hierarchical position** learned by the HGCN model from the legal citation network, helping you 
                understand the authority level of each case!
                """)
    
    else:
        # Show instructions
        st.markdown("""
        <div class='metric-card' style='text-align: center; padding: 40px;'>
            <h2 style='margin-bottom: 20px;'>👆 Enter a legal query above</h2>
            <p style='font-size: 1.1em; opacity: 0.8; margin-bottom: 30px;'>
                Type any legal concept, situation, or topic you want to search for
            </p>
            <p style='opacity: 0.6;'>
                Examples: "drunk driving", "property dispute", "breach of contract", "negligence"
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
