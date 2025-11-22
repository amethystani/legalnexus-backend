import streamlit as st
import time
import re
from hybrid_case_search import NovelHybridSearchSystem

# Page Config
st.set_page_config(
    page_title="Adversarial Legal AI (Llama3.2)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #e0e0e0; }
    .stTextInput > div > div > input {
        background-color: #1e212b; color: #ffffff;
        border: 1px solid #30363d; border-radius: 8px;
    }
    .reasoning-box {
        background-color: #161b22; border-left: 3px solid #3fb950;
        padding: 15px; margin-bottom: 15px; border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace; font-size: 0.9em; color: #8b949e;
    }
    .prosecutor-card {
        border: 1px solid #da3633; background-color: #2a0f0f;
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .defense-card {
        border: 1px solid #238636; background-color: #0f2a1a;
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .judge-card {
        border: 1px solid #d29922; background-color: #2a220f;
        padding: 20px; border-radius: 10px; margin-top: 20px;
    }
    .model-tag {
        background-color: #1f6feb; color: white;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.8em; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def parse_reasoning(text):
    """Extracts <think> content and the final answer"""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        final_answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return reasoning, final_answer
    return None, text

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚖️ Adversarial Legal AI")
    st.caption("Powered by **Llama3.2** • Optimized for Speed")
with col2:
    st.markdown("<br><div style='text-align:right'><span class='model-tag'>Llama3.2 (Fast)</span></div>", unsafe_allow_html=True)

# Initialize System
@st.cache_resource
def get_system():
    return NovelHybridSearchSystem()

try:
    with st.spinner("Initializing Llama3.2..."):
        system = get_system()
    st.sidebar.success("✅ System Ready")
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configuration")
top_k = st.sidebar.slider("Evidence Cases", 3, 8, 3)

# Main Interface
query = st.text_input("Enter your legal query:", placeholder="e.g., I hit a pedestrian but it was dark...")

if query:
    # Progress tracking
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    timer_placeholder = st.empty()
    start_time = time.time()
    
    # Step 1: Retrieval
    status_placeholder.info("🔍 Discovering relevant cases...")
    progress_bar.progress(20)
    
    analysis = system.query_expander.expand_query(query)
    weights, _ = system.weighting_engine.adapt_weights(analysis)
    candidates = system._retrieve_candidates(query, analysis.get("expanded_query", query), weights, top_k=top_k)
    case_docs = [doc for doc, score, breakdown in candidates]
    
    elapsed = time.time() - start_time
    timer_placeholder.caption(f"⏱️ {elapsed:.1f}s")
    progress_bar.progress(40)
    
    # Step 2: Prosecutor
    status_placeholder.warning("👨‍⚖️ Prosecutor arguing...")
    prosecutor_arg = system.prosecutor.debate_topic(query, case_docs)
    
    elapsed = time.time() - start_time
    timer_placeholder.caption(f"⏱️ {elapsed:.1f}s")
    progress_bar.progress(60)
    
    # Step 3: Defense
    status_placeholder.success("🛡️ Defense arguing...")
    defense_arg = system.defense.debate_topic(query, case_docs)
    
    elapsed = time.time() - start_time
    timer_placeholder.caption(f"⏱️ {elapsed:.1f}s")
    progress_bar.progress(80)
    
    # Step 4: Judge
    status_placeholder.info("⚖️ Judge deliberating...")
    ruling = system.judge.deliver_ruling(query, prosecutor_arg, defense_arg, case_docs)
    
    elapsed = time.time() - start_time
    progress_bar.progress(100)
    
    # Clear progress UI
    status_placeholder.empty()
    progress_bar.empty()
    timer_placeholder.empty()
    
    st.success(f"✅ Debate Complete ({elapsed:.1f}s)")
    
    results = {
        'prosecutor_argument': prosecutor_arg,
        'defense_argument': defense_arg,
        'judicial_ruling': ruling,
        'cases': candidates
    }

    # Display Results
    st.subheader("🏛️ The Courtroom Debate")
    
    col_pros, col_def = st.columns(2)
    
    with col_pros:
        st.markdown("### 👨‍⚖️ Prosecution")
        p_reasoning, p_arg = parse_reasoning(results['prosecutor_argument'])
        st.markdown('<div class="prosecutor-card">', unsafe_allow_html=True)
        if p_reasoning:
            with st.expander("💭 View Chain of Thought"):
                st.markdown(f"<div class='reasoning-box'>{p_reasoning}</div>", unsafe_allow_html=True)
        st.markdown(f"**Argument:**\n\n{p_arg}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_def:
        st.markdown("### 🛡️ Defense")
        d_reasoning, d_arg = parse_reasoning(results['defense_argument'])
        st.markdown('<div class="defense-card">', unsafe_allow_html=True)
        if d_reasoning:
            with st.expander("💭 View Chain of Thought"):
                st.markdown(f"<div class='reasoning-box'>{d_reasoning}</div>", unsafe_allow_html=True)
        st.markdown(f"**Argument:**\n\n{d_arg}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("⚖️ Chief Justice's Ruling")
    j_reasoning, j_ruling = parse_reasoning(results['judicial_ruling'])
    
    st.markdown('<div class="judge-card">', unsafe_allow_html=True)
    if j_reasoning:
        with st.expander("💭 View Judicial Deliberation"):
             st.markdown(f"<div class='reasoning-box'>{j_reasoning}</div>", unsafe_allow_html=True)
    st.markdown(f"### Final Judgment\n\n{j_ruling}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📚 Evidence Cited")
    for i, (doc, score, breakdown) in enumerate(results['cases']):
        # Create score breakdown display
        score_display = f"**Overall: {score:.3f}** | Semantic: {breakdown['semantic']:.3f} | Text: {breakdown['text']:.3f}"
        
        with st.expander(f"Exhibit {i+1}: {doc.metadata.get('title', 'Untitled')} - Relevance: {score:.2f}"):
            st.caption(f"Court: {doc.metadata.get('court', 'Unknown')} | Date: {doc.metadata.get('date', 'Unknown')}")
            st.info(score_display)
            st.markdown(f"**Excerpt:** {doc.page_content[:500]}...")
