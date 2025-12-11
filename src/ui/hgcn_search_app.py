"""
HGCN Hyperbolic Search App
Run with: streamlit run hgcn_search_app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import time
import sys
import os
import gc
import json
import glob
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("google-generativeai not installed. Install with: pip install google-generativeai")

# Page Config
st.set_page_config(
    page_title="LegalNexus Hyperbolic Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 🎨 PREMIUM STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    
    /* Cards */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.6);
        color: white;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 12px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Custom Classes */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s;
    }
    .result-card:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateX(4px);
        border-left: 3px solid #818cf8;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-sc { background: rgba(236, 72, 153, 0.2); color: #fbcfe8; border: 1px solid rgba(236, 72, 153, 0.3); }
    .badge-hc { background: rgba(59, 130, 246, 0.2); color: #bfdbfe; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-lc { background: rgba(16, 185, 129, 0.2); color: #a7f3d0; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🧠 MODEL & DATA LOADING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_data_light():
    """Load only the lightweight HGCN embeddings and IDs."""
    try:
        with open('models/hgcn_embeddings.pkl', 'rb') as f:
            hgcn_embeddings = pickle.load(f)
        case_ids = [k for k in hgcn_embeddings.keys() if k != 'filename']
        return hgcn_embeddings, case_ids
    except Exception as e:
        st.error(f"Failed to load HGCN embeddings: {e}")
        return {}, []

@st.cache_resource(show_spinner=False)
def load_jina_cache():
    """Load the Jina embeddings cache (medium memory usage)."""
    try:
        with open('data/case_embeddings_cache.pkl', 'rb') as f:
            jina_embeddings = pickle.load(f)
        return jina_embeddings
    except Exception as e:
        st.warning(f"Could not load Jina embeddings cache: {e}")
        return {}

# Define search tool for Gemini function calling
search_cases_tool = {
    "function_declarations": [
        {
            "name": "search_legal_cases",
            "description": "Search for legal cases using semantic similarity. Use this when initial results are not relevant enough or you need to refine the search with different keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant legal cases. Can be refined based on what you're looking for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

@st.cache_resource(show_spinner=False)
def load_gemini_model():
    """Load Gemini AI model for intelligent case analysis."""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        
        # Use Gemini Flash Lite for faster responses with function calling
        try:
            model = genai.GenerativeModel(
                model_name="models/gemini-flash-lite-latest",
                tools=[search_cases_tool]
            )
        except:
            # Fallback to regular model if function calling not available
            model = genai.GenerativeModel(model_name="models/gemini-flash-lite-latest")
        
        return model
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def load_jina_model():
    """Load embedding model that matches cache dimensions (768D)."""
    try:
        # Import here to avoid overhead if not used
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sentence_transformers import SentenceTransformer
        
        # Force garbage collection before loading
        gc.collect()
        
        # Use a 768-dimensional model to match the cache
        # all-mpnet-base-v2 is a high-quality 768-dim model
        # Alternative: 'paraphrase-multilingual-mpnet-base-v2' for multilingual support
        model_path = "sentence-transformers/all-mpnet-base-v2"
        
        print(f"Loading {model_path} (768 dimensions to match cache)...")
        model = SentenceTransformer(model_path)
        print(f"✓ Loaded {model_path}")
        
        # Wrap in a simple interface
        class EmbeddingModel:
            def __init__(self, st_model):
                self.model = st_model
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
                return embedding[0].tolist()
        
        return EmbeddingModel(model)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_case_texts():
    """Load case text content from CSV files. Returns dict mapping case_id -> text."""
    case_texts = {}
    
    try:
        # Load from binary dataset
        binary_path = "data/binary_dev/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv"
        if os.path.exists(binary_path):
            df = pd.read_csv(binary_path, header=None, names=['case_id', 'text', 'label'])
            for _, row in df.iterrows():
                case_id = str(row['case_id'])
                case_texts[case_id] = str(row.get('text', ''))
        
        # Load from ternary dataset
        ternary_path = "data/ternary_dev/CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv"
        if os.path.exists(ternary_path):
            df = pd.read_csv(ternary_path, header=None, names=['case_id', 'text', 'label'])
            for _, row in df.iterrows():
                case_id = str(row['case_id'])
                # Don't overwrite if already loaded from binary
                if case_id not in case_texts:
                    case_texts[case_id] = str(row.get('text', ''))
        
        # Load from JSON files
        json_files = glob.glob("data/legal_cases/*.json")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    case_id = data.get('id', os.path.basename(json_file).replace('.json', ''))
                    content = data.get('content', '')
                    if case_id and content:
                        case_texts[case_id] = content
            except Exception as e:
                continue
                
    except Exception as e:
        st.warning(f"Could not load some case texts: {e}")
    
    return case_texts

@st.cache_data(show_spinner=False)
def load_citation_network():
    """Load citation network from pickle file. Returns dict with edges and metadata."""
    citation_data = {
        'edges': [],
        'case_ids': [],
        'metadata': {}
    }
    
    try:
        # Try to load citation network
        for path in ['data/citation_network.pkl', 'data/citation_network_full.pkl']:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle different formats
                    if isinstance(data, dict):
                        citation_data['edges'] = data.get('edges', data.get('edge_list', []))
                        citation_data['case_ids'] = data.get('nodes', data.get('case_ids', []))
                        citation_data['metadata'] = data.get('metadata', {})
                    break
    except Exception as e:
        st.warning(f"Could not load citation network: {e}. Citation edges will be simulated.")
    
    return citation_data

# -----------------------------------------------------------------------------
# 🧮 UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def get_court_info(radius):
    """Return (Emoji, Name, BadgeClass) based on hyperbolic radius."""
    if radius < 0.10:
        return "🏛️", "Supreme Court", "badge-sc"
    elif radius < 0.20:
        return "⚖️", "High Court", "badge-hc"
    else:
        return "📜", "Lower Court", "badge-lc"

def poincare_distance(u, v):
    """Calculate Poincaré distance between two vectors."""
    sq_u = np.sum(u * u)
    sq_v = np.sum(v * v)
    sq_dist = np.sum((u - v) ** 2)
    val = 1 + 2 * sq_dist / ((1 - sq_u) * (1 - sq_v))
    # Numerical stability
    val = max(1.0, val)
    return np.arccosh(val)

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors with dimension checking."""
    a = np.array(a)
    b = np.array(b)
    
    # Check dimensions match
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} != {len(b)}")
    
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)

def calculate_hyperbolic_geodesic(p1, p2, num_points=20):
    """Calculate points along a hyperbolic geodesic in Poincaré disk model.
    
    Args:
        p1: Starting point (2D or 3D numpy array)
        p2: Ending point (2D or 3D numpy array)
        num_points: Number of points to generate along the geodesic
    
    Returns:
        Array of points along the geodesic curve
    """
    # Project to 2D if 3D
    if len(p1) > 2:
        p1_2d = p1[:2]
        p2_2d = p2[:2]
    else:
        p1_2d = p1
        p2_2d = p2
    
    # Ensure points are in unit disk
    norm1 = np.linalg.norm(p1_2d)
    norm2 = np.linalg.norm(p2_2d)
    if norm1 >= 1:
        p1_2d = p1_2d / norm1 * 0.95
    if norm2 >= 1:
        p2_2d = p2_2d / norm2 * 0.95
    
    # Check if points are very close or collinear with origin
    if np.linalg.norm(p1_2d - p2_2d) < 1e-6:
        # Same point - return linear interpolation
        t = np.linspace(0, 1, num_points)
        if len(p1) > 2:
            return np.array([p1 * (1-ti) + p2 * ti for ti in t])
        else:
            return np.array([p1_2d * (1-ti) + p2_2d * ti for ti in t])
    
    # Check if geodesic passes through origin (straight line case)
    cross_prod = p1_2d[0] * p2_2d[1] - p1_2d[1] * p2_2d[0]
    if abs(cross_prod) < 1e-6:
        # Geodesic is a straight line through origin
        t = np.linspace(0, 1, num_points)
        if len(p1) > 2:
            return np.array([p1 * (1-ti) + p2 * ti for ti in t])
        else:
            points_2d = np.array([p1_2d * (1-ti) + p2_2d * ti for ti in t])
            return points_2d
    
    # General case: geodesic is a circular arc orthogonal to boundary
    # Find center and radius of the circle
    # Using the formula for circle through two points orthogonal to unit circle
    
    x1, y1 = p1_2d
    x2, y2 = p2_2d
    
    # Coefficients for finding center (cx, cy) such that:
    # (x1-cx)^2 + (y1-cy)^2 = (x2-cx)^2 + (y2-cy)^2
    # cx^2 + cy^2 - r^2 = 1 (orthogonality condition)
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Midpoint perpendicular bisector
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    
    # Perpendicular direction
    if abs(dx) < 1e-6:
        # Vertical line between points
        cx = mx
        cy = (1 - cx**2) / (2 * my)
    elif abs(dy) < 1e-6:
        # Horizontal line between points
        cy = my
        cx = (1 - cy**2) / (2 * mx)
    else:
        # General case
        # Center lies on perpendicular bisector: cy = my - (dx/dy)*(cx - mx)
        # And satisfies: cx^2 + cy^2 = 1 + r^2, where r^2 = (x1-cx)^2 + (y1-cy)^2
        
        # Use perpendicular bisector constraint
        slope_perp = -dx / dy
        # cy = my + slope_perp * (cx - mx)
        
        # From orthogonality and bisector constraints:
        a = 1 + slope_perp**2
        b = -2 * (mx + slope_perp * (my - slope_perp * mx))
        c = mx**2 + (my - slope_perp * mx)**2 - 1 - ((x1-mx)**2 + (y1-my)**2)
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            # Fallback to straight line
            t = np.linspace(0, 1, num_points)
            if len(p1) > 2:
                return np.array([p1 * (1-ti) + p2 * ti for ti in t])
            else:
                return np.array([p1_2d * (1-ti) + p2_2d * ti for ti in t])
        
        cx1 = (-b + np.sqrt(discriminant)) / (2*a)
        cx2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Choose the center that gives the smaller arc
        cy1 = my + slope_perp * (cx1 - mx)
        cy2 = my + slope_perp * (cx2 - mx)
        
        # Pick center that's farther from both points (gives smaller arc inside disk)
        dist1 = min(np.linalg.norm(p1_2d - np.array([cx1, cy1])), 
                   np.linalg.norm(p2_2d - np.array([cx1, cy1])))
        dist2 = min(np.linalg.norm(p1_2d - np.array([cx2, cy2])), 
                   np.linalg.norm(p2_2d - np.array([cx2, cy2])))
        
        if dist1 < dist2:
            cx, cy = cx2, cy2
        else:
            cx, cy = cx1, cy1
    
    center = np.array([cx, cy])
    radius = np.linalg.norm(p1_2d - center)
    
    # Calculate angles for arc
    angle1 = np.arctan2(y1 - cy, x1 - cx)
    angle2 = np.arctan2(y2 - cy, x2 - cx)
    
    # Choose shorter arc
    angle_diff = angle2 - angle1
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi
    
    # Generate points along arc
    angles = np.linspace(angle1, angle1 + angle_diff, num_points)
    points_2d = np.array([[cx + radius * np.cos(a), cy + radius * np.sin(a)] for a in angles])
    
    # If original points were 3D, interpolate Z coordinate linearly
    if len(p1) > 2:
        z_coords = np.linspace(p1[2], p2[2], num_points)
        points_3d = np.column_stack([points_2d, z_coords])
        return points_3d
    
    return points_2d

def perform_semantic_search(query: str, jina_model, jina_cache, case_ids, case_texts, hgcn_embeddings, top_k: int = 5):
    """
    Perform semantic search and return top results.
    This function can be called by Gemini via function calling.
    """
    # Embed query
    query_vec = jina_model.embed_query(query)
    query_vec = np.array(query_vec)
    query_dim = len(query_vec)
    
    # Hybrid search: Combine semantic similarity with keyword matching
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scores = []
    for idx, cid in enumerate(case_ids):
        sid = str(idx)
        if sid in jina_cache:
            vec = np.array(jina_cache[sid])
            if len(vec) == query_dim:
                # Semantic similarity
                sem_sim = cosine_sim(query_vec, vec)
                
                # Keyword matching bonus
                keyword_bonus = 0.0
                if cid in case_texts:
                    case_text_lower = case_texts[cid].lower()
                    matching_words = sum(1 for word in query_words if word in case_text_lower and len(word) > 2)
                    if matching_words > 0:
                        keyword_bonus = min(0.2, matching_words * 0.05)
                
                combined_score = sem_sim + keyword_bonus
                scores.append((cid, combined_score, sem_sim, keyword_bonus))
    
    # Sort and get top K
    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:top_k]
    
    # Format results with case text
    formatted_results = []
    for score_tuple in top_results:
        if len(score_tuple) == 4:
            cid, combined_score, sem_sim, keyword_bonus = score_tuple
        else:
            cid, combined_score = score_tuple
        
        case_text = case_texts.get(cid, "Text not available")
        case_info = {
            "case_id": cid,
            "score": combined_score,
            "text": case_text[:1000] if len(case_text) > 1000 else case_text,  # Truncate for AI
            "full_text": case_text
        }
        
        # Add hyperbolic info if available
        if cid in hgcn_embeddings:
            h_vec = np.array(hgcn_embeddings[cid])
            case_info["radius"] = float(np.linalg.norm(h_vec))
        
        formatted_results.append(case_info)
    
    return formatted_results

def ai_analyze_results(query: str, initial_results: List[Dict], gemini_model, jina_model, jina_cache, 
                       case_ids, case_texts, hgcn_embeddings, max_iterations: int = 2):
    """
    AI-powered analysis of search results with iterative refinement.
    Returns final results and AI reasoning.
    """
    if not gemini_model:
        return initial_results, "AI analysis unavailable. Showing initial results."
    
    # Prepare context for AI
    results_text = "\n\n".join([
        f"Case {i+1} (ID: {r['case_id']}, Score: {r['score']:.3f}):\n{r['text']}"
        for i, r in enumerate(initial_results)
    ])
    
    prompt = f"""You are a legal research assistant analyzing search results for a user query.

User Query: "{query}"

Initial Search Results (Top 5):
{results_text}

Your task:
1. Analyze if these results are relevant to the user's query
2. If results are relevant (score > 0.3 and content matches query), return them
3. If results are NOT relevant enough, use the search_legal_cases function to search again with a refined query

Guidelines:
- Be strict: Only return results if they genuinely match the query
- If searching again, refine the query to be more specific or use different keywords
- Consider legal terminology and concepts
- Maximum {max_iterations} search iterations allowed

Respond with your analysis and either:
- Return the relevant cases from the initial results, OR
- Call search_legal_cases with a refined query"""
    
    iteration = 0
    current_results = initial_results
    reasoning = []
    
    while iteration < max_iterations:
        iteration += 1
        reasoning.append(f"🤖 AI Analysis (Iteration {iteration})...")
        
        try:
            # Call Gemini with function calling
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )
            
            # Check if function was called
            function_called = False
            text_response = ""
            
            # Get full response text first
            full_response_text = ""
            if hasattr(response, 'text'):
                full_response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts if hasattr(candidate.content, 'parts') else []
                    for part in parts:
                        if hasattr(part, 'text'):
                            full_response_text += part.text
            
            # Check for function call in API format first
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check for function calls in API format
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts if hasattr(candidate.content, 'parts') else []
                    
                    for part in parts:
                        # Check for function call in API format
                        if hasattr(part, 'function_call') and part.function_call:
                            func_call = part.function_call
                            if func_call.name == "search_legal_cases":
                                function_called = True
                                # Extract parameters
                                args = dict(func_call.args) if hasattr(func_call, 'args') else {}
                                refined_query = args.get('query', query)
                                refined_top_k = args.get('top_k', 5)
                                
                                reasoning.append(f"🔍 Refining search with: '{refined_query}'")
                                
                                # Perform new search
                                current_results = perform_semantic_search(
                                    refined_query, jina_model, jina_cache, case_ids, 
                                    case_texts, hgcn_embeddings, refined_top_k
                                )
                                
                                # Update prompt for next iteration
                                results_text = "\n\n".join([
                                    f"Case {i+1} (ID: {r['case_id']}, Score: {r['score']:.3f}):\n{r['text']}"
                                    for i, r in enumerate(current_results)
                                ])
                                prompt = f"""Refined Search Results:
{results_text}

Analyze these results. If they are relevant to "{query}", return them. Otherwise, search again with a different refined query."""
                                break
                        
                        # Extract text response
                        if hasattr(part, 'text'):
                            text_response += part.text
            
            # If no API function call, check if function call is in text (JSON format)
            if not function_called and full_response_text:
                # Look for function call in text format - handle both JSON and text formats
                # Pattern 1: "function_call": { "name": "search_legal_cases", "arguments": {...} }
                # Pattern 2: search_legal_cases with query in arguments
                
                # Try to find function call name
                if '"search_legal_cases"' in full_response_text or 'search_legal_cases' in full_response_text:
                    # Extract query from various formats
                    query_patterns = [
                        r'"query"\s*:\s*"([^"]+)"',  # Standard JSON
                        r'"query"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',  # JSON with escaped quotes
                        r'query["\']?\s*:\s*["\']([^"\']+)["\']',  # Flexible quotes
                        r'query["\']?\s*:\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']',  # With escapes
                    ]
                    
                    refined_query = None
                    for pattern in query_patterns:
                        query_match = re.search(pattern, full_response_text, re.IGNORECASE)
                        if query_match:
                            refined_query = query_match.group(1)
                            # Clean up escaped characters
                            refined_query = refined_query.replace('\\"', '"').replace('\\n', ' ').replace('\\/', '/')
                            break
                    
                    # Also try to extract from the specific format the user showed
                    if not refined_query:
                        # Look for the exact format: "query": "\"driving under the influence\" OR ..."
                        # Handle multi-line JSON with escaped quotes
                        complex_query_match = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)+)"', full_response_text, re.DOTALL)
                        if complex_query_match:
                            refined_query = complex_query_match.group(1)
                            # Unescape JSON string
                            refined_query = refined_query.replace('\\"', '"').replace('\\n', ' ').replace('\\/', '/')
                            # Remove outer quotes if present
                            refined_query = refined_query.strip('"')
                    
                    if refined_query:
                        function_called = True
                        
                        # Extract top_k if present
                        top_k_match = re.search(r'"top_k"\s*:\s*(\d+)', full_response_text)
                        refined_top_k = int(top_k_match.group(1)) if top_k_match else 5
                        
                        reasoning.append(f"🔍 AI requested search refinement: '{refined_query}'")
                        reasoning.append(f"⚙️ Executing search function...")
                        
                        # Perform new search
                        current_results = perform_semantic_search(
                            refined_query, jina_model, jina_cache, case_ids, 
                            case_texts, hgcn_embeddings, refined_top_k
                        )
                        
                        reasoning.append(f"✅ Found {len(current_results)} cases with refined query")
                        
                        # Update prompt for next iteration
                        results_text = "\n\n".join([
                            f"Case {i+1} (ID: {r['case_id']}, Score: {r['score']:.3f}):\n{r['text']}"
                            for i, r in enumerate(current_results)
                        ])
                        prompt = f"""Refined Search Results (from function call):
{results_text}

Analyze these results. If they are relevant to the original query "{query}", explain why and return them. 
If they are still not relevant, you may search once more with a different approach."""
                    
                    if not function_called:
                        # AI decided results are good
                        if text_response:
                            reasoning.append(f"✅ AI Analysis:\n{text_response}")
                        else:
                            reasoning.append("✅ AI confirmed results are relevant.")
                        break
                else:
                    # No content, check text directly
                    if hasattr(response, 'text'):
                        text_response = response.text
                        reasoning.append(f"✅ AI Analysis:\n{text_response}")
                    else:
                        reasoning.append("✅ Using initial results.")
                    break
            else:
                # Fallback: use text response
                if hasattr(response, 'text'):
                    text_response = response.text
                    reasoning.append(f"✅ AI Analysis:\n{text_response}")
                else:
                    reasoning.append("✅ Using initial results.")
                break
                
        except Exception as e:
            reasoning.append(f"⚠️ AI analysis error: {str(e)}. Using initial results.")
            break
    
    final_reasoning = "\n".join(reasoning)
    return current_results, final_reasoning

# -----------------------------------------------------------------------------
# 🖥️ MAIN UI
# -----------------------------------------------------------------------------

def main():
    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### 🧠 Memory Management")
        use_text_search = st.toggle("Enable Text Search (Requires RAM)", value=False)
        use_ai_analysis = st.toggle("🤖 Enable AI Analysis (ChatGPT-like)", value=True)
        
        if use_text_search:
            st.info("⚠️ Text search loads the embedding model (~1GB RAM). Disable if system is slow.")
        
        if use_ai_analysis:
            st.info("✨ AI will analyze results and refine search if needed. Requires GOOGLE_API_KEY in .env")
        
        st.markdown("### 🔍 Search Parameters")
        top_k = st.slider("Initial Results", 5, 50, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Hierarchy Guide")
        st.caption("Lower radius = Higher Authority")
        st.markdown("""
        - **< 0.10**: 🏛️ Supreme Court
        - **0.10 - 0.20**: ⚖️ High Court
        - **> 0.20**: 📜 Lower Courts
        """)

    # Main Header
    st.title("⚖️ LegalNexus")
    st.markdown("### Hyperbolic Case Search Engine")
    
    # Load Data (Light)
    with st.spinner("Loading knowledge base..."):
        hgcn_embeddings, case_ids = load_data_light()
        case_texts = load_case_texts()  # Load case text content
    
    if not hgcn_embeddings:
        st.stop()

    # Search Interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if use_text_search:
            query = st.text_input("🔎 Search Query", placeholder="e.g., 'drunk driving accident liability'")
            search_type = "text"
        else:
            query = st.selectbox("📂 Select Case ID", options=[""] + case_ids[:1000], format_func=lambda x: "Select a case..." if x == "" else x)
            search_type = "case"
            st.caption("💡 Enable 'Text Search' in sidebar to type queries.")

    with col2:
        st.write("") # Spacer
        st.write("")
        search_btn = st.button("🚀 Search", use_container_width=True)
    
    # Initialize session state for search results
    if 'search_completed' not in st.session_state:
        st.session_state.search_completed = False
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = None
    
    # Reset search state if new query is entered
    if search_btn and query:
        if st.session_state.last_search_query != query:
            st.session_state.search_completed = False
            st.session_state.chat_history = []  # Clear chat when new search
            st.session_state.last_search_query = query

    # Logic
    if search_btn and query:
        results = []
        start_time = time.time()
        
        if search_type == "text":
            # Load heavy resources only now
            with st.spinner("Loading Models (this happens once)..."):
                jina_model = load_jina_model()
                jina_cache = load_jina_cache()
                gemini_model = load_gemini_model() if use_ai_analysis else None
            
            if jina_model and jina_cache:
                # 1. Embed Query
                query_vec = jina_model.embed_query(query)
                query_vec = np.array(query_vec)
                query_dim = len(query_vec)
                
                # 2. Check dimension compatibility with cached embeddings
                # Find first cached embedding to check dimensions
                cached_dim = None
                for key in jina_cache.keys():
                    if key != 'filename':
                        cached_vec = np.array(jina_cache[key])
                        cached_dim = len(cached_vec)
                        break
                
                if cached_dim is None:
                    st.error("No cached embeddings found. Please regenerate the embeddings cache.")
                    st.stop()
                
                if query_dim != cached_dim:
                    st.error(
                        f"**Dimension Mismatch Error**\n\n"
                        f"The query embedding has {query_dim} dimensions, but the cached embeddings "
                        f"have {cached_dim} dimensions. This should not happen with the correct model.\n\n"
                        f"**Please ensure you're using a 768-dimensional embedding model.**"
                    )
                    st.stop()
                
                # 3. Semantic Search (Cosine)
                # Optimization: We iterate through cache keys that map to case_ids
                # For demo speed, we might limit to N cases if cache is huge, but 50k is fast for numpy
                
                # Convert cache to matrix for speed if possible, or loop
                # Looping 50k items in python is slow. Let's try a smarter way if keys match indices
                # Assuming jina_cache keys are '0', '1'... mapping to case_ids list order
                
                scores = []
                # Fast path: Check if we can build a matrix (only do this once in real app)
                # For now, simple loop with progress bar if needed
                
                progress_bar = st.progress(0)
                batch_size = 1000
                
                # We need to map jina_cache keys to case_ids
                # Assuming jina_cache keys '0' -> case_ids[0]
                
                # Let's just do a quick scan of first 5000 for responsiveness if full scan is too slow
                # Or scan all if user is patient. 50k dot products is fast in numpy, loop overhead is the issue.
                
                # Vectorized approach attempt:
                # Extract all embeddings into a matrix? (Memory heavy)
                # Let's stick to a safe loop for now to avoid crashing
                
                # Hybrid search: Combine semantic similarity with keyword matching
                query_lower = query.lower()
                query_words = set(query_lower.split())
                
                count = 0
                for idx, cid in enumerate(case_ids):
                    sid = str(idx)
                    if sid in jina_cache:
                        vec = np.array(jina_cache[sid])
                        # Double-check dimension before computing similarity
                        if len(vec) == query_dim:
                            # Semantic similarity (0 to 1)
                            sem_sim = cosine_sim(query_vec, vec)
                            
                            # Keyword matching bonus (0 to 0.2)
                            keyword_bonus = 0.0
                            if cid in case_texts:
                                case_text_lower = case_texts[cid].lower()
                                # Count matching words
                                matching_words = sum(1 for word in query_words if word in case_text_lower and len(word) > 2)
                                if matching_words > 0:
                                    # Bonus proportional to word matches, capped at 0.2
                                    keyword_bonus = min(0.2, matching_words * 0.05)
                            
                            # Combined score: semantic similarity + keyword bonus
                            combined_score = sem_sim + keyword_bonus
                            scores.append((cid, combined_score, sem_sim, keyword_bonus))
                    
                    if idx % 5000 == 0:
                        progress_bar.progress(min(idx / len(case_ids), 1.0))
                
                progress_bar.empty()
                # Sort by combined score
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # Get initial top results
                top_semantic = scores[:top_k]
                
                # Format initial results for AI analysis
                initial_results = []
                for score_tuple in top_semantic:
                    if len(score_tuple) == 4:
                        cid, combined_score, sem_sim, keyword_bonus = score_tuple
                        sim = combined_score
                    else:
                        cid, sim = score_tuple
                    
                    case_text = case_texts.get(cid, "Text not available")
                    result_item = {
                        "case_id": cid,
                        "score": sim,
                        "text": case_text[:1000] if len(case_text) > 1000 else case_text,
                        "full_text": case_text
                    }
                    
                    if cid in hgcn_embeddings:
                        h_vec = np.array(hgcn_embeddings[cid])
                        result_item["radius"] = float(np.linalg.norm(h_vec))
                    
                    initial_results.append(result_item)
                
                # AI Analysis (if enabled)
                if use_ai_analysis:
                    gemini_model = load_gemini_model()
                    if gemini_model:
                        with st.spinner("🤖 AI is analyzing results and refining search..."):
                            ai_results, ai_reasoning = ai_analyze_results(
                                query, initial_results, gemini_model, jina_model, 
                                jina_cache, case_ids, case_texts, hgcn_embeddings, max_iterations=2
                            )
                        
                        # Display AI reasoning
                        with st.expander("🤖 AI Analysis Process", expanded=True):
                            st.markdown(ai_reasoning)
                        
                        # Use AI-refined results
                        final_results = ai_results
                    else:
                        st.warning("⚠️ Gemini API not configured. Using initial results. Set GOOGLE_API_KEY in .env")
                        final_results = initial_results
                else:
                    final_results = initial_results
                
                # Convert to display format
                for r in final_results:
                    cid = r['case_id']
                    if cid in hgcn_embeddings:
                        h_vec = np.array(hgcn_embeddings[cid])
                        rad = np.linalg.norm(h_vec)
                        results.append({
                            "id": cid,
                            "score": r.get('score', 0.0),
                            "radius": r.get('radius', rad),
                            "type": "AI-Refined Match" if use_ai_analysis else "Semantic Match"
                        })

        elif search_type == "case":
            if query in hgcn_embeddings:
                q_vec = np.array(hgcn_embeddings[query])
                q_rad = np.linalg.norm(q_vec)
                
                # Hyperbolic Search
                dists = []
                for cid in case_ids:
                    if cid == query: continue
                    c_vec = np.array(hgcn_embeddings[cid])
                    dist = poincare_distance(q_vec, c_vec)
                    dists.append((cid, dist))
                
                dists.sort(key=lambda x: x[1])
                
                for cid, dist in dists[:top_k]:
                    h_vec = np.array(hgcn_embeddings[cid])
                    rad = np.linalg.norm(h_vec)
                    results.append({
                        "id": cid,
                        "score": dist, # Distance (lower is better)
                        "radius": rad,
                        "type": "Hyperbolic Match"
                    })
                
                st.info(f"Query Case Radius: **{q_rad:.4f}** ({get_court_info(q_rad)[1]})")

        # Mark search as completed
        st.session_state.search_completed = True
        st.session_state.last_search_query = query
        
        # Display Results
        duration = time.time() - start_time
        st.success(f"Found {len(results)} cases in {duration:.2f}s")
        
        if results:
            # Hierarchy Distribution Chart
            radii = [r['radius'] for r in results]
            hist_data = pd.DataFrame(radii, columns=["Radius"])
            
            col_chart, col_stats = st.columns([2, 1])
            with col_chart:
                st.caption("Hierarchy Distribution (Left = Higher Court)")
                st.bar_chart(hist_data, height=150)
            
            with col_stats:
                avg_rad = np.mean(radii)
                emoji, name, _ = get_court_info(avg_rad)
                st.metric("Avg Hierarchy", f"{avg_rad:.3f}", f"{emoji} {name}")

            st.markdown("### 🏆 Top Results")
            
            for rank, item in enumerate(results, 1):
                emoji, court_name, badge_class = get_court_info(item['radius'])
                score_label = "Similarity" if search_type == "text" else "Distance"
                score_val = f"{item['score']:.4f}"
                
                # Get case text content
                case_id = item['id']
                case_text = case_texts.get(case_id, "Text content not available")
                
                # Header with case info
                col_header1, col_header2 = st.columns([3, 1])
                with col_header1:
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="font-size:1.2em; font-weight:bold;">#{rank}</span>
                        <span style="font-size:1.1em; color:#e2e8f0;">{case_id}</span>
                        <span class="badge {badge_class}" style="margin-left:10px;">{emoji} {court_name}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_header2:
                    st.markdown(f"""
                    <div style="text-align:right;">
                        <div style="font-size:0.9em; color:#94a3b8;">{score_label}</div>
                        <div style="font-size:1.2em; color:#818cf8; font-weight:bold;">{score_val}</div>
                        <div style="font-size:0.8em; color:#64748b;">Radius: {item['radius']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Case text content in expander
                with st.expander(f"📄 View Case Content ({len(case_text)} chars)", expanded=False):
                    if case_text and case_text != "Text content not available":
                        st.text_area("", value=case_text, height=200, key=f"case_text_{rank}", disabled=True, label_visibility="collapsed")
                    else:
                        st.info("Case text content not available in the dataset.")
                
                st.markdown("---")
            
            # 3D Hyperbolic Case Network Visualization
            if results and len(results) > 0:
                st.markdown("### 🌐 3D Hyperbolic Citation Network")
                st.caption("""
                **Interactive Poincaré Ball Visualization** - Explore case relationships in hyperbolic space:
                - 🔵 **Curved edges** represent citation relationships as hyperbolic geodesics
                - ➡️ **Arrows** indicate citation direction (A → B means A cites B)
                - 🟡🔵🟢 **Node colors** show court hierarchy (Gold=Supreme, Blue=High, Green=Lower)
                - 📊 **Node size** reflects citation count
                - 🌐 **Poincaré sphere** boundary visualizes the hyperbolic space
                - 🖱️ **Rotate, zoom, and hover** to explore the network
                """)
                
                try:
                    viz_html = create_3d_case_network(results, case_texts, hgcn_embeddings, query if search_type == "case" else None)
                    if viz_html:
                        components.html(viz_html, height=900, scrolling=False)
                    else:
                        st.info("3D visualization requires plotly. Install with: pip install plotly")
                except Exception as e:
                    st.warning(f"Could not generate 3D visualization: {str(e)}")

            
            # AI Legal Advisor Chatbot - only show if search was completed
            if use_ai_analysis and results and st.session_state.search_completed:
                st.markdown("---")
                col_chat_header, col_clear = st.columns([3, 1])
                with col_chat_header:
                    st.markdown("### 💬 Ask AI Legal Advisor")
                    st.caption("Chat with AI to understand what might happen based on these cases, ask questions, and get detailed case explanations.")
                with col_clear:
                    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
                        st.session_state.chat_history = []
                        st.rerun()
                
                # Initialize chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Store original query and results in session state for chat context
                if 'chat_query' not in st.session_state or st.session_state.chat_query != query:
                    st.session_state.chat_query = query
                    st.session_state.chat_results = results
                
                # Prepare case context for AI
                case_context = []
                for r in results[:5]:  # Use top 5 cases
                    case_id = r['id']
                    case_text = case_texts.get(case_id, "")
                    # Get court info
                    emoji, court_name, _ = get_court_info(r['radius'])
                    case_context.append({
                        "case_id": case_id,
                        "score": r['score'],
                        "court": court_name,
                        "text": case_text[:2000] if len(case_text) > 2000 else case_text,  # Limit for context
                        "full_text": case_text
                    })
                
                # Initial AI analysis if chat is empty
                if len(st.session_state.chat_history) == 0:
                    gemini_model = load_gemini_model()
                    if gemini_model:
                        with st.spinner("🤖 AI is analyzing the cases and preparing legal insights..."):
                            initial_analysis = generate_legal_analysis(
                                query, case_context, gemini_model
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": initial_analysis
                            })
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            with st.chat_message("user"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(message["content"])
                
                # Chat input - use form to prevent refresh
                with st.form(key="chat_form", clear_on_submit=True):
                    user_input = st.text_input(
                        "Ask about the cases, what might happen, or any legal questions...",
                        key="chat_input_field",
                        label_visibility="collapsed"
                    )
                    submit_chat = st.form_submit_button("Send", use_container_width=True)
                
                if submit_chat and user_input and st.session_state.search_completed:
                    # Prevent processing if this is a duplicate
                    last_user_msg = None
                    if st.session_state.chat_history:
                        for msg in reversed(st.session_state.chat_history):
                            if msg.get("role") == "user":
                                last_user_msg = msg.get("content")
                                break
                    
                    if last_user_msg != user_input:
                        # Add user message to history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        
                        # Get AI response
                        gemini_model = load_gemini_model()
                        jina_model = load_jina_model()
                        jina_cache = load_jina_cache()
                        
                        if gemini_model and jina_model and jina_cache:
                            with st.spinner("🤖 AI is thinking..."):
                                # Use stored query and results from session state
                                chat_query = st.session_state.get('chat_query', query)
                                chat_results = st.session_state.get('chat_results', results)
                                
                                # Rebuild case context from stored results
                                chat_case_context = []
                                for r in chat_results[:5]:
                                    case_id = r['id']
                                    case_text = case_texts.get(case_id, "")
                                    emoji, court_name, _ = get_court_info(r['radius'])
                                    chat_case_context.append({
                                        "case_id": case_id,
                                        "score": r['score'],
                                        "court": court_name,
                                        "text": case_text[:2000] if len(case_text) > 2000 else case_text,
                                        "full_text": case_text
                                    })
                                
                                # Check if AI wants to search for more cases based on user's answer
                                ai_response, should_search, search_query = chat_with_legal_advisor(
                                    user_input, st.session_state.chat_history, 
                                    chat_query, chat_case_context, gemini_model, jina_model, 
                                    jina_cache, case_ids, case_texts, hgcn_embeddings
                                )
                                
                                # If AI wants to search, perform search and update context
                                if should_search and search_query:
                                    with st.spinner(f"🔍 Searching for cases related to: {search_query}..."):
                                        new_results = perform_semantic_search(
                                            search_query, jina_model, jina_cache, 
                                            case_ids, case_texts, hgcn_embeddings, top_k=3
                                        )
                                        
                                        # Add new relevant cases to context
                                        for new_case in new_results:
                                            if new_case['case_id'] not in [c['case_id'] for c in chat_case_context]:
                                                chat_case_context.append(new_case)
                                        
                                        # Update AI response to mention new search
                                        if new_results:
                                            ai_response += f"\n\n🔍 I found {len(new_results)} additional relevant cases. Let me incorporate this information into my analysis."
                                            
                                            # Regenerate response with new cases
                                            ai_response, _, _ = chat_with_legal_advisor(
                                                user_input, st.session_state.chat_history, 
                                                chat_query, chat_case_context, gemini_model, jina_model, 
                                                jina_cache, case_ids, case_texts, hgcn_embeddings, 
                                                skip_search=True
                                            )
                                
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": ai_response
                                })
                            
                            # Rerun to show new messages
                            st.rerun()
                        else:
                            st.error("⚠️ Gemini API not configured. Please set GOOGLE_API_KEY in .env")

def generate_legal_analysis(user_query: str, case_context: List[Dict], gemini_model) -> str:
    """Generate initial legal analysis based on found cases."""
    
    cases_text = "\n\n".join([
        f"**Case {i+1} (ID: {case['case_id']}, Court: {case.get('court', 'Unknown')}, Relevance Score: {case['score']:.3f}):**\n{case['text']}\n"
        for i, case in enumerate(case_context)
    ])
    
    prompt = f"""You are an expert legal advisor analyzing Indian legal cases for a user. You MUST always provide a helpful response to the user, no matter what.

User's Query: "{user_query}"

Cases Available:
{cases_text}

CRITICAL RULES:
1. **NEVER tell the user that cases are not relevant** - that's for your internal use only
2. **ALWAYS provide a helpful response** - even if cases don't perfectly match, use them to provide general legal insights
3. **If cases are relevant**: Cite them and explain what happened
4. **If cases are less relevant**: Still provide helpful legal guidance based on what you know, and ask clarifying questions that can be used as search queries
5. **When asking questions**: Frame them as specific legal scenarios that could be searched (e.g., "Was this a workplace accident?" can become search query "workplace accident liability")

Your task:
1. **Analyze what might happen to the user** - be specific about legal consequences, outcomes, or implications
2. **If you need more information**: Ask 2-3 specific clarifying questions that are:
   - Clear and helpful
   - Structured as legal scenarios (so they can be used as search queries later)
   - Focused on getting relevant case information
3. **Cite cases ONLY if they are genuinely relevant** - don't mention irrelevant cases
4. **For each cited case**: Explain what happened, what the court decided, and how it relates
5. **Be conversational, helpful, and empathetic**

Format your response:
- If cases are relevant: "Based on similar cases I found, here's what might happen..." and cite them
- If you need more info: "To give you the most accurate advice, I'd like to understand..." and ask questions
- Always end positively: "Feel free to ask me more questions or provide additional details."

Response (NEVER mention that cases aren't relevant - always be helpful):"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"I apologize, but I encountered an error while analyzing the cases: {str(e)}. Please try again."

def create_3d_case_network(results: List[Dict], case_texts: Dict, hgcn_embeddings: Dict, 
                          query_case_id: str = None) -> str:
    """Create an enhanced 3D interactive HTML visualization showing case relationships with hyperbolic geodesics."""
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return None
    
    # Load citation network
    citation_network = load_citation_network()
    citation_edges = citation_network.get('edges', [])
    
    # Prepare nodes
    nodes = []
    node_positions = []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_citations_in = []
    node_citations_out = []
    
    # Build case ID to index mapping
    case_to_idx = {}
    
    # Get query case if specified
    query_case = None
    if query_case_id and query_case_id in hgcn_embeddings:
        query_case = query_case_id
    
    # Add nodes for each result
    for i, r in enumerate(results[:20]):  # Limit to 20 for performance
        case_id = r['id']
        if case_id in hgcn_embeddings:
            # Get 3D position from hyperbolic embedding
            h_vec = np.array(hgcn_embeddings[case_id])
            # Normalize to unit ball for visualization
            norm = np.linalg.norm(h_vec)
            if norm > 0:
                pos_3d = (h_vec[:3] / norm) * min(norm, 0.95)  # Keep in unit ball
            else:
                pos_3d = np.random.rand(3) * 0.1  # Small random if zero
            
            nodes.append(case_id)
            case_to_idx[case_id] = i
            node_positions.append(pos_3d)
            
            # Count citations for this case
            cites_count_in = sum(1 for edge in citation_edges if edge[1] == case_id or (isinstance(edge, dict) and edge.get('target') == case_id))
            cites_count_out = sum(1 for edge in citation_edges if edge[0] == case_id or (isinstance(edge, dict) and edge.get('source') == case_id))
            node_citations_in.append(cites_count_in)
            node_citations_out.append(cites_count_out)
            
            # Color by court hierarchy (radius)
            radius = r.get('radius', 0.5)
            if radius < 0.10:
                color = '#fbbf24'  # Gold for Supreme Court
                base_size = 18
            elif radius < 0.20:
                color = '#3b82f6'  # Blue for High Court
                base_size = 15
            else:
                color = '#10b981'  # Green for Lower Court
                base_size = 12
            
            # Highlight query case
            if case_id == query_case:
                color = '#ec4899'  # Magenta for query
                base_size = 22
            
            # Size by citation count (both in and out)
            total_citations = cites_count_in + cites_count_out
            size = base_size + min(total_citations, 10)  # Cap at +10
            
            node_colors.append(color)
            node_sizes.append(size)
            
            # Enhanced tooltip text
            case_text = case_texts.get(case_id, "")[:150]
            court_name = get_court_info(radius)[1]
            tooltip = (f"<b>Case: {case_id}</b><br>"
                      f"Court: {court_name}<br>"
                      f"Radius: {radius:.3f}<br>"
                      f"Score: {r['score']:.3f}<br>"
                      f"Citations In: {cites_count_in}<br>"
                      f"Citations Out: {cites_count_out}<br>"
                      f"Preview: {case_text}...")
            node_texts.append(tooltip)
    
    if not nodes:
        return None
    
    # Convert to arrays
    node_positions = np.array(node_positions)
    
    # Create figure
    fig = go.Figure()
    
    # Add Poincaré disk boundary (unit sphere in 3D)
    # Draw a semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 0.98 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.98 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.05,
        colorscale=[[0, '#818cf8'], [1, '#818cf8']],
        showscale=False,
        hoverinfo='skip',
        name='Poincaré Disk Boundary'
    ))
    
    # Add concentric spheres for distance reference
    for r_val in [0.3, 0.6, 0.9]:
        x_ref = r_val * np.outer(np.cos(u), np.sin(v))
        y_ref = r_val * np.outer(np.sin(u), np.sin(v))
        z_ref = r_val * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(
            x=x_ref, y=y_ref, z=z_ref,
            opacity=0.02,
            colorscale=[[0, '#64748b'], [1, '#64748b']],
            showscale=False,
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Create citation edges with hyperbolic geodesics
    edge_traces = []
    edges_added = set()
    
    # Build edge list from citation network
    for edge in citation_edges[:100]:  # Limit edges for performance
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source_id, target_id = edge[0], edge[1]
        elif isinstance(edge, dict):
            source_id = edge.get('source')
            target_id = edge.get('target')
        else:
            continue
        
        # Check if both nodes are in our visualization
        if source_id in case_to_idx and target_id in case_to_idx:
            source_idx = case_to_idx[source_id]
            target_idx = case_to_idx[target_id]
            
            # Avoid duplicate edges
            edge_key = tuple(sorted([source_idx, target_idx]))
            if edge_key in edges_added:
                continue
            edges_added.add(edge_key)
            
            # Get positions
            pos_source = node_positions[source_idx]
            pos_target = node_positions[target_idx]
            
            # Calculate hyperbolic geodesic
            try:
                geodesic_points = calculate_hyperbolic_geodesic(pos_source, pos_target, num_points=30)
                
                # Determine edge color based on relationship
                # Blue for citation, with slight variation
                edge_color = 'rgba(59, 130, 246, 0.4)'  # Blue with transparency
                edge_width = 2
                
                # Add geodesic curve
                fig.add_trace(go.Scatter3d(
                    x=geodesic_points[:, 0],
                    y=geodesic_points[:, 1],
                    z=geodesic_points[:, 2],
                    mode='lines',
                    line=dict(color=edge_color, width=edge_width),
                    showlegend=False,
                    hovertemplate=f'<b>Citation</b><br>{source_id} → {target_id}<extra></extra>',
                    name='Citation'
                ))
                
                # Add arrow at midpoint to show direction
                mid_idx = len(geodesic_points) // 2
                if mid_idx > 0 and mid_idx < len(geodesic_points) - 1:
                    arrow_pos = geodesic_points[mid_idx]
                    arrow_dir = geodesic_points[mid_idx + 1] - geodesic_points[mid_idx - 1]
                    arrow_dir = arrow_dir / np.linalg.norm(arrow_dir) * 0.05  # Scale arrow
                    
                    # Add small cone for arrow
                    fig.add_trace(go.Cone(
                        x=[arrow_pos[0]],
                        y=[arrow_pos[1]],
                        z=[arrow_pos[2]],
                        u=[arrow_dir[0]],
                        v=[arrow_dir[1]],
                        w=[arrow_dir[2]],
                        sizemode='absolute',
                        sizeref=0.3,
                        showscale=False,
                        colorscale=[[0, edge_color], [1, edge_color]],
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            except Exception as e:
                # Fallback to straight line if geodesic calculation fails
                fig.add_trace(go.Scatter3d(
                    x=[pos_source[0], pos_target[0]],
                    y=[pos_source[1], pos_target[1]],
                    z=[pos_source[2], pos_target[2]],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # If no citation edges found, connect similar cases (top 5)
    if len(edges_added) == 0:
        for i in range(min(5, len(nodes))):
            for j in range(i+1, min(5, len(nodes))):
                if (query_case and (nodes[i] == query_case or nodes[j] == query_case)) or (i < 3 and j < 3):
                    pos1 = node_positions[i]
                    pos2 = node_positions[j]
                    
                    try:
                        geodesic_points = calculate_hyperbolic_geodesic(pos1, pos2, num_points=20)
                        fig.add_trace(go.Scatter3d(
                            x=geodesic_points[:, 0],
                            y=geodesic_points[:, 1],
                            z=geodesic_points[:, 2],
                            mode='lines',
                            line=dict(color='rgba(125, 125, 125, 0.2)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    except:
                        # Straight line fallback
                        fig.add_trace(go.Scatter3d(
                            x=[pos1[0], pos2[0]],
                            y=[pos1[1], pos2[1]],
                            z=[pos1[2], pos2[2]],
                            mode='lines',
                            line=dict(color='rgba(125, 125, 125, 0.2)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
    
    # Add nodes (after edges so they appear on top)
    fig.add_trace(go.Scatter3d(
        x=node_positions[:, 0],
        y=node_positions[:, 1],
        z=node_positions[:, 2],
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.9,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=[n[:12] + "..." if len(n) > 12 else n for n in nodes],
        textposition="top center",
        textfont=dict(size=9, color='white', family='Arial Black'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=node_texts,
        name='Cases'
    ))
    
    # Create legend annotations
    legend_text = (
        "<b>Court Hierarchy:</b><br>"
        "🟡 Gold: Supreme Court (radius < 0.10)<br>"
        "🔵 Blue: High Court (0.10 - 0.20)<br>"
        "🟢 Green: Lower Court (> 0.20)<br>"
        "🔴 Magenta: Query Case<br><br>"
        "<b>Node Size:</b> Citation count<br>"
        "<b>Edges:</b> Citation relationships<br>"
        "<b>Curved Edges:</b> Hyperbolic geodesics<br>"
        "Arrows show citation direction"
    )
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': '<b>3D Hyperbolic Citation Network</b><br><sub>Interactive Poincaré Ball Model</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(
                title='X',
                range=[-1.1, 1.1],
                backgroundcolor='rgba(15, 23, 42, 0.9)',
                gridcolor='rgba(100, 116, 139, 0.2)',
                showbackground=True,
                zerolinecolor='rgba(148, 163, 184, 0.3)'
            ),
            yaxis=dict(
                title='Y',
                range=[-1.1, 1.1],
                backgroundcolor='rgba(15, 23, 42, 0.9)',
                gridcolor='rgba(100, 116, 139, 0.2)',
                showbackground=True,
                zerolinecolor='rgba(148, 163, 184, 0.3)'
            ),
            zaxis=dict(
                title='Z',
                range=[-1.1, 1.1],
                backgroundcolor='rgba(15, 23, 42, 0.9)',
                gridcolor='rgba(100, 116, 139, 0.2)',
                showbackground=True,
                zerolinecolor='rgba(148, 163, 184, 0.3)'
            ),
            bgcolor='rgba(15, 23, 42, 1)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),  # Better default angle
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=800,
        paper_bgcolor='rgba(15, 23, 42, 1)',
        plot_bgcolor='rgba(15, 23, 42, 1)',
        font=dict(color='white', family='Arial'),
        showlegend=False,
        annotations=[
            dict(
                text=legend_text,
                xref='paper',
                yref='paper',
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                showarrow=False,
                bgcolor='rgba(30, 41, 59, 0.85)',
                bordercolor='rgba(99, 102, 241, 0.5)',
                borderwidth=2,
                borderpad=10,
                font=dict(size=11, color='#e2e8f0', family='Arial')
            )
        ],
        hoverlabel=dict(
            bgcolor='rgba(30, 41, 59, 0.95)',
            font_size=12,
            font_family='Arial',
            bordercolor='rgba(99, 102, 241, 0.8)'
        )
    )
    
    # Convert to HTML
    html_str = fig.to_html(include_plotlyjs='cdn', div_id="case-network-3d")
    return html_str


def chat_with_legal_advisor(user_message: str, chat_history: List[Dict], original_query: str, 
                           case_context: List[Dict], gemini_model, jina_model=None, 
                           jina_cache=None, case_ids=None, case_texts=None, 
                           hgcn_embeddings=None, skip_search=False) -> tuple:
    """Handle conversational chat with legal advisor."""
    
    # Build conversation context
    conversation_context = f"""You are a helpful legal advisor chatbot specializing in Indian law. The user originally asked: "{original_query}"

Cases Available (use these when relevant, ignore if not relevant):
"""
    for i, case in enumerate(case_context):
        conversation_context += f"\n**Case {i+1} (ID: {case['case_id']}, Court: {case.get('court', 'Unknown')}):**\n{case['text'][:800]}...\n"
    
    conversation_context += "\n\nRecent Conversation:\n"
    # Include all history except the last user message (which is the current one)
    for msg in chat_history[:-1]:  # All except the last message (current user input)
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_context += f"{role}: {msg['content']}\n"
    
    conversation_context += f"\n\nCurrent User Message: {user_message}\n\n"
    
    # Determine if we should search for more cases
    should_search = False
    search_query = None
    
    if not skip_search and jina_model and jina_cache:
        # Check if user's message answers a question that could be used as a search query
        # Look for specific legal scenarios, facts, or situations mentioned
        search_indicators = [
            "it was", "it is", "this is", "it happened", "the situation is",
            "yes", "no", "it's a", "it's an", "type of", "kind of"
        ]
        
        user_lower = user_message.lower()
        if any(indicator in user_lower for indicator in search_indicators):
            # Extract potential search query from user's answer
            # Try to identify the legal scenario they're describing
            conversation_context += """BEFORE RESPONDING, analyze if the user's message contains information that could help find more relevant cases.
If yes, extract a search query (2-5 words) that describes the legal scenario. Format: SEARCH_QUERY: [query]
If no, format: SEARCH_QUERY: None

Example:
User: "Yes, it was a workplace accident"
SEARCH_QUERY: workplace accident compensation

User: "It's a contract dispute"
SEARCH_QUERY: contract dispute breach

User: "Just asking a general question"
SEARCH_QUERY: None

Now analyze:"""
            
            try:
                search_analysis = gemini_model.generate_content(conversation_context + "\n\nExtract search query:")
                search_text = search_analysis.text if hasattr(search_analysis, 'text') else str(search_analysis)
                
                # Extract search query from response
                search_match = re.search(r'SEARCH_QUERY:\s*(.+?)(?:\n|$)', search_text, re.IGNORECASE)
                if search_match:
                    potential_query = search_match.group(1).strip()
                    if potential_query.lower() != "none" and len(potential_query) > 3:
                        should_search = True
                        search_query = potential_query
            except:
                pass
    
    conversation_context += """CRITICAL RULES:
1. **NEVER mention that cases are not relevant** - that's for your internal use only
2. **ALWAYS provide a helpful response** - use cases if relevant, provide general guidance if not
3. **Cite cases ONLY when they are genuinely relevant** - don't force irrelevant citations
4. **If you need better cases**: Ask clarifying questions that are structured as searchable legal scenarios
5. **When asking questions**: Frame them so they can be used as search queries (e.g., "Was this a commercial dispute?" → searchable as "commercial dispute")

Instructions:
- Answer naturally and conversationally
- **Cite cases by ID ONLY when relevant** (e.g., "In Case [ID]...")
- **Explain what happened** in cited cases - situation, court decision, and relevance
- If they ask "what will happen to me", analyze based on relevant cases or provide general legal guidance
- Be empathetic, helpful, and clear
- If you need more information, ask questions that can be used as search queries
- Keep responses detailed but not overwhelming
- **Always be helpful** - never say you don't have relevant cases

Your response (NEVER mention irrelevant cases or that cases don't match):"""
    
    try:
        response = gemini_model.generate_content(conversation_context)
        ai_response = response.text if hasattr(response, 'text') else str(response)
        return ai_response, should_search, search_query
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again.", False, None
    
    try:
        response = gemini_model.generate_content(conversation_context)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."

if __name__ == "__main__":
    main()
