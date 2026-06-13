"""
Streamlit frontend for the Mini Agentic RAG Application.

Communicates with the FastAPI backend over HTTP. Provides:
- Sidebar: PDF upload, URL ingestion, system status
- Main panel: Q&A interface with trace visualisation
- Bottom section: Recent trace log table
"""

import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
    }
    .trace-path-rag {
        color: #00d4aa;
        font-weight: bold;
    }
    .trace-path-tool {
        color: #ff6b6b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────

def get_health():
    """Fetch health status from the backend."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None
    return None


def post_query(query: str):
    """Send a query to the backend and return the response."""
    try:
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"query": query},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", "Unknown error")
            return None, f"Error {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the FastAPI server running?"
    except Exception as exc:
        return None, f"Request failed: {exc}"


def post_load_pdf(file):
    """Upload a PDF file to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        resp = requests.post(f"{BASE_URL}/load", files=files, timeout=120)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", "Unknown error")
            return None, f"Error {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the FastAPI server running?"
    except Exception as exc:
        return None, f"Upload failed: {exc}"


def post_load_url(url: str):
    """Submit a URL for ingestion to the backend."""
    try:
        resp = requests.post(
            f"{BASE_URL}/load",
            data={"url": url},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", "Unknown error")
            return None, f"Error {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the FastAPI server running?"
    except Exception as exc:
        return None, f"URL ingestion failed: {exc}"


def get_traces():
    """Fetch recent traces from the backend."""
    try:
        resp = requests.get(f"{BASE_URL}/traces", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("traces", [])
    except Exception:
        pass
    return []


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🤖 Agentic RAG")
    st.markdown("---")

    # System status
    st.markdown("### 📊 System Status")
    health = get_health()

    if health:
        status_color = "🟢" if health["milvus"] == "connected" else "🔴"
        st.markdown(f"**Backend:** 🟢 Online")
        st.markdown(f"**Milvus:** {status_color} {health['milvus'].title()}")
        st.markdown(f"**Documents Indexed:** `{health['documents_indexed']}`")
    else:
        st.markdown("**Backend:** 🔴 Offline")
        st.warning("Start the backend with:\n```\nuvicorn backend.main:app --reload\n```")

    st.markdown("---")

    # Data ingestion tabs
    st.markdown("### 📥 Load Knowledge")
    pdf_tab, url_tab = st.tabs(["📄 PDF Upload", "🌐 URL"])

    with pdf_tab:
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to add it to the knowledge base",
        )
        if uploaded_file is not None:
            if st.button("📤 Load PDF", key="load_pdf", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    result, error = post_load_pdf(uploaded_file)
                if error:
                    st.error(error)
                else:
                    st.success(
                        f"✅ Loaded **{result['chunks_loaded']}** chunks "
                        f"from `{result['source']}`"
                    )

    with url_tab:
        url_input = st.text_input(
            "Enter a URL",
            placeholder="https://example.com/article",
            help="Enter a webpage URL to scrape and add to the knowledge base",
        )
        if url_input:
            if st.button("🌐 Load URL", key="load_url", use_container_width=True):
                with st.spinner("Fetching and processing URL..."):
                    result, error = post_load_url(url_input)
                if error:
                    st.error(error)
                else:
                    st.success(
                        f"✅ Loaded **{result['chunks_loaded']}** chunks "
                        f"from URL"
                    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Built with Gemini + Groq + Milvus"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Panel ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🧠 Mini Agentic RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Ask questions about your documents or the web — the agent decides the best path."
    "</p>",
    unsafe_allow_html=True,
)

# Query input
query = st.text_area(
    "Your question",
    placeholder="e.g., What is retrieval-augmented generation? / Calculate 45 * 27",
    height=100,
    label_visibility="collapsed",
)

col_btn, col_space = st.columns([1, 4])
with col_btn:
    ask_clicked = st.button("🚀 Ask", use_container_width=True, type="primary")

# ── Process query ──────────────────────────────────────────────────────────

if ask_clicked and query and query.strip():
    with st.spinner("🔄 Processing your query..."):
        result, error = post_query(query.strip())

    if error:
        st.error(f"❌ {error}")
    elif result:
        # ── Display answer ──────────────────────────────────────────
        st.markdown("### 💡 Answer")
        st.success(result["answer"])

        # ── Display trace ───────────────────────────────────────────
        trace = result.get("trace", {})
        with st.expander("🔍 Query Trace", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                # Retrieval Hit
                retrieval_hit = trace.get("retrieval_hit", False)
                st.markdown(f"**Retrieval Hit:** {'✅ Yes' if retrieval_hit else '❌ No'}")

                # Path Taken
                path = trace.get("path_taken", "unknown")
                if path == "rag":
                    st.markdown(
                        '**Path Taken:** <span class="trace-path-rag">📚 RAG</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '**Path Taken:** <span class="trace-path-tool">🔧 Tool</span>',
                        unsafe_allow_html=True,
                    )

                # Model Used
                model = trace.get("primary_model", "unknown")
                st.markdown(f"**Model Used:** `{model}`")

                # Fallback
                fallback = trace.get("fallback_triggered", False)
                st.markdown(f"**Fallback:** {'✅ Triggered' if fallback else '❌ No'}")

            with col2:
                # Similarity Score
                similarity = trace.get("similarity_score", 0.0)
                st.markdown(f"**Similarity Score:** `{similarity:.4f}`")
                st.progress(min(similarity, 1.0))

                # Latency
                latency = trace.get("response_time_ms", 0.0)
                if latency >= 1000:
                    st.markdown(f"**Latency:** `{latency / 1000:.2f}s`")
                else:
                    st.markdown(f"**Latency:** `{latency:.0f}ms`")

                # Tool Used
                tool = trace.get("tool_used")
                if tool:
                    st.markdown(f"**Tool Used:** `{tool}`")
                else:
                    st.markdown("**Tool Used:** _None_")

                # Routing Reason
                reason = trace.get("routing_reason", "")
                if reason:
                    st.markdown(f"**Routing Reason:** _{reason}_")

elif ask_clicked:
    st.warning("⚠️ Please enter a question first.")

# ── Recent Traces ──────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("📋 Recent Traces", expanded=False):
    traces = get_traces()
    if traces:
        # Build a table from traces
        table_data = []
        for t in reversed(traces):
            table_data.append({
                "Time": t.get("timestamp", "")[:19],
                "Query": t.get("query", "")[:60],
                "Path": t.get("path_taken", ""),
                "Similarity": f"{t.get('similarity_score', 0.0):.3f}",
                "Model": t.get("primary_model", ""),
                "Fallback": "Yes" if t.get("fallback_triggered") else "No",
                "Latency": f"{t.get('response_time_ms', 0.0):.0f}ms",
                "Tool": t.get("tool_used") or "-",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No traces recorded yet. Ask a question to generate traces.")
