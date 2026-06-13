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

BASE_URL = "http://127.0.0.1:8000"

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
    except Exception:
        return None
    return None


def post_query(query: str, source: str = None):
    """Send a query to the backend and return the response."""
    try:
        payload = {"query": query}
        if source:
            payload["source"] = source
        resp = requests.post(
            f"{BASE_URL}/query",
            json=payload,
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


def get_traces():
    """Fetch recent traces from the backend."""
    try:
        resp = requests.get(f"{BASE_URL}/traces", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("traces", [])
    except Exception:
        pass
    return []


# ── Ingestion Result Message Handler ──────────────────────────────────────

# Render any load results from the previous run
if "upload_message" in st.session_state and st.session_state["upload_message"]:
    if st.session_state.get("upload_success"):
        st.success(st.session_state["upload_message"])
    else:
        st.error(st.session_state["upload_message"])
    # Reset message states
    st.session_state["upload_message"] = None
    st.session_state["upload_success"] = None


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
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{BASE_URL}/load", files=files, timeout=120)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state["upload_message"] = f"✅ Loaded {data['chunks_loaded']} chunks from {data['source']}"
                            st.session_state["upload_success"] = True
                            st.session_state["active_source"] = data['source']
                        else:
                            st.session_state["upload_message"] = f"❌ {response.status_code}: {response.text}"
                            st.session_state["upload_success"] = False
                    except Exception as exc:
                        st.session_state["upload_message"] = f"❌ Connection error: {exc}"
                        st.session_state["upload_success"] = False
                st.rerun()

    with url_tab:
        url_input = st.text_input(
            "Enter a URL",
            placeholder="https://example.com/article",
            help="Enter a webpage URL to scrape and add to the knowledge base",
        )
        if url_input:
            if st.button("🌐 Load URL", key="load_url", use_container_width=True):
                with st.spinner("Processing URL..."):
                    try:
                        response = requests.post(
                            f"{BASE_URL}/load",
                            data={"url": url_input},
                            timeout=120,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state["upload_message"] = f"✅ Loaded {data['chunks_loaded']} chunks from {data['source']}"
                            st.session_state["upload_success"] = True
                            st.session_state["active_source"] = data['source']
                        else:
                            st.session_state["upload_message"] = f"❌ {response.status_code}: {response.text}"
                            st.session_state["upload_success"] = False
                    except Exception as exc:
                        st.session_state["upload_message"] = f"❌ Connection error: {exc}"
                        st.session_state["upload_success"] = False
                st.rerun()

    st.markdown("---")

    # Active Source UI
    active_source = st.session_state.get("active_source")
    if active_source:
        st.markdown("### 🎯 Active Document")
        st.info(f"Searching only in:\n**{active_source}**")
        if st.button("✖️ Clear Active Document", use_container_width=True):
            st.session_state.pop("active_source", None)
            st.rerun()

    st.markdown("---")

    # Database Reset
    st.markdown("### ⚠️ Danger Zone")
    if st.button("🗑️ Clear Vector Database", type="primary", use_container_width=True):
        with st.spinner("Clearing database..."):
            try:
                response = requests.post(f"{BASE_URL}/reset", timeout=30)
                if response.status_code == 200:
                    st.success("✅ Vector database cleared!")
                    st.session_state.pop("upload_message", None)
                    st.session_state.pop("active_source", None)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to clear: {response.text}")
            except Exception as exc:
                st.error(f"❌ Connection error: {exc}")

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
        source_to_use = st.session_state.get("active_source")
        result, error = post_query(query.strip(), source=source_to_use)

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
