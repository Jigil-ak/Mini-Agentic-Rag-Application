"""
Streamlit frontend for the Mini Agentic RAG Application.

Communicates with the FastAPI backend over HTTP. Provides:
- Sidebar: PDF upload, URL ingestion, system status
- Main panel: Q&A interface with trace visualisation
- Bottom section: Recent trace log table
"""

import os

import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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
        with st.expander("🔍 Agent Trace", expanded=True):
            if trace:
                col1, col2 = st.columns(2)
                with col1:
                    # Tools Used with status
                    tools = trace.get("selected_tools", [])
                    st.markdown("**🛠️ Tools Used:**")
                    icons = {
                        "knowledge_base": "📚 Knowledge Base",
                        "web_search": "🌐 Web Search",
                        "calculator": "🧮 Calculator",
                    }
                    for t in tools:
                        status = trace.get("tool_execution_status", {}).get(t, "unknown")
                        status_icon = "✅" if status == "success" else "⚠️" if status == "empty" else "❌"
                        st.markdown(f"- {icons.get(t, t)} {status_icon} `{status}`")

                    # Routing type
                    routing_type = trace.get("routing_type", "unknown")
                    routing_colors = {
                        "knowledge_base": "🟢",
                        "web_search": "🔵",
                        "calculator": "🟡",
                        "multi_tool": "🟣",
                        "unknown": "⚪",
                    }
                    st.markdown(
                        f"**🛣️ Routing Type:** {routing_colors.get(routing_type, '⚪')} `{routing_type}`"
                    )

                    # Routing method (LLM vs fallback)
                    routing_method = trace.get("routing_method", "unknown")
                    if routing_method == "keyword_fallback":
                        st.warning("⚠️ LLM routing unavailable — keyword fallback used")
                    else:
                        st.success("✅ LLM-based routing")

                    # KB Retrieval
                    hit = trace.get("retrieval_hit", False)
                    st.markdown(f"**📥 KB Retrieval:** {'✅ Hit' if hit else '❌ Miss'}")

                    # Model
                    st.markdown(f"**🤖 Model:** `{trace.get('primary_model', 'unknown')}`")

                    # Fallback
                    fb = trace.get("fallback_triggered", False)
                    st.markdown(f"**🔄 Fallback:** {'⚠️ Groq (rate limit)' if fb else '✅ Gemini'}")

                    # Latency
                    st.markdown(f"**⚡ Latency:** `{trace.get('response_time_ms', 0):.0f}ms`")

                    # Gemini Status
                    gemini_status = trace.get("gemini_status", "not_attempted")
                    gemini_icons = {
                        "success": "🟢 success",
                        "rate_limited": "⚠️ rate_limited",
                        "failed": "🔴 failed",
                        "not_attempted": "⏭️ not_attempted"
                    }
                    st.markdown(f"**🔑 Gemini Status:** `{gemini_icons.get(gemini_status, gemini_status)}`")

                    # Docs Indexed
                    docs_indexed = trace.get("documents_indexed_at_query_time", 0)
                    st.markdown(f"**📦 Docs Indexed:** `{docs_indexed}`")

                    # Document Relevance Score
                    relevance_score = trace.get("document_relevance_score", 0.0)
                    st.markdown(f"**🎯 Doc Relevance Score:** `{relevance_score:.4f}`")

                    # Phase 2 Evidence
                    top_sim = trace.get("top_similarity_score", 0.0)
                    if top_sim:
                        st.markdown(f"**📈 Top Similarity Score:** `{top_sim:.4f}`")
                    
                    top_src = trace.get("top_chunk_source", "N/A")
                    if top_src != "N/A":
                        st.markdown(f"**📄 Top Chunk Source:** `{top_src}`")
                        
                    src_filter = trace.get("source_filter_used")
                    if src_filter:
                        st.markdown(f"**🔍 Source Filter Used:** `{src_filter}`")
                        
                    retrieved_chunks = trace.get("retrieved_chunk_count", 0)
                    if retrieved_chunks:
                        st.markdown(f"**📚 Chunks Retrieved:** `{retrieved_chunks}`")

                    top_preview = trace.get("top_chunk_preview", "N/A")
                    if top_preview != "N/A":
                        with st.expander("📝 Top Chunk Preview"):
                            st.caption(top_preview)

                with col2:
                    # Agent reasoning
                    st.markdown("**🧠 Agent Reasoning:**")
                    st.info(trace.get("tool_reasoning", "No reasoning recorded"))

                    st.markdown(f"**🛣️ Path:** `{trace.get('path_taken', 'unknown')}`")

                    rdt = trace.get("routing_decision_timestamp", "")
                    if rdt:
                        st.markdown(f"**🕐 Decision Time:** `{rdt}`")

                    chunks = trace.get("chunks_used", 0)
                    if chunks:
                        st.markdown(f"**📄 Chunks Used:** `{chunks}`")

                    # Routing Explanation
                    routing_explanation = trace.get("routing_explanation", "")
                    if routing_explanation:
                        st.markdown("**📋 Routing Explanation:**")
                        st.info(routing_explanation)

                    # Web Search Results Count
                    if "web_search" in trace.get("selected_tools", []):
                        ws_count = trace.get("web_search_results_count", 0)
                        st.markdown(f"**🔍 Web Results:** `{ws_count}`")

                    # KB Results Found
                    if "knowledge_base" in trace.get("selected_tools", []):
                        kb_found = trace.get("knowledge_base_results_found", 0)
                        st.markdown(f"**📚 KB Results Found:** `{kb_found}`")

                    # Tool Failure Reason
                    failure_reason = trace.get("tool_failure_reason")
                    if failure_reason:
                        st.warning(f"⚠️ Tool Failure: {failure_reason}")

                # Tool outputs (full width below columns)
                tool_results = trace.get("tool_execution_results", {})
                if tool_results:
                    st.markdown("**📊 Tool Outputs:**")
                    for tool_name, output in tool_results.items():
                        with st.expander(f"Output: {tool_name}"):
                            st.text(str(output)[:1000])

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
            selected = t.get("selected_tools", [])
            table_data.append({
                "Time": t.get("timestamp", "")[:19],
                "Query": t.get("query", "")[:60],
                "Routing": t.get("routing_type", t.get("path_taken", "")),
                "Tools": ", ".join(selected) if selected else t.get("tool_used", "-") or "-",
                "Method": t.get("routing_method", "-"),
                "Model": t.get("primary_model", ""),
                "Fallback": "Yes" if t.get("fallback_triggered") else "No",
                "Latency": f"{t.get('response_time_ms', 0.0):.0f}ms",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No traces recorded yet. Ask a question to generate traces.")
