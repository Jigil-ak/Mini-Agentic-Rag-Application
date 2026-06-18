"""
Single Agent — LLM-powered intent classification with tool execution.

Replaces the old threshold-based router with a true agent architecture:
1. Gemini LLM classifies intent and selects tools (primary)
2. Keyword-scoring fallback handles 429/timeout/network failures
3. Selected tools execute independently
4. LLM synthesises final answer from tool outputs

The agent never uses Milvus similarity scores for routing.
Vector search happens ONLY inside KnowledgeBaseTool after the
agent has already decided to use it.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import TOP_K
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import search_documents, get_document_count
from backend.tools.calculator import CalculatorTool
from backend.tools.web_search import WebSearchTool
from backend.agent.fallback import call_with_fallback

logger = logging.getLogger(__name__)


class RAGAgent:
    """Single agent that classifies intent via LLM and executes tools."""

    _REALTIME_REGEX = re.compile(
        r'\b(latest|current|today|right now|live|recent|fresh|news|headlines|update|updates|trending|'
        r'weather|forecast|temperature|rain|snow|wind|humidity|climate|'
        r'stock price|stocks|stock market|ticker|share price|nasdaq|dow jones|sp500|crypto|bitcoin|ethereum|solana|'
        r'yesterday|tomorrow|this week|this month|this year|tonight|currently|at the moment|'
        r'202[4-9]|203[0-9]|'
        r'score|match|game|standing|standings|versus|vs|tournament|olympics|championship)\b',
        re.IGNORECASE
    )

    def _is_realtime_query(self, query: str) -> bool:
        """Detect if query requires fresh external information."""
        return bool(self._REALTIME_REGEX.search(query))

    def __init__(self) -> None:
        self.top_similarity_score = 0.0
        self.top_chunk_source = "N/A"
        self.top_chunk_preview = "N/A"
        self.retrieved_chunk_count = 0

    def _get_retrieval_evidence(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Perform a pre-classification retrieval to gather evidence.
        
        Returns:
            Dict containing retrieval metrics for the query.
        """
        doc_count = 0
        top_similarity = 0.0
        top_source = "N/A"
        top_preview = "N/A"
        retrieved_chunk_count = 0
        source_match = False
        doc_keyword_match = False

        try:
            doc_count = get_document_count()
            logger.info("Retrieval evidence gathering - doc_count=%d", doc_count)
            if doc_count > 0:
                # Guarantee collection is loaded in memory before querying or searching
                from backend.vectordb.milvus_client import _get_client, COLLECTION_NAME
                client = _get_client()
                try:
                    client.load_collection(COLLECTION_NAME)
                    logger.info("[MilvusClient] Collection '%s' loaded into memory successfully for evidence check", COLLECTION_NAME)
                except Exception as load_err:
                    logger.warning("[MilvusClient] Failed to load collection '%s': %s", COLLECTION_NAME, load_err)

                # 1. Source metadata check: check for keyword overlap with unique sources
                try:
                    query_res = client.query(
                        collection_name=COLLECTION_NAME,
                        output_fields=["source"],
                        limit=100
                    )
                    unique_sources = set(r.get("source", "") for r in query_res if r.get("source"))
                except Exception as query_err:
                    logger.warning("Failed to query unique sources for metadata check: %s", query_err)
                    unique_sources = set()

                query_lower = query.lower()
                query_words = set(re.findall(r'\b\w{3,}\b', query_lower))

                for src in unique_sources:
                    src_clean = os.path.basename(src).lower()
                    src_clean = re.sub(r'\.[a-zA-Z0-9]+$', '', src_clean)
                    src_clean = src_clean.replace("_", " ").replace("-", " ")
                    src_words = set(re.findall(r'\b\w{3,}\b', src_clean))
                    if query_words.intersection(src_words):
                        source_match = True
                        break

                # 2. Document keywords match check
                doc_keywords = [
                    "document", "pdf", "resume", "cv", "uploaded file", 
                    "this file", "this document", "projects in the document"
                ]
                if any(kw in query_lower for kw in doc_keywords):
                    doc_keyword_match = True

                # 3. Similarity search check: get top-1 chunk
                if source_match or doc_keyword_match:
                    top_similarity = 1.0
                    retrieved_chunk_count = 1
                    top_source = "Metadata Match"
                else:
                    query_embedding = embed_query(query)
                    chunks = search_documents(query_embedding, top_k=1, source=source)
                    if chunks:
                        top_similarity = chunks[0].get("similarity", 0.0)
                        top_source = chunks[0].get("source", "unknown")
                        top_preview = chunks[0].get("text", "")[:200]
                        retrieved_chunk_count = len(chunks)

        except Exception as e:
            logger.warning("Error gathering retrieval evidence: %s", e)

        # Update instance properties for trace diagnostics
        self.top_similarity_score = top_similarity
        self.top_chunk_source = top_source
        self.top_chunk_preview = top_preview
        self.retrieved_chunk_count = retrieved_chunk_count

        logger.info(
            "Evidence Summary - query='%s', doc_count=%d, source_match=%s, doc_keyword_match=%s, "
            "top_similarity=%.4f, top_source='%s', retrieved_chunk_count=%d",
            query, doc_count, source_match, doc_keyword_match, top_similarity, top_source, retrieved_chunk_count
        )

        return {
            "doc_count": doc_count,
            "top_similarity": top_similarity,
            "top_source": top_source,
            "top_preview": top_preview,
            "retrieved_chunk_count": retrieved_chunk_count,
            "active_source": source,
            "source_match": source_match,
            "doc_keyword_match": doc_keyword_match,
        }

    def _is_document_related(self, query: str, source: Optional[str] = None) -> tuple[bool, float]:
        """Determine if the query is related to the indexed documents.

        Returns:
            Tuple of (is_related: bool, relevance_score: float)
        """
        evidence = self._get_retrieval_evidence(query, source=source)
        source_val = 1.0 if evidence["source_match"] else 0.0
        keyword_val = 1.0 if evidence["doc_keyword_match"] else 0.0
        chunk_val = 1.0 if evidence["retrieved_chunk_count"] > 0 else 0.0
        
        evidence_score = (0.45 * source_val) + (0.25 * keyword_val) + (0.50 * evidence["top_similarity"]) + (0.10 * chunk_val)
        is_rel = (evidence_score >= 0.30)
        return is_rel, evidence["top_similarity"]

    # ── Intent Classification (Primary: LLM) ──────────────────────────────

    def _analyze_intent(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Use Gemini as the intent classifier.

        Falls back to keyword scoring ONLY if Gemini fails
        (429, timeout, content block, etc.)

        Returns:
            Dict with keys: tools, reasoning, routing_method,
            routing_decision_timestamp, gemini_status, document_relevance_score
        """
        from backend.config import GEMINI_API_KEY
        
        evidence = self._get_retrieval_evidence(query, source=source)
        doc_count = evidence["doc_count"]
        top_similarity = evidence["top_similarity"]
        top_source = evidence["top_source"]
        retrieved_chunk_count = evidence["retrieved_chunk_count"]
        active_source = evidence["active_source"]

        if not GEMINI_API_KEY:
            print("[Agent] Gemini API key not configured, falling back to keyword scoring")
            fallback_res = self._analyze_intent_fallback(query, source=source)
            fallback_res["gemini_status"] = "not_attempted"
            fallback_res["document_relevance_score"] = top_similarity
            return fallback_res

        relation_info = ""
        if doc_count > 0:
            relation_info = f"""Knowledge Base Evidence:
- Total documents currently indexed in database: {doc_count}
- Active source filter in request: {active_source if active_source else 'None'}
- Chunks matching query in database: {retrieved_chunk_count}
- Top chunk similarity score: {top_similarity:.4f}
- Top chunk source document: '{top_source}'

Instructions:
- Use the similarity score and top chunk source as evidence of document relevance.
- If documents are present and top similarity is high or query is related to the top chunk source document, you should prioritize 'knowledge_base' before 'web_search'.
- Do NOT select 'web_search' unless you are confident the information cannot be found in the user's uploaded documents.
- If the query is an explicit arithmetic expression, select 'calculator'."""
        else:
            relation_info = "No documents are currently indexed in the knowledge base. You MUST NOT select 'knowledge_base'."

        routing_prompt = f"""You are an AI routing agent. Analyze the user query and select the appropriate tools based on the provided Knowledge Base Evidence.

Available tools:
1. knowledge_base — Search the user's uploaded documents.
2. web_search — Search the internet for current/external information not available in uploaded documents.
3. calculator — Use ONLY for explicit arithmetic expressions.

Knowledge Base Context Evidence:
{relation_info}

Rules:
- You must evaluate the retrieval evidence as a human would. If there is a high top similarity score or a matching chunk count and top chunk source matches the query's topic, you MUST select 'knowledge_base'.
- You may select multiple tools if the query benefits from both (e.g. comparing personal info with industry trends).
- calculator is mutually exclusive — if selected, select nothing else.
- Default to web_search if intent is unclear and no relevant documents exist.

User query: "{query}"

Respond ONLY with valid JSON, no explanation, no markdown:
{{"tools": ["tool1", "tool2"], "reasoning": "one sentence explanation"}}"""

        try:
            from backend.llm.primary_llm import generate as gemini_generate

            response = gemini_generate(routing_prompt)

            # Strip markdown fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.strip("`")
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()

            decision = json.loads(clean)

            # Validate tools are from the allowed set
            allowed = {"knowledge_base", "web_search", "calculator"}
            tools = [t for t in decision.get("tools", []) if t in allowed]
            if not tools:
                if doc_count > 0 and (top_similarity >= 0.20 or retrieved_chunk_count > 0):
                    tools = ["knowledge_base"]
                else:
                    tools = ["web_search"]

            reasoning = decision.get("reasoning", "LLM-based routing")

            print(f"[Agent] LLM routing decision: {tools}")
            print(f"[Agent] LLM reasoning: {reasoning}")

            logger.info(
                "Routing Stage - query='%s', routing_method='llm', selected_tools=%s",
                query,
                tools,
            )

            return {
                "tools": tools,
                "reasoning": reasoning,
                "routing_method": "llm",
                "routing_decision_timestamp": datetime.now(timezone.utc).isoformat(),
                "gemini_status": "success",
                "document_relevance_score": top_similarity,
            }

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "resource_exhausted" in err_msg.lower():
                gemini_status = "rate_limited"
            else:
                gemini_status = "failed"
            print(
                f"[Agent] LLM routing failed ({type(e).__name__}: {e}), "
                "falling back to keyword scoring"
            )
            fallback_res = self._analyze_intent_fallback(query, source=source)
            fallback_res["gemini_status"] = gemini_status
            fallback_res["document_relevance_score"] = top_similarity
            return fallback_res

    # ── Intent Classification (Fallback: Keyword Scoring) ─────────────────

    def _analyze_intent_fallback(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Keyword-scoring fallback — only used when Gemini routing fails.

        Handles 429 rate limits, network failures, and content blocks
        gracefully so the system always produces a routing decision.
        """
        query_lower = query.lower().strip()
        selected_tools: List[str] = []
        reasons: List[str] = []

        # ── Calculator detection ──────────────────────────────────────
        if re.search(r'\b\d+\s*[\+\-\*\/\%\^]\s*\d+\b', query_lower):
            selected_tools = ["calculator"]
            reasons.append("Arithmetic expression detected")
            logger.info(
                "Routing Stage - query='%s', routing_method='keyword_fallback', selected_tools=%s",
                query,
                selected_tools,
            )
            return {
                "tools": selected_tools,
                "reasoning": "[FALLBACK ROUTING] " + " | ".join(reasons),
                "routing_method": "keyword_fallback",
                "routing_decision_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        if re.search(r'\b(calculate|compute|evaluate|solve)\b', query_lower):
            selected_tools = ["calculator"]
            reasons.append("Math action keyword")
            logger.info(
                "Routing Stage - query='%s', routing_method='keyword_fallback', selected_tools=%s",
                query,
                selected_tools,
            )
            return {
                "tools": selected_tools,
                "reasoning": "[FALLBACK ROUTING] " + " | ".join(reasons),
                "routing_method": "keyword_fallback",
                "routing_decision_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── Real-time query detection ──────────────────────────────────
        is_rt = self._is_realtime_query(query)

        # ── Document-relevance check (weighted evidence score) ─────────
        evidence = self._get_retrieval_evidence(query, source=source)
        doc_count = evidence["doc_count"]
        top_similarity = evidence["top_similarity"]
        retrieved_chunk_count = evidence["retrieved_chunk_count"]
        source_match = evidence["source_match"]
        doc_keyword_match = evidence["doc_keyword_match"]

        source_val = 1.0 if source_match else 0.0
        keyword_val = 1.0 if doc_keyword_match else 0.0
        chunk_val = 1.0 if retrieved_chunk_count > 0 else 0.0
        
        evidence_score = (0.45 * source_val) + (0.25 * keyword_val) + (0.50 * top_similarity) + (0.10 * chunk_val)
        is_rel = (evidence_score >= 0.30)

        logger.info(
            "Fallback weighted evidence check - evidence_score=%.4f (threshold=0.30), is_rel=%s",
            evidence_score, is_rel
        )

        # Fallback routing logic based on document availability and relevance
        if doc_count > 0 and is_rel and not is_rt:
            selected_tools = ["knowledge_base"]
            reasons.append(f"Fallback: documents exist and evidence score {evidence_score:.3f} >= 0.30")
        else:
            selected_tools = ["web_search"]
            if doc_count == 0:
                reasons.append("Fallback: no documents indexed, using web search")
            elif is_rt:
                reasons.append("Fallback: query is real-time, using web search")
            else:
                reasons.append(f"Fallback: query is not related to indexed documents (evidence score {evidence_score:.3f} < 0.30), using web search")

        # Support comparison detection for multi-tool fallback
        if re.search(r'\b(compare|vs|versus|difference between|contrast)\b', query_lower):
            if doc_count > 0 and is_rel:
                selected_tools = ["knowledge_base", "web_search"]
                reasons.append("Fallback: comparison query, using KB and web search")

        print(f"[Agent] Fallback selected tools: {selected_tools}")

        logger.info(
            "Routing Stage - query='%s', routing_method='keyword_fallback', selected_tools=%s",
            query,
            selected_tools,
        )

        return {
            "tools": selected_tools,
            "reasoning": "[FALLBACK ROUTING] " + " | ".join(reasons),
            "routing_method": "keyword_fallback",
            "routing_decision_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Tool Execution ────────────────────────────────────────────────────

    def _execute_tools(
        self,
        tools: List[str],
        query: str,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute each selected tool independently.

        Args:
            tools: List of tool names to execute, in order.
            query: The original user query.
            source: Optional source document filter for KB search.

        Returns:
            Dict with tool outputs keyed by tool name, plus private
            metadata prefixed with ``_`` (not included in public trace).
        """
        results: Dict[str, Any] = {}

        if "calculator" in tools:
            try:
                expr = re.sub(r'[a-zA-Z\s\?\!\.]+', ' ', query).strip()
                expr = re.sub(r'\s+', '', expr)
                calc = CalculatorTool()
                output = calc.calculate(expr)
                results["calculator"] = str(output)
                results["_status_calculator"] = "success"
                print(f"[Agent] Calculator: '{expr}' = {output}")
            except Exception as e:
                results["calculator"] = f"Calculation error: {str(e)}"
                results["_status_calculator"] = "failed"
                print(f"[Agent] Calculator failed: {e}")

        if "knowledge_base" in tools:
            # KnowledgeBaseTool: agent already decided to use KB.
            # Milvus search happens HERE, inside the tool — not before routing.
            try:
                query_embedding = embed_query(query)
                # Pass source filter through — None means search all documents
                chunks = search_documents(query_embedding, top_k=TOP_K, source=source)
                print(
                    f"[Agent] KB search with source filter: {source!r}, "
                    f"returned {len(chunks)} chunks"
                )

                chunks_retrieved = len(chunks)
                sources = list(set(c.get("source", "unknown") for c in chunks))
                max_similarity = max((c.get("similarity", 0.0) for c in chunks), default=0.0)

                logger.info(
                    "KB Stage - query='%s', similarity_score=%.4f, chunks_retrieved=%d, source_names=%s",
                    query,
                    max_similarity,
                    chunks_retrieved,
                    sources,
                )

                if chunks:
                    context = "\n\n".join([
                        f"[Chunk {i + 1} | Source: {c.get('source', 'unknown')}]\n{c['text']}"
                        for i, c in enumerate(chunks)
                    ])
                    results["knowledge_base"] = context
                    results["_status_knowledge_base"] = "success"
                    results["_retrieval_hit"] = True
                    results["_kb_sources"] = list(
                        set(c.get("source", "") for c in chunks)
                    )
                    results["_chunks_used"] = len(chunks)
                else:
                    results["knowledge_base"] = (
                        "No relevant information found in knowledge base."
                    )
                    results["_status_knowledge_base"] = "empty"
                    results["_retrieval_hit"] = False
                    results["_chunks_used"] = 0
            except Exception as e:
                print(f"[Agent] KnowledgeBaseTool error: {e}")
                results["knowledge_base"] = f"Knowledge base unavailable: {str(e)}"
                results["_status_knowledge_base"] = "failed"
                results["_retrieval_hit"] = False
                results["_chunks_used"] = 0

        if "web_search" in tools:
            try:
                searcher = WebSearchTool()
                output = searcher.search(query, max_results=3)
                results["web_search"] = output
                results["_status_web_search"] = (
                    "success"
                    if output and output != "No search results found."
                    else "empty"
                )
                results["_web_search_count"] = getattr(searcher, "last_results_count", 0)
                print(f"[Agent] WebSearchTool: {len(output)} chars returned")
            except Exception as e:
                print(f"[Agent] WebSearchTool error: {e}")
                results["web_search"] = f"Web search failed: {str(e)}"
                results["_status_web_search"] = "failed"
                results["_web_search_count"] = 0

        return results

    # ── Prompt Assembly ───────────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        tool_results: Dict[str, Any],
        intent: Any,
    ) -> str:
        """Build the final LLM prompt from tool outputs.

        Assembles context sections from each tool's output and wraps them
        in a structured prompt for Gemini / Groq to synthesise an answer.
        """
        tools_list = intent.get("tools", []) if isinstance(intent, dict) else intent
        sections: List[str] = []

        if "knowledge_base" in tools_list and "knowledge_base" in tool_results:
            sections.append(
                "=== KNOWLEDGE BASE RESULTS ===\n"
                f"{tool_results['knowledge_base']}"
            )

        if "web_search" in tools_list and "web_search" in tool_results:
            sections.append(
                "=== WEB SEARCH RESULTS ===\n"
                f"{tool_results['web_search']}"
            )

        if "calculator" in tools_list and "calculator" in tool_results:
            sections.append(
                "=== CALCULATOR RESULT ===\n"
                f"{tool_results['calculator']}"
            )

        context_block = "\n\n".join(sections)

        prompt = (
            "You are a helpful assistant. Answer the user's question based on "
            "the provided context. If the context is insufficient, say so "
            "clearly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    # ── Main Orchestrator ─────────────────────────────────────────────────

    def run(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full single-agent pipeline.

        Args:
            query: The user's question.
            source: Optional source document to restrict KB search.

        Returns:
            Dict with keys: answer, selected_tools, tool_reasoning,
            routing_decision_timestamp, routing_method,
            tool_execution_results, tool_execution_status,
            retrieval_hit, model_used, fallback_triggered,
            sources, chunks_used, plus the new 8 trace fields.
        """
        print(f"\n[Agent] ===== QUERY: {query[:80]} =====")

        try:
            doc_count = get_document_count()
        except Exception:
            doc_count = 0

        # 1. Classify intent (LLM primary, keyword fallback)
        intent = self._analyze_intent(query, source=source)
        gemini_status = intent.get("gemini_status", "not_attempted")

        # 2. Execute selected tools
        tools = list(intent["tools"])
        tool_results = self._execute_tools(tools, query, source=source)

        kb_selected = "knowledge_base" in intent["tools"]
        kb_attempted = kb_selected

        # KB Escalation to Web Search:
        # If KB was selected but returned no results, and web_search is not in tools:
        if (kb_selected and 
            tool_results.get("_status_knowledge_base") in ("empty", "failed") and
            "web_search" not in tools):
            print("[Agent] KB returned empty — escalating to web_search")
            ws_results = self._execute_tools(["web_search"], query, source=None)
            tool_results.update(ws_results)
            tools.append("web_search")

        # Check if all executed tools are empty/failed
        all_empty = True
        tool_failure_reasons = []
        for tool_name in tools:
            status = tool_results.get(f"_status_{tool_name}", "unknown")
            if status == "success":
                all_empty = False
            else:
                tool_failure_reasons.append(f"{tool_name}: {status}")

        # 3. Build synthesis prompt or return explicit failure
        if all_empty:
            answer = "I could not find usable information to answer your question. The tools did not return relevant results."
            model_used = "none"
            fallback_triggered = False
        else:
            prompt = self._build_prompt(query, tool_results, tools)
            # 4. Generate answer with LLM fallback
            # Bypass Gemini if routing has already failed or rate-limited
            bypass_gemini = (gemini_status in ("rate_limited", "failed"))
            llm_result = call_with_fallback(
                prompt,
                bypass_gemini=bypass_gemini,
                initial_gemini_status=gemini_status
            )
            answer = llm_result["response"]
            model_used = llm_result["model_used"]
            fallback_triggered = llm_result["fallback_triggered"]
            if not bypass_gemini:
                gemini_status = llm_result.get("gemini_status", gemini_status)

        # 5. Separate public outputs from private metadata
        public_outputs = {
            k: v for k, v in tool_results.items() if not k.startswith("_")
        }
        execution_status = {
            k.replace("_status_", ""): v
            for k, v in tool_results.items()
            if k.startswith("_status_")
        }

        # 6. Construct routing explanation
        is_rel, relevance_score = self._is_document_related(query, source=source)
        is_rt = self._is_realtime_query(query)
        explanation_parts = []
        explanation_parts.append(f"Documents indexed: {doc_count}.")
        if doc_count > 0:
            explanation_parts.append(f"Query document-relevance check: {'Passed' if is_rel else 'Failed'}.")
            explanation_parts.append(f"Real-time query check: {'True' if is_rt else 'False'}.")
        explanation_parts.append(f"Routing method: {intent['routing_method']}.")
        explanation_parts.append(f"Selected tools: {intent['tools']}.")
        routing_explanation = " ".join(explanation_parts)

        logger.info(
            "LLM Stage - query='%s', model_selected='%s', fallback_triggered=%s",
            query,
            model_used,
            fallback_triggered,
        )

        return {
            "answer": answer,
            "selected_tools": intent["tools"],
            "tool_reasoning": intent["reasoning"],
            "routing_decision_timestamp": intent["routing_decision_timestamp"],
            "routing_method": intent["routing_method"],
            "tool_execution_results": public_outputs,
            "tool_execution_status": execution_status,
            "retrieval_hit": tool_results.get("_retrieval_hit", False),
            "model_used": model_used,
            "fallback_triggered": fallback_triggered,
            "sources": tool_results.get("_kb_sources", []),
            "chunks_used": tool_results.get("_chunks_used", 0),
            # New diagnostic fields
            "gemini_status": gemini_status,
            "knowledge_base_attempted": kb_attempted,
            "knowledge_base_selected": kb_selected,
            "knowledge_base_results_found": tool_results.get("_chunks_used", 0),
            "web_search_results_count": tool_results.get("_web_search_count", 0),
            "tool_failure_reason": " | ".join(tool_failure_reasons) if tool_failure_reasons else None,
            "routing_explanation": routing_explanation,
            "documents_indexed_at_query_time": doc_count,
            "document_relevance_score": relevance_score,
            # Phase 2 Evidence diagnostics
            "top_similarity_score": self.top_similarity_score,
            "top_chunk_source": self.top_chunk_source,
            "top_chunk_preview": self.top_chunk_preview,
            "source_filter_used": source,
            "retrieved_chunk_count": self.retrieved_chunk_count,
        }
