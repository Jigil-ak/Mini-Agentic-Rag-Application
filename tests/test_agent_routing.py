import pytest
from unittest.mock import patch, MagicMock
from backend.agent.agent import RAGAgent


@pytest.fixture
def agent():
    return RAGAgent()


class TestRealtimeQueryDetection:
    def test_realtime_keywords(self, agent):
        assert agent._is_realtime_query("What are the latest headlines?") is True
        assert agent._is_realtime_query("weather in New York") is True
        assert agent._is_realtime_query("Apple stock price today") is True
        assert agent._is_realtime_query("who won the match yesterday?") is True
        assert agent._is_realtime_query("events in 2025") is True

    def test_non_realtime_keywords(self, agent):
        assert agent._is_realtime_query("What is photosynthesis?") is False
        assert agent._is_realtime_query("Explain quantum computing") is False


class TestDocumentRelatedDetection:
    @patch("backend.agent.agent.get_document_count", return_value=0)
    def test_no_documents(self, mock_count, agent):
        assert agent._is_document_related("Who is Jigil?")[0] is False

    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.search_documents")
    @patch("backend.agent.agent.embed_query")
    @patch("backend.vectordb.milvus_client._get_client")
    def test_keyword_match_in_source(self, mock_get_client, mock_embed, mock_search, mock_count, agent):
        # Mock milvus query for unique sources
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.query.return_value = [{"source": "Oscar_Wikipedia.pdf"}]

        assert agent._is_document_related("Tell me about Oscar statuettes")[0] is True
        mock_search.assert_not_called() # keyword match shortcuts search

    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.search_documents")
    @patch("backend.agent.agent.embed_query")
    @patch("backend.vectordb.milvus_client._get_client")
    def test_similarity_match(self, mock_get_client, mock_embed, mock_search, mock_count, agent):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.query.return_value = [{"source": "Oscar_Wikipedia.pdf"}]
        mock_embed.return_value = [0.1] * 384
        
        # Test similarity score >= threshold (0.3)
        mock_search.return_value = [{"similarity": 0.75, "text": "Oscar details", "source": "Oscar_Wikipedia.pdf"}]
        assert agent._is_document_related("What are statuettes made of?")[0] is True

        # Test similarity score < threshold (0.3)
        mock_search.return_value = [{"similarity": 0.15, "text": "Python language", "source": "Oscar_Wikipedia.pdf"}]
        assert agent._is_document_related("What is Python?")[0] is False


class TestAnalyzeIntent:
    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.RAGAgent._get_retrieval_evidence")
    @patch("backend.llm.primary_llm.generate")
    @patch("backend.config.GEMINI_API_KEY", "mock_key")
    def test_gemini_routing_success(self, mock_generate, mock_evidence, mock_count, agent):
        mock_evidence.return_value = {
            "doc_count": 5,
            "top_similarity": 0.75,
            "top_source": "Oscar_Wikipedia.pdf",
            "top_preview": "Oscar details",
            "retrieved_chunk_count": 1,
            "active_source": None,
            "source_match": True,
            "doc_keyword_match": False
        }
        mock_generate.return_value = '{"tools": ["knowledge_base"], "reasoning": "matches uploaded docs"}'
        result = agent._analyze_intent("Who is Jigil?")
        assert result["routing_method"] == "llm"
        assert result["tools"] == ["knowledge_base"]
        assert result["gemini_status"] == "success"

    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.RAGAgent._get_retrieval_evidence")
    @patch("backend.agent.agent.RAGAgent._is_realtime_query", return_value=False)
    @patch("backend.llm.primary_llm.generate")
    @patch("backend.config.GEMINI_API_KEY", "mock_key")
    def test_gemini_routing_rate_limit_fallback(self, mock_generate, mock_is_rt, mock_evidence, mock_count, agent):
        from google.api_core.exceptions import ResourceExhausted
        mock_generate.side_effect = ResourceExhausted("Rate limit exceeded 429")
        mock_evidence.return_value = {
            "doc_count": 5,
            "top_similarity": 0.75,
            "top_source": "Oscar_Wikipedia.pdf",
            "top_preview": "Oscar details",
            "retrieved_chunk_count": 1,
            "active_source": None,
            "source_match": True,
            "doc_keyword_match": False
        }

        result = agent._analyze_intent("Who is Jigil?")
        assert result["routing_method"] == "keyword_fallback"
        assert result["tools"] == ["knowledge_base"]
        assert result["gemini_status"] == "rate_limited"


class TestAntiHallucinationAndEscalation:
    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.RAGAgent._analyze_intent")
    @patch("backend.agent.agent.RAGAgent._execute_tools")
    @patch("backend.agent.agent.RAGAgent._is_document_related", return_value=(False, 0.0))
    def test_all_empty_returns_failure_response(self, mock_is_doc, mock_execute, mock_intent, mock_count, agent):
        mock_intent.return_value = {
            "tools": ["knowledge_base"],
            "reasoning": "related to docs",
            "routing_method": "llm",
            "routing_decision_timestamp": "timestamp",
            "gemini_status": "success"
        }
        # Both tools return failure or empty
        mock_execute.return_value = {
            "_status_knowledge_base": "empty",
            "_chunks_used": 0
        }

        result = agent.run("unrelated query")
        assert "could not find usable information" in result["answer"].lower()
        assert result["tool_execution_status"] == {"knowledge_base": "empty"}

    @patch("backend.agent.agent.get_document_count", return_value=5)
    @patch("backend.agent.agent.RAGAgent._analyze_intent")
    @patch("backend.agent.agent.RAGAgent._execute_tools")
    @patch("backend.agent.agent.call_with_fallback")
    @patch("backend.agent.agent.RAGAgent._is_document_related", return_value=(False, 0.0))
    def test_kb_escalation_to_web_search(self, mock_is_doc, mock_call_llm, mock_execute, mock_intent, mock_count, agent):
        mock_intent.return_value = {
            "tools": ["knowledge_base"],
            "reasoning": "related to docs",
            "routing_method": "llm",
            "routing_decision_timestamp": "timestamp",
            "gemini_status": "success"
        }

        # Setup mock execution results
        def side_effect(tools, query, source=None):
            if "knowledge_base" in tools:
                return {
                    "knowledge_base": "No relevant info",
                    "_status_knowledge_base": "empty",
                    "_chunks_used": 0
                }
            elif "web_search" in tools:
                return {
                    "web_search": "Web result contents",
                    "_status_web_search": "success",
                    "_web_search_count": 3
                }
            return {}

        mock_execute.side_effect = side_effect
        mock_call_llm.return_value = {
            "response": "Final synthesized answer",
            "model_used": "gemini-2.0-flash",
            "fallback_triggered": False,
            "gemini_status": "success"
        }

        result = agent.run("escalate query")
        # Assert web search was escalated
        assert mock_execute.call_count == 2
        assert result["answer"] == "Final synthesized answer"
        assert "web_search" in result["tool_execution_status"]


class TestValidationQueries:
    @patch("backend.agent.agent.get_document_count", return_value=1050)
    @patch("backend.agent.agent.RAGAgent._get_retrieval_evidence")
    @patch("backend.agent.agent.call_with_fallback")
    @patch("backend.config.GEMINI_API_KEY", "mock_key")
    @patch("backend.llm.primary_llm.generate")
    def test_validation_queries(self, mock_generate, mock_call_llm, mock_evidence, mock_count, agent):
        # 1. Query: "First motorcycle companies are?" -> expected knowledge_base
        mock_evidence.return_value = {
            "doc_count": 1050,
            "top_similarity": 0.78,
            "top_source": "https://en.wikipedia.org/wiki/Motorcycle",
            "top_preview": "Some early motorcycle builders...",
            "retrieved_chunk_count": 3,
            "active_source": "https://en.wikipedia.org/wiki/Motorcycle",
            "source_match": True,
            "doc_keyword_match": False
        }
        mock_generate.return_value = '{"tools": ["knowledge_base"], "reasoning": "related to motorcycles"}'
        result = agent._analyze_intent("First motorcycle companies are?", source="https://en.wikipedia.org/wiki/Motorcycle")
        assert "knowledge_base" in result["tools"]

        # 2. Query: "Who owns Oscar statuettes?" -> expected knowledge_base
        mock_evidence.return_value = {
            "doc_count": 1050,
            "top_similarity": 0.82,
            "top_source": "Oscar_Wikipedia.pdf",
            "top_preview": "statuettes are owned by academy...",
            "retrieved_chunk_count": 3,
            "active_source": None,
            "source_match": True,
            "doc_keyword_match": False
        }
        mock_generate.return_value = '{"tools": ["knowledge_base"], "reasoning": "related to Oscars"}'
        result = agent._analyze_intent("Who owns Oscar statuettes?")
        assert "knowledge_base" in result["tools"]

        # 3. Query: "Tell me about snails" -> expected knowledge_base
        mock_evidence.return_value = {
            "doc_count": 1050,
            "top_similarity": 0.75,
            "top_source": "Snail_Wikipedia.pdf",
            "top_preview": "snails are gastropod molluscs...",
            "retrieved_chunk_count": 3,
            "active_source": None,
            "source_match": True,
            "doc_keyword_match": False
        }
        mock_generate.return_value = '{"tools": ["knowledge_base"], "reasoning": "related to snails"}'
        result = agent._analyze_intent("Tell me about snails")
        assert "knowledge_base" in result["tools"]

        # 4. Query: "What is Python?" -> expected web_search
        mock_evidence.return_value = {
            "doc_count": 1050,
            "top_similarity": 0.12,
            "top_source": "Snail_Wikipedia.pdf",
            "top_preview": "Some text...",
            "retrieved_chunk_count": 1,
            "active_source": None,
            "source_match": False,
            "doc_keyword_match": False
        }
        mock_generate.return_value = '{"tools": ["web_search"], "reasoning": "general python query"}'
        result = agent._analyze_intent("What is Python?")
        assert "web_search" in result["tools"]

        # 5. Query: "45+27" -> expected calculator
        mock_generate.return_value = '{"tools": ["calculator"], "reasoning": "math expression"}'
        result = agent._analyze_intent("45+27")
        assert "calculator" in result["tools"]
