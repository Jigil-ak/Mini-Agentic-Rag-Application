"""
Structured JSONL trace logger for every agent invocation.

Each query produces one trace record containing routing decisions,
model usage, latency, and error information. Traces are appended
as single JSON lines to logs/agent_logs.jsonl.
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import LOG_FILE

logger = logging.getLogger(__name__)


class TraceManager:
    """Manages lifecycle of a single query trace: start → update → finish."""

    def __init__(self) -> None:
        self._active_traces: Dict[str, Dict[str, Any]] = {}
        self._log_path = Path(LOG_FILE)

    # ── Public API ─────────────────────────────────────────────────────────

    def start_trace(self, query: str) -> str:
        """Begin a new trace for a query. Returns a unique trace_id."""
        trace_id = str(uuid.uuid4())
        self._active_traces[trace_id] = {
            "trace_id": trace_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_hit": False,
            "similarity_score": 0.0,
            "path_taken": "",
            "routing_reason": "",
            "tool_used": None,
            "tool_output_preview": None,
            "primary_model": "",
            "fallback_triggered": False,
            "chunks_used": None,
            "response_time_ms": 0.0,
            "error": None,
            "_start_time": time.perf_counter(),
        }
        return trace_id

    def update_trace(self, trace_id: str, **kwargs: Any) -> None:
        """Update fields on an in-progress trace.

        Accepts any key that matches the trace schema. Unknown keys
        are silently ignored so callers don't need to worry about
        forward-compatibility.
        """
        trace = self._active_traces.get(trace_id)
        if trace is None:
            logger.warning("update_trace called with unknown trace_id=%s", trace_id)
            return

        for key, value in kwargs.items():
            if key in trace and not key.startswith("_"):
                trace[key] = value

    def finish_trace(self, trace_id: str) -> Dict[str, Any]:
        """Finalise the trace: compute latency, persist to JSONL, return record.

        Raises KeyError if the trace_id is not found (programming error).
        """
        trace = self._active_traces.pop(trace_id, None)
        if trace is None:
            raise KeyError(f"No active trace with id={trace_id}")

        # Compute elapsed time
        start = trace.pop("_start_time")
        trace["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # Truncate tool_output_preview to 200 chars
        preview = trace.get("tool_output_preview")
        if preview and len(preview) > 200:
            trace["tool_output_preview"] = preview[:200]

        # Persist
        self._write_jsonl(trace)
        return trace

    def get_recent_traces(self, n: int = 20) -> List[Dict[str, Any]]:
        """Read the last *n* traces from the JSONL log file.

        Returns an empty list if the log file doesn't exist yet.
        """
        if not self._log_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSONL line")
                        continue
        except OSError as exc:
            logger.error("Failed to read trace log: %s", exc)
            return []

        # Return the most recent n traces (tail of file)
        return traces[-n:]

    # ── Internal ───────────────────────────────────────────────────────────

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        """Append a single JSON line to the log file, creating dirs if needed."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to write trace: %s", exc)
