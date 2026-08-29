"""Read-only Mubit boundary used by the STATE-Bench learning hook."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from typing import Any

from mubit_state_bench.config import MubitStateBenchConfig
from mubit_state_bench.telemetry import JsonlTelemetrySink
from state_bench.agents.base import AgentRuntimeContext


class MubitReadOnlyStore:
    """Retrieve global procedural lessons without exposing mutation methods."""

    def __init__(
        self,
        *,
        client: Any,
        config: MubitStateBenchConfig,
        runtime_context: AgentRuntimeContext,
        telemetry: JsonlTelemetrySink,
    ):
        self._client = client
        self._config = config
        self._runtime_context = runtime_context
        self._telemetry = telemetry

    @classmethod
    def from_env(cls, runtime_context: AgentRuntimeContext) -> "MubitReadOnlyStore":
        from mubit import Client

        config = MubitStateBenchConfig.from_env(runtime_context)
        client = Client(
            endpoint=config.endpoint,
            transport=config.transport,
            run_id=config.run_id,
            api_key=config.api_key,
        )
        return cls(
            client=client,
            config=config,
            runtime_context=runtime_context,
            telemetry=JsonlTelemetrySink(config.telemetry_path),
        )

    def retrieve(self, query: str, top_k: int) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Mubit retrieval query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("Mubit retrieval top_k must be an integer >= 1")

        started = time.perf_counter()
        base_event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "mubit_retrieval",
            "domain": self._runtime_context.domain,
            "task_id": self._runtime_context.task_id,
            "run_id": self._config.run_id,
            "experiment_id": self._config.experiment_id,
            "arm": self._config.arm,
            "artifact_sha256": self._config.artifact_sha256,
            "run_number": self._config.run_number,
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "requested_top_k": top_k,
            "request": {
                "mode": "direct_bypass",
                "direct_lane": "semantic_search",
                "evidence_only": True,
                "entry_types": ["lesson"],
                "include_linked_runs": False,
                "include_working_memory": False,
                "budget": self._config.budget,
                "rank_by": self._config.rank_by,
                "limit": top_k,
            },
        }

        try:
            response = self._client.recall(
                session_id=self._config.run_id,
                query=query,
                mode="direct_bypass",
                direct_lane="semantic_search",
                include_linked_runs=False,
                limit=top_k,
                entry_types=["lesson"],
                include_working_memory=False,
                budget=self._config.budget,
                rank_by=self._config.rank_by,
                explain=True,
                prefer_current_run=False,
                evidence_only=True,
            )
        except Exception as exc:
            self._telemetry.write(
                {
                    **base_event,
                    "status": "error",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        raw_evidence = response.get("evidence", []) if isinstance(response, dict) else []
        evidence = raw_evidence if isinstance(raw_evidence, list) else []
        contents: list[str] = []
        telemetry_evidence: list[dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if len(contents) >= top_k:
                break
            contents.append(content)
            telemetry_evidence.append(
                {
                    "id": item.get("id"),
                    "reference_id": item.get("reference_id"),
                    "entry_type": item.get("entry_type"),
                    "retrieval_mode": item.get("retrieval_mode"),
                    "score": item.get("score"),
                    "knowledge_confidence": item.get("knowledge_confidence"),
                    "is_stale": item.get("is_stale", False),
                    "content": content,
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )

        self._telemetry.write(
            {
                **base_event,
                "status": "ok",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "response_mode": response.get("mode") if isinstance(response, dict) else None,
                "returned_count": len(contents),
                "evidence": telemetry_evidence,
            }
        )
        return contents
