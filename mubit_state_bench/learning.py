"""Offline Mubit ingestion and one-shot reflection for training trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mubit_state_bench.config import (
    MubitCredentialRole,
    MubitStateBenchConfig,
    redact_configured_secrets,
)
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.trajectory import (
    DECISION_TURN_SCHEMA,
    TrainTrajectoryLoader,
    TrainTrajectorySource,
    parse_decision_turns,
    sha256_text,
)

RAW_REFLECTION_SCHEMA = "raw_mubit_reflection_v1"


def _reflection_quality(response: Any) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not isinstance(response, dict):
        return "degraded", ["reflection response is not an object"]
    if response.get("degraded") is True:
        reasons.append("Mubit marked the reflection degraded")
    status = str(response.get("status", "")).lower()
    if status in {"failed", "error"}:
        reasons.append(f"Mubit reflection status is {status}")
    if not isinstance(response.get("lessons"), list):
        reasons.append("reflection response has no lessons list")
    return ("degraded", reasons) if reasons else ("ok", [])


class MubitTrajectoryLearner:
    """Write deterministic decision turns, then explicitly reflect once."""

    def __init__(
        self,
        *,
        client: Any,
        config: MubitStateBenchConfig,
        loader: TrainTrajectoryLoader,
    ):
        if config.role is not MubitCredentialRole.BUILD:
            raise ValueError("MubitTrajectoryLearner requires a BUILD credential/config")
        self._client = client
        self._config = config
        self._loader = loader

    def learn(self, source: TrainTrajectorySource, output_path: Path) -> dict[str, Any]:
        self._loader.assert_train_source(source)
        parsed = parse_decision_turns(source)
        base_record: dict[str, Any] = {
            "schema_version": RAW_REFLECTION_SCHEMA,
            "parser_schema_version": DECISION_TURN_SCHEMA,
            "domain": source.domain,
            "task_id": source.task_id,
            "source_path": source.source_relative_path,
            "source_sha256": source.source_sha256,
            "run_id": self._config.run_id,
            "terminal_marker_seen": parsed.terminal_marker_seen,
            "terminal_marker_interpreted_as_outcome": False,
            "parsed_turn_count": len(parsed.turns),
        }
        ingested_count = 0
        try:
            for turn in parsed.turns:
                content = turn.canonical_content()
                content_sha256 = sha256_text(content)
                idempotency_sha256 = canonical_sha256(
                    {
                        "run_id": self._config.run_id,
                        "source_sha256": source.source_sha256,
                        "turn_index": turn.turn_index,
                        "content_sha256": content_sha256,
                    }
                )
                self._client.remember(
                    session_id=self._config.run_id,
                    agent_id="statebench-phase2-builder",
                    item_id=f"decision-turn-{turn.turn_index:03d}",
                    content=content,
                    intent="trace",
                    source="statebench-train-trajectory",
                    upsert_key=(f"{source.domain}:{source.task_id}:{DECISION_TURN_SCHEMA}:{turn.turn_index:03d}"),
                    idempotency_key=f"statebench-phase2:{idempotency_sha256}",
                    metadata={
                        "benchmark": "microsoft-state-bench",
                        "phase": "phase2",
                        "domain": source.domain,
                        "task_id": source.task_id,
                        "source_path": source.source_relative_path,
                        "source_sha256": source.source_sha256,
                        "parser_schema_version": DECISION_TURN_SCHEMA,
                        "turn_index": turn.turn_index,
                        "content_sha256": content_sha256,
                    },
                    wait=True,
                )
                ingested_count += 1

            reflection = self._client.reflect(
                session_id=self._config.run_id,
                include_linked_runs=False,
                last_n_items=len(parsed.turns),
            )
        except Exception as exc:
            record = {
                **base_record,
                "status": "failed",
                "failure_stage": "ingest" if ingested_count < len(parsed.turns) else "reflect",
                "ingested_item_count": ingested_count,
                "degraded_reasons": [],
                "raw_reflection_response": None,
                "raw_reflection_response_sha256": None,
                "error_type": type(exc).__name__,
                "error": redact_configured_secrets(str(exc)),
            }
            write_json_atomic(output_path, record)
            return record

        status, degraded_reasons = _reflection_quality(reflection)
        reflection_sha256 = canonical_sha256(reflection)
        record = {
            **base_record,
            "status": status,
            "failure_stage": None,
            "ingested_item_count": ingested_count,
            "degraded_reasons": degraded_reasons,
            "raw_reflection_response": reflection,
            "raw_reflection_response_sha256": reflection_sha256,
            "error_type": None,
            "error": None,
        }
        write_json_atomic(output_path, record)
        return record
