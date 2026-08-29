"""Offline Mubit ingestion and one-shot reflection for training trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mubit_state_bench.config import (
    MubitCredentialRole,
    MubitStateBenchConfig,
    redact_configured_secrets,
)
from mubit_state_bench.durability import DURABLE_WRITE_RECEIPT_SCHEMA, MubitDurableWriter
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.trajectory import (
    DECISION_TURN_SCHEMA,
    TrainTrajectoryLoader,
    TrainTrajectorySource,
    parse_decision_turns,
    sha256_text,
)

RAW_REFLECTION_SCHEMA = "raw_mubit_reflection_v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        durable_writer: MubitDurableWriter | None = None,
    ):
        if config.role is not MubitCredentialRole.BUILD:
            raise ValueError("MubitTrajectoryLearner requires a BUILD credential/config")
        self._client = client
        self._config = config
        self._loader = loader
        self._durable_writer = durable_writer or MubitDurableWriter(client)

    def learn(self, source: TrainTrajectorySource, output_path: Path) -> dict[str, Any]:
        self._loader.assert_train_source(source)
        if output_path.exists():
            raise FileExistsError(f"Raw reflection already exists and will not be overwritten: {output_path}")
        parsed = parse_decision_turns(source)
        base_record: dict[str, Any] = {
            "schema_version": RAW_REFLECTION_SCHEMA,
            "parser_schema_version": DECISION_TURN_SCHEMA,
            "domain": source.domain,
            "task_id": source.task_id,
            "source_path": source.source_relative_path,
            "source_sha256": source.source_sha256,
            "experiment_id": self._config.experiment_id,
            "run_id": self._config.run_id,
            "terminal_marker_seen": parsed.terminal_marker_seen,
            "terminal_marker_interpreted_as_outcome": False,
            "parsed_turn_count": len(parsed.turns),
        }
        durable_receipts: list[dict[str, Any]] = []
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
                item_id = f"decision-turn-{turn.turn_index:03d}"
                receipt = self._durable_writer.remember_durable(
                    expected_item_id=item_id,
                    session_id=self._config.run_id,
                    agent_id="statebench-phase2-builder",
                    item_id=item_id,
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
                )
                durable_receipts.append(receipt.to_dict())

            if len(durable_receipts) != len(parsed.turns):
                raise RuntimeError("Not every parsed trace has a durable Mubit write receipt")

            reflection = self._client.reflect(
                session_id=self._config.run_id,
                include_linked_runs=False,
                last_n_items=len(parsed.turns),
            )
        except Exception as exc:
            record = {
                **base_record,
                "status": "failed",
                "failure_stage": "ingest" if len(durable_receipts) < len(parsed.turns) else "reflect",
                "durable_ingested_item_count": len(durable_receipts),
                "durable_ingest_receipts": durable_receipts,
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
            "durable_ingested_item_count": len(durable_receipts),
            "durable_ingest_receipts": durable_receipts,
            "degraded_reasons": degraded_reasons,
            "raw_reflection_response": reflection,
            "raw_reflection_response_sha256": reflection_sha256,
            "error_type": None,
            "error": None,
        }
        write_json_atomic(output_path, record)
        return record


def validate_raw_reflection_record(
    record: Any,
    *,
    source: TrainTrajectorySource,
    expected_experiment_id: str,
    expected_run_id: str | None = None,
) -> None:
    """Fully validate a persisted reflection before reuse or artifact freezing."""

    if not isinstance(record, dict) or record.get("schema_version") != RAW_REFLECTION_SCHEMA:
        raise ValueError("Raw reflection has an unsupported schema")
    parsed = parse_decision_turns(source)
    expected = {
        "parser_schema_version": DECISION_TURN_SCHEMA,
        "domain": source.domain,
        "task_id": source.task_id,
        "source_path": source.source_relative_path,
        "source_sha256": source.source_sha256,
        "experiment_id": expected_experiment_id,
        "terminal_marker_seen": parsed.terminal_marker_seen,
        "terminal_marker_interpreted_as_outcome": False,
        "parsed_turn_count": len(parsed.turns),
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(f"Raw reflection {field} mismatch for {source.task_id}")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Raw reflection run_id is invalid for {source.task_id}")
    expected_run_suffix = f":{source.domain}:train:{expected_experiment_id}:{source.task_id}"
    if not run_id.endswith(expected_run_suffix):
        raise ValueError(f"Raw reflection run_id provenance mismatch for {source.task_id}")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"Raw reflection run_id mismatch for {source.task_id}")

    receipts = record.get("durable_ingest_receipts")
    if not isinstance(receipts, list):
        raise ValueError(f"Raw reflection durable receipts are missing for {source.task_id}")
    expected_item_ids = [f"decision-turn-{turn.turn_index:03d}" for turn in parsed.turns]
    if [receipt.get("item_id") if isinstance(receipt, dict) else None for receipt in receipts] != expected_item_ids[
        : len(receipts)
    ]:
        raise ValueError(f"Raw reflection durable receipt order mismatch for {source.task_id}")
    for receipt in receipts:
        if receipt.get("schema_version") != DURABLE_WRITE_RECEIPT_SCHEMA:
            raise ValueError(f"Raw reflection durable receipt schema mismatch for {source.task_id}")
        if not isinstance(receipt.get("job_id"), str) or not receipt["job_id"]:
            raise ValueError(f"Raw reflection durable receipt job ID is invalid for {source.task_id}")
        if (
            not isinstance(receipt.get("record_ids"), list)
            or not receipt["record_ids"]
            or any(not isinstance(value, str) or not value for value in receipt["record_ids"])
        ):
            raise ValueError(f"Raw reflection durable receipt record IDs are invalid for {source.task_id}")
        if (
            not isinstance(receipt.get("storage_memory_types"), list)
            or len(receipt["storage_memory_types"]) != len(receipt["record_ids"])
            or any(not isinstance(value, str) or not value for value in receipt["storage_memory_types"])
        ):
            raise ValueError(f"Raw reflection durable storage types are invalid for {source.task_id}")
        job_sha256 = receipt.get("job_sha256")
        if not isinstance(job_sha256, str) or not _SHA256.fullmatch(job_sha256):
            raise ValueError(f"Raw reflection durable receipt hash is invalid for {source.task_id}")
        if not isinstance(receipt.get("submission_deduplicated"), bool) or not isinstance(
            receipt.get("job_deduplicated"), bool
        ):
            raise ValueError(f"Raw reflection durable receipt deduplication fields are invalid for {source.task_id}")
    if record.get("durable_ingested_item_count") != len(receipts):
        raise ValueError(f"Raw reflection durable item count mismatch for {source.task_id}")

    status = record.get("status")
    if status not in {"ok", "degraded", "failed"}:
        raise ValueError(f"Raw reflection status is invalid for {source.task_id}")
    response = record.get("raw_reflection_response")
    response_sha256 = record.get("raw_reflection_response_sha256")
    if status in {"ok", "degraded"}:
        if len(receipts) != len(parsed.turns):
            raise ValueError(f"Raw reflection is missing durable receipts for {source.task_id}")
        if canonical_sha256(response) != response_sha256:
            raise ValueError(f"Raw reflection response hash mismatch for {source.task_id}")
        quality, reasons = _reflection_quality(response)
        if status != quality or record.get("degraded_reasons") != reasons:
            raise ValueError(f"Raw reflection quality fields mismatch for {source.task_id}")
        if record.get("failure_stage") is not None or record.get("error") is not None:
            raise ValueError(f"Raw reflection success/failure fields conflict for {source.task_id}")
        if record.get("error_type") is not None:
            raise ValueError(f"Raw reflection success contains an error type for {source.task_id}")
    else:
        if record.get("failure_stage") not in {"ingest", "reflect"}:
            raise ValueError(f"Raw reflection failure stage is invalid for {source.task_id}")
        if record.get("failure_stage") == "reflect" and len(receipts) != len(parsed.turns):
            raise ValueError(f"Raw reflection reflect failure lacks durable receipts for {source.task_id}")
        if record.get("failure_stage") == "ingest" and len(receipts) >= len(parsed.turns):
            raise ValueError(f"Raw reflection ingest failure has inconsistent durable receipts for {source.task_id}")
        if response is not None or response_sha256 is not None:
            raise ValueError(f"Raw reflection failure contains a reflection response for {source.task_id}")
        if not isinstance(record.get("error_type"), str) or not isinstance(record.get("error"), str):
            raise ValueError(f"Raw reflection failure details are invalid for {source.task_id}")
