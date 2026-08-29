"""Deterministic frozen lesson artifacts derived from raw Mubit reflections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.learning import RAW_REFLECTION_SCHEMA
from mubit_state_bench.trajectory import DECISION_TURN_SCHEMA, TrainTrajectoryLoader, sha256_text

FROZEN_ARTIFACT_SCHEMA = "frozen_mubit_lessons_v1"


def _validate_reflection_record(record: Any, domain: str, loader: TrainTrajectoryLoader) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != RAW_REFLECTION_SCHEMA:
        raise ValueError("Raw reflection has an unsupported schema")
    if record.get("domain") != domain:
        raise ValueError("Raw reflection domain does not match the artifact domain")
    task_id = record.get("task_id")
    source_path = record.get("source_path")
    if not isinstance(task_id, str) or not isinstance(source_path, str):
        raise ValueError("Raw reflection is missing task/source provenance")
    expected_relative = f"datasets/train_task_trajectories/{domain}/{task_id}.json"
    if source_path != expected_relative:
        raise ValueError(f"Raw reflection source is not the checked-in train path: {source_path}")
    source = loader.load_path(domain, loader.repo_root / source_path)
    if source.source_sha256 != record.get("source_sha256"):
        raise ValueError(f"Raw reflection source hash mismatch for {task_id}")
    raw_response = record.get("raw_reflection_response")
    response_sha256 = record.get("raw_reflection_response_sha256")
    if raw_response is not None and canonical_sha256(raw_response) != response_sha256:
        raise ValueError(f"Raw reflection response hash mismatch for {task_id}")


def _lesson_is_degraded(lesson: dict[str, Any]) -> bool:
    if lesson.get("degraded") is True:
        return True
    return str(lesson.get("status", "")).lower() in {"degraded", "failed", "error"}


def build_frozen_artifact(
    *,
    domain: str,
    experiment_id: str,
    reflection_dir: Path,
    output_path: Path,
    loader: TrainTrajectoryLoader | None = None,
) -> dict[str, Any]:
    loader = loader or TrainTrajectoryLoader()
    reflection_paths = sorted(reflection_dir.glob("*.json"), key=lambda path: path.name)
    if not reflection_paths:
        raise ValueError(f"No raw reflections found in {reflection_dir}")

    lessons_by_exact_content: dict[str, dict[str, Any]] = {}
    excluded_reflections: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for reflection_path in reflection_paths:
        record = json.loads(reflection_path.read_text(encoding="utf-8"))
        _validate_reflection_record(record, domain, loader)
        task_id = record["task_id"]
        if task_id in seen_tasks:
            raise ValueError(f"Duplicate raw reflection task: {task_id}")
        seen_tasks.add(task_id)
        if record.get("status") != "ok":
            excluded_reflections.append(
                {
                    "task_id": task_id,
                    "status": record.get("status"),
                    "reasons": record.get("degraded_reasons") or [record.get("error") or "unknown"],
                }
            )
            continue

        response = record["raw_reflection_response"]
        for lesson_index, lesson in enumerate(response["lessons"]):
            if not isinstance(lesson, dict) or _lesson_is_degraded(lesson):
                continue
            content = lesson.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            provenance = {
                "task_id": task_id,
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "reflection_file": reflection_path.name,
                "reflection_response_sha256": record["raw_reflection_response_sha256"],
                "reflection_lesson_index": lesson_index,
                "mubit_lesson_id": lesson.get("lesson_id"),
            }
            existing = lessons_by_exact_content.get(content)
            if existing is not None:
                existing["provenance"].append(provenance)
                continue
            conditions = lesson.get("conditions")
            if not isinstance(conditions, list) or any(not isinstance(item, str) for item in conditions):
                conditions = []
            lessons_by_exact_content[content] = {
                "content": content,
                "content_sha256": sha256_text(content),
                "lesson_type": lesson.get("lesson_type") or "observation",
                "importance": lesson.get("importance") or "medium",
                "conditions": conditions,
                "provenance": [provenance],
            }

    payload: dict[str, Any] = {
        "schema_version": FROZEN_ARTIFACT_SCHEMA,
        "parser_schema_version": DECISION_TURN_SCHEMA,
        "source_reflection_schema_version": RAW_REFLECTION_SCHEMA,
        "experiment_id": experiment_id,
        "domain": domain,
        "reflection_count": len(reflection_paths),
        "excluded_reflections": excluded_reflections,
        "lesson_count": len(lessons_by_exact_content),
        "lessons": list(lessons_by_exact_content.values()),
    }
    artifact = {**payload, "artifact_sha256": canonical_sha256(payload)}
    write_json_atomic(output_path, artifact)
    return artifact


def load_frozen_artifact(path: Path, *, expected_domain: str | None = None) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != FROZEN_ARTIFACT_SCHEMA:
        raise ValueError("Frozen lesson artifact has an unsupported schema")
    artifact_sha256 = artifact.get("artifact_sha256")
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if canonical_sha256(payload) != artifact_sha256:
        raise ValueError("Frozen lesson artifact SHA-256 verification failed")
    if expected_domain is not None and artifact.get("domain") != expected_domain:
        raise ValueError("Frozen lesson artifact domain mismatch")
    lessons = artifact.get("lessons")
    if not isinstance(lessons, list) or artifact.get("lesson_count") != len(lessons):
        raise ValueError("Frozen lesson artifact lesson count is invalid")
    return artifact
