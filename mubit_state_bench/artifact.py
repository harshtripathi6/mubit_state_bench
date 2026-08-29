"""Deterministic frozen lesson artifacts derived from raw Mubit reflections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.learning import RAW_REFLECTION_SCHEMA, validate_raw_reflection_record
from mubit_state_bench.trajectory import DECISION_TURN_SCHEMA, TrainTrajectoryLoader, sha256_text

FROZEN_ARTIFACT_SCHEMA = "frozen_mubit_lessons_v2"


def _validate_reflection_record(
    record: Any,
    domain: str,
    experiment_id: str,
    reflection_path: Path,
    loader: TrainTrajectoryLoader,
) -> None:
    if not isinstance(record, dict):
        raise ValueError("Raw reflection is not an object")
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or reflection_path.name != f"{task_id}.json":
        raise ValueError("Raw reflection filename/task provenance mismatch")
    source = loader.load_path(domain, loader.domain_root(domain) / f"{task_id}.json")
    validate_raw_reflection_record(
        record,
        source=source,
        expected_experiment_id=experiment_id,
    )


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
    official_full_training_set: bool,
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
        _validate_reflection_record(record, domain, experiment_id, reflection_path, loader)
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

    if official_full_training_set:
        expected_tasks = {path.stem for path in loader.paths(domain)}
        if len(expected_tasks) != 100:
            raise ValueError(
                f"Official artifact requires exactly 100 checked-in training tasks; found {len(expected_tasks)}"
            )
        missing = sorted(expected_tasks - seen_tasks)
        extra = sorted(seen_tasks - expected_tasks)
        if missing or extra:
            raise ValueError(
                f"Official artifact training-set coverage mismatch: missing={missing or []}, extra={extra or []}"
            )

    lessons = list(lessons_by_exact_content.values())
    lesson_set_sha256 = canonical_sha256(sorted(lesson["content"] for lesson in lessons))

    payload: dict[str, Any] = {
        "schema_version": FROZEN_ARTIFACT_SCHEMA,
        "parser_schema_version": DECISION_TURN_SCHEMA,
        "source_reflection_schema_version": RAW_REFLECTION_SCHEMA,
        "experiment_id": experiment_id,
        "domain": domain,
        "training_set_mode": "official_full" if official_full_training_set else "partial_pilot",
        "reflection_count": len(reflection_paths),
        "excluded_reflections": excluded_reflections,
        "lesson_count": len(lessons),
        "lesson_set_sha256": lesson_set_sha256,
        "lessons": lessons,
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
    contents: list[str] = []
    for lesson in lessons:
        if not isinstance(lesson, dict) or not isinstance(lesson.get("content"), str):
            raise ValueError("Frozen lesson artifact contains an invalid lesson")
        if sha256_text(lesson["content"]) != lesson.get("content_sha256"):
            raise ValueError("Frozen lesson content SHA-256 verification failed")
        contents.append(lesson["content"])
    if canonical_sha256(sorted(contents)) != artifact.get("lesson_set_sha256"):
        raise ValueError("Frozen lesson-set SHA-256 verification failed")
    return artifact
