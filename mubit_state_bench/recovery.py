"""Read-only recovery of a publication manifest after a post-audit mismatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mubit_state_bench.artifact import load_frozen_artifact
from mubit_state_bench.config import MubitCredentialRole, MubitStateBenchConfig
from mubit_state_bench.io_utils import write_json_atomic
from mubit_state_bench.remote_audit import MubitEvalAuditor


def recover_publication_manifest(
    *,
    artifact_path: Path,
    publication_path: Path,
    config: MubitStateBenchConfig,
    auditor: MubitEvalAuditor,
    original_clean_preflight: dict[str, int],
    original_durable_write_count: int,
) -> dict[str, Any]:
    """Recover a local manifest using remote reads only; never submit memory writes."""

    if config.role is not MubitCredentialRole.EVAL:
        raise ValueError("Publication recovery requires an EVAL credential/config")
    if original_clean_preflight != {"visible_activity_count": 0, "global_lesson_count": 0}:
        raise ValueError("Publication recovery requires the original strict clean-preflight result")
    if publication_path.exists():
        raise FileExistsError(f"Publication manifest already exists: {publication_path}")

    artifact = load_frozen_artifact(artifact_path, expected_domain=config.domain)
    if artifact["artifact_sha256"] != config.artifact_sha256:
        raise ValueError("Recovery config artifact SHA does not match the frozen artifact")
    if artifact["lesson_set_sha256"] != config.lesson_set_sha256:
        raise ValueError("Recovery config lesson-set SHA does not match the frozen artifact")
    if original_durable_write_count != artifact["lesson_count"]:
        raise ValueError("Original durable write count does not match the frozen artifact")

    post_publication_audit = auditor.await_exact_lesson_set(
        {lesson["content_sha256"] for lesson in artifact["lessons"]}
    )
    if post_publication_audit["global_lesson_count"] != artifact["lesson_count"]:
        raise ValueError("Read-only recovery audit lesson count does not match the frozen artifact")

    manifest = {
        "schema_version": "mubit_artifact_publication_recovery_v1",
        "publication_status": "recovered_after_strict_activity_audit_mismatch",
        "recovery_remote_access": "read_only",
        "domain": config.domain,
        "experiment_id": config.experiment_id,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact["artifact_sha256"],
        "lesson_set_sha256": artifact["lesson_set_sha256"],
        "clean_eval_instance_confirmed": True,
        "remote_cleanliness_preflight": {
            **original_clean_preflight,
            "evidence": "observed during the original publication attempt before any write",
        },
        "published_count": artifact["lesson_count"],
        "durable_write_count": original_durable_write_count,
        "durable_receipts_persisted": False,
        "durability_evidence": (
            "the original publisher reached post-publication audit, which is called only after every durable write"
        ),
        "recovery_reason": (
            "the original audit required total ListActivity count to equal the global lesson count; "
            "Mubit also returned server-generated auto-reflection activity history"
        ),
        "remote_post_publication_audit": post_publication_audit,
    }
    write_json_atomic(publication_path, manifest)
    return manifest
