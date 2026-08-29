"""Publish only a verified frozen artifact into a clean EVAL instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mubit_state_bench.artifact import load_frozen_artifact
from mubit_state_bench.config import MubitCredentialRole, MubitStateBenchConfig
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic


class FrozenArtifactPublisher:
    def __init__(self, *, client: Any, config: MubitStateBenchConfig):
        if config.role is not MubitCredentialRole.EVAL:
            raise ValueError("FrozenArtifactPublisher requires an EVAL credential/config")
        self._client = client
        self._config = config

    def publish(
        self,
        artifact_path: Path,
        publication_path: Path,
        *,
        clean_eval_instance_confirmed: bool,
    ) -> dict[str, Any]:
        if not clean_eval_instance_confirmed:
            raise ValueError("Publication requires explicit confirmation that the isolated EVAL instance is clean")
        artifact = load_frozen_artifact(artifact_path, expected_domain=self._config.domain)
        if artifact["artifact_sha256"] != self._config.artifact_sha256:
            raise ValueError("Publication config artifact SHA does not match the frozen artifact")
        if publication_path.exists():
            raise FileExistsError(
                f"Publication manifest already exists: {publication_path}. "
                "Refusing to publish into the EVAL instance twice."
            )

        published: list[dict[str, Any]] = []
        for lesson in artifact["lessons"]:
            content_sha256 = lesson["content_sha256"]
            stable_key = f"statebench:{self._config.domain}:frozen:{content_sha256}"
            idempotency_sha256 = canonical_sha256(
                {
                    "artifact_sha256": artifact["artifact_sha256"],
                    "stable_key": stable_key,
                }
            )
            response = self._client.remember(
                session_id=self._config.run_id,
                agent_id="statebench-phase2-publisher",
                item_id=f"frozen-lesson-{content_sha256[:24]}",
                content=lesson["content"],
                intent="lesson",
                lesson_type=lesson["lesson_type"],
                lesson_scope="global",
                lesson_importance=lesson["importance"],
                lesson_conditions=lesson["conditions"],
                source="statebench-frozen-artifact",
                upsert_key=stable_key,
                idempotency_key=f"statebench-phase2-publish:{idempotency_sha256}",
                metadata={
                    "benchmark": "microsoft-state-bench",
                    "phase": "phase2",
                    "domain": self._config.domain,
                    "experiment_id": self._config.experiment_id,
                    "artifact_sha256": artifact["artifact_sha256"],
                    "content_sha256": content_sha256,
                    "provenance": lesson["provenance"],
                },
                wait=True,
            )
            published.append(
                {
                    "content_sha256": content_sha256,
                    "accepted": bool(response),
                }
            )

        manifest = {
            "schema_version": "mubit_artifact_publication_v1",
            "domain": self._config.domain,
            "experiment_id": self._config.experiment_id,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "clean_eval_instance_confirmed": True,
            "published_count": len(published),
            "published": published,
        }
        write_json_atomic(publication_path, manifest)
        return manifest
