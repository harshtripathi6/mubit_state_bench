"""Publish only a verified frozen artifact into a clean EVAL instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mubit_state_bench.artifact import load_frozen_artifact
from mubit_state_bench.config import MubitCredentialRole, MubitStateBenchConfig
from mubit_state_bench.durability import MubitDurableWriter
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.remote_audit import MubitEvalAuditor


class FrozenArtifactPublisher:
    def __init__(
        self,
        *,
        client: Any,
        config: MubitStateBenchConfig,
        durable_writer: MubitDurableWriter | None = None,
        auditor: MubitEvalAuditor | None = None,
    ):
        if config.role is not MubitCredentialRole.EVAL:
            raise ValueError("FrozenArtifactPublisher requires an EVAL credential/config")
        self._client = client
        self._config = config
        self._durable_writer = durable_writer or MubitDurableWriter(client)
        self._auditor = auditor or MubitEvalAuditor(client)

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
        if artifact["lesson_set_sha256"] != self._config.lesson_set_sha256:
            raise ValueError("Publication config lesson-set SHA does not match the frozen artifact")
        if publication_path.exists():
            raise FileExistsError(
                f"Publication manifest already exists: {publication_path}. "
                "Refusing to publish into the EVAL instance twice."
            )

        preflight = self._auditor.assert_clean()
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
            item_id = f"frozen-lesson-{content_sha256[:24]}"
            receipt = self._durable_writer.remember_durable(
                expected_item_id=item_id,
                expected_memory_type="lesson",
                session_id=self._config.run_id,
                agent_id="statebench-phase2-publisher",
                item_id=item_id,
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
                    "lesson_set_sha256": artifact["lesson_set_sha256"],
                    "content_sha256": content_sha256,
                    "provenance": lesson["provenance"],
                },
            )
            published.append(
                {
                    "content_sha256": content_sha256,
                    "durable_receipt": receipt.to_dict(),
                }
            )

        post_publication_audit = self._auditor.await_exact_lesson_set(
            {lesson["content_sha256"] for lesson in artifact["lessons"]}
        )

        manifest = {
            "schema_version": "mubit_artifact_publication_v2",
            "domain": self._config.domain,
            "experiment_id": self._config.experiment_id,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "lesson_set_sha256": artifact["lesson_set_sha256"],
            "clean_eval_instance_confirmed": True,
            "remote_cleanliness_preflight": preflight,
            "published_count": len(published),
            "published": published,
            "remote_post_publication_audit": post_publication_audit,
        }
        write_json_atomic(publication_path, manifest)
        return manifest
