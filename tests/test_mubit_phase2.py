from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mubit_state_bench.artifact import build_frozen_artifact, load_frozen_artifact
from mubit_state_bench.build_trajectories import build_training_reflections
from mubit_state_bench.config import (
    MubitCredentialRole,
    MubitStateBenchConfig,
    validate_credential_separation,
)
from mubit_state_bench.durability import IngestDurabilityError, MubitDurableWriter
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.learning import RAW_REFLECTION_SCHEMA, MubitTrajectoryLearner
from mubit_state_bench.phase2_paths import Phase2Paths
from mubit_state_bench.publication import FrozenArtifactPublisher
from mubit_state_bench.remote_audit import EvalAuditError, MubitEvalAuditor
from mubit_state_bench.trajectory import (
    DECISION_TURN_SCHEMA,
    TERMINAL_MARKER,
    TrainTrajectoryLoader,
    parse_decision_turns,
)
from state_bench.agents.base import AgentRuntimeContext


class RecordingBuildClient:
    def __init__(self, reflection: object):
        self.calls: list[tuple[str, dict]] = []
        self.reflection = reflection
        self.jobs: dict[str, dict] = {}
        self.advanced = SimpleNamespace(get_ingest_job=self.get_ingest_job)

    def remember(self, **kwargs):
        self.calls.append(("remember", kwargs))
        job_id = f"job-{len(self.jobs)}"
        self.jobs[job_id] = kwargs
        return {"accepted": True, "job_id": job_id, "status": "pending"}

    def get_ingest_job(self, **kwargs):
        self.calls.append(("get_ingest_job", kwargs))
        item_id = self.jobs[kwargs["job_id"]]["item_id"]
        return {
            "job_id": kwargs["job_id"],
            "run_id": kwargs["run_id"],
            "status": "completed",
            "done": True,
            "traces": [
                {
                    "item_id": item_id,
                    "writes": [{"memory_type": "knowledge", "record_id": f"record-{item_id}", "success": True}],
                }
            ],
        }

    def reflect(self, **kwargs):
        self.calls.append(("reflect", kwargs))
        if isinstance(self.reflection, Exception):
            raise self.reflection
        return self.reflection


class RecordingPublishClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def remember(self, **kwargs):
        self.calls.append(("remember", kwargs))
        return {"accepted": True, "job_id": "unused", "status": "pending"}


class RecordingDurableWriter:
    def __init__(self, calls: list[tuple[str, dict]]):
        self.calls = calls

    def remember_durable(self, **kwargs):
        self.calls.append(("remember_durable", kwargs))
        return SimpleNamespace(
            to_dict=lambda: {
                "schema_version": "mubit_durable_write_receipt_v2",
                "item_id": kwargs["expected_item_id"],
                "job_id": f"job-{kwargs['expected_item_id']}",
                "status": "completed",
                "submission_deduplicated": False,
                "job_deduplicated": False,
                "record_ids": [f"record-{kwargs['expected_item_id']}"],
                "storage_memory_types": ["knowledge"],
                "job_sha256": "d" * 64,
            }
        )


class RecordingAuditor:
    def __init__(self, calls: list[tuple[str, dict]]):
        self.calls = calls

    def assert_clean(self):
        self.calls.append(("preflight", {}))
        return {"visible_activity_count": 0, "global_lesson_count": 0}

    def await_exact_lesson_set(self, expected):
        self.calls.append(("post_audit", {"expected": expected}))
        return {
            "visible_activity_count": len(expected),
            "global_lesson_count": len(expected),
            "content_sha256s": sorted(expected),
        }


def _config(
    role: MubitCredentialRole,
    *,
    artifact_sha256: str = "",
    lesson_set_sha256: str = "",
    run_id: str = "statebench:travel:train:test:task",
) -> MubitStateBenchConfig:
    return MubitStateBenchConfig(
        domain="travel",
        api_key=f"{role.value}-test-key",
        endpoint="https://api.mubit.ai",
        transport="auto",
        namespace="statebench",
        run_id=run_id,
        telemetry_path=Path("unused.jsonl"),
        role=role,
        experiment_id="phase2-unit",
        arm=role.value,
        artifact_sha256=artifact_sha256,
        lesson_set_sha256=lesson_set_sha256,
        instance_id=f"{role.value}-instance",
    )


def _reflection_record(
    source,
    response: dict,
    *,
    status: str = "ok",
    reasons: list[str] | None = None,
    experiment_id: str = "phase2-unit",
):
    parsed = parse_decision_turns(source)
    receipts = [
        {
            "schema_version": "mubit_durable_write_receipt_v2",
            "item_id": f"decision-turn-{turn.turn_index:03d}",
            "job_id": f"job-{turn.turn_index}",
            "status": "completed",
            "submission_deduplicated": False,
            "job_deduplicated": False,
            "record_ids": [f"record-{turn.turn_index}"],
            "storage_memory_types": ["knowledge"],
            "job_sha256": f"{turn.turn_index % 10}" * 64,
        }
        for turn in parsed.turns
    ]
    return {
        "schema_version": RAW_REFLECTION_SCHEMA,
        "parser_schema_version": DECISION_TURN_SCHEMA,
        "domain": source.domain,
        "task_id": source.task_id,
        "source_path": source.source_relative_path,
        "source_sha256": source.source_sha256,
        "experiment_id": experiment_id,
        "run_id": f"statebench:{source.domain}:train:{experiment_id}:{source.task_id}",
        "terminal_marker_seen": parsed.terminal_marker_seen,
        "terminal_marker_interpreted_as_outcome": False,
        "parsed_turn_count": len(parsed.turns),
        "status": status,
        "failure_stage": None,
        "durable_ingested_item_count": len(receipts),
        "durable_ingest_receipts": receipts,
        "degraded_reasons": reasons or [],
        "raw_reflection_response": response,
        "raw_reflection_response_sha256": canonical_sha256(response),
        "error_type": None,
        "error": None,
    }


def _build_test_artifact(tmp_path: Path):
    loader = TrainTrajectoryLoader()
    first, second, third = loader.load("travel", limit=3)
    reflection_dir = tmp_path / "raw_reflections"
    response_one = {
        "degraded": False,
        "lessons": [
            {
                "lesson_id": "mubit-1",
                "content": "Preview a cancellation before confirming it.",
                "lesson_type": "rule",
                "importance": "high",
                "conditions": ["flight cancellation"],
            },
            {"content": "", "lesson_type": "observation"},
        ],
    }
    response_two = {
        "degraded": False,
        "lessons": [
            {
                "lesson_id": "mubit-2",
                "content": "Preview a cancellation before confirming it.",
                "lesson_type": "rule",
                "importance": "high",
                "conditions": ["flight cancellation"],
            },
            {"content": "Never silently substitute a different booking.", "status": "degraded"},
        ],
    }
    response_three = {"degraded": True, "lessons": [{"content": "Must not enter artifact."}]}
    write_json_atomic(reflection_dir / f"{first.task_id}.json", _reflection_record(first, response_one))
    write_json_atomic(reflection_dir / f"{second.task_id}.json", _reflection_record(second, response_two))
    write_json_atomic(
        reflection_dir / f"{third.task_id}.json",
        _reflection_record(
            third,
            response_three,
            status="degraded",
            reasons=["Mubit marked the reflection degraded"],
        ),
    )
    artifact_path = tmp_path / "frozen_lessons.json"
    artifact = build_frozen_artifact(
        domain="travel",
        experiment_id="phase2-unit",
        reflection_dir=reflection_dir,
        output_path=artifact_path,
        official_full_training_set=False,
        loader=loader,
    )
    return artifact, artifact_path, reflection_dir


def test_role_credentials_cannot_collapse_to_the_same_key(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_SMOKE_TRAVEL_API_KEY", "same-secret")
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "same-secret")

    with pytest.raises(ValueError, match="credential isolation violation") as exc_info:
        validate_credential_separation()

    assert "same-secret" not in str(exc_info.value)


def test_distinct_keys_cannot_collapse_to_the_same_instance(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_SMOKE_TRAVEL_API_KEY", "mbt_shared_key1_secret1")
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "mbt_shared_key2_secret2")

    with pytest.raises(ValueError, match="instance isolation violation") as exc_info:
        validate_credential_separation()

    assert "secret1" not in str(exc_info.value)
    assert "secret2" not in str(exc_info.value)


def test_explicit_instance_identity_must_match_hosted_key(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "mbt_actual_key_secret")
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_INSTANCE_ID", "different")

    with pytest.raises(ValueError, match="instance identity mismatch"):
        MubitStateBenchConfig.for_build("travel", "task-1", "phase2-unit")


def test_seed_build_and_eval_configs_accept_only_their_role_keys(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY", "eval-only")
    with pytest.raises(ValueError, match="SMOKE_TRAVEL_API_KEY"):
        MubitStateBenchConfig.for_seed("travel")
    with pytest.raises(ValueError, match="BUILD_TRAVEL_API_KEY"):
        MubitStateBenchConfig.for_build("travel", "task-1", "phase2-unit")


def test_eval_config_uses_eval_key_and_runtime_run_index(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY", "mbt_evalinstance_key_secret")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EXPERIMENT_ID", "phase2-eval")
    monkeypatch.setenv("MUBIT_STATE_BENCH_ARM", "mubit")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_LESSON_SET_SHA256", "b" * 64)
    monkeypatch.setenv("MUBIT_STATE_BENCH_RUN_NUMBER", "99")
    runtime_context = AgentRuntimeContext(
        task_id="task-1",
        user_id="user-1",
        domain="travel",
        now="2026-06-15T10:00:00",
        run_idx=4,
    )

    config = MubitStateBenchConfig.from_env(runtime_context)

    assert config.role is MubitCredentialRole.EVAL
    assert config.api_key == "mbt_evalinstance_key_secret"
    assert config.instance_id == "evalinstance"
    assert config.lesson_set_sha256 == "b" * 64
    assert config.run_number == 4
    assert config.run_id == "statebench:travel:eval:phase2-eval:run-4:task-1"
    assert config.telemetry_path == Path("outputs/phase2-eval/retrieval/travel.jsonl")


def test_train_loader_defaults_to_100_and_rejects_arbitrary_or_test_paths():
    loader = TrainTrajectoryLoader()

    assert len(loader.load("travel")) == 100
    assert len(loader.load("travel", limit=3)) == 3
    held_out_path = loader.repo_root / "state_bench" / "domains" / "travel" / "tasks" / "1-cancel_economy_domestic.json"
    with pytest.raises(ValueError, match="only datasets/train_task_trajectories"):
        loader.load_path("travel", held_out_path)
    with pytest.raises(ValueError, match="only datasets/train_task_trajectories"):
        loader.load_path("travel", Path("/tmp/arbitrary.json"))


def test_decision_turn_v1_is_deterministic_and_never_treats_task_done_as_success():
    source = TrainTrajectoryLoader().load("travel", limit=1)[0]

    first = parse_decision_turns(source)
    second = parse_decision_turns(source)

    assert [turn.canonical_content() for turn in first.turns] == [turn.canonical_content() for turn in second.turns]
    assert first.terminal_marker_seen is True
    assert all(TERMINAL_MARKER not in context for turn in first.turns for context in turn.user_context)
    assert all("success" not in turn.to_dict() and "outcome" not in turn.to_dict() for turn in first.turns)
    assert first.turns[0].user_context == ("Hi, I need to cancel my upcoming flight to Atlanta.",)
    assert first.turns[0].tool_calls[0].name == "get_user_reservations"
    assert first.turns[0].tool_calls[0].arguments == {"user_id": "user_002"}
    assert first.turns[0].tool_calls[0].result["booking_ids"] == ["BK-1000", "BK-1001"]


def test_learner_writes_ordered_trace_items_then_reflects_exactly_once(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    reflection = {
        "degraded": False,
        "lessons": [{"lesson_id": "lesson-1", "content": "Confirm before cancellation."}],
    }
    client = RecordingBuildClient(reflection)
    output_path = tmp_path / "reflection.json"
    learner = MubitTrajectoryLearner(client=client, config=_config(MubitCredentialRole.BUILD), loader=loader)

    record = learner.learn(source, output_path)

    parsed = parse_decision_turns(source)
    assert [name for name, _ in client.calls] == [
        item for _ in parsed.turns for item in ("remember", "get_ingest_job")
    ] + ["reflect"]
    remember_calls = [kwargs for name, kwargs in client.calls if name == "remember"]
    assert [json.loads(call["content"])["turn_index"] for call in remember_calls] == list(range(len(parsed.turns)))
    assert all(call["intent"] == "trace" and call["wait"] is False for call in remember_calls)
    assert all("outcome" not in call for call in remember_calls)
    reflect_calls = [kwargs for name, kwargs in client.calls if name == "reflect"]
    assert reflect_calls == [
        {
            "session_id": "statebench:travel:train:test:task",
            "include_linked_runs": False,
            "last_n_items": len(parsed.turns),
        }
    ]
    assert record["status"] == "ok"
    assert record["durable_ingested_item_count"] == len(parsed.turns)
    assert len(record["durable_ingest_receipts"]) == len(parsed.turns)
    assert record["terminal_marker_interpreted_as_outcome"] is False
    assert record["raw_reflection_response"] == reflection
    assert json.loads(output_path.read_text()) == record


def test_learner_rejects_a_forged_non_train_source_before_any_mubit_call(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    forged = replace(
        source,
        source_path=loader.repo_root
        / "state_bench"
        / "domains"
        / "travel"
        / "tasks"
        / "1-cancel_economy_domestic.json",
    )
    client = RecordingBuildClient({"degraded": False, "lessons": []})
    learner = MubitTrajectoryLearner(client=client, config=_config(MubitCredentialRole.BUILD), loader=loader)

    with pytest.raises(ValueError, match="only datasets/train_task_trajectories"):
        learner.learn(forged, tmp_path / "must-not-exist.json")

    assert client.calls == []
    assert not (tmp_path / "must-not-exist.json").exists()


def test_failed_or_degraded_reflection_is_persisted_and_visible(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    degraded_client = RecordingBuildClient({"degraded": True, "lessons": [{"content": "provisional"}]})
    failed_client = RecordingBuildClient(RuntimeError("reflection unavailable"))

    degraded = MubitTrajectoryLearner(
        client=degraded_client,
        config=_config(MubitCredentialRole.BUILD),
        loader=loader,
    ).learn(source, tmp_path / "degraded.json")
    failed = MubitTrajectoryLearner(
        client=failed_client,
        config=_config(MubitCredentialRole.BUILD),
        loader=loader,
    ).learn(source, tmp_path / "failed.json")

    assert degraded["status"] == "degraded"
    assert degraded["raw_reflection_response"]["lessons"]
    assert failed["status"] == "failed"
    assert failed["failure_stage"] == "reflect"
    assert failed["raw_reflection_response"] is None


def test_artifact_exact_dedupes_retains_provenance_and_hashes_deterministically(tmp_path):
    artifact, artifact_path, reflection_dir = _build_test_artifact(tmp_path)
    second_path = tmp_path / "frozen_lessons-again.json"
    rebuilt = build_frozen_artifact(
        domain="travel",
        experiment_id="phase2-unit",
        reflection_dir=reflection_dir,
        output_path=second_path,
        official_full_training_set=False,
    )

    assert artifact["artifact_sha256"] == rebuilt["artifact_sha256"]
    assert artifact_path.read_text() == second_path.read_text()
    assert artifact["lesson_count"] == 1
    assert len(artifact["lessons"][0]["provenance"]) == 2
    assert len(artifact["excluded_reflections"]) == 1
    assert load_frozen_artifact(artifact_path, expected_domain="travel") == artifact


def test_artifact_rejects_reflection_with_non_train_provenance(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    response = {"degraded": False, "lessons": [{"content": "contaminated"}]}
    record = _reflection_record(source, response)
    record["source_path"] = "state_bench/domains/travel/tasks/1-cancel_economy_domestic.json"
    reflection_dir = tmp_path / "raw_reflections"
    write_json_atomic(reflection_dir / f"{source.task_id}.json", record)

    with pytest.raises(ValueError, match="source_path mismatch"):
        build_frozen_artifact(
            domain="travel",
            experiment_id="phase2-unit",
            reflection_dir=reflection_dir,
            output_path=tmp_path / "artifact.json",
            official_full_training_set=False,
            loader=loader,
        )


def test_publication_writes_only_verified_artifact_entries_as_global_lessons(tmp_path):
    artifact, artifact_path, _ = _build_test_artifact(tmp_path)
    client = RecordingPublishClient()
    config = _config(
        MubitCredentialRole.EVAL,
        artifact_sha256=artifact["artifact_sha256"],
        lesson_set_sha256=artifact["lesson_set_sha256"],
        run_id=f"statebench:travel:publish:{artifact['artifact_sha256'][:16]}",
    )
    audit_calls: list[tuple[str, dict]] = []
    durable_calls: list[tuple[str, dict]] = []
    publisher = FrozenArtifactPublisher(
        client=client,
        config=config,
        durable_writer=RecordingDurableWriter(durable_calls),
        auditor=RecordingAuditor(audit_calls),
    )

    with pytest.raises(ValueError, match="clean"):
        publisher.publish(artifact_path, tmp_path / "unconfirmed.json", clean_eval_instance_confirmed=False)
    manifest = publisher.publish(
        artifact_path,
        tmp_path / "publication.json",
        clean_eval_instance_confirmed=True,
    )

    assert [name for name, _ in durable_calls] == ["remember_durable"] * artifact["lesson_count"]
    assert [call["content"] for _, call in durable_calls] == [lesson["content"] for lesson in artifact["lessons"]]
    for _, call in durable_calls:
        assert call["intent"] == "lesson"
        assert call["lesson_scope"] == "global"
        assert call["metadata"]["artifact_sha256"] == artifact["artifact_sha256"]
        assert "wait" not in call
    assert [name for name, _ in audit_calls] == ["preflight", "post_audit"]
    assert manifest["published_count"] == artifact["lesson_count"]


def test_durable_writer_requires_terminal_successful_expected_write():
    client = MagicMock()
    client.remember.return_value = {"accepted": True, "job_id": "job-1", "status": "pending"}
    client.advanced.get_ingest_job.side_effect = [
        {"job_id": "job-1", "status": "running", "done": False},
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "status": "completed",
            "done": True,
            "traces": [
                {
                    "item_id": "trace-1",
                    "writes": [{"memory_type": "knowledge", "record_id": "record-1", "success": True}],
                }
            ],
        },
    ]
    writer = MubitDurableWriter(client, timeout_seconds=1, poll_interval_seconds=0, sleeper=lambda _: None)

    receipt = writer.remember_durable(
        expected_item_id="trace-1",
        session_id="run-1",
        item_id="trace-1",
        content="trace",
    )

    assert receipt.record_ids == ("record-1",)
    assert receipt.storage_memory_types == ("knowledge",)
    assert client.remember.call_args.kwargs["wait"] is False
    assert client.advanced.get_ingest_job.call_count == 2


@pytest.mark.parametrize(
    "job",
    [
        {"job_id": "job-1", "run_id": "run-1", "status": "failed", "done": True, "error": "write failed", "traces": []},
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "status": "completed",
            "done": True,
            "traces": [
                {
                    "item_id": "trace-1",
                    "writes": [{"memory_type": "trace", "record_id": "", "success": False}],
                }
            ],
        },
    ],
)
def test_durable_writer_rejects_terminal_jobs_without_expected_durable_write(job):
    client = MagicMock()
    client.remember.return_value = {"accepted": True, "job_id": "job-1"}
    client.advanced.get_ingest_job.return_value = job
    writer = MubitDurableWriter(client, timeout_seconds=1, poll_interval_seconds=0)

    with pytest.raises(IngestDurabilityError):
        writer.remember_durable(
            expected_item_id="trace-1",
            session_id="run-1",
            item_id="trace-1",
            content="trace",
        )


def test_learner_never_reflects_after_an_ingest_durability_failure(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    client = RecordingBuildClient({"degraded": False, "lessons": []})
    client.advanced.get_ingest_job = MagicMock(
        return_value={"status": "failed", "done": True, "error": "not persisted", "traces": []}
    )
    learner = MubitTrajectoryLearner(client=client, config=_config(MubitCredentialRole.BUILD), loader=loader)

    record = learner.learn(source, tmp_path / "failed.json")

    assert record["status"] == "failed"
    assert record["failure_stage"] == "ingest"
    assert not any(name == "reflect" for name, _ in client.calls)


def test_build_resume_validates_and_skips_without_constructing_a_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "mbt_buildinstance_key_secret")
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    output_path = Phase2Paths("phase2-unit").reflection_path("travel", source.task_id)
    write_json_atomic(
        output_path,
        _reflection_record(source, {"degraded": False, "lessons": []}),
    )

    with patch("mubit.Client") as client_class:
        manifest = build_training_reflections("travel", "phase2-unit", limit=1, resume=True)

    client_class.assert_not_called()
    assert manifest["tasks"][0]["action"] == "resumed"


def test_build_resume_refuses_invalid_existing_record_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "mbt_buildinstance_key_secret")
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    output_path = Phase2Paths("phase2-unit").reflection_path("travel", source.task_id)
    record = _reflection_record(source, {"degraded": False, "lessons": []})
    record["source_sha256"] = "0" * 64
    write_json_atomic(output_path, record)
    original = output_path.read_bytes()

    with patch("mubit.Client") as client_class, pytest.raises(ValueError, match="source_sha256 mismatch"):
        build_training_reflections("travel", "phase2-unit", limit=1, resume=True)

    client_class.assert_not_called()
    assert output_path.read_bytes() == original


def test_official_artifact_requires_all_100_exact_training_task_ids(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    reflection_dir = tmp_path / "raw"
    write_json_atomic(
        reflection_dir / f"{source.task_id}.json",
        _reflection_record(source, {"degraded": False, "lessons": []}),
    )

    with pytest.raises(ValueError, match="training-set coverage mismatch"):
        build_frozen_artifact(
            domain="travel",
            experiment_id="phase2-unit",
            reflection_dir=reflection_dir,
            output_path=tmp_path / "official.json",
            official_full_training_set=True,
            loader=loader,
        )

    partial = build_frozen_artifact(
        domain="travel",
        experiment_id="phase2-unit",
        reflection_dir=reflection_dir,
        output_path=tmp_path / "partial.json",
        official_full_training_set=False,
        loader=loader,
    )
    assert partial["training_set_mode"] == "partial_pilot"


def test_official_artifact_accepts_exactly_all_100_training_records(tmp_path):
    loader = TrainTrajectoryLoader()
    reflection_dir = tmp_path / "raw"
    for source in loader.load("travel"):
        write_json_atomic(
            reflection_dir / f"{source.task_id}.json",
            _reflection_record(source, {"degraded": False, "lessons": []}),
        )

    artifact = build_frozen_artifact(
        domain="travel",
        experiment_id="phase2-unit",
        reflection_dir=reflection_dir,
        output_path=tmp_path / "official.json",
        official_full_training_set=True,
        loader=loader,
    )

    assert artifact["training_set_mode"] == "official_full"
    assert artifact["reflection_count"] == 100


def test_lesson_set_hash_is_independent_of_experiment_and_provenance(tmp_path):
    loader = TrainTrajectoryLoader()
    source = loader.load("travel", limit=1)[0]
    lesson = {"degraded": False, "lessons": [{"content": "Always preview a state-changing action."}]}
    artifacts = []
    for experiment_id in ("experiment-a", "experiment-b"):
        reflection_dir = tmp_path / experiment_id / "raw"
        write_json_atomic(
            reflection_dir / f"{source.task_id}.json",
            _reflection_record(source, lesson, experiment_id=experiment_id),
        )
        artifacts.append(
            build_frozen_artifact(
                domain="travel",
                experiment_id=experiment_id,
                reflection_dir=reflection_dir,
                output_path=tmp_path / experiment_id / "artifact.json",
                official_full_training_set=False,
                loader=loader,
            )
        )

    assert artifacts[0]["lesson_set_sha256"] == artifacts[1]["lesson_set_sha256"]
    assert artifacts[0]["artifact_sha256"] != artifacts[1]["artifact_sha256"]


class AuditClient:
    def __init__(self, activity: list[dict], lessons: list[dict]):
        self.advanced = SimpleNamespace(
            list_activity=MagicMock(
                return_value={"entries": activity, "next_page_token": "", "total_visible": len(activity)}
            )
        )
        self.lessons = MagicMock(return_value={"lessons": lessons})


def test_remote_eval_cleanliness_checks_activity_and_global_lessons():
    assert MubitEvalAuditor(AuditClient([], [])).assert_clean() == {
        "visible_activity_count": 0,
        "global_lesson_count": 0,
    }

    with pytest.raises(EvalAuditError, match="not clean"):
        MubitEvalAuditor(AuditClient([{"entry_type": "trace", "content": "old"}], [])).assert_clean()
    with pytest.raises(EvalAuditError, match="not clean"):
        MubitEvalAuditor(AuditClient([], [{"content": "old lesson"}])).assert_clean()


def test_remote_activity_audit_uses_sdk_0132_transport_fallback():
    transport = MagicMock()
    transport.invoke.return_value = {"entries": [], "next_page_token": "", "total_visible": 0}
    client = SimpleNamespace(advanced=SimpleNamespace(), _transport=transport)

    assert MubitEvalAuditor(client).list_all_activity() == []

    operation, payload = transport.invoke.call_args.args
    assert operation["grpc"] == {"method": "ListActivity", "service": "ControlService"}
    assert operation["http"] == {"method": "POST", "path": "/v2/control/activity"}
    assert payload["run_id"] == ""
    assert payload["projection"] == "full"


def test_remote_eval_post_audit_requires_exact_activity_and_lesson_content_set():
    content = "A frozen procedural lesson."
    expected = {__import__("hashlib").sha256(content.encode()).hexdigest()}
    exact_client = AuditClient(
        [{"entry_type": "lesson", "content": content}],
        [{"content": content}],
    )
    audit = MubitEvalAuditor(exact_client).await_exact_lesson_set(expected)
    assert audit["content_sha256s"] == sorted(expected)

    times = iter([0.0, 2.0])
    mismatched_client = AuditClient(
        [{"entry_type": "lesson", "content": "unexpected"}],
        [{"content": content}],
    )
    with pytest.raises(EvalAuditError, match="did not converge exactly"):
        MubitEvalAuditor(
            mismatched_client,
            timeout_seconds=1,
            poll_interval_seconds=0,
            clock=lambda: next(times),
            sleeper=lambda _: None,
        ).await_exact_lesson_set(expected)


def test_publication_does_not_write_manifest_when_post_audit_fails(tmp_path):
    artifact, artifact_path, _ = _build_test_artifact(tmp_path)
    config = _config(
        MubitCredentialRole.EVAL,
        artifact_sha256=artifact["artifact_sha256"],
        lesson_set_sha256=artifact["lesson_set_sha256"],
    )
    auditor = MagicMock(spec=MubitEvalAuditor)
    auditor.assert_clean.return_value = {"visible_activity_count": 0, "global_lesson_count": 0}
    auditor.await_exact_lesson_set.side_effect = EvalAuditError("mismatch")
    publication_path = tmp_path / "publication.json"
    publisher = FrozenArtifactPublisher(
        client=RecordingPublishClient(),
        config=config,
        durable_writer=RecordingDurableWriter([]),
        auditor=auditor,
    )

    with pytest.raises(EvalAuditError, match="mismatch"):
        publisher.publish(artifact_path, publication_path, clean_eval_instance_confirmed=True)

    assert not publication_path.exists()


def test_publication_does_not_audit_or_manifest_an_unconfirmed_write(tmp_path):
    artifact, artifact_path, _ = _build_test_artifact(tmp_path)
    config = _config(
        MubitCredentialRole.EVAL,
        artifact_sha256=artifact["artifact_sha256"],
        lesson_set_sha256=artifact["lesson_set_sha256"],
    )
    durable_writer = MagicMock(spec=MubitDurableWriter)
    durable_writer.remember_durable.side_effect = IngestDurabilityError("not durable")
    auditor = MagicMock(spec=MubitEvalAuditor)
    auditor.assert_clean.return_value = {"visible_activity_count": 0, "global_lesson_count": 0}
    publication_path = tmp_path / "publication.json"
    publisher = FrozenArtifactPublisher(
        client=RecordingPublishClient(),
        config=config,
        durable_writer=durable_writer,
        auditor=auditor,
    )

    with pytest.raises(IngestDurabilityError, match="not durable"):
        publisher.publish(artifact_path, publication_path, clean_eval_instance_confirmed=True)

    auditor.await_exact_lesson_set.assert_not_called()
    assert not publication_path.exists()
