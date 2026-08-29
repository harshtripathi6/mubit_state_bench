from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mubit_state_bench.artifact import build_frozen_artifact, load_frozen_artifact
from mubit_state_bench.config import (
    MubitCredentialRole,
    MubitStateBenchConfig,
    validate_credential_separation,
)
from mubit_state_bench.io_utils import canonical_sha256, write_json_atomic
from mubit_state_bench.learning import RAW_REFLECTION_SCHEMA, MubitTrajectoryLearner
from mubit_state_bench.publication import FrozenArtifactPublisher
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

    def remember(self, **kwargs):
        self.calls.append(("remember", kwargs))
        return {"done": True}

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
        return {"done": True}


def _config(
    role: MubitCredentialRole,
    *,
    artifact_sha256: str = "",
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
    )


def _reflection_record(source, response: dict, *, status: str = "ok", reasons: list[str] | None = None):
    return {
        "schema_version": RAW_REFLECTION_SCHEMA,
        "parser_schema_version": DECISION_TURN_SCHEMA,
        "domain": source.domain,
        "task_id": source.task_id,
        "source_path": source.source_relative_path,
        "source_sha256": source.source_sha256,
        "run_id": f"statebench:{source.domain}:train:phase2-unit:{source.task_id}",
        "terminal_marker_seen": True,
        "terminal_marker_interpreted_as_outcome": False,
        "parsed_turn_count": 1,
        "status": status,
        "failure_stage": None,
        "ingested_item_count": 1,
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
        _reflection_record(third, response_three, status="degraded", reasons=["Mubit marked degraded"]),
    )
    artifact_path = tmp_path / "frozen_lessons.json"
    artifact = build_frozen_artifact(
        domain="travel",
        experiment_id="phase2-unit",
        reflection_dir=reflection_dir,
        output_path=artifact_path,
        loader=loader,
    )
    return artifact, artifact_path, reflection_dir


def test_role_credentials_cannot_collapse_to_the_same_key(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_SMOKE_TRAVEL_API_KEY", "same-secret")
    monkeypatch.setenv("MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY", "same-secret")

    with pytest.raises(ValueError, match="credential isolation violation") as exc_info:
        validate_credential_separation()

    assert "same-secret" not in str(exc_info.value)


def test_seed_build_and_eval_configs_accept_only_their_role_keys(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY", "eval-only")
    with pytest.raises(ValueError, match="SMOKE_TRAVEL_API_KEY"):
        MubitStateBenchConfig.for_seed("travel")
    with pytest.raises(ValueError, match="BUILD_TRAVEL_API_KEY"):
        MubitStateBenchConfig.for_build("travel", "task-1", "phase2-unit")


def test_eval_config_uses_eval_key_and_runtime_run_index(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY", "eval-only")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EXPERIMENT_ID", "phase2-eval")
    monkeypatch.setenv("MUBIT_STATE_BENCH_ARM", "mubit")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256", "a" * 64)
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
    assert config.api_key == "eval-only"
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
    assert [name for name, _ in client.calls] == ["remember"] * len(parsed.turns) + ["reflect"]
    remember_calls = [kwargs for name, kwargs in client.calls if name == "remember"]
    assert [json.loads(call["content"])["turn_index"] for call in remember_calls] == list(range(len(parsed.turns)))
    assert all(call["intent"] == "trace" and call["wait"] is True for call in remember_calls)
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

    with pytest.raises(ValueError, match="not the checked-in train path"):
        build_frozen_artifact(
            domain="travel",
            experiment_id="phase2-unit",
            reflection_dir=reflection_dir,
            output_path=tmp_path / "artifact.json",
            loader=loader,
        )


def test_publication_writes_only_verified_artifact_entries_as_global_lessons(tmp_path):
    artifact, artifact_path, _ = _build_test_artifact(tmp_path)
    client = RecordingPublishClient()
    config = _config(
        MubitCredentialRole.EVAL,
        artifact_sha256=artifact["artifact_sha256"],
        run_id=f"statebench:travel:publish:{artifact['artifact_sha256'][:16]}",
    )
    publisher = FrozenArtifactPublisher(client=client, config=config)

    with pytest.raises(ValueError, match="clean"):
        publisher.publish(artifact_path, tmp_path / "unconfirmed.json", clean_eval_instance_confirmed=False)
    manifest = publisher.publish(
        artifact_path,
        tmp_path / "publication.json",
        clean_eval_instance_confirmed=True,
    )

    assert [name for name, _ in client.calls] == ["remember"] * artifact["lesson_count"]
    assert [call["content"] for _, call in client.calls] == [lesson["content"] for lesson in artifact["lessons"]]
    for _, call in client.calls:
        assert call["intent"] == "lesson"
        assert call["lesson_scope"] == "global"
        assert call["metadata"]["artifact_sha256"] == artifact["artifact_sha256"]
        assert call["wait"] is True
    assert manifest["published_count"] == artifact["lesson_count"]
