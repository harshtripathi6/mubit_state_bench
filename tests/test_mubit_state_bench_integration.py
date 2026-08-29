from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mubit_state_bench.agent import MubitStateBenchAgent
from mubit_state_bench.config import MubitCredentialRole, MubitStateBenchConfig
from mubit_state_bench.memory import MubitReadOnlyStore
from mubit_state_bench.seed_synthetic_lessons import seed_travel_lessons
from mubit_state_bench.synthetic_lessons import TRAVEL_SYNTHETIC_LESSONS
from mubit_state_bench.telemetry import JsonlTelemetrySink
from state_bench.agents.base import AgentRuntimeContext
from state_bench.agents.loader import load_root_agent_class
from state_bench.agents.state_bench import RETRIEVE_LEARNINGS_TOOL_NAME
from state_bench.client import PooledLLMClient
from state_bench.domain import get_domain_config
from state_bench.env_loader import load_task_environment
from state_bench.orchestrator import run_task
from state_bench.paths import domain_tasks_dir
from state_bench.schemas import TaskDefinition

REAL_DOMAIN_NAME = "travel"
REAL_TASK_ID = "101-challenge_mixed_strategy_shared_budget"


class RecordingRecallClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def recall(self, **kwargs):
        self.calls.append(("recall", kwargs))
        return {
            "mode": "direct_bypass",
            "evidence": [
                {
                    "id": f"lesson-{index}",
                    "content": f"procedural lesson {index}",
                    "entry_type": "lesson",
                    "score": 1 - (index / 10),
                }
                for index in range(1, 6)
            ],
        }


class RecordingMemoryStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[str]:
        self.calls.append((query, top_k))
        return ["Preview a state-changing action, explain its consequences, then obtain explicit confirmation."]


def _runtime_context() -> AgentRuntimeContext:
    return AgentRuntimeContext(
        task_id=REAL_TASK_ID,
        user_id="user-1",
        domain=REAL_DOMAIN_NAME,
        now="2026-06-15T10:00:00",
    )


def _config(telemetry_path: Path) -> MubitStateBenchConfig:
    return MubitStateBenchConfig(
        domain=REAL_DOMAIN_NAME,
        api_key="isolated-test-key",
        endpoint="https://api.mubit.ai",
        transport="auto",
        namespace="statebench",
        run_id=f"statebench:{REAL_DOMAIN_NAME}:eval:{REAL_TASK_ID}",
        telemetry_path=telemetry_path,
        role=MubitCredentialRole.EVAL,
        experiment_id="phase2-test",
        arm="mubit",
        artifact_sha256="a" * 64,
        lesson_set_sha256="b" * 64,
        run_number=2,
    )


def _make_response(response_id: str, output_items: list, output_text: str = "") -> MagicMock:
    response = MagicMock()
    response.id = response_id
    response.output = output_items
    response.output_text = output_text
    response.status = "completed"
    response.incomplete_details = None
    response.usage = None
    return response


def _make_function_call(call_id: str, name: str, arguments: dict[str, object]) -> MagicMock:
    item = MagicMock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = json.dumps(arguments)
    return item


def _make_text_item(text: str) -> MagicMock:
    item = MagicMock()
    item.type = "message"
    item.text = text
    return item


def _load_real_task_and_env():
    domain = get_domain_config(REAL_DOMAIN_NAME)
    task = TaskDefinition.load(domain_tasks_dir(REAL_DOMAIN_NAME) / f"{REAL_TASK_ID}.json")
    env_data, _ = load_task_environment(domain, task)
    return domain, task, env_data


def test_retrieval_is_one_read_only_direct_bypass_call_and_caps_top_k(tmp_path):
    client = RecordingRecallClient()
    telemetry_path = tmp_path / "retrieval.jsonl"
    store = MubitReadOnlyStore(
        client=client,
        config=_config(telemetry_path),
        runtime_context=_runtime_context(),
        telemetry=JsonlTelemetrySink(telemetry_path),
    )

    learnings = store.retrieve("cancel only one flight booking safely", top_k=3)

    assert learnings == ["procedural lesson 1", "procedural lesson 2", "procedural lesson 3"]
    assert [name for name, _ in client.calls] == ["recall"]
    recall = client.calls[0][1]
    assert recall == {
        "session_id": f"statebench:{REAL_DOMAIN_NAME}:eval:{REAL_TASK_ID}",
        "query": "cancel only one flight booking safely",
        "mode": "direct_bypass",
        "direct_lane": "semantic_search",
        "include_linked_runs": False,
        "limit": 3,
        "entry_types": ["lesson"],
        "include_working_memory": False,
        "budget": "mid",
        "rank_by": "relevance",
        "explain": True,
        "prefer_current_run": False,
        "evidence_only": True,
    }

    events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["status"] == "ok"
    assert events[0]["returned_count"] == 3
    assert events[0]["request"]["mode"] == "direct_bypass"
    assert events[0]["request"]["evidence_only"] is True
    assert events[0]["experiment_id"] == "phase2-test"
    assert events[0]["arm"] == "mubit"
    assert events[0]["artifact_sha256"] == "a" * 64
    assert events[0]["lesson_set_sha256"] == "b" * 64
    assert events[0]["run_number"] == 2


def test_retrieval_telemetry_redacts_configured_api_key_from_exceptions(tmp_path):
    telemetry_path = tmp_path / "retrieval.jsonl"
    config = _config(telemetry_path)
    client = MagicMock()
    client.recall.side_effect = RuntimeError(f"request rejected for credential {config.api_key}")
    store = MubitReadOnlyStore(
        client=client,
        config=config,
        runtime_context=_runtime_context(),
        telemetry=JsonlTelemetrySink(telemetry_path),
    )

    with pytest.raises(RuntimeError):
        store.retrieve("cancel safely", top_k=3)

    telemetry = telemetry_path.read_text()
    assert config.api_key not in telemetry
    assert "[redacted]" in telemetry


def test_real_state_bench_task_invokes_mubit_retrieval_without_state_mutation():
    domain, task, env_data = _load_real_task_and_env()
    memory_store = RecordingMemoryStore()
    retrieve_call = _make_function_call(
        "call_mubit",
        RETRIEVE_LEARNINGS_TOOL_NAME,
        {"query": "cancel one flight while preserving the rest of the trip", "top_k": 99},
    )
    mock_complete = MagicMock(
        side_effect=[
            _make_response("resp_001", [retrieve_call]),
            _make_response(
                "resp_002", [_make_text_item("I checked the relevant procedure.")], "I checked the relevant procedure."
            ),
        ]
    )

    pinned_client = MagicMock()
    pinned_client.complete_with_tools = mock_complete
    pinned_context = MagicMock()
    pinned_context.__enter__ = MagicMock(return_value=pinned_client)
    pinned_context.__exit__ = MagicMock(return_value=False)
    client = MagicMock(spec=PooledLLMClient)
    client.pinned.return_value = pinned_context

    simulator = MagicMock()
    simulator.respond.return_value = "[TASK_DONE]"

    with (
        patch("mubit_state_bench.agent.MubitReadOnlyStore.from_env", return_value=memory_store),
        patch("state_bench.orchestrator.UserSimulator", return_value=simulator),
    ):
        trajectory = run_task(
            task=task,
            env_data=env_data,
            user_id=task.user_id,
            client=client,
            simulator_client=MagicMock(),
            domain=domain,
            agent_class=MubitStateBenchAgent,
            retrieve_learnings_top_k=3,
            agent_reasoning_effort="medium",
        )

    assert memory_store.calls == [("cancel one flight while preserving the rest of the trip", 3)]
    retrieval_calls = [
        call
        for message in trajectory.conversation
        for call in (message.get("tool_calls") or [])
        if call["name"] == RETRIEVE_LEARNINGS_TOOL_NAME
    ]
    assert retrieval_calls == [
        {
            "name": RETRIEVE_LEARNINGS_TOOL_NAME,
            "arguments": {
                "query": "cancel one flight while preserving the rest of the trip",
                "top_k": 99,
            },
            "result": {
                "learnings": [
                    "Preview a state-changing action, explain its consequences, then obtain explicit confirmation."
                ]
            },
        }
    ]
    assert trajectory.state_diff is not None
    assert trajectory.state_diff.is_empty()

    first_model_call = mock_complete.call_args_list[0].kwargs
    assert RETRIEVE_LEARNINGS_TOOL_NAME in {tool["name"] for tool in first_model_call["tools"]}
    assert first_model_call["reasoning_effort"] == "medium"


def test_root_agent_loader_discovers_mubit_subclass():
    loaded = load_root_agent_class("MubitStateBenchAgent", root=Path(__file__).parents[1])

    assert loaded is MubitStateBenchAgent


def test_config_requires_a_domain_specific_mubit_instance(monkeypatch):
    monkeypatch.delenv("MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY", raising=False)
    monkeypatch.setenv("MUBIT_STATE_BENCH_API_KEY", "shared-key-must-not-be-used")
    monkeypatch.setenv("MUBIT_API_KEY", "generic-key-must-not-be-used")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EXPERIMENT_ID", "phase2-test")
    monkeypatch.setenv("MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("MUBIT_STATE_BENCH_RUN_NUMBER", "1")

    with pytest.raises(ValueError, match="MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY"):
        MubitStateBenchConfig.from_env(_runtime_context())


def test_seed_script_writes_exactly_five_global_lessons(monkeypatch):
    monkeypatch.setenv("MUBIT_STATE_BENCH_SMOKE_TRAVEL_API_KEY", "mbt_smokeinstance_key_secret")
    client = MagicMock()
    client.remember.side_effect = [
        {"accepted": True, "job_id": f"job-{index}", "status": "pending"} for index in range(5)
    ]
    client.advanced.get_ingest_job.side_effect = [
        {
            "job_id": f"job-{index}",
            "run_id": "statebench:travel:smoke:phase1-synthetic",
            "status": "completed",
            "done": True,
            "traces": [
                {
                    "item_id": lesson.item_id,
                    "writes": [{"memory_type": "knowledge", "record_id": f"record-{index}", "success": True}],
                }
            ],
        }
        for index, lesson in enumerate(TRAVEL_SYNTHETIC_LESSONS)
    ]

    with patch("mubit.Client", return_value=client):
        results = seed_travel_lessons()

    assert len(TRAVEL_SYNTHETIC_LESSONS) == 5
    assert len(results) == 5
    assert client.remember.call_count == 5
    assert {call.kwargs["item_id"] for call in client.remember.call_args_list} == {
        lesson.item_id for lesson in TRAVEL_SYNTHETIC_LESSONS
    }
    for call in client.remember.call_args_list:
        assert call.kwargs["session_id"] == "statebench:travel:smoke:phase1-synthetic"
        assert call.kwargs["intent"] == "lesson"
        assert call.kwargs["lesson_scope"] == "global"
        assert call.kwargs["wait"] is False
        assert call.kwargs["metadata"]["domain"] == "travel"
        assert call.kwargs["metadata"]["synthetic"] is True
