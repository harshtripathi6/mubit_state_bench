"""Train-only loading and deterministic ``decision_turn_v1`` parsing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mubit_state_bench.config import STATE_BENCH_DOMAINS

DECISION_TURN_SCHEMA = "decision_turn_v1"
TERMINAL_MARKER = "[TASK_DONE]"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainTrajectorySource:
    domain: str
    task_id: str
    source_path: Path
    source_relative_path: str
    source_sha256: str
    document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DecisionToolCall:
    name: str
    arguments: Any
    result: Any

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}


@dataclass(frozen=True, slots=True)
class DecisionTurn:
    turn_index: int
    user_context: tuple[str, ...]
    assistant_text: str
    tool_calls: tuple[DecisionToolCall, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DECISION_TURN_SCHEMA,
            "turn_index": self.turn_index,
            "user_context": list(self.user_context),
            "assistant_text": self.assistant_text,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }

    def canonical_content(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ParsedTrainTrajectory:
    domain: str
    task_id: str
    source_path: Path
    source_relative_path: str
    source_sha256: str
    turns: tuple[DecisionTurn, ...]
    terminal_marker_seen: bool


class TrainTrajectoryLoader:
    """Load only checked-in Agent Learning Track training trajectories."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        self.dataset_root = (self.repo_root / "datasets" / "train_task_trajectories").resolve()

    def domain_root(self, domain: str) -> Path:
        if domain not in STATE_BENCH_DOMAINS:
            raise ValueError(f"Unsupported STATE-Bench training domain: {domain!r}")
        root = (self.dataset_root / domain).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Checked-in training trajectory directory not found: {root}")
        return root

    def paths(self, domain: str, limit: int | None = None) -> list[Path]:
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError("limit must be an integer >= 1 or None")
        paths = sorted(self.domain_root(domain).glob("*.json"), key=lambda path: path.name)
        return paths if limit is None else paths[:limit]

    def load(self, domain: str, limit: int | None = None) -> list[TrainTrajectorySource]:
        return [self.load_path(domain, path) for path in self.paths(domain, limit)]

    def load_path(self, domain: str, path: Path) -> TrainTrajectorySource:
        """Load a path only if it is a direct child of the selected train directory."""

        domain_root = self.domain_root(domain)
        resolved = path.resolve()
        if resolved.parent != domain_root or resolved.suffix != ".json":
            raise ValueError(
                f"Trajectory source rejected: only datasets/train_task_trajectories/{domain}/*.json may be loaded"
            )
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        raw = resolved.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("conversation"), list):
            raise ValueError(f"Training trajectory must contain a conversation list: {resolved.name}")
        return TrainTrajectorySource(
            domain=domain,
            task_id=resolved.stem,
            source_path=resolved,
            source_relative_path=resolved.relative_to(self.repo_root).as_posix(),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            document=document,
        )

    def assert_train_source(self, source: TrainTrajectorySource) -> None:
        expected = self.load_path(source.domain, source.source_path)
        if (
            expected.task_id != source.task_id
            or expected.source_relative_path != source.source_relative_path
            or expected.source_sha256 != source.source_sha256
        ):
            raise ValueError("Trajectory source does not match the checked-in training file")


def parse_decision_turns(source: TrainTrajectorySource) -> ParsedTrainTrajectory:
    """Parse conversation data without inference, scoring, or outcome synthesis."""

    pending_user_context: list[str] = []
    turns: list[DecisionTurn] = []
    terminal_marker_seen = False
    for message_index, message in enumerate(source.document["conversation"]):
        if not isinstance(message, dict):
            raise ValueError(f"conversation[{message_index}] must be an object")
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"conversation[{message_index}].content must be a string")
            if content.strip() == TERMINAL_MARKER:
                terminal_marker_seen = True
            else:
                pending_user_context.append(content)
            continue
        if role != "assistant":
            raise ValueError(f"Unsupported conversation role at index {message_index}: {role!r}")

        assistant_text = message.get("content")
        if assistant_text is None:
            assistant_text = ""
        if not isinstance(assistant_text, str):
            raise ValueError(f"conversation[{message_index}].content must be a string or null")
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ValueError(f"conversation[{message_index}].tool_calls must be a list or null")
        tool_calls: list[DecisionToolCall] = []
        for tool_index, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("name"), str):
                raise ValueError(f"conversation[{message_index}].tool_calls[{tool_index}] is malformed")
            tool_calls.append(
                DecisionToolCall(
                    name=tool_call["name"],
                    arguments=tool_call.get("arguments"),
                    result=tool_call.get("result"),
                )
            )
        turns.append(
            DecisionTurn(
                turn_index=len(turns),
                user_context=tuple(pending_user_context),
                assistant_text=assistant_text,
                tool_calls=tuple(tool_calls),
            )
        )
        pending_user_context.clear()

    return ParsedTrainTrajectory(
        domain=source.domain,
        task_id=source.task_id,
        source_path=source.source_path,
        source_relative_path=source.source_relative_path,
        source_sha256=source.source_sha256,
        turns=tuple(turns),
        terminal_marker_seen=terminal_marker_seen,
    )
