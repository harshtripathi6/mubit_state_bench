"""Role-separated Mubit configuration for smoke, build, and evaluation."""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from state_bench.agents.base import AgentRuntimeContext

STATE_BENCH_DOMAINS = ("travel", "customer_support", "shopping_assistant")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MubitCredentialRole(StrEnum):
    SMOKE = "smoke"
    BUILD = "build"
    EVAL = "eval"


def _domain_env_suffix(domain: str) -> str:
    if domain not in STATE_BENCH_DOMAINS:
        raise ValueError(f"Unsupported STATE-Bench domain: {domain!r}")
    return domain.upper()


def _credential_env_name(role: MubitCredentialRole, domain: str) -> str:
    return f"MUBIT_STATE_BENCH_{role.value.upper()}_{_domain_env_suffix(domain)}_API_KEY"


def _first_nonempty(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _require_safe_id(value: str, field_name: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field_name} must match {_SAFE_ID.pattern}")
    return value


def validate_credential_separation() -> None:
    """Reject any configured role/domain credentials with identical values."""

    configured: list[tuple[str, str]] = []
    for role in MubitCredentialRole:
        for domain in STATE_BENCH_DOMAINS:
            name = _credential_env_name(role, domain)
            value = os.getenv(name, "").strip()
            if value:
                configured.append((name, value))

    for index, (left_name, left_value) in enumerate(configured):
        for right_name, right_value in configured[index + 1 :]:
            if hmac.compare_digest(left_value, right_value):
                raise ValueError(
                    "Mubit credential isolation violation: "
                    f"{left_name} and {right_name} contain the same key value. "
                    "Use a distinct Mubit instance/key for every role and domain."
                )


@dataclass(frozen=True, slots=True)
class MubitStateBenchConfig:
    """One domain- and role-isolated Mubit client configuration."""

    domain: str
    api_key: str
    endpoint: str
    transport: str
    namespace: str
    run_id: str
    telemetry_path: Path
    budget: str = "mid"
    rank_by: str = "relevance"
    role: MubitCredentialRole = MubitCredentialRole.EVAL
    experiment_id: str = ""
    arm: str = ""
    artifact_sha256: str = ""
    run_number: int | None = None

    @classmethod
    def _from_role_env(
        cls,
        *,
        domain: str,
        role: MubitCredentialRole,
        run_id: str,
        telemetry_path: Path,
        experiment_id: str = "",
        arm: str = "",
        artifact_sha256: str = "",
        run_number: int | None = None,
    ) -> "MubitStateBenchConfig":
        suffix = _domain_env_suffix(domain)
        credential_name = _credential_env_name(role, domain)
        api_key = os.getenv(credential_name, "").strip()
        if not api_key:
            raise ValueError(f"Missing {role.value} Mubit credential. Set {credential_name}.")
        validate_credential_separation()

        role_suffix = role.value.upper()
        endpoint = (
            _first_nonempty(
                f"MUBIT_STATE_BENCH_{role_suffix}_{suffix}_ENDPOINT",
                f"MUBIT_STATE_BENCH_{role_suffix}_ENDPOINT",
                "MUBIT_STATE_BENCH_ENDPOINT",
            )
            or "https://api.mubit.ai"
        )
        budget = os.getenv("MUBIT_STATE_BENCH_BUDGET", "mid").strip() or "mid"
        if budget not in {"low", "mid", "high"}:
            raise ValueError("MUBIT_STATE_BENCH_BUDGET must be low, mid, or high")
        rank_by = os.getenv("MUBIT_STATE_BENCH_RANK_BY", "relevance").strip() or "relevance"
        if rank_by not in {"relevance", "freshness", "balanced"}:
            raise ValueError("MUBIT_STATE_BENCH_RANK_BY must be relevance, freshness, or balanced")

        return cls(
            domain=domain,
            api_key=api_key,
            endpoint=endpoint,
            transport=os.getenv("MUBIT_STATE_BENCH_TRANSPORT", "auto").strip() or "auto",
            namespace=os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench",
            run_id=run_id,
            telemetry_path=telemetry_path,
            budget=budget,
            rank_by=rank_by,
            role=role,
            experiment_id=experiment_id,
            arm=arm,
            artifact_sha256=artifact_sha256,
            run_number=run_number,
        )

    @classmethod
    def from_env(cls, runtime_context: AgentRuntimeContext) -> "MubitStateBenchConfig":
        """Load the evaluation-only configuration used by the benchmark agent."""

        domain = runtime_context.domain
        namespace = os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench"
        experiment_id = _require_safe_id(
            os.getenv("MUBIT_STATE_BENCH_EXPERIMENT_ID", "").strip(),
            "MUBIT_STATE_BENCH_EXPERIMENT_ID",
        )
        arm = _require_safe_id(
            os.getenv("MUBIT_STATE_BENCH_ARM", "mubit").strip() or "mubit",
            "MUBIT_STATE_BENCH_ARM",
        )
        artifact_sha256 = os.getenv("MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256", "").strip().lower()
        if not _SHA256.fullmatch(artifact_sha256):
            raise ValueError("MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256 must be a 64-character lowercase SHA-256")
        if runtime_context.run_idx is not None:
            if not isinstance(runtime_context.run_idx, int) or isinstance(runtime_context.run_idx, bool):
                raise ValueError("STATE-Bench runtime run_idx must be an integer >= 1")
            run_number = runtime_context.run_idx
        else:
            run_number_text = os.getenv("MUBIT_STATE_BENCH_RUN_NUMBER", "").strip()
            if not run_number_text.isdigit() or int(run_number_text) < 1:
                raise ValueError("MUBIT_STATE_BENCH_RUN_NUMBER must be an integer >= 1")
            run_number = int(run_number_text)
        if run_number < 1:
            raise ValueError("STATE-Bench run number must be >= 1")
        task_component = runtime_context.task_id.replace("/", "_")
        run_id = f"{namespace}:{domain}:eval:{experiment_id}:run-{run_number}:{task_component}"
        telemetry_path = Path("outputs") / experiment_id / "retrieval" / f"{domain}.jsonl"

        return cls._from_role_env(
            domain=domain,
            role=MubitCredentialRole.EVAL,
            run_id=run_id,
            telemetry_path=telemetry_path,
            experiment_id=experiment_id,
            arm=arm,
            artifact_sha256=artifact_sha256,
            run_number=run_number,
        )

    @classmethod
    def for_seed(cls, domain: str) -> "MubitStateBenchConfig":
        namespace = os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench"
        return cls._from_role_env(
            domain=domain,
            role=MubitCredentialRole.SMOKE,
            run_id=f"{namespace}:{domain}:smoke:phase1-synthetic",
            telemetry_path=Path("outputs") / "smoke" / "retrieval" / f"{domain}.jsonl",
        )

    @classmethod
    def for_build(cls, domain: str, task_id: str, experiment_id: str) -> "MubitStateBenchConfig":
        experiment_id = _require_safe_id(experiment_id, "experiment_id")
        namespace = os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench"
        task_component = task_id.replace("/", "_")
        return cls._from_role_env(
            domain=domain,
            role=MubitCredentialRole.BUILD,
            run_id=f"{namespace}:{domain}:train:{experiment_id}:{task_component}",
            telemetry_path=Path("outputs") / experiment_id / "build" / domain / "build.jsonl",
            experiment_id=experiment_id,
            arm="build",
        )

    @classmethod
    def for_publication(
        cls,
        domain: str,
        experiment_id: str,
        artifact_sha256: str,
    ) -> "MubitStateBenchConfig":
        experiment_id = _require_safe_id(experiment_id, "experiment_id")
        if not _SHA256.fullmatch(artifact_sha256):
            raise ValueError("artifact_sha256 must be a 64-character lowercase SHA-256")
        namespace = os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench"
        return cls._from_role_env(
            domain=domain,
            role=MubitCredentialRole.EVAL,
            run_id=f"{namespace}:{domain}:publish:{artifact_sha256[:16]}",
            telemetry_path=Path("outputs") / experiment_id / "publication" / f"{domain}.jsonl",
            experiment_id=experiment_id,
            arm="publication",
            artifact_sha256=artifact_sha256,
        )


def redact_configured_secrets(message: str) -> str:
    redacted = message
    for role in MubitCredentialRole:
        for domain in STATE_BENCH_DOMAINS:
            value = os.getenv(_credential_env_name(role, domain), "").strip()
            if value:
                redacted = redacted.replace(value, "[redacted]")
    return redacted
