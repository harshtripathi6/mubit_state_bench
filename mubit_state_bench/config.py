"""Environment-backed configuration for the isolated Mubit evaluation store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from state_bench.agents.base import AgentRuntimeContext


def _domain_env_suffix(domain: str) -> str:
    return domain.upper().replace("-", "_")


def _first_nonempty(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True, slots=True)
class MubitStateBenchConfig:
    """Configuration shared by the seeder and read-only retrieval adapter.

    Credentials intentionally use STATE-Bench-specific variable names. There is
    no fallback to ``MUBIT_API_KEY``: accidentally querying a developer's normal
    Mubit instance would violate domain isolation and make a run irreproducible.
    """

    domain: str
    api_key: str
    endpoint: str
    transport: str
    namespace: str
    run_id: str
    telemetry_path: Path
    budget: str = "mid"
    rank_by: str = "relevance"

    @classmethod
    def from_env(cls, runtime_context: AgentRuntimeContext) -> "MubitStateBenchConfig":
        domain = runtime_context.domain
        suffix = _domain_env_suffix(domain)
        api_key = _first_nonempty(f"MUBIT_STATE_BENCH_{suffix}_API_KEY")
        if not api_key:
            raise ValueError(f"Missing domain-isolated Mubit credentials. Set MUBIT_STATE_BENCH_{suffix}_API_KEY.")

        endpoint = (
            _first_nonempty(
                f"MUBIT_STATE_BENCH_{suffix}_ENDPOINT",
                "MUBIT_STATE_BENCH_ENDPOINT",
            )
            or "https://api.mubit.ai"
        )
        namespace = os.getenv("MUBIT_STATE_BENCH_NAMESPACE", "statebench").strip() or "statebench"
        task_component = runtime_context.task_id.replace("/", "_")
        run_id = f"{namespace}:{domain}:eval:{task_component}"
        telemetry_override = os.getenv("MUBIT_STATE_BENCH_TELEMETRY_PATH", "").strip()
        telemetry_path = (
            Path(telemetry_override) if telemetry_override else Path("outputs") / "mubit_retrieval" / f"{domain}.jsonl"
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
            namespace=namespace,
            run_id=run_id,
            telemetry_path=telemetry_path,
            budget=budget,
            rank_by=rank_by,
        )

    @classmethod
    def for_seed(cls, domain: str, telemetry_path: Path | None = None) -> "MubitStateBenchConfig":
        runtime_context = AgentRuntimeContext(
            task_id="phase1-synthetic-seed",
            user_id="synthetic",
            domain=domain,
            now="",
        )
        config = cls.from_env(runtime_context)
        return cls(
            domain=config.domain,
            api_key=config.api_key,
            endpoint=config.endpoint,
            transport=config.transport,
            namespace=config.namespace,
            run_id=f"{config.namespace}:{domain}:seed:phase1-synthetic",
            telemetry_path=telemetry_path or config.telemetry_path,
            budget=config.budget,
            rank_by=config.rank_by,
        )
