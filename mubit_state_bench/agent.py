"""STATE-Bench built-in agent with a read-only Mubit learning hook."""

from __future__ import annotations

from typing import Any, Callable

from mubit_state_bench.memory import MubitReadOnlyStore
from state_bench.agents.base import AgentRuntimeContext
from state_bench.agents.state_bench import StateBenchAgent
from state_bench.client import LLMClient, PooledLLMClient


class MubitStateBenchAgent(StateBenchAgent):
    """Microsoft's built-in agent loop plus Mubit ``retrieve_learnings``.

    STATE-Bench continues to own the system prompt, model calls, domain tools,
    tool handlers, simulator, environment, and scoring. This subclass only
    implements the benchmark's read-only procedural-learning hook.
    """

    def __init__(
        self,
        client: LLMClient | PooledLLMClient,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable],
        runtime_context: AgentRuntimeContext | None = None,
        retrieve_learnings_top_k: int = 3,
        agent_reasoning_effort: str | None = None,
        memory_store: MubitReadOnlyStore | None = None,
    ):
        if runtime_context is None:
            raise ValueError("MubitStateBenchAgent requires STATE-Bench runtime_context")
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            runtime_context=runtime_context,
            retrieve_learnings_top_k=retrieve_learnings_top_k,
            agent_reasoning_effort=agent_reasoning_effort,
        )
        self._memory_store = memory_store or MubitReadOnlyStore.from_env(runtime_context)

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """Return at most ``top_k`` evidence strings from read-only Mubit recall."""

        return self._memory_store.retrieve(query=query, top_k=top_k)[:top_k]
