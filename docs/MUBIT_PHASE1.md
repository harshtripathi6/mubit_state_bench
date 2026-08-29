# Mubit phase-one integration

This integration keeps Microsoft's built-in `StateBenchAgent` and adds only its
official Agent Learning Track hook. `MubitStateBenchAgent` does not replace the
agent loop, system prompt, domain tools, simulator, environment, or scoring.

Phase one deliberately supports retrieval and synthetic seeding only. It does
not ingest, reflect on, or learn from training trajectories.

## Isolation and retrieval contract

Use one Mubit instance/API key per STATE-Bench domain:

```dotenv
MUBIT_STATE_BENCH_TRAVEL_API_KEY="mbt_..."
MUBIT_STATE_BENCH_CUSTOMER_SUPPORT_API_KEY="mbt_..."
MUBIT_STATE_BENCH_SHOPPING_ASSISTANT_API_KEY="mbt_..."
```

There is no shared-key fallback. Requiring the domain-specific variable makes
it harder to accidentally point two evaluation domains at the same instance,
and the integration also intentionally ignores a generic `MUBIT_API_KEY`.

Every retrieval is one Mubit `recall` request with:

- `mode="direct_bypass"` and `direct_lane="semantic_search"`
- `evidence_only=True`
- `entry_types=["lesson"]`
- `include_working_memory=False`
- a response cap equal to STATE-Bench's benchmark-fixed `top_k`

The adapter exposes no write method. Retrieval telemetry is appended to
`outputs/mubit_retrieval/<domain>.jsonl`, separately from STATE-Bench's task
trajectory and environment state.

## Setup and seed

```bash
uv sync --group dev
cp .env.example .env
# Add the isolated travel Mubit API key to .env.
uv run mubit-state-bench-seed --domain travel
```

The seed command writes exactly five fixed synthetic travel lessons. Each write
uses `intent="lesson"`, `lesson_scope="global"`, a stable upsert key, and a
stable idempotency key. This explicit global scope is intentional: it makes the
lessons available to frozen evaluation runs without waiting for gradual scope
promotion.

## Deterministic task-level proof

```bash
uv run pytest tests/test_mubit_state_bench_integration.py -q
```

The proof loads the checked-in travel task
`101-challenge_mixed_strategy_shared_budget` and its actual STATE-Bench
environment, runs it through the benchmark orchestrator with
`MubitStateBenchAgent`, and makes the mocked model invoke
`retrieve_learnings`. It asserts that:

- the query reaches the Mubit retrieval boundary with benchmark-fixed
  `top_k=3`, even if the model asks for more;
- the retrieved lesson appears in the recorded tool result;
- the retrieval tool is included in Microsoft's built-in agent prompt/tool
  loop; and
- the environment state diff remains empty because retrieval is read-only.

The model and Mubit transport are mocked only to make this smoke proof
deterministic and credential-free; the real task, environment, orchestrator,
and built-in agent loop are used.

## One live task

After seeding and configuring the STATE-Bench model/simulator credentials, run:

```bash
uv run python -m state_bench.scripts.run_task \
  --domain travel \
  --task 101-challenge_mixed_strategy_shared_budget \
  --agent-class MubitStateBenchAgent \
  --agent-provider openai \
  --agent-model-name YOUR_MODEL \
  --agent-model-reasoning-level medium \
  --retrieve-learnings-top-k 3 \
  --num-runs 1 \
  --num-workers 1 \
  --no-score \
  --output-dir outputs/mubit-phase1-travel
```

Keep the provider, model, reasoning effort, and any provider-side sampling
configuration identical for the no-memory, flat-RAG, and Mubit arms. The live
trajectory records the `retrieve_learnings` tool call, while the separate Mubit
JSONL file records retrieval latency and evidence metadata.
