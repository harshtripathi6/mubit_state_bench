# Mubit phase-two pipeline

Phase 2 creates a one-way, auditable boundary:

```text
checked-in train trajectories
        -> BUILD instance traces
        -> one raw reflection per task
        -> deterministic frozen artifact
        -> clean EVAL instance
        -> read-only benchmark retrieval
```

It does not inspect held-out task outcomes, infer success from `[TASK_DONE]`, or
call `record_outcome()`.

## Credential isolation

Provision separate Mubit instances and API keys for every role and domain. For the
initial travel pilot, `.env` needs three distinct values:

```dotenv
MUBIT_STATE_BENCH_SMOKE_TRAVEL_API_KEY="mbt_..."
MUBIT_STATE_BENCH_BUILD_TRAVEL_API_KEY="mbt_..."
MUBIT_STATE_BENCH_EVAL_TRAVEL_API_KEY="mbt_..."
```

For hosted `mbt_<instance>_...` keys, startup compares the documented
non-secret instance segment as well as the full key value. Distinct keys for
the same instance are rejected. For a legacy or non-hosted key, set the
matching `MUBIT_STATE_BENCH_<ROLE>_<DOMAIN>_INSTANCE_ID` value so isolation can
still be established without exposing a secret.

The seeder accepts only SMOKE, the trajectory learner accepts only BUILD, and
the evaluator and artifact publisher accept only EVAL. Startup rejects any
identical key values across all configured roles and domains without printing
the values. `.env` is ignored by Git and must never be copied into output,
telemetry, raw reflections, or an artifact.

## 1. Pilot-build 3–5 travel reflections

The learner has no input-path option. It can read only direct children of
`datasets/train_task_trajectories/<domain>/`, verifies the source path and raw
SHA-256, and defaults to all 100 files. Use `--limit` only for development:

```bash
uv run mubit-state-bench-build \
  --domain travel \
  --experiment-id phase2-travel-pilot-001 \
  --limit 5
```

Each assistant decision is parsed locally as `decision_turn_v1`. The canonical
trace preserves the immediately preceding user context, assistant text, and
ordered tool names, arguments, and results. No model is used during parsing.

The BUILD run ID is unique per experiment/task:

```text
statebench:travel:train:phase2-travel-pilot-001:<task-id>
```

Every trace write uses a deterministic item ID, upsert key, and idempotency key.
The client submits asynchronously, polls its exact ingest job, and requires a
successful durable storage record for the expected trace item ID. Only after every
turn has a persisted receipt can the learner call `reflect()` exactly once.
Configure the BUILD instance without a server-side automatic
reflection cadence if strict one-reflection-per-task accounting is required.

Raw results are stored under:

```text
outputs/phase2-travel-pilot-001/build/travel/raw_reflections/<task-id>.json
```

Every file contains the checked-in source path/hash and the complete raw Mubit
response. Failed and degraded responses remain visible and are not silently
dropped. Inspect these files before continuing.

To continue an interrupted build, add `--resume`. An existing task is skipped
only if its schema, experiment/run identity, current source path and bytes,
parser facts, ordered durable receipts, reflection response hash, and status
fields all validate. Invalid or older files stop the build; they are never
overwritten or silently reflected again.

## 2. Freeze lessons

```bash
uv run mubit-state-bench-freeze \
  --domain travel \
  --experiment-id phase2-travel-pilot-001 \
  --partial
```

The freezer revalidates every source against the checked-in train directory,
verifies raw reflection hashes, excludes failed/degraded reflections and
empty/degraded lessons, and performs exact-string deduplication only. Duplicate
strings merge provenance rather than losing it.

`--partial` is an explicit pilot mode. Use `--official` only for the official
artifact: it requires reflection records for exactly all 100 checked-in source
task IDs. Missing or extra task IDs fail freezing.

The artifact and its deterministic SHA are written to:

```text
outputs/phase2-travel-pilot-001/artifacts/travel/frozen_lessons.json
```

The artifact contains both an artifact SHA (which covers provenance and
experiment metadata) and a `lesson_set_sha256` computed only from the sorted,
exact lesson strings. This is also the canonical lesson set for a later
flat-vector baseline: index
the artifact's `lessons[*].content` strings unchanged.

## 3. Publish into a clean EVAL instance

Provision a new, empty EVAL instance. The publisher never deletes or cleans an
instance and therefore requires an explicit operator acknowledgement:

```bash
uv run mubit-state-bench-publish \
  --domain travel \
  --experiment-id phase2-travel-pilot-001 \
  --confirm-clean-eval-instance
```

Publication verifies both hashes and uses the activity and global-lesson APIs
to prove the remote EVAL instance is empty before writing. Each global lesson
must receive a successful durable write receipt. Publication then audits both
remote surfaces until their exact content-SHA set equals the frozen artifact;
only then is a local manifest written. It refuses to republish when a local
manifest already exists. Do not use the EVAL instance for synthetic seeding,
training traces, reflection, or outcomes.

## 4. Run read-only evaluation

Copy the non-secret SHA printed by the freezer into `.env` and set the run
identity:

```dotenv
MUBIT_STATE_BENCH_EXPERIMENT_ID="phase2-travel-pilot-001"
MUBIT_STATE_BENCH_ARM="mubit"
MUBIT_STATE_BENCH_EVAL_ARTIFACT_SHA256="<artifact SHA printed above>"
MUBIT_STATE_BENCH_EVAL_LESSON_SET_SHA256="<lesson-set SHA printed above>"
```

The standard `run_task` and `run_batch` entry points pass their actual run
index into the agent runtime context, so multi-run telemetry is labeled
correctly. `MUBIT_STATE_BENCH_RUN_NUMBER` is only a fallback for direct custom
orchestrator calls that do not supply `run_idx`.

Then run `MubitStateBenchAgent` using the same provider, model, reasoning
effort, and provider-side sampling settings as every comparison arm. Retrieval
telemetry now lands in:

```text
outputs/<experiment-id>/retrieval/<domain>.jsonl
```

Every event includes `experiment_id`, `arm`, artifact SHA, lesson-set SHA, and run number in
addition to the query, retrieval settings, latency, and returned evidence.

## Validation

```bash
uv run pytest tests/test_mubit_phase2.py -q
uv run pytest -q
```

The tests cover role/key/instance collisions, explicit ingest durability,
resume validation, arbitrary and held-out path rejection, deterministic parsing
and both artifact hashes, complete official-set enforcement, absence of outcome
calls, degraded reflection persistence, exact deduplication/provenance, remote
EVAL pre/post audits, artifact-only global publication, and secret-redacted
retrieval telemetry.
