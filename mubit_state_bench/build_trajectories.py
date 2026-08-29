"""Ingest checked-in training trajectories and persist raw Mubit reflections."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from mubit_state_bench.config import STATE_BENCH_DOMAINS, MubitStateBenchConfig
from mubit_state_bench.io_utils import write_json_atomic
from mubit_state_bench.learning import MubitTrajectoryLearner
from mubit_state_bench.phase2_paths import Phase2Paths
from mubit_state_bench.trajectory import TrainTrajectoryLoader


def build_training_reflections(domain: str, experiment_id: str, limit: int | None = None) -> dict:
    from mubit import Client

    paths = Phase2Paths(experiment_id)
    loader = TrainTrajectoryLoader()
    sources = loader.load(domain, limit=limit)
    results: list[dict] = []
    for source in sources:
        output_path = paths.reflection_path(domain, source.task_id)
        if output_path.exists():
            raise FileExistsError(
                f"Raw reflection already exists for {source.task_id}: {output_path}. "
                "Use a new experiment_id instead of reflecting the same build run twice."
            )
        config = MubitStateBenchConfig.for_build(domain, source.task_id, experiment_id)
        client = Client(
            endpoint=config.endpoint,
            transport=config.transport,
            run_id=config.run_id,
            api_key=config.api_key,
        )
        learner = MubitTrajectoryLearner(client=client, config=config, loader=loader)
        results.append(learner.learn(source, output_path))

    manifest = {
        "schema_version": "phase2_build_manifest_v1",
        "experiment_id": experiment_id,
        "domain": domain,
        "requested_limit": limit,
        "task_count": len(results),
        "tasks": [
            {
                "task_id": result["task_id"],
                "source_sha256": result["source_sha256"],
                "status": result["status"],
            }
            for result in results
        ],
    }
    write_json_atomic(paths.reflection_dir(domain).parent / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=STATE_BENCH_DOMAINS, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only cap. Omit to process all 100 checked-in training trajectories.",
    )
    args = parser.parse_args()
    load_dotenv()
    manifest = build_training_reflections(args.domain, args.experiment_id, args.limit)
    print(json.dumps(manifest, indent=2))
    if any(task["status"] != "ok" for task in manifest["tasks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
