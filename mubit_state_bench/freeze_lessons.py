"""Build a deterministic frozen lesson artifact from managed raw reflections."""

from __future__ import annotations

import argparse
import json

from mubit_state_bench.artifact import build_frozen_artifact
from mubit_state_bench.config import STATE_BENCH_DOMAINS
from mubit_state_bench.phase2_paths import Phase2Paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=STATE_BENCH_DOMAINS, required=True)
    parser.add_argument("--experiment-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--official",
        action="store_true",
        help="Require complete coverage of exactly all 100 checked-in training task IDs.",
    )
    mode.add_argument(
        "--partial",
        action="store_true",
        help="Explicitly create a pilot artifact from an incomplete training subset.",
    )
    args = parser.parse_args()
    paths = Phase2Paths(args.experiment_id)
    artifact = build_frozen_artifact(
        domain=args.domain,
        experiment_id=args.experiment_id,
        reflection_dir=paths.reflection_dir(args.domain),
        output_path=paths.artifact_path(args.domain),
        official_full_training_set=args.official,
    )
    print(
        json.dumps(
            {
                "domain": args.domain,
                "artifact_path": str(paths.artifact_path(args.domain)),
                "artifact_sha256": artifact["artifact_sha256"],
                "lesson_set_sha256": artifact["lesson_set_sha256"],
                "training_set_mode": artifact["training_set_mode"],
                "lesson_count": artifact["lesson_count"],
                "excluded_reflection_count": len(artifact["excluded_reflections"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
