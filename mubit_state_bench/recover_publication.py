"""Recover a publication manifest using read-only verification of an existing EVAL instance."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from mubit_state_bench.artifact import load_frozen_artifact
from mubit_state_bench.config import STATE_BENCH_DOMAINS, MubitStateBenchConfig
from mubit_state_bench.phase2_paths import Phase2Paths
from mubit_state_bench.recovery import recover_publication_manifest
from mubit_state_bench.remote_audit import MubitEvalAuditor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=STATE_BENCH_DOMAINS, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--confirm-original-clean-preflight",
        action="store_true",
        help="Confirm that the original publication attempt observed zero activity and zero global lessons.",
    )
    parser.add_argument(
        "--original-durable-write-count",
        type=int,
        required=True,
        help="Durable writes completed before the original post-publication audit mismatch.",
    )
    args = parser.parse_args()
    if not args.confirm_original_clean_preflight:
        raise ValueError("Recovery requires confirmation of the original strict clean preflight")

    load_dotenv()
    paths = Phase2Paths(args.experiment_id)
    artifact_path = paths.artifact_path(args.domain)
    artifact = load_frozen_artifact(artifact_path, expected_domain=args.domain)
    config = MubitStateBenchConfig.for_publication(
        args.domain,
        args.experiment_id,
        artifact["artifact_sha256"],
        artifact["lesson_set_sha256"],
    )

    from mubit import Client

    client = Client(
        endpoint=config.endpoint,
        transport=config.transport,
        run_id=config.run_id,
        api_key=config.api_key,
    )
    manifest = recover_publication_manifest(
        artifact_path=artifact_path,
        publication_path=paths.publication_path(args.domain),
        config=config,
        auditor=MubitEvalAuditor(client),
        original_clean_preflight={"visible_activity_count": 0, "global_lesson_count": 0},
        original_durable_write_count=args.original_durable_write_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
