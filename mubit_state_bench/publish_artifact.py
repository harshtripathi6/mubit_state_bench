"""Publish a managed frozen lesson artifact into an isolated clean EVAL instance."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from mubit_state_bench.artifact import load_frozen_artifact
from mubit_state_bench.config import STATE_BENCH_DOMAINS, MubitStateBenchConfig
from mubit_state_bench.phase2_paths import Phase2Paths
from mubit_state_bench.publication import FrozenArtifactPublisher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=STATE_BENCH_DOMAINS, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--confirm-clean-eval-instance",
        action="store_true",
        help="Required acknowledgement that this role/domain EVAL instance contains no other memory.",
    )
    args = parser.parse_args()
    load_dotenv()
    paths = Phase2Paths(args.experiment_id)
    artifact = load_frozen_artifact(paths.artifact_path(args.domain), expected_domain=args.domain)
    config = MubitStateBenchConfig.for_publication(
        args.domain,
        args.experiment_id,
        artifact["artifact_sha256"],
    )

    from mubit import Client

    client = Client(
        endpoint=config.endpoint,
        transport=config.transport,
        run_id=config.run_id,
        api_key=config.api_key,
    )
    publisher = FrozenArtifactPublisher(client=client, config=config)
    manifest = publisher.publish(
        paths.artifact_path(args.domain),
        paths.publication_path(args.domain),
        clean_eval_instance_confirmed=args.confirm_clean_eval_instance,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
