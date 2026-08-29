"""Seed five synthetic global lessons into an isolated Mubit travel instance."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from mubit_state_bench.config import MubitStateBenchConfig
from mubit_state_bench.synthetic_lessons import TRAVEL_SYNTHETIC_LESSONS


def seed_travel_lessons() -> list[dict[str, object]]:
    from mubit import Client

    config = MubitStateBenchConfig.for_seed("travel")
    client = Client(
        endpoint=config.endpoint,
        transport=config.transport,
        run_id=config.run_id,
        api_key=config.api_key,
    )
    results: list[dict[str, object]] = []
    for lesson in TRAVEL_SYNTHETIC_LESSONS:
        response = client.remember(
            session_id=config.run_id,
            agent_id="statebench-phase1-seeder",
            item_id=lesson.item_id,
            content=lesson.content,
            intent="lesson",
            lesson_type=lesson.lesson_type,
            lesson_scope="global",
            lesson_importance="high",
            lesson_conditions=list(lesson.conditions),
            source="synthetic",
            upsert_key=lesson.item_id,
            idempotency_key=f"{config.run_id}:{lesson.item_id}",
            metadata={
                "benchmark": "microsoft-state-bench",
                "phase": "phase1",
                "domain": "travel",
                "synthetic": True,
            },
            wait=True,
        )
        results.append(
            {
                "item_id": lesson.item_id,
                "content": lesson.content,
                "accepted": bool(response),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["travel"],
        default="travel",
        help="Phase one seeds the travel smoke-test corpus only.",
    )
    args = parser.parse_args()
    load_dotenv()
    results = seed_travel_lessons()
    print(json.dumps({"domain": args.domain, "seeded": len(results), "lessons": results}, indent=2))


if __name__ == "__main__":
    main()
