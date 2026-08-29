"""Run one or more tasks and write trajectories.

Usage:
    uv run python -m state_bench.scripts.run_task
    uv run python -m state_bench.scripts.run_task --task 1-cancel_economy_domestic
    uv run python -m state_bench.scripts.run_task --task 1-cancel_economy_domestic 5-cancel_airline_cancelled --num-workers 2
    uv run python -m state_bench.scripts.run_task --task 1-cancel_economy_domestic,5-cancel_airline_cancelled --num-workers 2
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from state_bench.agents.base import BaseAgent
from state_bench.agents.loader import load_root_agent_class, load_root_client_class
from state_bench.agents.state_bench import StateBenchAgent
from state_bench.client import (
    BaseLLMClient,
    LLMClient,
    PooledLLMClient,
    build_llm_client,
    build_locked_judge_client,
    build_user_sim_client,
)
from state_bench.domain import get_domain_config
from state_bench.env_loader import load_task_environment
from state_bench.orchestrator import run_task
from state_bench.paths import domain_tasks_dir
from state_bench.protocol import format_domain_not_in_protocol_error, load_default_protocol, load_split_task_ids
from state_bench.schemas import TaskDefinition
from state_bench.scoring import TaskRequirementsJudge, UXQualityJudge
from state_bench.scripts.score import score_one


def _build_agent_model_metadata(args: argparse.Namespace) -> dict[str, str | None]:
    model_name = (args.agent_model_name or "").strip()
    if not model_name:
        raise ValueError("--agent-model-name is required")
    reasoning_level = args.agent_model_reasoning_level
    if isinstance(reasoning_level, str):
        reasoning_level = reasoning_level.strip() or None
    return {"model_name": model_name, "reasoning_level": reasoning_level}


def _parse_task_ids(raw_values: list[str]) -> list[str]:
    task_ids: list[str] = []
    for value in raw_values:
        task_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return task_ids


def _load_requested_tasks(tasks_dir: Path, task_ids: list[str]) -> list[TaskDefinition]:
    missing: list[str] = []
    tasks: list[TaskDefinition] = []
    for task_id in task_ids:
        task_path = tasks_dir / f"{task_id}.json"
        if not task_path.exists():
            missing.append(task_id)
            continue
        tasks.append(TaskDefinition.load(task_path))
    if missing:
        available = sorted(f.stem for f in tasks_dir.glob("*.json"))
        print(f"Task(s) not found: {', '.join(missing)}")
        print(f"Available ({len(available)}):")
        for task_id in available:
            print(f"  {task_id}")
        raise SystemExit(1)
    return tasks


def _arg_was_provided(argv: list[str], flag: str) -> bool:
    return flag in argv


def _validate_agent_client_args(args: argparse.Namespace, argv: list[str]) -> None:
    custom_agent_requested = bool(args.agent_class)
    custom_client_requested = bool(args.agent_client_class)
    if custom_client_requested and not custom_agent_requested:
        raise ValueError("--agent-client-class requires --agent-class.")
    if custom_client_requested and (
        _arg_was_provided(argv, "--agent-provider") or _arg_was_provided(argv, "--agent-api-key-var")
    ):
        raise ValueError(
            "--agent-provider and --agent-api-key-var are only valid with the built-in client. "
            "Custom clients should read their own configuration in from_env()."
        )


def _run_single_task(
    task: TaskDefinition,
    run_idx: int,
    user_override: str | None,
    client: BaseLLMClient | None,
    simulator_client: LLMClient | PooledLLMClient | None,
    domain,
    output_dir: Path,
    protocol=None,
    agent_model: dict[str, str | None] | None = None,
    agent_class: type[BaseAgent] | None = None,
    retrieve_learnings_top_k: int = 3,
    agent_reasoning_effort: str | None = None,
    task_requirements_judge: TaskRequirementsJudge | None = None,
    ux_judge: UXQualityJudge | None = None,
) -> dict:
    user_id = user_override or task.user_id
    if not user_id:
        return {"task_id": task.task_id, "run_idx": run_idx, "status": "ERR", "error": "no user_id"}

    try:
        env_data, env_path = load_task_environment(domain, task)
    except FileNotFoundError:
        return {
            "task_id": task.task_id,
            "run_idx": run_idx,
            "status": "ERR",
            "error": (
                f"checked-in task environment not found for domain {domain.name}; "
                "the benchmark package or repository checkout is incomplete"
            ),
        }

    metadata = {}
    if protocol is not None:
        metadata.update(protocol.simulator_metadata(domain.name))
        metadata["agent_name"] = (agent_class or StateBenchAgent).__name__
        if agent_model is not None:
            metadata["agent_model"] = agent_model

    trajectory = run_task(
        task,
        env_data,
        user_id,
        client,
        domain=domain,
        agent=None,
        env=None,
        trajectory_metadata=metadata,
        simulator_client=simulator_client,
        agent_class=agent_class,
        retrieve_learnings_top_k=retrieve_learnings_top_k,
        agent_reasoning_effort=agent_reasoning_effort,
        runtime_output_dir=str(output_dir),
        runtime_run_idx=run_idx,
    )

    tool_calls = []
    for msg in trajectory.conversation:
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])

    output_path = output_dir / f"{task.task_id}.json"
    trajectory.save(output_path)
    result = {
        "task_id": task.task_id,
        "run_idx": run_idx,
        "status": "OK",
        "output_path": str(output_path),
        "env_path": str(env_path),
        "task_summary": task.task_summary,
    }
    if task_requirements_judge is not None or ux_judge is not None:
        score_result = score_one(
            output_path,
            domain_tasks_dir(domain.name),
            task_requirements_judge,
            ux_judge,
            output_path,
            protocol,
            domain.name,
        )
        result["scoring_status"] = score_result.get("status")
        if score_result.get("status") == "ERR":
            result["scoring_error"] = score_result.get("error", "unknown scoring error")
        if "ux_score" in score_result:
            result["ux_score"] = score_result["ux_score"]
    if trajectory.efficiency:
        result["efficiency"] = {
            "turns": trajectory.efficiency.turns,
            "tool_calls": trajectory.efficiency.tool_calls,
            "tool_errors": trajectory.efficiency.tool_errors,
            "redundant_calls": trajectory.efficiency.redundant_calls,
        }
    if trajectory.token_usage is not None:
        result["token_usage"] = trajectory.token_usage.to_dict()
        if trajectory.token_usage.total_cost_usd:
            result["cost_usd"] = trajectory.token_usage.total_cost_usd
    return result


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run one or more benchmark tasks")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        required=False,
        help="One or more task IDs. Supports space-separated values and comma-separated lists.",
    )
    parser.add_argument("--user", type=str, default=None, help="Override user ID (default: from task definition)")
    parser.add_argument("--domain", type=str, default="travel", help="Domain name (default: travel)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: outputs/<domain>)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument(
        "--num-runs-idx-start",
        type=int,
        default=1,
        help="Starting run index for output directories (default: 1)",
    )
    parser.add_argument("--num-workers", dest="workers", type=int, default=None, help="Number of parallel task workers")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "test"],
        help="Task split to run when --task is omitted (default: all)",
    )
    parser.add_argument(
        "--agent-class",
        type=str,
        default=None,
        help=(
            "Agent class name under repo-root agents/. StateBenchAgent subclasses can use the built-in "
            "client; fully custom providers also require --agent-client-class."
        ),
    )
    parser.add_argument(
        "--agent-client-class",
        type=str,
        default=None,
        help="BaseLLMClient subclass name under repo-root clients/. Required for fully custom provider clients.",
    )
    parser.add_argument(
        "--retrieve-learnings-top-k",
        type=int,
        default=3,
        help="Benchmark-fixed retrieve_learnings top_k value (default: 3)",
    )
    parser.add_argument(
        "--agent-provider",
        type=str,
        default=os.environ.get("STATE_BENCH_AGENT_PROVIDER", "azure_openai"),
        choices=["azure_openai", "openai"],
        help="Provider for the built-in agent client",
    )
    parser.add_argument(
        "--agent-api-key-var",
        type=str,
        default=os.environ.get("STATE_BENCH_AGENT_API_KEY_VAR", "STATE_BENCH_AGENT_API_KEY"),
        help="API key env var for the built-in agent client",
    )
    parser.add_argument(
        "--agent-model-name",
        type=str,
        required=True,
        help="Agent model name reported in trajectories and metrics",
    )
    parser.add_argument(
        "--agent-model-reasoning-level",
        type=str,
        default=None,
        help="Optional agent model reasoning level reported in trajectories and metrics",
    )
    parser.add_argument(
        "--no-ux-score",
        action="store_true",
        help="Score task/state requirements inline but skip UX judge calls.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable all inline judge scoring.",
    )
    parser.add_argument(
        "--score-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for inline judge scoring (default: protocol judge setting)",
    )
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.num_runs_idx_start < 1:
        parser.error("--num-runs-idx-start must be >= 1")
    if args.task and args.split != "all":
        parser.error("--task and --split are mutually exclusive")
    if args.retrieve_learnings_top_k < 1:
        parser.error("--retrieve-learnings-top-k must be >= 1")
    if args.no_score and args.no_ux_score:
        parser.error("--no-score and --no-ux-score are mutually exclusive")
    try:
        _validate_agent_client_args(args, sys.argv[1:])
    except ValueError as exc:
        parser.error(str(exc))
    domain = get_domain_config(args.domain)
    protocol = load_default_protocol()
    try:
        agent_model = _build_agent_model_metadata(args)
    except ValueError as exc:
        parser.error(str(exc))
    protocol_errors = protocol.validate_prompt_hashes()
    if protocol_errors:
        parser.error("Protocol prompt validation failed:\n" + "\n".join(protocol_errors))
    if args.domain not in protocol.domains:
        parser.error(format_domain_not_in_protocol_error(args.domain, protocol))
    tasks_dir = domain_tasks_dir(args.domain)
    base_output = Path(args.output_dir) if args.output_dir else Path(f"outputs/{args.domain}")
    agent_class = load_root_agent_class(args.agent_class) if args.agent_class else None
    if agent_class is not None and args.agent_client_class is None and not issubclass(agent_class, StateBenchAgent):
        parser.error("--agent-class without --agent-client-class must be a StateBenchAgent subclass")

    print("Initializing agent LLM client...")
    if args.agent_client_class:
        client_class = load_root_client_class(args.agent_client_class)
        client = client_class.from_env()
        if not isinstance(client, BaseLLMClient):
            raise TypeError(f"{args.agent_client_class}.from_env() must return a BaseLLMClient")
    else:
        client = build_llm_client(
            provider=args.agent_provider,
            api_key_var=args.agent_api_key_var,
        )
    user_sim_client = build_user_sim_client()
    task_requirements_judge = None
    ux_judge = None
    if not args.no_score:
        judge_client = build_locked_judge_client()
        score_reasoning_effort = args.score_reasoning_effort or protocol.judge_reasoning_effort
        task_requirements_judge = TaskRequirementsJudge(
            client=judge_client,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=score_reasoning_effort,
        )
        if not args.no_ux_score:
            ux_judge = UXQualityJudge(
                client=judge_client,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=score_reasoning_effort,
            )

    task_ids = (
        _parse_task_ids(args.task)
        if args.task
        else load_split_task_ids(args.domain, args.split, protocol.split_version)
    )
    tasks = _load_requested_tasks(tasks_dir, task_ids)
    work_items = [
        (run_idx, task)
        for run_idx in range(args.num_runs_idx_start, args.num_runs_idx_start + args.num_runs)
        for task in tasks
    ]
    worker_count = args.workers or (1 if len(work_items) == 1 else min(len(work_items), 25))
    if worker_count < 1:
        parser.error("--num-workers must be >= 1")

    print(f"Loaded {len(tasks)} task(s):")
    for task in tasks:
        print(f"  {task.task_id}: {task.task_summary.splitlines()[0]}")

    agent_name = (agent_class or StateBenchAgent).__name__
    print(f"Agent: {agent_name}")
    print("Agent cost: user-reported via BaseAgent.add_cost_usd() when provided")
    if not args.no_score:
        score_mode = "task/state only" if args.no_ux_score else "task/state + UX"
        print(
            f"Scoring: inline {score_mode} "
            f"(reasoning_effort={args.score_reasoning_effort or protocol.judge_reasoning_effort})"
        )
    else:
        print("Scoring: off (--no-score set; use state_bench.scripts.score for local scoring)")

    run_indices = list(range(args.num_runs_idx_start, args.num_runs_idx_start + args.num_runs))
    run_dirs = {run_idx: base_output / f"run{run_idx}" for run_idx in run_indices}
    for run_dir in run_dirs.values():
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(
        f"# Running {len(tasks)} task(s) across run indices {run_indices[0]}..{run_indices[-1]}"
        f" => {len(work_items)} total job(s) with {worker_count} worker(s)"
    )
    print(
        f"# Scoring: {'off' if args.no_score else ('inline task/state only' if args.no_ux_score else 'inline task/state + UX')}"
    )
    print(f"{'#' * 60}")

    results: list[dict] = []
    if worker_count == 1:
        for run_idx, task in work_items:
            print(f"\n{'=' * 60}")
            print(f"Running: {task.task_id} | User: {args.user or task.user_id} | Run index {run_idx}")
            print(f"{'=' * 60}")
            result = _run_single_task(
                task,
                run_idx,
                args.user,
                client,
                user_sim_client,
                domain,
                run_dirs[run_idx],
                protocol,
                agent_model,
                agent_class,
                args.retrieve_learnings_top_k,
                (agent_model or {}).get("reasoning_level"),
                task_requirements_judge,
                ux_judge,
            )
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_single_task,
                    task,
                    run_idx,
                    args.user,
                    client,
                    user_sim_client,
                    domain,
                    run_dirs[run_idx],
                    protocol,
                    agent_model,
                    agent_class,
                    args.retrieve_learnings_top_k,
                    (agent_model or {}).get("reasoning_level"),
                    task_requirements_judge,
                    ux_judge,
                ): (run_idx, task)
                for run_idx, task in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                suffix = ""
                print(f"  [done] run{result['run_idx']} {result['task_id']}: {result['status']}{suffix}")

    results.sort(key=lambda item: (item["run_idx"], item["task_id"]))
    for result in results:
        print(f"\nTrajectory saved: {result.get('output_path', '<none>')}")
        if result["status"] != "OK":
            print(f"Run failed: {result.get('error', 'unknown error')}")
            continue
        print(f"  Task: {result['task_id']} | Run index {result['run_idx']}")
        print(f"  Environment: {result['env_path']}")
        if "efficiency" in result:
            efficiency = result["efficiency"]
            print(
                "  Efficiency: "
                f"{efficiency['turns']} turns, "
                f"{efficiency['tool_calls']} tool calls, "
                f"{efficiency['tool_errors']} errors, "
                f"{efficiency['redundant_calls']} redundant"
            )
        if "token_usage" in result:
            usage = result["token_usage"]
            print(
                "  Tokens: "
                f"input={usage['input_tokens']}, "
                f"cached_input={usage['cached_input_tokens']}, "
                f"output={usage['output_tokens']}, "
                f"cost=${usage['total_cost_usd']:.4f}"
            )
            print(
                "  Cost breakdown: "
                f"agent=${usage.get('agent_turn_cost_usd', 0.0):.4f}, "
                f"memory_ingest=${usage.get('memory_ingestion_cost_usd', 0.0):.4f}, "
                f"memory_retrieval=${usage.get('memory_retrieval_cost_usd', 0.0):.4f}"
            )
        if result.get("scoring_status"):
            suffix = f"; ux={result['ux_score']}" if "ux_score" in result else ""
            print(f"  Scoring: {result['scoring_status']}{suffix}")
        if result.get("scoring_error"):
            print(f"  Scoring error: {result['scoring_error']}")

    if not args.no_score:
        print(
            f"\nMetrics with: uv run python -m state_bench.scripts.compute_metrics --domain {args.domain} "
            f"--results-dir {base_output} "
            f"--num-runs {args.num_runs} --num-runs-idx-start {args.num_runs_idx_start} "
            f"--split {args.split} "
            f"--save-filepath {base_output / 'metrics.json'}"
        )
    else:
        print(
            f"\nScore with: uv run python -m state_bench.scripts.score --domain {args.domain} "
            f"--results-dir {base_output} "
            f"--num-runs {args.num_runs} --num-runs-idx-start {args.num_runs_idx_start} "
            f"--split {args.split}"
        )


if __name__ == "__main__":
    main()
