#!/usr/bin/env python3
"""Stream an XPolicyLab evaluation text log to a dedicated W&B run.

The evaluator remains the source of truth.  This sidecar only parses its
per-episode records, so it works for an already-running evaluation as well as
new ones and does not import the simulator or policy adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

import wandb


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPISODE_RECORD = re.compile(
    r"episode=(?P<episode>\d+).*?success=(?P<success>True|False).*?"
    r"batch success:\s*(?P<successes>\d+)\s*/\s*(?P<total>\d+)",
    re.DOTALL,
)


def checkpoint_step(checkpoint: str) -> int | None:
    """Return a numeric checkpoint directory component when one is present."""
    for part in reversed(Path(checkpoint).parts):
        if part.isdigit():
            return int(part)
    return None


def screen_is_running(name: str) -> bool:
    result = subprocess.run(
        ["screen", "-ls"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    # screen prefixes a session name with its PID.  Match the complete session
    # name, rather than a substring, so this logger's own ``-wandb`` session
    # does not keep it alive after the evaluator exits.
    return re.search(rf"^\s*\d+\.{re.escape(name)}\s+\(", result.stdout, re.MULTILINE) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("smolvla", "lte", "tempo"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="", help="Stable experiment ID shared with the training run.")
    parser.add_argument("--screen-session", help="Stop after this evaluation screen session exits.")
    parser.add_argument("--process-id", type=int, help="Stop after the evaluator process exits.")
    parser.add_argument("--training-run-id", default="", help="Source W&B training run ID.")
    parser.add_argument("--policy-name", choices=("TEMPO_SmolVLA", "TEMPO_Pi05"),
                        help="Exact TEMPO adapter; otherwise read from checkpoint metadata.")
    parser.add_argument("--tag", action="append", default=[], help="Additional W&B tag (repeatable).")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--metadata-helper", type=Path,
                        help="Explicit bundled tempo_wandb.py for older pinned policy releases.")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate CLI/metadata and print resolved logging config without creating a W&B run.")
    parser.add_argument("--once", action="store_true", help="Backfill an existing log once, without waiting for a process.")
    parser.add_argument("--checkpoint-in-name", action="store_true", help="Distinguish checkpoint-sweep evals by step in the display name.")
    parser.add_argument("--wandb-run-id", help="Resume a verified existing evaluation run, never a training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step = checkpoint_step(args.checkpoint)
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not os.environ.get("WANDB_API_KEY") or not entity or not project:
        raise SystemExit("WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT are required")
    group = args.group or f"RMBench-{args.task}-{args.env_cfg}-seed{args.seed}"
    policy_name = {"lte":"LTE_SmolVLA", "tempo":"TEMPO_SmolVLA", "smolvla":"SmolVLA"}[args.mode]
    run_name = args.run_name
    tags = ["RMBench", "evaluation", args.mode, args.task, args.env_cfg,
            *([f"checkpoint-{step}"] if step is not None else [])]
    if args.mode == "tempo" or (Path(args.checkpoint) / "tempo_run.json").is_file():
        helper = getattr(args, "metadata_helper", None)
        if helper:
            spec = importlib.util.spec_from_file_location("eval_tempo_wandb", helper)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        else:
            from XPolicyLab.utils import tempo_wandb as module
        checkpoint_config, infer_policy = module.checkpoint_config, module.infer_policy
        merge_tags, tempo_tags = module.merge_tags, module.tempo_tags
        evaluation_display_name = module.evaluation_display_name
        metadata = checkpoint_config(args.checkpoint)
        run_name = evaluation_display_name(metadata, run_name)
        policy_name = getattr(args, "policy_name", None) or infer_policy(metadata) or policy_name
        tags = merge_tags(tags, tempo_tags(metadata, policy_name=policy_name, job_type="evaluation",
                                          task=args.task, env_cfg=args.env_cfg, checkpoint_step=step))
    # Passing explicit tags to wandb.init overrides WANDB_TAGS, so retain those too.
    for tag in [*os.environ.get("WANDB_TAGS", "").split(","), *getattr(args, "tag", [])]:
        if tag.strip() and tag.strip() not in tags:
            tags.append(tag.strip())
    if getattr(args, "checkpoint_in_name", False) and step is not None:
        prefix = run_name[:-5] if run_name.endswith('_eval') else run_name
        run_name = f'{prefix}_checkpoint_{step}_eval'

    config = {
        "benchmark": "RMBench", "policy": policy_name, "task": args.task,
        "env_cfg": args.env_cfg, "checkpoint": args.checkpoint,
        "checkpoint_step": step, "experiment_id": group, "seed": args.seed,
        "training_run_id": args.training_run_id,
    }
    if getattr(args, "check_config", False):
        print(json.dumps({"name": run_name, "group": group, "tags": tags, "config": config}))
        return
    resume_id = getattr(args, 'wandb_run_id', None)
    if resume_id and resume_id == args.training_run_id:
        raise ValueError('Refusing to resume the training run as an evaluation')
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="evaluation",
        group=group,
        tags=tags,
        config=config,
        **({'id': resume_id, 'resume': 'must'} if resume_id else {}),
    )
    resume_after_step = int(run.step)-1 if resume_id else -1

    seen_episodes: set[int] = set()
    final_total = 0
    final_successes = 0
    try:
        while True:
            if args.log_file.exists():
                text = ANSI_ESCAPE.sub("", args.log_file.read_text(errors="replace"))
                for match in EPISODE_RECORD.finditer(text):
                    episode = int(match["episode"])
                    if episode in seen_episodes:
                        continue
                    seen_episodes.add(episode)
                    successes = int(match["successes"])
                    total = int(match["total"])
                    final_total = max(final_total, total)
                    if total >= final_total:
                        final_successes = successes
                    metrics = {
                            "eval/episode": episode,
                            "eval/episode_success": int(match["success"] == "True"),
                            "eval/successes": successes,
                            "eval/episodes_completed": total,
                            "eval/success_rate": successes / total,
                    }
                    if step is not None:
                        metrics["eval/checkpoint_step"] = step
                    # Parallel workers finish episode IDs out of order. W&B's
                    # step follows completed-count order; retain the episode ID
                    # as a metric so no valid result is silently dropped.
                    if total > resume_after_step:
                        run.log(metrics, step=total)

            if getattr(args, "once", False):
                break
            if args.screen_session and not screen_is_running(args.screen_session):
                break
            if args.process_id:
                try:
                    os.kill(args.process_id, 0)
                except ProcessLookupError:
                    break
            time.sleep(args.poll_seconds)
    finally:
        if final_total:
            run.summary["eval/final_successes"] = final_successes
            run.summary["eval/final_episodes"] = final_total
            run.summary["eval/final_success_rate"] = final_successes / final_total
        if step is not None:
            run.summary["eval/checkpoint_step"] = step
        run.finish()


if __name__ == "__main__":
    main()
