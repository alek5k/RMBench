#!/usr/bin/env python3
"""Stream an XPolicyLab evaluation text log to a dedicated W&B run.

The evaluator remains the source of truth.  This sidecar only parses its
per-episode records, so it works for an already-running evaluation as well as
new ones and does not import the simulator or policy adapter.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--mode", choices=("smolvla", "lte"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--screen-session", help="Stop after this evaluation screen session exits.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step = checkpoint_step(args.checkpoint)
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not os.environ.get("WANDB_API_KEY") or not entity or not project:
        raise SystemExit("WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT are required")

    run = wandb.init(
        entity=entity,
        project=project,
        name=args.run_name,
        job_type="evaluation",
        group=f"RMBench-{args.task}-{args.env_cfg}-seed{args.seed}",
        tags=[
            "RMBench", "evaluation", args.mode, args.task, args.env_cfg,
            *([f"checkpoint-{step}"] if step is not None else []),
        ],
        config={
            "benchmark": "RMBench",
            "policy": "LTE_SmolVLA" if args.mode == "lte" else "SmolVLA",
            "task": args.task,
            "env_cfg": args.env_cfg,
            "checkpoint": args.checkpoint,
            "checkpoint_step": step,
            "seed": args.seed,
        },
    )

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
                    run.log(metrics, step=episode + 1)

            if args.screen_session and not screen_is_running(args.screen_session):
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
