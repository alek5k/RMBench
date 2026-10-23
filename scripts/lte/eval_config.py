#!/usr/bin/env python3
"""Validate an RMBench evaluation manifest and expose shell-safe values."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


POLICY_BY_MODE = {
    "smolvla": "SmolVLA",
    "lte": "LTE_SmolVLA",
}
LAUNCHER_KEYS = {
    "mode",
    "task",
    "env_cfg",
    "checkpoint",
    "policy_conda_env",
    "eval_conda_env",
    "policy_gpu_id",
    "env_gpu_id",
}
EVALUATION_KEYS = {
    "action_type",
    "episodes",
    "num_workers",
    "protocol",
    "seed",
    "task_config",
    "instruction_type",
    "expert_check",
    "frequency",
    "max_seed_attempts",
}
WANDB_KEYS = {"enabled", "entity", "project", "run_name"}


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {section} keys: {', '.join(sorted(unknown))}")


def positive_int(value: Any, key: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return result


def boolean(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be true or false")


def resolve_path(value: Any, config_path: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = config_path.resolve().parent / expanded
    return str(expanded.resolve())


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping")

    unknown = set(config) - {"launcher", "evaluation", "wandb"}
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")

    launcher = require_mapping(config, "launcher")
    evaluation = require_mapping(config, "evaluation")
    wandb = config.get("wandb", {})
    if not isinstance(wandb, dict):
        raise ValueError("wandb must be a mapping")

    reject_unknown(launcher, LAUNCHER_KEYS, "launcher")
    reject_unknown(evaluation, EVALUATION_KEYS, "evaluation")
    reject_unknown(wandb, WANDB_KEYS, "wandb")

    required_launcher = {"mode", "task", "env_cfg", "checkpoint"}
    missing = required_launcher - set(launcher)
    if missing:
        raise ValueError(f"Missing launcher keys: {', '.join(sorted(missing))}")

    mode = str(launcher["mode"])
    if mode not in POLICY_BY_MODE:
        raise ValueError("launcher.mode must be 'smolvla' or 'lte'")
    if evaluation.get("action_type", "joint") != "joint":
        raise ValueError("SmolVLA evaluation currently supports only action_type: joint")
    if evaluation.get("protocol", "ws") != "ws":
        raise ValueError("Parallel identity-aware evaluation requires protocol: ws")

    episodes = positive_int(evaluation.get("episodes", 100), "evaluation.episodes")
    workers = positive_int(evaluation.get("num_workers", 1), "evaluation.num_workers")
    if workers > episodes:
        raise ValueError("evaluation.num_workers cannot exceed evaluation.episodes")
    if "frequency" in evaluation:
        positive_int(evaluation["frequency"], "evaluation.frequency")
    if "max_seed_attempts" in evaluation:
        positive_int(evaluation["max_seed_attempts"], "evaluation.max_seed_attempts")
    if evaluation.get("instruction_type", "seen") not in {"seen", "unseen"}:
        raise ValueError("evaluation.instruction_type must be 'seen' or 'unseen'")
    if "expert_check" in evaluation:
        boolean(evaluation["expert_check"], "evaluation.expert_check")
    if "enabled" in wandb:
        boolean(wandb["enabled"], "wandb.enabled")

    return config


def shell_values(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    launcher = config["launcher"]
    evaluation = config["evaluation"]
    wandb = config.get("wandb", {})
    mode = str(launcher["mode"])
    values = {
        "MODE": mode,
        "POLICY": POLICY_BY_MODE[mode],
        "TASK": str(launcher["task"]),
        "ENV_CFG": str(launcher["env_cfg"]),
        "CKPT_NAME": resolve_path(launcher["checkpoint"], config_path),
        "POLICY_CONDA_ENV": str(launcher.get("policy_conda_env", "smolvla")),
        "EVAL_CONDA_ENV": str(launcher.get("eval_conda_env", "rmbench-eval")),
        "POLICY_GPU_ID": str(launcher.get("policy_gpu_id", 0)),
        "ENV_GPU_ID": str(launcher.get("env_gpu_id", 0)),
        "ACTION_TYPE": str(evaluation.get("action_type", "joint")),
        "TEST_NUM": str(int(evaluation.get("episodes", 100))),
        "NUM_WORKERS": str(int(evaluation.get("num_workers", 1))),
        "PROTOCOL": str(evaluation.get("protocol", "ws")),
        "SEED": str(int(evaluation.get("seed", 0))),
        "TASK_CONFIG": str(evaluation.get("task_config", "demo_clean")),
        "INSTRUCTION_TYPE": str(evaluation.get("instruction_type", "seen")),
        "EXPERT_CHECK": str(evaluation.get("expert_check", True)).lower(),
        "FREQUENCY": str(int(evaluation.get("frequency", 30))),
        "MAX_SEED_ATTEMPTS": str(
            int(evaluation.get("max_seed_attempts", max(int(evaluation.get("episodes", 100)) * 50, 20)))
        ),
        "WANDB_ENABLED": str(wandb.get("enabled", False)).lower(),
        "WANDB_ENTITY_CONFIG": str(wandb.get("entity", "")),
        "WANDB_PROJECT_CONFIG": str(wandb.get("project", "")),
        "WANDB_RUN_NAME": str(wandb.get("run_name", "")),
    }
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("command", choices=("shell", "validate"))
    cli = parser.parse_args()

    try:
        config = load_config(cli.config)
        if cli.command == "shell":
            for key, value in shell_values(config, cli.config).items():
                print(f"{key}={shlex.quote(value)}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Invalid evaluation configuration: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
