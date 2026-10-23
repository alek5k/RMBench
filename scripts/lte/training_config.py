#!/usr/bin/env python3
"""Read launcher metadata and exact XPolicyLab training arguments from YAML."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


LAUNCHER_KEYS = {
    "task",
    "env_cfg",
    "num_processes",
    "cuda_visible_devices",
    "conda_env",
    "lte_smolvla_source",
    "single_camera",
    "wait_for_evals",
    "dataset",
}
DATASET_SETUP_KEYS = {"repo_ids", "download"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping")

    unknown = set(config) - {"launcher", "args"}
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")

    launcher = config.get("launcher", {})
    args = config.get("args")
    if not isinstance(launcher, dict):
        raise ValueError("launcher must be a mapping")
    if not isinstance(args, dict) or not args:
        raise ValueError("args must be a non-empty mapping")

    unknown = set(launcher) - LAUNCHER_KEYS
    if unknown:
        raise ValueError(f"Unknown launcher keys: {', '.join(sorted(unknown))}")
    if "task" not in launcher or "env_cfg" not in launcher:
        raise ValueError("launcher.task and launcher.env_cfg are required")

    dataset = launcher.get("dataset", {})
    if dataset and not isinstance(dataset, dict):
        raise ValueError("launcher.dataset must be a mapping")
    unknown = set(dataset) - DATASET_SETUP_KEYS
    if unknown:
        raise ValueError(f"Unknown launcher.dataset keys: {', '.join(sorted(unknown))}")

    for key in args:
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"Invalid trainer argument key: {key!r}")
    if "dataset.repo_id" not in args:
        raise ValueError("args must contain dataset.repo_id")
    return config


def scalar(value: Any, key: str, config_path: Path) -> str:
    if value is None:
        raise ValueError(f"Trainer argument {key} cannot be null")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if key == "output_dir" and not os.path.isabs(expanded):
        return str((config_path.resolve().parent / expanded).resolve())
    return expanded


def trainer_args(config: dict[str, Any], config_path: Path) -> list[str]:
    return [
        f"--{key}={scalar(value, key, config_path)}"
        for key, value in config["args"].items()
    ]


def launcher_value(config: dict[str, Any], key: str) -> Any:
    launcher = config.get("launcher", {})
    if key.startswith("dataset."):
        value: Any = launcher.get("dataset", {})
        for part in key.split(".")[1:]:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value
    return launcher.get(key)


def print_value(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, list):
        print(",".join(str(item) for item in value))
    else:
        print(os.path.expandvars(os.path.expanduser(str(value))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("args")
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("key")
    subparsers.add_parser("shell")
    subparsers.add_parser("validate")

    cli = parser.parse_args()
    try:
        config = load_config(cli.config)
        if cli.command == "args":
            for argument in trainer_args(config, cli.config):
                print(argument)
        elif cli.command == "get":
            print_value(launcher_value(config, cli.key))
        elif cli.command == "shell":
            print(shlex.join(trainer_args(config, cli.config)))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Invalid training configuration: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
