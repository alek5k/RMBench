#!/usr/bin/env python3
"""Merge local LeRobot datasets whose roots are separate directories.

LeRobot 0.4.4's ``lerobot-edit-dataset`` merge command accepts one shared
dataset root.  RMBench conversions use an exact root for each local repo, so
joint baseline controls need this thin adapter around LeRobot's public merge
API instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("REPO_ID", "ROOT"),
        required=True,
        help="A local LeRobot repo ID and its exact dataset root. Repeat per source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.output_root / "meta" / "info.json").is_file():
        print(f"Merged dataset already exists: {args.output_root}")
        return
    if args.output_root.exists():
        raise SystemExit(f"Output root already exists and is incomplete: {args.output_root}")

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = []
    for repo_id, root_text in args.source:
        root = Path(root_text)
        if not (root / "meta" / "info.json").is_file():
            raise SystemExit(f"Missing local LeRobot dataset: {root}")
        datasets.append(LeRobotDataset(repo_id, root=root))

    merged = merge_datasets(
        datasets,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_root,
    )
    print(
        f"Merged {len(datasets)} datasets into {args.output_root} "
        f"({merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames)"
    )


if __name__ == "__main__":
    main()
