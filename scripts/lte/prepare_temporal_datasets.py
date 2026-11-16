#!/usr/bin/env python3
"""Download and convert the published WaitAtGoal and LiftQA datasets.

The original demonstrations are public Zarr archives rather than Hub-native
LeRobot datasets.  This script owns the RMBench provisioning path: it fetches
the archives, preserves them below ``<output-root>/downloads``, and writes the
two LeRobot v3 datasets directly below ``<output-root>``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import zarr


SOURCE_REPOSITORY = "erdrsvhr9/temporally_conditioned_diffusion_policy"
DATASETS = (
    {
        "name": "waitatgoal_v4",
        "archive": "wait_at_goal_dataset_v4.zarr.zip",
        "store": "wait_at_goal_dataset_v4.zarr",
        "repo_id": "local/waitatgoal_v4",
        "robot_type": "waitatgoal",
        "task": "wait at the goal and return to the start",
        "fps": 10,
        "extra": {
            "agent_velocity": "observation.agent_velocity",
            "goal_pose": "observation.goal_pose",
            "goal_state": "observation.goal_state",
            "reward": "next.reward",
            "step_count": "observation.step_count",
            "time_at_rest": "observation.time_at_rest",
            "wait_time": "observation.wait_time",
            "wait_times_each_visit": "observation.wait_times_each_visit",
        },
    },
    {
        "name": "liftqa_v2",
        "archive": "lift_qa_v2.zarr.zip",
        "store": "lift_qa_v2.zarr",
        "repo_id": "local/liftqa_v2",
        "robot_type": "liftqa",
        "task": "lift the object, wait at the goal, and return to the start",
        "fps": 10,
        "extra": {
            "agent_velocity": "observation.agent_velocity",
            "cur_time": "observation.cur_time",
            "done": "next.done",
            "gripper_velocity": "observation.gripper_velocity",
            "joint_velocity": "observation.joint_velocity",
            "reward": "next.reward",
            "step_count": "observation.step_count",
            "wait_time": "observation.wait_time",
            "wait_times_each_visit": "observation.wait_times_each_visit",
        },
    },
)


def vector_feature(array: object) -> dict[str, object]:
    shape = tuple(int(dimension) for dimension in array.shape[1:])
    return {"dtype": "float32", "shape": shape or (1,), "names": [f"dim_{index}" for index in range(len(shape or (1,)))]}


def value(array: object, index: int) -> np.ndarray:
    result = np.asarray(array[index], dtype=np.float32)
    return result.reshape(1) if result.ndim == 0 else result


def image(value_: object) -> np.ndarray:
    result = np.asarray(value_)
    if result.ndim != 3:
        raise ValueError(f"Expected CHW image, received {result.shape}")
    if result.dtype != np.uint8:
        result = np.clip(result * 255 if np.issubdtype(result.dtype, np.floating) and result.max() <= 1 else result, 0, 255).astype(np.uint8)
    return result


def archive_store(spec: dict[str, object], downloads: Path, raw: Path) -> Path:
    archive = hf_hub_download(
        repo_id=SOURCE_REPOSITORY,
        repo_type="dataset",
        filename=f"demonstration/{spec['archive']}",
        local_dir=downloads,
    )
    store = raw / str(spec["store"])
    if not store.is_dir():
        with zipfile.ZipFile(archive) as contents:
            for member in contents.infolist():
                destination = (raw / member.filename).resolve()
                if not destination.is_relative_to(raw.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            contents.extractall(raw)
    if not store.is_dir():
        raise FileNotFoundError(f"Zarr store missing after extraction: {store}")
    return store


def convert(spec: dict[str, object], source: Path, output_root: Path) -> None:
    destination = output_root / str(spec["name"])
    source_data = zarr.open(str(source), mode="r")["data"]
    action, state, frames = source_data["action"], source_data["agent_pose"], source_data["full_image"]
    episode_ends = np.asarray(zarr.open(str(source), mode="r")["meta"]["episode_ends"], dtype=np.int64)
    if not len(episode_ends) or episode_ends[-1] != len(action):
        raise ValueError(f"Invalid episode boundaries in {source}")
    if (destination / "meta" / "info.json").is_file():
        info = json.loads((destination / 'meta/info.json').read_text())
        if (info.get('total_frames'), info.get('total_episodes')) != (len(action), len(episode_ends)):
            raise FileExistsError(f"Incomplete dataset exists: {destination}")
        if not (destination / 'meta/stats.json').is_file() or not any((destination/'data').rglob('*.parquet')):
            raise FileExistsError(f"Dataset lacks finalized data/statistics: {destination}")
        print(f"[skip complete] {spec['name']}: {destination}", flush=True)
        return
    if destination.exists():
        raise FileExistsError(f"Incomplete dataset exists: {destination}")
    first_image = image(frames[0])
    extras = {target: source_data[key] for key, target in dict(spec["extra"]).items() if key in source_data}
    features: dict[str, dict[str, object]] = {
        "action": vector_feature(action),
        "observation.state": vector_feature(state),
        "observation.image": {"dtype": "image", "shape": tuple(int(dimension) for dimension in first_image.shape), "names": ["channel", "height", "width"]},
        **{target: vector_feature(array) for target, array in extras.items()},
    }
    dataset = LeRobotDataset.create(repo_id=str(spec["repo_id"]), root=destination, fps=int(spec["fps"]), features=features, use_videos=False, robot_type=str(spec["robot_type"]))
    try:
        start = 0
        for episode, end in enumerate(episode_ends, start=1):
            for index in range(start, int(end)):
                frame = {"action": value(action, index), "observation.state": value(state, index), "observation.image": image(frames[index]), "task": str(spec["task"])}
                frame.update({target: value(array, index) for target, array in extras.items()})
                dataset.add_frame(frame)
            dataset.save_episode()
            start = int(end)
            print(f"[{spec['name']}] episode {episode}/{len(episode_ends)}", flush=True)
    finally:
        dataset.finalize()
    (destination/'temporal_conversion_complete.json').write_text(json.dumps({
        'source_repository': SOURCE_REPOSITORY, 'source_archive': spec['archive'],
        'episodes': len(episode_ends), 'frames': len(action), 'complete': True,
    }, indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=[str(spec['name']) for spec in DATASETS],
                        help="Prepare only these datasets (default: both temporal tasks).")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    downloads, raw = output_root / "downloads", output_root / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS:
        if args.datasets and spec['name'] not in args.datasets:
            continue
        # Shared by early CPU preparation and the later training queue. Hold
        # through download, extraction and finalization, not just file creation.
        with (output_root / f".prepare-{spec['name']}.lock").open('a') as lock:
            print(f"[lock] Waiting for {spec['name']} preparation lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            convert(spec, archive_store(spec, downloads, raw), output_root)


if __name__ == "__main__":
    main()
