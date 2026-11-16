#!/usr/bin/env python3
"""Evaluate an XPolicyLab SmolVLA-family checkpoint in WaitAtGoal or LiftQA.

The temporal environments are not RMBench robot schemas, so they cannot use
the generic XPolicyLab policy-server evaluator.  This runner keeps the exact
XPolicyLab checkpoint and processor path, while loading only the environment
implementation from an LTE-DP checkout supplied at launch.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch


PROMPTS = {
    "waitatgoal": "go to the goal area, wait at least two seconds, and return to the start",
    "liftqa": "lift the object, wait at the goal, and return to the start",
}
ENV_ACTION_DIMS = {"waitatgoal": 2, "liftqa": 3}
ENV_MAX_STEPS = {"waitatgoal": 400, "liftqa": 1000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy-kind", choices=("smolvla", "lte_smolvla", "tempo_smolvla"), default="lte_smolvla")
    parser.add_argument("--environment", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--temporal-env-root", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, help="Optional per-episode cap; useful for a smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--experiment-id")
    parser.add_argument("--training-run-id", default="")
    parser.add_argument("--vlm-assets", type=Path, help="Local tokenizer/config metadata for a relocated full checkpoint.")
    parser.add_argument("--prompt", help="Exact instruction used in training; defaults to the legacy task prompt.")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_environment(environment: str, root: Path, seed: int) -> tuple[Any, int, Callable[[np.ndarray], np.ndarray]]:
    root = root.expanduser().resolve()
    if environment == "waitatgoal":
        module = load_module("rmbench_waitatgoal_env", root / "ltedp/envs/waitatgoal/waitatgoal_env.py")
        return (
            module.WaitAtGoal(render_size=144, render_mode="rgb_array"),
            ENV_MAX_STEPS[environment],
            lambda action: action,
        )

    # LiftQA's Robosuite import can otherwise enable Numba's on-disk cache in
    # environments where it has no cache locator. The simulation needs no JIT.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    module = load_module("rmbench_liftqa_env", root / "ltedp/envs/liftqa/lift_qa.py")
    environment_obj = module.create_env(
        render=False,
        control_hz=10,
        max_episode_length=ENV_MAX_STEPS[environment],
        camera_height_width=(144, 144),
        seed=seed,
    )
    return environment_obj, ENV_MAX_STEPS[environment], module.translate_3_dim_action_to_7d


def reset(environment: Any, seed: int) -> dict[str, Any]:
    # WaitAtGoal follows Gymnasium's reset(seed=...) protocol, whereas the
    # legacy Robosuite LiftQA environment fixes its seed at construction.
    try:
        result = environment.reset(seed=seed)
    except TypeError:
        result = environment.reset()
    return result[0] if isinstance(result, tuple) else result


def step(environment: Any, action: np.ndarray) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    result = environment.step(action)
    if len(result) == 5:
        observation, _, terminated, truncated, info = result
        return observation, bool(terminated or truncated), info
    observation, _, done, info = result
    return observation, bool(done), info


def to_policy_batch(observation: dict[str, Any], prompt: str, state_dim: int) -> dict[str, Any]:
    image = np.asarray(observation["full_image"])
    if image.ndim != 3:
        raise ValueError(f"Expected image with three dimensions, got {image.shape}")
    if image.shape[-1] in (1, 3):
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] not in (1, 3):
        raise ValueError(f"Expected CHW image, got {image.shape}")
    image = image.astype(np.float32, copy=False)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0

    state = np.asarray(observation["agent_pose"], dtype=np.float32).reshape(-1)
    if state.shape[0] > state_dim:
        raise ValueError(f"Environment state has {state.shape[0]} values but checkpoint accepts {state_dim}")
    padded_state = np.zeros(state_dim, dtype=np.float32)
    padded_state[: state.shape[0]] = state
    return {
        "observation.state": torch.from_numpy(padded_state).unsqueeze(0),
        "observation.images.camera1": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
        "task": [prompt],
    }


def checkpoint_step(path: Path) -> int | None:
    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)
    return None


def postprocess_action(postprocessor: Any, action: torch.Tensor) -> np.ndarray:
    result = postprocessor(action)
    if isinstance(result, dict):
        result = result["action"]
    return result.detach().float().cpu().numpy()


class TempoActionQueue:
    """Observe every environment frame, even while executing a cached action chunk."""
    def __init__(self, policy, postprocessor):
        self.policy,self.postprocessor=policy,postprocessor
        self.memory=None
        self.pending=deque()

    def next_action(self, batch):
        self.memory=self.policy.observe(batch,self.memory)
        if not self.pending:
            chunk=self.policy.act(batch,self.memory)[0]
            actions=postprocess_action(self.postprocessor,chunk)
            if actions.ndim!=2 or not len(actions) or not np.isfinite(actions).all():
                raise ValueError('Invalid TEMPO action chunk')
            self.pending.extend(actions[:self.policy.config.n_action_steps])
        return self.pending.popleft()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Not a LeRobot checkpoint: {checkpoint}")

    # Import after CLI parsing so this runner can be inspected without a GPU.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    if args.policy_kind == "lte_smolvla":
        from XPolicyLab.policy.LTE_SmolVLA.modeling_lte_smolvla import LTESmolVLAPolicy as Policy
    elif args.policy_kind == "tempo_smolvla":
        from XPolicyLab.policy.TEMPO_SmolVLA.modeling_tempo import TEMPOSmolVLAPolicy as Policy
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy

    device = torch.device(args.device)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if args.vlm_assets:
        assets=args.vlm_assets.expanduser().resolve()
        if not (assets/'config.json').is_file():raise FileNotFoundError(assets)
        config.vlm_model_name=str(assets)
        config.load_vlm_weights=False
    policy = Policy.from_pretrained(checkpoint, config=config, strict=True).to(device).eval()
    processor_overrides={"device_processor":{"device":str(device)}}
    if args.vlm_assets:
        processor_overrides['tokenizer_processor']={'tokenizer_name':str(assets)}
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint,
        preprocessor_overrides=processor_overrides)
    state_dim = int(config.robot_state_feature.shape[0])
    prompt = args.prompt or PROMPTS[args.environment]
    environment, max_steps, action_transform = make_environment(args.environment, args.temporal_env_root, args.seed)
    if args.max_steps is not None:
        max_steps = min(max_steps, args.max_steps)

    wandb_run = None
    if args.wandb_entity or args.wandb_project or args.wandb_run_name or args.experiment_id:
        if not all((args.wandb_entity, args.wandb_project, args.wandb_run_name, args.experiment_id)):
            raise ValueError("W&B logging requires entity, project, run name, and experiment ID")
        import wandb
        from XPolicyLab.utils.tempo_wandb import checkpoint_config, evaluation_display_name

        policy_name={"lte_smolvla":"LTE_SmolVLA","tempo_smolvla":"TEMPO_SmolVLA","smolvla":"SmolVLA"}[args.policy_kind]
        run_name = evaluation_display_name(checkpoint_config(checkpoint), args.wandb_run_name)
        tags=["RMBench", "evaluation", args.policy_kind, args.environment]
        if args.policy_kind=='tempo_smolvla':
            from XPolicyLab.utils.tempo_wandb import checkpoint_config,tempo_tags
            tags=tempo_tags(checkpoint_config(checkpoint),policy_name=policy_name,job_type='evaluation',
                            task=args.environment,env_cfg='temporal',checkpoint_step=checkpoint_step(checkpoint))
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            group=args.experiment_id,
            job_type="evaluation",
            tags=tags,
            config={
                "benchmark": "temporal",
                "policy": policy_name,
                "environment": args.environment,
                "checkpoint": str(checkpoint),
                "checkpoint_step": checkpoint_step(checkpoint),
                "seed": args.seed,
                "episodes": args.episodes,
                "max_steps": max_steps,
                "instruction": prompt,
                "experiment_id": args.experiment_id,
                "training_run_id": args.training_run_id,
            },
        )

    episode_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for episode in range(args.episodes):
            policy.reset()
            tempo_queue=TempoActionQueue(policy,postprocessor) if args.policy_kind=='tempo_smolvla' else None
            observation = reset(environment, args.seed + episode)
            success = False
            for step_index in range(max_steps):
                raw_batch = to_policy_batch(observation, prompt, state_dim)
                batch = preprocessor(raw_batch)
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                if tempo_queue is not None:
                    action=tempo_queue.next_action(batch)
                else:
                    normalized_action = policy.select_action(batch)
                    action = postprocess_action(postprocessor, normalized_action)[0]
                environment_action = action_transform(action[: ENV_ACTION_DIMS[args.environment]])
                observation, done, info = step(environment, environment_action)
                success = success or bool(info.get("is_success", False))
                if done:
                    break

            row = {"episode": episode + 1, "success": success, "steps": step_index + 1}
            episode_rows.append(row)
            successes = sum(item["success"] for item in episode_rows)
            metrics = {
                "eval/episode": episode + 1,
                "eval/episode_success": int(success),
                "eval/successes": successes,
                "eval/episodes_completed": episode + 1,
                "eval/success_rate": successes / (episode + 1),
                "eval/episode_steps": step_index + 1,
            }
            print(
                f"episode={episode + 1} success={success} "
                f"batch success: {successes} / {episode + 1}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=episode + 1)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    duration = time.monotonic() - started
    successes = sum(item["success"] for item in episode_rows)
    result = {
        "environment": args.environment,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "training_run_id": args.training_run_id,
        "policy_kind": args.policy_kind,
        "seed": args.seed,
        "episodes": len(episode_rows),
        "successes": successes,
        "success_rate": successes / len(episode_rows),
        "mean_episode_steps": float(np.mean([item["steps"] for item in episode_rows])),
        "wall_time_seconds": duration,
        "episode_results": episode_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if wandb_run is not None:
        wandb_run.summary.update({
            "eval/final_successes": successes,
            "eval/final_episodes": len(episode_rows),
            "eval/final_success_rate": result["success_rate"],
            "eval/mean_episode_steps": result["mean_episode_steps"],
            "eval/wall_time_seconds": duration,
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
