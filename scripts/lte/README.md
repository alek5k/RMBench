# Brev: XPolicyLab SmolVLA and LTE_SmolVLA

This workflow trains XPolicyLab's pinned SmolVLA and the local LTE_SmolVLA adapter on an existing Brev GPU.

## Configuration

Training recipes are YAML launcher manifests under [../../config/lte/](../../config/lte/). They have two sections:

```yaml
launcher:
  task: rmbench
  env_cfg: aloha_agilex
  num_processes: 8
  cuda_visible_devices: "0,1,2,3,4,5,6,7"
args:
  policy.type: lte_smolvla
  batch_size: 16
  dataset.repo_id: example/dataset
```

Every key under `args` is the exact XPolicyLab/LeRobot trainer argument. The launcher only adds `--` and serializes the value; it does not rename variables or insert training defaults. The sole path convenience is `output_dir`: a relative value is resolved from the YAML file's directory, so `../../checkpoints/<recipe>` lands under the RMBench checkout on both local and remote machines. Therefore `batch_size` is the single batch-size control for either policy. LTE recipes explicitly declare their canonical `policy.input_features` and `policy.output_features`; startup derives the post-rename/post-padding dataset schema and fails if it differs, without modifying the configured schema. Policy defaults are documented separately in [`smolvla_defaults.yaml`](../../config/lte/smolvla_defaults.yaml) and [`lte_smolvla_defaults.yaml`](../../config/lte/lte_smolvla_defaults.yaml); recipes do not inherit from these reference files.

`launcher` contains only orchestration and dataset-staging metadata. In particular, `num_processes` and `cuda_visible_devices` select DDP processes but are not trainer arguments. Each YAML describes exactly one policy run. Baseline SmolVLA uses `policy.path`; LTE_SmolVLA uses `policy.type: lte_smolvla` and wraps the configured `policy.base_policy_path`.

Inspect the exact command without starting training:

```bash
bash scripts/lte/run_training.sh --dry-run "$PWD" lte \
  config/lte/lte_smolvla_rmbench_8gpu_60k.yaml
```

## Credentials

Copy credentials into this script directory. The file is mode `600` and ignored by Git:

```bash
bash scripts/lte/stage_env.sh /absolute/path/to/.env
```

`scripts/lte/.env` contains secrets only (`GITHUB_TOKEN`, `HF_TOKEN`, `WANDB_API_KEY`, and `NTFY_AUTH_TOKEN`). Keep non-secret W&B settings in YAML as exact keys such as `wandb.enable`, `wandb.entity`, and `wandb.project`. The API key remains an environment credential and is never put on the command line.

## Launch on Brev

```bash
bash scripts/lte/launch_brev.sh entire-aquamarine-penguin lte \
  config/lte/lte_smolvla_battery_try_1gpu_5k.yaml

bash scripts/lte/launch_brev.sh <eight-gpu-instance> lte \
  config/lte/lte_smolvla_rmbench_8gpu_60k.yaml
```

The full-suite recipe trains on `yinchenghust/rmbench_lerobot` for 60,000 updates with eight DDP processes. Its single `batch_size: 8` applies per process (effective global batch 64). It retains only intentional experiment settings; omitted values use XPolicyLab's defaults.

Bootstrap runs detached in `~/.lte-bootstrap/setup.log`. Training runs in a named `screen` session and writes logs below `~/RMBench-lte/logs/`.

## Local temporal benchmark recipe

```bash
bash scripts/lte/train_waitatgoal_liftqa_embedding_cache_local.sh \
  config/lte/lte_smolvla_waitatgoal_liftqa_2gpu_20k.yaml
```

This two-GPU recipe keeps the raw image cache and detached embedding cache on GPU. Its `batch_size` is per DDP process.

## Simulator evaluation

Training and simulator evaluation use separate Conda environments. Prepare the evaluator once:

```bash
bash scripts/lte/setup_eval_remote.sh ~/RMBench-lte
```

Then launch from a reusable evaluation manifest:

```bash
bash scripts/lte/run_eval.sh ~/RMBench-lte \
  config/lte/eval_smolvla_battery_try_5k_parallel.yaml
```

The manifest records the checkpoint, task, episode count, worker count, protocol,
seed, task config, expert check, and W&B metadata. Relative checkpoint paths are
resolved from the manifest. Parallel identity-aware policies require `protocol: ws`;
each worker has a stable environment ID and resets only its own policy state.
Credentials remain in `scripts/lte/.env`, not in the manifest.

The older positional form remains available for sequential compatibility:

```bash
bash scripts/lte/run_eval.sh ~/RMBench-lte lte
bash scripts/lte/run_eval.sh ~/RMBench-lte lte /absolute/path/to/pretrained_model 2
```

Monitor Brev with:

```bash
brev exec entire-aquamarine-penguin 'screen -ls'
brev exec entire-aquamarine-penguin 'tail -f ~/RMBench-lte/logs/*.log'
```
