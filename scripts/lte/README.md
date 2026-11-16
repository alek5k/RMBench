# Brev: XPolicyLab SmolVLA and LTE_SmolVLA

This workflow trains XPolicyLab's pinned SmolVLA or LTE_SmolVLA on an existing Brev GPU. Deployment uses one immutable source bundle assembled locally from RMBench and XPolicyLab; the remote host does not fetch either repository and no policy overlay is applied.

## Configuration

Training recipes are YAML launcher manifests under [../../config/lte/](../../config/lte/). They have two sections:

```yaml
launcher:
  task: rmbench
  env_cfg: aloha_agilex
  experiment_id: 261024_lte_smolvla_battery_blocks_ranking
  num_processes: 8
  cuda_visible_devices: "0,1,2,3,4,5,6,7"
args:
  policy.type: lte_smolvla
  batch_size: 16
  dataset.repo_id: example/dataset
```

Every key under `args` is the exact XPolicyLab/LeRobot trainer argument. The launcher only adds `--` and serializes the value; it does not rename variables or insert training defaults. Locally, a relative `output_dir` resolves from the YAML file's directory. On Brev, the launcher sets `RMBENCH_OUTPUT_ROOT`, so the same relative value is redirected to the external state directory, `~/RMBench-state/checkpoints/<output-name>`.

`batch_size` is the single batch-size control for either policy. LTE recipes explicitly declare their canonical `policy.input_features` and `policy.output_features`; startup derives the post-rename/post-padding dataset schema and fails if it differs, without modifying the configured schema. Policy defaults are documented separately in [`smolvla_defaults.yaml`](../../config/lte/smolvla_defaults.yaml) and [`lte_smolvla_defaults.yaml`](../../config/lte/lte_smolvla_defaults.yaml); recipes do not inherit from these reference files.

`launcher` contains only orchestration and dataset-staging metadata. In particular, `num_processes` and `cuda_visible_devices` select DDP processes but are not trainer arguments. Each YAML describes exactly one policy run. Baseline SmolVLA uses `policy.path`; LTE_SmolVLA uses `policy.type: lte_smolvla` and wraps the configured `policy.base_policy_path`.

`launcher.experiment_id` is an optional stable W&B group identifier. Use the form `<YYMMDD>_<algo>_<split>`, for example `261024_lte_smolvla_battery_blocks_ranking`. Put the same value under `wandb.experiment_id` in each evaluation manifest for that training run. It links the runs without relying on a mutable W&B display name; the evaluator retains its exact checkpoint path separately.

For the local temporal benchmark, `launcher.post_train_temporal_evals: true` makes a successful LTE WaitAtGoal+LiftQA training run evaluate its final checkpoint immediately afterwards. It runs 100 episodes on WaitAtGoal and then LiftQA, with both evaluation runs grouped under the training `experiment_id`. A failed training run never triggers evaluation.

Inspect the exact command without starting training:

```bash
bash scripts/lte/run_training.sh --dry-run "$PWD" lte \
  config/lte/lte_smolvla_rmbench_8gpu_60k.yaml
```

## Credentials

Stage credentials into this script directory. The file is mode `600` and ignored by Git:

```bash
bash scripts/lte/stage_env.sh /absolute/path/to/.env
```

`scripts/lte/.env` contains secrets only (`GITHUB_TOKEN`, `HF_TOKEN`, `WANDB_API_KEY`, and `NTFY_AUTH_TOKEN`). Keep non-secret W&B settings in YAML as exact keys such as `wandb.enable`, `wandb.entity`, and `wandb.project`. The source-bundle builder selects files through Git's ignore rules and rejects credential-like selected paths; `.env` is never included in the archive. If present, it is uploaded separately and installed as `~/RMBench-state/credentials.env` with mode `600`.

## Immutable source bundle

The launcher runs [`build_source_bundle.py`](build_source_bundle.py) locally. It selects each repository with:

```bash
git ls-files --cached --others --exclude-standard
```

This includes tracked source and intentional untracked work while respecting nested `.gitignore` files. RMBench's XPolicyLab gitlink is replaced with the separately selected XPolicyLab source under `RMBench/XPolicyLab/`. The resulting `tar.zst` preserves executable modes and safe relative symlinks, excludes Git metadata and secrets, and records source revisions, dirty state, file hashes, excluded unsafe symlinks, recipe, creation time, and bundle ID in `RMBench/SOURCE_BUNDLE.json`. A SHA-256 file accompanies the archive.

Remote setup checks the archive digest and member/link paths before extracting atomically. Releases and mutable state are separate:

```text
~/RMBench-releases/
├── <bundle-id>/RMBench/    # versioned source release
└── current -> <bundle-id>/

~/RMBench-state/
├── assets/
├── checkpoints/
├── data/
├── envs/
├── eval_result/
├── logs/
├── runs/
├── wandb/
└── credentials.env
```

Tracked code remains in the release, while generated children such as `assets/objects`, `assets/embodiments`, `data/demo_clean`, `data/download_cache`, and `envs/curobo` are state symlinks. LeRobot source is installed under `~/RMBench-state/envs/lerobot-v0.4.4` and linked at XPolicyLab's expected `policy/SmolVLA/smovla` path. An existing release with the same bundle ID is reused, never overlaid.

Build a bundle without launching:

```bash
python3 scripts/lte/build_source_bundle.py \
  --rmbench-root "$PWD" \
  --xpolicylab-root "$PWD/XPolicyLab" \
  --recipe config/lte/lte_smolvla_battery_try_1gpu_5k.yaml \
  --output-dir /tmp/rmbench-bundle
```

## Launch on Brev

```bash
bash scripts/lte/launch_brev.sh entire-aquamarine-penguin lte \
  config/lte/lte_smolvla_battery_try_1gpu_5k.yaml

bash scripts/lte/launch_brev.sh <eight-gpu-instance> lte \
  config/lte/lte_smolvla_rmbench_8gpu_60k.yaml
```

The full-suite recipe trains on `yinchenghust/rmbench_lerobot` for 60,000 updates with eight DDP processes. Its single `batch_size: 8` applies per process (effective global batch 64). It retains only intentional experiment settings; omitted values use XPolicyLab's defaults.

Bootstrap logs use `~/.lte-bootstrap/setup-<bundle-id>.log`. Training runs in a named `screen` session and writes logs below `~/RMBench-state/logs/`. A launch creates a new release and does not modify or restart an experiment already running from an older path.

## Local temporal benchmark recipe

```bash
bash scripts/lte/train_waitatgoal_liftqa_embedding_cache_local.sh \
  config/lte/lte_smolvla_waitatgoal_liftqa_2gpu_20k.yaml
```

This two-GPU recipe keeps the raw image cache and detached embedding cache on GPU. Its `batch_size` is per DDP process.

## Simulator evaluation

Training and simulator evaluation use separate Conda environments. Prepare the evaluator against a release while directing mutable dependencies to state:

```bash
RMBENCH_STATE_ROOT="$HOME/RMBench-state" \
RMBENCH_CREDENTIAL_FILE="$HOME/RMBench-state/credentials.env" \
  bash "$HOME/RMBench-releases/current/RMBench/scripts/lte/setup_eval_remote.sh" \
  "$HOME/RMBench-releases/current/RMBench"
```

Then launch from a reusable evaluation manifest:

```bash
RMBENCH_STATE_ROOT="$HOME/RMBench-state" \
RMBENCH_CREDENTIAL_FILE="$HOME/RMBench-state/credentials.env" \
  bash "$HOME/RMBench-releases/current/RMBench/scripts/lte/run_eval.sh" \
  "$HOME/RMBench-releases/current/RMBench" \
  config/lte/eval_smolvla_battery_try_5k_parallel.yaml
```

The manifest records the checkpoint, task, episode count, worker count, protocol, seed, task config, expert check, and W&B metadata. Relative checkpoint paths are resolved from the manifest; for immutable deployments, prefer an absolute path below `~/RMBench-state/checkpoints`. Parallel identity-aware policies require `protocol: ws`; each worker has a stable environment ID and resets only its own policy state.

The older positional form remains available for sequential compatibility:

```bash
bash "$HOME/RMBench-releases/current/RMBench/scripts/lte/run_eval.sh" \
  "$HOME/RMBench-releases/current/RMBench" lte /absolute/path/to/pretrained_model 100
```

Monitor Brev with:

```bash
brev exec entire-aquamarine-penguin 'screen -ls'
brev exec entire-aquamarine-penguin 'tail -f ~/RMBench-state/logs/*.log'
```
