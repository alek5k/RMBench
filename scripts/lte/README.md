# Brev: XPolicyLab SmolVLA and LTE_SmolVLA

This workflow trains XPolicyLab's pinned SmolVLA (`LeRobot v0.4.4`) and the local `LTE_SmolVLA` adapter on an existing Brev GPU. It does **not** use LTE-DP's setup scripts or its separate LeRobot installation.

## Configuration and credentials

The versioned single-task configuration is [../../config/lte/battery_try.env](../../config/lte/battery_try.env). It selects RMBench's `battery_try` demonstrations for `aloha_agilex`, converts them to a local LeRobot-v3 cache, and trains baseline then LTE sequentially. Its `local/...` dataset ID is only a cache namespace; neither conversion nor training uploads a dataset. All paths are under the Brev user's home directory automatically. It defines the shared SmolVLA policy reference and feature schema; LTE inherits that base configuration, then adds only its temporal settings. The bootstrap also applies a small, versioned LTE overlay fix so explicit shared command-line policy settings are retained during inheritance.

RMBench trajectories use the XPolicyLab `xspark` HDF5 layout, so the converter uses its compatible `RoboDojo` loader internally; model run names and checkpoints remain labelled `RMBench`.

Copy credentials into this script directory. The file is mode `600` and ignored by Git:

```bash
cd ~/Repos/RMBench
bash scripts/lte/stage_env.sh /absolute/path/to/.env
```

`scripts/lte/.env` contains credentials only: GitHub, Hugging Face, Weights & Biases (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`), and ntfy. Never put run configuration or training settings in it.

## Launch

```bash
cd ~/Repos/RMBench
bash scripts/lte/launch_brev.sh entire-aquamarine-penguin both
```

### Eight-GPU LTE recipe aligned to SimpleMemVLA

`config/lte/rmbench_lte_smolvla_simplememvla_8gpu.env` is an LTE-only,
single-node DDP recipe for an eight-GPU Brev instance. It trains on
SimpleMemVLA's published full-suite, 10-task RMBench dataset
(`yinchenghust/rmbench_lerobot`) rather than a `battery_try` subset, and maps
the compatible parts of
SimpleMemVLA's RMBench recipe: batch 8 per GPU (effective batch 64), 60,000
updates, checkpoints every 3,000 updates, 1,000 warmup updates, and its
differential learning rates.  The SmolVLA VLM—including its vision
encoder—uses 5e-6; the action expert, state projector, and LTE modules use
5e-5. It also follows SimpleMemVLA's AdamW treatment of biases and
normalization parameters (zero weight decay). Launch it on an existing
eight-GPU instance with:

```bash
bash scripts/lte/launch_brev.sh <eight-gpu-brev-instance> lte \
  config/lte/rmbench_lte_smolvla_simplememvla_8gpu.env
```

The training script uses `torchrun --nproc_per_node=8`, and fails before model
loading if fewer than eight GPUs are visible. It does not create a Brev
instance. The LTE run is named `rmbench_lte_smolvla_60k` in W&B. The older
`battery_try_lte_simplememvla_8gpu.env` stays available for a single-task
ablation.

SimpleMemVLA's README notes that its published step budgets target a much
larger multi-node global batch (about 512); its checked-in eight-GPU launcher
itself defaults to an effective batch of 64. This LTE recipe deliberately
matches that executable per-GPU batch (8 × 8 GPUs). The pinned LeRobot trainer
has no gradient-accumulation option, so it does not pretend to reproduce a
512-sample update by silently increasing the per-GPU batch.

This is parameter-aligned, not an architectural reproduction: SimpleMemVLA
uses Qwen3.5-4B and a DiT action head, whereas LTE uses SmolVLA's action
expert plus temporal modules. Pinned SmolVLA keeps its own small DDP-safety
exclusions (its output head, final norm, and selected final VLM layers), so
the unfreezing is not bit-for-bit identical. LTE also requires image
augmentation disabled because its temporal embedding cache must remain
frame-consistent; the full-finetune recipe refreshes that detached cache every
1,000 optimizer updates.

### Joint WaitAtGoal + LiftQA, raw GPU-frame queries

`config/lte/waitatgoal_liftqa_lte_smolvla_2gpu_15k_gpu_queries.env` is a
separate 15,000-update LTE-only profile for the temporal benchmark datasets
the local `waitatgoal_v4` and `liftqa_v2` LeRobot exports. It launches two DDP
processes, fully fine-tunes SmolVLA as in the SimpleMemVLA-aligned recipe, and
uses batch size 8 per GPU.

It intentionally does not allocate an embedding cache. Each rank preloads the
raw uint8 frames for the one temporal camera onto its GPU, selects 16 history
queries for each example, then passes the resulting 128 images through the
vision encoder in one call. The LTE training assertion fails if that call is
not exactly `batch_size × num_history_queries`. W&B records
`temporal_query_vision_seconds`, `temporal_query_total_step_seconds`, and
`temporal_query_vision_pct_of_step` (the query-path time as a percentage of a
synchronised full optimizer step).

WaitAtGoal's 2-D state/action representation is right-padded to the shared
3-D LiftQA schema. Its padded action dimension is masked from the diffusion
loss, so it does not train an artificial zero-valued action.

```bash
bash scripts/lte/train_waitatgoal_liftqa_local.sh
```

### Joint WaitAtGoal + LiftQA, raw GPU-frame cache

`config/lte/waitatgoal_liftqa_lte_smolvla_2gpu_20k_gpu_cache.env` is the
otherwise matched 20,000-update, effective-global-batch-32 comparison to the
uncached recipe. Each DDP rank decodes the full selected dataset once into its
own local-GPU uint8 cache, then gathers all 256 temporal query frames directly
on GPU before the single vision-encoder invocation. It does not use a detached
embedding cache, so vision-tower gradients still flow on every training update.

The cache builder temporarily uses 32 CPU conversion threads on each rank.
`torchrun` ordinarily forces one OMP thread, which made image conversion
several minutes long; the setting is restored before optimisation begins.
For Parquet-backed datasets it also reads image columns in large batches,
avoiding per-frame decoding of unrelated state and action fields.

W&B logs `temporal_image_cache_build_seconds` (the one-time local-rank build
time) as well as the query, encoder, temporal encoder/decoder/loss, and action
timings. The usual distributed metric reduction means the recorded build value
is the mean across the two ranks.

```bash
bash scripts/lte/train_waitatgoal_liftqa_gpu_cache_local.sh
```

### TODO: lazy, bounded temporal embedding cache

For uncached temporal-query training, add a **lazy per-rank embedding cache**
keyed by dataset frame index. On a temporal query, look up every requested
frame before decoding an image. Encode only cache misses, then store their
detached visual embeddings with the optimizer-update number at which they were
computed. At the start of each update, discard entries whose embedding was
computed more than a configurable x optimizer updates ago; a cache hit does
not extend that lifetime. This deliberately bounds the staleness introduced
while the vision encoder is being fine-tuned.

The temporal path should then concatenate cache hits and newly encoded misses
back into the original query order before the temporal encoder. Instrument
W&B with hit rate, misses encoded, evictions, and the same encoder/decoder/loss
timings used by the uncached recipe.

`launch_brev.sh` uploads the credential-only `.env`, the selected configuration, and the current uncommitted `~/Repos/XPolicyLab/policy/LTE_SmolVLA` source as a small overlay. The remote checkout is created automatically at `~/RMBench-lte`; credentials remain at `scripts/lte/.env` with mode `600`.

Bootstrap continues detached in `~/.lte-bootstrap/setup.log`. Training runs in a named `screen` session; attach with `screen -r <session-name>` and detach with `Ctrl-a d`.

The bootstrap explicitly clones `https://github.com/alek5k/RMBench.git` at `main`, then initializes and checks out `https://github.com/alek5k/XPolicyLab.git` at `main`. The fork and dependency pins are implementation defaults; the small versioned config contains only the task and training controls.

The remote setup creates `~/miniforge3`, an isolated `smolvla` environment, then runs XPolicyLab's own `policy/SmolVLA/install.sh`. LTE uses the same environment and pinned base source. It intentionally does not upload the local `policy/SmolVLA/smovla` checkout.

## Simulator evaluation

Training and simulator evaluation intentionally use different Conda environments. The policy server stays in `smolvla`; the RMBench simulator client uses `rmbench-eval`, pinned to Python 3.10, PyTorch CUDA 12.1, CUDA toolkit 12.1.1, CuRobo v0.7.8, Warp 1.12.0, and setuptools 69.5.1. This keeps Sapien/CuRobo build dependencies out of training.

On a Brev instance with the remote checkout, prepare the simulator once:

```bash
bash scripts/lte/setup_eval_remote.sh ~/RMBench-lte
```

Then launch each standard 100-episode `battery_try` rollout in `screen`:

```bash
bash scripts/lte/run_eval.sh ~/RMBench-lte smolvla
bash scripts/lte/run_eval.sh ~/RMBench-lte lte
```

For a complete training/evaluation smoke test, pass an explicit step checkpoint
and a two-episode budget. This prevents the adapter from falling back to the
latest full run:

```bash
bash scripts/lte/run_eval.sh ~/RMBench-lte smolvla /absolute/path/to/checkpoints/000005/pretrained_model 2
```

The scripts use `smolvla` for the policy server and `rmbench-eval` for the simulator client. Monitor with `screen -ls` and `tail -f ~/RMBench-lte/logs/*eval*.log`.

When `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` are present in `scripts/lte/.env`, each launch also starts a separate W&B evaluation run. A lightweight sidecar tails the evaluator log and records every episode's success, cumulative success count, and success rate, then writes the final rate to the W&B summary. It does not change the simulator or policy process.

## Monitoring

```bash
brev exec entire-aquamarine-penguin 'tail -f ~/RMBench-lte/logs/*.log'
brev exec entire-aquamarine-penguin 'screen -ls'
```

`TRAIN_STEPS` is the shared count of optimizer updates for both models; it is not wall-clock time or epochs. `TRAIN_SAVE_FREQ` and `TRAIN_LOG_FREQ` control checkpoint and W&B cadence. The default batch sizes are 32 for baseline and 8 for LTE. The base and LTE output directories are respectively `XPolicyLab/policy/SmolVLA/checkpoints/` and `XPolicyLab/policy/LTE_SmolVLA/checkpoints/` beneath the remote workspace.

For a W&B connectivity and configuration smoke test, use the separate five-update configuration. Its `RMBENCH_RUN_TAG` puts checkpoints under `checkpoints/wandb-smoke-5/`, separate from real training:

```bash
bash scripts/lte/launch_brev.sh entire-aquamarine-penguin smolvla config/lte/battery_try_wandb_smoke.env
```
