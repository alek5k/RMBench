# TEMPO policies on Brev

## Opt-in 16-query loss-balanced Cover Blocks test and final shutdown barrier

After the 32/64-query study (including both final evals),
`queue_cover_blocks_balanced_history.yaml` schedules one more fresh
100k/50-predict/50-act run on `calm-crimson-urial`, then 100-episode final eval.
It restores 16 time-lag reconstruction queries and changes only loss-value
balancing; the four memory-read queries and all other reference settings stay
fixed. W&B ID `b016a9c4`, tags `reconstruction-effectiveness`,
`history-queries-16`, `history-loss-balanced`.

Only that training recipe enables `history_loss_balance`: a 100-update rolling
mean action/raw-reconstruction ratio, detached and bounded to [0.01,1000]. The
ratio uses available batches until the window fills, with denominator floor
1e-8. It targets equal numerical contributions, not equal gradient norms.
All existing/default recipes retain fixed weighting. W&B logs the actual
coefficient and raw/weighted contributions; full window state is saved in each
checkpoint. Full-batch train/reload and two-episode simulator smoke tests gate
the 100k run. State: `/data/RMBench-state/cover-blocks-loss-balance-261110.json`.
Screen/log basename: `cover-blocks-loss-balance-261110`.

`stop_l40s_after_queues.py` uses `stop_l40s_after_all_queues.yaml` to supervise
both L40S hosts locally. It waits for every specified train/eval queue on BOTH
hosts (nine exact final-checkpoint evals including this new test), validates
result/checkpoint IDs, counts and files, and requires two consecutive GPU/job
idle checks. Only then does it issue individual `brev stop` commands for
`normal-scarlet-vole` (`pk47azbb5`) and `calm-crimson-urial` (`zcb67bfcs`),
verifying STOPPED. It never deletes instances or files, stops a failed/pending
queue, or touches CPU instances. Add any newly requested future runs to this
barrier before letting it shut down the hosts.

## Cover Blocks time-lag query sweep (32 and 64)

`queue_cover_blocks_queries32_queries64.yaml` runs on `calm-crimson-urial`
only after both WaitAtGoal reconstruction-target trains and their exact final
100-episode evaluations complete. Then it prepares the full official Cover
Blocks data and runs 32-query train -> eval -> 64-query train -> eval.
Both fresh runs use 100k updates, checkpoints every 10k, batch 8, seed 0,
50-predict/50-act and the original native 1e-4 -> 2.5e-6 schedule (1000 warmup,
floor reached at 30k and held). The control is `8d2f6a10` at update 100k;
its 200k total duration does not change the shared first-100k LR schedule.

Only `memory.history_queries` changes from 16 to 32/64. These are the
decoder's sampled **time-lag reconstruction queries**, not memory-read queries:
`num_tokens: 4`, width 64, two GLA layers, history loss weight 1, spatial 2x2,
camera1 temporal input, three native cameras, frozen vision/LLM and joint
action/history loss remain unchanged. Targets remain time-specific history,
not the WaitAtGoal mean/current ablation targets. Query MSE is averaged, not
summed, so doubling queries does not double the configured loss weight.

Configs: `cover_blocks_tempo_predict50_act50_100k_queries32_seed0.yaml`
(`31c8ad72`) and `cover_blocks_tempo_predict50_act50_100k_queries64_seed0.yaml`
(`64b2e9f1`). Each final eval uses 100 episodes/four workers and its exact
100k checkpoint. Training/eval tags include `reconstruction-effectiveness`
and `history-queries-32` or `history-queries-64`; checkpoint uploads stay off.

`run_cover_query_study_remote.sh <release-root> <queue.yaml>` starts the
guarded queue in an existing pinned environment. Before production training,
both modes must pass full-batch train/save/reload/inference and two-episode
simulator smoke tests. The queue verifies reference weights, counts, saved
normalization, unchanged pinned benchmark/conversion source, full official
dataset receipt and disk headroom. No current jobs are interrupted or deleted.
State: `/data/RMBench-state/cover-blocks-query-study-261110.json`.
Screen: `cover-blocks-query-study-261110`.
Log: `/data/RMBench-state/logs/cover-blocks-query-study-261110.log`.

## WaitAtGoal reconstruction-target controls (opt-in)

`queue_waitatgoal_reconstruction_baselines.yaml` schedules two fresh seed-1
10k runs, each followed by a 100-episode eval: `history_mean` then
`current_frame`. Both match reference `5547de5e` (98/100 eval `jvpaxv80`):
batch 8, predict/act 10/10, native mild LR 1e-4 -> 7.5e-5 with 333 effective
warmup updates, all 203 episodes/14008 frames, frozen vision/LLM, expert+GLA
trainable, joint action/reconstruction gradients and reconstruction weight 1.
No holdout or checkpoint artifact uploads. The mean is computed independently
for each example/camera over valid strictly past frames; the current target
shares the existing native vision pass. Both retain 16 lag queries and the
same decoder. These are easier-target training controls, not alternative
history samplers. Normal recipes/checkpoints default to the original `history`
target; neither the original recipes nor global defaults are changed.

Separate fully materialized configs:
`waitatgoal_tempo_history_mean_horizon10_10k_seed1.yaml` (`e2a713c0`) and
`waitatgoal_tempo_current_frame_horizon10_10k_seed1.yaml` (`9d35f6b1`).
Their `history_reconstruction_target` and `history_reconstruction_diagnostics`
settings are opt-in and checkpoint/W&B persisted. The diagnostics compare the
decoder with actual sampled past embeddings and two non-learned predictions
(history mean and current observation) on the same queries. Different-target
training losses alone must not be interpreted as better history reconstruction.

After staging an immutable source release and the hash-verified simulator
sources from `config/tempo/temporal_environment_source.json`, run:

```bash
bash scripts/tempo/run_reconstruction_baselines_remote.sh "$SOURCE_ROOT" \
  "$SOURCE_ROOT/config/tempo/queue_waitatgoal_reconstruction_baselines.yaml"
```

The wrapper downloads/converts the public Hugging Face WaitAtGoal archive,
isolates Zarr tooling and simulator dependencies from the policy environment,
checks all 75 model regressions, and smokes both full-batch train/save/reload/
inference and simulator evaluation before either long run. The queue refuses
config drift, existing outputs, non-joint objectives or wrong eval checkpoints.
Screen: `waitatgoal-reconstruction-baselines-261110` on `calm-crimson-urial`.
State: `/data/RMBench-state/waitatgoal-reconstruction-baselines-261110.json`.
Log: `/data/RMBench-state/logs/waitatgoal-reconstruction-baselines-261110.log`.
No instance shutdown/deletion is scheduled.

Filter these training/evaluation runs with W&B tag `reconstruction-effectiveness`.
The local training recipes include it; evaluations inherit training tags from
their checkpoints. For runs already running/queued from an older immutable
release, `tag_reconstruction_runs.py --watch` appends the tag as each matching
run appears, identified by training/source run IDs, without restarting jobs or
changing their released configs. Existing tags are preserved.

## Swap T 50-predict / 50-act, 100k

`config/tempo/swap_T_tempo_predict50_act50_100k_seed0.yaml` starts fresh on
`calm-crimson-urial` (L40S), matching Cover Blocks `9b4a762e` except task,
task-specific data/cache/identity and execution horizon 20 -> 50. It uses
100k updates, batch 8, seed 0, shuffled full official `swap_T/demo_clean`
demonstrations, native 1000-step warmup and 1e-4 -> 2.5e-6 by 30k, with the
LR floor held afterwards. Vision and LLM are frozen; expert/projections +
GLA are trainable. Current observations use all three cameras; temporal
history uses camera1. No holdout or checkpoint artifact upload is enabled.

The verified `e363a647604d798fa69e` source bundle is reused unchanged.
New setup/controllers and `prepare_swap_T.yaml` are staged separately under
`/data/RMBench-state/queues/swap-T-p50-a50-b871c40f`. Only the Cover Blocks
resolved config and processor JSON settings are copied for comparison, never
its trained weights. Swap T is downloaded from `TianxingChen/RMBench` at
`5d8ff21038fba32ea549e9eee8497aaba9aab5cf`; conversion verifies every raw episode,
encoded video frame count, sample decode and finite numeric statistics.
The runner rejects temporal truncation and checks native 50-to-50 execution,
reference policy/weight/count parity and Swap-specific normalization after a
full-batch three-update save/reload/inference smoke.

Fresh-machine entrypoint: `bootstrap_rmbench_task_remote.sh`, then
`setup_rmbench_task_remote.sh`, then `run_rmbench_task.py`. The bootstrap
recovers an unreachable Ubuntu HTTP archive by verifying the official HTTPS
endpoint and saving the original apt configuration before switching it.

W&B ID: `b871c40f`; group: `261109_tempo_swap_T_predict50_act50_100k_seed0`.
Screen: `tempo-swap-T-p50-a50-100k-b871c40f`.
Log: `/data/RMBench-state/logs/tempo-swap-T-p50-a50-100k-b871c40f.log`.
State: `/data/RMBench-state/tempo-training-b871c40f.json`.
Checkpoints every 10k at
`/data/RMBench-state/checkpoints/swap_T_tempo_predict50_act50_100k_seed0`.
No evaluation or instance shutdown is scheduled; existing runs on other
instances are not interrupted.
Verified preparation: 50 episodes, 17149 frames at 15 Hz, maximum episode
358 frames; conversion/validation took 172.9 seconds. The embedding build
took 95.6 seconds. All 59 model/trainer tests and full-batch three-update
smoke/reference/normalizer checks passed. Trainable parameters: 100221728
(expert 98245840); frozen VLM: 350165184.

The subsequently requested final eval is queued separately via
`queue_swap_T_final_eval.yaml`: 100 episodes, four workers, seed 0, seen
instructions, expert checks and the saved final **100000** checkpoint with
50/50 horizons. W&B uses the same training group/source ID, logs checkpoint
100000, and names the eval `swap_T_tempo_predict50_act50_100k_seed0_eval`.
`queue_final_rmbench_eval.py` waits for verified successful training completion,
the exact complete checkpoint, a validated simulator runtime and an idle GPU.
It never pauses/restarts training or evaluates a partial/intermediate checkpoint.

Screen: `eval-swap-T-after-100k-b871c40f`.
State: `/data/RMBench-state/tempo-swap-T-final-eval-b871c40f.json`.
Log: `/data/RMBench-state/logs/queue-swap-T-final-eval-b871c40f.log`.
Staging: `/data/RMBench-state/queues/swap-T-final-eval-b871c40f`.
The separate simulator environment, CuRobo and assets are copied at identical
absolute prefixes from the proven L40S runtime. Background screen
`prepare-swap-T-eval-runtime` waits for extraction, checks file counts/byte
totals/critical hashes against `swap_T_eval_runtime_manifest.json`, validates
CPU package imports, a short-lived CuRobo CUDA task import and W&B logger config, then writes
`/data/RMBench-state/swap_T-eval-runtime-ready.json`. It installs nothing into
the active policy environment and launches no rollouts during training.
Its status/log are `/data/RMBench-state/swap_T-eval-runtime-setup.json` and
`/data/RMBench-state/logs/prepare-swap-T-eval-runtime.log`.

The initial streamed runtime copy was interrupted. Recovery uses
`repair_eval_runtime_copy.py`, detached in **local screen**
`repair-swap-eval-runtime-b871c40f`: inventory both prefixes, package only
missing/incomplete/mode-or-link-mismatched files, transfer the archive with
`scp -3`, verify SHA-256, extract, then rerun the complete runtime validator.
Controller state/log: `/tmp/repair-swap-eval-runtime-b871c40f.json` and `.log`.
This avoids leaving a long transfer tied to an interactive tool session.
CuRobo constructs a CUDA Pose in a default method argument during task import;
the validator checks that separately with CUDA visible rather than treating
the expected CPU-only import failure as a missing/corrupt runtime.

An independently requested Swap T 30k quick eval uses
`eval_swap_T_tempo_predict50_act50_30k_quick.yaml`: 10 episodes, one worker,
same seed/settings as the final eval. `run_swap_T_quick_eval_remote.sh` validates
the saved checkpoint/source/horizons, waits for the repaired runtime receipt,
and rechecks VRAM before starting alongside training. Screen:
`eval-swap-T-30k-b871c40f`; log:
`/data/RMBench-state/logs/eval-swap-T-30k-b871c40f.log`.
It uses the same training group and source ID, with checkpoint 30000 in W&B.

## Cover Blocks 50-predict / 20-act, 100k

`config/tempo/cover_blocks_tempo_predict50_act20_100k_seed0.yaml` matches
Battery Try reference `455049d4`: seed 0, batch 8, native 1000-step warmup,
1e-4 to 2.5e-6 by 30k, frozen vision/LLM and trainable expert/projections +
GLA. It starts from the pinned pretrained base, not a Battery Try checkpoint.
Official Cover Blocks data (50 demonstrations, 51077 frames) is downloaded
from the pinned Hugging Face revision and converted with the shared helper.
Current VLA input uses three cameras; temporal memory uses camera1 only.

Run ID: `9b4a762e`. Launcher: `run_cover_blocks_predict50_act20_remote.sh`.
It waits for the verified preparation receipt, checks disk budget and matching
reference settings, runs regression tests plus a full-batch three-update smoke,
and verifies Cover Blocks-specific normalization and the actual 50-to-20
execution slice before production. The immutable `e363a647604d798fa69e` training
release is reused; controller updates are staged separately.

Screen: `tempo-cover-p50-a20-100k-9b4a762e` on `normal-scarlet-vole`.
State: `/data/RMBench-state/tempo-training-9b4a762e.json`.
Log: `/data/RMBench-state/logs/tempo-cover-predict50-act20-100k-9b4a762e.log`.
Checkpoints are every 10k on the home disk at
`/home/ubuntu/RMBench-state/checkpoints/cover_blocks_tempo_predict50_act20_100k_seed0`;
older `/data` checkpoints are preserved. W&B receives metrics/config only,
not checkpoint artifacts. No evaluation or instance shutdown is scheduled.
Preparation completed in 487 seconds; the first embedding build took 283.5
seconds. All 59 model/trainer tests and the full-batch save/reload/inference,
reference policy/weight/count, Cover-specific normalizer and execution checks
passed. The staged `run_battery_predict50_act10.py` must be the current
parameterized helper (the old immutable release's helper accepts no horizon
argument). The initial helper-compatibility failure is retained separately as
`tempo-training-9b4a762e.failed-helper-preflight.json`; no production run was
started during that failed attempt.

A separately requested concurrent quick evaluation uses
`eval_cover_blocks_tempo_predict50_act20_30k_quick.yaml`: saved checkpoint
030000, 10 episodes, one worker, seed 0, seen instructions and expert checks.
It retains the trained 50/20 horizons and uses the same policy/source release.
`run_cover_blocks_quick_eval_remote.sh` validates checkpoint identity and at
least 16 GB free VRAM before launching; it does not pause or change training.
W&B uses the training group/source ID and logs checkpoint step 30000, with
`checkpoint_in_name` distinguishing this intermediate quick check.
Screen: `eval-cover-30k-9b4a762e`; launcher log:
`/data/RMBench-state/logs/eval-cover-30k-9b4a762e.log`; outputs:
`/data/RMBench-state/evaluations/cover_blocks_tempo_predict50_act20_30k_quick`.
The matching 70k quick eval is
`eval_cover_blocks_tempo_predict50_act20_70k_quick.yaml`; only checkpoint/output
differ from 30k. Its separate screen/log are `eval-cover-70k-9b4a762e` and
`/data/RMBench-state/logs/eval-cover-70k-9b4a762e.log`. Both retain 10 episodes,
one worker and 50/20 horizons, with checkpoint-specific W&B display names.

## Queued Battery Try 50-predict / 50-act, 500k + eval

`queue_battery_try_tempo_predict50_act50_500k_after_eval.yaml` waits for the
entire `455049d4` 50/20 training-and-eval sequence to finish and validates its
100-episode final result. It then runs fresh pretrained **50/50 TEMPO training
for 500k updates**, followed by 100-episode/four-worker evaluation of checkpoint
500000. Training run ID: `d8fa2f2e`; group:
`261108_tempo_battery_try_predict50_act50_500k_seed0`.

Only total updates, execution horizon and identity/output metadata differ.
Batch 8, seed 0, full Battery Try data, frozen vision/LLM, trainable expert + GLA,
and the pinned source/environment/cache stay unchanged. The native schedule
also stays unchanged: warmup 1000, 1e-4 -> 2.5e-6 at 30k, then floor hold to
500k; this does **not** stretch decay over the longer duration. Checkpoints
remain every 10k, all local, with metrics/config but no artifacts sent to W&B.

Fifty checkpoints plus smoke require about 82 GB. At queue creation, the
projected remaining space after both runs was about 19 GB. The queue rechecks
the full checkpoint budget plus a 10 GiB reserve before training; insufficient
space stops it without deleting anything. Smoke validates a 50-action adapter
output and exact reference configuration/normalizers before production.

Screen: `tempo-battery-p50-a50-500k-d8fa2f2e`.
State: `/data/RMBench-state/tempo-predict50-act50-500k-train-eval-d8fa2f2e.json`.
Training state: `/data/RMBench-state/tempo-training-d8fa2f2e.json`.
Staging: `/data/RMBench-state/queues/tempo-predict50-act50-500k-d8fa2f2e`.
Log: `/data/RMBench-state/logs/tempo-battery-predict50-act50-500k-train-eval-d8fa2f2e.log`.
The generic queue/smoke helpers accept explicit duration and execution horizon
while preserving the existing 100k/10-act and 100k/20-act defaults. Active
releases and the predecessor queue were not edited or interrupted.

## Queued Battery Try 50-predict / 20-act, 100k + eval

`queue_battery_try_tempo_predict50_act20_after_eval.yaml` waits for the current
50/10 final evaluation (`06075719`, W&B eval `5qvdjvgn`) to produce its exact
100-episode `completed.json`, including successful logging. It then waits for
GPU cleanup, runs regression/full-batch save/reload/normalization checks, trains
`battery_try_tempo_predict50_act20_100k_seed0.yaml` fresh from the pinned base,
and evaluates the final 100k checkpoint on 100 Battery Try episodes/four workers.
Order is **current 50/10 eval -> 50/20 train -> 50/20 eval**. Predecessor failure
or incomplete results stop the queue; it never interrupts the active eval.

Training run ID `455049d4`, group `261107_tempo_battery_try_predict50_act20_100k_seed0`.
Only `n_action_steps: 10 -> 20` and identity/output metadata differ from `06075719`.
Prediction remains 50; updates 100k, batch 8, seed 0, native 1k warmup/30k decay
to 2.5e-6, frozen vision/LLM and trainable expert/projections + GLA remain matched.
The exact same `e363a647604d798fa69e` trainer/model release and existing verified
dataset/cache/environments are reused. No package installs, data/cache rebuilds,
checkpoint artifact uploads or lifecycle actions are scheduled. Checkpoints stay
on disk every 10k. The adapter's 50-to-20 execution slice is explicitly checked.

The parameterized smoke/training helper and queue/config files are staged under
`/data/RMBench-state/queues/tempo-predict50-act20-455049d4`; existing releases are
not edited. Screen: `tempo-battery-p50-a20-455049d4`.
Queue state: `/data/RMBench-state/tempo-predict50-act20-train-eval-455049d4.json`.
Training state: `/data/RMBench-state/tempo-training-455049d4.json`.
Log: `/data/RMBench-state/logs/tempo-battery-predict50-act20-train-eval-455049d4.log`.
Final eval shares the new training group/source ID and logs checkpoint 100000.

## Battery Try 50-predict / 10-act, 100k

`config/tempo/battery_try_tempo_predict50_act10_100k_seed0.yaml` starts fresh
from the pinned SmolVLA base, matching the 74% reference `e84cae63` except
`n_action_steps: 10`. Prediction/training targets remain `chunk_size: 50`.
Other settings remain 100k updates, seed 0, batch 8, 1k warmup, native cosine
1e-4 to 2.5e-6 at 30k then held, shuffled full Battery Try data, frozen vision/LLM,
and trainable expert/projections + GLA. All ten checkpoints stay on disk; W&B
receives metrics/config only. The training runner schedules no follow-up or
lifecycle action; the final eval below was separately requested after completion.

The latest local RMBench and standalone XPolicyLab sources were bundled into
immutable release `e363a647604d798fa69e` on `normal-scarlet-vole`; all 8257
deployed files were verified against the source manifest. A first preflight
caught missing `TEMPO_Pi05` imports in the local checkout. Its four required
shared helper files were restored exactly from the saved successful-run source;
the new bundle retains current horizon/validation/naming code, with no changes
to the GLA/decoder mathematics. All 59 model/trainer regression tests pass.

`run_battery_predict50_act10_remote.sh` uses the existing pinned environment,
dataset and frozen embedding cache; it does not reinstall or reconvert data.
The runner checks the real adapter's first-10 slice, then runs a full-batch
three-update save/reload/inference smoke before production. Resolved policy
must match the old reference except execution horizon; pretrained SHA,
parameter counts, frame count and normalization tensors must still match.

W&B training ID: `06075719`. Screen: `tempo-battery-predict50-act10-100k`.
State: `/data/RMBench-state/tempo-training-06075719.json`.
Log: `/data/RMBench-state/logs/battery-try-tempo-predict50-act10-100k-seed0-06075719.log`.
Checkpoints: `/data/RMBench-state/checkpoints/battery_try_tempo_predict50_act10_100k_seed0`.
The failed initial preflight state is preserved separately; no production
training or checkpoint directory was overwritten.

After training completed, the user requested a final-checkpoint evaluation:
`config/tempo/eval_battery_try_tempo_predict50_act10_100k_seed0.yaml`. It runs
100 Battery Try episodes/four workers on the same L40S as the earlier reference,
with seed 0, seen instructions, demo_clean, expert checks, joint actions and
30 Hz. The saved 50/10 horizons remain unchanged. The current training release
is reused, with links to the existing shared simulator assets and the existing
policy/simulator environments; no checkpoint transfer or package install occurs.
W&B shares source training ID `06075719` and its group, logging checkpoint 100000.
The wrapper validates those identities/horizons, and the launcher checks logger
compatibility before allocating the policy/simulator. Config and eval wrapper
are staged in `/data/RMBench-state/queues/tempo-predict50-act10-06075719-eval`.
Screen: `eval-tempo-p50-a10-06075719`. Output:
`/data/RMBench-state/evaluations/battery_try_tempo_predict50_act10_100k_seed0`.
This eval is independent of the cancelled local checkpoint sweep and does not
interrupt the earlier-checkpoint local sanity evaluation.

## Local Battery Try `58ae3446` checkpoint sweep

`bash scripts/tempo/run_local_battery_checkpoint_evals.sh` uses
`config/tempo/eval_battery_try_tempo_horizon10_local_all_checkpoints.yaml` to
smoke-test the final checkpoint on two episodes, then evaluate **every saved
checkpoint, 10k through 100k**, on 100 Battery Try episodes each. It runs
sequentially in screen session `battery-58ae3446-local-evals`, on local GPU 0
only (selected by UUID); GPU 1 and cloud instances are untouched.

The saved `a05b1b9e3db937c5f157` training-source bundle is extracted into
`.deps/eval-source-a05/RMBench`; TEMPO configuration/GLA/modeling file hashes
match the trained source. Existing local assets are linked into this isolated
runtime. Model code is unmodified. Eval child processes remove the inherited
`LD_LIBRARY_PATH` override: the workstation otherwise mixes system cuBLASLt 12.9
with PyTorch's cuBLAS 12.8, causing invalid-value errors in original GLA readout
and BF16 matrix products. Both original operations pass with the matched libraries.
Portable checkpoint views under `.deps/tempo-eval` symlink unchanged
model/normalizer tensors and rewrite only VLM/tokenizer absolute paths, never
the verified backups. A separate `.venv-tempo-rmbench-eval` inherits the installed
RMBench simulator and overlays NumPy 1.26.4/OpenCV 4.10 to match Brev, leaving
the original environment untouched. Policy env stays `.venv-tempo-smolvla`.
SAPIEN's bundled OIDN 2.0.1 cannot denoise on RTX 5090. The simulator subprocess
alone preloads official OIDN 2.3.3 from `.deps/oidn` (Blackwell support), retaining
ray tracing and OIDN rather than switching renderers/denoisers. This library
version differs from Brev and is recorded explicitly in each materialized eval
config. Neither installed SAPIEN nor the policy subprocess is changed. OIDN
errors now fail the smoke/eval instead of silently accepting invalid rendering.

Evaluation settings match the Brev run: horizon 10/10, four workers, env seed 0,
`demo_clean`, seen instructions, expert checks, joint actions, frequency 30.
Results/state/logs are in
`/mnt/wdblack/tempo/evaluations/battery_try_tempo_horizon10_58ae3446_local`.
Each W&B eval shares the training group/source ID and records the checkpoint
step; names end `_checkpoint_<step>_eval`. No model artifacts are uploaded.
`--resume` is allowed only for this failed sweep and reuses verified successful
results; the file lock prevents a second concurrent sweep. Smoke is not logged
to W&B. Model action-sampling RNG remains unseeded, matching the previous eval.

The first 10k evaluation was explicitly skipped by the user after stalling at
99/100 episodes (zero successes). Its raw logs are retained. The separate
`010000/skipped.json` records this incomplete, user-authorized result; it is
never treated as a successful `completed.json` or included in completed results.
The sweep runner validates the source ID/step/episode count of each skip marker.
**The user subsequently cancelled the entire remaining sweep.** The supervisor
and `battery-sweep-after-reference` resume watcher were stopped; do not resume
20k–100k. The reference sanity eval remains active and is not cancelled.
The original W&B eval `kjykgoyu` is explicitly marked
`eval/status=skipped_by_user`, `eval/complete=false`; its partial metrics remain.

### Earlier 74% checkpoint: local sanity check

`bash scripts/tempo/run_local_reference_sanity_eval.sh` downloads the original
`e84cae63` final 100k checkpoint from `normal-scarlet-vole`, including optimizer
and processors, to `/mnt/wdblack/tempo/battery_try_tempo_reference_100k_e84cae63`.
All eight file hashes are checked against the remote checkpoint before inference.
Its original **50/50 prediction/execution horizon** is preserved. Local inference
source, renderer, seed and 100-episode/four-worker recipe match the current sweep.
Results and controller state are in
`/mnt/wdblack/tempo/evaluations/battery_try_tempo_reference_e84cae63_local_sanity`.
W&B links this eval to `e84cae63`, with a `reference-sanity-check` tag.

The screen session `tempo-reference-local-sanity` suspends only the verified
sweep supervisor, not its active eval. After that eval releases GPU 0, the
reference runs next, then the supervisor resumes the remaining checkpoints.
GPU 1 and other jobs are untouched. Exceptions/SIGTERM/Ctrl-C resume the parent
automatically. If the controller is forcibly killed with SIGKILL, inspect
`state.json`'s saved `sweep_supervisor` PID/command/start identity before manually
resuming that exact process with SIGCONT. Do not start a second sweep.

## Final Battery Try / Cover Blocks horizon-10 evaluations

**Cancelled by user:** Cover Blocks never started. Battery Try `58ae3446`
finished 100k; its simulator completed 100 episodes (2 successes), but the old
pinned logger rejected a newer launcher's `--policy-name` argument. The entire
follow-up watcher and waiting Cover Blocks controller were cancelled before
recovering the logging result; do not restart these historical queue recipes.

For pinned releases, evaluation recipes now explicitly select a bundled
`log_eval_wandb.py` plus its matching `tempo_wandb.py` metadata utility. The
launcher checks CLI and metadata with `--check-config` before starting GPU
processes. `--once` can backfill an already-finished simulator log without
re-running episodes. A successful simulator result is saved separately as
`simulator_completed.json` even if the W&B sidecar subsequently fails; the
overall `completed.json` still requires both simulation and logging to succeed.

`queue_battery_cover_horizon10_final_evals.yaml` schedules 100-episode evals of
both final 100k checkpoints on the existing L40S `normal-scarlet-vole`, which
already has the real RMBench simulator/assets. A workstation `screen` watcher
(`queue_cross_host_rmbench_evals.py`) polls training on `binding-lavender-bedbug`,
copies each final checkpoint through the workstation's existing SSH access
(no credential files or optimizer copied), verifies complete file SHA256 manifests,
and launches each eval in a remote screen session. The watcher must remain running
on the workstation; no new cloud instance or shared-env mutation is required.

Order: **Battery Try train -> Battery Try eval -> Cover Blocks train -> Cover
Blocks eval**. The waiting Cover Blocks controller now checks an atomic result
handoff before starting any GPU work. Only a completed 100-episode Battery Try
100k result releases this gate; training failure or cancellation stops the queue.
Eval uses four simulator workers, environment seed 0, seen instructions,
`demo_clean`, expert checks and joint actions, matching the prior Battery Try eval.
W&B names are the training name plus `_eval`, grouped with their source run and
logging source run ID/checkpoint step. Original H100 checkpoints are retained.

## Cover Blocks horizon-10 after Battery Try

`queue_cover_blocks_tempo_horizon10_after_battery.yaml` waits for Battery Try
`58ae3446` to complete successfully and verifies its final 100k checkpoint,
then waits for an idle GPU before starting `cover_blocks_tempo_horizon10_100k_seed0.yaml`
(`5c31d850`). It uses the same horizon-enabled release and **all the same training
and GLA settings**: 100k updates, seed 0, batch 8, horizon 10/10, native 1000-step
warmup, LR 1e-4 -> 2.5e-6 at 30k then held, frozen vision/LLM, trainable action
expert/projections + GLA, and local-only checkpoints every 10k.

Only task/data/cache and run identity change. This is **Cover Blocks, not Blocks
Ranking**: all 50 official demonstrations / 51,077 frames at 15 Hz, three 240x320
cameras and 14D state/actions. Camera 1 supplies temporal history. It starts from
the same pinned pretrained SmolVLA base, never from Battery Try weights. It uses
Cover Blocks normalization statistics and a separate frozen embedding cache,
built only after Battery Try finishes. A full-batch 3-update checkpoint/reload/
inference smoke checks actual policy, trainable counts, base SHA, dataset size,
processor settings and task-specific normalization before production training.

On `binding-lavender-bedbug`, state is
`/data/RMBench-state/cover-blocks-tempo-horizon10-100k-seed0-queue.json` and staging
is `/data/RMBench-state/queues/cover-blocks-tempo-horizon10-100k-seed0-5c31d850`.
The queue runs in its own screen session and stops if Battery Try fails or is
cancelled. The separate final-eval watcher above supplies the evaluation gate;
no instance lifecycle action is queued.

## Battery Try horizon-10 follow-up after LiftQA

`queue_battery_try_tempo_horizon10_after_liftqa.yaml` waits for both LiftQA
horizon-10 train/eval pairs (`895576c1`, `e035a5c9`) to finish, then trains
`battery_try_tempo_horizon10_100k_seed0.yaml` (`58ae3446`) from the original
pinned pretrained base. The reference is the latest completed Battery Try
TEMPO run `e84cae63`, not the earlier episode-grouped/legacy-scheduler run.
It retains **100k updates, seed 0**, batch 8, shuffled frames, eight workers,
native 1e-4 -> 2.5e-6 decay, full 1000-update warmup and floor reached at 30k
(held to 100k). The only learning change is **10/10 prediction/execution**
instead of 50/50. GLA architecture and frozen vision/LLM are unchanged.

The original 50-demo / 33,653-frame, 15 Hz, three-camera 240x320, 14D dataset
and camera-1 frozen cache were copied from `normal-scarlet-vole` and verified
byte-identical (both complete asset-tree checksums). No checkpoint weights or
credentials were transferred. Small saved reference metadata/normalizers are
retained for exact full-batch checkpoint/reload/inference smoke comparison.
The preprocessor comparison permits only the added `observation.image` to
`observation.images.camera1` compatibility alias when that source key is absent
from the dataset. All used camera mappings, tokenizer/normalizer settings and
normalization tensor files must still match the reference. Failed pre-production
attempts are archived with their state and smoke output before a fresh retry;
production always starts from the pinned pretrained base, not smoke weights.
Model configuration, GLA implementation and TEMPO modeling source hashes match
the original Battery Try source. The existing horizon-enabled trainer is reused.

Relative to recent WaitAtGoal/LiftQA recipes: longer run, seed 0, full rather
than duration-scaled warmup, 30k decay endpoint plus floor hold, three current
VLA cameras rather than one, 14D rather than 2D/3D state/actions, and task-specific
data statistics. All use camera 1 for temporal memory, 16 history queries,
four 64D memory tokens/two layers/four heads, batch 8, AdamW .9/.95, clip 10,
and state/action mean-std normalization. Native input resize remains 512x512.

Suite state: `/data/RMBench-state/battery-try-tempo-horizon10-100k-seed0-queue.json`.
Stage: `/data/RMBench-state/queues/battery-try-tempo-horizon10-100k-seed0-58ae3446`.
Checkpoints are local every 10k, W&B metrics/config only. This follow-up is
**training only**; no Battery Try eval or instance lifecycle action is scheduled.

## Queued LiftQA TEMPO horizon-10 20k seeds 1 and 2

`queue_liftqa_tempo_horizon10_20k_seeds1_2.yaml` waits for both original-decay
WaitAtGoal 10k train/eval pairs (`c95a9891`, `aad3737e`) to complete, then runs
**LiftQA TEMPO seed 1 train -> eval -> seed 2 train -> eval**. New run IDs:
`895576c1`, `e035a5c9`; names `liftqa_tempo_horizon10_20k_seed{1,2}`.

The previous LiftQA recipe is retained: fresh pinned pretrained weights,
full 200-episode / 34,465-frame dataset, batch 8, original native decay 1e-4
to 2.5e-6, 666 effective warmup updates, frozen vision/LLM, trainable action
expert + GLA, same memory/history settings and existing LiftQA embedding cache.
Only seed/identity and **10/10 prediction/execution horizons** differ from the
seed-0 reference. Local checkpoints at 10k and 20k, no W&B checkpoint artifacts.
Each final 20k eval runs 100 LiftQA episodes with environment seeds 0..99,
the same instruction and 1000-step cap, logging source ID and checkpoint step.

The existing LiftQA repeat runner now supports a TEMPO-only horizon study and
an explicit predecessor gate. It verifies both exact predecessor IDs and
completed 100-episode result files before any GPU smoke/training. Failed or
cancelled predecessor work stops the follow-up. For each new seed it performs
a full-batch reload/inference smoke, compares actual resolved policy (allowing
only the declared horizons), counts/weights/frames/normalizers, and runs a
one-episode/five-step simulator smoke before production. No active code/env
changes, downloads, package installs or instance lifecycle actions.

Suite state: `/data/RMBench-state/liftqa-tempo-horizon10-20k-seeds1-2-queue.json`.
Generated configs: `/data/RMBench-state/queues/liftqa-tempo-horizon10-20k-seeds1-2/generated`.
Logs: `/data/RMBench-state/logs/liftqa-tempo-horizon10-20k-seeds1-2`.
Source remains `/data/RMBench-releases/tempo-horizon10-1dd44cee/RMBench`.

## TEMPO horizon-10 original-decay 10k runs, seeds 1 and 2

`queue_waitatgoal_tempo_original_decay_horizon10_10k_seeds1_2.yaml` runs fresh
TEMPO **seed 1 train -> eval -> seed 2 train -> eval**, IDs `c95a9891` and
`aad3737e`. Each matches its completed mild-decay 10k reference (`5547de5e`
and `5f291249`), changing only final LR and run identity. Horizon stays 10/10,
batch 8, full data/no holdout, frozen vision/LLM, trainable action expert + GLA,
same pretrained revisions/memory/cache. The explicitly requested original
native schedule is **1e-4 to 2.5e-6**, with **333 effective warmup updates**
and decay finishing at 10k. Nominal preset remains 1000/30000. An opt-in
`waitatgoal_original_decay.yaml` validates this study; global mild defaults
and completed recipes are unchanged.

Each final local 10k checkpoint receives the same 100-episode evaluation,
environment seeds 0..99, instruction and 400-step cap. The per-run gates verify
the corresponding mild-decay run and eval have finished; the suite executes
sequentially and stops on failures. No checkpoint artifacts are sent to W&B.
No policy/trainer/simulator or package changes are needed.

Suite state: `/data/RMBench-state/tempo-original-decay-horizon10-10k-seeds1-2-queue.json`.
Per-run states: `/data/RMBench-state/tempo-original-decay-horizon10-10k-seed{1,2}-{training,eval}.json`.
Logs: `/data/RMBench-state/logs/tempo-original-decay-horizon10-10k-seeds1-2/seed{1,2}.log`.
The preceding mild-decay 10k pair completed at 98% (seed 1), 100% (seed 2).

## TEMPO horizon-10 fresh 10k runs, seeds 1 and 2

`queue_waitatgoal_tempo_horizon10_10k_seeds1_2.yaml` schedules **TEMPO seed 1
train -> eval -> TEMPO seed 2 train -> eval**, run IDs `5547de5e` and `5f291249`.
These are fresh pretrained runs, not evaluations of the earlier 20k runs'
10k checkpoints. Each matches its seed's 20k horizon-10 TEMPO recipe apart
from duration and run identity: batch 8, full data/no holdout, frozen vision/LLM,
trainable action expert + GLA, 10/10 horizons, same memory and frozen cache.
Native mild decay stays 1e-4 to 7.5e-5, with **333 effective warmup updates**
and decay finishing at 10k. Nominal 1000/30000 scheduler settings are unchanged.
Only the final 10k checkpoint is saved locally; no W&B checkpoint artifacts.

The shared sequential queue now accepts an explicit policy variant and duration
while preserving the old native 20k suite's defaults. It delegates to the same
training/smoke and eval runners, explicitly validates the sole duration change,
matches GLA memory settings, and stops on a failed run. No policy/trainer,
package or simulator changes are required. Both final 10k evals use 100
episodes, environment seeds 0..99, the same instruction and 400-step cap.

Suite state: `/data/RMBench-state/tempo-horizon10-10k-seeds1-2-queue.json`.
Per-run states: `/data/RMBench-state/tempo-horizon10-10k-seed{1,2}-{training,eval}.json`.
Logs: `/data/RMBench-state/logs/tempo-horizon10-10k-seeds1-2/seed{1,2}.log`.
Source remains `/data/RMBench-releases/tempo-horizon10-1dd44cee/RMBench`.

## Plain SmolVLA horizon-10 comparisons, seeds 1 and 2

`queue_waitatgoal_smolvla_horizon10_seeds1_2.yaml` schedules **SmolVLA seed 1
train -> eval -> SmolVLA seed 2 train -> eval**, after the TEMPO seed-1 final
100-episode eval completes. Native run IDs: `b9f47197` (seed 1), `da10ab2d`
(seed 2). Each is matched to its actual TEMPO reference (`c7e551e2` and
`1dd44cee` respectively): 20k updates, batch 8, 10/10 prediction/execution,
native mild decay 1e-4 to 7.5e-5, 666 effective warmup updates, same full data,
pretrained revisions, shuffling, normalization and frozen vision/LLM. Only
the action expert/projections train; native SmolVLA has no temporal modules,
history loss or embedding-cache dependency. Saved checkpoint config includes
the actual horizons and schedule. Local checkpoints at 10k/20k, no W&B artifacts.

The suite calls the existing shared training/smoke and eval queue runners
synchronously. It checks matching shared settings before launch, gates on the
TEMPO eval's exact source/step/result, and stops on any failure rather than
releasing the next run. Both final 20k evals use the same 100 episodes with
environment seeds 0..99, instruction and 400-step cap, logging source ID and
checkpoint step to W&B under training name + `_eval`. This does not change
the existing caveat that policy-sampling RNG is not explicitly seeded by eval.

Suite state: `/data/RMBench-state/smolvla-horizon10-20k-seeds1-2-queue.json`.
Per-run states: `/data/RMBench-state/smolvla-horizon10-20k-seed{1,2}-{training,eval}.json`.
Logs: `/data/RMBench-state/logs/smolvla-horizon10-20k-seeds1-2/seed{1,2}.log`.
Same horizon-enabled source release and existing train/eval environments;
no package installation, downloads, checkpoint uploads or instance actions.

## WaitAtGoal ten-action-horizon seed-1 repeat (seed 0 not scheduled)

`waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed1.yaml` (`c7e551e2`)
repeats completed horizon-10 seed-2 run `1dd44cee` from fresh pretrained weights.
Only the training seed and run identity change: 20k updates, batch 8, 10/10
prediction/execution horizons, native mild decay 1e-4 to 7.5e-5 with 666
effective warmup updates, full data/no holdout, frozen vision/LLM and trainable
action expert + GLA. It reuses exactly the same pinned horizon-enabled source
and embedding cache. Local checkpoints at 10k/20k; no W&B checkpoint uploads.

Training state: `/data/RMBench-state/gla-training-c7e551e2.json`.
The final 20k checkpoint is followed by the same 100-episode eval using
environment seeds 0..99, 400-step cap and the same instruction. Eval queue:
`queue_eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed1.yaml`; state:
`/data/RMBench-state/waitatgoal-tempo-horizon10-20k-seed1-eval-queue.json`.
The user withdrew the seed-0 repeat before launch; no seed-0 run is queued.

## WaitAtGoal TEMPO mild decay, ten-action horizon, seed 2

`waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2.yaml` (`1dd44cee`)
replaces the unlaunched seed-2 50k/original-decay request. It starts fresh from
the pinned pretrained base for 20k updates, seed 2, batch 8, mild native decay
1e-4 to 7.5e-5, and 666 effective warmup updates. Both `chunk_size` (prediction)
and `n_action_steps` (execution before replanning) are 10, rather than 50.
This does not change the ten denoising iterations. Other settings match the
full-data/no-holdout `c556dd74` recipe; frozen vision/LLM, trainable action expert
+ GLA, existing frozen embedding cache, and local 10k/20k checkpoints.

The trainer now accepts explicit validated horizon overrides; omitted keys
preserve pretrained defaults. Deployment uses an isolated copy of the pinned
release with only that trainer wiring and the horizon helper added, not the
newer local trainer's unrelated changes. `run_gla_seed_repeat_remote.sh` uses
`--allow-action-horizon-change` to allow exactly seed/horizon/identity changes,
and checks resolved settings, pretrained checksum, parameter counts, data,
processors and ten-action smoke inference before production training.

Training state: `/data/RMBench-state/gla-training-1dd44cee.json`.
The final 20k checkpoint is followed by a 100-episode eval, environment seeds
0..99 with the same 400-step cap/instruction, and W&B metrics/source/step IDs.
Eval queue: `queue_eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2.yaml`.
State: `/data/RMBench-state/waitatgoal-tempo-horizon10-20k-seed2-eval-queue.json`.
No checkpoint uploads or instance lifecycle changes.

## WaitAtGoal TEMPO second 10x LR increase, seed 1

`waitatgoal_tempo_very_high_lr_20k_seed1.yaml` (`4c5862b9`) repeats run
`265f96b5` with peak LR **1e-2** and final LR **1e-3**, fresh from the same
pinned pretrained base. All non-LR training settings remain identical,
including seed 1, 20k updates, batch 8, 666 effective warmup updates, frozen
vision/LLM, trainable action expert + GLA, and local 10k/20k checkpoints.
Global mild-decay defaults remain unchanged; W&B receives no checkpoints.

Training queue: `queue_waitatgoal_tempo_very_high_lr_20k_seed1.yaml`; state:
`/data/RMBench-state/waitatgoal-tempo-very-high-lr-20k-seed1-queue.json`.
After successful training, `queue_eval_waitatgoal_tempo_very_high_lr_20k_seed1.yaml`
evaluates the final 20k checkpoint for 100 episodes, environment seeds 0..99,
400-step cap and the same instruction. Eval name is training name + `_eval`.
Eval state: `/data/RMBench-state/waitatgoal-tempo-very-high-lr-20k-seed1-eval-queue.json`.
The explicit evaluation wrapper reuses the proven simulator environment and
evaluator without the historical release's old queue gate. No package,
policy-code, pretrained-weight or instance lifecycle changes are required.

## WaitAtGoal TEMPO high-LR seed-1 20k experiment

`waitatgoal_tempo_high_lr_20k_seed1.yaml` (`265f96b5`) starts fresh from the
pinned pretrained base, changing only peak LR to **1e-3** and final LR to
**1e-4** relative to seed-1 20k run `c556dd74`. Batch 8, full data/no holdout,
action expert + GLA trainable, frozen vision/LLM, same memory/cache/optimizer
settings. Native warmup remains **666 effective updates**, with cosine decay
ending at 20k. Local checkpoints at 10k and 20k; W&B metrics/config only.
Global mild-decay defaults and historical configs are not modified.

Queue recipe: `queue_waitatgoal_tempo_high_lr_20k_seed1.yaml`; state:
`/data/RMBench-state/waitatgoal-tempo-high-lr-20k-seed1-queue.json`.
Stage/screen/log basename: `waitatgoal-tempo-high-lr-20k-seed1-265f96b5`.
The queue checks the actual completed reference config, explicitly permits only
the two LR changes and new identity, waits for an idle GPU, and performs a
full-batch checkpoint/reload/inference smoke before training. The existing
pinned model/trainer and package environment are unchanged. Training only;
no eval or instance lifecycle action is scheduled by this recipe.

## WaitAtGoal seed-1 50k pair after the LiftQA suite

`queue_waitatgoal_original_decay_50k_seed1_after_liftqa.yaml` appends fresh
SmolVLA (`9fc7a033`) -> final eval -> TEMPO (`e63ed34e`) -> final eval after
all four LiftQA seed-1/2 train/eval pairs finish. It verifies the predecessor's
exact four IDs and completed 100-episode result files, not just an idle GPU.
A failed/cancelled LiftQA suite blocks the follow-up without starting training.

Both WaitAtGoal jobs: seed 1, 50k updates, batch 8, full data/no holdout,
action expert/projections trainable (plus GLA for TEMPO), vision/LLM frozen.
The user explicitly requested the **original**, not default mild, decay:
native scheduler, 1e-4 to **2.5e-6**, **1000** effective warmup updates and
decay ending at **30k**, followed by the minimum LR through 50k. Checkpoints
are local at 10k intervals. Each final `050000` eval uses 100 episodes,
environment seed 0, 400-step cap and the original instruction. W&B names use
`waitatgoal_{smolvla,tempo}_original_decay_50k_seed1` plus `_eval` for evals.

State: `/data/RMBench-state/waitatgoal-original-decay-50k-seed1-queue.json`.
Stage/screen: `waitatgoal-original-decay-50k-seed1` under the usual queue root.
Logs: `/data/RMBench-state/logs/waitatgoal-original-decay-50k-seed1`.
The pinned model/trainer/evaluator and active LiftQA queue are not modified.
Reference configs, normalizers, full-batch reload/inference and simulator smoke
are checked before each production run. No checkpoint artifacts, package
installs, deletion, or instance lifecycle actions.

## LiftQA SmolVLA / TEMPO seeds 1 and 2

`queue_liftqa_20k_seeds1_2.yaml` runs four fresh train/eval pairs on the existing
H100, in this order: SmolVLA seed 1 (`de64d576`), TEMPO seed 1 (`30980ebb`),
SmolVLA seed 2 (`0dac0ddb`), TEMPO seed 2 (`4254be1e`). Each trains 20k updates
and immediately evaluates its final 20k checkpoint for 100 LiftQA episodes
before the next training starts. Eval environment seed is always 0, matching
the original LiftQA comparisons, not the variable training seed.

Native reference: `5f96e30c`; TEMPO reference: `ab7b0d92`. Their actual saved
resolved configs are verified against staged source recipes. Only training
seed and identity/output metadata change: full 200-episode / 34,465-frame
`liftqa_v2`, batch 8, native scheduler with 666 effective warmup updates,
LR 1e-4 to **2.5e-6** (the original LiftQA schedule, not WaitAtGoal's mild decay),
action expert/projections trainable and vision/LLM frozen. TEMPO also trains
the same one-camera GLA/history decoder and reuses the frozen embedding cache.
No holdout; local checkpoints at 10k and 20k; W&B metrics/config only.

The queue runs regression tests once, then full-batch training/reload/inference
smoke plus a one-episode/five-step simulator smoke for every run. Resolved
policy, parameter counts, pretrained checksum, dataset frames and saved
normalizers must match the reference before production. Each full eval result
is checked for exact source ID, final checkpoint, all 100 episodes and seed 0.
Any failure stops the suite. No package installs, deletes, or instance lifecycle
actions. Existing pinned model/trainer/simulator code is unchanged.

Run names: `liftqa_{smolvla,tempo}_20k_seed{1,2}`, with `_eval` appended for
evaluations. State: `/data/RMBench-state/liftqa-20k-seeds1-2-queue.json`.
Staging: `/data/RMBench-state/queues/liftqa-20k-seeds1-2`; ordinary materialized
train/eval configs are in its `generated` directory. Per-run logs are below
`/data/RMBench-state/logs/liftqa-20k-seeds1-2`. Screen: `liftqa-20k-seeds1-2`.

## WaitAtGoal TEMPO 20k seed-1 replacement and final eval

The user cancelled seed-0 run `47102a1e` and replaced it with seed 1:
`waitatgoal_tempo_mild_lr_decay_20k_seed1.yaml` (`c556dd74`). Fresh pretrained
initialization; same full data, batch 8, native mild decay/warmup (666 effective
updates), frozen vision/LLM, action expert + GLA trainable, local checkpoints
at 10k and 20k. No holdout or checkpoint artifacts uploaded to W&B.

`eval_waitatgoal_tempo_mild_lr_decay_20k_seed1_100ep.yaml` evaluates `020000`
after successful training: 100 WaitAtGoal episodes, the same environment base
seed **0** used in all prior comparisons, same instruction and 400-step cap.
Eval W&B name is `waitatgoal_tempo_mild_lr_decay_20k_seed1_eval`; source run
ID and group match training. The watcher rejects failed/cancelled training or
missing/wrong checkpoints. No instance lifecycle action.

Train state: `/data/RMBench-state/gla-training-c556dd74.json`.
Eval state: `/data/RMBench-state/waitatgoal-tempo-20k-seed1-eval-queue.json`.
Staging/screen/log basename: `waitatgoal-tempo-20k-seed1-c556dd74`, with `-eval`
for the evaluation screen/log. Existing pinned model/trainer remains unchanged.
The reference guard compares seed/identity-only differences against the actual
cancelled seed-0 resolved config, including processor hashes from its completed
smoke checkpoint when no final checkpoint exists.

### Cancelled seed-0 20k launch (do not restart)

`waitatgoal_tempo_mild_lr_decay_20k_seed0.yaml` (`47102a1e`) starts fresh from
the same pretrained base after reproduction `81f33025` **and its final
100-episode eval** complete successfully. It does not resume the 10k checkpoint.
Same full dataset, seed 0, batch 8, action expert + GLA trainable, frozen
vision/LLM, and native mild-decay preset. Increasing the duration rescales
effective warmup from 333 to **666** updates and ends decay at 20k.
Local checkpoints remain every 10k (10k and 20k); W&B metrics/config only.
No additional 20k eval or instance lifecycle action is scheduled.

Recipe: `queue_waitatgoal_tempo_20k_after_seed0_repro_eval.yaml`.
State: `/data/RMBench-state/waitatgoal-tempo-20k-seed0-queue.json`.
Stage/screen/log basename: `waitatgoal-tempo-20k-seed0-47102a1e`.
Pinned policy/trainer source is unchanged. The queue checks actual predecessor
config, duration-only differences, shared defaults and final eval identity,
waits for an idle GPU, then performs a full-batch smoke before training.

## Seed-0 reproduction of the 96% mild-decay result

`waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml` (`81f33025`) reproduces
`c74d109e`, whose eval `9085vnyb` scored 96/100. It retains the reference's
**native scheduler, 333 effective warmup updates and mild decay 1e-4 to 7.5e-5**.
The initial 97%-reference draft was corrected before any remote launch; no
constant-LR/no-warmup reproduction was started.
Seed 0, 10k updates, batch 8, full 203-episode dataset, frozen vision/LLM and
trainable action expert + GLA match the reference. Only identity/output
metadata changes. Pinned model/trainer release remains `a05b1b9e3db937c5f157`.
The completed reproduction's final 100-episode eval scored **97/100** versus
the reference's 96/100 (recorded result under
`/data/RMBench-state/eval/tempo_waitatgoal_mild_lr_decay_10k_seed0_repro_100ep/results.json`).

The existing repeat supervisor's `--exact-reproduction` mode checks the actual
reference `resolved_config.json`, requires the same seed, and verifies smoke
policy config, parameter counts, pretrained checksum, frame count and saved
normalizer/processor files against the reference before production starts.
Source/eval sampling behavior is otherwise unchanged: matching environment
seeds do not imply a fixed policy-noise seed in the historical evaluator.

After final checkpoint `010000`, `queue_eval_waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml`
runs 100 episodes with the reference environment seed 0, instruction and
400-step cap. W&B eval name is the training name plus `_eval`. No checkpoint
artifacts or instance lifecycle action. Train state:
`/data/RMBench-state/gla-training-81f33025.json`; eval state:
`/data/RMBench-state/waitatgoal-tempo-mild-lr-decay-10k-seed0-repro-eval-queue.json`.
Staging/screen/log basename is `waitatgoal-tempo-mild-lr-decay-10k-seed0-repro-81f33025`.

## W&B display-name defaults

New GLA training/eval display names use `tempo`, replacing `tempo_smolvla_gla`:
for example, `waitatgoal_tempo_mild_lr_decay_10k_seed1` and
`waitatgoal_tempo_mild_lr_decay_10k_seed1_eval`. Constant-LR studies include
`constant_lr`; the mild recipe (floor = 0.75 * peak) includes `mild_lr_decay`.
Labels come from actual LR settings and preserve seed/holdout/duration suffixes.
Evaluators derive `<training name>_eval`
from checkpoint metadata when available; otherwise they shorten the configured
fallback name. This affects display names only: adapter names, tags, group/run
IDs, checkpoint paths and saved recipe metadata keep their original values.
Historical YAML recipes are preserved. Active pinned training code is not
hot-patched; its existing W&B display name is renamed through the API instead.

## WaitAtGoal GLA 10k seed-1 repeat (no holdout)

The user clarified that this repeat should be **GLA**, not plain SmolVLA.
`waitatgoal_smolvla_gla_mild_decay_warmup_10k_seed1.yaml` (`2139a773`)
repeats the 96%-success GLA run `c74d109e`, changing only training seed (1)
and run identity/output metadata. Full dataset, batch 8, action expert + GLA
trainable, vision/LLM frozen; native 333-update warmup and mild decay.
No holdout, checkpoint artifact upload, or instance lifecycle action.
After training completed, the user requested a final-checkpoint 100-episode
eval: `eval_waitatgoal_tempo_10k_seed1_100ep.yaml`, with environment base seed 0
and W&B name `waitatgoal_tempo_mild_lr_decay_10k_seed1_eval`. Queue recipe:
`queue_eval_waitatgoal_tempo_10k_seed1.yaml`; state:
`/data/RMBench-state/waitatgoal-tempo-10k-seed1-eval-queue.json`.
The original pinned trainer is unchanged. `run_gla_seed_repeat_remote.sh`
checks the config against the actual reference `resolved_config.json`, runs
regression tests and a full-batch GPU checkpoint/reload/inference smoke, then
trains. State: `/data/RMBench-state/gla-training-2139a773.json`.
Screen/log basename: `waitatgoal-gla-10k-seed1-2139a773`.
Staging: `/data/RMBench-state/queues/waitatgoal-gla-10k-seed1-2139a773`.

### Stopped mistaken native launch

`config/tempo/waitatgoal_smolvla_mild_decay_warmup_10k_seed1.yaml`
(`ee7d5161`) was stopped immediately after the user's GLA clarification.
It was configured to repeat native seed-0 run `91f4a6b2` with training seed 1.
Only seed and run identity/output metadata differ. It uses all 203 episodes /
14,008 frames, batch 8, frozen vision/LLM, and native warmup (333 effective
updates) followed by mild decay from 1e-4 to 7.5e-5. There is no validation
holdout or scheduled eval. The final full checkpoint is saved locally at 10k;
W&B receives metrics/config only, not checkpoint artifacts.

Source remains the original pinned release
`/data/RMBench-releases/a05b1b9e3db937c5f157/RMBench`; the new config is staged
separately under `/data/RMBench-state/queues/waitatgoal-smolvla-10k-seed1-ee7d5161`.
The existing native runner performs regression tests and a full-batch GPU
checkpoint/reload/inference smoke before production. State is
`/data/RMBench-state/smolvla-training-ee7d5161.json`; screen/log basename is
`waitatgoal-smolvla-10k-seed1-ee7d5161` (logs below `RMBench-state/logs`).
This is a fresh request, not a restart of the cancelled 12k seed-repeat queue.

## WaitAtGoal episode holdout / validation runs

`waitatgoal_smolvla_holdout10_12k.yaml` (`5ca813e7`) and
`waitatgoal_smolvla_gla_holdout10_12k.yaml` (`7bd249f0`) keep the default
12k/batch-8/native-warmup/mild-decay settings and frozen vision/LLM. They add
a deterministic 10% **whole-episode** holdout using split seed 0:
183 train episodes / 12,626 frames, 20 validation episodes / 1,382 frames.
No held-out episode can provide training action targets or GLA histories.
Both action/state normalizers are fitted on training frames only, then shared
with validation and saved in each full checkpoint. The frozen pretrained vision
embedding cache may contain all frames; it has no learned fit on held-out data.

The shared trainer's `validation.py` runs the full holdout every **500 updates**
with no gradients, restoring module modes and Python/NumPy/PyTorch/CUDA RNG
afterward. Validation noise/history queries are fixed per batch; GLA auxiliary
sampling cannot shift the next batch's action-noise draw. A private loader RNG
keeps validation iteration from affecting the training shuffle/noise stream.
Smokes perform two real validation batches after each of three smoke updates;
production validates all 1,382 held-out frames without a batch cap.

W&B logs `validation/action_loss`, `validation/history_loss` (GLA),
`validation/total_loss`, sample/batch counts and wall seconds. Corresponding
`train/action_loss_window`, history/total window means cover the preceding
500 updates, weighted by valid action targets/history queries rather than
unweighted batch means. `validation/best_action_loss` / `best_action_step`
track metrics only; checkpointing remains at the final 12k update. These are
held-out flow-matching/reconstruction losses, **not** closed-loop success rates.
`timing/step_seconds` remains regular optimizer-update time; validation duration
and `timing/step_with_validation_seconds` report its separate overhead.

Split IDs and train-normalizer hashes are logged in W&B and saved as
`validation_split.json` in the run root and checkpoints. The GLA queue compares
its smoke manifest against native's production manifest before training; any
split/statistics discrepancy blocks launch. Tests cover episode/RNG/statistics
separation, weighted aggregation, checkpoint provenance and queue guards.

The validation-only source overlay is
`/data/RMBench-releases/79fc0fcb8c0052ce680a/RMBench`, based on the prior pinned
release, with `VALIDATION_OVERLAY.json` recording the changed trainer files.
All other policy code is verified unchanged. GLA's real GPU smoke passed.
The new sequence is native 12k -> final 100-episode eval -> GLA 12k -> final
100-episode eval. Training/eval use matching W&B groups and source IDs; no
checkpoint artifacts upload and no instance lifecycle action is scheduled.
Native state: `/data/RMBench-state/smolvla-training-5ca813e7.json`.
Other states/logs/screens use `waitatgoal-{smolvla,gla}-holdout10-12k`, with
`-eval` for evaluations; states append `-queue.json` below `/data/RMBench-state`,
logs append `.log` below `/data/RMBench-state/logs`.

## Current SmolVLA scheduler policy

New SmolVLA/TEMPO comparisons use `scheduler: smolvla`, the pinned native
warmup/cosine implementation, not `legacy`. Constant LR means native warmup
followed by a flat LR (`minimum_lr == learning_rate`); do not disable warmup.
The local XPolicyLab trainer's fallback now selects `smolvla`; historical
explicit legacy recipes/results below are retained as provenance, not defaults.
The current Brev comparison explicitly selects native scheduling in both configs
while retaining the original model/trainer release unchanged.

### Cover Blocks 50/50 follow-up suite (joint loss, 2026-11-09)

`queue_cover_blocks_predict50_act50_suite.yaml` waits for the current
`9b4a762e` final 100k/100-episode eval to finish, then runs these fresh pretrained
experiments sequentially, each immediately followed by its final 100-episode,
four-worker eval:

1. `8d2f6a10`: 200k updates, history weight 1, frozen vision/connector/LLM.
2. `5c90b73a`: 100k updates, history weight 5, frozen vision/connector/LLM.
3. `c74e092b`: 100k updates, history weight 1, LLM unfrozen; vision/connector frozen.

All predict/act 50/50, use batch 8, seed 0, full official Cover Blocks data,
unchanged GLA architecture, and **explicit `temporal_objective: joint`**. The
queue rejects reconstruction-only routing. The prior WaitAtGoal reconstruction
ablation does not change any defaults. LR remains the reference 1e-4 with 1k
warmup, native decay to 2.5e-6 at 30k and a floor thereafter (not stretched to 200k).
Each run starts from the same pinned SmolVLA base, not the previous experiment.
10k checkpoint intervals and no W&B checkpoint uploads remain unchanged.

The queue validates all configs up front, rejects identity/parameter drift,
waits for an idle GPU, verifies a conservative disk budget for all three runs,
then smoke-tests all three modes before beginning the long first production
run. Every smoke checks joint routing, full-batch finite gradients, strict
checkpoint reload, 50x14 action sampling, real trainability counts and identical
normalization. Tests verify LLM gradients while the vision encoder and connector
stay frozen; the same immutable frozen visual cache remains valid.

State: `/data/RMBench-state/cover-blocks-predict50-act50-suite-261109.json`.
Screen: `cover-blocks-50-50-suite-261109`. No lifecycle action on this instance.
The user-authorized cleanup deleted 98 old intermediate checkpoints (~147 GiB),
preserving every affected run's latest/final and all smoke checkpoints; its
audit manifest is `/data/RMBench-state/checkpoint-cleanup-plan-20261109.json`.

Swap T's existing final-100k/100-episode eval queue remains in place. Local
`stop_brev_after_eval.py` monitors it with `stop_swap_T_after_final_eval.yaml`;
only after a complete, matching result and idle GPU does it issue
`brev stop calm-crimson-urial` (ID `zcb67bfcs`), then confirm STOPPED. It never
deletes the instance/checkpoints or stops on a failed/incomplete evaluation.
Local state: `swap_T_stop_after_eval_20261109.json` (ignored runtime state).
Screen: `stop-swap-T-after-eval-b871c40f` on the local workstation.

### Cover Blocks final 100k evaluation

`eval_cover_blocks_tempo_predict50_act20_100k_seed0.yaml` evaluates the completed
`9b4a762e` run's full `100000` checkpoint, preserving its original immutable
source `e363a647604d798fa69e` and 50-predict/20-act execution. It runs 100 episodes
with four workers on `normal-scarlet-vole`, matching the 30k/70k quick checks'
seed 0, seen instructions, demo_clean, expert solvability checks and 30 Hz.
The two earlier quick evaluations each finished with 0/10 successes.

The launcher checks successful training, exact final checkpoint/source/horizon,
pinned simulator versions, the Cover Blocks CUDA import and W&B metadata before
using the shared final-evaluation controller. It makes no package/model changes,
waits for an idle GPU, and rejects an incomplete or mismatched result. New output
uses the home disk rather than the nearly full `/data` volume:
`/home/ubuntu/RMBench-state/evaluations/cover_blocks_tempo_predict50_act20_100k_seed0/`.
State: `/data/RMBench-state/tempo-cover-blocks-final-eval-9b4a762e.json`.
Stage: `/data/RMBench-state/queues/eval-cover-blocks-100k-9b4a762e`.
Screen: `eval-cover-100k-9b4a762e`. W&B name is the training name plus `_eval`,
with the training group, source run ID and checkpoint step 100000 logged.

### Reconstruction-only memory objective ablation (10k, seed 1)

`waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1.yaml` matches the
W&B-verified seed-1 run `5547de5e`, whose eval `jvpaxv80` achieved 98/100
successes. Its exact config/evaluation provenance is saved in
`config/tempo/references/waitatgoal_5547de5e.json`. Keep the 10/10 horizon,
batch 8, full 203-episode/14,008-frame WaitAtGoal dataset, frozen vision/LLM,
GLA architecture, 16 queries, reconstruction weight 1, and native mild-decay
1e-4 -> 7.5e-5 with 333 effective warmup updates. No holdout or checkpoint uploads.

New run `b6a941ce` starts fresh from the same pinned pretrained base. The new
`training.temporal_objective: reconstruction_only` selects disjoint objective
updates, not a detached forward connection or a duplicate temporal encoder:
memory/history decoder update from reconstruction; action expert/state/action
projections/memory-to-prefix projection update from action loss. Each objective
also clips its own gradients so the other objective cannot alter its scale.
Historical runs/configs remain joint by default. W&B logs the saved setting,
objective ownership, both gradient norms and existing losses/timings.

`run_temporal_objective_trial_remote.sh <release-root> <training-config>`
uses the existing policy environment without changing its packages. Zarr
conversion extras live in a separate target directory and affect only the
dataset-preparation subprocess. The runner checks exact reference/default
agreement, full dataset counts, at least 10 GiB GPU headroom, all unit tests,
then a batch-8 three-update save/reload/action-sampling smoke before production.
It refuses existing output/state paths and saves only the final 10k checkpoint
locally. State: `/data/RMBench-state/tempo-training-b6a941ce.json`.
The selected host is the Cover Blocks L40S, `normal-scarlet-vole`; its existing
jobs are not stopped. New checkpoints use the roomier home disk under
`/home/ubuntu/RMBench-state/checkpoints/waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1/`.
The subsequently requested final evaluation uses
`eval_waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1_100ep.yaml`
and `queue_eval_waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1.yaml`.
It requires successful training and the full final `010000` checkpoint, prepares
a simulator-only overlay without changing the policy environment, runs a
one-episode/five-step unlogged smoke, then evaluates 100 episodes with seeds
0..99, the same instruction and 400-step cap as reference `jvpaxv80`.
W&B name is the training name plus `_eval`, with the same group and explicit
`training_run_id` and `checkpoint_step`. Queue state:
`/data/RMBench-state/waitatgoal-reconstruction-b6a941ce-eval-queue.json`.
No instance lifecycle action is scheduled.

### Default WaitAtGoal recipe and fresh 12k comparison

`config/tempo/waitatgoal_defaults.yaml` is the authoritative default LR recipe:
native `smolvla` scheduler, peak 1e-4, floor 7.5e-5, warmup preset 1000,
decay preset 30000. New WaitAtGoal queues reference it through `defaults_config`
and reject materialized configs that disagree; the trainer still reads the same
plain YAML format for native and TEMPO, with no new model/loader logic.
Historical experiments are not rewritten. Explicit duration/checkpoint changes
are validated separately from LR changes and cannot waive unrelated parameters.

The fresh sequence on the same H100 is native SmolVLA `a62f81c4` (12k), its
100-episode eval, TEMPO GLA `db90e7a3` (12k), then its 100-episode eval.
Recipes: `waitatgoal_smolvla_mild_decay_warmup_12k.yaml` and
`waitatgoal_smolvla_gla_mild_decay_warmup_12k.yaml`. These retain the completed
10k comparison's shared hyperparameters and memory architecture; only duration,
checkpoint interval and run identity change. Both start fresh from pretrained
weights. Native scaling gives **400 effective warmup updates**, then mild decay
to 7.5e-5 at 12k. Schedules are tested for equality at every update.

Screens/states/logs follow the same naming convention below with `12k` instead
of `10k`. Final checkpoints are `012000` under the corresponding
`smolvla_waitatgoal_mild_decay_warmup_12k` and
`tempo_smolvla_waitatgoal_gla_mild_decay_warmup_12k` checkpoint roots.
The native runner validates against completed native `91f4a6b2`; the GLA runner
requires the new native run and its 100-episode eval to succeed before launch.
The overlay is `/data/RMBench-state/queues/waitatgoal-default-12k-a62f81c4`;
model/trainer source remains `a05b1b9e3db937c5f157`. Each trainer gets a
full-batch checkpoint/reload/inference smoke test. No checkpoint artifact upload
or instance lifecycle action is scheduled.

### Cancelled/deleted 12k training-seed repeats

Cancelled at the user's request on 2026-11-05. SmolVLA seed 1 was stopped;
the other three repeats never started. W&B run `e185b4c7` was deleted, and
no other seed-repeat training/eval runs existed on W&B. Generated run configs,
seed-1 checkpoint/smoke outputs, per-run log and local W&B sync files were
moved into recoverable trash at
`/data/RMBench-state/trash/seed-repeats-cancelled-20261105T082014Z` (see its
`manifest.json`). The persistent queue state is `cancelled_by_user`, no repeat
screen remains, and the GPU is idle. Seed-0 runs/results and the instance are
preserved. The original reusable source recipe/implementation below remains
for provenance; it is not an active queue.

`config/tempo/queue_waitatgoal_12k_seed_repeats.yaml` appends four fresh runs
after the current seed-0 TEMPO 12k training **and** its final 100-episode eval
finish successfully. Order: native seed 1 (`e185b4c7`), TEMPO seed 1
(`f2a761d8`), native seed 2 (`b3948ec1`), TEMPO seed 2 (`d4c962a5`). Each
trains for 12k updates and is immediately evaluated for 100 episodes before
the next starts. The **training** seeds vary; every **evaluation** retains
base environment seed 0 (episode seeds 0..99), matching the seed-0 controls.
The existing evaluator's sampling behavior is otherwise unchanged.

`queue_waitatgoal_seed_repeats.py` validates the actual active reference configs
against its staged templates, then materializes ordinary full YAML recipes
under `/data/RMBench-state/queues/waitatgoal-12k-seed-repeats/generated`.
Only training seed and identity/output metadata differ from the seed-0
12k recipes. Schedule/defaults, architecture, data, normalization, optimizer,
pretrained revisions, batch size and final-checkpoint eval settings are retained.
Full-batch checkpoint/reload/inference smoke tests precede each fresh run.
Checkpoint/result run and step identities are verified at each transition;
failed/cancelled predecessors or any failed repeat stop the remainder.

Persistent screen: `waitatgoal-12k-seed-repeats` on `binding-lavender-bedbug`.
State: `/data/RMBench-state/waitatgoal-12k-seed-repeats-queue.json`.
Supervisor log: `/data/RMBench-state/logs/waitatgoal-12k-seed-repeats.log`.
Per-run train/eval logs: `/data/RMBench-state/logs/waitatgoal-12k-seed-repeats/`.
W&B groups/names/output roots have `_seed1`/`_seed2` suffixes with independent
run IDs. Model/trainer release remains `a05b1b9e3db937c5f157`; the active seed-0
jobs and their watcher processes are not changed. W&B receives metrics/config
only, no checkpoint artifacts, and no instance stop/deletion is scheduled.

### Completed 10k comparison

The preceding WaitAtGoal sequence on `binding-lavender-bedbug` was:

1. Native SmolVLA `91f4a6b2`, named `waitatgoal_smolvla_10k`, 10k updates.
2. Its final `010000` checkpoint, 100-episode eval.
3. TEMPO GLA `c74d109e`, named `waitatgoal_tempo_smolvla_gla_10k`, 10k updates.
4. Its final `010000` checkpoint, 100-episode eval.

Recipes: `waitatgoal_smolvla_mild_decay_warmup_10k.yaml` and
`waitatgoal_smolvla_gla_mild_decay_warmup_10k.yaml`. Both use batch 8, seed 0,
the same pretrained weights/data/optimizer and native `warmup_steps: 1000`,
`scheduler_decay_steps: 30000`, peak `1e-4`, floor `7.5e-5`. Native short-run
scaling produces **333 effective warmup updates and decay ending at 10k**.
Both schedules were checked against each other for every update using the
actual pinned native scheduler. Vision/LLM stay frozen; expert/projections
train, with only TEMPO additionally training its unchanged GLA/history modules.
The GLA embedding cache is reused. Each policy starts fresh from pretrained
weights, not from the predecessor's trained checkpoint.

All four persistent screen names share `waitatgoal-`, then `smolvla` or `gla`,
then `-mild-decay-warmup-10k`, with `-eval` on eval screens. States use those
names plus `-queue.json` under `/data/RMBench-state/`; logs use `.log` below
`/data/RMBench-state/logs/`. The plain training queue references old completed
`94501382` only to validate shared settings/checkpoint provenance. The GLA
queue explicitly requires the new native eval's successful completion and
100-episode result identity before smoke-testing/training. Each stage refuses
existing output/state paths and failed/cancelled predecessors. New training
queues require native scheduling and positive warmup; both production runs
are preceded by full-batch three-update checkpoint/reload/inference smoke tests.

The orchestration overlay lives at
`/data/RMBench-state/queues/waitatgoal-native-warmup-comparison-91f4a6b2`;
model/trainer source remains pinned to release `a05b1b9e3db937c5f157`.
Eval settings match earlier controls: seed 0, same instruction and 400-step cap.
W&B logs metrics/config/source checkpoint step/group, not checkpoint artifacts.
No instance shutdown/deletion is scheduled.

Both completed: native SmolVLA evaluated 39/100, TEMPO GLA 96/100; W&B and
saved result/checkpoint identities were verified. These results are separate
from the fresh 12k runs above.

The experiment config lives in `config/tempo/battery_try_action_expert_100k.yaml`: battery_try only, 100k optimizer updates, batch 8, Pi05 vision/projector/LLM frozen, action expert and TEMPO trainable, full local checkpoints every 20k. Checkpoints contain model weights, optimizer state, config and normalization assets. W&B receives metrics/config only; checkpoint files are not uploaded.

The selected instance is `normal-scarlet-vole` (L40S). Its attached 256 GB disk is mounted at `/data`; models, raw/converted datasets, embedding cache, logs and checkpoints use `/data/RMBench-state`. The Python environment is under `~/RMBench-state/envs/tempo-pi05`.

## Setup and launch

Install `uv`, `screen`, `ffmpeg` and standard OpenCV system libraries on the instance, then copy `install_environment.sh`, `requirements.txt`, `overrides.txt` and `download_base.sh` to `~/RMBench-state/bootstrap/tempo/`. Run these in screen:

```bash
bash ~/RMBench-state/bootstrap/tempo/install_environment.sh
bash ~/RMBench-state/bootstrap/tempo/download_base.sh
```

The installer selects PyTorch 2.10 CUDA 12.6 for the instance's driver and CPU JAX 0.5.3 for OpenPI utilities. NumPy, pandas, Arrow, datasets and OpenCV are pinned for compatibility. Rerun is unused by training; a NumPy override follows the vendored OpenPI environment's NumPy <2 policy.

From the local RMBench repo, after `brev refresh` has configured SSH:

```bash
bash scripts/tempo/launch_brev.sh normal-scarlet-vole config/tempo/battery_try_action_expert_100k.yaml
```

This reuses `scripts/lte/build_source_bundle.py` to include the current RMBench source and standalone sibling `XPolicyLab` checkout. It preserves their Git source metadata and ignore rules. Credentials from `scripts/lte/.env` are uploaded separately with mode 600; they are never in the source archive. Training starts in screen `tempo-battery-100k` from an immutable release at `/data/RMBench-releases/<bundle>/RMBench`.

`run_remote.sh` applies the vendored OpenPI Transformers patches, downloads only the battery_try archive from `TianxingChen/RMBench` at revision `5d8ff21038fba32ea549e9eee8497aaba9aab5cf`, converts all 50 demos through XPolicyLab's official v3 converter, and computes the same delta-action normalization used in training. The base is a format-only OpenPI Pi05 conversion from `yyshi0619/pi05-base-pytorch@3e7adadea2068316728d07e22a2bb84361debb1c`, verified against SHA256 `f722b8d13a9c70ee2fe7451568fce1ded46d28d30f0fb7b89b801237514d85b1`.

Before the long run, it builds the frozen vision embedding cache, performs three real training updates, saves/reloads a full checkpoint, and verifies finite action sampling. The production run starts only if that smoke test passes. The smoke outputs have `_smoke` appended to their output directory. Existing production outputs are never overwritten or auto-resumed.

## Inspect

```bash
ssh normal-scarlet-vole 'tail -30 /data/RMBench-state/logs/tempo-battery-100k.log'
ssh normal-scarlet-vole 'screen -ls; nvidia-smi'
```

The optimization replacement uses screen `tempo-battery-100k-optimized` and log
`/data/RMBench-state/logs/tempo-battery-100k-optimized.log`.

Checkpoints: `/data/RMBench-state/checkpoints/tempo_pi05_battery_try_action_expert_100k_optimized/{020000,040000,060000,080000,100000}/`. W&B credentials/entity are read from the credential environment. Cache build seconds, separate action/history losses, trainable/frozen parameter counts and step duration are logged.

Initial run: [battery_try_tempo_pi05_action_expert_100k](https://wandb.ai/uts_robot_lab/RMBench/runs/iu21ne7n), launched 2026-11-04. All 50 demos converted to 33,653 frames. The one-time feature cache took 282.584 seconds. Three real training updates, checkpoint reload and finite action sampling passed before the production run started.

The optimized recipe batches visual projection across time, disables recurrent
activation checkpointing while retaining Pi05 gradient checkpointing, and groups
similar history lengths in random pools of 32 batches. No history is truncated or
subsampled. Across all 33,653 frames, padding falls from 48.9% to 3.3%.
A controlled L40S benchmark on the same 128 examples (16 batch-8 updates after
warmup) measured 4.511 s/update originally, 2.776 s with the memory optimizations,
and 2.104 s with length bucketing: **2.14× faster**, with 14.36 GiB peak allocated
VRAM. Timing includes collation, transfer and the complete optimizer update; video
reads are preloaded outside timing. These are throughput measurements, not a
policy-success result. SIGTERM/SIGINT now save a full checkpoint after the current
optimizer update before stopping.

Optimized run: [52vcbl9w](https://wandb.ai/uts_robot_lab/RMBench/runs/52vcbl9w),
source release `770b2a43d4d8ef631dcd`. It starts from the same pretrained base;
the original run had only about 900 updates and had not saved its first checkpoint.
The replacement passed its own three-update checkpoint/reload/action-sampling
smoke test. Its output directory is separate, retaining the original logs and
W&B history.

## TEMPO SmolVLA

`config/tempo/battery_try_smolvla_action_expert_100k.yaml` selects the separate
`TEMPO_SmolVLA` adapter: batch 8, 100k steps, all battery_try demonstrations,
20k full local checkpoints, frozen vision/connector/LLM and trainable action
expert/state/action projections plus TEMPO. It uses native SmolVLA state/action
normalization and its pretrained architecture/image resolution. The optimizer
settings match the TEMPO Pi05 experiment. It retains one-frame recurrence.

```bash
bash scripts/tempo/launch_smolvla_brev.sh normal-scarlet-vole config/tempo/battery_try_smolvla_action_expert_100k.yaml
```

Requires the prepared official LeRobot v3 dataset at the configured `dataset_root`
(the existing instance already has it). The isolated environment is
`~/RMBench-state/envs/tempo-smolvla`; Pi05's environment is not modified.
`SMOVLA_ROOT=/data/RMBench-state/deps/smovla` points to the same pinned LeRobot
commit used by the local XPolicyLab SmolVLA adapter. The installer enables the
native dependency set (Transformers 4.57.1, unpatched), and model/config assets
are downloaded automatically from the pinned Hugging Face revisions.

The launcher creates an immutable source release, transfers credentials separately,
then runs a short-history real-model preflight, cache extraction, three real smoke updates, checkpoint save/reload and
finite action sampling before the production run. Checkpoints are under
`/data/RMBench-state/checkpoints/tempo_smolvla_battery_try_action_expert_100k/`.
Do not launch again into an existing output directory; auto-resume is not supported.
Screen is `tempo-smolvla-train`. The first manually staged run uses log
`/data/RMBench-state/logs/tempo-smolvla-train.log`; subsequent launcher logs are
timestamped and their path is printed at launch.

W&B records one-time cache-build time, whole-step wall time and CUDA-event timings
for forward/backward/optimizer plus memory replay, SmolVLA forward and reconstruction.
The latter three partition the forward pass, so they must not be added to the
parent forward time. CUDA-event durations include dispatch idle gaps; use wall
time across many updates for completion ETA. No checkpoint artifacts are uploaded.

Active replacement (2026-11-04): [TEMPO SmolVLA epszmw8c](https://wandb.ai/uts_robot_lab/RMBench/runs/epszmw8c),
source `85e790e2f8a3ed2b33e5`, log
`/data/RMBench-state/logs/tempo-smolvla-20261104T040019Z.log`.
The Pi05 run is stopped, with a safe full checkpoint at
`/data/RMBench-state/checkpoints/tempo_pi05_battery_try_action_expert_100k_optimized/000320`.
SmolVLA's cache build took 623.15 seconds. The three-update full-history smoke test
passed, including full checkpoint reload and finite `[50,14]` actions; both plain
and encoded-RGB websocket debug tests passed. These are functional checks, not
simulator success-rate evaluations.

### One-camera TEMPO ablation

```bash
bash scripts/tempo/launch_smolvla_brev.sh normal-scarlet-vole config/tempo/battery_try_smolvla_action_expert_1cam_100k.yaml
```

This selects `observation.images.camera1` (`cam_high`) for TEMPO only; the native
SmolVLA prefix still receives all three cameras. The 100k/batch-8 optimizer,
freezing, history and checkpoint settings are unchanged from the three-camera
TEMPO recipe. The run starts from the same pretrained base rather than resuming
the differently shaped memory model. It uses a separate cache and output path
ending in `tempo_smolvla_battery_try_action_expert_1cam_100k`.

The three-camera run `epszmw8c` was safely stopped for this replacement at update
479; its full local checkpoint is preserved at
`/data/RMBench-state/checkpoints/tempo_smolvla_battery_try_action_expert_100k/000479`.
The launcher gates training on adapter unit tests, real-weight preflight and the
full-history three-update checkpoint/reload/inference smoke test.

One-camera run: [kqo9t9pb](https://wandb.ai/uts_robot_lab/RMBench/runs/kqo9t9pb),
source `f566bb818f1c6ba1ed01`, log
`/data/RMBench-state/logs/tempo-smolvla-20261104T041159Z.log`.
Verified through update 200: cache 188.78 seconds, 14 unit tests and both debug
image paths passed. Matched steps 10–200 measured 0.9644 s/update versus 0.9809
for three cameras (1.68% less time), projecting about 27 hours for 100k updates.
Peak allocated GPU memory at step 200 fell from 5.89 to 4.81 GiB. Cache extraction
is much faster; serial full-history training remains the main runtime cost.

### Shared-prefix replay

```bash
bash scripts/tempo/launch_smolvla_brev.sh normal-scarlet-vole config/tempo/battery_try_smolvla_action_expert_1cam_shared_100k.yaml
```

Four nearby supervised points per episode group share one full-prefix autograd
graph. The 32-frame grouping window is **not** a replay limit: every preceding
frame is processed, without temporal subsampling or truncated gradients. The
batch still contains eight action samples (apart from an epoch's remainder).
All samples are visited once per epoch, but same-episode grouping introduces
more within-batch correlation. No memory states are cached across weight updates.

The one-camera independent run was safely stopped at full checkpoint `000753`.
New source: `709dca6705ca683829d8`; log:
`/data/RMBench-state/logs/tempo-smolvla-20261104T043206Z.log`.
Run: [ql8htdv5](https://wandb.ai/uts_robot_lab/RMBench/runs/ql8htdv5).
The same frozen vision cache is reused, and checkpoints go to
`/data/RMBench-state/checkpoints/tempo_smolvla_battery_try_action_expert_1cam_shared_100k/`.

Before launching, 18 tests, real-model gradient equivalence and a three-update
shared-replay checkpoint/reload/inference smoke test passed. The isolated paired
benchmark (12 identical batches after warmup, forward/backward only) measured
1.0531 seconds independent versus 1.0601 seconds shared: no meaningful speedup.
Losses matched and temporal-gradient relative L2 error was 0.000111. Sharing
reduces logical frame-state work but not the serial loop depth; the independent
implementation already batches histories on GPU. W&B logs the paired benchmark,
replay reuse factor, unique episodes, frame-update counts and normal step timings.

Production steps 120–250 (after all debug processes exited) measured 0.9329 s/update,
projecting about **26 hours remaining**, with the first 20k checkpoint about 5.1
hours away at step 250. This is an early ETA, not proof of a speedup over different
samples in the old run. The matched benchmark above found no meaningful change.
Both plain/encoded RGB server checks passed using the shared-trained smoke
checkpoint. Supported robot/action interface remains `aloha_agilex` / `joint`.

### GLA final evaluation and matched native baseline

The 100k GLA run is `5lhbf4r8`. The foreground queue
`queue_gla_eval_baseline.py` waits for its verified process to exit, requires a
complete numeric `100000` checkpoint, evaluates it, smoke-tests the native
baseline, then starts baseline training. It is persistent in screen
`tempo-gla-eval-baseline` on `normal-scarlet-vole`.

Evaluation recipe: `config/tempo/eval_battery_try_gla_100k.yaml`. It selects 100
battery_try episodes, four workers, demo_clean, seen instructions, seed 0,
30 Hz, and the expert solvability check. Each simulator worker has an isolated
TEMPO memory state. Results log to W&B with `checkpoint_step: 100000` and
`training_run_id: 5lhbf4r8`, plus policy/client text logs and `completed.json`.
An incomplete evaluation or rollout error prevents the baseline from launching.

Baseline recipe: `config/tempo/battery_try_smolvla_matched_action_expert_100k.yaml`.
`policy_variant: smolvla` creates the native `SmolVLAPolicy` with a native
`SmolVLAConfig`; no temporal parameters are created. The shared trainer matches
the GLA run's masked action-loss reduction, data/normalization, base/tokenizer
revisions, three current cameras, frozen vision/LLM, trainable expert and
state/action projections, TF32, seed, episode-aware sampler, batch 8, 100k
updates, AdamW/LR schedule, clipping, workers, and 20k local checkpoint interval.
There is no temporal cache read or history reconstruction loss in the baseline.
The queue validates all shared recipe settings before waiting.

```bash
# Read-only inspection of the persistent queue.
ssh normal-scarlet-vole 'cat /data/RMBench-state/tempo-gla-eval-baseline-queue.json'
ssh normal-scarlet-vole 'tail -30 /data/RMBench-state/logs/tempo-gla-eval-baseline-queue.log'
```

`setup_smolvla_eval.sh` provisions the existing RMBench simulator recipe in a
separate `rmbench-eval` environment, using shared state-backed assets. An
exclusive setup lock lets the background provisioning process and the queue
coordinate. The running trainer uses its original immutable source release.
The baseline saves native full-model checkpoints under
`/data/RMBench-state/checkpoints/smolvla_battery_try_matched_action_expert_100k/`.

### Confirmed SmolVLA-reference GLA run

The native-baseline follow-up above was cancelled before it started. Its old
evaluation continues to completion. The replacement GLA recipe is
`config/tempo/battery_try_smolvla_gla_reference_100k.yaml`, confirmed against the
actual W&B config of `p29v0sde` (`battery_try_smolvla_100k`).

It uses 100k updates, batch 8, eight workers, LR 1e-4, 1k warmup, native SmolVLA
cosine decay to 2.5e-6 at 30k then a floor hold, clipping 10, AdamW betas .9/.95,
epsilon 1e-8, weight decay 1e-10, and checkpoints every 10k updates. Uniform
shuffled current frames replace episode grouping; the native final partial batch
is retained. Full causal histories remain chronological and untruncated. Vision
and LLM stay frozen; expert/projections and the unchanged GLA/decoder are trained.

`run_training_eval.py` performs a full-batch smoke test (or verifies its existing
marker), trains, requires the final checkpoint, then runs
`config/tempo/eval_battery_try_gla_reference_100k.yaml`: 100 battery_try episodes,
four workers, seed 0, seen/demo_clean, expert check, 30 Hz. Both stages share W&B
group `261104_tempo_smolvla_gla_battery_try_reference`; the training run ID is
`e84cae63`. Evaluation logs the checkpoint number and training run ID. Its W&B
step follows episode completion count so out-of-order worker IDs are preserved.

Screen: `tempo-gla-reference-train-eval`. Status:
`/data/RMBench-state/tempo-gla-reference-train-eval.json`. Log:
`/data/RMBench-state/logs/tempo-gla-reference-100k.log`.
Production source: `4a82afbf257d680da347`; the existing policy environment and
verified frozen embedding cache are reused. Scheduler state is saved alongside
the optimizer/RNG state. Output:
`/data/RMBench-state/checkpoints/tempo_smolvla_battery_try_gla_reference_100k/`.

### W&B tags

TEMPO training and evaluation use `XPolicyLab.utils.tempo_wandb`. Tags describe
the benchmark, stage, adapter/backbone, memory backend (`gla` or `rmt`), task,
robot and fine-tuning mode; evaluations also carry `checkpoint-<step>`.
`action-expert-only` describes which backbone components are fine-tuned: TEMPO
memory/projections/decoder also remain trainable. Evaluation reads checkpoint
JSON only, without loading tensors. Custom `training.wandb_tags`, evaluation
`wandb.tags`, and `WANDB_TAGS` are retained and deduplicated. Native SmolVLA
controls are not tagged as TEMPO or GLA.

Backfill existing TEMPO runs using saved configuration and explicitly linked
training runs, rather than inferring architecture from mutable display names:

```bash
python scripts/tempo/backfill_wandb_tags.py \
  --xpolicylab-root /path/to/XPolicyLab --report /tmp/tempo-tags-preview.json
# Review the preview; add --apply to update tags only.
python scripts/tempo/backfill_wandb_tags.py \
  --xpolicylab-root /path/to/XPolicyLab --apply --report /tmp/tempo-tags-applied.json
```

The script preserves existing tags, verifies each update, and is idempotent.
It avoids `Run.update()` because that method also re-saves other run metadata
and the summary. Names, groups, configs, summaries and histories are not
included in the tags-only mutation.

On 2026-11-05, runs `5lhbf4r8`, `5yrvyg38`, `e84cae63`, and `ay1dxoe7` were
backfilled and re-fetched successfully. Names, states, groups, job types,
configs and summaries were verified identical before/after; a second dry run
proposed zero additions. Audit reports are on `normal-scarlet-vole` under
`/data/RMBench-state/maintenance/tempo-wandb-tags-20261105-1SOgpW/`.

### Native SmolVLA reference run (2026-11-05)

`battery_try_smolvla_reference_100k.yaml` is the native control for the latest
GLA run `e84cae63`, not the earlier grouped/LR-2.5e-5 recipe. All 24 shared
initialization/data/optimizer/scheduler/sampler/checkpoint settings match:
100k updates, batch 8, LR 1e-4, native 1k warmup/30k cosine then 2.5e-6 floor,
clip 10, eight workers, shuffled frames, seed 0, all 50 battery_try demos.
Vision/LLM remain frozen; action expert and native projections are fine-tuned.
There are no temporal modules, auxiliary loss or embedding-cache accesses.

```bash
bash scripts/tempo/launch_smolvla_brev.sh normal-scarlet-vole \
  config/tempo/battery_try_smolvla_reference_100k.yaml \
  config/tempo/eval_battery_try_smolvla_reference_100k.yaml
```

Passing the optional eval manifest runs regression tests, a full-batch
three-update checkpoint/reload/inference smoke test, two smoke simulator
episodes (not logged to W&B), then production training and 100 final-checkpoint
episodes. A failed smoke test prevents production training. The foreground
runner persists under screen; no local timer is required. This configured run
uses W&B ID `bd463459`, group `261105_smolvla_battery_try_reference`, checkpoints
every 10k, and status `/data/RMBench-state/smolvla-train-eval-bd463459.json`.
Do not relaunch this exact run ID/output path once it has started; create a
fresh recipe/ID instead. Existing installations can run
`run_smolvla_remote.sh <release> <train-config> <eval-config>` directly without
reinstalling their verified policy environment.

This run was started on `normal-scarlet-vole` using immutable source release
`f626818935cd92b0df14`. Screen: `smolvla-battery-reference-100k`; log:
`/data/RMBench-state/logs/smolvla-battery-reference-100k-bd463459.log`.
Thirty-six model/training regressions, three logging regressions and two
pipeline-ordering tests passed. The full-batch three-update smoke checkpoint
strictly reloaded and produced finite `[50,14]` actions with no memory state.
Two complete simulator smoke episodes finished without runtime errors before
production training began; their 0/2 result is not a trained-policy score and
was intentionally not logged to W&B. The actual production W&B config was
compared to `e84cae63`: shared optimizer/data/scheduler settings and pretrained
SHA matched, with zero trainable VLM parameters and no temporal config.
Training was verified past update 300 on W&B and 340 in the live log, with
finite losses, about 0.185 s/update and 94% sampled GPU utilization.

### Prepared GLA 50k run: binding-lavender-bedbug

**Superseded before launch:** the requested task is now WaitAtGoal, using the
recipes in the following section. The battery_try 50k recipe was never started.

Recipes `battery_try_smolvla_gla_reference_50k.yaml` and
`eval_battery_try_gla_reference_50k.yaml` are prepared, not launched. They retain
the latest GLA architecture/settings (`e84cae63`) and the shared parameters of
native baseline `bd463459`; only duration and run-specific metadata/paths differ.
Training is 50k updates, batch 8, LR 1e-4 with 1k warmup and the unchanged 30k
decay-to-floor schedule. Checkpoints: 10k, 20k, 30k, 40k, 50k. All 50 battery_try
demonstrations are used, with no validation holdout, matching the reference.
Vision/LLM remain frozen. Action expert/native projections and the complete
GLA memory path (including prefix projection and history decoder) are trainable.
One camera supplies temporal features; all three supply the current VLA input.

After the new instance has the policy/simulator environments, dataset and assets
provisioned, launch the persistent smoke/train/final-eval pipeline with:

```bash
bash scripts/tempo/launch_smolvla_brev.sh binding-lavender-bedbug \
  config/tempo/battery_try_smolvla_gla_reference_50k.yaml \
  config/tempo/eval_battery_try_gla_reference_50k.yaml
```

The evaluation is tied to the exact `050000` checkpoint: 100 episodes, four
workers, seed 0, seen/demo_clean, expert check, 30 Hz. Training/evaluation share
W&B group `261105_tempo_smolvla_gla_battery_try_50k` and source training ID
`136454d4`. The trainer supplies semantic tags automatically. No job has been
started or queued on either instance for this prepared recipe.

### WaitAtGoal GLA 50k on binding-lavender-bedbug

Training recipe: `config/tempo/waitatgoal_smolvla_gla_reference_50k.yaml`.
The H100 is provisioning this run in screen `tempo-waitatgoal-gla-50k`, source
release `bb9c21afef84ba0527bd`. Log:
`/data/RMBench-state/logs/tempo-waitatgoal-gla-50k-c952aa61.log`.
W&B ID `c952aa61`; group `261105_tempo_smolvla_gla_waitatgoal_50k`.

It uses all 203 WaitAtGoal-v4 episodes / 14,008 frames: one RGB image and 2D
state/actions. Only actual state/images enter the policy; goal, timing and reward
annotations are excluded from policy features. Optimizer/scheduler, batch 8,
seed 0 and the GLA mechanism match the latest battery_try reference; duration is
50k and checkpoints are every 10k. Vision/LLM remain frozen. Action expert,
native projections and the full GLA memory/projection/decoder path are trained.

`wandb_log_checkpoints: false` is explicit and enforced: W&B logs metrics/config
only; full checkpoints and optimizer states remain on disk. The training data,
environment and checkpoints use the attached `/ephemeral` disk via task-specific
`/data/RMBench-state` and `/data/RMBench-releases` symlinks.

The supplied `eval_waitatgoal_gla_reference_50k.yaml` has `execution: local`.
After the three-update model smoke test and successful 50k training, the remote
supervisor records `awaiting_local_download` instead of provisioning a simulator
or evaluating remotely. No instance deletion or shutdown is configured. Local
download/evaluation setup follows only after training is verified active.
Local evaluation targets the final `050000` checkpoint for 100 episodes.

Training is verified active on source release `4175645e058e79682a81` (the first
release stopped at a unit-test fixture limitation before starting training).
All 46 regressions passed. Real pretrained-weight smoke training/save/reload/
inference passed with `[50,2]` actions and `[1,4,64]` memory readout; cache build
took 57.22 seconds and production reuses it. W&B run `c952aa61` was verified
advancing with a frozen VLM, state/image-only inputs and zero model artifacts.
Current log: `.../logs/tempo-waitatgoal-gla-50k-c952aa61-verified.log`.

After verifying Brev training, a separate local Python 3.11 environment was
provisioned at `.venv-tempo-smolvla`, with the same policy versions and
PyTorch 2.10 CUDA 12.8 wheels for RTX 5090. Existing environments were untouched.
WaitAtGoal reset/render/step passed; five transfer/memory-queue tests passed.
The temporal evaluator observes every frame, but predicts a new action chunk
only when its queue is empty, with fresh memory on each episode. Local VLM and
tokenizer path overrides preserve the original downloaded checkpoint bytes.

Local screen `tempo-waitatgoal-gla-followup-c952aa61` is armed using
`followup_waitatgoal_gla_reference_50k.yaml`. It waits for the actual remote
`awaiting_local_download` marker, inventories all five complete checkpoints,
downloads them plus VLM metadata and the exact source bundle, verifies all
SHA-256 hashes, then evaluates the 50k checkpoint locally for 100 episodes.
There are no stop/delete or checkpoint-upload operations. The instance remains
running. Interrupted transfers can be retried without deleting local files.
Status: `checkpoints/tempo_smolvla_waitatgoal_gla_reference_50k/followup_status.json`.
Log: `logs/tempo/waitatgoal-gla-followup-c952aa61.log`.
The local full-policy rollout has not yet been exercised with a downloaded
checkpoint; that stage awaits the training-complete marker and verified files.

### Queued native WaitAtGoal SmolVLA control

**Cancelled before launch (2026-11-05):** `f0e1fd9a` never started. The user
then stopped GLA at update 16,078 and requested the 10k native run below instead.

`config/tempo/waitatgoal_smolvla_reference_50k.yaml` is queued on the same H100
after GLA run `c952aa61`. It starts fresh from the identical pinned SmolVLA base;
it does not load GLA-trained weights. All shared training settings match the
active GLA config, including 50k updates, batch 8, seed 0, shuffled sampling,
LR 1e-4, 1k warmup/30k decay to 2.5e-6, grad clip 10 and checkpoints every 10k.
Vision/LLM stay frozen; expert/native projections are fine-tuned. TEMPO memory,
history decoder/loss and embedding-cache accesses are absent. W&B receives only
metrics/config; checkpoints remain on the attached disk. Its future run ID is
`f0e1fd9a`, name `waitatgoal_smolvla_50k`, group `261105_smolvla_waitatgoal_50k`.

Queue recipe: `queue_waitatgoal_gla_then_smolvla_50k.yaml`. The persistent remote
screen is `waitatgoal-smolvla-50k-queue`, running immutable release
`c9869ec17313d82541cb`. State:
`/data/RMBench-state/waitatgoal-gla-then-smolvla-50k-queue.json`; log:
`/data/RMBench-state/logs/waitatgoal-smolvla-50k-queue-f0e1fd9a.log`.
The queue is completion-gated, not ETA-gated: it checks the predecessor's run
identity/success marker and final checkpoint, waits for no GPU compute processes,
then performs a full-batch three-update native checkpoint/reload/inference smoke
test before production training. Four queue regressions passed locally/on Brev,
and shared settings were compared against the active immutable GLA recipe.
No packages or running-job state were changed, and the queue was verified in
`waiting_for_predecessor_training` while GLA continued training.

This baseline queue schedules training only. GLA's existing local download/eval
watcher is independent and does not delay the baseline once the GPU is released.
No instance shutdown/deletion is scheduled.

### Native WaitAtGoal SmolVLA 10k replacement

`config/tempo/waitatgoal_smolvla_reference_10k.yaml` replaces the cancelled
50k baseline queue on `binding-lavender-bedbug`. GLA `c952aa61` was stopped
gracefully with a complete checkpoint at `016078`; its 10k spot evaluation
`mdvkz0kx` completed 10/10 episodes successfully. The obsolete local 50k
download/eval watcher was cancelled. Existing checkpoints and W&B runs remain.

The new native run is `94501382`, named `waitatgoal_smolvla_10k`. It uses
the same shared trainer and action-loss reduction as GLA, but no temporal
modules, history loss, or embedding cache. All other recipe settings match
the cancelled 50k native recipe except duration and run identity/output paths.
The native scheduler automatically rescales its 1k/30k preset to 333 warmup
steps and decay ending at 10k; this is not the first 10k of the 50k schedule.
Vision/LLM remain frozen; action expert/native projections are trainable.

`run_smolvla_train_only_remote.sh <release-root> <training.yaml>` waits for an
idle GPU, runs native-baseline/training-control/metadata regressions, performs
three full-batch smoke updates and strict checkpoint/reload/action inference,
then trains 10k fresh updates. It reuses the existing environment without
installing packages. Final checkpoint is local to the instance at
`/data/RMBench-state/checkpoints/smolvla_waitatgoal_reference_10k/010000`.
W&B receives metrics/config only. No evaluation or instance shutdown is queued.

### Queued WaitAtGoal GLA 10k follow-up

`queue_waitatgoal_smolvla_then_gla_10k.yaml` waits for native baseline
`94501382` to finish successfully, verifies its final 10k checkpoint, then
waits for an idle GPU. It smoke-tests and starts the fresh GLA recipe
`waitatgoal_smolvla_gla_reference_10k.yaml` as run `28d25e4b`
(`waitatgoal_tempo_smolvla_gla_10k`). This is not a continuation of either run.
The previous GLA architecture/settings are retained exactly; only duration
and run-specific identity/output fields differ from `c952aa61`. Shared
hyperparameters and the effective 333-step warmup/10k decay match the native
10k baseline. Vision/LLM remain frozen; expert/projections and GLA are trained.
The compatible frozen embedding cache is reused. Checkpoints stay on Brev;
no checkpoint artifacts, evaluation, download, or shutdown are scheduled.

The queue validates both the prior GLA recipe and the actual predecessor's
immutable config, rejects cancelled/failed predecessors, and requires finite
full-batch checkpoint/reload/inference with the expected memory shape before
training. Seven queue regressions cover these gates. Start under screen with
direct, profile-free shell arguments (no nested login-shell command string):

```bash
screen -L -Logfile /data/RMBench-state/logs/waitatgoal-smolvla-then-gla-10k.log \
  -dmS waitatgoal-smolvla-then-gla-10k /bin/bash --noprofile --norc \
  "$ROOT/scripts/tempo/run_queued_training.sh" "$ROOT" \
  "$ROOT/config/tempo/queue_waitatgoal_smolvla_then_gla_10k.yaml" \
  /data/RMBench-state/waitatgoal-smolvla-then-gla-10k-queue.json
```

`ROOT` is the verified immutable release's RMBench directory. Inspect the
queue state before relaunching; existing state/output directories are refused.

### LiftQA 20k GLA and three final evaluations

`queue_waitatgoal_then_liftqa_gla_and_evals.yaml` extends the chain without
changing either existing WaitAtGoal run. After GLA `28d25e4b` completes, it
downloads/converts the public `lift_qa_v2.zarr.zip` demonstrations from
`erdrsvhr9/temporally_conditioned_diffusion_policy`, validates the 3D
state/action schema, and smoke-tests the new GLA checkpoint and LiftQA rollout.
Fresh run `ab7b0d92` (`liftqa_tempo_smolvla_gla_20k`) trains all LiftQA v2
episodes for 20k updates, saving full local checkpoints at 10k and 20k.
The pretrained base, frozen vision/LLM, trainable expert/GLA, batch 8 and GLA
architecture are unchanged; its native LR preset rescales to 666 warmup steps
and decay ending at 20k. LiftQA has its own dataset/output/embedding cache.

Once training succeeds, the suite evaluates, sequentially, 100 episodes each:

1. WaitAtGoal native SmolVLA `94501382`, final `010000`.
2. WaitAtGoal GLA `28d25e4b`, final `010000`.
3. LiftQA GLA `ab7b0d92`, final `020000`.

Each has an explicit eval manifest in `config/tempo`, the exact training
instruction, seed 0 and native task success criteria. WaitAtGoal caps episodes
at 400 steps; LiftQA at 1000. W&B receives independent evaluation runs with
their source training IDs, checkpoint steps and matching groups. Results stay
under `/data/RMBench-state/eval/`. Checkpoints are never uploaded to W&B.
No instance shutdown/deletion or local download is scheduled by this suite.

`setup_temporal_eval_remote.sh` creates a simulator-only overlay at
`/data/RMBench-state/envs/tempo-temporal-eval`, with read-only access to pinned
policy dependencies. It does not modify the live training environment. MuJoCo
3.5.0 and Robosuite commit `95743f6687ad7394ccb6e865fa4e1b99114e580d`
match LTE-DP's LiftQA environment requirements. Only the simulator source
files are staged under `/data/RMBench-state/deps/temporal-eval/ltedp/envs/`;
LTE-DP training/setup scripts are not used. Environment source hashes should
be recorded alongside that staging directory. LiftQA reset/render/step passed
on the H100 before scheduling; the real trained-policy smoke test is gated
until its own dataset/cache and three-update checkpoint exist.

Run under screen via `run_temporal_suite_remote.sh <release-root> <suite.yaml>
<state.json>`. Its state records successful checkpoints, training stages and
each completed evaluation. A failed/cancelled predecessor never starts
training. An individual final eval failure is recorded without skipping the
other requested evaluations. The queue refuses duplicate state/output paths.

LiftQA can also be prepared ahead of time while WaitAtGoal trains:
`prepare_liftqa_remote.sh <release-root>` disables CUDA, uses two CPU math
threads, downloads/converts LiftQA only, and validates source episode/frame
counts, feature shapes, normalization statistics and first/last action chunks.
It writes `/data/RMBench-state/datasets/liftqa_v2_prepared.json` on success.
The downloaded LiftQA v2 source has 200 episodes / 34,465 frames, 3D actions
and state, RGB images `[3,144,144]`, and a maximum episode length of 224 frames.
The simulator is already provisioned; GPU embedding extraction remains queued.
The converter's per-dataset lock is shared with the later training queue, so
the queue waits for complete finalization instead of reading partial output.
Existing datasets are reused only if frame/episode counts and finalized data
and statistics are present. Incomplete output is refused, not deleted.

The early-preparation deployment uses immutable release
`a05b1b9e3db937c5f157`, screen `liftqa-dataset-prep`, and log
`/data/RMBench-state/logs/liftqa-dataset-prep.log`. Only the waiting LiftQA/eval
queue was replaced to pick up the shared lock; the active WaitAtGoal GLA run
was not changed. Its active follow-up state is now
`/data/RMBench-state/waitatgoal-liftqa-gla-three-evals-prepared-queue.json`
under screen `waitatgoal-liftqa-gla-three-evals-prepared`. The old queue state
is retained with its replacement pointer. Training IDs and all three final
evaluation recipes are unchanged.

Early preparation completed successfully on 2026-11-05: all 200 episodes /
34,465 frames converted and validated, including source counts, finite stats,
first/last decoded 50-action chunks, and the policy's state/image-only inputs.
The receipt reports `sample_decode_passed: true`, `cpu_only: true`, and
`gpu_cache_built: false`. The later queue reuses this finalized dataset.

### LiftQA native baseline and rescheduled four-policy evaluation

The three-eval schedule above is superseded by
`config/tempo/queue_liftqa_smolvla_then_four_evals.yaml`. The existing LiftQA
GLA run `ab7b0d92` finishes unchanged, then fresh native SmolVLA run
`5f96e30c` trains LiftQA for 20k updates. Its recipe is
`config/tempo/liftqa_smolvla_reference_20k.yaml`: same base-weight revisions,
all 200 episodes, batch 8, LR 1e-4, scheduler, optimizer and frozen vision/LLM
as GLA, with only the action expert/projections trainable. There is no GLA,
history loss or embedding cache. Local checkpoints are saved at 10k and 20k;
no checkpoint artifacts are uploaded to W&B.

After both LiftQA trainings, evaluate 100 episodes each in this order:

1. WaitAtGoal SmolVLA `94501382`, checkpoint 10k.
2. WaitAtGoal GLA `28d25e4b`, checkpoint 10k.
3. LiftQA GLA `ab7b0d92`, checkpoint 20k.
4. LiftQA SmolVLA `5f96e30c`, checkpoint 20k.

`run_liftqa_baseline_suite_remote.sh <original-release-root> <suite.yaml>
<state.json>` keeps model/trainer/eval Python code pinned to the original release.
New orchestration/config files are staged in
`/data/RMBench-state/queues/liftqa-smolvla-four-evals-5f96e30c`.
The old eval shell launcher is wrapped with `run_temporal_eval_handoff.sh`;
the original launcher is preserved byte-for-byte in the queue directory.
The barrier holds only evals launched by the identified original supervisor,
leaving that supervisor and its training child running normally. Only after
successful GLA completion and both complete checkpoints does native training
start. Its smoke eval bypasses the barrier. After native training succeeds,
the original three evals are released, followed by the new native LiftQA eval.
Full-batch three-update training/reload and real policy/simulator smoke tests
gate the new baseline. A failed training never releases final evals. A failed
individual eval does not skip the remaining requested evaluations.

Persistent screen: `liftqa-smolvla-four-evals`.
State: `/data/RMBench-state/liftqa-smolvla-four-evals-v2-queue.json`.
Log: `/data/RMBench-state/logs/liftqa-smolvla-four-evals-v2.log`.
The old state remains live for the original three evals; the new state records
the combined four-eval schedule. All evals retain W&B metrics, source run IDs, checkpoint numbers and
matching groups. No instance shutdown/deletion or local download is scheduled.

### Local archive of all comparison checkpoints

`python3 scripts/tempo/archive_temporal_checkpoints.py --config
config/tempo/archive_waitatgoal_liftqa.yaml` copies every complete numeric
checkpoint of runs `94501382`, `28d25e4b`, `ab7b0d92` and `5f96e30c` from
`binding-lavender-bedbug` to `/mnt/wdblack/tempo/<run-name>_<run-id>/<step>`.
This covers both WaitAtGoal 10k checkpoints and both LiftQA 10k/20k checkpoints.
Existing checkpoints transfer immediately; it polls every 60 seconds for
remaining saves and exits once all six expected checkpoints are verified.
Three-update smoke-test checkpoints are not production checkpoints.

Every copy includes full model weights, optimizer state, policy configuration,
processors and normalization statistics. SHA-256 hashes and run/step identity
are checked before an `.incoming` directory becomes the final local checkpoint.
Existing conflicting copies are never overwritten. Training recipes, resolved
configs, remote inventory and per-checkpoint verification receipts are retained.
Shared tokenizer/VLM metadata is dereferenced once into `vlm_assets/<revision>`;
pass that directory as `--vlm-assets` when relocating an evaluation checkpoint.
No checkpoint JSON is rewritten, and no credentials or datasets are copied.

Run the archiver in local screen `tempo-checkpoint-archive`; progress and pending
saves are recorded in `/mnt/wdblack/tempo/archive_status.json`. It only reads
Brev and writes the requested local archive: no eval, training, W&B upload,
remote deletion or instance shutdown is performed.

### Older versus fresh WaitAtGoal 10k checkpoint comparison

`eval_waitatgoal_gla_50k_schedule_10k_100ep.yaml` evaluates the older
`c952aa61/010000` checkpoint over 100 episodes on the same Brev H100 as the
fresh `28d25e4b/010000` evaluation (`4lant4rk`, 79/100). It uses the same pinned
source `a05b1b9e3db937c5f157`, simulator overlay, environment seed 0, 400-step
cap, instruction, action chunks and per-frame GLA memory updates. Policy noise
is not explicitly seeded, matching the previous Brev evaluator behavior.
Metrics go to a separate W&B eval linked to `c952aa61`; previous results stay
unchanged and no model artifacts are uploaded.

The older checkpoint was copied and SHA-256 verified locally at
`/mnt/wdblack/tempo/waitatgoal_tempo_smolvla_gla_50k_c952aa61/010000`.
A local RTX 5090 smoke rollout failed in GLA's existing `torch.einsum` call
with `CUBLAS_STATUS_INVALID_VALUE`. The local comparison was cancelled after
the checkpoint copy finished; model math was not changed to work around it.
The Brev eval uses screen `waitatgoal-old-10k-100ep-eval`, log
`/data/RMBench-state/logs/waitatgoal-old-10k-100ep-eval.log`, and output
`/data/RMBench-state/eval/tempo_waitatgoal_gla_50k_schedule_10k_100ep/`.

Completed comparison: older 10k checkpoint `c952aa61` achieved **99/100**
(`zalq9q5n`), versus **79/100** for fresh 10k run `28d25e4b` (`4lant4rk`).
Both result JSONs were copied and SHA-256 verified under
`/mnt/wdblack/tempo/eval/` using their corresponding remote directory names.
The performance gap persists with 100 episodes; the verified effective LR
schedule difference is a plausible explanation, not an isolated causal test.

### Constant-LR WaitAtGoal GLA 10k trial

`waitatgoal_smolvla_gla_constant_lr_10k.yaml` defines fresh run `666e444e`:
the same 10k GLA architecture, seed, batch 8, dataset, pretrained weights,
frozen vision/LLM and trainable expert/projections/GLA as `28d25e4b`, with
fixed LR **1e-4 at every update**, no warmup and no decay. The pinned trainer
already provides a direct-LR (`scheduler: legacy`) path; setting its peak and
floor both to 1e-4 makes the expression constant without modifying model or
trainer code. The retained `scheduler_decay_steps` field is unused on this path.

`run_constant_lr_trial_remote.sh <original-release-root> <training.yaml>
<state.json>` validates the only intended changes against the deployed 10k
reference and tests the LR expression for every update. It requires 16 GiB of
free GPU memory, runs a full-batch three-update checkpoint/reload smoke test,
then starts fresh production training. It never stops other jobs or modifies
environments. The existing compatible frozen embedding cache is reused.

Model/trainer source remains pinned to `a05b1b9e3db937c5f157`. Screen:
`waitatgoal-gla-constant-lr-10k`. State:
`/data/RMBench-state/waitatgoal-gla-constant-lr-10k-queue.json`. Log:
`/data/RMBench-state/logs/waitatgoal-gla-constant-lr-10k.log`. Full checkpoint:
`/data/RMBench-state/checkpoints/tempo_smolvla_waitatgoal_gla_constant_lr_10k/010000`.
W&B receives training metrics/config only; no checkpoint upload, eval or
instance shutdown is scheduled by this training-only runner.

The later requested final eval is scheduled separately by
`queue_eval_waitatgoal_gla_constant_lr_10k.yaml` and
`queue_temporal_eval_after_training.py`, without restarting/modifying the
active trainer. It waits for successful training completion and verifies the
full `010000` checkpoint's step/run identity before starting 100 WaitAtGoal
episodes with seed 0, 400-step cap and the same instruction/evaluator as the
earlier controls. Eval W&B group/source ID match `666e444e`; only metrics and
config are logged, not checkpoint artifacts. The original training-only
state's `evaluation_scheduled: false` is not modified; this external queue is
the authoritative eval schedule.

Persistent eval screen: `waitatgoal-constant-lr-10k-eval`.
State: `/data/RMBench-state/waitatgoal-gla-constant-lr-10k-eval-queue.json`.
Log: `/data/RMBench-state/logs/waitatgoal-gla-constant-lr-10k-eval.log`.
Results: `/data/RMBench-state/eval/tempo_waitatgoal_gla_constant_lr_10k_100ep/results.json`.
Failed/cancelled training or a missing final checkpoint prevents evaluation;
duplicate state/output paths are refused. No instance lifecycle action occurs.

### Native WaitAtGoal constant-LR 10k control

`waitatgoal_smolvla_constant_lr_10k.yaml` defines fresh regular SmolVLA run
`8ed0d03f`, matched to GLA `666e444e` but without temporal modules, history
loss or embedding cache. All shared settings, including batch 8, seed 0,
the pretrained weight revisions, optimizer and fixed **1e-4** LR (no warmup
or decay), are unchanged. Vision/LLM stay frozen; expert/projections train.

As requested, it starts only after the GLA 100-episode eval completes. That
eval finished **97/100**, and its result/run/checkpoint identity was verified
before launch on the idle H100. The existing pinned
`run_smolvla_train_only_remote.sh` runs regressions, a full-batch three-update
native checkpoint/reload smoke test, then 10k fresh production updates.
There is no new loader or trainer implementation, package installation,
checkpoint artifact upload, instance lifecycle action, or auto-eval schedule.

Source: `a05b1b9e3db937c5f157`. Screen:
`waitatgoal-smolvla-constant-lr-10k`. State:
`/data/RMBench-state/smolvla-training-8ed0d03f.json`. Log:
`/data/RMBench-state/logs/waitatgoal-smolvla-constant-lr-10k.log`. Final checkpoint:
`/data/RMBench-state/checkpoints/smolvla_waitatgoal_constant_lr_10k/010000`.

The later requested native evaluation uses
`queue_eval_waitatgoal_smolvla_constant_lr_10k.yaml` and
`eval_waitatgoal_smolvla_constant_lr_10k_100ep.yaml`: 100 episodes on the
verified final checkpoint, seed 0, the same instruction and 400-step cap.
It runs alongside the mild-decay GLA trainer on the H100 without modifying
training or its queued evaluation. Screen: `waitatgoal-smolvla-constant-lr-10k-eval`.
State: `/data/RMBench-state/waitatgoal-smolvla-constant-lr-10k-eval-queue.json`.
Log: `/data/RMBench-state/logs/waitatgoal-smolvla-constant-lr-10k-eval.log`.
W&B uses the native training group/source ID; no checkpoint artifacts upload.

### Mild-decay WaitAtGoal GLA 10k follow-up

`waitatgoal_smolvla_gla_mild_decay_10k.yaml` defines fresh TEMPO GLA run
`404f54a1`, matching constant-LR GLA `666e444e` except for a cosine LR
schedule from **1e-4 to 7.5e-5 over 10k updates**, with no warmup. Batch 8,
seed 0, frozen vision/LLM, trainable expert/projections/GLA, pretrained
weights, dataset and embedding cache remain unchanged. The existing
`scheduler: legacy` path uses the total training steps for this cosine;
the retained `scheduler_decay_steps` metadata is unused.

`queue_waitatgoal_smolvla_then_gla_mild_decay_10k.yaml` waits for native
`8ed0d03f` to finish successfully, verifies its final checkpoint and an idle
GPU, then smoke-tests and trains the new GLA policy from pretrained weights.
It does not resume from the native policy. The queue permits only the
explicitly declared LR-floor change against the prior GLA recipe; other
shared training and memory settings must match. Model/trainer code remains
pinned to `a05b1b9e3db937c5f157`.

`queue_eval_waitatgoal_gla_mild_decay_10k.yaml` separately waits for the new
GLA training to succeed, then evaluates its final `010000` checkpoint for
100 WaitAtGoal episodes with the existing seed, instruction and 400-step cap.
Both watchers are persistent screen sessions on `binding-lavender-bedbug`:
`waitatgoal-gla-mild-decay-10k` and `waitatgoal-gla-mild-decay-10k-eval`.
Their states are `/data/RMBench-state/waitatgoal-gla-mild-decay-10k-queue.json`
and `/data/RMBench-state/waitatgoal-gla-mild-decay-10k-eval-queue.json`;
logs use the corresponding screen names below `/data/RMBench-state/logs/`.
W&B group: `261105_tempo_smolvla_gla_waitatgoal_mild_decay_10k`.
Training/evaluation log metrics and config only, not checkpoint artifacts.
No instance shutdown or deletion is scheduled.

### Native mild-decay control after TEMPO evaluation

`queue_waitatgoal_gla_eval_then_smolvla_mild_decay_10k.yaml` waits for both
TEMPO `404f54a1` training and its 100-episode final-checkpoint evaluation to
finish successfully before launching native SmolVLA `596ab8f2`. The optional
`predecessor_eval_state` gate in `queue_training_after_run.py` verifies the
evaluation result's run ID, checkpoint path/step and all 100 episodes before
the existing idle-GPU and full-batch checkpoint/reload smoke-test gates.
Failed/cancelled evaluation prevents launch; a low success rate does not.

`waitatgoal_smolvla_mild_decay_10k.yaml` matches the TEMPO run's shared training
settings exactly: 10k updates, batch 8, seed 0, no warmup and cosine LR
1e-4 to 7.5e-5. It starts fresh from the same pretrained weights, without
TEMPO modules/history loss/cache; vision/LLM remain frozen and expert/projections
train. Original model/trainer release `a05b1b9e3db937c5f157` is unchanged.

Screen: `waitatgoal-smolvla-mild-decay-10k` on `binding-lavender-bedbug`.
State: `/data/RMBench-state/waitatgoal-smolvla-mild-decay-10k-queue.json`.
Log: `/data/RMBench-state/logs/waitatgoal-smolvla-mild-decay-10k.log`.
W&B group: `261105_smolvla_waitatgoal_mild_decay_10k`. Checkpoint artifacts
remain disabled. This follow-up schedules native training only, with no
native auto-eval or instance lifecycle action.

### Cover Blocks data preparation only

`prepare_cover_blocks.yaml` selects only official RMBench `cover_blocks`
demonstrations at revision `5d8ff21038fba32ea549e9eee8497aaba9aab5cf`.
`prepare_rmbench_remote.sh <release-root> <preparation.yaml>` downloads raw
data below `/data/RMBench-state/data/demo_clean/cover_blocks/aloha_agilex`
and converts through XPolicyLab's shared RGB/data helpers into
`/data/RMBench-state/datasets/lerobot/local/rmbench-cover_blocks-aloha_agilex-v30`.
It never starts training or changes the existing training/evaluation queues.

The recipe preserves the converter's native FPS from `env_cfg`, 14D
state/actions, three policy cameras at 240x320, AV1 CRF18/preset12/GOP2 and
native normalization statistics. CUDA is disabled, CPU math threads and
per-camera encoding threads are limited to two, and full-episode encoder
queues prevent dropped frames. Validation checks every raw episode, source
frame counts, sample TorchCodec decoding, finite statistics and the actual
encoded frame total for every camera. Existing dataset output is retained
and validated, never overwritten. A successful preparation writes
`/data/RMBench-state/datasets/cover_blocks_prepared.json`.

System FFmpeg is needed by TorchCodec for video datasets; the image-only
temporal tasks do not exercise it. If absent, install the OS FFmpeg package
without reinstalling policy/Torch packages. The preparation runs in screen
`cover-blocks-dataset-prep`, with log
`/data/RMBench-state/logs/cover-blocks-dataset-prep.log`.

Preparation completed on 2026-11-05: all 50 demos / 51,077 frames at 15 Hz,
14D state/actions and three 240x320 cameras. TorchCodec sample decoding,
finite normalization statistics and exact encoded frame counts passed.
CPU conversion/validation took 376.66 seconds after the raw archive download.
The receipt is marked complete. No Cover Blocks training/evaluation is queued.
