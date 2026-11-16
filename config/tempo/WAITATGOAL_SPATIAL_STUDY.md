# WaitAtGoal spatial reconstruction study

Both recipes match the seed-1 10k reference `5547de5e`: batch 8, predict/act
10/10, frozen vision+LLM, trainable action expert+GLA, joint gradients,
16 stratified past-time queries, four width-64 memory read tokens, and
native SmolVLA warmup/mild LR decay 1e-4 to 7.5e-5 (333 effective warmup).
Each ends with 100 evaluation episodes at seed 0, maximum 400 steps, and
the original instruction. Metrics go to W&B; checkpoints remain on disk.

| Recipe | Temporal input per camera/frame | Reconstruction output per time query |
| --- | --- | --- |
| spatial4 | Existing adaptive 2x2 spatial pool: 4x960 | Each of the 4x960 tokens independently |
| native64 | All native 8x8 post-connector tokens: 64x960; no pooling | Each of the 64x960 tokens independently |

The native64 study also changes encoder spatial embeddings and the memory
frame projection's input size. It is not solely a target ablation. Both
decoders query final memory using lag+camera+learned spatial positions; no
target embedding enters the decoder. Target normalization is per patch,
and loss averages over patches as well as channels, retaining weight 1.
The unchanged default still reconstructs the mean visual embedding.

The four-token test reuses the existing pool2 cache; native64 builds its own
cache (~1.6 GiB, versus ~103 MiB for pool2) with the same batch size 64.
Runtime geometry checks require exactly the configured native token count;
there is no silent downsampling fallback. Neither option is a new default.

The original cloud queue `queue_waitatgoal_spatial_reconstruction.yaml` is
retained as provenance. Its waiting process on calm-crimson-urial was stopped
before any smoke or training started, at the user's request to move both jobs
to the local RTX 5090s. The cloud state is marked `moved_to_local` with its
original state backed up. Both unused W&B training identities are retained.

`queue_waitatgoal_spatial_local.yaml` runs spatial4 on local GPU 0 and native64
on GPU 1, independently at batch 8 (not a distributed batch). The `_local`
training/eval recipes change only machine paths; numerical settings, model
revisions, run identities, prompt, seeds and eval protocol are unchanged.
Both full-batch three-update train/checkpoint-reload and one-episode/five-step
simulator smoke tests must pass before either full training job starts. Then
each GPU trains its own 10k run followed by its own 100-episode eval.

The dataset was copied byte-for-byte from the prepared Brev dataset and the
pinned simulator source matches its SHA-256. Training uses the local CUDA12.8
PyTorch wheel (Blackwell support) and the same model/library versions. Outputs,
caches, logs, and state live in `/mnt/wdblack/tempo/waitatgoal-spatial-261110`.
The local supervisor never deletes checkpoints, touches cloud instances, or
shuts down this PC. The two-host Brev shutdown barrier now covers only the
nine remaining cloud evaluations. good-turquoise-basilisk remains untouched.

## Fresh local restart after temporary-file cleanup

The first local attempts (`a4f73c21`, `c8e164b0`) stopped at approximately
6280 and 5100 logged updates after worker IPC failed. The user confirmed a
temporary-file cleanup was running at the time. Their logs and metadata remain
at the original paths; neither full run reached its first 10k checkpoint.

`queue_waitatgoal_spatial_local_rerun1.yaml` starts fresh pretrained runs with
new W&B identities (`f43a91d2`, `d6497a30`) and separate `rerun1` output paths.
It reuses the verified dataset and completed frozen-vision caches. Training
parameters, 8 workers, pinning, persistent workers, 10k-only checkpointing, and
100-episode final evals are unchanged. There is no recovery/resume mechanism.
Temporary files remain under `/tmp`, explicitly requested by the user; avoid
cleaning active multiprocessing socket directories while these jobs run.

The cleanup also removed the original Hugging Face cache. Before the restart,
the pinned base model and VLM metadata were downloaded again, with the base
weights verified against the original SHA-256 above. Initial bootstrap logs
are preserved under `rerun1/bootstrap-failed-missing-hf-cache`.

On this exFAT data drive, direct safetensors mmap writes then blocked the smoke
checkpoint writers. The local launcher opts into
`TEMPO_CHECKPOINT_STAGING_DIR=/tmp`: serialize the same checkpoint on the native
filesystem and copy it sequentially to the unchanged atomic WD destination.
Saved weights, optimizer/RNG state, run metadata and save frequency are
unchanged; no extra checkpoints or recovery behavior were added. The stopped
slow-write smoke files are retained in `rerun1/bootstrap-stopped-exfat-mmap`.

The WD issue also affected ordinary sequential writes: a disposable 2MiB
buffered write took 30.2 seconds. The actual restart therefore uses
`queue_waitatgoal_spatial_local_rerun1_ssd.yaml`, with live state, logs and
checkpoints in `/home/sydney1/Repos/RMBench/logs/waitatgoal-spatial-261110-rerun1`.
After each final eval, its checkpoint directory and results are copied to
`/mnt/wdblack/tempo/waitatgoal-spatial-261110/rerun1/ssd-outputs/<mode>`.
Archival owns no GPU context and happens after—not before—the evaluation.
The `_ssd` recipes retain exactly the same training parameters and run IDs;
only local output paths differ. Temp files remain in `/tmp`.
