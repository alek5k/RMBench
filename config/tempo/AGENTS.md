# WaitAtGoal defaults

- New WaitAtGoal SmolVLA and TEMPO runs use the shared schedule in
  `waitatgoal_defaults.yaml`: native SmolVLA scheduler, peak LR 1e-4, floor
  7.5e-5, warmup preset 1000 and decay preset 30000. Keep warmup enabled.
- Training recipes remain fully materialized and compatible with the same
  trainer; new queue recipes set `defaults_config: waitatgoal_defaults.yaml`
  to validate that both policies agree with the authoritative default.
- Native short-run scaling gives 400 effective warmup updates for 12k steps
  (333 for 10k); report effective rather than only nominal warmup when asked.
- Preserve historical configs/results; do not overwrite old run IDs or
  output paths. Use fresh identities for every new training run.
