# Training schedule preference

- WaitAtGoal's default schedule is `config/tempo/waitatgoal_defaults.yaml`:
  native warmup and mild decay 1e-4 to 7.5e-5. New WaitAtGoal queue recipes
  reference this file with `defaults_config` and validate materialized settings.
- Use the native SmolVLA scheduler (`scheduler: smolvla`) for new SmolVLA and
  TEMPO runs. Do not choose `legacy` or silently fall back to it.
- Retain native LR warmup, including when the user requests a constant LR:
  constant means warmup followed by a flat LR (`minimum_lr == learning_rate`),
  not `warmup_steps: 0`.
- Match native and TEMPO scheduler settings and effective warmup exactly in
  comparisons. Report native short-run scaling (the 1000/30000 preset becomes
  333 effective warmup steps for 10000 updates, or 400 for 12000 updates).
- Historical legacy/no-warmup recipes and completed results are provenance;
  do not rewrite them to describe a schedule that was not actually used.
