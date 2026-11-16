#!/usr/bin/env python3
"""Patch the pinned LeRobot checkout to sanitize W&B artifact names fully.

The pinned v0.4.4 helper only replaces ``:`` and ``/``.  Composite local
dataset IDs also contain commas, which W&B rejects when a checkpoint is
uploaded.  This patch is deliberately exact and idempotent so setup fails
closed if the pinned implementation changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = '''def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")
'''

SANITIZED = '''def get_safe_wandb_artifact_name(name: str):
    """Return a W&B-valid artifact identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
'''

NEW = '''def get_safe_wandb_artifact_name(name: str):
    """Return a valid, deterministic W&B artifact identifier."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) <= 128:
        return safe
    import hashlib
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"{safe[:111]}_{digest}"
'''

GROUP_OLD = '''            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True),
'''

GROUP_NEW = '''            notes=self.cfg.notes,
            group=os.environ.get("WANDB_RUN_GROUP") or None,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True),
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lerobot_root", type=Path)
    args = parser.parse_args()
    target = args.lerobot_root / "src" / "lerobot" / "rl" / "wandb_utils.py"
    source = target.read_text()
    changed = False
    if NEW not in source:
        previous = SANITIZED if SANITIZED in source else OLD if OLD in source else None
        if previous is None:
            raise RuntimeError(f"Pinned LeRobot artifact-name helper did not match expected source: {target}")
        source = source.replace(previous, NEW, 1)
        changed = True
    if GROUP_NEW not in source:
        if GROUP_OLD not in source:
            raise RuntimeError(f"Pinned LeRobot W&B logger did not match expected source: {target}")
        source = source.replace(GROUP_OLD, GROUP_NEW, 1)
        changed = True
    if changed:
        target.write_text(source)
        print(f"Patched LeRobot W&B artifact naming and grouping: {target}")
    else:
        print(f"LeRobot W&B artifact naming and grouping patches already present: {target}")


if __name__ == "__main__":
    main()
