#!/usr/bin/env python3
"""Build one immutable RMBench/XPolicyLab source bundle for Brev."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_FORMAT = "rmbench-source-bundle-v1"
BUNDLE_ROOT = PurePosixPath("RMBench")
MANIFEST_PATH = BUNDLE_ROOT / "SOURCE_BUNDLE.json"
FORBIDDEN_BASENAMES = {
    ".env",
    "credentials.env",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}


def run(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.check_output(args, cwd=cwd)


def git_text(root: Path, *args: str) -> str:
    return os.fsdecode(run("git", "-C", str(root), *args)).strip()


def selected_paths(root: Path) -> list[PurePosixPath]:
    raw = run(
        "git",
        "-C",
        str(root),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    paths: list[PurePosixPath] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        relative = PurePosixPath(os.fsdecode(entry))
        validate_relative_path(relative)
        paths.append(relative)
    return sorted(paths, key=str)


def validate_relative_path(path: PurePosixPath) -> None:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe source path: {str(path)!r}")
    if ".git" in path.parts:
        raise ValueError(f"Git metadata may not be bundled: {path}")
    if path.name in FORBIDDEN_BASENAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"refusing to bundle credential-like path: {path}")
    if path.name.startswith(".env.") and path.name != ".env.example":
        raise ValueError(f"refusing to bundle credential-like path: {path}")


def validate_symlink(root: Path, relative: PurePosixPath, target: str) -> None:
    if os.path.isabs(target):
        raise ValueError(f"absolute symlink is not allowed: {relative} -> {target}")
    normalized = posixpath.normpath(posixpath.join(str(relative.parent), target))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"escaping symlink is not allowed: {relative} -> {target}")
    resolved = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"escaping symlink is not allowed: {relative} -> {target}") from error


def source_metadata(root: Path, paths: list[PurePosixPath]) -> dict[str, Any]:
    status = git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": git_text(root, "rev-parse", "HEAD"),
        "branch": git_text(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "file_count": len(paths),
    }


def materialize_entry(
    source_root: Path,
    relative: PurePosixPath,
    destination_root: Path,
    archive_prefix: PurePosixPath,
) -> dict[str, Any] | None:
    source = source_root / Path(*relative.parts)
    destination = destination_root / Path(*archive_prefix.parts) / Path(*relative.parts)
    if source.is_symlink():
        target = os.readlink(source)
        try:
            validate_symlink(source_root, relative, target)
        except ValueError as error:
            return {
                "path": str(archive_prefix / relative),
                "type": "excluded_symlink",
                "target": target,
                "reason": str(error),
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
        return {
            "path": str(archive_prefix / relative),
            "type": "symlink",
            "target": target,
        }
    if source.is_dir():
        return None
    if not source.is_file():
        raise ValueError(f"selected source is not a regular file or symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)
    mode = stat.S_IMODE(source.stat(follow_symlinks=False).st_mode)
    os.chmod(destination, mode)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": str(archive_prefix / relative),
        "type": "file",
        "mode": f"{mode:04o}",
        "size": destination.stat().st_size,
        "sha256": digest,
    }


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    return info


def write_tar(staging_root: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(
            staging_root / BUNDLE_ROOT.name,
            arcname=BUNDLE_ROOT.name,
            recursive=True,
            filter=tar_filter,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rmbench_root = args.rmbench_root.resolve()
    xpolicylab_root = args.xpolicylab_root.resolve()
    recipe = args.recipe.resolve()
    output_dir = args.output_dir.resolve()

    if not (rmbench_root / ".git").exists():
        raise ValueError(f"RMBench root is not a Git worktree: {rmbench_root}")
    if not (xpolicylab_root / ".git").exists():
        raise ValueError(f"XPolicyLab root is not a Git worktree: {xpolicylab_root}")
    recipe_relative = PurePosixPath(recipe.relative_to(rmbench_root).as_posix())

    rmbench_paths = [
        path
        for path in selected_paths(rmbench_root)
        if path != PurePosixPath("XPolicyLab")
    ]
    xpolicylab_paths = selected_paths(xpolicylab_root)
    if recipe_relative not in rmbench_paths:
        raise ValueError(f"recipe is ignored or outside the selected RMBench source set: {recipe}")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with tempfile.TemporaryDirectory(prefix="rmbench-source-bundle-") as temp_dir:
        temp_root = Path(temp_dir)
        staging_root = temp_root / "staging"
        entries: list[dict[str, Any]] = []
        for root, paths, prefix in (
            (rmbench_root, rmbench_paths, BUNDLE_ROOT),
            (xpolicylab_root, xpolicylab_paths, BUNDLE_ROOT / "XPolicyLab"),
        ):
            for relative in paths:
                entry = materialize_entry(root, relative, staging_root, prefix)
                if entry is not None:
                    entries.append(entry)

        source_files = sorted(
            (entry for entry in entries if entry["type"] != "excluded_symlink"),
            key=lambda entry: entry["path"],
        )
        exclusions = sorted(
            (entry for entry in entries if entry["type"] == "excluded_symlink"),
            key=lambda entry: entry["path"],
        )
        manifest: dict[str, Any] = {
            "format": BUNDLE_FORMAT,
            "created_at": created_at,
            "recipe": str(recipe_relative),
            "sources": {
                "rmbench": source_metadata(rmbench_root, rmbench_paths),
                "xpolicylab": source_metadata(xpolicylab_root, xpolicylab_paths),
            },
            "files": source_files,
            "exclusions": exclusions,
        }
        identity = {key: value for key, value in manifest.items() if key != "created_at"}
        identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        manifest["bundle_id"] = hashlib.sha256(identity_bytes).hexdigest()[:20]
        manifest_path = staging_root / Path(*MANIFEST_PATH.parts)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(manifest_path, 0o644)

        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"rmbench-source-{manifest['bundle_id']}"
        tar_path = temp_root / f"{base_name}.tar"
        archive_path = output_dir / f"{base_name}.tar.zst"
        checksum_path = output_dir / f"{base_name}.tar.zst.sha256"
        manifest_output = output_dir / f"{base_name}.manifest.json"
        write_tar(staging_root, tar_path)
        subprocess.run(
            ["zstd", "-q", "-10", "--threads=0", "--force", str(tar_path), "-o", str(archive_path)],
            check=True,
        )
        archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        checksum_path.write_text(f"{archive_digest}  {archive_path.name}\n", encoding="utf-8")
        shutil.copyfile(manifest_path, manifest_output)

    print(json.dumps({
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "manifest": str(manifest_output),
        "bundle_id": manifest["bundle_id"],
        "recipe": str(recipe_relative),
    }))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Source bundle build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
