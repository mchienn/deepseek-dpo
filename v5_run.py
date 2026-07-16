"""Shared reproducibility helpers for DPO v5 runs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_manifest(run_dir: str | Path, *, kind: str, config: dict[str, Any], inputs: dict[str, str | Path]) -> Path:
    run_path = Path(run_dir)
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": config,
        "inputs": {
            name: {"path": str(Path(path)), "sha256": sha256_file(path)}
            for name, path in inputs.items()
            if Path(path).is_file()
        },
    }
    target = run_path / "manifest.json"
    write_json(target, manifest)
    return target
