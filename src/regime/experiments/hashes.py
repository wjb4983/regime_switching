"""Stable hashing helpers for experiment inputs and outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from regime.logging import redact


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-like values or bytes."""
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            redact(value), sort_keys=True, default=repr, separators=(",", ":")
        ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return a SHA-256 hash for one local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hash(path: str | Path) -> str:
    """Return a deterministic hash for all files under a directory."""
    root = Path(path)
    entries = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {"path": file_path.relative_to(root).as_posix(), "sha256": file_hash(file_path)}
        )
    return stable_hash(entries)


def config_hash(config: Mapping[str, Any] | str | Path) -> str:
    """Hash a config mapping or config file."""
    path = Path(config) if isinstance(config, str | Path) else None
    if path is not None and path.exists():
        return file_hash(path)
    return stable_hash(config)


def dataset_hash(dataset: Any) -> str:
    """Hash dataset content or descriptors such as paths, tabular objects, or arrays."""
    if isinstance(dataset, str | Path):
        path = Path(dataset)
        if path.is_dir():
            return directory_hash(path)
        if path.exists():
            return file_hash(path)
    if hasattr(dataset, "to_json"):
        return stable_hash(dataset.to_json())
    if hasattr(dataset, "tolist"):
        return stable_hash(dataset.tolist())
    return stable_hash(dataset)


def feature_hash(features: Sequence[str] | Mapping[str, Any] | Any) -> str:
    """Hash feature definitions, selected feature names, or feature matrices."""
    return dataset_hash(features)


def model_hash(model: Any) -> str:
    """Hash model parameters or a fitted model descriptor."""
    if hasattr(model, "get_params"):
        return stable_hash(model.get_params(deep=True))
    if hasattr(model, "to_record"):
        return stable_hash(model.to_record())
    return stable_hash(repr(model))


__all__ = [
    "config_hash",
    "dataset_hash",
    "directory_hash",
    "feature_hash",
    "file_hash",
    "model_hash",
    "stable_hash",
]
