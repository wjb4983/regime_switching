"""Atomic, portable checkpoint persistence."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manage named checkpoints using atomic replacement in one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, suffix: str = ".pkl") -> Path:
        if not name or Path(name).name != name:
            raise ValueError("checkpoint name must be a non-empty file name")
        return self.directory / f"{name}{suffix}"

    def save(self, name: str, value: Any) -> Path:
        destination = self.path(name)
        return self._atomic_write(
            destination, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def load(self, name: str, default: Any = None) -> Any:
        source = self.path(name)
        return pickle.loads(source.read_bytes()) if source.exists() else default

    def save_json(self, name: str, value: Any) -> Path:
        payload = json.dumps(value, sort_keys=True, indent=2).encode("utf-8")
        return self._atomic_write(self.path(name, ".json"), payload)

    def load_json(self, name: str, default: Any = None) -> Any:
        source = self.path(name, ".json")
        return json.loads(source.read_text(encoding="utf-8")) if source.exists() else default

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> Path:
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}."
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination


__all__ = ["CheckpointManager"]
