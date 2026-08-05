"""Shared configuration loading primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

ConfigT = TypeVar("ConfigT", bound="RegimeBaseConfig")
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-?(?P<default>[^}]*))?\}")


class ConfigLoadError(ValueError):
    """Raised when structured configuration cannot be loaded or validated."""


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"extends", "includes"}:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def interpolate_env(value: Any) -> Any:
    """Recursively interpolate ``${VAR}`` and ``${VAR:-default}`` strings."""

    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            msg = (
                f"Environment variable {name!r} is required by the configuration. "
                "Set it in the environment or use ${VAR:-default}."
            )
            raise ConfigLoadError(msg)

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, Mapping):
        return {key: interpolate_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_env(item) for item in value]
    return value


def normalize_path(value: Any) -> Any:
    """Normalize local filesystem paths without requiring them to exist."""

    if value is None or isinstance(value, Path):
        path = value
    else:
        path = Path(str(value))
    if path is None:
        return None
    return path.expanduser().resolve(strict=False)


class RegimeBaseConfig(BaseModel):
    """Base class for typed workflow configuration objects."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )
    _path_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _interpolate_environment(cls, data: Any) -> Any:
        return interpolate_env(data)

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_declared_paths(cls, value: Any, info: Any) -> Any:
        if info.field_name in cls._path_fields:
            if isinstance(value, list):
                return [normalize_path(item) for item in value]
            return normalize_path(value)
        return value

    def stable_dict(self) -> dict[str, Any]:
        """Return a JSON-stable representation suitable for hashing."""

        return self.model_dump(mode="json", exclude_none=True, by_alias=True)

    def config_hash(self) -> str:
        """Generate a deterministic SHA-256 hash for the validated config."""

        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        """Load, compose, interpolate, and validate a YAML config file."""

        data = load_yaml_mapping(Path(path))
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigLoadError(format_validation_error(path, exc)) from exc


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return value
    return (value,)


def load_yaml_mapping(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML mapping with ``extends`` inheritance and ``includes`` composition."""

    resolved = path.expanduser().resolve(strict=False)
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ConfigLoadError(f"Config inheritance cycle detected: {chain}")
    if not resolved.exists():
        raise ConfigLoadError(f"Config file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigLoadError(
            f"Config file {resolved} must contain a YAML mapping at the top level."
        )

    merged: dict[str, Any] = {}
    next_seen = seen | {resolved}
    for parent in _as_sequence(raw.get("extends")):
        merged = _deep_merge(merged, load_yaml_mapping(resolved.parent / str(parent), next_seen))
    for included in _as_sequence(raw.get("includes")):
        merged = _deep_merge(merged, load_yaml_mapping(resolved.parent / str(included), next_seen))
    return _deep_merge(merged, raw)


def format_validation_error(path: str | Path, exc: ValidationError) -> str:
    """Format Pydantic errors with remediation hints."""

    lines = [f"Invalid configuration in {Path(path)}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']} (received {error.get('input')!r}).")
    lines.append("Review the field name, type, allowed values, and required fields shown above.")
    return "\n".join(lines)


def load_config(path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    """Load a YAML file into the requested typed config model."""

    return config_type.from_yaml(path)
