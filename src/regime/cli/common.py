"""Shared, shell-independent CLI behavior."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import typer

from regime.config.base import ConfigLoadError, load_yaml_mapping
from regime.errors import RegimeError
from regime.experiments.runner import ExperimentRun, RunRegistry
from regime.experiments.store import ExperimentStore
from regime.logging import redact

P = ParamSpec("P")
T = TypeVar("T")
EXPERIMENTS_DIR = Path("experiments")


def emit(payload: Mapping[str, Any], *, error: bool = False) -> None:
    """Write a redacted JSON record to the appropriate stream."""
    typer.echo(json.dumps(redact(payload), sort_keys=True, default=str), err=error)


def command_errors(function: Callable[P, T]) -> Callable[P, T]:
    """Map expected failures to stable exit codes and JSON error records."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except (ConfigLoadError, ValueError) as exc:
            emit(
                {"status": "error", "error": {"code": "invalid_input", "message": str(exc)}},
                error=True,
            )
            raise typer.Exit(2) from None
        except FileNotFoundError as exc:
            emit(
                {"status": "error", "error": {"code": "not_found", "message": str(exc)}}, error=True
            )
            raise typer.Exit(3) from None
        except RegimeError as exc:
            emit({"status": "error", "error": exc.to_record()}, error=True)
            raise typer.Exit(1) from None
        except Exception as exc:
            emit(
                {
                    "status": "error",
                    "error": {
                        "code": "operation_failed",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                error=True,
            )
            raise typer.Exit(1) from None

    return wrapped


def config_workflow(
    operation: str,
    config_path: Path,
    *,
    resume: bool,
    worker: Callable[[ExperimentRun, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load configuration and run a registered, resumable operation."""
    path = config_path.expanduser().resolve(strict=False)
    config = load_yaml_mapping(path)
    registry = RunRegistry(ExperimentStore(EXPERIMENTS_DIR))
    run = registry.start(name=f"{operation}:{path}", config=config, resume=resume)
    try:
        result = dict(worker(run, config) if worker else {})
        run.log_json(
            "manifest", "command.json", {"operation": operation, "config": str(path), **result}
        )
        registry.store.update_run(run.run_id, "completed")
    except Exception:
        registry.store.update_run(run.run_id, "failed")
        raise
    payload = {"status": "completed", "operation": operation, "run_id": run.run_id, **result}
    emit(payload)
    return payload


def config_option(help_text: str) -> Any:
    """Return the common required config option declaration."""
    return typer.Option(
        ...,
        "--config",
        help=help_text,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    )


def resume_option() -> Any:
    """Return the common resume flag declaration."""
    return typer.Option(
        True, "--resume/--no-resume", help="Resume the latest incomplete matching run."
    )
