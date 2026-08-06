"""Discover constructible model implementations from the command line."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from pydantic_core import PydanticUndefined

from regime.models import available_models, model_spec

app = typer.Typer(no_args_is_help=True, help="Discover registered regime models.")


@app.command("list")
def list_models() -> None:
    """List canonical model names, capabilities, and optional extras."""
    for spec in available_models():
        extra = spec.optional_dependency_group or "core"
        typer.echo(f"{spec.name}\t{','.join(sorted(spec.capabilities))}\t{extra}")


@app.command()
def describe(name: Annotated[str, typer.Argument(help="Canonical model name or alias.")]) -> None:
    """Describe configuration fields, defaults, aliases, and capabilities."""
    spec = model_spec(name)
    fields: dict[str, dict[str, object]] = {}
    for field_name, field in spec.config_class.model_fields.items():
        default: object = field.default
        if default is PydanticUndefined:
            default = "<required>"
        fields[field_name] = {
            "annotation": str(field.annotation),
            "default": default,
            "description": field.description,
        }
    payload = {
        "name": spec.name,
        "aliases": spec.aliases,
        "capabilities": sorted(spec.capabilities),
        "optional_dependency_group": spec.optional_dependency_group,
        "configuration": spec.config_class.__name__,
        "parameters": fields,
    }
    typer.echo(json.dumps(payload, indent=2, default=str))
