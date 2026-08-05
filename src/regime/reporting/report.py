"""Report document models and Jinja2 rendering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape

ProbabilityKind = Literal["filtered", "smoothed", "not applicable"]


@dataclass(frozen=True, slots=True)
class VisualizationMetadata:
    """Required research context displayed alongside every visualization."""

    research_question: str
    interpretation: str
    data_period: str
    model_version: str
    config_hash: str
    probability_kind: ProbabilityKind = "not applicable"

    @classmethod
    def from_config(
        cls,
        *,
        research_question: str,
        interpretation: str,
        data_period: str,
        model_version: str,
        config: Mapping[str, Any],
        probability_kind: ProbabilityKind = "not applicable",
    ) -> VisualizationMetadata:
        """Create metadata with a deterministic hash of a JSON-compatible config."""
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        config_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return cls(
            research_question=research_question,
            interpretation=interpretation,
            data_period=data_period,
            model_version=model_version,
            config_hash=config_hash,
            probability_kind=probability_kind,
        )

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if not str(value).strip():
                raise ValueError(f"{field} must not be empty")


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """Rendered figure fragment plus its mandatory context."""

    title: str
    html: str
    metadata: VisualizationMetadata
    kind: str = "figure"


class ReportBuilder:
    """Collect figures and write a portable, static HTML research report."""

    def __init__(self, title: str, *, subtitle: str = "") -> None:
        if not title.strip():
            raise ValueError("title must not be empty")
        self.title = title
        self.subtitle = subtitle
        self._figures: list[ReportFigure] = []

    @property
    def figures(self) -> tuple[ReportFigure, ...]:
        """Return an immutable view of report figures."""
        return tuple(self._figures)

    def add(self, figure: ReportFigure) -> ReportBuilder:
        """Append a visualization and return this builder for chaining."""
        self._figures.append(figure)
        return self

    def render(self, *, generated_at: datetime | date | None = None) -> str:
        """Render a complete HTML document with Plotly embedded once."""
        template_dir = files("regime.reporting").joinpath("templates")
        environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(("html", "xml")),
        )
        template = environment.get_template("report.html.j2")
        stamp = generated_at or datetime.now().astimezone()
        return template.render(
            title=self.title,
            subtitle=self.subtitle,
            figures=self._figures,
            generated_at=stamp.isoformat(),
        )

    def write(self, path: str | Path, *, generated_at: datetime | date | None = None) -> Path:
        """Atomically write the report and return its resolved destination."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(self.render(generated_at=generated_at), encoding="utf-8")
        temporary.replace(destination)
        return destination


def render_report(
    title: str,
    figures: Sequence[ReportFigure],
    output: str | Path,
    *,
    subtitle: str = "",
) -> Path:
    """Convenience API to build and write a report in one call."""
    builder = ReportBuilder(title, subtitle=subtitle)
    for figure in figures:
        builder.add(figure)
    return builder.write(output)
