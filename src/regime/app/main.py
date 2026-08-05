"""Thin Streamlit presentation layer for existing experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regime.artifacts import ArtifactBrowser, ArtifactSummary, matching_artifacts


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    return parser.parse_known_args()[0]


def _table(browser: ArtifactBrowser, artifact: ArtifactSummary) -> pd.DataFrame | None:
    try:
        return browser.load_table(artifact)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _probability_chart(browser: ArtifactBrowser, artifact: ArtifactSummary) -> None:
    frame = _table(browser, artifact)
    if frame is None or frame.empty:
        st.warning(f"Cannot display {artifact.filename} as a table.")
        return
    date_columns = [column for column in frame if str(column).casefold() in {"date", "timestamp"}]
    x = date_columns[0] if date_columns else frame.index
    numeric = list(frame.select_dtypes("number").columns)
    probability = [column for column in numeric if "prob" in str(column).casefold()]
    st.plotly_chart(px.area(frame, x=x, y=probability or numeric), use_container_width=True)


def _surface(browser: ArtifactBrowser, artifact: ArtifactSummary) -> None:
    frame = _table(browser, artifact)
    required = {"tenor", "moneyness", "implied_volatility"}
    if frame is None or not required.issubset(frame.columns):
        st.warning(f"{artifact.filename} needs tenor, moneyness, and implied_volatility columns.")
        return
    pivot = frame.pivot_table(
        index="tenor", columns="moneyness", values="implied_volatility", aggfunc="mean"
    )
    st.plotly_chart(
        px.imshow(pivot, labels={"x": "Moneyness", "y": "Tenor", "color": "IV"}),
        use_container_width=True,
    )


def main() -> None:
    """Render a read-only view of a pre-existing local experiment registry."""
    st.set_page_config(page_title="Regime experiment artifacts", layout="wide")
    st.title("Regime experiment artifacts")
    try:
        browser = ArtifactBrowser(_arguments().root)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    runs = browser.runs()
    selected = st.sidebar.multiselect(
        "Runs", options=runs, default=list(runs[:1]), format_func=lambda run: run.label
    )
    run_ids = tuple(run.run_id for run in selected)
    artifacts = browser.artifacts(run_ids)

    tabs = st.tabs(
        ["Model comparison", "Regime probabilities", "Downstream", "Options surfaces", "Downloads"]
    )
    with tabs[0]:
        comparison = browser.results(run_ids)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    with tabs[1]:
        candidates = matching_artifacts(artifacts, "probabilities", "probability")
        for artifact in candidates:
            st.subheader(artifact.filename)
            _probability_chart(browser, artifact)
    with tabs[2]:
        downstream = matching_artifacts(artifacts, "downstream", "performance", "metrics")
        for artifact in downstream:
            st.subheader(artifact.filename)
            frame = _table(browser, artifact)
            if frame is not None:
                st.dataframe(frame, use_container_width=True, hide_index=True)
    with tabs[3]:
        surfaces = matching_artifacts(artifacts, "option", "surface", "volatility")
        for artifact in surfaces:
            st.subheader(artifact.filename)
            _surface(browser, artifact)
    with tabs[4]:
        for artifact in artifacts:
            try:
                data, media_type = browser.download(artifact)
            except (FileNotFoundError, OSError) as error:
                st.caption(f"Unavailable: {artifact.filename} ({error})")
                continue
            st.download_button(
                f"Download {artifact.filename}", data, artifact.filename, media_type,
                key=artifact.artifact_id,
            )


if __name__ == "__main__":
    main()
