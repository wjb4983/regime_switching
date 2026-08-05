"""Project path helpers based on :class:`pathlib.Path`."""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = PACKAGE_DIR.parent
ROOT_DIR: Path = SRC_DIR.parent
CONFIGS_DIR: Path = ROOT_DIR / "configs"
DOCS_DIR: Path = ROOT_DIR / "docs"
EXAMPLES_DIR: Path = ROOT_DIR / "examples"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
REPORTS_DIR: Path = ROOT_DIR / "reports"
TESTS_DIR: Path = ROOT_DIR / "tests"

__all__ = [
    "CONFIGS_DIR",
    "DOCS_DIR",
    "EXAMPLES_DIR",
    "NOTEBOOKS_DIR",
    "PACKAGE_DIR",
    "REPORTS_DIR",
    "ROOT_DIR",
    "SRC_DIR",
    "TESTS_DIR",
]
