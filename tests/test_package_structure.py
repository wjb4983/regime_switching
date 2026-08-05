"""Import checks for the initial package structure."""

from importlib import import_module
from pathlib import Path

MODULES = [
    "regime.config",
    "regime.logging",
    "regime.errors",
    "regime.cli",
    "regime.data",
    "regime.features",
    "regime.datasets",
    "regime.models",
    "regime.models.clustering",
    "regime.validation",
    "regime.evaluation",
    "regime.backtesting",
    "regime.reporting",
    "regime.experiments",
    "regime.synthetic",
]


def test_package_modules_are_importable() -> None:
    """All planned top-level workflow packages should import cleanly."""
    for module_name in MODULES:
        assert import_module(module_name).__name__ == module_name


def test_project_paths_use_pathlib() -> None:
    """Repository path constants should be cross-platform pathlib paths."""
    from regime import CONFIGS_DIR, ROOT_DIR, SRC_DIR, TESTS_DIR

    assert isinstance(ROOT_DIR, Path)
    assert SRC_DIR == ROOT_DIR / "src"
    assert CONFIGS_DIR == ROOT_DIR / "configs"
    assert TESTS_DIR == ROOT_DIR / "tests"
