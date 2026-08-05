"""Optional adapters for fragile third-party option math packages."""

from importlib import import_module


def require_py_vollib_vectorized() -> object:
    """Import py_vollib_vectorized only when the optional options extra is installed."""
    return import_module("py_vollib_vectorized")
