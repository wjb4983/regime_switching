"""Reporting safeguards for representation-based regime research."""

EMBEDDING_INTERPRETATION_WARNING = (
    "Transformer or foundation-model embeddings are representation features, not inherently "
    "interpretable regimes. Treat downstream labels as regimes only when temporal stability, "
    "economic meaning, out-of-sample behavior, and other independent evidence support that claim."
)


def embedding_interpretation_warning() -> str:
    """Return the mandatory caveat for reports using transformer embeddings."""
    return EMBEDDING_INTERPRETATION_WARNING
