# Transformer representations

Install `regime-switching[transformers]`; the core package does not import Torch or
Hugging Face. Recommended quick-start order:

1. Validate a `TransformerConfig` for the numerical feature width and window.
2. Create `TimeSeriesTransformerEncoder` or `PatchTransformerEncoder` (or a foundation adapter).
3. Generate and inspect embeddings, checking alignment and preventing look-ahead leakage.
4. Attach exactly one HMM, HSMM, clustering, or jump-model head.
5. Validate labels out of sample and report stability, transitions, and economic evidence.
6. Include `embedding_interpretation_warning()` in every representation-based report.

Foundation models are representation generators here. **Their embeddings are not inherently
interpretable regimes.** A downstream label should be described as a regime only when temporal
stability, economic meaning, out-of-sample behavior, and independent evidence support it.
