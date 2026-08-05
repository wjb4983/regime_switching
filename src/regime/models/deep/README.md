# Deep regime models

Install this optional backend with `pip install 'regime-switching[deep]'`. The core
package does not import PyTorch.

## Quick start (recommended order)

1. Construct `DeepModelConfig`, selecting `device`, `precision`, `random_seed`, and
   a trailing `validation_window`.
2. Instantiate one of `LSTM`, `GRU`, `TemporalConvolutionalNetwork`, `NeuralHMM`,
   `DeepMarkovModel`, `VariationalStateSpaceModel`, `VectorQuantizedVAE`,
   `NeuralChangePointDetector`, or `GraphDependencyNetwork`.
3. Call `fit(series, config)`. Training windows and the trailing validation window
   are kept disjoint; early stopping monitors only the latter.
4. Optionally call `calibrate(calibration_series, state_labels)` for supervised
   temperature scaling. Without labels, fitting applies conservative entropy-based
   temperature calibration to the validation window.
5. Use `predict_proba`, `predict`, or `change_probabilities` for inference.
6. Pass `transform(series)` output to an HMM, HSMM, clustering, or jump-model head.
7. Persist the complete model with `save(path)` and restore it with `load(path)`.

`device="auto"` selects CUDA, then Apple MPS, then CPU. Explicit unavailable devices
fail rather than silently falling back. Mixed precision is enabled on CUDA; float32
is retained on other devices for numerical safety.
