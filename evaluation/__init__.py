"""ASR evaluation package: configuration, normalization, metrics, and execution pipeline.

Modules
-------
config:     Defines schema and helpers for generating test configurations.
normalization: Utilities to normalize ASR outputs to canonical numeric forms.
metrics:    Computation of WER, CER, Digit Error Rate, MAE, and aggregate stats.
pipeline:   Orchestrates evaluation across datasets and model configurations.
"""
from .config import EvaluationConfig, expand_parameter_grid, ModelConfig, InferenceConfig
from .metrics import (
    word_error_rate, char_error_rate, digit_error_rate,
    numeric_exact_match, mean_absolute_error_numbers,
)
from .pipeline import ASREvaluator

__all__ = [
    "EvaluationConfig",
    "ModelConfig",
    "InferenceConfig",
    "expand_parameter_grid",
    "ASREvaluator",
    "word_error_rate",
    "char_error_rate",
    "digit_error_rate",
    "numeric_exact_match",
    "mean_absolute_error_numbers",
]
