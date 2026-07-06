"""Evaluation metrics and pipeline for speaker identification."""

from .metrics import compute_eer, compute_dcf, compute_accuracy, compute_roc_auc
from .evaluator import Evaluator

__all__ = ["compute_eer", "compute_dcf", "compute_accuracy", "compute_roc_auc", "Evaluator"]
