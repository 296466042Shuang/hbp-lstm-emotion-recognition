"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy and macro-F1 for integer class labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
