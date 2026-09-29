from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _positive_class_index(classes: Sequence[str]) -> int:
    """Return the clinically positive class index for binary classification.

    Supported positive labels include the project's synthetic label
    ``glaucoma`` and the real HYGD label ``GON+``.
    """

    normalized = [
        str(c).strip().lower()
        for c in classes
    ]

    positive_candidates = [
        "glaucoma",
        "gon+",
        "gon-positive",
        "gon_positive",
    ]

    for candidate in positive_candidates:
        if candidate in normalized:
            return normalized.index(candidate)

    # Preserve the original project fallback.
    return 1


def binary_sensitivity_specificity(
    cm: np.ndarray,
    classes: Sequence[str],
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Return sensitivity, specificity, PPV and NPV."""

    if len(classes) != 2 or cm.shape != (2, 2):
        return None, None, None, None

    positive = _positive_class_index(classes)
    negative = 1 - positive

    tp = float(cm[positive, positive])
    fn = float(cm[positive, negative])
    fp = float(cm[negative, positive])
    tn = float(cm[negative, negative])

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    ppv = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    npv = (
        tn / (tn + fn)
        if tn + fn
        else 0.0
    )

    return (
        sensitivity,
        specificity,
        ppv,
        npv,
    )


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict:
    """Calculate standard GLAUCO-SENSE classification metrics."""

    labels = list(range(len(classes)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    if len(classes) == 2:
        positive = _positive_class_index(classes)

        precision = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            pos_label=positive,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            pos_label=positive,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            pos_label=positive,
            zero_division=0,
        )

    else:
        precision = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

    sensitivity, specificity, ppv, npv = (
        binary_sensitivity_specificity(
            cm,
            classes,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": (
            None
            if sensitivity is None
            else float(sensitivity)
        ),
        "specificity": (
            None
            if specificity is None
            else float(specificity)
        ),
        "ppv": (
            None
            if ppv is None
            else float(ppv)
        ),
        "npv": (
            None
            if npv is None
            else float(npv)
        ),
    }