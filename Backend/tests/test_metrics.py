import numpy as np

from app.ml.inference import DEFAULT_CLASSES
from ml.metrics import binary_sensitivity_specificity, classification_metrics


def test_binary_sensitivity_specificity():
    cm = np.array([[8, 2], [1, 9]])
    sensitivity, specificity, ppv, npv = binary_sensitivity_specificity(cm, ["normal", "glaucoma"])
    assert round(sensitivity, 6) == round(9 / 10, 6)
    assert round(specificity, 6) == round(8 / 10, 6)
    assert round(ppv, 6) == round(9 / 11, 6)
    assert round(npv, 6) == round(8 / 9, 6)


def test_classification_metrics_uses_glaucoma_as_positive():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["normal", "glaucoma"])
    assert metrics["accuracy"] == 0.75
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0


def test_default_classes_exist():
    assert DEFAULT_CLASSES == ["glaucoma", "normal"]
