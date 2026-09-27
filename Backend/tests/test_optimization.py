import torch
from torch import nn
from ml.optimization import build_classification_loss, build_optimizer


def test_default_classification_loss():
    criterion = build_classification_loss()
    assert isinstance(criterion, nn.CrossEntropyLoss)
    loss = criterion(torch.tensor([[2.0, 0.5], [0.2, 1.5]]), torch.tensor([0, 1]))
    assert torch.isfinite(loss) and loss.item() > 0.0


def test_weighted_classification_loss():
    criterion = build_classification_loss(class_weights=[1.0, 2.0])
    assert torch.equal(criterion.weight, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_optimizer_is_adamw():
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_invalid_learning_rate():
    model = nn.Linear(4, 2)
    try:
        build_optimizer(model, learning_rate=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid learning rate.")


def test_invalid_class_weights():
    try:
        build_classification_loss(class_weights=[1.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive weights.")
