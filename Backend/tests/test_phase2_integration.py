from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ml.model import build_baseline_model
from ml.optimization import build_classification_loss, build_optimizer
from ml.training import fit
from ml.train_baseline import set_seed
from ml.config import IMAGE_SIZE


def test_complete_phase2_pipeline(tmp_path: Path):
    set_seed(42)
    device = torch.device("cpu")
    train_x = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    train_y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    val_x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    val_y = torch.tensor([0, 1, 0, 1])
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=2, shuffle=False)
    model = build_baseline_model(num_classes=2, pretrained=False)
    criterion = build_classification_loss()
    assert isinstance(criterion, nn.CrossEntropyLoss)
    optimizer = build_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    checkpoint = tmp_path / "phase2_baseline.pt"
    history = fit(model, train_loader, val_loader, criterion, optimizer, device, 1, checkpoint)
    assert len(history) == 1
    record = history[0]
    assert record["epoch"] == 1
    assert record["train_loss"] >= 0 and record["val_loss"] >= 0
    assert 0 <= record["train_accuracy"] <= 1 and 0 <= record["val_accuracy"] <= 1
    assert checkpoint.exists()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "model_state_dict" in saved and "optimizer_state_dict" in saved
    assert "train_metrics" in saved and "val_metrics" in saved
