from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ml.training import fit, train_one_epoch, validate_one_epoch


def make_loader(num_samples: int = 8, batch_size: int = 4) -> DataLoader:
    torch.manual_seed(42)
    x = torch.randn(num_samples, 4)
    y = torch.randint(0, 2, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_train_one_epoch():
    loader = make_loader(); model = make_model(); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    assert "loss" in result and "accuracy" in result
    assert result["loss"] >= 0.0 and 0.0 <= result["accuracy"] <= 1.0


def test_validate_one_epoch():
    loader = make_loader(); model = make_model(); criterion = nn.CrossEntropyLoss()
    result = validate_one_epoch(model, loader, criterion, torch.device("cpu"))
    assert "loss" in result and "accuracy" in result
    assert result["loss"] >= 0.0 and 0.0 <= result["accuracy"] <= 1.0


def test_fit_creates_best_checkpoint(tmp_path: Path):
    train_loader = make_loader(); val_loader = make_loader(); model = make_model(); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint = tmp_path / "best_model.pt"
    history = fit(model, train_loader, val_loader, criterion, optimizer, torch.device("cpu"), 2, checkpoint)
    assert len(history) == 2 and checkpoint.exists()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "model_state_dict" in saved and "optimizer_state_dict" in saved
    assert "train_metrics" in saved and "val_metrics" in saved
