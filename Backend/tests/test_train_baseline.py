from pathlib import Path
from ml.train_baseline import run_smoke_training


def test_smoke_training_runs(tmp_path: Path):
    checkpoint = tmp_path / "baseline.pt"
    history = run_smoke_training(epochs=1, checkpoint_path=checkpoint)
    assert len(history) == 1
    record = history[0]
    assert record["epoch"] == 1
    assert record["train_loss"] >= 0.0 and record["val_loss"] >= 0.0
    assert 0.0 <= record["train_accuracy"] <= 1.0
    assert 0.0 <= record["val_accuracy"] <= 1.0
    assert checkpoint.exists()
