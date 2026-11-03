"""同期付き推論時間の測定。"""

import pytest
import torch
from torch import nn

from nn_compression.metrics import measure_synchronized_inference_latency


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))


def test_cpu_latency_returns_quartiles_without_moving_input():
    model = Tiny()
    images = torch.zeros(2, 4)
    before = images.device
    result = measure_synchronized_inference_latency(
        model, images, warmup=1, repetitions=5
    )
    assert images.device == before
    assert result["repetitions"] == 5
    assert result["batch_size"] == 2
    assert result["q1_seconds"] <= result["median_seconds"] <= result["q3_seconds"]
    assert model.training is True


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.inner = nn.Linear(3, 2)

    def forward(self, x):
        return self.inner(self.fc(x.view(x.shape[0], -1)))


class Exploding(Mixed):
    def forward(self, x):
        raise RuntimeError("forward failed")


def test_restores_mixed_train_and_eval_modules():
    model = Mixed()
    model.train()
    model.inner.eval()
    images = torch.zeros(2, 4)
    result = measure_synchronized_inference_latency(
        model, images, warmup=0, repetitions=1
    )
    assert model.training is True
    assert model.fc.training is True
    assert model.inner.training is False
    assert images.device.type == "cpu"
    assert result["batch_size"] == 2
    assert result["repetitions"] == 1


def test_keeps_model_that_started_in_eval():
    model = Mixed()
    model.eval()
    measure_synchronized_inference_latency(
        model, torch.zeros(2, 4), warmup=0, repetitions=1
    )
    assert model.training is False
    assert model.fc.training is False
    assert model.inner.training is False


def test_restores_modes_when_forward_raises():
    model = Exploding()
    model.train()
    model.inner.eval()
    with pytest.raises(RuntimeError, match="forward failed"):
        measure_synchronized_inference_latency(
            model, torch.zeros(2, 4), warmup=0, repetitions=1
        )
    assert model.training is True
    assert model.fc.training is True
    assert model.inner.training is False


def test_rejects_bad_counts():
    model = Tiny()
    with pytest.raises(ValueError):
        measure_synchronized_inference_latency(
            model, torch.zeros(1, 4), warmup=-1, repetitions=1
        )
