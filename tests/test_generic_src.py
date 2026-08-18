"""一般化した src API の回帰テスト。Fashion-MNIST 既存テストは壊さない。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import (
    factorize_named_conv2d,
    factorize_named_layers,
    factorize_named_linear,
    sweep_conv_svd_ranks,
)
from nn_compression.datasets import shuffled_index_splits
from nn_compression.metrics import collect_compression_metrics, estimate_cnn_macs
from nn_compression.models import CIFAR10CNN, FashionMNISTCNN
from nn_compression.training import fit_with_early_stopping
from nn_compression.utils import get_named_module, set_named_module


class NestedConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1))
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _tiny_loader(n: int = 8, c: int = 2, h: int = 8, classes: int = 3):
    images = torch.randn(n, c, h, h)
    labels = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_named_module_nested_roundtrip():
    model = NestedConv()
    original = get_named_module(model, "block.0")
    replacement = nn.Conv2d(2, 4, 3, padding=1)
    set_named_module(model, "block.0", replacement)
    assert get_named_module(model, "block.0") is replacement
    assert original is not replacement


def test_factorize_named_nested_conv_does_not_mutate():
    torch.manual_seed(0)
    model = NestedConv()
    original = get_named_module(model, "block.0")
    compressed = factorize_named_conv2d(model, "block.0", 2)
    assert get_named_module(model, "block.0") is original
    assert isinstance(get_named_module(compressed, "block.0"), nn.Sequential)


def test_factorize_named_layers_conv_and_linear():
    torch.manual_seed(0)
    model = NestedConv()
    compressed = factorize_named_layers(
        model,
        conv_ranks={"block.0": 2},
        linear_ranks={"fc": 2},
    )
    assert isinstance(get_named_module(compressed, "block.0"), nn.Sequential)
    assert isinstance(compressed.fc, nn.Sequential)
    assert isinstance(model.block[0], nn.Conv2d)
    x = torch.randn(2, 2, 8, 8)
    assert compressed(x).shape == model(x).shape


def test_shuffled_index_splits_reproducible():
    a = shuffled_index_splits(10, (4, 3, 3), seed=0)
    b = shuffled_index_splits(10, (4, 3, 3), seed=0)
    c = shuffled_index_splits(10, (4, 3, 3), seed=1)
    assert a == b
    assert a != c
    assert tuple(len(part) for part in a) == (4, 3, 3)
    assert sorted(a[0] + a[1] + a[2]) == list(range(10))


def test_cifar10cnn_forward_and_parameter_count():
    model = CIFAR10CNN()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 128_842


def test_estimate_cnn_macs_cifar_like_three_convs():
    model = CIFAR10CNN()
    baseline, compressed, reduction = estimate_cnn_macs(
        model,
        verbose=False,
        conv_ranks={"conv1": 9, "conv2": 16, "conv3": 32},
        linear_ranks={},
        conv_output_hw={
            "conv1": (32, 32),
            "conv2": (16, 16),
            "conv3": (8, 8),
        },
        linear_layer_names=("fc1", "fc2"),
    )
    assert compressed < baseline
    assert 0.0 < reduction < 1.0


def test_fit_reevaluate_train_false_logs_train_one_epoch_metrics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 3))
    images = torch.randn(16, 1, 2, 4)
    labels = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = fit_with_early_stopping(
        model,
        loader,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        max_epochs=2,
        patience=5,
        min_delta=0.0,
        reevaluate_train=False,
        log_every_epoch=False,
    )
    assert result["best_epoch"] is not None
    assert len(result["history"]) == 2
    assert "train_acc" in result["history"][0]


def test_sweep_conv_svd_ranks_one_rank():
    torch.manual_seed(0)
    model = NestedConv()
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    rows = sweep_conv_svd_ranks(
        model,
        "block.0",
        [2],
        (8, 8),
        loader,
        criterion,
        device,
        baseline_acc=0.0,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
        verbose=False,
    )
    assert len(rows) == 1
    assert rows[0]["layer"] == "block.0"
    assert rows[0]["rank"] == 2
    assert "validation_acc" in rows[0]
    assert isinstance(model.block[0], nn.Conv2d)


def test_collect_compression_metrics_keys():
    torch.manual_seed(0)
    model = NestedConv()
    compressed = factorize_named_linear(model, "fc", 2)
    row = collect_compression_metrics(
        model,
        compressed,
        _tiny_loader(),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.5,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
        compressed_macs=10,
        compute_reduction=0.1,
    )
    assert row["compressed_macs"] == 10
    assert "agreement" in row
    assert "model" not in row

    row_with_model = collect_compression_metrics(
        model,
        compressed,
        _tiny_loader(),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.5,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
        include_model=True,
    )
    assert row_with_model["model"] is compressed


def test_fashion_mnist_cnn_still_imports():
    model = FashionMNISTCNN()
    assert model.conv1.in_channels == 1


if __name__ == "__main__":
    tests = [
        test_named_module_nested_roundtrip,
        test_factorize_named_nested_conv_does_not_mutate,
        test_factorize_named_layers_conv_and_linear,
        test_shuffled_index_splits_reproducible,
        test_cifar10cnn_forward_and_parameter_count,
        test_estimate_cnn_macs_cifar_like_three_convs,
        test_fit_reevaluate_train_false_logs_train_one_epoch_metrics,
        test_sweep_conv_svd_ranks_one_rank,
        test_collect_compression_metrics_keys,
        test_fashion_mnist_cnn_still_imports,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all passed")
