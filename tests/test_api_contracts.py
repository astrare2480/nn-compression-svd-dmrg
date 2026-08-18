"""Codex レビュー指摘の回帰テスト。既存 Fashion-MNIST 数値テストは壊さない。"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import (
    factorize_conv2d_layer,
    factorize_linear_layer,
    make_two_layer_svd_model,
    sweep_conv_svd_ranks,
)
from nn_compression.compression.mlp_svd import make_one_layer_svd_model
from nn_compression.datasets import make_fashion_mnist_loaders, shuffled_index_splits
from nn_compression.metrics import (
    benchmark_inference,
    collect_compression_metrics,
    take_inference_batch,
)
from nn_compression.models import CIFAR10CNN, FashionMNISTCNN, MNISTMLP
from nn_compression.selection.knee import (
    find_knee_point,
    get_endpoints,
    get_first_point,
    line_equation,
)
from nn_compression.training import fit_with_early_stopping
from nn_compression.utils import get_named_module, set_named_module


class NestedConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1))
        self.features = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Conv2d(2, 4, 3, padding=1),
        )
        self.block1 = nn.Module()
        self.block1.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _tiny_loader(n: int = 16, *, shuffle: bool = False, generator=None):
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=shuffle,
        generator=generator,
    )


def test_make_two_layer_svd_model_legacy_keyword():
    torch.manual_seed(0)
    model = MNISTMLP()
    compressed = make_two_layer_svd_model(model, r1=8, r2=4)
    assert compressed.fc1[0].out_features == 8
    assert compressed.fc2[0].out_features == 4


def test_make_two_layer_svd_model_rejects_alias_conflict():
    model = MNISTMLP()
    with pytest.raises(ValueError, match="同時"):
        make_two_layer_svd_model(model, fc1_rank=8, r1=4, fc2_rank=4)


def test_compressed_mlp_does_not_share_fc3():
    torch.manual_seed(0)
    model = MNISTMLP()
    compressed = make_two_layer_svd_model(model, 8, 4)
    assert compressed.fc3 is not model.fc3
    assert compressed.fc3.weight.data_ptr() != model.fc3.weight.data_ptr()


def test_finetuning_compressed_does_not_change_baseline_fc3():
    torch.manual_seed(0)
    model = MNISTMLP()
    baseline_fc3 = model.fc3.weight.detach().clone()
    compressed = make_two_layer_svd_model(model, 8, 4)
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    optimizer = torch.optim.SGD(compressed.parameters(), lr=0.1)
    logits = compressed(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(model.fc3.weight, baseline_fc3)
    assert not torch.equal(compressed.fc3.weight, baseline_fc3)


def test_linear_factorize_keeps_float64():
    torch.manual_seed(0)
    layer = nn.Linear(6, 4, dtype=torch.float64)
    factorized = factorize_linear_layer(layer, 2)
    assert factorized[0].weight.dtype == torch.float64
    assert factorized[1].weight.dtype == torch.float64
    x = torch.randn(3, 6, dtype=torch.float64)
    assert factorized(x).dtype == torch.float64


def test_linear_and_conv_keep_device_and_dtype():
    cpu = torch.device("cpu")
    linear = nn.Linear(6, 4, device=cpu, dtype=torch.float64)
    factorized_linear = factorize_linear_layer(linear, 2)
    assert factorized_linear[0].weight.device == linear.weight.device
    assert factorized_linear[0].weight.dtype == torch.float64

    conv = nn.Conv2d(
        2,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        device=cpu,
        dtype=torch.float64,
    )
    factorized_conv = factorize_conv2d_layer(conv, 2)
    assert factorized_conv[0].weight.device == conv.weight.device
    assert factorized_conv[0].weight.dtype == torch.float64
    assert factorized_conv[0].stride == conv.stride
    assert factorized_conv[0].dilation == conv.dilation

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        gpu_linear = nn.Linear(6, 4, device=cuda)
        gpu_fact = factorize_linear_layer(gpu_linear, 2)
        assert gpu_fact[0].weight.device.type == "cuda"


def test_conv_groups_not_supported():
    conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    with pytest.raises(ValueError, match="groups"):
        factorize_conv2d_layer(conv, 2)


def test_conv_keeps_padding_mode():
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1, padding_mode="reflect")
    factorized = factorize_conv2d_layer(conv, 2)
    assert factorized[0].padding_mode == "reflect"
    assert factorized[0].stride == conv.stride
    assert factorized[0].padding == conv.padding
    assert factorized[0].dilation == conv.dilation


def test_requires_grad_false_is_preserved():
    layer = nn.Linear(5, 3)
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)
    factorized = factorize_linear_layer(layer, 2)
    assert factorized[0].weight.requires_grad is False
    assert factorized[1].weight.requires_grad is False
    assert factorized[1].bias.requires_grad is False

    conv = nn.Conv2d(2, 4, 3, padding=1)
    conv.weight.requires_grad_(False)
    conv.bias.requires_grad_(False)
    fact_conv = factorize_conv2d_layer(conv, 2)
    assert fact_conv[0].weight.requires_grad is False
    assert fact_conv[1].bias.requires_grad is False


@pytest.mark.parametrize("rank", [0, -1, 1000])
def test_linear_rank_validation(rank: int):
    layer = nn.Linear(4, 3)
    with pytest.raises(ValueError):
        factorize_linear_layer(layer, rank)


def test_conv_rank_validation_bounds():
    conv = nn.Conv2d(2, 4, 3, padding=1)
    max_rank = min(4, 2 * 3 * 3)
    with pytest.raises(ValueError):
        factorize_conv2d_layer(conv, 0)
    with pytest.raises(ValueError):
        factorize_conv2d_layer(conv, -3)
    with pytest.raises(ValueError):
        factorize_conv2d_layer(conv, max_rank + 1)


def test_nested_module_set_get():
    model = NestedConv()
    replacement = nn.Conv2d(2, 4, 3, padding=1)
    set_named_module(model, "block.0", replacement)
    assert get_named_module(model, "block.0") is replacement

    features_replacement = nn.Conv2d(2, 4, 3, padding=1)
    set_named_module(model, "features.3", features_replacement)
    assert get_named_module(model, "features.3") is features_replacement

    block1_replacement = nn.Conv2d(2, 4, 3, padding=1)
    set_named_module(model, "block1.conv", block1_replacement)
    assert get_named_module(model, "block1.conv") is block1_replacement

    with pytest.raises(AttributeError):
        set_named_module(model, "missing.0", replacement)


def test_knee_empty():
    df = pd.DataFrame(columns=["x", "y"])
    with pytest.raises(ValueError, match="空"):
        get_first_point(df, "x", "x", "y")


def test_knee_single_point():
    df = pd.DataFrame({"x": [0.2], "y": [0.8]})
    point, dist = find_knee_point(df, 1.0, 0.0, "x", "y")
    assert point == (0.2, 0.8)
    assert dist == 0.0


def test_knee_constant_x():
    df = pd.DataFrame({"x": [0.5, 0.5, 0.5], "y": [0.1, 0.4, 0.9]})
    p0, p1 = get_endpoints(df, "y", "x", "y")
    slope, intercept = line_equation(p0, p1)
    assert math.isinf(slope)
    point, dist = find_knee_point(df, slope, intercept, "x", "y")
    assert point[0] == 0.5
    assert dist == 0.0


def test_knee_constant_y():
    df = pd.DataFrame({"x": [0.1, 0.4, 0.9], "y": [0.3, 0.3, 0.3]})
    slope, intercept = line_equation((0.1, 0.3), (0.9, 0.3))
    assert slope == 0.0
    point, dist = find_knee_point(df, slope, intercept, "x", "y")
    assert point[1] == 0.3
    assert dist == 0.0


def test_knee_nan_and_inf():
    with pytest.raises(ValueError, match="NaN"):
        get_first_point(
            pd.DataFrame({"x": [0.1, float("nan")], "y": [0.2, 0.3]}),
            "x",
            "x",
            "y",
        )
    with pytest.raises(ValueError, match="inf"):
        get_first_point(
            pd.DataFrame({"x": [0.1, math.inf], "y": [0.2, 0.3]}),
            "x",
            "x",
            "y",
        )


def test_benchmark_repeats_zero():
    model = nn.Linear(4, 3)
    batch = torch.randn(2, 4)
    with pytest.raises(ValueError, match="repeats"):
        benchmark_inference(
            model,
            device=torch.device("cpu"),
            warmup=0,
            repeats=0,
            input_batch=batch,
        )


def test_benchmark_empty_loader():
    model = nn.Linear(4, 3)
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="空"):
        benchmark_inference(
            model,
            empty,
            torch.device("cpu"),
            warmup=0,
            repeats=1,
        )


def test_benchmark_does_not_consume_shuffle_generator():
    model = MNISTMLP()
    generator = torch.Generator().manual_seed(0)
    loader = _tiny_loader(shuffle=True, generator=generator)
    state_before = generator.get_state().clone()
    batch = torch.randn(4, 1, 28, 28)
    details = benchmark_inference(
        model,
        device=torch.device("cpu"),
        warmup=0,
        repeats=1,
        input_batch=batch,
        return_details=True,
    )
    assert torch.equal(generator.get_state(), state_before)
    assert details["batch_size"] == 4
    assert details["input_shape"] == (4, 1, 28, 28)
    with pytest.raises(ValueError, match="shuffle"):
        take_inference_batch(loader)


def test_early_stopping_does_not_extra_consume_train_shuffle_generator():
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)

    g_once = torch.Generator().manual_seed(0)
    loader_once = DataLoader(dataset, batch_size=4, shuffle=True, generator=g_once)
    for _ in loader_once:
        pass
    expected_state = g_once.get_state().clone()

    g_fit = torch.Generator().manual_seed(0)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=g_fit)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = MNISTMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fit_with_early_stopping(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        max_epochs=1,
        patience=5,
        min_delta=0.0,
        reevaluate_train=True,
    )
    assert torch.equal(g_fit.get_state(), expected_state)


def test_split_reproducibility():
    a = shuffled_index_splits(20, (10, 5, 5), seed=7)
    b = shuffled_index_splits(20, (10, 5, 5), seed=7)
    assert a == b


def test_fashion_mnist_existing_api():
    model = FashionMNISTCNN()
    assert model.conv1.in_channels == 1
    compressed = make_two_layer_svd_model(MNISTMLP(), 16, 8)
    x = torch.randn(2, 1, 28, 28)
    assert compressed(x).shape == (2, 10)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(images, labels)
    loaders = make_fashion_mnist_loaders(
        dataset,
        dataset,
        dataset,
        train_batch_size=4,
        validation_batch_size=4,
        test_batch_size=4,
        rank_validation_dataset=dataset,
    )
    assert set(loaders) >= {
        "train_loader",
        "validation_loader",
        "test_loader",
        "train_eval_loader",
        "validation_loader_rank",
    }
    assert not isinstance(loaders["train_eval_loader"].sampler, torch.utils.data.RandomSampler)


def test_fit_rejects_conflicting_train_eval_flags():
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = MNISTMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="train_eval_loader"):
        fit_with_early_stopping(
            model,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cpu"),
            max_epochs=1,
            patience=1,
            min_delta=0.0,
            reevaluate_train=False,
            train_eval_loader=eval_loader,
        )


def test_cifar_generic_rank_sweep_does_not_keep_models():
    torch.manual_seed(0)
    model = CIFAR10CNN()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    rows = sweep_conv_svd_ranks(
        model,
        "conv2",
        [4, 8],
        (16, 16),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.0,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
        verbose=False,
    )
    assert len(rows) == 2
    assert all("model" not in row for row in rows)
    assert {row["rank"] for row in rows} == {4, 8}


def test_one_layer_svd_legacy_alias_and_no_share():
    torch.manual_seed(0)
    model = MNISTMLP()
    compressed = make_one_layer_svd_model(model, r1=8, r2=8)
    assert compressed.fc3 is not model.fc3
    assert compressed.fc1.out_features == 512


def test_collect_metrics_optional_model():
    torch.manual_seed(0)
    model = NestedConv()
    compressed = make_two_layer_svd_model(MNISTMLP(), 4, 4)
    # NestedConv ではなく MLP で指標を取る
    loader = _tiny_loader()
    row = collect_compression_metrics(
        MNISTMLP(),
        compressed,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.0,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
    )
    assert "model" not in row
    assert "benchmark_batch_size" in row
