"""全 TT core を同時に更新する全データ GD を、小さい合成データで検証する。"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import factorize_named_tt_linear
from nn_compression.training.gpu_resident_batches import GpuResidentTensorBatches
from nn_compression.training.tt_linear_full_batch_sweep import NonFiniteLocalUpdate
from nn_compression.training import tt_linear_global_full_batch as global_update
from nn_compression.tt_mps import tt_matrix_ranks

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


def _model():
    torch.manual_seed(0)
    model = factorize_named_tt_linear(TinyMLP(), "fc1", OUT_MODES, IN_MODES, max_rank=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _data(n_samples=7, batch_size=4):
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(n_samples, 8, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (n_samples,), generator=generator)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    return features, labels, loader


def _manual_gradients(model, features, labels):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    named = list(model.named_parameters())
    saved = {name: parameter.requires_grad for name, parameter in named}
    for _, parameter in named:
        parameter.requires_grad_(True)
        parameter.grad = None
    total = features.shape[0]
    for start in range(0, total, 4):
        end = min(start + 4, total)
        loss = criterion(model(features[start:end]), labels[start:end])
        (loss * ((end - start) / total)).backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(saved[name])
        parameter.grad = None
    return grads


def test_module_does_not_implement_pair_svd():
    tree = ast.parse(inspect.getsource(global_update))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        if isinstance(node, ast.Name):
            names.append(node.id)
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
    referenced = " ".join(names)
    assert "split_merged_tt_matrix" not in referenced
    assert "merge_tt_matrix_cores" not in referenced
    assert "tt_linear_scoped_update" not in referenced


def test_one_step_updates_every_parameter_and_keeps_shapes():
    model = _model()
    features, labels, loader = _data()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    shapes = {name: tuple(parameter.shape) for name, parameter in before.items()}
    ranks = tuple(tt_matrix_ranks(list(model.fc1.cores)))
    result = global_update.update_tt_linear_global_full_batch(
        model,
        loader,
        layer_name="fc1",
        learning_rate=0.1,
        batch_size=None,
        device=torch.device("cpu"),
    )
    assert result.processed_samples == features.shape[0]
    assert result.core_shapes == tuple(shapes[name] for name, _ in model.named_parameters() if name.startswith("fc1.cores."))
    assert tuple(tt_matrix_ranks(list(model.fc1.cores))) == ranks
    for name, parameter in model.named_parameters():
        assert tuple(parameter.shape) == shapes[name]
        assert not torch.equal(parameter, before[name])
        assert parameter.requires_grad is False
    assert math_finite(result.gradient_norm)
    assert math_finite(result.relative_update)
    assert result.relative_update == pytest.approx(
        0.1 * result.gradient_norm / result.parameter_norm
    )


def math_finite(value):
    return value == value and value not in {float("inf"), float("-inf")}


def test_gradient_matches_manual_full_batch_mean():
    model = _model()
    features, labels, loader = _data()
    manual = _manual_gradients(model, features, labels)
    grads, loss, seen = global_update.accumulate_tt_linear_global_full_batch_gradients(
        model,
        loader,
        layer_name="fc1",
        batch_size=None,
        device=torch.device("cpu"),
    )
    assert seen == features.shape[0]
    assert math_finite(loss)
    for name, gradient in manual.items():
        assert torch.allclose(grads[name], gradient, rtol=1e-8, atol=1e-8)
    resident = GpuResidentTensorBatches(features, labels)
    resident_grads, resident_loss, resident_seen = global_update.accumulate_tt_linear_global_full_batch_gradients(
        model,
        resident,
        layer_name="fc1",
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert resident_seen == seen
    assert resident_loss == pytest.approx(loss)
    for name, gradient in grads.items():
        assert torch.allclose(resident_grads[name], gradient, rtol=1e-8, atol=1e-8)


def test_zero_learning_rate_does_not_change_values():
    model = _model()
    _, _, loader = _data()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    global_update.update_tt_linear_global_full_batch(
        model,
        loader,
        layer_name="fc1",
        learning_rate=0.0,
        batch_size=None,
        device=torch.device("cpu"),
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_non_finite_input_does_not_write():
    class NanLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.reduction = "mean"

        def forward(self, logits, labels):
            return torch.full((), float("nan"), dtype=logits.dtype)

    model = _model()
    features, labels, loader = _data()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(NonFiniteLocalUpdate):
        global_update.update_tt_linear_global_full_batch(
            model,
            loader,
            layer_name="fc1",
            learning_rate=0.1,
            batch_size=None,
            device=torch.device("cpu"),
            criterion=NanLoss(),
        )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    assert features.shape[0] == labels.shape[0]


class _InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction):
        ctx.shape = prediction.shape
        return prediction.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, float("inf"), dtype=grad_output.dtype, device=grad_output.device)


class InfiniteGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "mean"

    def forward(self, logits, labels):
        return _InfiniteGradient.apply(logits)


class _HugeFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction):
        ctx.shape = prediction.shape
        return prediction.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, 1e308, dtype=grad_output.dtype, device=grad_output.device)


class HugeFiniteGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "mean"

    def forward(self, logits, labels):
        return _HugeFiniteGradient.apply(logits)


def test_parameters_hold_until_accumulation_finishes_and_chunking_matches():
    """不均等な chunk でも、一括の平均勾配および 1 step 後の parameter と一致する。"""
    model = _model()
    features, labels, _loader = _data(n_samples=7, batch_size=4)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    full_loader = DataLoader(TensorDataset(features, labels), batch_size=7, shuffle=False)
    chunk_loader = DataLoader(TensorDataset(features, labels), batch_size=3, shuffle=False)

    class _Guard(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.reduction = "mean"

        def forward(self, logits, labels):
            self.calls += 1
            for name, parameter in model.named_parameters():
                assert torch.equal(parameter, before[name])
            return nn.functional.cross_entropy(logits, labels, reduction="mean")

    guard = _Guard()
    chunk_grads, chunk_loss, chunk_seen = global_update.accumulate_tt_linear_global_full_batch_gradients(
        model, chunk_loader, layer_name="fc1", batch_size=3, device=torch.device("cpu"), criterion=guard,
    )
    assert guard.calls == 3
    assert chunk_seen == 7
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    full_grads, full_loss, full_seen = global_update.accumulate_tt_linear_global_full_batch_gradients(
        model, full_loader, layer_name="fc1", batch_size=7, device=torch.device("cpu"),
    )
    assert full_seen == 7
    assert full_loss == pytest.approx(chunk_loss)
    for name, gradient in full_grads.items():
        assert torch.allclose(chunk_grads[name], gradient, rtol=1e-8, atol=1e-8)

    full_model = _model()
    chunk_model = _model()
    global_update.update_tt_linear_global_full_batch(
        full_model, full_loader, layer_name="fc1", learning_rate=0.1, batch_size=7, device=torch.device("cpu"),
    )
    global_update.update_tt_linear_global_full_batch(
        chunk_model, chunk_loader, layer_name="fc1", learning_rate=0.1, batch_size=3, device=torch.device("cpu"),
    )
    for name, parameter in full_model.named_parameters():
        other = dict(chunk_model.named_parameters())[name]
        assert torch.allclose(parameter, other, rtol=1e-8, atol=1e-8)


def test_non_finite_gradient_or_candidate_does_not_write():
    _, _, loader = _data()
    model = _model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(NonFiniteLocalUpdate):
        global_update.update_tt_linear_global_full_batch(
            model, loader, layer_name="fc1", learning_rate=0.1, batch_size=None,
            device=torch.device("cpu"), criterion=InfiniteGradientLoss(),
        )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    with pytest.raises(NonFiniteLocalUpdate):
        global_update.update_tt_linear_global_full_batch(
            model, loader, layer_name="fc1", learning_rate=10.0, batch_size=None,
            device=torch.device("cpu"), criterion=HugeFiniteGradientLoss(),
        )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
