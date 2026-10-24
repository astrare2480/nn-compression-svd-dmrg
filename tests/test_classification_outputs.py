"""``metrics.classification_outputs`` のテスト。"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.metrics import (
    ClassificationMetrics,
    LogitsComparison,
    agreement,
    classification_metrics_from_logits,
    collect_model_logits,
    compare_logits,
    logits_rmse,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _Net(nn.Module):
    """Dropout と BatchNorm を含む小さな分類器（training 状態の保存確認用）。"""

    def __init__(self, in_features=6, classes=4, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(in_features, 8)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, classes)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.bn(self.fc1(x)))))


def _data(n=10, in_features=6, classes=4, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_features, generator=generator)
    y = torch.randint(0, classes, (n,), generator=generator)
    return x, y


def _loader(x, y, batch_size=3):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# collect_model_logits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 4, 10, 64])
def test_collect_model_logits_keeps_loader_order_for_any_batch_size(batch_size):
    x, y = _data(10)
    model = _Net().eval()
    with torch.no_grad():
        expected = model(x)

    logits, labels = collect_model_logits(model, _loader(x, y, batch_size), "cpu")

    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(labels, y)
    assert logits.shape == (10, 4)


def test_collect_model_logits_is_row_aligned_between_two_models():
    x, y = _data(9)
    loader = _loader(x, y, 4)

    logits_a, labels_a = collect_model_logits(_Net(seed=0), loader, "cpu")
    logits_b, labels_b = collect_model_logits(_Net(seed=0), loader, "cpu")

    assert torch.equal(logits_a, logits_b)
    assert torch.equal(labels_a, labels_b)


def test_collect_model_logits_runs_in_eval_mode_during_forward():
    x, y = _data(6)
    model = _Net().train()
    seen = []
    model.drop.register_forward_hook(lambda module, args, output: seen.append(module.training))
    model.bn.register_forward_hook(lambda module, args, output: seen.append(module.training))

    collect_model_logits(model, _loader(x, y, 3), "cpu")

    assert seen and not any(seen)


def test_collect_model_logits_is_deterministic_despite_dropout_in_train_mode():
    x, y = _data(8)
    model = _Net().train()

    first, _ = collect_model_logits(model, _loader(x, y, 3), "cpu")
    second, _ = collect_model_logits(model, _loader(x, y, 3), "cpu")

    assert torch.equal(first, second)


def test_collect_model_logits_restores_root_training_state():
    x, y = _data(6)
    for initial in (True, False):
        model = _Net().train(initial)
        collect_model_logits(model, _loader(x, y), "cpu")
        assert model.training is initial
        assert all(module.training is initial for module in model.modules())


def test_collect_model_logits_restores_per_submodule_training_state():
    x, y = _data(6)
    model = _Net().train()
    model.bn.eval()  # 凍結 BatchNorm 相当（root は train のまま）
    before = {name: module.training for name, module in model.named_modules()}

    collect_model_logits(model, _loader(x, y), "cpu")

    assert {name: module.training for name, module in model.named_modules()} == before
    assert model.training is True and model.bn.training is False


def test_collect_model_logits_restores_training_state_when_forward_raises():
    x, y = _data(6)
    model = _Net().train()
    model.bn.eval()
    model.drop.register_forward_hook(lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        collect_model_logits(model, _loader(x, y), "cpu")

    assert model.training is True and model.bn.training is False and model.drop.training is True


def test_collect_model_logits_does_not_modify_model_or_batch_norm_statistics():
    x, y = _data(6)
    model = _Net().train()
    state_before = copy.deepcopy(model.state_dict())
    flags_before = {name: p.requires_grad for name, p in model.named_parameters()}

    collect_model_logits(model, _loader(x, y), "cpu")

    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name]), name  # running_mean などの buffer を含む
    assert {name: p.requires_grad for name, p in model.named_parameters()} == flags_before
    assert all(p.grad is None for p in model.parameters())


def test_collect_model_logits_returns_ordinary_tensors_without_graph():
    x, y = _data(6)

    logits, labels = collect_model_logits(_Net(), _loader(x, y), "cpu")

    assert not logits.is_inference() and not labels.is_inference()
    assert not logits.requires_grad
    logits.add_(1.0)  # inference tensor なら inference_mode の外では in-place 更新できない


def test_collect_model_logits_does_not_pass_labels_to_model_and_keeps_label_dtype():
    x, _ = _data(6)
    labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int32)
    received = []

    class _Recorder(nn.Module):
        def forward(self, inputs):
            received.append(inputs.shape)
            return inputs[:, :4]

    _, collected = collect_model_logits(_Recorder(), _loader(x, labels, 4), "cpu")

    assert received == [(4, 6), (2, 6)]
    assert collected.dtype == torch.int32
    assert torch.equal(collected, labels)


def test_collect_model_logits_does_not_modify_input_data():
    x, y = _data(6)
    x_before, y_before = x.clone(), y.clone()

    collect_model_logits(_Net(), _loader(x, y), "cpu")

    assert torch.equal(x, x_before) and torch.equal(y, y_before)


def test_collect_model_logits_accepts_list_batches_and_str_or_device_arguments():
    x, y = _data(4)
    batches = [[x[:2], y[:2]], (x[2:], y[2:])]

    logits, labels = collect_model_logits(
        _Net(), batches, torch.device("cpu"), output_device=torch.device("cpu")
    )

    assert logits.shape == (4, 4) and torch.equal(labels, y)


def test_collect_model_logits_rejects_empty_loader():
    with pytest.raises(ValueError, match="空"):
        collect_model_logits(_Net(), [], "cpu")
    empty = DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long)), batch_size=2)
    with pytest.raises(ValueError, match="空"):
        collect_model_logits(_Net(), empty, "cpu")


def test_collect_model_logits_restores_training_state_for_empty_loader():
    model = _Net().train()
    with pytest.raises(ValueError):
        collect_model_logits(model, [], "cpu")
    assert model.training is True


@pytest.mark.parametrize(
    "batch",
    [torch.zeros(2, 6), (torch.zeros(2, 6),), (torch.zeros(2, 6), torch.zeros(2), torch.zeros(2)), {"x": 1}],
)
def test_collect_model_logits_rejects_batches_that_are_not_input_label_pairs(batch):
    with pytest.raises(TypeError, match="inputs, labels"):
        collect_model_logits(_Net(), [batch], "cpu")


def test_collect_model_logits_rejects_non_tensor_batch_items():
    with pytest.raises(TypeError, match="torch.Tensor"):
        collect_model_logits(_Net(), [(torch.zeros(2, 6), [0, 1])], "cpu")


def test_collect_model_logits_rejects_non_module():
    with pytest.raises(TypeError, match="nn.Module"):
        collect_model_logits(lambda x: x, [], "cpu")


def test_collect_model_logits_rejects_non_two_dimensional_output():
    x, y = _data(4)
    with pytest.raises(ValueError, match="num_classes"):
        collect_model_logits(nn.Flatten(start_dim=0), [(x, y)], "cpu")
    with pytest.raises(ValueError, match="num_classes"):
        collect_model_logits(nn.Identity(), [(x.reshape(4, 2, 3), y)], "cpu")


def test_collect_model_logits_rejects_label_count_mismatch():
    x, y = _data(4)
    with pytest.raises(ValueError, match="labels"):
        collect_model_logits(_Net(), [(x, y[:3])], "cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_collect_model_logits_device_arguments(device):
    x, y = _data(6)
    model = _Net().to(device)

    logits, labels = collect_model_logits(model, _loader(x, y), device)
    assert logits.device.type == "cpu" and labels.device.type == "cpu"  # 既定は CPU

    logits_on_device, labels_on_device = collect_model_logits(
        model, _loader(x, y), device, output_device=device
    )
    assert logits_on_device.device.type == torch.device(device).type
    assert labels_on_device.device.type == torch.device(device).type
    torch.testing.assert_close(logits_on_device.cpu(), logits, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# classification_metrics_from_logits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_classification_metrics_match_reference_computation(dtype, device):
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(11, 5, generator=generator, dtype=torch.float64).to(dtype).to(device)
    labels = torch.randint(0, 5, (11,), generator=generator).to(device)

    result = classification_metrics_from_logits(logits, labels)

    expected_loss = float(F.cross_entropy(logits.double().cpu(), labels.cpu()))
    expected_accuracy = float((logits.argmax(dim=1) == labels).double().mean())
    assert result.loss == pytest.approx(expected_loss, rel=1e-12)
    assert result.accuracy == pytest.approx(expected_accuracy, abs=1e-15)
    assert result.num_samples == 11


def test_classification_metrics_known_values():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1])

    result = classification_metrics_from_logits(logits, labels)

    loss_first = math.log1p(math.exp(-2.0))
    loss_second = math.log1p(math.exp(2.0))
    loss_third = math.log(2.0)
    assert result.loss == pytest.approx((loss_first + loss_second + loss_third) / 3.0, rel=1e-12)
    # argmax: [0, 1, 0]（同点は先頭）→ 正解は1件目のみ
    assert result.accuracy == pytest.approx(1.0 / 3.0)


def test_classification_metrics_loss_is_sample_mean_independent_of_chunking():
    x, y = _data(10)
    logits, labels = collect_model_logits(_Net(), _loader(x, y, 3), "cpu")

    whole = classification_metrics_from_logits(logits, labels)
    parts = [
        classification_metrics_from_logits(logits[:6], labels[:6]),
        classification_metrics_from_logits(logits[6:], labels[6:]),
    ]

    weighted = sum(part.loss * part.num_samples for part in parts) / 10
    assert whole.loss == pytest.approx(weighted, rel=1e-12)


@pytest.mark.parametrize("label_dtype", [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64])
def test_classification_metrics_accept_integer_label_dtypes_without_modifying_them(label_dtype):
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1]).to(label_dtype)
    labels_before = labels.clone()

    classification_metrics_from_logits(logits, labels)

    assert labels.dtype == label_dtype and torch.equal(labels, labels_before)


def test_classification_metrics_large_logits_stay_finite():
    logits = torch.tensor([[1e30, 0.0], [0.0, 1e30]], dtype=torch.float32)
    labels = torch.tensor([0, 1])

    result = classification_metrics_from_logits(logits, labels)

    assert result.loss == 0.0 and result.accuracy == 1.0


def test_classification_metrics_return_python_numbers_in_a_frozen_dataclass():
    result = classification_metrics_from_logits(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))

    assert isinstance(result, ClassificationMetrics)
    assert type(result.loss) is float and type(result.accuracy) is float
    assert type(result.num_samples) is int
    assert list(dataclasses.asdict(result)) == ["loss", "accuracy", "num_samples"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.loss = 0.0


def test_classification_metrics_do_not_modify_inputs_and_ignore_grad():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    logits_before = logits.detach().clone()

    classification_metrics_from_logits(logits, labels)

    assert torch.equal(logits.detach(), logits_before) and logits.grad is None


def test_classification_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="空"):
        classification_metrics_from_logits(torch.empty(0, 3), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="空"):
        classification_metrics_from_logits(torch.empty(2, 0), torch.tensor([0, 0]))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_classification_metrics_reject_non_finite_logits(bad_value):
    logits = torch.zeros(3, 3)
    logits[1, 1] = bad_value
    with pytest.raises(ValueError, match="NaN / inf"):
        classification_metrics_from_logits(logits, torch.tensor([0, 1, 2]))


@pytest.mark.parametrize("labels", [torch.tensor([0, 1, 3]), torch.tensor([-1, 0, 1]), torch.tensor([0, 1, 100])])
def test_classification_metrics_reject_out_of_range_labels(labels):
    with pytest.raises(ValueError, match="0 以上 3 未満"):
        classification_metrics_from_logits(torch.zeros(3, 3), labels)


def test_classification_metrics_reject_invalid_shapes():
    with pytest.raises(ValueError, match="2階"):
        classification_metrics_from_logits(torch.zeros(3), torch.tensor([0, 0, 0]))
    with pytest.raises(ValueError, match="2階"):
        classification_metrics_from_logits(torch.zeros(2, 3, 4), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="labels"):
        classification_metrics_from_logits(torch.zeros(3, 3), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="labels"):
        classification_metrics_from_logits(torch.zeros(3, 3), torch.tensor([[0], [1], [2]]))


def test_classification_metrics_reject_invalid_types():
    with pytest.raises(TypeError, match="torch.Tensor"):
        classification_metrics_from_logits(np.zeros((3, 3), dtype=np.float32), torch.tensor([0, 1, 2]))
    with pytest.raises(TypeError, match="torch.Tensor"):
        classification_metrics_from_logits(torch.zeros(3, 3), [0, 1, 2])
    with pytest.raises(TypeError, match="dtype"):
        classification_metrics_from_logits(torch.zeros(3, 3, dtype=torch.float16), torch.tensor([0, 1, 2]))
    for bad_dtype in (torch.bool, torch.float32):
        with pytest.raises(TypeError, match="整数"):
            classification_metrics_from_logits(torch.zeros(3, 3), torch.tensor([0, 1, 1]).to(bad_dtype))


def test_classification_metrics_reject_device_mismatch():
    with pytest.raises(ValueError, match="device"):
        classification_metrics_from_logits(torch.zeros(3, 3), torch.tensor([0, 1, 2], device="meta"))


# ---------------------------------------------------------------------------
# compare_logits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_compare_logits_matches_reference_computation(dtype, device):
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(12, 5, generator=generator, dtype=torch.float64).to(dtype)
    candidate = (reference.double() + 0.3 * torch.randn(12, 5, generator=generator, dtype=torch.float64)).to(dtype)

    result = compare_logits(reference.to(device), candidate.to(device))

    diff = reference.double().numpy() - candidate.double().numpy()
    expected_agreement = float(np.mean(reference.argmax(1).numpy() == candidate.argmax(1).numpy()))
    assert result.prediction_agreement == pytest.approx(expected_agreement, abs=1e-15)
    assert result.logits_rmse == pytest.approx(float(np.sqrt(np.mean(diff**2))), rel=1e-12)
    assert result.logits_max_abs_error == pytest.approx(float(np.abs(diff).max()), rel=1e-12)


def test_compare_logits_matches_existing_model_loader_functions():
    """既存の ``agreement`` / ``logits_rmse`` と同じ意味の値になる。"""
    x, y = _data(13)
    loader = _loader(x, y, 4)
    baseline = _Net(seed=0)
    perturbed = copy.deepcopy(baseline)
    with torch.no_grad():
        perturbed.fc2.weight.add_(0.4 * torch.randn_like(perturbed.fc2.weight))

    reference_logits, _ = collect_model_logits(baseline, loader, "cpu")
    candidate_logits, _ = collect_model_logits(perturbed, loader, "cpu")
    result = compare_logits(reference_logits, candidate_logits)

    assert result.prediction_agreement == pytest.approx(agreement(baseline, perturbed, loader, "cpu"), abs=1e-15)
    assert result.logits_rmse == pytest.approx(logits_rmse(baseline, perturbed, loader, "cpu"), rel=1e-5)


def test_compare_logits_identical_inputs():
    logits = torch.randn(5, 3)

    result = compare_logits(logits, logits.clone())

    assert result == LogitsComparison(1.0, 0.0, 0.0)


def test_compare_logits_agreement_counts_rows_and_uses_first_max_on_ties():
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    candidate = torch.tensor([[1.0, 0.0], [1.0, 0.0], [5.0, 5.0], [0.0, 3.0]])

    result = compare_logits(reference, candidate)

    # 行ごとの argmax: ref=[0,1,0,0], cand=[0,0,0,1] → 一致は 0・2 行目
    assert result.prediction_agreement == pytest.approx(0.5)


def test_compare_logits_allows_zero_reference():
    result = compare_logits(torch.zeros(3, 4), torch.ones(3, 4))

    assert result.logits_rmse == pytest.approx(1.0)
    assert result.logits_max_abs_error == pytest.approx(1.0)
    assert result.prediction_agreement == 1.0  # どちらも全同点 → argmax は 0


def test_compare_logits_large_values_do_not_overflow():
    reference = torch.full((3, 4), 1e30, dtype=torch.float32)
    candidate = torch.full((3, 4), 2e30, dtype=torch.float32)

    result = compare_logits(reference, candidate)

    assert result.logits_rmse == pytest.approx(1e30, rel=1e-6)
    assert result.logits_max_abs_error == pytest.approx(1e30, rel=1e-6)


def test_compare_logits_does_not_modify_inputs_and_returns_python_numbers():
    reference, candidate = torch.randn(5, 3), torch.randn(5, 3)
    reference_before, candidate_before = reference.clone(), candidate.clone()

    result = compare_logits(reference, candidate)

    assert torch.equal(reference, reference_before) and torch.equal(candidate, candidate_before)
    assert all(type(value) is float for value in dataclasses.astuple(result))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.logits_rmse = 0.0


def test_compare_logits_rejects_shape_dtype_device_mismatch():
    with pytest.raises(ValueError, match="shape"):
        compare_logits(torch.zeros(3, 4), torch.zeros(3, 5))
    with pytest.raises(ValueError, match="shape"):
        compare_logits(torch.zeros(3, 4), torch.zeros(2, 4))
    with pytest.raises(TypeError, match="dtype"):
        compare_logits(torch.zeros(3, 4, dtype=torch.float32), torch.zeros(3, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="device"):
        compare_logits(torch.zeros(3, 4), torch.zeros(3, 4, device="meta"))


def test_compare_logits_rejects_non_two_dimensional_empty_and_non_finite_inputs():
    with pytest.raises(ValueError, match="2階"):
        compare_logits(torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="2階"):
        compare_logits(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="空"):
        compare_logits(torch.zeros(0, 3), torch.zeros(0, 3))
    bad = torch.zeros(3, 3)
    bad[0, 0] = math.nan
    with pytest.raises(ValueError, match="reference_logits"):
        compare_logits(bad, torch.zeros(3, 3))
    bad[0, 0] = math.inf
    with pytest.raises(ValueError, match="candidate_logits"):
        compare_logits(torch.zeros(3, 3), bad)


def test_compare_logits_rejects_non_tensor_and_unsupported_dtype():
    with pytest.raises(TypeError, match="torch.Tensor"):
        compare_logits([[1.0]], torch.zeros(1, 1))
    with pytest.raises(TypeError, match="dtype"):
        compare_logits(torch.zeros(2, 2, dtype=torch.float16), torch.zeros(2, 2, dtype=torch.float16))
