"""``metrics.tensor_comparison.tensor_approximation_metrics`` のテスト。"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from nn_compression.metrics import (
    TensorApproximationMetrics,
    relative_frobenius_error,
    tensor_approximation_metrics,
)
from nn_compression.metrics import tensor_comparison

DTYPES = [torch.float32, torch.float64]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _pair(shape, dtype, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
    approximation = (reference.to(torch.float64) + 0.1 * torch.randn(shape, generator=generator, dtype=torch.float64)).to(dtype)
    return reference.to(device), approximation.to(device)


# ---------------------------------------------------------------------------
# 値
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5), (2, 3, 4, 5)])
def test_metrics_match_float64_numpy_reference(shape, dtype, device):
    reference, approximation = _pair(shape, dtype, device)

    result = tensor_approximation_metrics(reference, approximation)

    ref = reference.double().cpu().numpy()
    approx = approximation.double().cpu().numpy()
    diff = ref - approx
    expected_abs = float(np.linalg.norm(diff.ravel()))
    assert result.absolute_frobenius_error == pytest.approx(expected_abs, rel=1e-12)
    assert result.relative_frobenius_error == pytest.approx(expected_abs / np.linalg.norm(ref.ravel()), rel=1e-12)
    assert result.rmse == pytest.approx(float(np.sqrt(np.mean(diff**2))), rel=1e-12)
    assert result.max_abs_error == pytest.approx(float(np.abs(diff).max()), rel=1e-12)


def test_identical_tensors_give_zero_errors():
    reference, _ = _pair((3, 4), torch.float64)

    result = tensor_approximation_metrics(reference, reference.clone())

    assert result == TensorApproximationMetrics(0.0, 0.0, 0.0, 0.0)


def test_scalar_and_single_element_tensors_are_supported():
    result = tensor_approximation_metrics(torch.tensor(3.0), torch.tensor(1.0))

    assert result.absolute_frobenius_error == pytest.approx(2.0)
    assert result.relative_frobenius_error == pytest.approx(2.0 / 3.0)
    assert result.rmse == pytest.approx(2.0)
    assert result.max_abs_error == pytest.approx(2.0)


def test_relative_error_reuses_existing_function(monkeypatch):
    reference, approximation = _pair((4, 4), torch.float64)
    calls = []
    original = tensor_comparison.relative_frobenius_error

    def spy(x, x_hat):
        calls.append((tuple(x.shape), tuple(x_hat.shape)))
        return original(x, x_hat)

    monkeypatch.setattr(tensor_comparison, "relative_frobenius_error", spy)

    result = tensor_approximation_metrics(reference, approximation)

    assert calls == [((4, 4), (4, 4))]
    assert result.relative_frobenius_error == pytest.approx(float(relative_frobenius_error(reference, approximation)), rel=1e-14)


def test_scaling_is_exact_for_normal_range_values():
    """2 のべき乗でのスケールは丸めを作らず、素朴な float64 計算と一致する。"""
    reference, approximation = _pair((16, 16), torch.float64, seed=3)
    expected_abs = float(torch.linalg.vector_norm(reference - approximation))
    expected_rel = float(relative_frobenius_error(reference, approximation))

    result = tensor_approximation_metrics(reference, approximation)

    assert result.absolute_frobenius_error == pytest.approx(expected_abs, rel=1e-13)
    assert result.relative_frobenius_error == pytest.approx(expected_rel, rel=1e-13)


# ---------------------------------------------------------------------------
# 零 reference の契約
# ---------------------------------------------------------------------------


def test_zero_reference_is_rejected_because_relative_error_is_undefined():
    with pytest.raises(ValueError, match="零"):
        tensor_approximation_metrics(torch.zeros(3, 3), torch.ones(3, 3))


def test_zero_reference_is_rejected_even_when_approximation_is_zero():
    with pytest.raises(ValueError, match="零"):
        tensor_approximation_metrics(torch.zeros(3, 3), torch.zeros(3, 3))


def test_zero_approximation_is_allowed_and_gives_relative_error_one():
    reference, _ = _pair((3, 3), torch.float64)

    result = tensor_approximation_metrics(reference, torch.zeros_like(reference))

    assert result.relative_frobenius_error == pytest.approx(1.0, rel=1e-14)
    assert result.absolute_frobenius_error == pytest.approx(float(torch.linalg.vector_norm(reference)), rel=1e-14)


# ---------------------------------------------------------------------------
# overflow
# ---------------------------------------------------------------------------


def test_float32_large_finite_values_do_not_overflow():
    """float32 で 1e30 の二乗（1e60）は float32 の範囲外だが、結果は有限になる。"""
    reference = torch.full((4, 4), 1e30, dtype=torch.float32)
    approximation = torch.full((4, 4), 2e30, dtype=torch.float32)

    result = tensor_approximation_metrics(reference, approximation)

    assert result.absolute_frobenius_error == pytest.approx(1e30 * 4.0, rel=1e-6)
    assert result.relative_frobenius_error == pytest.approx(1.0, rel=1e-6)
    assert result.rmse == pytest.approx(1e30, rel=1e-6)
    assert result.max_abs_error == pytest.approx(1e30, rel=1e-6)
    for value in dataclasses.astuple(result):
        assert math.isfinite(value)


def test_float64_large_finite_values_do_not_overflow():
    reference = torch.full((3, 3), 1e300, dtype=torch.float64)
    approximation = torch.full((3, 3), 3e300, dtype=torch.float64)

    result = tensor_approximation_metrics(reference, approximation)

    assert result.absolute_frobenius_error == pytest.approx(2e300 * 3.0, rel=1e-12)
    assert result.relative_frobenius_error == pytest.approx(2.0, rel=1e-12)
    assert result.rmse == pytest.approx(2e300, rel=1e-12)


def test_error_beyond_float64_range_is_inf_not_nan():
    big = torch.finfo(torch.float64).max
    reference = torch.tensor([big, big], dtype=torch.float64)
    approximation = torch.tensor([-big, -big], dtype=torch.float64)

    result = tensor_approximation_metrics(reference, approximation)

    assert result.absolute_frobenius_error == math.inf
    assert result.max_abs_error == math.inf
    assert not math.isnan(result.rmse)
    assert result.relative_frobenius_error == pytest.approx(2.0, rel=1e-12)


def test_tiny_values_keep_relative_precision():
    reference = torch.full((5,), 1e-30, dtype=torch.float64)
    approximation = reference * 1.5

    result = tensor_approximation_metrics(reference, approximation)

    assert result.relative_frobenius_error == pytest.approx(0.5, rel=1e-12)
    assert result.max_abs_error == pytest.approx(0.5e-30, rel=1e-12)


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        tensor_approximation_metrics(torch.ones(2, 3), torch.ones(3, 2))


def test_dtype_mismatch_is_rejected_without_implicit_cast():
    with pytest.raises(TypeError, match="dtype"):
        tensor_approximation_metrics(torch.ones(2, 3, dtype=torch.float32), torch.ones(2, 3, dtype=torch.float64))


def test_device_mismatch_is_rejected_without_implicit_transfer():
    with pytest.raises(ValueError, match="device"):
        tensor_approximation_metrics(torch.ones(2, 3), torch.ones(2, 3, device="meta"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_unsupported_dtype_is_rejected(dtype):
    tensor = torch.ones(2, 3).to(dtype)
    with pytest.raises(TypeError, match="dtype"):
        tensor_approximation_metrics(tensor, tensor.clone())


@pytest.mark.parametrize("bad", [[1.0, 2.0], np.ones(3), 1.0, None])
def test_non_tensor_inputs_are_rejected(bad):
    with pytest.raises(TypeError, match="torch.Tensor"):
        tensor_approximation_metrics(bad, torch.ones(2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        tensor_approximation_metrics(torch.ones(2), bad)


def test_empty_tensors_are_rejected():
    with pytest.raises(ValueError, match="空"):
        tensor_approximation_metrics(torch.empty(0, 3), torch.empty(0, 3))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected_and_the_argument_is_named(bad_value):
    good = torch.ones(2, 2, dtype=torch.float64)
    bad = good.clone()
    bad[1, 1] = bad_value

    with pytest.raises(ValueError, match="reference"):
        tensor_approximation_metrics(bad, good)
    with pytest.raises(ValueError, match="approximation"):
        tensor_approximation_metrics(good, bad)


# ---------------------------------------------------------------------------
# 入力非破壊・戻り値
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_inputs_are_not_modified(dtype):
    reference, approximation = _pair((5, 6), dtype)
    reference_before, approximation_before = reference.clone(), approximation.clone()

    tensor_approximation_metrics(reference, approximation)

    assert torch.equal(reference, reference_before)
    assert torch.equal(approximation, approximation_before)
    assert reference.dtype == dtype and approximation.dtype == dtype


def test_tensors_requiring_grad_are_accepted_and_get_no_gradient():
    reference, approximation = _pair((4, 4), torch.float64)
    reference.requires_grad_(True)
    approximation.requires_grad_(True)

    result = tensor_approximation_metrics(reference, approximation)

    assert all(isinstance(value, float) for value in dataclasses.astuple(result))
    assert reference.grad is None and approximation.grad is None


def test_result_is_a_frozen_dataclass_of_python_floats():
    reference, approximation = _pair((4, 4), torch.float32)

    result = tensor_approximation_metrics(reference, approximation)

    assert isinstance(result, TensorApproximationMetrics)
    assert all(type(value) is float for value in dataclasses.astuple(result))
    assert list(dataclasses.asdict(result)) == [
        "absolute_frobenius_error",
        "relative_frobenius_error",
        "rmse",
        "max_abs_error",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rmse = 0.0


def test_result_can_be_stored_in_a_dataframe_row():
    pd = pytest.importorskip("pandas")
    reference, approximation = _pair((4, 4), torch.float64)

    frame = pd.DataFrame([dataclasses.asdict(tensor_approximation_metrics(reference, approximation))])

    assert frame.shape == (1, 4)
    assert (frame.dtypes == "float64").all()


# ---------------------------------------------------------------------------
# スケール差が大きいときの相対誤差（P2-1）
# ---------------------------------------------------------------------------


def _assert_absolute_errors_are_not_nan(result):
    for value in (
        result.absolute_frobenius_error,
        result.rmse,
        result.max_abs_error,
    ):
        assert not math.isnan(value)


def test_tiny_nonzero_reference_against_huge_approximation_returns_inf_relative_error():
    """非零の小さい reference を、共通スケールで 0 と誤認しない。"""
    reference = torch.tensor([1e-300, 2e-300], dtype=torch.float64)
    approximation = torch.tensor([1e300, -1e300], dtype=torch.float64)
    reference_before, approximation_before = reference.clone(), approximation.clone()

    result = tensor_approximation_metrics(reference, approximation)

    assert result.relative_frobenius_error == math.inf
    assert not math.isnan(result.relative_frobenius_error)
    _assert_absolute_errors_are_not_nan(result)
    assert torch.equal(reference, reference_before)
    assert torch.equal(approximation, approximation_before)


def test_subnormal_nonzero_reference_is_not_treated_as_zero():
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    subnormal = torch.tensor([smallest.item(), 0.0], dtype=torch.float64)
    assert 0.0 < subnormal[0].item() < torch.finfo(torch.float64).tiny
    huge = torch.tensor([1e308, -1e308], dtype=torch.float64)

    result = tensor_approximation_metrics(subnormal, huge)

    assert result.relative_frobenius_error == math.inf
    _assert_absolute_errors_are_not_nan(result)

    # 同程度の subnormal 同士なら相対誤差は有限で、直接の定義と一致する。
    doubled = torch.tensor(
        [subnormal[0].item() * 2.0, 0.0],
        dtype=torch.float64,
    )
    close = tensor_approximation_metrics(subnormal, doubled)
    assert close.relative_frobenius_error == pytest.approx(1.0, rel=1e-12)
    assert math.isfinite(close.absolute_frobenius_error)


def test_large_but_representable_scale_gap_stays_finite_and_matches_definition():
    """二乗が underflow する小さい reference でも、比が float64 に収まるならその値を返す。"""
    reference = torch.tensor([1e-200], dtype=torch.float64)
    approximation = torch.tensor([1.0], dtype=torch.float64)

    result = tensor_approximation_metrics(reference, approximation)

    assert result.relative_frobenius_error == pytest.approx(1e200, rel=1e-9)
    assert result.absolute_frobenius_error == pytest.approx(1.0, rel=1e-12)
    assert result.max_abs_error == pytest.approx(1.0, rel=1e-12)
    _assert_absolute_errors_are_not_nan(result)


def test_exact_zero_reference_is_still_rejected_for_extreme_approximation():
    with pytest.raises(ValueError, match="零"):
        tensor_approximation_metrics(
            torch.zeros(2, dtype=torch.float64),
            torch.tensor([1e300, -1e300], dtype=torch.float64),
        )


def test_normal_values_match_direct_float64_computation():
    reference = torch.tensor([1.5, -2.0, 0.25, 4.0], dtype=torch.float64)
    approximation = torch.tensor([1.0, -1.0, 0.25, 3.5], dtype=torch.float64)
    difference = reference - approximation

    result = tensor_approximation_metrics(reference, approximation)

    assert result.relative_frobenius_error == pytest.approx(
        float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)),
        rel=0,
        abs=0,
    )
    assert result.absolute_frobenius_error == pytest.approx(float(torch.linalg.vector_norm(difference)))
    assert result.rmse == pytest.approx(float(torch.linalg.vector_norm(difference) / math.sqrt(difference.numel())))
    assert result.max_abs_error == pytest.approx(float(difference.abs().amax()))


def test_float32_moderate_values_keep_the_existing_relative_error():
    reference = torch.tensor([1e-3, 2e-3, -4e-3], dtype=torch.float32)
    approximation = torch.tensor([1.25e-3, 2e-3, -3e-3], dtype=torch.float32)
    reference64 = reference.double()
    approximation64 = approximation.double()
    expected = float(
        torch.linalg.vector_norm(reference64 - approximation64) / torch.linalg.vector_norm(reference64)
    )

    result = tensor_approximation_metrics(reference, approximation)

    assert result.relative_frobenius_error == pytest.approx(expected, rel=1e-6)
    assert all(math.isfinite(value) for value in dataclasses.astuple(result))
