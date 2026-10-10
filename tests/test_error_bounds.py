"""重み摂動による出力誤差上界（``metrics.error_bounds``）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/01_ttlinear_output_error_norm_bounds.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/02_relu_hidden_activation_error_todo.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/03_second_linear_logits_error_propagation.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb
"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.metrics import (
    LinearOutputErrorBounds,
    linear_output_error_bounds,
    mlp_logits_error_bound,
    mlp_sample_logits_error_bounds,
)

DTYPES = [torch.float32, torch.float64]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません"),
    ),
]


def _tol(dtype: torch.dtype) -> float:
    return 1e-4 if dtype == torch.float32 else 1e-10


def _randn(*shape, dtype=torch.float64, device="cpu", seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=dtype).to(device)


# ---------------------------------------------------------------------------
# linear_output_error_bounds
# ---------------------------------------------------------------------------


def test_linear_output_error_bounds_hand_computed_example():
    # X = I_2: ||X||_F = sqrt(2), ||X||_2 = 1。
    # delta_W = diag(3, 4): ||dW||_2 = 4, ||dW||_F = 5。
    X = torch.eye(2, dtype=torch.float64)
    delta_W = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))

    bounds = linear_output_error_bounds(X, delta_W)

    assert isinstance(bounds, LinearOutputErrorBounds)
    torch.testing.assert_close(
        bounds.frobenius_x_spectral_dw,
        torch.tensor(math.sqrt(2.0) * 4.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        bounds.spectral_x_frobenius_dw, torch.tensor(5.0, dtype=torch.float64)
    )
    # 実際の ||Delta_Y||_F = ||X dW^T||_F = 5。2番目の上界はこの例で tight。
    delta_Y = X @ delta_W.T
    assert float(torch.linalg.vector_norm(delta_Y)) <= float(bounds.spectral_x_frobenius_dw) + 1e-12


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_output_error_bounds_dominate_actual_error(dtype):
    X = _randn(7, 6, dtype=dtype, seed=1)
    delta_W = 0.1 * _randn(5, 6, dtype=dtype, seed=2)

    bounds = linear_output_error_bounds(X, delta_W)

    actual = torch.linalg.vector_norm(X @ delta_W.T)
    slack = 10 * torch.finfo(dtype).eps
    assert float(actual) <= float(bounds.frobenius_x_spectral_dw) * (1 + slack)
    assert float(actual) <= float(bounds.spectral_x_frobenius_dw) * (1 + slack)
    assert float(bounds.frobenius_x_spectral_dw) > 0
    assert float(bounds.spectral_x_frobenius_dw) > 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_output_error_bounds_match_norm_formulas(dtype):
    X = _randn(4, 3, dtype=dtype, seed=3)
    delta_W = _randn(2, 3, dtype=dtype, seed=4)

    bounds = linear_output_error_bounds(X, delta_W)

    expected_1 = torch.linalg.matrix_norm(X, ord="fro") * torch.linalg.matrix_norm(delta_W, ord=2)
    expected_2 = torch.linalg.matrix_norm(X, ord=2) * torch.linalg.matrix_norm(delta_W, ord="fro")
    torch.testing.assert_close(bounds.frobenius_x_spectral_dw, expected_1, rtol=_tol(dtype), atol=0)
    torch.testing.assert_close(bounds.spectral_x_frobenius_dw, expected_2, rtol=_tol(dtype), atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_output_error_bounds_keep_dtype_device_and_are_scalars(dtype, device):
    X = _randn(4, 3, dtype=dtype, device=device)
    delta_W = _randn(2, 3, dtype=dtype, device=device, seed=1)

    bounds = linear_output_error_bounds(X, delta_W)

    for bound in (bounds.frobenius_x_spectral_dw, bounds.spectral_x_frobenius_dw):
        assert bound.ndim == 0
        assert bound.dtype == dtype
        assert bound.device == X.device


def test_linear_output_error_bounds_zero_error():
    X = _randn(4, 3)
    bounds = linear_output_error_bounds(X, torch.zeros(2, 3, dtype=torch.float64))

    assert float(bounds.frobenius_x_spectral_dw) == 0.0
    assert float(bounds.spectral_x_frobenius_dw) == 0.0


def test_linear_output_error_bounds_do_not_mutate_inputs():
    X = _randn(4, 3)
    delta_W = _randn(2, 3, seed=1)
    X_saved, dW_saved = X.clone(), delta_W.clone()

    linear_output_error_bounds(X, delta_W)

    assert torch.equal(X, X_saved)
    assert torch.equal(delta_W, dW_saved)


def test_linear_output_error_bounds_are_differentiable_with_finite_gradients():
    X = _randn(4, 3).requires_grad_(True)
    delta_W = _randn(2, 3, seed=1).requires_grad_(True)

    bounds = linear_output_error_bounds(X, delta_W)
    assert bounds.frobenius_x_spectral_dw.requires_grad
    (bounds.frobenius_x_spectral_dw + bounds.spectral_x_frobenius_dw).backward()

    for tensor in (X, delta_W):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_linear_output_error_bounds_reject_shape_mismatch():
    with pytest.raises(ValueError):
        linear_output_error_bounds(_randn(4, 3), _randn(2, 4))  # in_features 不一致
    with pytest.raises(ValueError):
        linear_output_error_bounds(_randn(3), _randn(2, 3))  # X が1階
    with pytest.raises(ValueError):
        linear_output_error_bounds(_randn(4, 3), _randn(2, 3, 1))  # delta_W が3階
    with pytest.raises(ValueError):
        linear_output_error_bounds(_randn(4, 3), _randn(3))  # delta_W が1階


def test_linear_output_error_bounds_reject_empty_tensors():
    with pytest.raises(ValueError):
        linear_output_error_bounds(torch.zeros(0, 3, dtype=torch.float64), _randn(2, 3))


def test_linear_output_error_bounds_reject_dtype_problems():
    with pytest.raises(TypeError):
        linear_output_error_bounds(_randn(4, 3), _randn(2, 3, dtype=torch.float32))
    with pytest.raises(TypeError):
        linear_output_error_bounds(_randn(4, 3).half(), _randn(2, 3).half())
    with pytest.raises(TypeError):
        linear_output_error_bounds(torch.ones(4, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        linear_output_error_bounds([[1.0, 2.0]], _randn(2, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_linear_output_error_bounds_reject_device_mismatch():
    with pytest.raises(ValueError):
        linear_output_error_bounds(_randn(4, 3), _randn(2, 3, device="cuda"))


# ---------------------------------------------------------------------------
# mlp_logits_error_bound / mlp_sample_logits_error_bounds
# ---------------------------------------------------------------------------


def test_mlp_logits_error_bound_hand_computed_example():
    # x = (3, 4): ||x||_2 = 5。delta_W1 = 0.1 I: ||.||_2 = 0.1。W2 = diag(2, 1): ||.||_2 = 2。
    X = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    delta_W1 = 0.1 * torch.eye(2, dtype=torch.float64)
    W2 = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))

    batch_bound = mlp_logits_error_bound(X, delta_W1, W2)
    sample_bounds = mlp_sample_logits_error_bounds(X, delta_W1, W2)

    torch.testing.assert_close(batch_bound, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(sample_bounds, torch.tensor([1.0], dtype=torch.float64))


def test_mlp_bounds_distinguish_batch_and_sample_wise_shapes():
    X = torch.tensor([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]], dtype=torch.float64)
    delta_W1 = 0.1 * torch.eye(2, dtype=torch.float64)
    W2 = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))

    batch_bound = mlp_logits_error_bound(X, delta_W1, W2)
    sample_bounds = mlp_sample_logits_error_bounds(X, delta_W1, W2)

    assert batch_bound.shape == ()
    assert sample_bounds.shape == (3,)
    # ||x_n||_2 = (5, 0, 10) -> U_n = ||x_n|| * 0.1 * 2
    torch.testing.assert_close(
        sample_bounds, torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    )
    # batch 上界は sample 上界の ℓ2 合成: ||X||_F^2 = sum ||x_n||^2。
    torch.testing.assert_close(batch_bound, torch.linalg.vector_norm(sample_bounds))
    # batch 上界と sample 上界を混同しない: 小さい sample の U_n は batch 上界より小さい。
    assert float(sample_bounds[0]) < float(batch_bound)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mlp_bounds_dominate_actual_logits_error(dtype):
    X = _randn(9, 6, dtype=dtype, seed=1)
    W1 = _randn(5, 6, dtype=dtype, seed=2)
    b1 = _randn(5, dtype=dtype, seed=3)
    W2 = _randn(3, 5, dtype=dtype, seed=4)
    b2 = _randn(3, dtype=dtype, seed=5)
    delta_W1 = 0.05 * _randn(5, 6, dtype=dtype, seed=6)

    logits = torch.relu(X @ W1.T + b1) @ W2.T + b2
    logits_tilde = torch.relu(X @ (W1 + delta_W1).T + b1) @ W2.T + b2
    delta_L = logits_tilde - logits

    batch_bound = mlp_logits_error_bound(X, delta_W1, W2)
    sample_bounds = mlp_sample_logits_error_bounds(X, delta_W1, W2)

    slack = 10 * torch.finfo(dtype).eps
    assert float(torch.linalg.vector_norm(delta_L)) <= float(batch_bound) * (1 + slack)
    actual_per_sample = torch.linalg.vector_norm(delta_L, dim=1)
    assert bool((actual_per_sample <= sample_bounds * (1 + slack)).all())
    # ℓ∞ <= ℓ2 なので、argmax 判定に使う ℓ∞ 誤差も sample 上界以下。
    assert bool((delta_L.abs().amax(dim=1) <= sample_bounds * (1 + slack)).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mlp_bounds_keep_dtype_and_device(dtype, device):
    X = _randn(4, 3, dtype=dtype, device=device)
    delta_W1 = _randn(2, 3, dtype=dtype, device=device, seed=1)
    W2 = _randn(5, 2, dtype=dtype, device=device, seed=2)

    batch_bound = mlp_logits_error_bound(X, delta_W1, W2)
    sample_bounds = mlp_sample_logits_error_bounds(X, delta_W1, W2)

    assert batch_bound.dtype == dtype and batch_bound.device == X.device
    assert sample_bounds.dtype == dtype and sample_bounds.device == X.device


def test_mlp_bounds_zero_error():
    X = _randn(4, 3)
    W2 = _randn(5, 2, seed=1)
    zero = torch.zeros(2, 3, dtype=torch.float64)

    assert float(mlp_logits_error_bound(X, zero, W2)) == 0.0
    assert bool((mlp_sample_logits_error_bounds(X, zero, W2) == 0).all())


def test_mlp_bounds_do_not_mutate_inputs():
    X = _randn(4, 3)
    delta_W1 = _randn(2, 3, seed=1)
    W2 = _randn(5, 2, seed=2)
    saved = [t.clone() for t in (X, delta_W1, W2)]

    mlp_logits_error_bound(X, delta_W1, W2)
    mlp_sample_logits_error_bounds(X, delta_W1, W2)

    for tensor, original in zip((X, delta_W1, W2), saved):
        assert torch.equal(tensor, original)


@pytest.mark.parametrize("function", [mlp_logits_error_bound, mlp_sample_logits_error_bounds])
def test_mlp_bounds_are_differentiable_with_finite_gradients(function):
    X = _randn(4, 3).requires_grad_(True)
    delta_W1 = _randn(2, 3, seed=1).requires_grad_(True)
    W2 = _randn(5, 2, seed=2).requires_grad_(True)

    out = function(X, delta_W1, W2)
    assert out.requires_grad
    out.sum().backward()

    for tensor in (X, delta_W1, W2):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("function", [mlp_logits_error_bound, mlp_sample_logits_error_bounds])
def test_mlp_bounds_reject_shape_mismatch(function):
    X = _randn(4, 3)
    delta_W1 = _randn(2, 3, seed=1)
    W2 = _randn(5, 2, seed=2)

    with pytest.raises(ValueError):
        function(X, _randn(2, 4), W2)  # in_features 不一致
    with pytest.raises(ValueError):
        function(X, delta_W1, _randn(5, 3))  # hidden_features 不一致
    with pytest.raises(ValueError):
        function(X, delta_W1, _randn(2))  # W2 が1階
    with pytest.raises(ValueError):
        function(_randn(3), delta_W1, W2)  # X が1階


@pytest.mark.parametrize("function", [mlp_logits_error_bound, mlp_sample_logits_error_bounds])
def test_mlp_bounds_reject_dtype_problems(function):
    X = _randn(4, 3)
    delta_W1 = _randn(2, 3, seed=1)
    W2 = _randn(5, 2, seed=2)

    with pytest.raises(TypeError):
        function(X, delta_W1, W2.to(torch.float32))
    with pytest.raises(TypeError):
        function(X, delta_W1.to(torch.float32), W2)
    with pytest.raises(TypeError):
        function(X, delta_W1, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
@pytest.mark.parametrize("function", [mlp_logits_error_bound, mlp_sample_logits_error_bounds])
def test_mlp_bounds_reject_device_mismatch(function):
    with pytest.raises(ValueError):
        function(_randn(4, 3), _randn(2, 3, seed=1), _randn(5, 2, seed=2, device="cuda"))


# ---------------------------------------------------------------------------
# Bugbot: NaN / inf は SVD の _LinAlgError ではなく ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("which", ["X", "delta_W"])
def test_linear_output_error_bounds_reject_non_finite(bad, which):
    X = _randn(4, 3)
    delta_W = _randn(2, 3, seed=1)
    target = {"X": X, "delta_W": delta_W}[which]
    poisoned = target.clone()
    poisoned.view(-1)[0] = bad

    kwargs = {"X": X, "delta_W": delta_W}
    kwargs[which] = poisoned
    with pytest.raises(ValueError, match=rf"{which} に NaN / inf"):
        linear_output_error_bounds(kwargs["X"], kwargs["delta_W"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("which", ["X", "delta_W1", "W2"])
@pytest.mark.parametrize("function", [mlp_logits_error_bound, mlp_sample_logits_error_bounds])
def test_mlp_bounds_reject_non_finite(bad, which, function):
    values = {
        "X": _randn(4, 3),
        "delta_W1": _randn(2, 3, seed=1),
        "W2": _randn(5, 2, seed=2),
    }
    poisoned = values[which].clone()
    poisoned.view(-1)[0] = bad
    values[which] = poisoned

    with pytest.raises(ValueError, match=rf"{which} に NaN / inf"):
        function(values["X"], values["delta_W1"], values["W2"])


# ---------------------------------------------------------------------------
# 有限な極端入力でも inf * 0 を NaN にしない
# ---------------------------------------------------------------------------


def _assert_exact_scalar_zero(value: torch.Tensor, reference: torch.Tensor) -> None:
    assert value.ndim == 0
    assert value.shape == ()
    assert value.dtype == reference.dtype
    assert value.device == reference.device
    assert not bool(torch.isnan(value))
    assert float(value) == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_bounds_exact_zero_for_extreme_finite_partners(dtype, device):
    fmax = torch.finfo(dtype).max
    X = torch.full((3, 4), fmax, dtype=dtype, device=device)
    delta_W = torch.zeros((2, 4), dtype=dtype, device=device)
    X_saved, delta_saved = X.clone(), delta_W.clone()

    bounds = linear_output_error_bounds(X, delta_W)

    assert torch.equal(X, X_saved)
    assert torch.equal(delta_W, delta_saved)
    _assert_exact_scalar_zero(bounds.frobenius_x_spectral_dw, X)
    _assert_exact_scalar_zero(bounds.spectral_x_frobenius_dw, X)

    X_zero = torch.zeros((3, 4), dtype=dtype, device=device)
    delta_extreme = torch.full((2, 4), fmax, dtype=dtype, device=device)
    zero_input = linear_output_error_bounds(X_zero, delta_extreme)
    _assert_exact_scalar_zero(zero_input.frobenius_x_spectral_dw, X_zero)
    _assert_exact_scalar_zero(zero_input.spectral_x_frobenius_dw, X_zero)

    # どちらも非零でノルムが溢れるときは inf でよく、NaN にはしない。
    both_extreme = linear_output_error_bounds(X, delta_extreme)
    for field in (both_extreme.frobenius_x_spectral_dw, both_extreme.spectral_x_frobenius_dw):
        assert field.ndim == 0
        assert field.dtype == dtype
        assert field.device == X.device
        assert not bool(torch.isnan(field))
        assert bool(torch.isinf(field))

    X_grad = X.detach().requires_grad_(True)
    delta_grad = delta_W.detach().requires_grad_(True)
    connected = linear_output_error_bounds(X_grad, delta_grad)
    assert connected.frobenius_x_spectral_dw.requires_grad
    assert connected.spectral_x_frobenius_dw.requires_grad
    connected.frobenius_x_spectral_dw.backward()
    assert X_grad.grad is not None and delta_grad.grad is not None
    assert bool(torch.isfinite(X_grad.grad).all())
    assert bool(torch.isfinite(delta_grad.grad).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mlp_batch_bound_exact_zero_for_extreme_zero_factors(dtype, device):
    fmax = torch.finfo(dtype).max
    X = torch.full((4, 3), fmax, dtype=dtype, device=device)
    delta_zero = torch.zeros((2, 3), dtype=dtype, device=device)
    W2 = torch.full((5, 2), fmax, dtype=dtype, device=device)

    zero_delta = mlp_logits_error_bound(X, delta_zero, W2)
    _assert_exact_scalar_zero(zero_delta, X)

    delta_extreme = torch.full((2, 3), fmax, dtype=dtype, device=device)
    W2_zero = torch.zeros((5, 2), dtype=dtype, device=device)
    zero_w2 = mlp_logits_error_bound(X, delta_extreme, W2_zero)
    _assert_exact_scalar_zero(zero_w2, X)

    X_zero = torch.zeros((4, 3), dtype=dtype, device=device)
    zero_x = mlp_logits_error_bound(X_zero, delta_extreme, W2)
    _assert_exact_scalar_zero(zero_x, X_zero)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mlp_sample_bounds_zero_factors_and_mixed_extreme_rows(dtype, device):
    fmax = torch.finfo(dtype).max
    X_extreme = torch.full((3, 4), fmax, dtype=dtype, device=device)
    delta_zero = torch.zeros((2, 4), dtype=dtype, device=device)
    W2 = torch.eye(2, dtype=dtype, device=device)
    # hidden=2 に合わせる。eye(2) は (2, 2)。delta は (2, 4)。
    zero_delta = mlp_sample_logits_error_bounds(X_extreme, delta_zero, W2)
    assert zero_delta.shape == (3,)
    assert zero_delta.dtype == dtype
    assert zero_delta.device == X_extreme.device
    assert not bool(torch.isnan(zero_delta).any())
    assert bool(torch.equal(zero_delta, torch.zeros(3, dtype=dtype, device=device)))

    delta_extreme = torch.full((2, 4), fmax, dtype=dtype, device=device)
    W2_zero = torch.zeros((2, 2), dtype=dtype, device=device)
    zero_w2 = mlp_sample_logits_error_bounds(X_extreme, delta_extreme, W2_zero)
    assert zero_w2.shape == (3,)
    assert zero_w2.dtype == dtype
    assert zero_w2.device == X_extreme.device
    assert not bool(torch.isnan(zero_w2).any())
    assert bool(torch.equal(zero_w2, torch.zeros(3, dtype=dtype, device=device)))

    mixed = torch.zeros((2, 4), dtype=dtype, device=device)
    mixed[1] = fmax
    delta_finite = torch.zeros((2, 4), dtype=dtype, device=device)
    delta_finite[0, 0] = 1
    delta_finite[1, 1] = 1
    finite_factor = mlp_sample_logits_error_bounds(mixed, delta_finite, W2)
    assert finite_factor.shape == (2,)
    assert finite_factor.dtype == dtype
    assert finite_factor.device == mixed.device
    assert float(finite_factor[0]) == 0.0
    assert not bool(torch.isnan(finite_factor[1]))
    assert bool(torch.isinf(finite_factor[1]))

    rows = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    huge_delta = torch.full((2, 4), fmax, dtype=dtype, device=device)
    huge_w2 = torch.full((2, 2), fmax, dtype=dtype, device=device)
    huge_factor = mlp_sample_logits_error_bounds(rows, huge_delta, huge_w2)
    assert huge_factor.shape == (2,)
    assert huge_factor.dtype == dtype
    assert huge_factor.device == rows.device
    assert float(huge_factor[0]) == 0.0
    assert not bool(torch.isnan(huge_factor[1]))
    assert bool(torch.isinf(huge_factor[1]))

    rows_grad = rows.detach().requires_grad_(True)
    delta_grad = delta_zero.detach().requires_grad_(True)
    w2_grad = W2.detach().requires_grad_(True)
    connected = mlp_sample_logits_error_bounds(rows_grad, delta_grad, w2_grad)
    assert connected.requires_grad
    assert connected.shape == (2,)
    connected.sum().backward()
    assert rows_grad.grad is not None
    assert bool(torch.isfinite(rows_grad.grad).all())
