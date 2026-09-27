"""TT/MPS canonical form・center移動の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/04_gauge_freedom_qr_L2.ipynb
    notebooks/30_tt_mps/00_fundamentals/05_right_orthogonalization_right_block.ipynb
    notebooks/30_tt_mps/00_fundamentals/06_mixed_canonical_form_fixed.ipynb
    notebooks/30_tt_mps/00_fundamentals/07_orthogonality_center_qr_move.ipynb
    notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb
"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.compression import (
    tt_canonicality_errors,
    tt_canonicalize,
    tt_move_center,
    tt_reconstruct,
    tt_svd_exact,
)

SHAPE_4D = (2, 3, 2, 4)


def _tensor(seed: int, shape: tuple[int, ...] = SHAPE_4D, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _left_orthogonality_error(core: torch.Tensor) -> torch.Tensor:
    r_left, n_k, r_right = core.shape
    mat = core.reshape(r_left * n_k, r_right)
    gram = mat.T @ mat
    I = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.matrix_norm(gram - I, ord="fro")


def _right_orthogonality_error(core: torch.Tensor) -> torch.Tensor:
    r_left, n_k, r_right = core.shape
    mat = core.reshape(r_left, n_k * r_right)
    gram = mat @ mat.T
    I = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.matrix_norm(gram - I, ord="fro")


def _clone_cores(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [core.clone() for core in cores]


def _assert_cores_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


# ---------------------------------------------------------------------------
# tt_canonicalize: 再構成不変性・直交性・center norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_tt_canonicalize_preserves_reconstruction(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)

    canon = tt_canonicalize(cores, center)
    X_hat = tt_reconstruct(canon)

    assert X_hat.shape == X.shape
    assert torch.allclose(X_hat, X, atol=1e-10)


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_tt_canonicalize_left_and_right_orthogonality(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center)

    for k in range(center):
        assert _left_orthogonality_error(canon[k]).item() < 1e-8

    for k in range(center + 1, len(canon)):
        assert _right_orthogonality_error(canon[k]).item() < 1e-8


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_tt_canonicalize_center_norm_matches_global_norm(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center)

    global_norm = torch.linalg.vector_norm(tt_reconstruct(canon).reshape(-1))
    center_norm = torch.linalg.vector_norm(canon[center].reshape(-1))
    assert abs(global_norm.item() - center_norm.item()) < 1e-8


def test_tt_canonicalize_handles_d_equal_one():
    torch.manual_seed(0)
    core = torch.randn(1, 5, 1, dtype=torch.float64)
    cores = [core]

    canon = tt_canonicalize(cores, center=0)
    assert len(canon) == 1
    assert torch.equal(canon[0], core)
    assert canon[0] is not core  # 複製であること


def test_tt_canonicalize_does_not_mutate_input():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    saved = _clone_cores(cores)

    tt_canonicalize(cores, center=2)

    _assert_cores_equal(cores, saved)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_canonicalize_preserves_dtype_and_device(dtype):
    X = _tensor(seed=1, dtype=dtype)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=1)

    for core in canon:
        assert core.dtype == dtype
        assert core.device == cores[0].device


@pytest.mark.parametrize("center", [-1, 4, 10])
def test_tt_canonicalize_rejects_out_of_range_center(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(ValueError):
        tt_canonicalize(cores, center)


@pytest.mark.parametrize("center", [True, False])
def test_tt_canonicalize_rejects_bool_center(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_canonicalize(cores, center)


@pytest.mark.parametrize("center", [1.5, "1", None])
def test_tt_canonicalize_rejects_non_integer_center(center):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_canonicalize(cores, center)


# ---------------------------------------------------------------------------
# tt_move_center
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source,target", [(0, 3), (3, 0), (1, 2), (2, 1)])
def test_tt_move_center_preserves_reconstruction(source, target):
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=source)

    moved = tt_move_center(canon, source, target)
    X_hat = tt_reconstruct(moved)

    assert torch.allclose(X_hat, X, atol=1e-10)


@pytest.mark.parametrize("source,target", [(0, 3), (3, 0), (1, 2), (2, 1)])
def test_tt_move_center_produces_mixed_canonical_form_at_target(source, target):
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=source)

    moved = tt_move_center(canon, source, target)

    for k in range(target):
        assert _left_orthogonality_error(moved[k]).item() < 1e-8
    for k in range(target + 1, len(moved)):
        assert _right_orthogonality_error(moved[k]).item() < 1e-8


def test_tt_move_center_source_equal_target_returns_equivalent_clone():
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=1)

    moved = tt_move_center(canon, 1, 1)

    _assert_cores_equal(moved, canon)
    assert moved is not canon
    for m, c in zip(moved, canon):
        assert m is not c


def test_tt_move_center_handles_d_equal_one():
    torch.manual_seed(0)
    core = torch.randn(1, 5, 1, dtype=torch.float64)
    moved = tt_move_center([core], 0, 0)
    assert torch.equal(moved[0], core)


def test_tt_move_center_does_not_mutate_input():
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=0)
    saved = _clone_cores(canon)

    tt_move_center(canon, 0, 3)

    _assert_cores_equal(canon, saved)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_move_center_preserves_dtype_and_device(dtype):
    X = _tensor(seed=2, dtype=dtype)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=0)
    moved = tt_move_center(canon, 0, 3)

    for core in moved:
        assert core.dtype == dtype
        assert core.device == canon[0].device


@pytest.mark.parametrize("source,target", [(-1, 1), (1, -1), (10, 1), (1, 10)])
def test_tt_move_center_rejects_out_of_range_indices(source, target):
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=0)
    with pytest.raises(ValueError):
        tt_move_center(canon, source, target)


@pytest.mark.parametrize("value", [True, False])
def test_tt_move_center_rejects_bool_indices(value):
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=0)
    with pytest.raises(TypeError):
        tt_move_center(canon, value, 1)
    with pytest.raises(TypeError):
        tt_move_center(canon, 1, value)


@pytest.mark.parametrize("value", [1.5, "1", None])
def test_tt_move_center_rejects_non_integer_indices(value):
    X = _tensor(seed=2)
    cores = tt_svd_exact(X)
    canon = tt_canonicalize(cores, center=0)
    with pytest.raises(TypeError):
        tt_move_center(canon, value, 1)


# ---------------------------------------------------------------------------
# tt_canonicality_errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_tt_canonicality_errors_are_small_for_canonicalized_tt(center):
    X = _tensor(seed=13)
    canon = tt_canonicalize(tt_svd_exact(X), center=center)

    errors = tt_canonicality_errors(canon, center=center)

    assert len(errors["left"]) == center
    assert len(errors["right"]) == len(canon) - center - 1
    for error in errors["left"] + errors["right"]:
        assert error.item() < 1e-8
        assert error.dtype == X.dtype
        assert error.device == X.device


def test_tt_canonicality_errors_detect_noncanonical_core_without_mutation():
    X = _tensor(seed=14)
    canon = tt_canonicalize(tt_svd_exact(X), center=2)
    canon[0] = 2.0 * canon[0]
    saved = _clone_cores(canon)

    errors = tt_canonicality_errors(canon, center=2)

    assert errors["left"][0].item() > 1.0
    _assert_cores_equal(canon, saved)


def test_tt_canonicality_errors_for_single_core_are_empty():
    core = torch.randn(1, 5, 1, dtype=torch.float64)
    errors = tt_canonicality_errors([core], center=0)
    assert errors == {"left": [], "right": []}


def test_tt_canonicality_errors_preserves_autograd_graph():
    cores = [core.detach().requires_grad_() for core in tt_svd_exact(_tensor(seed=19))]

    errors = tt_canonicality_errors(cores, center=2)
    torch.stack(errors["left"] + errors["right"]).sum().backward()

    for index, core in enumerate(cores):
        if index == 2:
            # center coreは定義上、診断式へ含めない。
            assert core.grad is None
        else:
            assert core.grad is not None
            assert torch.isfinite(core.grad).all()


@pytest.mark.parametrize(
    "dtype,tiny",
    [(torch.float32, 1e-30), (torch.float64, 1e-200)],
)
@pytest.mark.parametrize("center,side", [(1, "left"), (0, "right")])
def test_tt_canonicality_errors_preserves_tiny_nonzero_error(dtype, tiny, center, side):
    """Gram差自体は表現可能だが、素朴な二乗ではunderflowする反例。"""
    perturbed = torch.tensor([[1.0, tiny], [0.0, 1.0]], dtype=dtype)
    identity = torch.eye(2, dtype=dtype)

    if side == "left":
        cores = [perturbed.reshape(1, 2, 2), identity.reshape(2, 2, 1)]
    else:
        cores = [identity.reshape(1, 2, 2), perturbed.reshape(2, 2, 1)]

    error = tt_canonicality_errors(cores, center=center)[side][0]

    assert error.item() > 0.0
    assert error.item() == pytest.approx(math.sqrt(2.0) * tiny, rel=1e-5)


@pytest.mark.parametrize("center", [-1, 4])
def test_tt_canonicality_errors_rejects_out_of_range_center(center):
    cores = tt_svd_exact(_tensor(seed=15))
    with pytest.raises(ValueError):
        tt_canonicality_errors(cores, center=center)


@pytest.mark.parametrize("center", [True, 1.5, "1", None])
def test_tt_canonicality_errors_rejects_non_integer_center(center):
    cores = tt_svd_exact(_tensor(seed=15))
    with pytest.raises(TypeError):
        tt_canonicality_errors(cores, center=center)
