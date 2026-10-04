"""TT-matrix（dense reconstruction・tensorization・TT-Linear forward）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb
    notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nn_compression.compression import (
    dense_to_tt_matrix_cores,
    dense_to_tt_matrix_tensor,
    tt_cores_to_tt_matrix_cores,
    tt_linear_forward,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
import nn_compression.compression.tt_matrix as tt_matrix_module

# ---------------------------------------------------------------------------
# テスト用ヘルパー（src化しない、本テストファイル専用）
# ---------------------------------------------------------------------------


def _random_tt_matrix_cores(
    out_modes: tuple[int, ...],
    in_modes: tuple[int, ...],
    ranks: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    seed: int = 0,
) -> list[torch.Tensor]:
    """検証用のランダムTT-matrix core列を作る。"""
    assert len(ranks) == len(out_modes) + 1
    assert ranks[0] == 1
    assert ranks[-1] == 1
    assert len(out_modes) == len(in_modes)

    torch.manual_seed(seed)
    cores = []
    for k, (m_k, n_k) in enumerate(zip(out_modes, in_modes)):
        cores.append(
            torch.randn(ranks[k], m_k, n_k, ranks[k + 1], dtype=dtype, device=device)
        )
    return cores


def _clone_cores(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [core.clone() for core in cores]


def _assert_cores_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


OUT_MODES_3 = (2, 3, 2)
IN_MODES_3 = (2, 2, 3)
RANKS_3 = (1, 2, 3, 1)


# ---------------------------------------------------------------------------
# tensorization・復元
# ---------------------------------------------------------------------------


def test_tt_matrix_to_dense_shape_d1():
    out_modes, in_modes = (5,), (4,)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 1), seed=1)
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W.shape == (5, 4)


def test_tt_matrix_to_dense_shape_d2():
    out_modes, in_modes = (2, 3), (3, 2)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 4, 1), seed=2)
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W.shape == (6, 6)


def test_tt_matrix_to_dense_shape_d3():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=3)
    W = tt_matrix_to_dense(cores, OUT_MODES_3, IN_MODES_3)
    assert W.shape == (12, 12)


def test_tt_matrix_to_dense_non_square_weight():
    out_modes, in_modes = (2, 2), (5, 3)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 2, 1), seed=4)
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W.shape == (4, 15)


def test_tt_matrix_to_dense_different_modes_per_site():
    out_modes, in_modes = (2, 4, 3), (5, 2, 2)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 2, 3, 1), seed=5)
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W.shape == (24, 20)


def test_tt_matrix_to_dense_matches_explicit_bond_contraction_element():
    """denseの1要素と、明示的なbond積（Notebook 14 TODO3相当）を比較する。"""
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=6)
    W = tt_matrix_to_dense(cores, OUT_MODES_3, IN_MODES_3)

    i1, i2, i3 = 1, 0, 1
    j1, j2, j3 = 0, 1, 2
    m1, m2, m3 = OUT_MODES_3
    n1, n2, n3 = IN_MODES_3
    i = i1 * (m2 * m3) + i2 * m3 + i3
    j = j1 * (n2 * n3) + j2 * n3 + j3

    G1, G2, G3 = cores
    A1 = G1[0, i1, j1, :]
    A2 = G2[:, i2, j2, :]
    A3 = G3[:, i3, j3, 0]
    expected = torch.einsum("a,ab,b->", A1, A2, A3)

    torch.testing.assert_close(W[i, j], expected, rtol=1e-12, atol=1e-12)


def test_dense_to_tt_matrix_tensor_element_correspondence():
    """dense W[i,j] と combined tensor A[a_1,...,a_d] の要素対応を確認する。"""
    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    m, n = math.prod(out_modes), math.prod(in_modes)

    torch.manual_seed(7)
    W = torch.randn(m, n, dtype=torch.float64)
    A = dense_to_tt_matrix_tensor(W, out_modes, in_modes)
    assert A.shape == tuple(mk * nk for mk, nk in zip(out_modes, in_modes))

    i_modes = (1, 2, 0)
    j_modes = (0, 1, 2)

    i_flat = 0
    for ik, mk in zip(i_modes, out_modes):
        i_flat = i_flat * mk + ik
    j_flat = 0
    for jk, nk in zip(j_modes, in_modes):
        j_flat = j_flat * nk + jk

    a_modes = tuple(ik * nk + jk for ik, jk, nk in zip(i_modes, j_modes, in_modes))

    torch.testing.assert_close(W[i_flat, j_flat], A[a_modes], rtol=0.0, atol=0.0)


def test_dense_to_tt_matrix_cores_max_rank_none_reconstructs_exactly():
    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    m, n = math.prod(out_modes), math.prod(in_modes)
    torch.manual_seed(8)
    W = torch.randn(m, n, dtype=torch.float64)

    cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=None)
    W_hat = tt_matrix_to_dense(cores, out_modes, in_modes)
    torch.testing.assert_close(W_hat, W, rtol=1e-10, atol=1e-10)


def test_dense_to_tt_matrix_cores_max_rank_bounds_rank():
    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    m, n = math.prod(out_modes), math.prod(in_modes)
    torch.manual_seed(9)
    W = torch.randn(m, n, dtype=torch.float64)

    cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=2)
    ranks = tt_matrix_ranks(cores)
    assert all(r <= 2 for r in ranks[1:-1])
    assert ranks[0] == 1 and ranks[-1] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("max_rank", [None, 1, 4])
def test_dense_to_tt_matrix_cores_d1_reconstructs_and_forwards(dtype, max_rank):
    """1-site は TT-SVD せず、``W.reshape(1, m, n, 1)`` 相当の1 core を返す。"""
    out_modes, in_modes = (5,), (4,)
    torch.manual_seed(70)
    W = torch.randn(*out_modes, *in_modes, dtype=dtype).reshape(5, 4)
    saved = W.clone()

    cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=max_rank)

    assert len(cores) == 1
    assert tuple(cores[0].shape) == (1, 5, 4, 1)
    assert cores[0].dtype == dtype
    assert cores[0].device == W.device
    torch.testing.assert_close(cores[0].reshape(5, 4), W, rtol=0.0, atol=0.0)
    torch.testing.assert_close(W, saved, rtol=0.0, atol=0.0)
    assert tt_matrix_ranks(cores) == (1, 1)

    tol = 1e-5 if dtype == torch.float32 else 1e-12
    W_hat = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W_hat.dtype == dtype
    torch.testing.assert_close(W_hat, W, rtol=tol, atol=tol)

    x = torch.randn(3, 4, dtype=dtype)
    bias = torch.randn(5, dtype=dtype)
    y = tt_linear_forward(x, cores, out_modes, in_modes, bias)
    torch.testing.assert_close(y, F.linear(x, W, bias), rtol=tol, atol=tol)
    y_no_bias = tt_linear_forward(x, cores, out_modes, in_modes, bias=None)
    torch.testing.assert_close(y_no_bias, F.linear(x, W, None), rtol=tol, atol=tol)


def test_dense_to_tt_matrix_cores_d1_large_max_rank_matches_exact():
    """内部 bond がないので、max_rank が 1 より大きくても 1-core のまま。"""
    W = torch.randn(5, 4, dtype=torch.float64)
    cores_none = dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=None)
    cores_one = dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=1)
    cores_large = dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=8)

    torch.testing.assert_close(cores_one[0], cores_none[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(cores_large[0], cores_none[0], rtol=0.0, atol=0.0)
    assert tt_matrix_ranks(cores_large) == (1, 1)


def test_dense_to_tt_matrix_cores_d1_preserves_autograd():
    W = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    cores = dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=None)
    cores[0].sum().backward()
    assert W.grad is not None
    torch.testing.assert_close(W.grad, torch.ones_like(W), rtol=0.0, atol=0.0)

    W_capped = torch.randn(5, 4, dtype=torch.float32, requires_grad=True)
    cores_capped = dense_to_tt_matrix_cores(W_capped, (5,), (4,), max_rank=3)
    cores_capped[0].sum().backward()
    assert W_capped.grad is not None
    torch.testing.assert_close(
        W_capped.grad, torch.ones_like(W_capped), rtol=0.0, atol=0.0
    )


def test_dense_to_tt_matrix_cores_d1_does_not_call_tt_svd(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("d=1 では tt_svd / tt_svd_exact を呼ばない")

    monkeypatch.setattr(tt_matrix_module, "tt_svd", boom)
    monkeypatch.setattr(tt_matrix_module, "tt_svd_exact", boom)

    W = torch.randn(5, 4, dtype=torch.float64)
    for max_rank in (None, 1, 6):
        cores = dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=max_rank)
        assert tuple(cores[0].shape) == (1, 5, 4, 1)


@pytest.mark.parametrize("max_rank", [0, -1])
def test_dense_to_tt_matrix_cores_d1_rejects_non_positive_max_rank(max_rank):
    W = torch.randn(5, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=max_rank)


@pytest.mark.parametrize("max_rank", [True, False, 1.5, "2"])
def test_dense_to_tt_matrix_cores_d1_rejects_invalid_max_rank_type(max_rank):
    W = torch.randn(5, 4, dtype=torch.float64)
    with pytest.raises(TypeError):
        dense_to_tt_matrix_cores(W, (5,), (4,), max_rank=max_rank)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_matrix_round_trip_matches_dtype(dtype):
    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    m, n = math.prod(out_modes), math.prod(in_modes)
    torch.manual_seed(10)
    W = torch.randn(m, n, dtype=dtype)

    cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=None)
    W_hat = tt_matrix_to_dense(cores, out_modes, in_modes)
    assert W_hat.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(W_hat, W, rtol=tol, atol=tol)


def test_dense_to_tt_matrix_tensor_does_not_mutate_input():
    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    m, n = math.prod(out_modes), math.prod(in_modes)
    torch.manual_seed(11)
    W = torch.randn(m, n, dtype=torch.float64)
    saved = W.clone()

    dense_to_tt_matrix_tensor(W, out_modes, in_modes)

    torch.testing.assert_close(W, saved, rtol=0.0, atol=0.0)


def test_tt_matrix_to_dense_does_not_mutate_cores():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=12)
    saved = _clone_cores(cores)

    tt_matrix_to_dense(cores, OUT_MODES_3, IN_MODES_3)

    _assert_cores_equal(cores, saved)


def test_tt_cores_to_tt_matrix_cores_does_not_mutate_input():
    from nn_compression.compression import tt_svd_exact

    out_modes, in_modes = OUT_MODES_3, IN_MODES_3
    combined = dense_to_tt_matrix_tensor(
        torch.randn(math.prod(out_modes), math.prod(in_modes), dtype=torch.float64),
        out_modes,
        in_modes,
    )
    cores_3d = tt_svd_exact(combined)
    saved = _clone_cores(cores_3d)

    tt_cores_to_tt_matrix_cores(cores_3d, out_modes, in_modes)

    _assert_cores_equal(cores_3d, saved)


def test_tt_matrix_ranks_and_num_parameters():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=13)
    assert tt_matrix_ranks(cores) == RANKS_3
    expected_params = sum(core.numel() for core in cores)
    assert tt_matrix_num_parameters(cores) == expected_params


# ---------------------------------------------------------------------------
# forward値: tt_linear_forward vs F.linear(dense) vs x @ W.T + b
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("batch", [1, 4])
def test_tt_linear_forward_matches_dense_d3(dtype, has_bias, batch):
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    cores = _random_tt_matrix_cores(
        OUT_MODES_3, IN_MODES_3, RANKS_3, dtype=dtype, seed=14
    )
    torch.manual_seed(100)
    n = math.prod(IN_MODES_3)
    m = math.prod(OUT_MODES_3)
    x = torch.randn(batch, n, dtype=dtype)
    bias = torch.randn(m, dtype=dtype) if has_bias else None

    y_tt = tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bias)
    W = tt_matrix_to_dense(cores, OUT_MODES_3, IN_MODES_3)
    y_dense_manual = x @ W.T + (bias if bias is not None else 0.0)
    y_f_linear = F.linear(x, W, bias)

    torch.testing.assert_close(y_tt, y_dense_manual, rtol=tol, atol=tol)
    torch.testing.assert_close(y_tt, y_f_linear, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_linear_forward_matches_dense_d1(dtype):
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    out_modes, in_modes = (5,), (4,)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 1), dtype=dtype, seed=15)
    x = torch.randn(3, 4, dtype=dtype)
    bias = torch.randn(5, dtype=dtype)

    y_tt = tt_linear_forward(x, cores, out_modes, in_modes, bias)
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    y_f_linear = F.linear(x, W, bias)

    torch.testing.assert_close(y_tt, y_f_linear, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_linear_forward_matches_dense_d2_different_bond_rank(dtype):
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    out_modes, in_modes = (2, 3), (3, 2)
    for rank in (1, 2, 5):
        cores = _random_tt_matrix_cores(
            out_modes, in_modes, (1, rank, 1), dtype=dtype, seed=16 + rank
        )
        x = torch.randn(4, 6, dtype=dtype)
        bias = torch.randn(6, dtype=dtype)

        y_tt = tt_linear_forward(x, cores, out_modes, in_modes, bias)
        W = tt_matrix_to_dense(cores, out_modes, in_modes)
        y_f_linear = F.linear(x, W, bias)

        torch.testing.assert_close(y_tt, y_f_linear, rtol=tol, atol=tol)


def test_tt_linear_forward_batch_size_one():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=20)
    x = torch.randn(1, math.prod(IN_MODES_3), dtype=torch.float64)
    bias = torch.randn(math.prod(OUT_MODES_3), dtype=torch.float64)

    y_tt = tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bias)
    W = tt_matrix_to_dense(cores, OUT_MODES_3, IN_MODES_3)
    y_f_linear = F.linear(x, W, bias)

    assert y_tt.shape == (1, math.prod(OUT_MODES_3))
    torch.testing.assert_close(y_tt, y_f_linear, rtol=1e-9, atol=1e-9)


def test_tt_linear_forward_does_not_mutate_inputs():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=21)
    saved_cores = _clone_cores(cores)
    x = torch.randn(3, math.prod(IN_MODES_3), dtype=torch.float64)
    saved_x = x.clone()
    bias = torch.randn(math.prod(OUT_MODES_3), dtype=torch.float64)
    saved_bias = bias.clone()

    tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bias)

    _assert_cores_equal(cores, saved_cores)
    torch.testing.assert_close(x, saved_x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bias, saved_bias, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------


def test_tt_linear_forward_gradients_match_dense_reconstruction():
    out_modes, in_modes = (2, 3), (3, 2)
    cores_a = _random_tt_matrix_cores(out_modes, in_modes, (1, 4, 1), seed=30)
    for core in cores_a:
        core.requires_grad_(True)
    cores_b = [core.detach().clone().requires_grad_(True) for core in cores_a]

    torch.manual_seed(31)
    x_a = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    bias_a = torch.randn(6, dtype=torch.float64, requires_grad=True)
    bias_b = bias_a.detach().clone().requires_grad_(True)

    y_direct = tt_linear_forward(x_a, cores_a, out_modes, in_modes, bias_a)
    W = tt_matrix_to_dense(cores_b, out_modes, in_modes)
    y_dense = x_b @ W.T + bias_b

    (y_direct**2).sum().backward()
    (y_dense**2).sum().backward()

    torch.testing.assert_close(x_a.grad, x_b.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(bias_a.grad, bias_b.grad, rtol=1e-9, atol=1e-9)
    for core_a, core_b in zip(cores_a, cores_b):
        torch.testing.assert_close(core_a.grad, core_b.grad, rtol=1e-9, atol=1e-9)


def test_tt_linear_forward_gradcheck_small_tt_matrix():
    out_modes, in_modes = (2, 2), (2, 2)
    cores = _random_tt_matrix_cores(out_modes, in_modes, (1, 2, 1), seed=32)
    tensors = tuple(core.clone().requires_grad_(True) for core in cores)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float64, requires_grad=True)

    def fn(x_in, c0, c1, b):
        return tt_linear_forward(x_in, [c0, c1], out_modes, in_modes, b)

    assert torch.autograd.gradcheck(
        fn, (x, *tensors, bias), atol=1e-6, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# dense非依存の確認
# ---------------------------------------------------------------------------


def test_tt_linear_forward_does_not_call_tt_matrix_to_dense(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("tt_matrix_to_dense は tt_linear_forward から呼ばれてはならない")

    monkeypatch.setattr(tt_matrix_module, "tt_matrix_to_dense", boom)

    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=40)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64)
    bias = torch.randn(math.prod(OUT_MODES_3), dtype=torch.float64)

    # patch後もtt_linear_forwardは成功する（内部でtt_matrix_to_denseを呼ばない）。
    y = tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bias)
    assert y.shape == (2, math.prod(OUT_MODES_3))


# ---------------------------------------------------------------------------
# validation: TT-matrix cores
# ---------------------------------------------------------------------------


def test_tt_matrix_to_dense_rejects_empty_cores():
    with pytest.raises(ValueError):
        tt_matrix_to_dense([], (), ())


def test_tt_matrix_to_dense_rejects_mode_count_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=41)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(cores, OUT_MODES_3[:-1], IN_MODES_3)


@pytest.mark.parametrize("bad_mode", [0, -1, True, False])
def test_tt_matrix_to_dense_rejects_invalid_modes(bad_mode):
    cores = _random_tt_matrix_cores((2, 2), (2, 2), (1, 2, 1), seed=42)
    bad_out_modes = (bad_mode, 2)
    with pytest.raises((TypeError, ValueError)):
        tt_matrix_to_dense(cores, bad_out_modes, (2, 2))


def test_tt_matrix_to_dense_rejects_3d_core():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=43)
    broken = list(cores)
    r_left, m_k, n_k, r_right = broken[0].shape
    broken[0] = broken[0].reshape(r_left, m_k * n_k, r_right)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_5d_core():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=44)
    broken = list(cores)
    broken[0] = broken[0].unsqueeze(0)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_physical_dimension_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=45)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(cores, (OUT_MODES_3[0] + 1, *OUT_MODES_3[1:]), IN_MODES_3)


def test_tt_matrix_to_dense_rejects_bond_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=46)
    broken = list(cores)
    r_left, m_k, n_k, r_right = broken[1].shape
    broken[1] = torch.randn(r_left + 1, m_k, n_k, r_right, dtype=broken[1].dtype)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_left_boundary_rank_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=47)
    broken = list(cores)
    r_left, m_k, n_k, r_right = broken[0].shape
    broken[0] = torch.randn(2, m_k, n_k, r_right, dtype=broken[0].dtype)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_right_boundary_rank_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=48)
    broken = list(cores)
    r_left, m_k, n_k, r_right = broken[-1].shape
    broken[-1] = torch.randn(r_left, m_k, n_k, 2, dtype=broken[-1].dtype)
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_dtype_mismatch_between_cores():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=49)
    broken = list(cores)
    broken[1] = broken[1].to(torch.float32)
    with pytest.raises(TypeError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_device_mismatch_between_cores_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDAが利用できないため device mismatch を検証できません。")
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=50)
    broken = list(cores)
    broken[1] = broken[1].to(torch.device("cuda"))
    with pytest.raises(ValueError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


def test_tt_matrix_to_dense_rejects_unsupported_dtype():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=51)
    broken = [core.to(torch.float16) for core in cores]
    with pytest.raises(TypeError):
        tt_matrix_to_dense(broken, OUT_MODES_3, IN_MODES_3)


# ---------------------------------------------------------------------------
# validation: dense weight (dense_to_tt_matrix_tensor)
# ---------------------------------------------------------------------------


def test_dense_to_tt_matrix_tensor_rejects_shape_mismatch():
    W = torch.randn(10, 10, dtype=torch.float64)
    with pytest.raises(ValueError):
        dense_to_tt_matrix_tensor(W, OUT_MODES_3, IN_MODES_3)


def test_dense_to_tt_matrix_tensor_rejects_1d_input():
    W = torch.randn(12, dtype=torch.float64)
    with pytest.raises(ValueError):
        dense_to_tt_matrix_tensor(W, OUT_MODES_3, IN_MODES_3)


def test_dense_to_tt_matrix_tensor_rejects_unsupported_dtype():
    W = torch.randn(12, 12).to(torch.float16)
    with pytest.raises(TypeError):
        dense_to_tt_matrix_tensor(W, OUT_MODES_3, IN_MODES_3)


# ---------------------------------------------------------------------------
# validation: tt_linear_forward の x / bias
# ---------------------------------------------------------------------------


def test_tt_linear_forward_rejects_x_last_dim_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=60)
    x = torch.randn(2, math.prod(IN_MODES_3) + 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3)


def test_tt_linear_forward_rejects_x_ndim_not_2():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=61)
    x = torch.randn(math.prod(IN_MODES_3), dtype=torch.float64)
    with pytest.raises(ValueError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3)


def test_tt_linear_forward_rejects_x_dtype_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=62)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float32)
    with pytest.raises(TypeError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3)


def test_tt_linear_forward_rejects_x_device_mismatch_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDAが利用できないため device mismatch を検証できません。")
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=63)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3)


def test_tt_linear_forward_allows_bias_none():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=64)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64)
    y = tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bias=None)
    assert y.shape == (2, math.prod(OUT_MODES_3))


def test_tt_linear_forward_rejects_bias_shape_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=65)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64)
    bad_bias = torch.randn(math.prod(OUT_MODES_3) + 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bad_bias)


def test_tt_linear_forward_rejects_bias_dtype_mismatch():
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=66)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64)
    bad_bias = torch.randn(math.prod(OUT_MODES_3), dtype=torch.float32)
    with pytest.raises(TypeError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bad_bias)


def test_tt_linear_forward_rejects_bias_device_mismatch_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDAが利用できないため device mismatch を検証できません。")
    cores = _random_tt_matrix_cores(OUT_MODES_3, IN_MODES_3, RANKS_3, seed=67)
    x = torch.randn(2, math.prod(IN_MODES_3), dtype=torch.float64)
    bad_bias = torch.randn(math.prod(OUT_MODES_3), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):
        tt_linear_forward(x, cores, OUT_MODES_3, IN_MODES_3, bad_bias)
