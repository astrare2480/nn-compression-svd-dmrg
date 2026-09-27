"""TT間contraction（内積・Frobeniusノルム）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/13_tt_inner_frobenius_norm_and_distance.ipynb

``tt_distance_sq`` / ``tt_distance`` はBugbotの4回のレビューで反例が
出続けたため、現時点ではpublic APIとして公開していない
（``src/nn_compression/compression/tt_contraction.py`` のモジュール
docstring参照）。このファイルは ``tt_inner`` / ``tt_fro_norm`` /
``_safe_sqrt`` だけを対象とする。
"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.compression import tt_fro_norm, tt_inner, tt_reconstruct
from nn_compression.compression.tt_contraction import _safe_sqrt


def _random_tt(
    physical_dims: tuple[int, ...],
    ranks: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    seed: int = 0,
) -> list[torch.Tensor]:
    """検証用のランダムTTを作る（テスト専用ヘルパー、src化しない）。"""
    assert len(ranks) == len(physical_dims) + 1
    assert ranks[0] == 1
    assert ranks[-1] == 1

    torch.manual_seed(seed)
    cores = []
    for k, n_k in enumerate(physical_dims):
        cores.append(
            torch.randn(ranks[k], n_k, ranks[k + 1], dtype=dtype, device=device)
        )
    return cores


def _clone_cores(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [core.clone() for core in cores]


def _assert_cores_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


PHYSICAL_DIMS = (2, 3, 2)
RANKS_A = (1, 2, 3, 1)
RANKS_B = (1, 3, 2, 1)


# ---------------------------------------------------------------------------
# dense照合（float64）
# ---------------------------------------------------------------------------


def test_tt_inner_matches_dense_inner_product():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)

    inner_tt = tt_inner(cores_a, cores_b)

    dense_a = tt_reconstruct(cores_a)
    dense_b = tt_reconstruct(cores_b)
    inner_dense = torch.sum(dense_a * dense_b)

    torch.testing.assert_close(inner_tt, inner_dense, atol=1e-10, rtol=1e-10)
    assert inner_tt.shape == ()


def test_tt_fro_norm_matches_dense_frobenius_norm():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)

    norm_tt = tt_fro_norm(cores_a)
    dense_a = tt_reconstruct(cores_a)
    norm_dense = torch.linalg.vector_norm(dense_a.reshape(-1))

    torch.testing.assert_close(norm_tt, norm_dense, atol=1e-10, rtol=1e-10)


def test_tt_inner_allows_different_tt_ranks():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)
    # rankがA/Bで異なっていても正しく計算できる（本テストの前提そのもの）
    assert [c.shape[-1] for c in cores_a[:-1]] != [c.shape[-1] for c in cores_b[:-1]]
    result = tt_inner(cores_a, cores_b)
    assert result.shape == ()


def test_tt_contraction_handles_d_equal_one():
    cores_a = _random_tt((5,), (1, 1), seed=3)
    cores_b = _random_tt((5,), (1, 1), seed=4)

    inner_tt = tt_inner(cores_a, cores_b)
    dense_a = tt_reconstruct(cores_a)
    dense_b = tt_reconstruct(cores_b)
    inner_dense = torch.sum(dense_a * dense_b)

    torch.testing.assert_close(inner_tt, inner_dense, atol=1e-10, rtol=1e-10)


def test_tt_contraction_handles_zero_tensor():
    cores_a = [torch.zeros(1, n, 1, dtype=torch.float64) for n in PHYSICAL_DIMS]
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)

    assert tt_fro_norm(cores_a).item() == pytest.approx(0.0, abs=1e-12)
    assert tt_inner(cores_a, cores_b).item() == pytest.approx(0.0, abs=1e-12)


def test_tt_contraction_does_not_mutate_input():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)
    saved_a = _clone_cores(cores_a)
    saved_b = _clone_cores(cores_b)

    tt_inner(cores_a, cores_b)
    tt_fro_norm(cores_a)

    _assert_cores_equal(cores_a, saved_a)
    _assert_cores_equal(cores_b, saved_b)


def test_tt_inner_backward_populates_gradients():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)
    for core in cores_a:
        core.requires_grad_(True)

    result = tt_inner(cores_a, cores_b)
    result.backward()

    for core in cores_a:
        assert core.grad is not None
        assert torch.isfinite(core.grad).all()


def test_tt_inner_backward_matches_dense_gradient():
    """``tt_inner`` の勾配が、dense再構成経由の勾配と一致すること（値まで比較）。"""
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)
    tt_cores_a = [core.clone().requires_grad_(True) for core in cores_a]
    dense_cores_a = [core.clone().requires_grad_(True) for core in cores_a]

    tt_inner(tt_cores_a, cores_b).backward()

    dense_b = tt_reconstruct(cores_b)
    (torch.sum(tt_reconstruct(dense_cores_a) * dense_b)).backward()

    for tt_core, dense_core in zip(tt_cores_a, dense_cores_a):
        torch.testing.assert_close(tt_core.grad, dense_core.grad, atol=1e-8, rtol=1e-6)


def test_tt_fro_norm_gradient_at_nonzero_matches_dense():
    cores = [core.clone().requires_grad_(True) for core in _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)]
    dense_cores = [core.detach().clone().requires_grad_(True) for core in cores]

    tt_fro_norm(cores).backward()
    torch.linalg.vector_norm(tt_reconstruct(dense_cores).reshape(-1)).backward()

    for tt_core, dense_core in zip(cores, dense_cores):
        torch.testing.assert_close(tt_core.grad, dense_core.grad, atol=1e-8, rtol=1e-6)


def test_tt_inner_gradcheck_small_tt():
    cores_a = _random_tt((2, 3), (1, 2, 1), seed=0)
    cores_b = _random_tt((2, 3), (1, 2, 1), seed=1)
    tensors = tuple(core.detach().clone().requires_grad_(True) for core in cores_a + cores_b)

    def inner_of_flat(*flat: torch.Tensor) -> torch.Tensor:
        return tt_inner(list(flat[:2]), list(flat[2:]))

    assert torch.autograd.gradcheck(inner_of_flat, tensors, atol=1e-6, rtol=1e-4)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_tt_inner_rejects_order_mismatch():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    # 境界rankが1の正常な短い(2階)TTを作り、order検証まで到達させる。
    cores_b_short = _random_tt((2, 3), (1, 3, 1), seed=5)

    with pytest.raises(ValueError):
        tt_inner(cores_a, cores_b_short)


def test_tt_inner_rejects_physical_shape_mismatch():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt((2, 4, 2), RANKS_B, seed=2)

    with pytest.raises(ValueError):
        tt_inner(cores_a, cores_b)


def test_tt_inner_rejects_dtype_mismatch():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1, dtype=torch.float64)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2, dtype=torch.float32)

    with pytest.raises(TypeError):
        tt_inner(cores_a, cores_b)


def test_tt_inner_rejects_device_mismatch_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDAが利用できないため device mismatch を検証できません。")

    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2, device=torch.device("cuda"))

    with pytest.raises(ValueError):
        tt_inner(cores_a, cores_b)


def test_tt_inner_rejects_broken_bond_in_either_tt():
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, seed=1)
    cores_b = _random_tt(PHYSICAL_DIMS, RANKS_B, seed=2)

    broken = _clone_cores(cores_a)
    r0, n0, _ = broken[0].shape
    broken[0] = torch.randn(r0, n0, 99, dtype=broken[0].dtype)

    with pytest.raises(ValueError):
        tt_inner(broken, cores_b)


# ---------------------------------------------------------------------------
# tt_fro_norm: zero / tiny values / dtype
# ---------------------------------------------------------------------------


def test_tt_fro_norm_backward_has_no_nan_gradient_for_zero_tt():
    """零TTのFrobeniusノルムのbackwardがNaN勾配を生んではならない。"""
    cores = [
        torch.zeros(1, n, 1, dtype=torch.float64, requires_grad=True)
        for n in PHYSICAL_DIMS
    ]

    norm = tt_fro_norm(cores)
    norm.backward()

    assert norm.item() == pytest.approx(0.0, abs=1e-12)
    for core in cores:
        assert core.grad is not None
        assert not torch.isnan(core.grad).any()


def test_tt_fro_norm_matches_sqrt_for_tiny_positive_float32():
    """float32の微小な正値でも、ノルムが ``sqrt`` より小さくならないこと。"""
    value = torch.finfo(torch.float32).eps / 10
    core = torch.tensor([[[value]]], dtype=torch.float32)
    norm = tt_fro_norm([core])
    expected = torch.sqrt(torch.tensor(value * value, dtype=torch.float32))
    torch.testing.assert_close(norm, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_fro_norm_matches_dense_for_both_dtypes(dtype):
    cores_a = _random_tt(PHYSICAL_DIMS, RANKS_A, dtype=dtype, seed=1)
    norm_tt = tt_fro_norm(cores_a)
    dense_a = tt_reconstruct(cores_a)
    norm_dense = torch.linalg.vector_norm(dense_a.reshape(-1))
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(norm_tt, norm_dense, atol=tol, rtol=tol)


def test_tt_fro_norm_overflow_and_underflow_are_documented_not_silently_wrong():
    """極端なスケールでのoverflow/underflowは、明示的な ``inf``/``0`` として現れる。

    このテストは「安全に丸める」実装ではないことの特性テストである。
    ``tt_inner`` / ``tt_fro_norm`` はスケール補正を行わないため、
    ``1e200`` 程度の極端な値では ``inf`` を返し得る（黙って0や有限値へ
    丸めることはしない）。将来この挙動を変える場合は、ここを更新すること。
    """
    huge = torch.tensor([[[1e200]]], dtype=torch.float64)
    norm = tt_fro_norm([huge, huge])
    assert torch.isinf(norm.detach()) or norm.item() > 1e300

    tiny = torch.tensor([[[1e-200]]], dtype=torch.float64)
    norm_tiny = tt_fro_norm([tiny, tiny])
    assert norm_tiny.item() == pytest.approx(0.0, abs=1e-300) or norm_tiny.item() > 0.0


def test_tt_fro_norm_does_not_hide_overflow_cancellation_nan_as_zero_norm():
    """Bugbot反例: 有限値TT ``[1e308, 1e308]`` と ``[1e-308, -0.5e-308]``。

    denseな再構成値は有限（約 ``0.5``）だが、``tt_inner`` 内部の縮約は
    ``1e308`` の二乗でoverflowし、符号の異なる ``inf`` の加算で ``NaN``
    になる。``_safe_sqrt`` が ``NaN`` を ``0`` へ変換すると、
    ``tt_fro_norm`` は誤った零ノルムを返してしまう。
    """
    core1 = torch.tensor([1e308, 1e308], dtype=torch.float64).reshape(1, 1, 2)
    core2 = torch.tensor([1e-308, -0.5e-308], dtype=torch.float64).reshape(2, 1, 1)
    cores = [core1, core2]

    dense_value = tt_reconstruct(cores).item()
    assert math.isfinite(dense_value)  # 前提: denseな値そのものは有限

    inner = tt_inner(cores, cores)
    assert torch.isnan(inner)  # 前提: 内部縮約は実際にNaNになる

    norm = tt_fro_norm(cores)
    assert torch.isnan(norm), (
        f"tt_fro_norm はNaNの縮約結果を隠さずNaNを返すべきだが、{norm.item()!r} を返した。"
    )


# ---------------------------------------------------------------------------
# _safe_sqrt: x>0 / x=0 / x=-0.0 / NaN / +inf / -inf / 負値(有意・微小)
# ---------------------------------------------------------------------------


def test_safe_sqrt_matches_torch_sqrt_for_positive_values():
    values = torch.tensor(
        [1e-20, 1e-12, 1.1920929e-8, 1e-4, 1.0, 1e8],
        dtype=torch.float64,
    )
    torch.testing.assert_close(_safe_sqrt(values), torch.sqrt(values), atol=0, rtol=0)


def test_safe_sqrt_zero_is_exact_and_backward_is_finite():
    value = torch.zeros((), dtype=torch.float64, requires_grad=True)
    result = _safe_sqrt(value)
    result.backward()

    assert result.item() == 0.0
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_safe_sqrt_negative_zero_is_exact_zero_and_does_not_raise():
    """``-0.0`` は ``-0.0 < 0`` が ``False`` なので、負値扱いにならず0になる。"""
    value = torch.tensor(-0.0, dtype=torch.float64)
    result = _safe_sqrt(value)
    assert result.item() == 0.0


def test_safe_sqrt_does_not_convert_nan_to_zero():
    """回帰: ``x=NaN`` で ``positive = x_nonnegative > 0`` が ``False`` になり、

    最後の ``torch.where`` が誤って0を選んでしまっていた。数値計算の失敗
    （overflow・cancellationなど）をNaNのまま気づけるように、
    ``_safe_sqrt(NaN)`` は0であってはならない。
    """
    value = torch.tensor(float("nan"), dtype=torch.float64)
    result = _safe_sqrt(value)
    assert torch.isnan(result), f"NaNが握り潰されて {result.item()!r} になった。"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_safe_sqrt_nan_is_preserved_elementwise_within_a_tensor(dtype):
    """NaNと通常値が混在するTensorでも、NaNの要素だけが0にならないこと。"""
    values = torch.tensor([float("nan"), 4.0, 0.0], dtype=dtype)
    result = _safe_sqrt(values)
    assert torch.isnan(result[0])
    torch.testing.assert_close(result[1], torch.sqrt(values[1]))
    assert result[2].item() == 0.0


def test_safe_sqrt_positive_infinity_matches_sqrt_of_infinity():
    value = torch.tensor(float("inf"), dtype=torch.float64)
    result = _safe_sqrt(value)
    assert torch.isinf(result)
    assert result.item() > 0.0


def test_safe_sqrt_negative_infinity_raises_value_error():
    """回帰（今回の修正1）: ``-inf`` は丸め誤差ではなく上流計算の失敗なので

    0へ変換せず ``ValueError`` を送出する。
    """
    value = torch.tensor(float("-inf"), dtype=torch.float64)
    with pytest.raises(ValueError):
        _safe_sqrt(value)


def test_safe_sqrt_significant_negative_value_raises_value_error():
    """回帰（今回の修正1）: ``x=-1.0`` のような有意な負値を0へ隠さない。"""
    value = torch.tensor(-1.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        _safe_sqrt(value)


def test_safe_sqrt_tiny_negative_value_raises_value_error():
    """回帰（今回の修正1）: 微小負値（``-1e-12``）も丸め誤差として0へ

    隠さず、``ValueError`` を送出する（旧contractの
    「丸め誤差として0へ丸める」動作は今回廃止した）。
    """
    for value in (-1e-12, -1e-30):
        with pytest.raises(ValueError):
            _safe_sqrt(torch.tensor(value, dtype=torch.float64))


def test_safe_sqrt_negative_value_does_not_use_nan_to_num():
    """``nan_to_num`` 相当の変換で0へ落としていないことの特性テスト。

    負値を含むTensor全体に対して呼び出したとき、結果が「0を含む有限の
    Tensor」ではなく例外になることを確認する（実装が
    ``torch.nan_to_num(..., neginf=0)`` のような変換を使っていないことの
    間接的な確認）。
    """
    values = torch.tensor([4.0, -1.0, float("nan")], dtype=torch.float64)
    with pytest.raises(ValueError):
        _safe_sqrt(values)


def test_safe_sqrt_positive_gradient_matches_reciprocal_two_sqrt():
    """正の通常値では、勾配が ``1 / (2 * sqrt(x))`` と数値的に一致すること。"""
    value = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    result = _safe_sqrt(value)
    result.backward()

    expected_value = torch.sqrt(torch.tensor(4.0, dtype=torch.float64))
    expected_grad = 1.0 / (2.0 * expected_value)
    torch.testing.assert_close(result.detach(), expected_value, atol=0, rtol=0)
    torch.testing.assert_close(value.grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_safe_sqrt_tiny_positive_gradient_is_finite_and_matches_reciprocal_two_sqrt():
    value = torch.tensor(1e-20, dtype=torch.float64, requires_grad=True)
    result = _safe_sqrt(value)
    result.backward()

    expected_grad = 1.0 / (2.0 * math.sqrt(1e-20))
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad.item(), expected_grad, atol=0, rtol=1e-9)
