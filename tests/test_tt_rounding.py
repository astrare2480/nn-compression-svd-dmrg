"""単一bond truncationとTT-roundingの回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/09_truncated_svd_and_bond_rank_truncation.ipynb
    notebooks/30_tt_mps/00_fundamentals/10_eckart_young_mirsky_and_single_bond_optimality.ipynb
    notebooks/30_tt_mps/00_fundamentals/12_tt_rounding_right_canonicalization_and_truncated_svd.ipynb
"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.compression import (
    tt_bond_singular_values,
    tt_canonicalize,
    tt_num_parameters,
    tt_physical_shape,
    tt_ranks,
    tt_reconstruct,
    tt_round,
    tt_svd_exact,
    tt_truncate_bond,
)
from nn_compression.compression.tt_rounding import _choose_truncation_rank

SHAPE_4D = (2, 3, 2, 4)


def _tensor(seed: int, shape: tuple[int, ...] = SHAPE_4D, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _clone_cores(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [core.clone() for core in cores]


def _assert_cores_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def _padded_redundant_tt(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    """真のTT-rank(2,2)を、意図的に冗長なbond shape(4,3)へzero paddingしたTT。"""
    torch.manual_seed(7)
    g1_base = torch.randn(1, 4, 2, dtype=dtype)
    g2_base = torch.randn(2, 5, 2, dtype=dtype)
    g3_base = torch.randn(2, 3, 1, dtype=dtype)

    g1 = torch.zeros(1, 4, 4, dtype=dtype)
    g2 = torch.zeros(4, 5, 3, dtype=dtype)
    g3 = torch.zeros(3, 3, 1, dtype=dtype)

    g1[:, :, :2] = g1_base
    g2[:2, :, :2] = g2_base
    g3[:2, :, :] = g3_base
    return [g1, g2, g3]


# ---------------------------------------------------------------------------
# tt_truncate_bond
# ---------------------------------------------------------------------------


def test_tt_truncate_bond_rank_is_bounded_and_shapes_consistent():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    bond = 1
    rank_before = cores[bond].shape[-1]
    max_rank = rank_before - 1
    assert max_rank >= 1

    new_cores, diag = tt_truncate_bond(cores, bond, max_rank)

    assert diag["rank_before"] == rank_before
    assert diag["rank_after"] == max_rank
    assert new_cores[bond].shape[-1] == max_rank
    assert new_cores[bond + 1].shape[0] == max_rank
    # 他のcoreのshapeは変化しない
    assert new_cores[bond].shape[:-1] == cores[bond].shape[:-1]
    assert new_cores[bond + 1].shape[1:] == cores[bond + 1].shape[1:]


def test_tt_truncate_bond_measured_error_matches_discarded_singular_values():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    bond = 1
    max_rank = max(1, cores[bond].shape[-1] - 1)

    new_cores, diag = tt_truncate_bond(cores, bond, max_rank)

    X_hat = tt_reconstruct(new_cores)
    measured_error = torch.linalg.vector_norm(X - X_hat).item()

    assert measured_error == pytest.approx(diag["local_error"], abs=1e-8)
    assert diag["discarded_tail_sq"] == pytest.approx(measured_error**2, abs=1e-8)

    # singular_values は打ち切り前の特異値全体（rank_before本）を保持する
    assert diag["singular_values"].shape[0] == diag["rank_before"]
    tail_sq_from_values = (
        diag["singular_values"][diag["rank_after"] :].square().sum().item()
    )
    assert tail_sq_from_values == pytest.approx(diag["discarded_tail_sq"], abs=1e-10)


def test_tt_truncate_bond_achieves_local_eckart_young_optimum():
    """canonical環境下でのSVD打ち切りが、同rank以下の任意候補より劣らないことを確認する。"""
    X = _tensor(seed=3)
    cores = tt_svd_exact(X)
    bond = 1
    max_rank = max(1, cores[bond].shape[-1] - 1)

    truncated_cores, _ = tt_truncate_bond(cores, bond, max_rank)
    error_truncated = torch.linalg.vector_norm(
        X - tt_reconstruct(truncated_cores)
    ).item()

    canon = tt_canonicalize(cores, center=bond)
    r_left, n_bond, r_bond = canon[bond].shape

    torch.manual_seed(42)
    worst_case = float("inf")
    for _ in range(20):
        P = torch.randn(r_left * n_bond, max_rank, dtype=X.dtype)
        Q = torch.randn(r_bond, max_rank, dtype=X.dtype)
        candidate_matrix = P @ Q.T  # rank <= max_rank

        candidate_cores = _clone_cores(canon)
        candidate_cores[bond] = candidate_matrix.reshape(r_left, n_bond, r_bond)
        error_candidate = torch.linalg.vector_norm(
            X - tt_reconstruct(candidate_cores)
        ).item()
        worst_case = min(worst_case, error_candidate)

    assert error_truncated <= worst_case + 1e-8


def test_tt_truncate_bond_without_truncation_when_max_rank_at_least_current_rank():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    bond = 1
    rank_before = cores[bond].shape[-1]

    new_cores, diag = tt_truncate_bond(cores, bond, rank_before + 5)

    assert diag["rank_after"] == rank_before
    assert diag["discarded_tail_sq"] == pytest.approx(0.0, abs=1e-10)
    X_hat = tt_reconstruct(new_cores)
    assert torch.allclose(X_hat, X, atol=1e-8)


def test_tt_truncate_bond_does_not_mutate_input():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    saved = _clone_cores(cores)

    tt_truncate_bond(cores, 1, 1)

    _assert_cores_equal(cores, saved)


@pytest.mark.parametrize("bond", [-1, 3, 10])
def test_tt_truncate_bond_rejects_out_of_range_bond(bond):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(ValueError):
        tt_truncate_bond(cores, bond, 1)


@pytest.mark.parametrize("max_rank", [0, -1])
def test_tt_truncate_bond_rejects_non_positive_max_rank(max_rank):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(ValueError):
        tt_truncate_bond(cores, 1, max_rank)


@pytest.mark.parametrize("max_rank", [True, False])
def test_tt_truncate_bond_rejects_bool_max_rank(max_rank):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_truncate_bond(cores, 1, max_rank)


@pytest.mark.parametrize("max_rank", [1.5, "1", None])
def test_tt_truncate_bond_rejects_invalid_max_rank_type(max_rank):
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_truncate_bond(cores, 1, max_rank)


# ---------------------------------------------------------------------------
# tt_round
# ---------------------------------------------------------------------------


def test_tt_round_respects_relative_error_tolerance():
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    eps_rel = 1e-2

    rounded, diag = tt_round(cores, eps_rel)
    X_hat = tt_reconstruct(rounded)

    norm_x = torch.linalg.vector_norm(X).item()
    dense_error = torch.linalg.vector_norm(X - X_hat).item()

    tolerance = 1e-9 * max(norm_x, 1.0)
    assert dense_error <= eps_rel * norm_x + tolerance
    assert dense_error == pytest.approx(diag["global_error_estimate"], abs=1e-8)


def test_tt_round_does_not_increase_rank():
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    input_ranks = tt_ranks(cores)

    _, diag = tt_round(cores, 1e-2)

    assert len(diag["output_ranks"]) == len(input_ranks)
    for before, after in zip(input_ranks, diag["output_ranks"]):
        assert after <= before


def test_tt_round_reduces_rank_for_compressible_input():
    cores = _padded_redundant_tt()
    input_ranks = tt_ranks(cores)

    rounded, diag = tt_round(cores, 1e-8)

    assert tuple(diag["output_ranks"]) != tuple(input_ranks)
    assert tt_num_parameters(rounded) < tt_num_parameters(cores)

    X = tt_reconstruct(cores)
    X_hat = tt_reconstruct(rounded)
    assert torch.linalg.vector_norm(X - X_hat).item() < 1e-6


def test_tt_round_with_eps_zero_preserves_dense_and_does_not_increase_rank():
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    input_ranks = tt_ranks(cores)

    rounded, diag = tt_round(cores, 0.0)
    X_hat = tt_reconstruct(rounded)

    assert torch.allclose(X_hat, X, atol=1e-8)
    for before, after in zip(input_ranks, diag["output_ranks"]):
        assert after <= before
    assert diag["global_error_estimate"] == 0.0


def test_tt_round_handles_zero_tensor():
    physical_shape = (4, 5, 3)
    cores = [
        torch.zeros(1, n, 1, dtype=torch.float64) for n in physical_shape
    ]

    rounded, diag = tt_round(cores, 1e-8)
    X_hat = tt_reconstruct(rounded)

    assert torch.allclose(X_hat, torch.zeros_like(X_hat))
    assert all(r == 1 for r in diag["output_ranks"])
    assert tt_physical_shape(rounded) == physical_shape
    assert diag["norm_x"] == 0.0
    assert diag["global_error_estimate"] == 0.0


def test_tt_round_handles_d_equal_one():
    torch.manual_seed(0)
    core = torch.randn(1, 7, 1, dtype=torch.float64)
    rounded, diag = tt_round([core], 1e-2)

    assert len(rounded) == 1
    assert torch.equal(rounded[0], core)
    assert diag["local_errors"] == []
    assert diag["local_tail_sq"] == []


def test_tt_round_does_not_mutate_input():
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    saved = _clone_cores(cores)

    tt_round(cores, 1e-2)

    _assert_cores_equal(cores, saved)


def test_tt_round_diagnostics_contains_required_keys():
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    _, diag = tt_round(cores, 1e-2)

    required_keys = {
        "input_ranks",
        "output_ranks",
        "norm_x",
        "eps_requested",
        "delta",
        "local_errors",
        "local_tail_sq",
        "global_error_estimate",
        "relative_error_estimate",
    }
    assert required_keys.issubset(diag.keys())
    assert len(diag["local_errors"]) == len(cores) - 1
    assert len(diag["local_tail_sq"]) == len(cores) - 1
    assert diag["global_error_estimate"] == pytest.approx(
        math.sqrt(sum(diag["local_tail_sq"])), abs=1e-12
    )


@pytest.mark.parametrize("eps_rel", [-1e-3, float("nan"), float("inf")])
def test_tt_round_rejects_invalid_eps_rel_values(eps_rel):
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    with pytest.raises(ValueError):
        tt_round(cores, eps_rel)


@pytest.mark.parametrize("eps_rel", [True, False])
def test_tt_round_rejects_bool_eps_rel(eps_rel):
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_round(cores, eps_rel)


@pytest.mark.parametrize("eps_rel", ["0.1", None, [0.1]])
def test_tt_round_rejects_invalid_eps_rel_type(eps_rel):
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_round(cores, eps_rel)


@pytest.mark.parametrize("eps_rel", [torch.tensor(True), torch.tensor(False)])
def test_tt_round_rejects_zero_dim_bool_tensor_eps_rel(eps_rel):
    """0次元bool TensorはTrue/False経由で1.0/0.0として密かに受理されてはならない。"""
    X = _tensor(seed=4)
    cores = tt_svd_exact(X)
    with pytest.raises(TypeError):
        tt_round(cores, eps_rel)


def test_tt_round_does_not_treat_tiny_nonzero_float32_tensor_as_zero():
    """Bugbot反例: float32の ``diag(1e-30, 1e-30)`` を、``eps_rel=0.1`` で

    全零TTへ丸めてはならない（回帰: ``torch.linalg.vector_norm`` が内部で
    先に二乗するため、``1e-30`` の二乗 ``1e-60`` がfloat32でunderflowして
    厳密な0になり、zero-tensor分岐へ誤って入っていた）。
    """
    X = torch.diag(torch.tensor([1e-30, 1e-30], dtype=torch.float32))
    cores = tt_svd_exact(X)

    rounded, diag = tt_round(cores, 0.1)

    assert diag["norm_x"] > 0.0
    X_hat = tt_reconstruct(rounded)
    assert not torch.equal(X_hat, torch.zeros_like(X_hat))

    # 相対誤差は、float64へ変換したoracleで比較する（実装と同じscale-safe
    # helperをoracleに使わない）。
    dense_error = torch.linalg.vector_norm((X.double() - X_hat.double())).item()
    norm_x_oracle = torch.linalg.vector_norm(X.double()).item()
    assert norm_x_oracle > 0.0
    relative_error = dense_error / norm_x_oracle
    assert relative_error <= 0.1 + 1e-6


def test_tt_round_zero_branch_only_triggers_for_exact_zero_tensor():
    """本当の零TTだけがzero-tensor分岐へ入ることを確認する（衝突ケース）。"""
    physical_shape = (3, 3)
    zero_cores = [torch.zeros(1, n, 1, dtype=torch.float32) for n in physical_shape]

    rounded, diag = tt_round(zero_cores, 0.1)

    assert diag["norm_x"] == 0.0
    assert all(r == 1 for r in diag["output_ranks"])
    X_hat = tt_reconstruct(rounded)
    assert torch.equal(X_hat, torch.zeros_like(X_hat))


def test_tt_round_does_not_overshoot_tolerance_with_dominant_singular_value():
    """特異値が[1, 1e-8]のように大きく偏る場合、非常に小さいeps_relを

    要求すると、小さい方の特異値も正しく保持されなければならない
    （回帰: 打ち切り判定のslackが全体エネルギーに比例していたため、
    ``budget_sq``が小さいと``1e-8``側の特異値まるごとを誤って
    「丸め誤差の範囲内」と誤判定し、要求精度を桁違いに超過していた）。
    """
    X = torch.diag(torch.tensor([1.0, 1e-8], dtype=torch.float64))
    cores = tt_svd_exact(X)
    assert cores[0].shape[-1] == 2  # 打ち切り前は両方の特異値を保持している

    eps_rel = 1e-12
    rounded, diag = tt_round(cores, eps_rel)
    X_hat = tt_reconstruct(rounded)

    # 要求精度よりずっと粗い1e-8のオーダーへ緩められた誤差を出してはいけない。
    relative_error = (
        torch.linalg.vector_norm(X - X_hat).item()
        / torch.linalg.vector_norm(X).item()
    )
    assert relative_error <= eps_rel * 10  # 数値的な安全マージンのみ許容
    # 実際にはrankを落とさず両方の特異値を維持できるはず。
    assert rounded[0].shape[-1] == 2


def test_tt_round_huge_scale_does_not_overflow_error_budget():
    """Bugbotのoverflow反例: ``diag(3e155, 3e155)``、``eps_rel=0.1``。

    旧実装は ``budget_sq = delta**2`` を計算していたため、``delta``自体は
    有限でも二乗すると ``OverflowError`` になり得た（float64の表現上限
    ``~1.8e308`` を超えるため）。新実装は ``delta`` を二乗せず
    ``_choose_truncation_rank`` へ渡すので、``OverflowError`` にならず、
    rank選択が完了し、diagnosticsが不正なNaNにならないことを確認する。
    """
    X = torch.diag(torch.tensor([3e155, 3e155], dtype=torch.float64))
    cores = tt_svd_exact(X)

    rounded, diag = tt_round(cores, 0.1)

    assert math.isfinite(diag["norm_x"])
    assert math.isfinite(diag["delta"])
    assert not math.isnan(diag["global_error_estimate"])
    assert not math.isnan(diag["relative_error_estimate"])
    for tail_sq in diag["local_tail_sq"]:
        assert not math.isnan(tail_sq)

    X_hat = tt_reconstruct(rounded)
    assert torch.isfinite(X_hat).all()


def test_tt_round_tiny_scale_does_not_underflow_error_budget_to_zero():
    """Bugbotのunderflow反例: ``diag(1, 1e-200)``、``eps_rel=1e-190``。

    旧実装は ``delta**2`` がfloat64でunderflowして0になり、rank選択へ
    正しい予算を渡せなかった。新実装では``delta``を二乗せずに渡すので、
    ``1e-200`` 側の特異値を、要求予算``1e-190``の範囲内として正しく
    破棄できることを確認する（``1e-200 <= 1e-190`` なので破棄可能）。
    """
    X = torch.diag(torch.tensor([1.0, 1e-200], dtype=torch.float64))
    cores = tt_svd_exact(X)

    rounded, diag = tt_round(cores, 1e-190)

    assert diag["delta"] > 0.0  # deltaそのものが0になっていない
    assert not math.isnan(diag["global_error_estimate"])
    # 1e-200は予算(delta≈1e-190/sqrt(1))内なので破棄されるはず。
    assert rounded[0].shape[-1] == 1


# ---------------------------------------------------------------------------
# _choose_truncation_rank: budget（非二乗）を直接渡す新contract
# ---------------------------------------------------------------------------


def test_choose_truncation_rank_does_not_discard_representable_tiny_singular_value_float32():
    """Bugbot反例: ``singular_values=[1, 1e-30]``（float32）、``budget=1e-35``。

    真の捨てる誤差は ``1e-30`` で、要求予算 ``1e-35`` を大幅に超えるので、
    rankを落としてはならない（回帰: ``singular_values[k].square()`` が
    float32で ``1e-60`` へunderflowし、tail-squareが0と誤評価されて
    誤って捨てられていた）。
    """
    singular_values = torch.tensor([1.0, 1e-30], dtype=torch.float32)
    budget = 1e-35

    info = _choose_truncation_rank(singular_values, budget)

    assert info["new_rank"] == 2  # 何も捨てない
    assert info["tail_sq"] == pytest.approx(0.0, abs=1e-70)
    assert info["local_error"] == pytest.approx(0.0, abs=1e-35)


def test_choose_truncation_rank_still_discards_when_budget_exceeds_true_tail():
    """予算が実際のtailより大きい場合は、従来どおりrankを削減する。"""
    singular_values = torch.tensor([1.0, 1e-30], dtype=torch.float32)
    budget = 1e-20  # 真のtail(1e-30)よりずっと大きい予算

    info = _choose_truncation_rank(singular_values, budget)

    assert info["new_rank"] == 1
    assert info["local_error"] == pytest.approx(1e-30, rel=1e-2)
    assert info["tail_sq"] == pytest.approx(1e-60, rel=1e-1)


def test_choose_truncation_rank_handles_float64_tiny_singular_value():
    singular_values = torch.tensor([1.0, 1e-150], dtype=torch.float64)
    budget = 1e-160  # 予算は真のtailよりずっと小さい

    info = _choose_truncation_rank(singular_values, budget)

    assert info["new_rank"] == 2  # 捨てない


def test_choose_truncation_rank_normal_scale_matches_naive_computation():
    """通常スケールでは、素朴な二乗和と一致すること（回帰しない）。"""
    torch.manual_seed(0)
    singular_values = torch.sort(torch.rand(6, dtype=torch.float64), descending=True).values
    budget = 0.05 * float(singular_values[-1].item())

    info = _choose_truncation_rank(singular_values, budget)

    naive_tail_sq = 0.0
    for k in range(singular_values.shape[0] - 1, info["new_rank"] - 1, -1):
        naive_tail_sq += float(singular_values[k].item()) ** 2
    assert info["tail_sq"] == pytest.approx(naive_tail_sq, rel=1e-9, abs=1e-12)


def test_choose_truncation_rank_with_zero_budget_keeps_all_but_min_rank_logic():
    singular_values = torch.tensor([1.0, 0.5, 1e-10], dtype=torch.float64)
    info = _choose_truncation_rank(singular_values, budget=0.0)
    # budget=0なので、厳密に0の特異値以外は捨てられないはず。
    assert info["new_rank"] == 3


def test_choose_truncation_rank_skewed_singular_values_do_not_discard_needed_rank():
    """大きく偏った特異値でも、必要なrankを誤って落とさないこと。"""
    singular_values = torch.tensor([1e10, 1.0, 1e-30], dtype=torch.float64)
    budget = 1e-40  # 一番小さい特異値(1e-30)より遥かに小さい予算

    info = _choose_truncation_rank(singular_values, budget)

    assert info["new_rank"] == 3  # 何も捨てない


def test_choose_truncation_rank_rejects_negative_budget():
    singular_values = torch.tensor([1.0, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        _choose_truncation_rank(singular_values, budget=-1e-10)


def test_choose_truncation_rank_rejects_non_finite_budget():
    singular_values = torch.tensor([1.0, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        _choose_truncation_rank(singular_values, budget=float("inf"))
    with pytest.raises(ValueError):
        _choose_truncation_rank(singular_values, budget=float("nan"))


# ---------------------------------------------------------------------------
# _choose_truncation_rank: budget**2 のoverflow/underflow回帰（今回の修正2）
# ---------------------------------------------------------------------------


def test_choose_truncation_rank_huge_budget_does_not_overflow_when_squared_naively():
    """Bugbotのoverflow反例: budgetが ``3e155`` 程度でも ``OverflowError`` にならない。

    旧実装は呼び出し側で ``budget_sq = delta**2`` を計算してから渡していた
    ため、``delta≈3e155`` では ``delta**2≈9e310`` がfloat64の表現上限
    （``~1.8e308``）を超え ``inf`` になっていた。新contractでは ``budget``
    を二乗せずそのまま渡すので、この問題が起きないことを確認する。
    """
    singular_values = torch.tensor([3e155, 3e154], dtype=torch.float64)
    budget = 3e155 * 0.1  # 有限のbudget（deltaに相当）

    info = _choose_truncation_rank(singular_values, budget)

    assert math.isfinite(info["local_error"])
    assert not math.isnan(info["tail_sq"])
    assert info["new_rank"] in (1, 2)


def test_choose_truncation_rank_tiny_budget_does_not_underflow_to_zero_when_squared_naively():
    """Bugbotのunderflow反例: ``singular_values=[1, 1e-200]``、``budget=1e-190``。

    旧実装は ``budget_sq = budget**2 = 1e-380`` を渡していたが、float64の
    最小非正規化数（``~5e-324``）を下回りunderflowして厳密な0になり、
    rank選択に正しい予算を渡せなかった。新contractでは``budget``を
    二乗せずそのまま使うので、``1e-200 <= 1e-190``（真の破棄可能条件）を
    正しく判定できることを確認する。
    """
    singular_values = torch.tensor([1.0, 1e-200], dtype=torch.float64)
    budget = 1e-190

    info = _choose_truncation_rank(singular_values, budget)

    assert budget > 0.0  # budgetそのものが0になっていない
    assert info["new_rank"] == 1  # 1e-200は予算内として破棄できる
    assert info["local_error"] == pytest.approx(1e-200, rel=1e-6)


def test_choose_truncation_rank_strict_budget_does_not_discard_larger_tail():
    """strict budget反例: ``singular_values=[1, 1e-30]``、``budget=1e-35``。

    ``1e-30 > 1e-35`` なので、この特異値を破棄してはならない。
    """
    singular_values = torch.tensor([1.0, 1e-30], dtype=torch.float64)
    budget = 1e-35

    info = _choose_truncation_rank(singular_values, budget)

    assert info["new_rank"] == 2  # 破棄しない


# ---------------------------------------------------------------------------
# tt_truncate_bond diagnostics: 捨てたtailのscale-safeな報告
# ---------------------------------------------------------------------------


def test_tt_truncate_bond_diagnostics_are_scale_safe_for_tiny_float32_singular_values():
    """Bugbot反例: float32の特異値 ``[1e-30, 1e-30]`` をrank1へ切り詰める。

    真の局所誤差は ``1e-30`` であり、``discarded_tail_sq`` は
    ``1e-60`` 程度になるはず（回帰: ``S[rank_after:].square().sum()`` が
    float32の中で計算されるため厳密に0へunderflowしていた）。
    """
    core1 = torch.tensor([[1e-30, 0.0], [0.0, 1e-30]], dtype=torch.float32).reshape(1, 2, 2)
    core2 = torch.eye(2, dtype=torch.float32).reshape(2, 2, 1)
    cores = [core1, core2]

    _, diag = tt_truncate_bond(cores, 0, 1)

    # float64 oracle: 捨てた特異値をfloat64へ変換してから二乗する
    # （実装と同じprivate helperをoracleに使わない）。
    discarded_float64 = diag["singular_values"][diag["rank_after"] :].double()
    oracle_local_error = torch.linalg.vector_norm(discarded_float64).item()
    oracle_tail_sq = oracle_local_error**2

    assert diag["local_error"] == pytest.approx(oracle_local_error, rel=1e-6)
    assert diag["discarded_tail_sq"] == pytest.approx(oracle_tail_sq, rel=1e-3)
    assert diag["local_error"] == pytest.approx(1e-30, rel=1e-2)
    assert diag["discarded_tail_sq"] == pytest.approx(1e-60, rel=1e-1)
    assert diag["discarded_tail_sq"] > 0.0
    assert diag["local_error"] > 0.0


def test_tt_truncate_bond_diagnostics_match_float64_oracle_for_normal_scale():
    """通常スケールでは、float64 oracleとの一致を保つ（回帰しない）。"""
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)
    bond = 1
    max_rank = max(1, cores[bond].shape[-1] - 1)

    _, diag = tt_truncate_bond(cores, bond, max_rank)

    discarded_float64 = diag["singular_values"][diag["rank_after"] :].double()
    oracle_local_error = torch.linalg.vector_norm(discarded_float64).item()

    assert diag["local_error"] == pytest.approx(oracle_local_error, rel=1e-9, abs=1e-12)
    assert diag["discarded_tail_sq"] == pytest.approx(oracle_local_error**2, rel=1e-8, abs=1e-12)


# ---------------------------------------------------------------------------
# tt_bond_singular_values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bond", [0, 1, 2])
def test_tt_bond_singular_values_match_dense_cut_spectrum(bond):
    X = _tensor(seed=16)
    cores = tt_svd_exact(X)
    saved = _clone_cores(cores)

    actual = tt_bond_singular_values(cores, bond=bond)
    cut_matrix = X.reshape(math.prod(X.shape[: bond + 1]), -1)
    expected = torch.linalg.svdvals(cut_matrix)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual.dtype == X.dtype
    assert actual.device == X.device
    _assert_cores_equal(cores, saved)


@pytest.mark.parametrize("bond", [-1, 3])
def test_tt_bond_singular_values_rejects_out_of_range_bond(bond):
    cores = tt_svd_exact(_tensor(seed=17))
    with pytest.raises(ValueError):
        tt_bond_singular_values(cores, bond=bond)


@pytest.mark.parametrize("bond", [True, 1.5, "1", None])
def test_tt_bond_singular_values_rejects_non_integer_bond(bond):
    cores = tt_svd_exact(_tensor(seed=17))
    with pytest.raises(TypeError):
        tt_bond_singular_values(cores, bond=bond)


def test_tt_bond_singular_values_rejects_single_core_tt():
    core = torch.randn(1, 5, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        tt_bond_singular_values([core], bond=0)


def test_tt_bond_singular_values_preserves_autograd_graph():
    cores = [core.detach().requires_grad_() for core in tt_svd_exact(_tensor(seed=20))]

    singular_values = tt_bond_singular_values(cores, bond=1)
    singular_values.sum().backward()

    for core in cores:
        assert core.grad is not None
        assert torch.isfinite(core.grad).all()
