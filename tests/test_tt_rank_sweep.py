"""TT-matrix weight レベル rank sweep（``sweep_tt_matrix_weight_ranks``）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/07_tt_rank_bond_dimension_sweep_weight_tradeoff.ipynb
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nn_compression.compression import (
    dense_to_tt_matrix_cores,
    sweep_tt_matrix_weight_ranks,
    tt_matrix_num_parameters,
    tt_matrix_num_parameters_from_ranks,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from nn_compression.metrics import relative_frobenius_error

DTYPES = [torch.float32, torch.float64]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません"),
    ),
]

# site 数 d = 1, 2, 3, 4。
MODE_CASES = [
    pytest.param((6,), (4,), id="d1"),
    pytest.param((2, 3), (3, 2), id="d2"),
    pytest.param((2, 2, 2), (2, 2, 2), id="d3"),
    pytest.param((2, 2, 2, 2), (2, 2, 2, 2), id="d4"),
]

RECORD_KEYS = {
    "requested_rank",
    "actual_ranks",
    "tt_parameters",
    "dense_parameters",
    "compression_ratio",
    "absolute_error",
    "relative_error",
    "is_compressed",
}


def _weight(out_modes, in_modes, *, dtype=torch.float64, device="cpu", seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn(math.prod(out_modes), math.prod(in_modes), generator=generator, dtype=dtype)
    return W.to(device)


@pytest.mark.parametrize(("out_modes", "in_modes"), MODE_CASES)
def test_sweep_matches_existing_tt_matrix_functions(out_modes, in_modes):
    W = _weight(out_modes, in_modes)
    candidates = [1, 2, 3, 100]

    records = sweep_tt_matrix_weight_ranks(W, out_modes, in_modes, candidates)

    assert len(records) == len(candidates)
    for requested, record in zip(candidates, records):
        cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=requested)
        W_hat = tt_matrix_to_dense(cores, out_modes, in_modes)

        assert set(record) == RECORD_KEYS
        assert record["requested_rank"] == requested
        assert record["actual_ranks"] == tt_matrix_ranks(cores)
        assert record["tt_parameters"] == tt_matrix_num_parameters(cores)
        assert record["dense_parameters"] == W.numel()
        assert record["compression_ratio"] == pytest.approx(W.numel() / tt_matrix_num_parameters(cores))
        assert record["absolute_error"] == pytest.approx(
            float(torch.linalg.vector_norm(W - W_hat)), rel=1e-9, abs=1e-12
        )
        assert record["relative_error"] == pytest.approx(
            float(relative_frobenius_error(W, W_hat)), rel=1e-9, abs=1e-12
        )
        assert record["is_compressed"] == (tt_matrix_num_parameters(cores) < W.numel())


@pytest.mark.parametrize(("out_modes", "in_modes"), MODE_CASES)
def test_sweep_parameter_count_equals_formula_from_actual_ranks(out_modes, in_modes):
    W = _weight(out_modes, in_modes, seed=1)

    for record in sweep_tt_matrix_weight_ranks(W, out_modes, in_modes, [1, 2, 4, 64]):
        assert record["tt_parameters"] == tt_matrix_num_parameters_from_ranks(
            out_modes, in_modes, record["actual_ranks"]
        )


def test_sweep_hand_computed_three_core_case_matches_notebook_07_values():
    # Notebook 07 と同じ 8x8, modes (2,2,2) x (2,2,2): rank r の TT parameter 数は
    # 4*r + 8*r*r + 4*r（r=1,2 のとき）、full rank 4 では 96。
    W = _weight((2, 2, 2), (2, 2, 2))

    records = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [1, 2, 4])

    assert [r["actual_ranks"] for r in records] == [(1, 1, 1, 1), (1, 2, 2, 1), (1, 4, 4, 1)]
    assert [r["tt_parameters"] for r in records] == [12, 32, 96]
    assert [r["dense_parameters"] for r in records] == [64, 64, 64]
    assert records[0]["compression_ratio"] == pytest.approx(64 / 12)
    assert records[1]["compression_ratio"] == pytest.approx(2.0)
    assert records[2]["compression_ratio"] == pytest.approx(64 / 96)


def test_sweep_records_requested_and_actual_rank_separately():
    # 8x8, 3 core: bond rank の上限は 4。requested=100 でも actual は 4。
    W = _weight((2, 2, 2), (2, 2, 2))

    record = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [100])[0]

    assert record["requested_rank"] == 100
    assert record["actual_ranks"] == (1, 4, 4, 1)
    assert all(isinstance(r, int) for r in record["actual_ranks"])

    # d=2, 6x6 (2*3, 3*2): bond は min(6, 6) = 6 まで。requested=2 のときは 2。
    W2 = _weight((2, 3), (3, 2), seed=2)
    low, high = sweep_tt_matrix_weight_ranks(W2, (2, 3), (3, 2), [2, 50])
    assert low["actual_ranks"] == (1, 2, 1)
    assert high["requested_rank"] == 50 and high["actual_ranks"][1] <= 6


def test_sweep_full_rank_records_tt_parameters_above_dense_and_negligible_error():
    W = _weight((2, 2, 2), (2, 2, 2))

    low, full = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [1, 4])

    assert low["is_compressed"] is True
    assert low["compression_ratio"] > 1.0
    assert full["tt_parameters"] > full["dense_parameters"]
    assert full["is_compressed"] is False
    assert full["compression_ratio"] < 1.0
    assert full["relative_error"] < 1e-12
    assert full["absolute_error"] < 1e-11


def test_sweep_one_site_tt_matrix_is_uncompressed_and_exact_for_every_rank():
    W = _weight((6,), (4,))

    records = sweep_tt_matrix_weight_ranks(W, (6,), (4,), [1, 3, 99])

    for record in records:
        assert record["actual_ranks"] == (1, 1)
        assert record["tt_parameters"] == record["dense_parameters"] == 24
        assert record["is_compressed"] is False  # 等しいだけでは圧縮ではない
        assert record["compression_ratio"] == 1.0
        assert record["absolute_error"] == 0.0
        assert record["relative_error"] == 0.0


def test_sweep_keeps_candidate_order_without_sorting_or_deduplicating():
    W = _weight((2, 2, 2), (2, 2, 2))

    records = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [4, 1, 2, 2])

    assert [r["requested_rank"] for r in records] == [4, 1, 2, 2]
    assert records[2] == records[3]


def test_sweep_relative_error_is_absolute_error_over_weight_norm():
    W = _weight((2, 2, 2), (2, 2, 2), seed=3)
    norm = float(torch.linalg.vector_norm(W))

    for record in sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [1, 2, 3]):
        assert record["relative_error"] == pytest.approx(record["absolute_error"] / norm, rel=1e-9)


def test_sweep_returns_plain_python_records_without_tensors():
    W = _weight((2, 2), (2, 2))

    records = sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), [1, 2])

    assert isinstance(records, list) and all(type(r) is dict for r in records)
    for record in records:
        assert type(record["requested_rank"]) is int
        assert type(record["actual_ranks"]) is tuple
        assert type(record["tt_parameters"]) is int
        assert type(record["dense_parameters"]) is int
        assert type(record["compression_ratio"]) is float
        assert type(record["absolute_error"]) is float
        assert type(record["relative_error"]) is float
        assert type(record["is_compressed"]) is bool


@pytest.mark.parametrize("dtype", DTYPES)
def test_sweep_supports_float32_and_float64(dtype):
    W = _weight((2, 2, 2), (2, 2, 2), dtype=dtype)

    records = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [1, 2, 4])

    tolerance = 1e-4 if dtype == torch.float32 else 1e-12
    assert records[-1]["relative_error"] < tolerance
    assert records[0]["relative_error"] > records[-1]["relative_error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_sweep_on_cuda_matches_cpu_results():
    W_cpu = _weight((2, 2, 2), (2, 2, 2))

    cpu = sweep_tt_matrix_weight_ranks(W_cpu, (2, 2, 2), (2, 2, 2), [1, 2, 4])
    cuda = sweep_tt_matrix_weight_ranks(W_cpu.to("cuda"), (2, 2, 2), (2, 2, 2), [1, 2, 4])

    for a, b in zip(cpu, cuda):
        assert a["actual_ranks"] == b["actual_ranks"]
        assert a["tt_parameters"] == b["tt_parameters"]
        assert a["relative_error"] == pytest.approx(b["relative_error"], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_sweep_does_not_mutate_input_or_change_dtype_device(device):
    W = _weight((2, 2, 2), (2, 2, 2), dtype=torch.float32, device=device)
    saved = W.clone()

    sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [1, 2])

    assert torch.equal(W, saved)
    assert W.dtype == torch.float32
    assert W.device.type == torch.device(device).type


def test_sweep_accepts_weight_requiring_grad_without_building_graph():
    W = _weight((2, 2, 2), (2, 2, 2)).requires_grad_(True)

    records = sweep_tt_matrix_weight_ranks(W, (2, 2, 2), (2, 2, 2), [2])

    assert W.grad is None and W.requires_grad
    assert type(records[0]["relative_error"]) is float


def test_sweep_does_not_mutate_candidate_sequence_and_accepts_sequence_types():
    candidates = [2, 1]

    sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)), (2, 2), (2, 2), candidates)
    assert candidates == [2, 1]

    W = _weight((2, 2), (2, 2))
    from_tuple = sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), (1, 2))
    from_range = sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), range(1, 3))
    from_numpy_ints = sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), [np.int64(1), np.int32(2)])
    assert from_tuple == from_range == from_numpy_ints
    assert type(from_numpy_ints[0]["requested_rank"]) is int


def test_sweep_is_not_limited_to_three_cores():
    for out_modes, in_modes in [((16,), (16,)), ((4, 4), (4, 4)), ((2, 2, 2, 2), (2, 2, 2, 2))]:
        W = _weight(out_modes, in_modes, seed=5)
        record = sweep_tt_matrix_weight_ranks(W, out_modes, in_modes, [2])[0]
        assert len(record["actual_ranks"]) == len(out_modes) + 1


# --- 検証 -------------------------------------------------------------------


def test_sweep_rejects_zero_weight():
    with pytest.raises(ValueError, match="ゼロ"):
        sweep_tt_matrix_weight_ranks(torch.zeros(4, 4, dtype=torch.float64), (2, 2), (2, 2), [1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_sweep_rejects_non_finite_weight(bad):
    W = _weight((2, 2), (2, 2))
    W[0, 0] = bad
    with pytest.raises(ValueError, match="NaN / inf"):
        sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), [1])


def test_sweep_rejects_weight_shape_and_mode_problems():
    W = _weight((2, 2), (2, 2))
    with pytest.raises(ValueError):
        sweep_tt_matrix_weight_ranks(W, (2, 3), (2, 2), [1])  # prod 不一致
    with pytest.raises(ValueError):
        sweep_tt_matrix_weight_ranks(W, (2, 2), (4,), [1])  # modes 長さ不一致
    with pytest.raises(ValueError):
        sweep_tt_matrix_weight_ranks(W.reshape(16), (2, 2), (2, 2), [1])  # 2階でない
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks(W, (2, True), (2, 2), [1])


def test_sweep_rejects_unsupported_dtype_and_non_tensor():
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)).half(), (2, 2), (2, 2), [1])
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks(torch.ones(4, 4, dtype=torch.int64), (2, 2), (2, 2), [1])
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks([[1.0] * 4] * 4, (2, 2), (2, 2), [1])


def test_sweep_rejects_empty_rank_candidates():
    with pytest.raises(ValueError, match="空"):
        sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)), (2, 2), (2, 2), [])


@pytest.mark.parametrize("bad", [0, -1])
def test_sweep_rejects_non_positive_rank_candidates(bad):
    with pytest.raises(ValueError):
        sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)), (2, 2), (2, 2), [1, bad])


@pytest.mark.parametrize("bad", [True, 1.0, "2", None])
def test_sweep_rejects_non_integer_or_bool_rank_candidates(bad):
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)), (2, 2), (2, 2), [1, bad])


@pytest.mark.parametrize("bad", [3, "12", None, 2.0])
def test_sweep_rejects_non_sequence_rank_candidates(bad):
    with pytest.raises(TypeError):
        sweep_tt_matrix_weight_ranks(_weight((2, 2), (2, 2)), (2, 2), (2, 2), bad)


# ---------------------------------------------------------------------------
# Bugbot: float32 最大値付近でも rank / 誤差が overflow で壊れないこと
# ---------------------------------------------------------------------------


def _extreme_float32_weight(out_features: int, in_features: int, *, device="cpu") -> torch.Tensor:
    """符号の異なる finfo.max 近傍の有限行列。正規化すれば通常スケールに戻る。"""
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(out_features, in_features, generator=generator, dtype=torch.float32)
    base = base / base.abs().amax()
    base[0, 0] = 1
    base[0, 1] = -1
    return (base * torch.finfo(torch.float32).max).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("out_modes", "in_modes"),
    [((2, 2), (2, 2)), ((2, 2, 2), (2, 2, 2))],
)
def test_sweep_float32_near_max_matches_normalized_and_stays_finite(out_modes, in_modes, device):
    big = _extreme_float32_weight(math.prod(out_modes), math.prod(in_modes), device=device)
    saved = big.clone()
    normalized = big / big.abs().amax()
    candidates = [1, 2, 4, 100]

    big_records = sweep_tt_matrix_weight_ranks(big, out_modes, in_modes, candidates)
    ref_records = sweep_tt_matrix_weight_ranks(normalized, out_modes, in_modes, candidates)

    assert torch.equal(big, saved)
    assert big.dtype == torch.float32
    assert big.device.type == torch.device(device).type
    scale = float(big.abs().amax())
    for big_record, ref_record in zip(big_records, ref_records):
        assert set(big_record) == RECORD_KEYS
        assert big_record["actual_ranks"] == ref_record["actual_ranks"]
        assert big_record["tt_parameters"] == ref_record["tt_parameters"]
        assert big_record["dense_parameters"] == big.numel()
        assert big_record["compression_ratio"] == ref_record["compression_ratio"]
        assert big_record["is_compressed"] == ref_record["is_compressed"]
        assert math.isfinite(big_record["absolute_error"])
        assert math.isfinite(big_record["relative_error"])
        assert not math.isnan(big_record["absolute_error"])
        assert not math.isnan(big_record["relative_error"])
        assert big_record["relative_error"] == pytest.approx(
            ref_record["relative_error"], rel=1e-5, abs=1e-6
        )
        assert big_record["absolute_error"] == pytest.approx(
            ref_record["absolute_error"] * scale, rel=1e-4, abs=1e-3
        )

    full = big_records[-1]
    assert full["requested_rank"] == 100
    assert full["actual_ranks"] == ref_records[-1]["actual_ranks"]
    assert max(full["actual_ranks"]) > 1
    assert full["relative_error"] < 1e-3


def test_sweep_one_site_extreme_weight_stays_exact():
    fmax = torch.finfo(torch.float32).max
    W = torch.tensor([[fmax, -fmax], [-0.5 * fmax, 0.25 * fmax]], dtype=torch.float32)

    records = sweep_tt_matrix_weight_ranks(W, (2,), (2,), [1, 7])

    for record in records:
        assert record["actual_ranks"] == (1, 1)
        assert record["absolute_error"] == 0.0
        assert record["relative_error"] == 0.0
        assert math.isfinite(record["compression_ratio"])


def test_sweep_absolute_error_beyond_float64_is_inf_not_nan():
    """真の絶対誤差が Python float の範囲を超えるときは inf。NaN にはしない。"""
    fmax = torch.finfo(torch.float64).max
    W = torch.tensor(
        [
            [fmax, -fmax, fmax, -fmax],
            [-fmax, fmax, -fmax, fmax],
            [fmax, fmax, -fmax, -fmax],
            [-fmax, -fmax, fmax, fmax],
        ],
        dtype=torch.float64,
    )
    saved = W.clone()

    low, full = sweep_tt_matrix_weight_ranks(W, (2, 2), (2, 2), [1, 100])

    assert torch.equal(W, saved)
    assert low["absolute_error"] == float("inf")
    assert not math.isnan(low["relative_error"])
    assert math.isfinite(low["relative_error"])
    assert low["relative_error"] > 0
    assert len(full["actual_ranks"]) == 3
    assert full["actual_ranks"][0] == 1 and full["actual_ranks"][-1] == 1
    assert not math.isnan(full["absolute_error"])
    assert math.isfinite(full["relative_error"])


def test_sweep_zero_weight_is_rejected_before_scaled_svd():
    with pytest.raises(ValueError, match="ゼロ"):
        sweep_tt_matrix_weight_ranks(torch.zeros(4, 4, dtype=torch.float32), (2, 2), (2, 2), [1])
