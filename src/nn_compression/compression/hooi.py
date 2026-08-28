"""層種別・モデル構造に依存しない HOOI (Higher-Order Orthogonal Iteration)。

参照元:
    notebooks/20_tucker/00_fundamentals/02_hooi.ipynb
    notebooks/20_tucker/10_cifar10_cnn/04_hooi_tucker2.ipynb
    notebooks/20_tucker/10_cifar10_cnn/05_hosvd_vs_hooi_finetuning.ipynb

HOSVD で得た factor を初期値に、指定 mode だけ反復更新する。
``ranks`` の key が「更新・圧縮対象の mode」を表す partial HOOI も扱う。

Conv2d weight ``(C_out, C_in, kH, kW)`` に ``ranks={0: rank_out, 1: rank_in}``
を渡せば Tucker-2 HOOI（空間 mode kH / kW は圧縮しない）になる。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..metrics import relative_frobenius_error
from ..tensor.operations import mode_dot, unfold
from ..tensor.validation import validate_tensor_ndim_at_least_2
from .svd import truncated_svd
from .tucker import hosvd, reconstruct_tucker
from .tucker_validation import (
    validate_max_iter,
    validate_mode_index,
    validate_non_negative_tolerance,
    validate_real_dtype,
    validate_tucker_ranks,
)


def _projected_unfold_max_rank(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
    target_mode: int,
) -> int:
    """target_mode 更新時の truncated SVD で取りうる最大 rank。"""
    projected = X
    for other_mode in ranks:
        if other_mode == target_mode:
            continue
        projected = mode_dot(projected, factors[other_mode].T, other_mode)
    unfolded = unfold(projected, target_mode)
    return min(unfolded.shape[0], unfolded.shape[1])


def _validate_hooi_inputs(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> None:
    """HOOI 入出力の mode / rank / factor shape を検証する。"""
    validate_tensor_ndim_at_least_2(X, name="X")
    validate_real_dtype(X, name="X")
    if not isinstance(ranks, Mapping):
        raise TypeError(
            f"ranks は Mapping である必要があります: {type(ranks)!r}"
        )
    if len(ranks) == 0:
        raise ValueError("ranks は空にできません。")

    validated_ranks = validate_tucker_ranks(
        tuple(X.shape), ranks, allow_empty=False
    )

    for mode in factors:
        validate_mode_index(mode, X.ndim, name="factor mode")

    factor_modes = set(factors.keys())
    rank_modes = set(validated_ranks.keys())
    if factor_modes != rank_modes:
        extra = sorted(factor_modes - rank_modes)
        missing = sorted(rank_modes - factor_modes)
        if extra:
            raise ValueError(
                f"factors に ranks に無い mode があります: {extra}"
            )
        if missing:
            raise ValueError(
                f"factors に mode={missing[0]} がありません。"
                f" ranks の各 mode に対応する factor が必要です。"
            )

    for mode, rank in validated_ranks.items():
        expected_rank = rank
        U = factors[mode]
        if U.ndim != 2:
            raise ValueError(
                f"factors[{mode}] は2次元行列である必要があります: "
                f"shape={tuple(U.shape)}"
            )
        if U.shape != (X.shape[mode], expected_rank):
            raise ValueError(
                f"factors[{mode}].shape={tuple(U.shape)} は "
                f"({X.shape[mode]}, {expected_rank}) である必要があります。"
            )
        if U.dtype != X.dtype:
            raise ValueError(
                f"factors[{mode}].dtype={U.dtype} は X.dtype={X.dtype} "
                "と一致する必要があります。"
            )
        if U.device != X.device:
            raise ValueError(
                f"factors[{mode}].device={U.device} は X.device={X.device} "
                "と一致する必要があります。"
            )

    for target_mode in validated_ranks:
        max_rank = _projected_unfold_max_rank(
            X, factors, validated_ranks, target_mode
        )
        if validated_ranks[target_mode] > max_rank:
            raise ValueError(
                f"rank[{target_mode}]={validated_ranks[target_mode]} は "
                f"projected unfolding で実現可能な上限 {max_rank} を超えています。"
            )


def has_converged(
    error: float,
    prev_error: float,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> bool:
    """誤差の変化が十分小さいか（収束したか）を判定する。

    ``|error - prev_error| <= abs_tol + rel_tol * |prev_error|``
    なら True。``abs_tol`` は絶対許容、``rel_tol`` は前回誤差に比例する相対許容。
    """
    abs_tol = validate_non_negative_tolerance(abs_tol, name="abs_tol")
    rel_tol = validate_non_negative_tolerance(rel_tol, name="rel_tol")
    absolute_change = abs(error - prev_error)
    return absolute_change <= abs_tol + rel_tol * abs(prev_error)


def hooi_sweep(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> dict[int, torch.Tensor]:
    """``ranks`` に含まれる mode だけ factor を 1 回ずつ更新する（1 sweep）。

    各 ``target_mode`` について、**target 自身以外**の最新 factor で
    ``X`` を mode-n 積で縮約し、unfold → truncated SVD で factor を更新する。
    sweep 途中で更新済みの factor は、後続 mode の更新にそのまま使う。

    入力 ``factors`` は書き換えず、更新後の辞書を返す。
    """
    _validate_hooi_inputs(X, factors, ranks)

    updated_factors = {
        mode: U.clone()
        for mode, U in factors.items()
    }

    # ranks の挿入順で更新（Tucker-2 では mode 0 → mode 1）
    for target_mode in ranks:
        projected = X

        for other_mode in ranks:
            if other_mode == target_mode:
                # target 自身の factor はこの mode の projection に使わない
                continue
            U_other = updated_factors[other_mode]
            projected = mode_dot(projected, U_other.T, other_mode)

        unfolded = unfold(projected, target_mode)
        updated_U, _, _ = truncated_svd(unfolded, ranks[target_mode])
        updated_factors[target_mode] = updated_U

    return updated_factors


def core_from_factors(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor:
    """現在の factor から Tucker core を計算する。

    ``core = X ×_0 U_0^T ×_1 U_1^T × ...``。
    各 mode で元テンソルの dimension を rank まで落とした低 rank core になる。
    """
    validate_real_dtype(X, name="X")
    core = X
    for mode, U in factors.items():
        core = mode_dot(core, U.T, mode)
    return core


def hooi(
    X: torch.Tensor,
    ranks: dict[int, int],
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]:
    """HOSVD 初期化から HOOI を反復し、core / factors / 誤差履歴を返す。

    HOSVD は各 mode の factor を 1 回だけ求める初期値生成。
    その後 ``hooi_sweep`` で ``ranks`` に含まれる mode だけ反復更新する。

    返り値 ``history`` は相対 Frobenius 誤差の list。
    ``history[0]`` は HOSVD 初期値、以降は各 sweep 後の誤差。
    """
    validate_tensor_ndim_at_least_2(X, name="X")
    if not isinstance(ranks, Mapping):
        raise TypeError(
            f"ranks は Mapping である必要があります: {type(ranks)!r}"
        )
    if len(ranks) == 0:
        raise ValueError("ranks は空にできません。")
    if torch.linalg.vector_norm(X) == 0:
        raise ValueError(
            "X がゼロテンソルなので HOOI の相対 Frobenius 誤差を定義できません。"
        )

    max_iter = validate_max_iter(max_iter)
    abs_tol = validate_non_negative_tolerance(abs_tol, name="abs_tol")
    rel_tol = validate_non_negative_tolerance(rel_tol, name="rel_tol")

    core_hosvd, factors_hosvd = hosvd(X, ranks)
    _validate_hooi_inputs(X, factors_hosvd, ranks)
    X_hat = reconstruct_tucker(core_hosvd, factors_hosvd)
    error = float(relative_frobenius_error(X, X_hat).detach())

    history: list[float] = [error]
    prev_error = error
    updated_factors = {
        mode: U.clone()
        for mode, U in factors_hosvd.items()
    }
    core = core_hosvd

    for _ in range(max_iter):
        updated_factors = hooi_sweep(X, updated_factors, ranks)
        core = core_from_factors(X, updated_factors)
        X_hat = reconstruct_tucker(core, updated_factors)
        error = float(relative_frobenius_error(X, X_hat).detach())
        history.append(error)

        if has_converged(error, prev_error, abs_tol=abs_tol, rel_tol=rel_tol):
            break
        prev_error = error

    return core, updated_factors, history
