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

import torch

from ..metrics import relative_frobenius_error
from ..tensor.operations import mode_dot, unfold
from .svd import coerce_rank, truncated_svd
from .tucker import hosvd, reconstruct_tucker


def _validate_hooi_inputs(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> None:
    """HOOI 入出力の mode / rank / factor shape を検証する。"""
    if X.ndim < 1:
        raise ValueError(f"テンソルは1次元以上である必要があります: ndim={X.ndim}")
    if not ranks:
        raise ValueError("ranks は空にできません。")

    for mode, rank in ranks.items():
        if not isinstance(mode, int):
            raise TypeError(f"mode は整数である必要があります: {mode!r}")
        if not 0 <= mode < X.ndim:
            raise ValueError(
                f"mode={mode} は 0〜{X.ndim - 1} の範囲で指定してください。"
            )
        if mode not in factors:
            raise ValueError(
                f"factors に mode={mode} がありません。"
                f" ranks の各 mode に対応する factor が必要です。"
            )
        expected_rank = coerce_rank(rank, X.shape[mode], name=f"rank[{mode}]")
        U = factors[mode]
        if U.ndim != 2:
            raise ValueError(
                f"factors[{mode}] は2次元行列である必要があります: shape={tuple(U.shape)}"
            )
        if U.shape != (X.shape[mode], expected_rank):
            raise ValueError(
                f"factors[{mode}].shape={tuple(U.shape)} は "
                f"({X.shape[mode]}, {expected_rank}) である必要があります。"
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
    if max_iter < 0:
        raise ValueError(f"max_iter は 0 以上である必要があります: {max_iter}")

    core_hosvd, factors_hosvd = hosvd(X, ranks)
    X_hat = reconstruct_tucker(core_hosvd, factors_hosvd)
    error = float(relative_frobenius_error(X, X_hat))

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
        error = float(relative_frobenius_error(X, X_hat))
        history.append(error)

        if has_converged(error, prev_error, abs_tol=abs_tol, rel_tol=rel_tol):
            break
        prev_error = error

    return core, updated_factors, history
