"""層種別・モデル構造に依存しない HOSVD / Tucker の共通処理。

参照元:
    notebooks/20_tucker/00_fundamentals/00_tucker_hosvd_basics.ipynb
    notebooks/20_tucker/00_fundamentals/01_rank_error_tradeoff.ipynb

mode-n 演算そのものは ``nn_compression.tensor.operations``、
再構成誤差は ``nn_compression.metrics`` に置く。ここは
「rank を決めて分解し、保存要素数を数える」までを担当する。

Conv2d への Tucker-2 適用（``conv_tucker``）は別モジュールにする。
"""

from __future__ import annotations

import torch

from ..tensor.operations import mode_dot, unfold
from .svd import truncated_svd


def hosvd(
    X: torch.Tensor,
    ranks: dict[int, int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """ranks で指定した mode ごとに低 rank 近似し、core と factor を返す。

    X: Tucker分解するテンソル
    ranks: {mode: rank} 形式の辞書。
           keyが分解対象のmode、valueがそのmodeで残すrankを表す。

    返り値:
        core: 指定したmodeを低rank化したcore tensor
        factors: {mode: factor_matrix} 形式の辞書

    ``ranks`` に全 mode を含めれば truncated HOSVD、一部の mode だけを
    含めれば partial HOSVD になる（未指定 mode は core に元の dimension
    が残る）。factor は HOOI のような反復ではなく、いずれも **元の X の
    unfolding** から 1 回で求める。逐次更新した core から取ると別の
    アルゴリズムになるため、この順序を変えない。
    """
    factors: dict[int, torch.Tensor] = {}
    core = X
    for mode, rank in ranks.items():
        factors[mode], _, _ = truncated_svd(unfold(X, mode), rank)
        # truncated_svd が返す Uのshapeは、[:, :rank]
        # 元のmodeの大きさを縮めるには、Tをかける
        core = mode_dot(core, factors[mode].T, mode)
    return core, factors


def reconstruct_tucker(
    core: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor:
    """core tensor と factor matrix から、元テンソルの近似を再構成する。

    全mode分解と部分mode分解の両方を扱う。

    ``factors`` は ``{mode: U}`` なので、未指定 mode は掛け戻さず
    core の dimension がそのまま残る。
    """
    original = core
    for mode, U in factors.items():
        original = mode_dot(original, U, mode)

    return original


def tucker_parameter_count(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> int:
    """指定 rank で Tucker 表現を作ったときの総要素数を返す。

    shape: 元テンソルのshape
    ranks: {mode: rank} 形式の辞書

    ``core の要素数 + Σ_{分解した mode} I_n * r_n``。
    ``ranks`` に無い mode は partial HOSVD と同じく、core に元の
    dimension をそのまま残す（factor も持たない）。
    """
    core_count = 1
    factor_count = 0

    for mode, dim in enumerate(shape):
        if mode in ranks:
            rank = ranks[mode]

            core_count *= rank
            factor_count += dim * rank
        else:
            core_count *= dim

    return core_count + factor_count


def parameter_ratio(shape: tuple[int, ...], ranks: dict[int, int]) -> float:
    """Tucker 表現の要素数が元テンソルの何割かを返す。小さいほど圧縮。"""
    full_param_count = 1
    for param in shape:
        full_param_count = param * full_param_count
    reduced_param_count = tucker_parameter_count(shape, ranks)
    return reduced_param_count / full_param_count


def compression_factor(shape: tuple[int, ...], ranks: dict[int, int]) -> float:
    """元テンソルに対して何倍小さい表現かを返す。大きいほど圧縮。"""
    full_param_count = 1
    for param in shape:
        full_param_count = param * full_param_count
    reduced_param_count = tucker_parameter_count(shape, ranks)
    return full_param_count / reduced_param_count
