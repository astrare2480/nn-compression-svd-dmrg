"""Tensor Train (TT) SVD の共通処理。

参照元:
    notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb
    notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb
    notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb

mode-n unfolding（``nn_compression.tensor.operations.unfold``）とは違い、
TT の cut unfolding は先頭 ``k`` 軸をまとめて残りと切り分ける。この違いを
混同しないよう、TT 専用の unfolding は ``tt_unfold`` としてここに独立させる。

``tt_svd_exact``、``tt_svd(max_rank)``、``tt_svd_ranks(ranks)`` は
exact / 共通上限 / bond ごとの上限の違いをコード上でも明示するために
別関数のままにするが、逐次SVDループ本体（``_tt_svd_sweep``）は共有し、
reshape ロジック自体は変更しない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..tensor.validation import validate_tensor_shape
from .svd import coerce_integer_scalar, truncated_svd
from .tt_validation import validate_tt_cores, validate_tt_cut_index, validate_tt_dtype


def tt_unfold(X: torch.Tensor, k: int) -> torch.Tensor:
    """TT の cut ``1:k | k+1:d`` に対応する2次元unfoldingを返す。

    X: unfoldingするテンソル（``ndim >= 2``）
    k: cut位置。``1 <= k < X.ndim`` を満たす必要がある。

    戻り値の shape は ``(n_1 ... n_k,  n_{k+1} ... n_d)``。
    """
    validate_tensor_shape(X)
    validate_tt_cut_index(k, X.ndim)

    shape = X.shape
    left_dim = math.prod(shape[:k])
    right_dim = math.prod(shape[k:])

    return X.reshape(left_dim, right_dim)


def _tt_svd_sweep(
    X: torch.Tensor,
    max_rank: int | None = None,
    bond_ranks: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    """``tt_svd_exact`` / ``tt_svd`` / ``tt_svd_ranks`` が共有する逐次SVDループ本体。

    ``max_rank=None`` かつ ``bond_ranks=None`` なら打ち切りなし
    （各段階の数値rankをそのまま使う）。
    ``max_rank`` に整数を渡すと、各段階で
    ``min(max_rank, その段階の数値rank)`` を bond dimension として使う。
    ``bond_ranks`` を渡すと、第 ``k`` 段階は
    ``min(bond_ranks[k], その段階の数値rank)`` を使う。
    ``max_rank`` と ``bond_ranks`` は同時に指定しない。

    reshape ``mat = remainder.reshape(r_left * n_mode, -1)`` は
    notebook 01/02 のアルゴリズムのまま変更しない。``tt_unfold`` では
    代用しない（``tt_unfold`` は元テンソルの cut unfolding rank を
    検証するための独立した関数）。

    零テンソルなど、ある段階の数値rankが0になる入力でも例外にしない。
    TTのbond dimensionは正の整数として扱うため、その段階のみ最低
    ``1`` を bond dimension として使う（``effective_rank`` 参照）。
    通常（数値rank >= 1）の場合はこれまでと挙動が変わらない。
    """
    validate_tensor_shape(X)
    validate_tt_dtype(X, name="X")

    if max_rank is not None and bond_ranks is not None:
        raise ValueError("max_rank と bond_ranks は同時に指定できません。")

    shape = X.shape
    d = X.ndim
    if bond_ranks is not None and len(bond_ranks) != d - 1:
        raise ValueError(
            f"ranks の長さは bond 数 X.ndim-1={d - 1} である必要があります: "
            f"{len(bond_ranks)}"
        )
    cores: list[torch.Tensor] = []

    # remainder: 未分解の残り。shape は (r_left, n_mode, n_{mode+1}, ..., n_d)
    remainder = X
    r_left = 1  # 左端の bond dimension（最初は 1）

    # d 階テンソルなら SVD は d-1 回。各 core は G^(k) ∈ R^{r_{k-1} × n_k × r_k}
    for mode in range(d - 1):
        n_mode = shape[mode]
        # 切断 (r_left·n_mode) | (残り) に対応する 2 次元行列
        mat = remainder.reshape(r_left * n_mode, -1)

        numerical_rank = int(torch.linalg.matrix_rank(mat).item())
        # bond dimension は正の整数として扱うため、数値rankが0の段階
        # （例: 零テンソル）のみ最低rank 1を使う。数値rank >= 1 の通常
        # ケースでは effective_rank == numerical_rank で挙動は変わらない。
        effective_rank = max(1, numerical_rank)
        if bond_ranks is not None:
            r = min(bond_ranks[mode], effective_rank)
        elif max_rank is None:
            r = effective_rank
        else:
            r = min(max_rank, effective_rank)
        U, S, Vh = truncated_svd(mat, r)
        cores.append(U.reshape(r_left, n_mode, r))

        # Σ V^T を次の remainder へ。物理モード n_{mode+1}, ..., n_d を復元
        remainder = (torch.diag(S) @ Vh).reshape(r, *shape[mode + 1 :])
        r_left = r

    # 最後の core。右端 bond は TT 形式どおり 1
    cores.append(remainder.reshape(r_left, shape[-1], 1))
    return cores


def tt_svd_exact(X: torch.Tensor) -> list[torch.Tensor]:
    """任意の d 階テンソルを、数値 rank を保ったまま TT core 列へ分解する。

    打ち切りなし: 各段階の数値rankをそのままbond dimensionとして残す。
    ここでの「rank」は数学的な symbolic rank ではなく、
    ``torch.linalg.matrix_rank`` の default tolerance に基づく数値rank
    であることに注意する。

    Note:
        零テンソル（またはある段階の数値rankが0になる入力）では、
        cut unfolding rank（0）と bond dimension（最低1）は一致しない。
        TTのbond dimensionは正の整数として扱うため、数値rank 0の段階では
        最低rank 1を使う。この場合
        ``cut unfolding rank == bond dimension`` は成立しない。
    """
    return _tt_svd_sweep(X, max_rank=None)


def tt_svd(X: torch.Tensor, max_rank: int) -> list[torch.Tensor]:
    """rank上限 ``max_rank`` 付きの近似TT-SVD。

    各段階で ``min(max_rank, その段階の数値rank)`` をbond dimensionとする。
    ``max_rank`` は 1 以上の整数である必要があり、0 や負数を有効な
    rank へ丸め込むことはしない。
    """
    max_rank = coerce_integer_scalar(max_rank, name="max_rank")
    if max_rank < 1:
        raise ValueError(f"max_rank は 1 以上である必要があります: {max_rank}")

    return _tt_svd_sweep(X, max_rank=max_rank)


def tt_svd_ranks(X: torch.Tensor, ranks: Sequence[int]) -> list[torch.Tensor]:
    """bond ごとの rank 上限付き TT-SVD。

    ``ranks`` の長さは ``X.ndim - 1``。第 ``k`` 段階（0 始まり）の
    bond dimension は ``min(ranks[k], その段階の数値rank)``。
    数値 rank を超える指定は、``tt_svd`` と同じくその段階の数値 rank で頭打ちにする。
    各要素は 1 以上の整数である必要があり、0 や負数を有効な rank へ丸め込まない。
    """
    if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
        raise TypeError(
            f"ranks は bond ごとの整数列である必要があります: {ranks!r}"
        )

    bond_ranks: list[int] = []
    for index, rank in enumerate(ranks):
        rank_int = coerce_integer_scalar(rank, name=f"ranks[{index}]")
        if rank_int < 1:
            raise ValueError(
                f"ranks[{index}] は 1 以上である必要があります: {rank_int}"
            )
        bond_ranks.append(rank_int)

    return _tt_svd_sweep(X, bond_ranks=bond_ranks)


def _scale_safe_vector_norm(values: torch.Tensor) -> torch.Tensor:
    """1階Tensorの2-normを、最大絶対値で正規化してから計算する。"""
    if values.numel() == 0:
        # 空tailでも入力SVDへ接続された厳密な0を返し、autogradを維持する。
        return values.sum()

    scale = values.abs().amax()
    is_zero = scale == 0
    safe_scale = torch.where(is_zero, torch.ones_like(scale), scale)
    normalized_norm = torch.linalg.vector_norm(values / safe_scale)
    return torch.where(is_zero, torch.zeros_like(scale), scale * normalized_norm)


def tt_svd_error_bound(
    X: torch.Tensor,
    ranks: Sequence[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    r"""``tt_svd_ranks`` と同じ逐次SVDに対するFrobenius誤差上限を求める。

    第 ``k`` stepの未分解remainderを行列化した特異値を
    ``sigma_j^(k)``、実際に保持するrankを ``rho_k`` とすると、局所tail norm

    $$
    \delta_k
    =
    \left(\sum_{j > \rho_k} \left(\sigma_j^{(k)}\right)^2\right)^{1/2}
    $$

    と、その二乗和平方根

    $$
    \left(\sum_{k=1}^{d-1}\delta_k^2\right)^{1/2}
    $$

    を返す。``rho_k`` は ``tt_svd_ranks`` と同じく、指定rankと
    ``torch.linalg.matrix_rank`` が返す数値rankの小さい方（零行列では
    最低1）である。このため、数値rankによる追加の切り捨てもtailへ含み、
    後者は実際の ``tt_svd_ranks(X, ranks)`` の絶対Frobenius誤差を上から抑える。

    ``ranks`` の型・長さ・各要素のcontractは ``tt_svd_ranks`` と同じ。
    戻り値は ``(global_bound, local_tail_norms)`` で、いずれも
    ``X.dtype`` / ``X.device`` を維持するスカラーTensorである。
    入力Tensorは変更せず、TT core列やdense近似Tensorは構築しない。
    """
    validate_tensor_shape(X)
    validate_tt_dtype(X, name="X")

    if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
        raise TypeError(
            f"ranks は bond ごとの整数列である必要があります: {ranks!r}"
        )
    if len(ranks) != X.ndim - 1:
        raise ValueError(
            f"ranks の長さは bond 数 X.ndim-1={X.ndim - 1} である必要があります: "
            f"{len(ranks)}"
        )

    bond_ranks: list[int] = []
    for index, rank in enumerate(ranks):
        rank_int = coerce_integer_scalar(rank, name=f"ranks[{index}]")
        if rank_int < 1:
            raise ValueError(
                f"ranks[{index}] は 1 以上である必要があります: {rank_int}"
            )
        bond_ranks.append(rank_int)

    local_tail_norms: list[torch.Tensor] = []
    remainder = X
    r_left = 1
    for mode, rank_cap in enumerate(bond_ranks):
        n_mode = X.shape[mode]
        mat = remainder.reshape(r_left * n_mode, -1)

        # tt_svd_ranksと同じ数値rank contractで、実際の保持rankを決める。
        numerical_rank = int(torch.linalg.matrix_rank(mat).item())
        effective_rank = max(1, numerical_rank)
        rank = min(rank_cap, effective_rank)

        _, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        local_tail_norms.append(_scale_safe_vector_norm(S[rank:]))

        # 保持成分から次stepのremainderを作る。core列自体は構築しない。
        remainder = (S[:rank].unsqueeze(1) * Vh[:rank, :]).reshape(
            rank,
            *X.shape[mode + 1 :],
        )
        r_left = rank

    global_bound = _scale_safe_vector_norm(torch.stack(local_tail_norms))
    return global_bound, local_tail_norms


def tt_reconstruct(cores: list[torch.Tensor]) -> torch.Tensor:
    """TT core 列を左から bond 縮約し、dense テンソルを再構成する。"""
    validate_tt_cores(cores)

    result = cores[0]
    for core in cores[1:]:
        # 左 core の右 bond（最終軸）と右 core の左 bond（先頭軸）を縮約
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.squeeze(0).squeeze(-1)


def tt_num_parameters(cores: list[torch.Tensor]) -> int:
    """TT core列の総要素数 ``P_TT = Σ_k r_{k-1} n_k r_k`` を返す。"""
    validate_tt_cores(cores)
    return sum(core.numel() for core in cores)


def tt_physical_shape(cores: list[torch.Tensor]) -> tuple[int, ...]:
    """TT core列から物理次元 ``(n_1, ..., n_d)`` を返す。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。
        ``cores[k].shape == (r_{k-1}, n_k, r_k)``。

    戻り値:
        各 core の中央軸（物理軸）のサイズを並べた ``tuple[int, ...]``。
    """
    validate_tt_cores(cores)
    return tuple(int(core.shape[1]) for core in cores)


def tt_ranks(cores: list[torch.Tensor]) -> tuple[int, ...]:
    """境界 rank を含む TT-rank 列 ``(1, r_1, ..., r_{d-1}, 1)`` を返す。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。

    戻り値:
        長さ ``d + 1`` の ``tuple[int, ...]``。先頭・末尾は境界 rank
        （常に1）で、中間が各 bond の rank ``r_1, ..., r_{d-1}``。

    Note:
        Notebook 12 の「内部bondだけを返す補助関数」（``[core.shape[2]
        for core in cores[:-1]]`` 相当、長さ ``d - 1``）とは戻り値の
        長さ・境界の扱いが異なる。ここでは境界 rank（常に1）も含めた
        長さ ``d + 1`` の列を返す。
    """
    validate_tt_cores(cores)
    return tuple(int(core.shape[0]) for core in cores) + (int(cores[-1].shape[-1]),)
