r"""TT-matrix の隣接 2 core を結合・分割・正準化する汎用処理。

物理 DMRG の 3 階 MPS core とは軸も再正規化も違う。正準化は物理軸を
まとめた 3 階 TT に対する ``tt_canonicalize`` を使う。
``torch.nn`` や ``TTLinear`` には依存しない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..linalg.svd import coerce_integer_scalar
from .canonical import tt_canonicality_errors, tt_canonicalize
from .matrix import tt_cores_to_tt_matrix_cores, tt_matrix_cores_to_tt_cores
from .matrix_validation import validate_tt_matrix_cores
from .validation import validate_tt_dtype


def _validate_tt_matrix_core(core: torch.Tensor, *, name: str) -> None:
    """1個の4階 TT-matrix core の階数・正の次元・dtype を確認する。"""
    if not torch.is_tensor(core):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(core)!r}")
    if core.ndim != 4:
        raise ValueError(
            f"{name} は4階 (r_left, m, n, r_right) である必要があります: "
            f"ndim={core.ndim}, shape={tuple(core.shape)}"
        )
    for dim_index, dim in enumerate(core.shape):
        if dim <= 0:
            raise ValueError(
                f"{name}.shape[{dim_index}] は正の整数である必要があります: {dim}"
            )
    validate_tt_dtype(core, name=name)


def _validate_same_dtype_device(named: Sequence[tuple[str, torch.Tensor]]) -> None:
    """列の先頭と dtype / device が揃っていることを確認する。"""
    _, first = named[0]
    for name, tensor in named[1:]:
        if tensor.dtype != first.dtype:
            raise TypeError(
                f"{name}.dtype={tensor.dtype} は {named[0][0]}.dtype={first.dtype} と"
                "一致する必要があります。"
            )
        if tensor.device != first.device:
            raise ValueError(
                f"{name}.device={tensor.device} は {named[0][0]}.device={first.device} と"
                "一致する必要があります。"
            )


def _pair_index(pair_index: int, n_cores: int, *, name: str = "pair_index") -> int:
    """隣接2 core の左端 index を検証する。``0 <= pair_index <= n_cores - 2``。"""
    index = coerce_integer_scalar(pair_index, name=name)
    if n_cores < 2 or not 0 <= index <= n_cores - 2:
        raise ValueError(
            f"{name}={index} は、core 数 {n_cores} の隣接 pair として "
            f"0 <= {name} <= {n_cores - 2} を満たす必要があります。"
        )
    return index


def merge_tt_matrix_cores(
    left_core: torch.Tensor,
    right_core: torch.Tensor,
) -> torch.Tensor:
    r"""隣接する2つの TT-matrix core を、bond ``β`` で縮約して6階テンソルにする。

    left_core:
        ``G_k``。shape ``(r_{k-1}, m_k, n_k, r_k)``。変更しない。
    right_core:
        ``G_{k+1}``。shape ``(r_k, m_{k+1}, n_{k+1}, r_{k+1})``。
        左 bond は ``left_core`` の右 bond と一致する。変更しない。

    戻り値:
        結合テンソル ``T``。shape
        ``(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})``。

        .. math::

            T[α, i_k, j_k, i_{k+1}, j_{k+1}, γ]
            = \sum_β G_k[α, i_k, j_k, β]\, G_{k+1}[β, i_{k+1}, j_{k+1}, γ]

        dtype / device / autograd graph は入力を維持する。``detach`` しない。
        呼び出し側が独立な学習変数にしたいときは、戻り値を明示して
        ``detach().clone()`` する。

    成立条件:
        両 core は4階、dtype は float32 / float64、dtype と device は一致、
        共有 bond の次元が一致する。
    """
    _validate_tt_matrix_core(left_core, name="left_core")
    _validate_tt_matrix_core(right_core, name="right_core")
    _validate_same_dtype_device((("left_core", left_core), ("right_core", right_core)))
    if left_core.shape[3] != right_core.shape[0]:
        raise ValueError(
            f"left_core の right bond ({left_core.shape[3]}) と right_core の "
            f"left bond ({right_core.shape[0]}) が一致しません。"
        )
    # 軸: left (α, i_k, j_k, β), right (β, i_{k+1}, j_{k+1}, γ)
    return torch.tensordot(left_core, right_core, dims=([3], [0]))


def merged_tt_matrix_to_matrix(merged: torch.Tensor) -> torch.Tensor:
    r"""結合テンソル ``T`` を、左3軸と右3軸で行列化する。

    merged:
        shape ``(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})`` の6階 Tensor。
        変更しない。

    戻り値:
        ``M``。shape ``(r_{k-1} m_k n_k, m_{k+1} n_{k+1} r_{k+1})``。

        .. math::

            M[(α, i_k, j_k), (i_{k+1}, j_{k+1}, γ)]
            = T[α, i_k, j_k, i_{k+1}, j_{k+1}, γ]

        ``reshape`` のみなので autograd graph は維持する。
    """
    if not torch.is_tensor(merged):
        raise TypeError(f"merged は torch.Tensor である必要があります: {type(merged)!r}")
    if merged.ndim != 6:
        raise ValueError(
            "merged は6階 (r_left, m_k, n_k, m_next, n_next, r_right) である必要があります: "
            f"ndim={merged.ndim}, shape={tuple(merged.shape)}"
        )
    for dim_index, dim in enumerate(merged.shape):
        if dim <= 0:
            raise ValueError(
                f"merged.shape[{dim_index}] は正の整数である必要があります: {dim}"
            )
    validate_tt_dtype(merged, name="merged")
    r_left, m_k, n_k, m_next, n_next, r_right = merged.shape
    return merged.reshape(r_left * m_k * n_k, m_next * n_next * r_right)


@dataclass(frozen=True)
class TTMatrixPairSplit:
    """結合テンソルの方向付き SVD 分割。再正規化はしない。

    右向き（``direction="right"``、既定）では特異値を右 core に置く。

    left_core:
        ``G_k' = reshape(U_r)``。shape ``(r_{k-1}, m_k, n_k, r_keep)``。
    right_core:
        ``G_{k+1}' = reshape(Σ_r V_r^T)``。
        shape ``(r_keep, m_{k+1}, n_{k+1}, r_{k+1})``。

    左向き（``direction="left"``）では特異値を左 core に置く。

    left_core:
        ``G_k' = reshape(U_r Σ_r)``。
    right_core:
        ``G_{k+1}' = reshape(V_r^T)``。

    どちらの向きでも shape は上と同じである。
    retained_rank:
        保持した特異値の本数 ``r_keep``。
    singular_values:
        切り捨て**前**の還元 SVD が返した特異値全部（降順）。
        数値 rank で小さい成分を落とした結果ではない。
    discarded_squared_norm:
        捨てた特異値の二乗和 ``Σ_{j > r_keep} σ_j^2``。0階 Tensor。
        空の tail は0。入力 dtype / device を維持し、SVD の graph にもつながる。
    """

    left_core: torch.Tensor
    right_core: torch.Tensor
    retained_rank: int
    singular_values: torch.Tensor
    discarded_squared_norm: torch.Tensor


def split_merged_tt_matrix(
    merged: torch.Tensor,
    *,
    max_bond: int | None = None,
    direction: str = "right",
) -> TTMatrixPairSplit:
    r"""結合テンソルを還元 SVD し、特異値を指定した側の core に置いて戻す。

    merged:
        shape ``(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})``。変更しない。
        行列化は、左側 ``(r_{k-1}, m_k, n_k)``、右側
        ``(m_{k+1}, n_{k+1}, r_{k+1})`` である。
    max_bond:
        保持する bond 次元の上限。``None`` なら還元 SVD の全成分
        （``min(左次元, 右次元)`` 個。数値的に小さい特異値も残す）を保持する。
        1以上の整数なら ``min(max_bond, その本数)`` を保持する。
        bool は整数として受け付けない。
    direction:
        ``"right"``（既定）または ``"left"``。

        行列は ``M = U Σ V^T`` とする。

        - ``"right"``: ``G_k' = reshape(U_r)``、
          ``G_{k+1}' = reshape(Σ_r V_r^T)``。
          左 core の列が正規直交になり、直交中心は pair の右 site へ移る。
        - ``"left"``: ``G_k' = reshape(U_r Σ_r)``、
          ``G_{k+1}' = reshape(V_r^T)``。
          右 core の行が正規直交になり、直交中心は pair の左 site へ移る。

        既定は ``"right"`` なので、向きを省略した既存の呼び出しは右向きのままである。

    物理 DMRG の ``split_two_site_tensor`` と違い、特異値をノルムで割らない。
    ``max_bond=None`` のとき、再結合は丸め誤差を除いて ``T`` を再現する。
    ``max_bond`` を付けた切り捨ては、この行列の Frobenius ノルムに関する
    最良低 rank 近似であり、分類損失の最小化ではない。

    入力は変更しない。``detach`` / ``no_grad`` は使わない。
    学習側で SVD を逆伝播に含めない場合は、呼び出し前にテンソルを切る。
    """
    matrix = merged_tt_matrix_to_matrix(merged)
    if max_bond is not None:
        max_bond = coerce_integer_scalar(max_bond, name="max_bond")
        if max_bond < 1:
            raise ValueError(f"max_bond は 1 以上である必要があります: {max_bond}")

    if direction not in {"right", "left"}:
        raise ValueError(
            f'direction は "right" または "left" である必要があります: {direction!r}'
        )

    # full_matrices=False は min(m, n) 本を全部返す。matrix_rank による削減はしない。
    u_factor, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    available = int(singular_values.shape[0])
    retained = available if max_bond is None else min(available, max_bond)

    r_left, m_k, n_k, m_next, n_next, r_right = merged.shape
    kept_u = u_factor[:, :retained]
    kept_s = singular_values[:retained]
    kept_vh = vh[:retained, :]
    if direction == "right":
        left_core = kept_u.reshape(r_left, m_k, n_k, retained)
        right_core = (kept_s.unsqueeze(1) * kept_vh).reshape(retained, m_next, n_next, r_right)
    else:
        left_core = (kept_u * kept_s.unsqueeze(0)).reshape(r_left, m_k, n_k, retained)
        right_core = kept_vh.reshape(retained, m_next, n_next, r_right)
    discarded_squared_norm = singular_values[retained:].square().sum()
    return TTMatrixPairSplit(
        left_core=left_core,
        right_core=right_core,
        retained_rank=retained,
        singular_values=singular_values,
        discarded_squared_norm=discarded_squared_norm,
    )


def canonicalize_tt_matrix_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    center: int,
) -> list[torch.Tensor]:
    r"""TT-matrix core 列を、指定 site を中心とする混合正準形へ変換する。

    各 core を ``(r_{k-1}, m_k n_k, r_k)`` にまとめてから既存の
    ``tt_canonicalize`` を呼び、4階へ戻す。QR による gauge 変換なので、
    dense 重み ``W`` は（丸めを除いて）変わらない。

    center:
        直交中心の site index。``0 <= center < d``。
        2 core の局所更新で pair の左 site を ``center`` にすると、

        - ``center`` より左の core は左直交
          （左展開 ``(r_{k-1} m_k n_k, r_k)`` の列が正規直交）、
        - ``center`` より右の core は右直交
          （右展開 ``(r_{k-1}, m_k n_k r_k)`` の行が正規直交）。

        pair ``(k, k+1)`` に対して ``center=k`` とすると、pair の右 core も
        右直交に含まれる。pair の左 core は直交中心であり、それ自身には
        直交条件を課さない。このとき pair の外側は等長なので、結合テンソル
        ``T`` の Frobenius ノルムは重み ``W`` の Frobenius ノルムと一致する。

    戻り値:
        新しい4階 core 列。入力は変更しない。``tt_canonicalize`` と同様、
        reduced QR が bond 次元を ``min(行, 列)`` まで変えることはあるが、
        数値 rank による切り捨てはしない。
    """
    tt_cores = tt_matrix_cores_to_tt_cores(cores, out_modes, in_modes)
    canonical = tt_canonicalize(tt_cores, center)
    return tt_cores_to_tt_matrix_cores(canonical, out_modes, in_modes)


def tt_matrix_canonicality_errors(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    center: int,
) -> dict[str, list[torch.Tensor]]:
    r"""TT-matrix core 列の左右直交誤差を、既存の3階 TT の定義で返す。

    ``q_k = m_k n_k`` にまとめた core に ``tt_canonicality_errors`` を適用する。
    左直交誤差は ``||M^T M - I||_F``、右直交誤差は ``||N N^T - I||_F``。
    ``center`` 自身は評価しない。

    戻り値:
        ``{"left": [...], "right": [...]}``。``left`` は site ``0 .. center-1``、
        ``right`` は site ``center+1 .. d-1`` の順。各要素は入力と同じ
        dtype / device のスカラー Tensor。
    """
    tt_cores = tt_matrix_cores_to_tt_cores(cores, out_modes, in_modes)
    return tt_canonicality_errors(tt_cores, center)


def replace_tt_matrix_core_pair(
    cores: Sequence[torch.Tensor],
    pair_index: int,
    left_core: torch.Tensor,
    right_core: torch.Tensor,
) -> list[torch.Tensor]:
    """core 列の隣接 pair だけを差し替えた新しい list を返す。

    cores:
        元の TT-matrix core 列。list も各 Tensor も変更しない。
        pair 以外の要素は同じ Tensor を参照する（コピーも ``detach`` もしない）。
    pair_index:
        差し替える左 core の site index。
    left_core, right_core:
        新しい pair。外向きの bond が隣の core と一致する必要がある。

    戻り値:
        長さ ``len(cores)`` の新しい list。構造は ``validate_tt_matrix_cores``
        （mode なし）を満たす。
    """
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise ValueError(f"cores は空でない list または tuple である必要があります: {cores!r}")
    index = _pair_index(pair_index, len(cores))
    _validate_tt_matrix_core(left_core, name="left_core")
    _validate_tt_matrix_core(right_core, name="right_core")
    replaced = list(cores)
    replaced[index] = left_core
    replaced[index + 1] = right_core
    validate_tt_matrix_cores(replaced)
    _validate_same_dtype_device(
        [(f"cores[{i}]", core) for i, core in enumerate(replaced)]
    )
    return replaced

