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

    return _assemble_pair_split(
        merged,
        u_factor,
        singular_values,
        vh,
        retained,
        direction,
    )


def _assemble_pair_split(
    merged: torch.Tensor,
    u_factor: torch.Tensor,
    singular_values: torch.Tensor,
    vh: torch.Tensor,
    retained: int,
    direction: str,
) -> TTMatrixPairSplit:
    """還元 SVD の先頭 ``retained`` 成分から、方向付きの 2 core を組む。"""
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


@dataclass(frozen=True)
class BondRankDecision:
    """discarded weight による bond rank の選択結果。

    ``available_rank`` は SVD 行列の shape から保持できる本数であり、
    数値的 rank ではない。``threshold_required_rank`` は上限を掛ける前の値。
    ``relative_discarded_weight`` は ``retained_rank`` で実際に捨てた割合。
    比較の許容は ``comparison_atol`` と ``comparison_rtol`` だけを使い、
    既定はどちらも 0 である。
    """

    available_rank: int
    threshold_required_rank: int
    retained_rank: int
    relative_discarded_weight: float
    threshold_satisfied: bool
    cap_limited: bool
    min_rank_limited: bool
    zero_spectrum: bool


def _finite_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold は実数である必要があります: {threshold!r}")
    value = float(threshold)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"threshold は 0 以上の有限値である必要があります: {threshold!r}")
    return value


def _discarded_weight_meets_limit(
    values: torch.Tensor,
    tail_start: int,
    reported_relative: float,
    limit: float,
) -> bool:
    """閾値 0 では非零 tail を underflow した 0 と扱わない。

    正の閾値は、通常 scale の既存結果に合わせ、報告用の float 相対量で比べる。
    """
    if limit == 0.0:
        if tail_start >= int(values.shape[0]):
            return True
        return not bool(torch.any(values[tail_start:] > 0))
    return reported_relative <= limit


def _nonnegative_tolerance(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} は実数である必要があります: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} は 0 以上の有限値である必要があります: {value!r}")
    return number


def select_bond_rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    threshold: float,
    min_bond_rank: int,
    max_bond_rank: int,
    comparison_atol: float = 0.0,
    comparison_rtol: float = 0.0,
) -> BondRankDecision:
    r"""相対 discarded weight が閾値以下になる最小 rank を選ぶ。

    ``discarded_weight(r) = sum_{j>r} σ_j^2 / sum_j σ_j^2``。
    この量は相対 Frobenius 誤差の二乗であり、分類 accuracy の保証ではない。
    二乗和は float64 で計算する。モデル本体の dtype は変えない。

    探索範囲は ``1 .. available_rank``。``available_rank`` は引数の長さであり、
    還元 SVD が shape から返した本数をそのまま渡す。
    全特異値が 0 のときは ``threshold_required_rank = 1``、
    ``retained_rank = min_bond_rank``、切り捨て量 0、閾値達成とする。
    二乗和は最大特異値でスケールしてから計算し、tail は小さい側からの
    逆向き累積和で作る。NaN と Inf はゼロスペクトルにせず ``ValueError``。
    報告する相対量は、この float64 計算のままである。閾値がちょうど 0 の
    rank 判定だけは、正規化二乗が underflow して 0 になっても、tail に非零の
    特異値が残っていれば未達とする。
    """
    if not torch.is_tensor(singular_values):
        raise TypeError(
            f"singular_values は torch.Tensor である必要があります: {type(singular_values)!r}"
        )
    if singular_values.ndim != 1 or singular_values.shape[0] < 1:
        raise ValueError(
            "singular_values は長さ 1 以上の 1 階 Tensor である必要があります: "
            f"shape={tuple(singular_values.shape)}"
        )
    validate_tt_dtype(singular_values, name="singular_values")
    if not torch.isfinite(singular_values).all():
        raise ValueError(
            "特異値に NaN または Inf があります。ゼロスペクトルとは区別し、rank は選びません。"
        )
    if torch.any(singular_values < 0):
        raise ValueError("特異値は 0 以上である必要があります。")
    threshold = _finite_threshold(threshold)
    comparison_atol = _nonnegative_tolerance(comparison_atol, "comparison_atol")
    comparison_rtol = _nonnegative_tolerance(comparison_rtol, "comparison_rtol")
    min_bond_rank = coerce_integer_scalar(min_bond_rank, name="min_bond_rank")
    max_bond_rank = coerce_integer_scalar(max_bond_rank, name="max_bond_rank")
    if min_bond_rank < 1 or max_bond_rank < 1 or min_bond_rank > max_bond_rank:
        raise ValueError(
            f"1 <= min_bond_rank <= max_bond_rank である必要があります: "
            f"min={min_bond_rank}, max={max_bond_rank}"
        )
    available = int(singular_values.shape[0])
    if min_bond_rank > available:
        raise ValueError(
            f"min_bond_rank={min_bond_rank} は available_rank={available} を超えています。"
        )

    if torch.all(singular_values == 0):
        return BondRankDecision(
            available_rank=available,
            threshold_required_rank=1,
            retained_rank=min_bond_rank,
            relative_discarded_weight=0.0,
            threshold_satisfied=True,
            cap_limited=False,
            min_rank_limited=min_bond_rank > 1,
            zero_spectrum=True,
        )

    values = singular_values.detach().to(dtype=torch.float64)
    scale = values.max()
    if not torch.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("特異値のスケールが有限な正の値ではありません。")
    squares = (values / scale).square()
    if not torch.isfinite(squares).all():
        raise ValueError("スケール後の特異値二乗が有限ではありません。")
    total = squares.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise ValueError("特異値二乗和が有限な正の値ではありません。")
    inclusive_from_here = torch.cumsum(squares.flip(0), dim=0).flip(0)
    tail = torch.zeros_like(squares)
    if available > 1:
        tail[:-1] = inclusive_from_here[1:]
    if not torch.isfinite(tail).all() or float(tail.min()) < 0.0:
        raise ValueError("discarded weight の tail sum が有限な非負値ではありません。")
    relative = tail / total
    if not torch.isfinite(relative).all():
        raise ValueError("相対 discarded weight が有限ではありません。")
    limit = threshold + comparison_atol + comparison_rtol * abs(threshold)
    required = available
    for index in range(available):
        if _discarded_weight_meets_limit(values, index + 1, float(relative[index]), limit):
            required = index + 1
            break
    cap_limited = required > max_bond_rank
    retained = min(required, max_bond_rank, available)
    min_rank_limited = retained < min_bond_rank
    if min_rank_limited:
        retained = min_bond_rank
    relative_discarded_weight = float(relative[retained - 1])
    return BondRankDecision(
        available_rank=available,
        threshold_required_rank=required,
        retained_rank=retained,
        relative_discarded_weight=relative_discarded_weight,
        threshold_satisfied=_discarded_weight_meets_limit(
            values, retained, relative_discarded_weight, limit
        ),
        cap_limited=cap_limited,
        min_rank_limited=min_rank_limited,
        zero_spectrum=False,
    )


def split_merged_tt_matrix_by_discarded_weight(
    merged: torch.Tensor,
    *,
    threshold: float,
    min_bond_rank: int,
    max_bond_rank: int,
    direction: str = "right",
    comparison_atol: float = 0.0,
    comparison_rtol: float = 0.0,
) -> tuple[TTMatrixPairSplit, BondRankDecision]:
    """結合テンソルを 1 回だけ SVD し、discarded weight で rank を選んで分割する。

    core の dtype / device は ``split_merged_tt_matrix`` と同じく入力を維持する。
    rank の比較計算だけ float64 を使う。固定 rank の ``max_bond`` 分割は
    ``split_merged_tt_matrix`` のままである。
    """
    matrix = merged_tt_matrix_to_matrix(merged)
    if direction not in {"right", "left"}:
        raise ValueError(
            f'direction は "right" または "left" である必要があります: {direction!r}'
        )
    u_factor, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    decision = select_bond_rank_from_singular_values(
        singular_values,
        threshold=threshold,
        min_bond_rank=min_bond_rank,
        max_bond_rank=max_bond_rank,
        comparison_atol=comparison_atol,
        comparison_rtol=comparison_rtol,
    )
    split = _assemble_pair_split(
        merged,
        u_factor,
        singular_values,
        vh,
        decision.retained_rank,
        direction,
    )
    return split, decision


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


def truncate_tt_matrix_bond(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    pair_index: int,
    *,
    max_bond: int,
    direction: str = "right",
) -> tuple[list[torch.Tensor], TTMatrixPairSplit]:
    """1 本の TT-matrix bond を、rank 上限付き SVD で切り詰める。

    ``pair_index`` は切り詰める bond の左 site である。bond 番号を
    ``pair_index + 1`` と数える実験でも、この関数へ渡すのは左 site の番号。

    手順は、pair index の検証、その pair の左 site を中心にした正準化、
    隣接 2 core の結合、``max_bond`` による SVD 分割、core 列への書き戻し。
    入力の core 列も各 Tensor も変更しない。dtype、device、autograd graph は
    再利用する正準化・結合・分割が維持する。

    戻り値は、更新済み core 列と、その分割結果 ``TTMatrixPairSplit``。
    還元 SVD の本数が ``max_bond`` 未満のときは、その本数を保持する。
    """
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise ValueError(f"cores は空でない list または tuple である必要があります: {cores!r}")
    index = _pair_index(pair_index, len(cores))
    canonical = canonicalize_tt_matrix_cores(cores, out_modes, in_modes, index)
    merged = merge_tt_matrix_cores(canonical[index], canonical[index + 1])
    split = split_merged_tt_matrix(merged, max_bond=max_bond, direction=direction)
    replaced = replace_tt_matrix_core_pair(
        canonical,
        index,
        split.left_core,
        split.right_core,
    )
    return replaced, split


def truncate_tt_matrix_bonds(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    bond_targets: Sequence[tuple[int, int]],
    *,
    direction: str = "right",
) -> tuple[list[torch.Tensor], tuple[TTMatrixPairSplit, ...]]:
    """複数の TT-matrix bond を、与えた順に 1 本ずつ切り詰める。

    ``bond_targets`` は ``(pair_index, max_bond)`` の列である。
    ``pair_index`` は ``truncate_tt_matrix_bond`` と同じく、切り詰める bond の
    左 site を指す。各段は既存の ``truncate_tt_matrix_bond`` を、直前の結果に
    対して呼ぶ。

    切り詰める順序によって結果は変わり得る。後の bond は、前の切り詰めで
    変わった core を正準化し直してから SVD するため、同じ target でも
    逆順とは一致しないことがある。

    入力の core 列も各 Tensor も変更しない。dtype、device、autograd graph は
    各段の切り詰めが維持する。戻り値は更新済み core 列と、bond 順の
    ``TTMatrixPairSplit`` である。
    """
    if isinstance(bond_targets, (str, bytes)) or not isinstance(bond_targets, Sequence):
        raise TypeError("bond_targets は (pair_index, max_bond) の列である必要があります。")
    updated: Sequence[torch.Tensor] = cores
    splits: list[TTMatrixPairSplit] = []
    for position, target in enumerate(bond_targets):
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError(
                f"bond_targets[{position}] は (pair_index, max_bond) である必要があります: {target!r}"
            )
        pair_index, max_bond = target
        updated, split = truncate_tt_matrix_bond(
            updated,
            out_modes,
            in_modes,
            pair_index,
            max_bond=max_bond,
            direction=direction,
        )
        splits.append(split)
    return list(updated), tuple(splits)

