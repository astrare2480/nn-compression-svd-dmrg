r"""TT-matrix（TTで表現した行列）の dense 復元・tensorization・TT-Linear forward。

参照元:
    notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb
    notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb

Notebook 14/15 では、``nn.Linear`` の重み ``W in R^{m x n}`` を

    out_modes = (m_1, ..., m_d),  m = prod(m_k)
    in_modes  = (n_1, ..., n_d),  n = prod(n_k)

で分解した TT-matrix core 列

    G^{(k)} in R^{r_{k-1} x m_k x n_k x r_k},  r_0 = r_d = 1

として表現し、``Y = X W^T + b`` を dense weight を作らずに計算する
``tt_linear_forward`` を検証した。ここでは、その中の再利用可能な処理
（tensorization・dense復元・TT-SVDへの橋渡し・dense化しないforward）だけを
production API として独立させる。

Notebook 側の ``assert`` ベースの validation、``print`` によるshape表示、
``clone_cores``、固定shape専用の2-core SVDコード、添字を選んだ要素対応の
表示、bond indexを明示した教材用for-loopは、ここへは移さない
（Notebookに残す）。

``torch.nn.Module`` としての ``TTLinear`` クラス化は今回のスコープ外
（後続フェーズで設計する）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..linalg.svd import coerce_integer_scalar
from .core import tt_svd, tt_svd_exact
from .matrix_validation import (
    validate_dense_tt_matrix_weight,
    validate_tt_matrix_cores,
    validate_tt_matrix_modes,
)
from .validation import validate_tt_cores, validate_tt_dtype

def tt_matrix_ranks(cores: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """TT-matrix core 列から、境界 rank を含む TT-rank 列を返す。

    cores:
        ``validate_tt_matrix_cores`` の contract を満たす TT-matrix core 列
        （各 core の shape は ``(r_{k-1}, m_k, n_k, r_k)``）。

    戻り値:
        長さ ``d + 1`` の ``tuple[int, ...]``: ``(r_0, r_1, ..., r_d)``。
        先頭・末尾は境界 rank（常に1）。

        戻り値の型・形式は既存の ``tt_ranks``（通常の3階TT向け）に合わせる。
    """
    validate_tt_matrix_cores(cores)
    return tuple(int(core.shape[0]) for core in cores) + (int(cores[-1].shape[-1]),)

def tt_matrix_num_parameters(cores: Sequence[torch.Tensor]) -> int:
    """TT-matrix core 列の総要素数 ``sum_k r_{k-1} m_k n_k r_k`` を返す。

    cores:
        ``validate_tt_matrix_cores`` の contract を満たす TT-matrix core 列。
    """
    validate_tt_matrix_cores(cores)
    return sum(core.numel() for core in cores)

def dense_to_tt_matrix_tensor(
    W: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    r"""dense weight ``W`` を、TT-SVD 入力用の combined tensor へ並べ替える。

    次の変換だけを行う（TT-SVD 自体はここでは行わない）。

        W: (prod(m_k), prod(n_k))
        -> grouped:     (m_1, ..., m_d, n_1, ..., n_d)
        -> interleaved: (m_1, n_1, ..., m_d, n_d)
        -> combined:    (m_1 n_1, ..., m_d n_d)

    W:
        dense weight。shape ``(product(out_modes), product(in_modes))``。
        dtype は ``torch.float32`` / ``torch.float64``。device は制限しない
        （CPU固定にしない）。変更しない（``reshape`` / ``permute`` のみ使う
        ので、連続でない場合でも元の ``W`` 自体を書き換えることはない）。
    out_modes:
        出力側 physical modes ``(m_1, ..., m_d)``。
    in_modes:
        入力側 physical modes ``(n_1, ..., n_d)``。

    戻り値:
        shape ``(q_1, ..., q_d)``（``q_k = m_k * n_k``）の combined tensor。
        ``W`` の dtype / device / autograd（``requires_grad`` と勾配グラフ）
        を維持する。

    Note:
        ``W`` が ``out_modes`` / ``in_modes`` が要求する並びで contiguous
        でない場合、``reshape`` は内部でコピーを作ることがあるが、それは
        新しい Tensor であり、入力 ``W`` 自体を変更するものではない。
    """
    out_modes_t, in_modes_t = validate_dense_tt_matrix_weight(W, out_modes, in_modes)
    d = len(out_modes_t)

    grouped = W.reshape(*out_modes_t, *in_modes_t)

    # (m_1, ..., m_d, n_1, ..., n_d) -> (m_1, n_1, ..., m_d, n_d)
    perm: list[int] = []
    for k in range(d):
        perm.append(k)
        perm.append(d + k)
    interleaved = grouped.permute(*perm)

    q_modes = tuple(m_k * n_k for m_k, n_k in zip(out_modes_t, in_modes_t))
    return interleaved.reshape(*q_modes)

def tt_cores_to_tt_matrix_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> list[torch.Tensor]:
    r"""通常の3階TT core列を、4階のTT-matrix core列へ reshape する。

    各 core の物理軸 ``q_k = m_k * n_k`` を ``(m_k, n_k)`` にばらす。

        core_k: (r_{k-1}, q_k, r_k) -> (r_{k-1}, m_k, n_k, r_k)

    cores:
        ``validate_tt_cores`` の contract を満たす通常の TT core 列
        （``tt_svd_exact`` / ``tt_svd`` の戻り値）。変更しない。
    out_modes:
        出力側 physical modes ``(m_1, ..., m_d)``。
    in_modes:
        入力側 physical modes ``(n_1, ..., n_d)``。``len(out_modes) ==
        len(in_modes) == len(cores)`` かつ、各 ``k`` で
        ``cores[k].shape[1] == out_modes[k] * in_modes[k]`` である必要が
        ある。

    戻り値:
        4階 TT-matrix core 列（新しい ``list``）。``reshape`` のみ使うため、
        元の ``cores`` の各 Tensor（および autograd graph）は変更しない。

    Note:
        TT-SVD 自体はここでは行わない（reshape のみ）。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    if len(out_modes_t) != len(cores):
        raise ValueError(
            f"len(out_modes)={len(out_modes_t)} は len(cores)={len(cores)} と"
            "一致する必要があります。"
        )

    tt_matrix_cores: list[torch.Tensor] = []
    for index, (core, m_k, n_k) in enumerate(zip(cores, out_modes_t, in_modes_t)):
        r_left, q_k, r_right = core.shape
        if q_k != m_k * n_k:
            raise ValueError(
                f"cores[{index}].shape[1]={q_k} は "
                f"out_modes[{index}]*in_modes[{index}]={m_k * n_k} と"
                "一致する必要があります。"
            )
        tt_matrix_cores.append(core.reshape(r_left, m_k, n_k, r_right))

    return tt_matrix_cores

def tt_matrix_cores_to_tt_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> list[torch.Tensor]:
    r"""4階 TT-matrix core 列を、通常の3階 TT core 列へ reshape する。

    ``tt_cores_to_tt_matrix_cores`` の逆変換。各 core の出力・入力 mode
    ``(m_k, n_k)`` を、1本の物理軸 ``q_k = m_k * n_k`` にまとめる。

        core_k: (r_{k-1}, m_k, n_k, r_k) -> (r_{k-1}, q_k, r_k)

    軸順は C-order の reshape なので、``q_k`` の中では ``n_k`` が最も速く
    変わる。これは ``tt_cores_to_tt_matrix_cores`` が ``(r, q, r)`` を
    ``(r, m, n, r)`` に戻すときの順序と同じであり、
    ``dense_to_tt_matrix_tensor`` の ``a_k = i_k * n_k + j_k`` とも一致する。

    cores:
        ``validate_tt_matrix_cores`` の contract を満たす TT-matrix core 列。
        変更しない。
    out_modes, in_modes:
        各 core の ``shape[1]`` / ``shape[2]`` と一致する mode 列。

    戻り値:
        3階 TT core 列（新しい ``list``）。``reshape`` のみなので、元 Tensor
        の値・dtype・device・autograd graph は維持する。

    成立条件:
        左展開 ``(r_{k-1} * q_k, r_k)`` は、4階 core を
        ``(r_{k-1} * m_k * n_k, r_k)`` に展開したものと同一である。したがって
        3階 TT 用の ``tt_canonicalize`` / ``tt_canonicality_errors`` を、
        この変換の前後にはさんで TT-matrix に適用できる。
    """
    validate_tt_matrix_cores(cores, out_modes, in_modes)
    out_modes_t = tuple(int(mode) for mode in out_modes)
    in_modes_t = tuple(int(mode) for mode in in_modes)

    tt_cores: list[torch.Tensor] = []
    for core, m_k, n_k in zip(cores, out_modes_t, in_modes_t):
        r_left, _, _, r_right = core.shape
        tt_cores.append(core.reshape(r_left, m_k * n_k, r_right))
    return tt_cores

def _validate_tt_svd_max_rank(max_rank: int | None) -> None:
    """``tt_svd`` と同じ ``max_rank`` contract を適用する。

    ``None`` は打ち切りなし（exact）として許可する。それ以外は
    ``tt_svd`` と同じく、bool・非整数を ``TypeError``、0 以下を
    ``ValueError`` で拒否する。``tt_svd`` 本体の contract は変更しない。
    """
    if max_rank is None:
        return
    max_rank_int = coerce_integer_scalar(max_rank, name="max_rank")
    if max_rank_int < 1:
        raise ValueError(f"max_rank は 1 以上である必要があります: {max_rank_int}")

def dense_to_tt_matrix_cores(
    W: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> list[torch.Tensor]:
    r"""dense weight から TT-matrix core 列までを一括で作る wrapper。

    ``d >= 2`` では次の3段を結合する。

        dense_to_tt_matrix_tensor
        -> tt_svd_exact または tt_svd
        -> tt_cores_to_tt_matrix_cores

    ``d == 1``（1-site）では combined tensor が1階になり、``tt_svd_exact`` /
    ``tt_svd`` の「2階以上」という contract の対象外になる。この場合は
    TT-SVD を呼ばず、``W.reshape(1, m_1, n_1, 1)`` を唯一の core として返す。
    内部 bond がないため、``max_rank`` が 1 以上ならその値は結果を変えない。

    W:
        dense weight。``dense_to_tt_matrix_tensor`` と同じ contract。
        変更しない。dtype / device / autograd graph は戻り値の core に維持する。
    out_modes, in_modes:
        出力側・入力側の physical modes。
    max_rank:
        ``None``（既定）なら、``d >= 2`` では打ち切りなしの ``tt_svd_exact``
        を使う。``int`` なら、その値を上限とする ``tt_svd`` を使う。

        ``d == 1`` でも ``None`` と 1 以上の整数は受け付ける。``True``、
        ``0``、負数、非整数は ``tt_svd`` と同じ例外で拒否する
        （この関数側で先に検証し、``tt_svd`` 自体は呼ばない）。

        既存 ``tt_svd`` / ``tt_svd_exact`` の contract を拡張・緩和することは
        しない。

    戻り値:
        4階 TT-matrix core 列。
    """
    out_modes_t, in_modes_t = validate_dense_tt_matrix_weight(W, out_modes, in_modes)

    if len(out_modes_t) == 1:
        _validate_tt_svd_max_rank(max_rank)
        return [W.reshape(1, out_modes_t[0], in_modes_t[0], 1)]

    combined = dense_to_tt_matrix_tensor(W, out_modes_t, in_modes_t)

    if max_rank is None:
        cores_3d = tt_svd_exact(combined)
    else:
        cores_3d = tt_svd(combined, max_rank)

    return tt_cores_to_tt_matrix_cores(cores_3d, out_modes, in_modes)

def tt_matrix_to_dense(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    r"""TT-matrix core 列を bond 縮約し、dense weight ``W`` を復元する（参照検証・比較用API）。

    cores:
        ``validate_tt_matrix_cores`` の contract を満たす TT-matrix core 列。
        変更しない。
    out_modes:
        出力側 physical modes ``(m_1, ..., m_d)``。``m = product(out_modes)``。
    in_modes:
        入力側 physical modes ``(n_1, ..., n_d)``。``n = product(in_modes)``。

    戻り値:
        dense weight ``W``。shape ``(m, n)``。``cores`` の dtype / device /
        autograd を維持する。

    計算量:
        左から右へ ``torch.tensordot`` で bond を縮約していく（``tt_reconstruct``
        と同じ構成）。dense化は参照検証・比較専用であり、``tt_linear_forward``
        からは呼ばない。

    Note:
        入力 ``cores`` を clone しない。``tensordot`` / ``permute`` /
        ``reshape`` / ``squeeze`` はいずれも非破壊的な演算であり、入力
        Tensor 自体を書き換えない。
    """
    validate_tt_matrix_cores(cores, out_modes, in_modes)
    out_modes_t = tuple(int(m) for m in out_modes)
    in_modes_t = tuple(int(n) for n in in_modes)
    d = len(cores)

    # 左 -> 右へ bond 縮約。boundary rank（左端のr_0=1）をまず落とす。
    current = cores[0].squeeze(0)  # (m_1, n_1, r_1)
    for core in cores[1:]:
        current = torch.tensordot(current, core, dims=([-1], [0]))
    # 最終的な right boundary（r_d=1）を落とす。
    current = current.squeeze(-1)
    # current の axis は (m_1, n_1, m_2, n_2, ..., m_d, n_d)（interleaved）。

    # interleaved -> grouped: (m_1, ..., m_d, n_1, ..., n_d)
    perm = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
    grouped = current.permute(*perm)

    m = math.prod(out_modes_t)
    n = math.prod(in_modes_t)
    return grouped.reshape(m, n)
