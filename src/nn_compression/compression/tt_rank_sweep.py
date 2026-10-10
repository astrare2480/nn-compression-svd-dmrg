"""TT-matrix の weight レベル rank sweep。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/07_tt_rank_bond_dimension_sweep_weight_tradeoff.ipynb

dense weight を TT-matrix へ分解したとき、rank 上限ごとに parameter 数と
再構成誤差がどう変わるかを集める。Notebook 07 の3-core 固定の
``tt_svd_matrix_3core_with_rank_cap`` / ``reconstruct_tt_matrix_3core`` は移さず、
任意の site 数 ``d`` に使える既存 API
（``dense_to_tt_matrix_cores`` / ``tt_matrix_to_dense`` / ``tt_matrix_ranks`` /
``tt_matrix_num_parameters`` / ``relative_frobenius_error``）を組み合わせる。

行列 SVD 用の ``rank_sweep.sweep_layer_ranks`` は ``retained_energy`` を前提とし
model 全体の評価（accuracy・DataLoader）も行うため、ここでは流用しない。
この module は weight 単体の近似誤差と parameter 数の記録までを担当する。
model-level の accuracy 評価、fine-tuning、DataFrame 化、plot、表示は呼び出し側に置く。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..metrics.tensor_approximation import relative_frobenius_error
from .svd import coerce_integer_scalar
from .tt_matrix import (
    dense_to_tt_matrix_cores,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from .tt_matrix_validation import validate_dense_tt_matrix_weight


def _validate_rank_candidates(ranks: Sequence[int]) -> list[int]:
    if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
        raise TypeError(f"ranks は整数の列である必要があります: {ranks!r}")
    if len(ranks) == 0:
        raise ValueError("ranks は空にできません。")

    candidates: list[int] = []
    for index, rank in enumerate(ranks):
        rank_int = coerce_integer_scalar(rank, name=f"ranks[{index}]")
        if rank_int < 1:
            raise ValueError(f"ranks[{index}] は 1 以上である必要があります: {rank_int}")
        candidates.append(rank_int)
    return candidates


def sweep_tt_matrix_weight_ranks(
    W: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
) -> list[dict]:
    """rank 上限の候補ごとに、dense weight の TT-matrix 近似を評価して記録する。

    W:
        dense weight。shape ``(prod(out_modes), prod(in_modes))``、dtype は
        float32 / float64、device は問わない。変更しない。零行列と NaN / inf を含む
        weight は相対誤差が定義できないため ``ValueError``。
    out_modes, in_modes:
        出力側・入力側の mode 列。site 数 ``d`` に制限はない（``d = 1`` も可）。
    ranks:
        rank 上限の候補列（空でない、各要素は bool ではない 1 以上の整数）。
        重複や非昇順も許し、与えた順にそのまま記録する。

    戻り値:
        ``ranks`` と同じ順序の ``list[dict]``。各 record は次のキーを持つ。

        - ``requested_rank``: 渡された rank 上限。
        - ``actual_ranks``: 境界を含む実際の TT-rank ``(r_0, ..., r_d)``（tuple）。
          各 bond は ``min(requested_rank, その段階の数値 rank)`` になるため、
          ``requested_rank`` と一致するとは限らない。
        - ``tt_parameters``: 生成した core の総要素数。
        - ``dense_parameters``: ``W.numel()``。
        - ``compression_ratio``: ``dense_parameters / tt_parameters``。
          1 未満なら TT の方が parameter が多い。
        - ``absolute_error``: ``||W - W_hat||_F``。
        - ``relative_error``: ``||W - W_hat||_F / ||W||_F``。
        - ``is_compressed``: ``tt_parameters < dense_parameters``。
          full rank では TT の parameter 数が dense を上回りうるため、その場合は
          ``False`` が記録される。

        誤差が rank とともに単調に減ることは保証せず、検査もしない
        （multi-core TT-SVD の一般的な性質ではない）。

    Note:
        ``d = 1`` は内部 bond がないため、全 rank 候補で同じ結果になる。
        計算は ``W.detach()`` に対して行い、autograd の計算グラフは作らない。
        各 record の数値は Python の ``int`` / ``float`` / ``bool`` / ``tuple``。

        分解と誤差は ``scale = max |W_ij|`` で割った行列に対して計算する。
        rank と ``relative_error`` はスケール不変なので、正規化後の値を使う。
        ``absolute_error`` は ``scale * ||E_scaled||_F`` を float64 で戻した
        Python float。この積が float64 の最大有限値（約 1e308）を超えるときは
        ``inf`` になる。NaN にはしない。``relative_error`` は正規化空間で
        計算するため、その場合でも有限に留まる。零行列は割る前に拒否する。
    """
    out_modes_t, in_modes_t = validate_dense_tt_matrix_weight(W, out_modes, in_modes)
    candidates = _validate_rank_candidates(ranks)

    weight = W.detach()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("W に NaN / inf が含まれています。")
    scale = weight.abs().amax()
    if not bool(scale > 0):
        raise ValueError("W がゼロ行列なので相対誤差を定義できません。")
    # 最大絶対値で割ると要素は [-1, 1] に収まり、巨大な有限値でも SVD と
    # norm が overflow しない。入力 weight 自体は変更しない。
    weight_scaled = weight / scale

    dense_parameters = int(weight.numel())
    records: list[dict] = []

    with torch.no_grad():
        for requested_rank in candidates:
            cores = dense_to_tt_matrix_cores(
                weight_scaled, out_modes_t, in_modes_t, max_rank=requested_rank
            )
            weight_hat = tt_matrix_to_dense(cores, out_modes_t, in_modes_t)
            scaled_absolute = torch.linalg.vector_norm(weight_scaled - weight_hat)
            # float32 の積だと scale が大きいときに inf になる。Python float へ
            # 戻す前に float64 で掛ける。それでも範囲外なら inf（NaN ではない）。
            absolute_error = float(scale.to(dtype=torch.float64) * scaled_absolute.to(dtype=torch.float64))

            tt_parameters = int(tt_matrix_num_parameters(cores))
            records.append(
                {
                    "requested_rank": requested_rank,
                    "actual_ranks": tt_matrix_ranks(cores),
                    "tt_parameters": tt_parameters,
                    "dense_parameters": dense_parameters,
                    "compression_ratio": dense_parameters / tt_parameters,
                    "absolute_error": absolute_error,
                    "relative_error": float(relative_frobenius_error(weight_scaled, weight_hat)),
                    "is_compressed": tt_parameters < dense_parameters,
                }
            )

    return records
