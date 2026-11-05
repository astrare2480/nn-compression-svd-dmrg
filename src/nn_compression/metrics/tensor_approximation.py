"""テンソル近似そのものの誤差指標。

参照元:
    notebooks/20_tucker/00_fundamentals/00_tucker_hosvd_basics.ipynb
    notebooks/20_tucker/00_fundamentals/01_rank_error_tradeoff.ipynb

``model_comparison`` は圧縮前後のモデル出力を比べるが、こちらは
重みテンソル / 再構成テンソルを直接比べる。分解手法に依存しないので
Tucker 以外の低ランク近似からも使える。
"""

from __future__ import annotations

import math

import torch


def relative_frobenius_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
) -> torch.Tensor:
    """元テンソルと再構成テンソルの相対 Frobenius 誤差を返す。

    ``||X - X_hat||_F / ||X||_F``。

    スケール非依存にするため相対値で返す。分母が 0 だと比が定義できない
    ので、0 除算で inf / nan を返さずエラーにする。
    """
    if X.shape != X_hat.shape:
        raise ValueError(
            f"shapeが一致しません: X={tuple(X.shape)}, "
            f"X_hat={tuple(X_hat.shape)}"
        )

    # 分母になるXの大きさ計算
    denominator = torch.linalg.vector_norm(X)

    if denominator == 0:
        raise ValueError("Xがゼロテンソルなので相対誤差を定義できません。")

    return torch.linalg.vector_norm(X - X_hat) / denominator


def _nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise TypeError(f"{name} は有限な非負の実数である必要があります: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} は有限な非負の実数である必要があります: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} は有限な非負の実数である必要があります: {number}")
    return number


def relative_l2_from_squared_sums(
    difference_squared_sum: float,
    baseline_squared_sum: float,
) -> float:
    """二乗和から、データ全体の relative L2 を返す。

    ``sqrt(sum_i ||difference_i||_2^2) / sqrt(sum_i ||baseline_i||_2^2)``。
    batch ごとの比を平均しない。負値へ丸めず、分母が 0 のときは epsilon に
    置き換えず ``ValueError`` にする。feature map 全体を保持せず、streaming で
    積んだ二乗和を渡すための関数である。
    """
    difference = _nonnegative_finite_float(difference_squared_sum, "difference_squared_sum")
    baseline = _nonnegative_finite_float(baseline_squared_sum, "baseline_squared_sum")
    if baseline == 0.0:
        raise ValueError(
            "baseline の二乗和がゼロなので relative L2 を定義できません。"
            "epsilon には置き換えません。"
        )
    relative = math.sqrt(difference) / math.sqrt(baseline)
    if not math.isfinite(relative):
        raise ValueError(f"relative L2 が有限ではありません: {relative}")
    return relative
