"""テンソル近似そのものの誤差指標。

参照元:
    notebooks/20_tucker/00_fundamentals/00_tucker_hosvd_basics.ipynb
    notebooks/20_tucker/00_fundamentals/01_rank_error_tradeoff.ipynb

``model_comparison`` は圧縮前後のモデル出力を比べるが、こちらは
重みテンソル / 再構成テンソルを直接比べる。分解手法に依存しないので
Tucker 以外の低ランク近似からも使える。
"""

from __future__ import annotations

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
