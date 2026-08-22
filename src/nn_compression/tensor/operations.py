"""分解手法に依存しない mode-n テンソル演算。

参照元:
    notebooks/20_tucker/00_fundamentals/00_tucker_hosvd_basics.ipynb

Tucker / HOSVD だけでなく、mode 単位でテンソルを行列として扱う処理は
すべてここに置く。分解アルゴリズム側（``compression.tucker``）から
呼ばれる下位部品なので、この層は compression へ依存しない。

PyTorch には mode-n unfolding / folding / n-mode product の専用 API が
ないため自前で持つ。``torch.Tensor.unfold`` はスライディングウィンドウを
取り出す別の処理で、ここでいう unfolding とは異なる。
"""

from __future__ import annotations

import torch


def unfold(X: torch.Tensor, mode: int) -> torch.Tensor:
    """指定した mode を基準に、テンソルを 2 次元行列へ展開する。

    X: 展開するテンソル
    mode: 展開の基準にする軸番号

    戻り値の shape は ``(X.shape[mode], 残り全 mode の積)``。
    """
    if not 0 <= mode < X.ndim:
        raise ValueError(
            f"mode={mode} は 0〜{X.ndim - 1} の範囲で指定してください。"
        )
    # movedim(source, destination)はsourceをdestinationnにうつす
    return X.movedim(mode, 0).flatten(start_dim=1)


def fold(
    unfolded: torch.Tensor,
    mode: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """mode-n unfolding された 2 次元行列を、元の軸配置を持つテンソルへ戻す。

    unfolded: mode-n unfoldingされた2次元行列
    mode: unfolding時に基準とした軸番号
    shape: 復元したい元テンソルのshape

    ``unfold`` の逆操作。``shape`` は ``tuple`` でも ``torch.Size`` でもよい。
    """
    if not 0 <= mode < len(shape):
        raise ValueError(
            f"mode={mode} は 0〜{len(shape) - 1} の範囲で指定してください。"
        )
    # shape から mode 番目を一度抜いて、先頭に付け直している式
    moved_shape = (shape[mode],) + tuple(shape[:mode]) + tuple(shape[mode + 1:])

    # 並びだけが違うが全体としてはもとと等しいテンソルに戻す
    X_moved = unfolded.reshape(*moved_shape)

    # X_movedは展開軸が最初にあるので、それを元に戻す
    return X_moved.movedim(0, mode)


def mode_dot(
    X: torch.Tensor,
    matrix: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """テンソルの指定した mode に 2 次元行列を作用させる（n-mode product）。

    X: mode積を行うテンソル
    matrix: 指定したmodeに作用させる2次元行列
    matrix.shape == (新しいmodeのサイズ, X.shape[mode])
    mode: 行列を作用させる軸番号

    ``unfold`` して左から掛け、``fold`` で戻すため、この 3 関数の
    mode 規約は必ず一致させる。
    """
    if not 0 <= mode < len(X.shape):
        raise ValueError(
            f"mode={mode} は 0〜{len(X.shape) - 1} の範囲で指定してください。"
        )

    if not 2 == len(matrix.shape):
        raise ValueError(
            "matrixは2次元行列で指定してください。"
        )

    if matrix.shape[1] != X.shape[mode]:
        raise ValueError(
            f"matrix.shape[1]={matrix.shape[1]} と "
            f"X.shape[{mode}]={X.shape[mode]} は一致する必要があります。"
        )

    # matrixを左から掛ける
    result_unfold = torch.matmul(matrix, unfold(X, mode))

    # 出力shapeを作る
    result_shape = (
        tuple(X.shape[:mode])
        + (matrix.shape[0],)
        + tuple(X.shape[mode + 1:])
    )

    return fold(result_unfold, mode, result_shape)
