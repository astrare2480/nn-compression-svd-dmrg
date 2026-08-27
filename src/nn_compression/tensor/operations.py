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

from .validation import (
    validate_mode_index,
    validate_positive_shape,
    validate_tensor_shape,
)


def unfold(X: torch.Tensor, mode: int) -> torch.Tensor:
    """指定した mode を基準に、テンソルを 2 次元行列へ展開する。

    X: 展開するテンソル
    mode: 展開の基準にする軸番号

    戻り値の shape は ``(X.shape[mode], 残り全 mode の積)``。
    """
    validate_tensor_shape(X)
    validate_mode_index(mode, X.ndim)
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
    shape_tuple = tuple(shape)
    validate_positive_shape(shape_tuple)
    validate_mode_index(mode, len(shape_tuple))

    if unfolded.ndim != 2:
        raise ValueError(
            f"unfolded は2次元行列である必要があります: ndim={unfolded.ndim}"
        )

    expected_rows = shape_tuple[mode]
    if unfolded.shape[0] != expected_rows:
        raise ValueError(
            f"unfolded.shape[0]={unfolded.shape[0]} は "
            f"shape[{mode}]={expected_rows} と一致する必要があります。"
        )

    expected_numel = 1
    for dim in shape_tuple:
        expected_numel *= dim
    if unfolded.numel() != expected_numel:
        raise ValueError(
            f"unfolded の総要素数 {unfolded.numel()} は "
            f"shape={shape_tuple} の期待値 {expected_numel} と一致する必要があります。"
        )

    expected_cols = expected_numel // expected_rows
    if unfolded.shape[1] != expected_cols:
        raise ValueError(
            f"unfolded.shape[1]={unfolded.shape[1]} は "
            f"期待値 {expected_cols} と一致する必要があります。"
            " 転置された unfolded を黙って reshape しません。"
        )

    # shape から mode 番目を一度抜いて、先頭に付け直している式
    moved_shape = (shape_tuple[mode],) + tuple(shape_tuple[:mode]) + tuple(
        shape_tuple[mode + 1 :]
    )

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
    validate_tensor_shape(X)
    validate_mode_index(mode, X.ndim)

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
