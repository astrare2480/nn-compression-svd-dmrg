"""データセットを長さで分割するための汎用インデックス処理。

学習と評価で transform が違うときは ``Subset`` に同じ index を渡す。
"""

import torch


def shuffled_index_splits(
    n: int,
    lengths: tuple[int, ...],
    *,
    seed: int,
) -> tuple[list[int], ...]:
    """0..n-1 を再現可能に並べ替え、指定長の連続切片へ分ける。

    ``sum(lengths)`` は ``n`` 以下。余った index は使わない。
    """
    total = sum(lengths)
    if total > n:
        raise ValueError(
            f"分割長の合計 {total} がデータ件数 {n} を超えています。"
        )
    if any(length < 0 for length in lengths):
        raise ValueError("分割長は0以上にしてください。")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    splits: list[list[int]] = []
    start = 0
    for length in lengths:
        splits.append(indices[start : start + length])
        start += length
    return tuple(splits)
