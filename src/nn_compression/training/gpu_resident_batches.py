"""GPU上に置いた学習tensorを、連続sliceとして繰り返し供給する。

``tt_linear_full_batch_sweep`` と ``tt_linear_scoped_update`` のどちらからも
使う。後者が前者をimportするため、このクラスはどちらにも置かない。
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} は int である必要があります: {value!r}")
    if value < 1:
        raise ValueError(f"{name} は 1 以上である必要があります: {value}")
    return value


def _slice_bounds(n_samples: int, batch_size: int) -> list[tuple[int, int]]:
    n_samples = _require_positive_int(n_samples, "n_samples")
    batch_size = _require_positive_int(batch_size, "batch_size")
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        bounds.append((start, end))
        start = end
    cursor = 0
    for bound_start, bound_end in bounds:
        if bound_start != cursor or bound_end <= bound_start:
            raise RuntimeError("batch 境界に欠落または重複があります。")
        cursor = bound_end
    if cursor != n_samples:
        raise RuntimeError("batch 境界が全 sample を覆っていません。")
    return bounds


class GpuResidentTensorBatches:
    """すでに device 上にある入力とラベルを、連続 slice で繰り返し供給する。

    ``batches`` は呼ぶたびに sample 0 から走査する。戻り値の generator を
    このオブジェクトへ保存しない。
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("images と labels は torch.Tensor である必要があります。")
        if images.ndim < 1 or labels.ndim != 1:
            raise ValueError("images は1次元以上、labels は1次元である必要があります。")
        if images.shape[0] != labels.shape[0] or images.shape[0] < 1:
            raise ValueError("images と labels の sample 数が一致し、1件以上必要です。")
        if images.device != labels.device:
            raise ValueError("images と labels は同じ device にある必要があります。")
        if not images.is_contiguous() or not labels.is_contiguous():
            raise ValueError(
                "常駐 tensor は contiguous である必要があります。配置時に一度だけ contiguous にしてください。"
            )
        if images.requires_grad or labels.requires_grad:
            raise ValueError("常駐する入力とラベルは requires_grad=False である必要があります。")
        self.images = images
        self.labels = labels

    @property
    def n_samples(self) -> int:
        return int(self.images.shape[0])

    @property
    def device(self) -> torch.device:
        return self.images.device

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """sample 順を保った新しい走査を返す。最後の batch が小さくても打ち切らない。"""
        for start, end in _slice_bounds(self.n_samples, batch_size):
            yield self.images[start:end], self.labels[start:end]
