"""tensor 演算向けの入力 validation。"""

from __future__ import annotations

import torch


def validate_tensor_ndim_at_least_2(tensor: torch.Tensor, *, name: str = "テンソル") -> None:
    """public tensor API の contract: ndim >= 2。"""
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} は2次元以上である必要があります: ndim={tensor.ndim}"
        )


def validate_mode_index(mode: int, ndim: int, *, name: str = "mode") -> None:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError(
            f"{name} は bool 以外の整数である必要があります: {mode!r}"
        )
    if not 0 <= mode < ndim:
        raise ValueError(
            f"{name}={mode} は 0〜{ndim - 1} の範囲で指定してください。"
        )


def validate_positive_shape(shape: tuple[int, ...]) -> None:
    if not shape:
        raise ValueError("shape は空にできません。")
    for index, dim in enumerate(shape):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                f"shape[{index}] は bool 以外の整数である必要があります: {dim!r}"
            )
        if dim <= 0:
            raise ValueError(
                f"shape[{index}] は正の整数である必要があります: {dim}"
            )
