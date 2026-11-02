"""TTLinear の入力と bias の検証。

TT-matrix core 自体の検証は ``nn_compression.tt_mps.matrix_validation``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def validate_tt_linear_input(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    in_modes: Sequence[int],
    *,
    name: str = "x",
) -> None:
    """``tt_linear_forward`` の入力 ``x`` が満たすべき contract を検証する。

    - ``x`` が ``torch.Tensor``
    - 2階（``(batch, in_features)``）のみ正式対応
    - 最終次元が ``product(in_modes)`` と一致する
    - dtype / device が ``cores[0]`` と一致する

    ``cores`` 自体の構造検証（``validate_tt_matrix_cores``）は呼び出し側の
    責務とし、ここでは行わない（``cores[0]`` の dtype/device 参照のみ使う）。
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(
            f"{name} は2階（(batch, in_features)）である必要があります"
            f"（今回は2階入力のみ正式対応）: ndim={x.ndim}, shape={tuple(x.shape)}"
        )

    expected_n = math.prod(int(n) for n in in_modes)
    if x.shape[-1] != expected_n:
        raise ValueError(
            f"{name}.shape[-1]={x.shape[-1]} は product(in_modes)={expected_n} "
            "と一致する必要があります。"
        )

    core0 = cores[0]
    if x.dtype != core0.dtype:
        raise TypeError(
            f"{name}.dtype={x.dtype} は cores[0].dtype={core0.dtype} と"
            "一致する必要があります。"
        )
    if x.device != core0.device:
        raise ValueError(
            f"{name}.device={x.device} は cores[0].device={core0.device} と"
            "一致する必要があります。"
        )


def validate_tt_linear_bias(
    bias: torch.Tensor | None,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    *,
    name: str = "bias",
) -> None:
    """``tt_linear_forward`` の ``bias`` が満たすべき contract を検証する。

    - ``bias`` が ``None`` の場合は常に許可する
      （Notebook 14 の実装は ``bias is None`` でも ``bias.shape`` を参照して
      ``AttributeError`` になる不具合があったため、ここでは最初に ``None``
      チェックを行う）。
    - ``None`` でない場合、``torch.Tensor`` かつ shape が
      ``(product(out_modes),)``
    - dtype / device が ``cores[0]`` と一致する
    """
    if bias is None:
        return

    if not torch.is_tensor(bias):
        raise TypeError(f"{name} は torch.Tensor または None である必要があります: {type(bias)!r}")

    expected_m = math.prod(int(m) for m in out_modes)
    if tuple(bias.shape) != (expected_m,):
        raise ValueError(
            f"{name}.shape={tuple(bias.shape)} は (product(out_modes),)="
            f"({expected_m},) と一致する必要があります。"
        )

    core0 = cores[0]
    if bias.dtype != core0.dtype:
        raise TypeError(
            f"{name}.dtype={bias.dtype} は cores[0].dtype={core0.dtype} と"
            "一致する必要があります。"
        )
    if bias.device != core0.device:
        raise ValueError(
            f"{name}.device={bias.device} は cores[0].device={core0.device} と"
            "一致する必要があります。"
        )

