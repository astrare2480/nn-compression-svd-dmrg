"""学習可能な TT-Linear 層（``nn.Module``）と、dense ``nn.Linear`` からの変換。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb
    notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb
    notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb

Notebook 00 の教材用 ``TTLinear`` は、core を ``requires_grad=False`` にし、
bias を buffer にしていたため、fine-tuning には使えなかった。ここでは
core と bias を通常の ``nn.Parameter`` として保持し、optimizer で更新できる
production 向けの層にする。

forward は既存の ``tt_linear_forward`` を呼ぶだけで、dense weight を復元しない。
変換と層置換は、既存の ``dense_to_tt_matrix_cores`` と
``utils.get_named_module`` / ``utils.set_named_module`` を再利用する
（Notebook 内の ``get_module_by_path`` / ``set_module_by_path`` は移さない）。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
from torch import nn

from ..utils.modules import get_named_module, set_named_module
from .tt_matrix import dense_to_tt_matrix_cores, tt_linear_forward, tt_matrix_ranks
from .tt_matrix_validation import (
    validate_tt_linear_bias,
    validate_tt_matrix_cores,
    validate_tt_matrix_modes,
)


class TTLinear(nn.Module):
    """TT-matrix core 列で重みを保持し、``Y = X W^T + b`` を計算する線形層。

    cores:
        TT-matrix core 列（list / tuple）。各 core の shape は
        ``(r_{k-1}, m_k, n_k, r_k)``、``r_0 = r_d = 1``。
        contract は ``tt_matrix_validation.validate_tt_matrix_cores`` と同じ
        （4階、bond 一致、dtype は float32 / float64、core 間で dtype / device 一致）。
    out_modes:
        出力側 mode ``(m_1, ..., m_d)``。``out_features = prod(out_modes)``。
    in_modes:
        入力側 mode ``(n_1, ..., n_d)``。``in_features = prod(in_modes)``。
    bias:
        ``None`` または shape ``(out_features,)`` の Tensor。dtype / device は
        core と一致している必要がある。

    Parameter 構成:
        - ``self.cores``: ``nn.ParameterList``。要素 ``cores.{k}`` は
          ``(r_{k-1}, m_k, n_k, r_k)`` の leaf ``nn.Parameter``。
        - ``self.bias``: leaf ``nn.Parameter``、または ``None``。

        いずれも渡された Tensor を ``detach().clone()`` したコピーで、
        このモジュールが所有する。元の Tensor は変更しない。既定で
        ``requires_grad=True``。凍結したい場合は、呼び出し側が通常の PyTorch の方法
        （``param.requires_grad_(False)``）で設定する。

    入力:
        ``forward`` は ``(batch, in_features)`` の2階 Tensor のみ受け付ける
        （``tt_linear_forward`` の contract）。dtype / device は core と一致が必要。

    Note:
        ``.to(device)`` / ``.to(dtype)`` は通常の ``nn.Module`` として core と bias を
        移す。ただし ``torch.float32`` / ``torch.float64`` 以外へ変換した場合、
        forward は dtype 検証で ``TypeError`` になる。
    """

    def __init__(
        self,
        cores: Sequence[torch.Tensor],
        out_modes: Sequence[int],
        in_modes: Sequence[int],
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
        validate_tt_matrix_cores(cores, out_modes_t, in_modes_t)
        validate_tt_linear_bias(bias, cores, out_modes_t)

        self.out_modes: tuple[int, ...] = out_modes_t
        self.in_modes: tuple[int, ...] = in_modes_t

        self.cores = nn.ParameterList(
            [
                nn.Parameter(
                    core.detach().clone(memory_format=torch.contiguous_format)
                )
                for core in cores
            ]
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_parameter(
                "bias",
                nn.Parameter(bias.detach().clone(memory_format=torch.contiguous_format)),
            )

    @property
    def in_features(self) -> int:
        return math.prod(self.in_modes)

    @property
    def out_features(self) -> int:
        return math.prod(self.out_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tt_linear_forward(
            x,
            list(self.cores),
            self.out_modes,
            self.in_modes,
            self.bias,
        )

    def extra_repr(self) -> str:
        ranks = tt_matrix_ranks(list(self.cores))
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_modes={self.in_modes}, out_modes={self.out_modes}, "
            f"ranks={ranks}, bias={self.bias is not None}"
        )


def build_tt_linear(
    linear: nn.Linear,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> TTLinear:
    """dense ``nn.Linear`` から ``TTLinear`` を作る。

    linear:
        変換元。変更しない。dtype は float32 / float64 のみ。
    out_modes, in_modes:
        ``prod(out_modes) == linear.out_features``、
        ``prod(in_modes) == linear.in_features`` を満たす mode 列。
    max_rank:
        ``None``（既定）なら打ち切りなし（``tt_svd_exact``）。
        正の整数なら、その値を bond rank の上限とする ``tt_svd``。
        ``d = 1``（1-site）は内部 bond がないため、``max_rank`` に依らず 1 core。
        不正値は ``dense_to_tt_matrix_cores`` と同じ contract で拒否する。

    戻り値:
        ``TTLinear``。

        - dtype / device は ``linear.weight`` を維持する。
        - bias の有無を維持する。
        - Parameter は ``linear`` から独立（storage を共有しない）。
        - ``requires_grad`` は ``linear.weight`` / ``linear.bias`` の状態を引き継ぐ
          （凍結した層から学習可能な層を作らない）。
        - ``training`` フラグは ``linear.training`` に合わせる。
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"linear は nn.Linear である必要があります: {type(linear).__name__}"
        )

    cores = dense_to_tt_matrix_cores(
        linear.weight.detach(), out_modes, in_modes, max_rank=max_rank
    )
    layer = TTLinear(cores, out_modes, in_modes, bias=linear.bias)

    for core in layer.cores:
        core.requires_grad_(linear.weight.requires_grad)
    if layer.bias is not None:
        layer.bias.requires_grad_(linear.bias.requires_grad)

    layer.train(linear.training)
    return layer


def factorize_named_tt_linear(
    model: nn.Module,
    layer_name: str,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> nn.Module:
    """指定した名前の ``nn.Linear`` だけを ``TTLinear`` へ置換したコピーを返す。

    ``factorize_named_linear`` と同じ方針: ``model`` は変更せず ``deepcopy`` した
    コピー側だけを置換する。そのため、コピーを fine-tune しても元の model の
    Parameter は動かない。

    layer_name:
        ``"fc1"`` や ``"encoder.0"`` のようなドット区切りの名前。空文字は不可。
        存在しない名前は ``get_submodule`` と同じ ``AttributeError``。
    out_modes, in_modes, max_rank:
        ``build_tt_linear`` と同じ。

    例外:
        ``layer_name`` が ``str`` でなければ ``TypeError``、空なら ``ValueError``。
        指す module が ``nn.Linear`` でなければ ``TypeError``。

    Note:
        model 全体の dtype / device / 各 module の ``training`` 状態は ``deepcopy`` が
        そのまま保持する。置換される層の ``training`` は元の ``nn.Linear`` に揃える。
    """
    if not isinstance(layer_name, str):
        raise TypeError(
            f"layer_name は str である必要があります: {type(layer_name).__name__}"
        )
    if layer_name == "":
        raise ValueError("layer_name は空文字にできません。")

    layer = get_named_module(model, layer_name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"{layer_name!r} は Linear である必要があります: {type(layer).__name__}"
        )

    tt_layer = build_tt_linear(layer, out_modes, in_modes, max_rank=max_rank)

    compressed_model = copy.deepcopy(model)
    set_named_module(compressed_model, layer_name, tt_layer)
    return compressed_model
