"""複数の named 層を、層種別ごとに低ランク分解する。"""

import copy

from torch import nn

from ..utils.modules import get_named_module, set_named_module
from .conv_svd import factorize_conv2d_layer
from .linear_svd import factorize_linear_layer


def factorize_named_layers(
    model: nn.Module,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
) -> nn.Module:
    """``layer_name → rank`` で指定した層だけを分解したコピーを返す。

    Conv2d と Linear は分解の仕方が違うので、辞書を分ける。
    元の ``model`` は変更しない。分解は必ず未分解の元層から行う。
    ``deepcopy`` するため Fine-tune しても baseline の Parameter は動かない。
    """
    compressed_model = copy.deepcopy(model)

    for layer_name, rank in (conv_ranks or {}).items():
        layer = get_named_module(model, layer_name)
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(
                f"{layer_name!r} は Conv2d である必要があります: "
                f"{type(layer).__name__}"
            )
        set_named_module(
            compressed_model,
            layer_name,
            factorize_conv2d_layer(layer, rank),
        )

    for layer_name, rank in (linear_ranks or {}).items():
        layer = get_named_module(model, layer_name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"{layer_name!r} は Linear である必要があります: "
                f"{type(layer).__name__}"
            )
        set_named_module(
            compressed_model,
            layer_name,
            factorize_linear_layer(layer, rank),
        )

    return compressed_model
