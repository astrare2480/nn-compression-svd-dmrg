"""層名で指した module を、置換対象として取り出す非公開 helper。

``factorize_named_tt_linear`` と ``factorize_named_tt_conv2d`` は、
「層名の型・空文字を検証し、``get_named_module`` で取り出し、期待する層種別か
確かめる」前半が同じである。同じ検証を2か所へ置かないように、ここへ集める。
置換（``deepcopy`` と ``set_named_module``）は各 factorize 関数が行う。
"""

from __future__ import annotations

from torch import nn

from ..utils.modules import get_named_module


def _get_named_layer(
    model: nn.Module,
    layer_name: str,
    layer_type: type[nn.Module],
) -> nn.Module:
    """``layer_name`` が指す ``layer_type`` の層を返す。``model`` は変更しない。

    例外:
        ``layer_name`` が ``str`` でなければ ``TypeError``、空文字なら ``ValueError``。
        存在しない名前は ``get_submodule`` と同じ ``AttributeError``。
        指す module が ``layer_type`` でなければ ``TypeError``。
    """
    if not isinstance(layer_name, str):
        raise TypeError(
            f"layer_name は str である必要があります: {type(layer_name).__name__}"
        )
    if layer_name == "":
        raise ValueError("layer_name は空文字にできません。")

    layer = get_named_module(model, layer_name)
    if not isinstance(layer, layer_type):
        raise TypeError(
            f"{layer_name!r} は {layer_type.__name__} である必要があります: "
            f"{type(layer).__name__}"
        )
    return layer
