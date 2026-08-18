"""nn.Module を名前で取得・置換する汎用処理。

``\"conv2\"`` のような直下属性だけでなく、
``\"block1.conv\"`` のようなネストした名前も扱う。
"""

from torch import nn


def get_named_module(model: nn.Module, name: str) -> nn.Module:
    """``name`` で指す子モジュールを返す。

    ``model.get_submodule`` と同じく、ドット区切りなら親から辿る。
    """
    return model.get_submodule(name)


def set_named_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """``name`` の位置へ ``module`` を代入する。``model`` をその場で変更する。

    直下なら ``setattr``、ネストなら親の ``_modules`` を更新する。
    ``nn.Sequential`` の番号付き子（``\"0\"``）にも使える。
    """
    if "." not in name:
        parent = model
        child_name = name
    else:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)

    if child_name not in parent._modules:
        raise AttributeError(
            f"{type(parent).__name__} に子モジュール {child_name!r} がありません。"
        )
    parent._modules[child_name] = module
