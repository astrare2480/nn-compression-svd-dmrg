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

    PyTorch の ``set_submodule`` を使う。
    ``nn.Sequential`` の番号付き子（``\"block.0\"``）にも使える。
    存在しないパスは ``strict=True`` でエラーにする。
    """
    get_named_module(model, name)
    model.set_submodule(name, module, strict=True)
