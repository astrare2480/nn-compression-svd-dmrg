"""nn.Module の training 状態と parameter 値を、処理の前後で保存・復元する。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def _preserve_module_training_modes(*models):
    """model と全 submodule の training 状態を保存し、処理後に個別へ復元する。

    ``model.train(state)`` / ``model.eval()`` は全 submodule へ再帰的に
    同じ状態を設定してしまうため、frozen BatchNorm のように一部
    submodule だけ eval() にしていた場合、その個別状態を壊してしまう。
    ここでは各 module インスタンスの ``training`` 属性を直接保存・復元
    することで、この副作用を避ける。正常終了時だけでなく、例外発生時も
    finally で必ず復元する。
    """
    saved_states = [
        [(module, module.training) for module in model.modules()]
        for model in models
    ]
    try:
        yield
    finally:
        for module_states in saved_states:
            for module, state in module_states:
                module.training = state


def snapshot_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """parameter 名から、CPU 上の独立な clone への mapping を返す。

    登録順は結果の意味に影響しない。モデルも parameter も変更しない。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    return {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in model.named_parameters()
    }


def restore_named_parameters(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """名前で対応する parameter へ snapshot の値を書き戻す。

    先に key 集合、shape、dtype をすべて確認する。1 つでも違えば何も書き戻さない。
    書き込みの前に、全 snapshot Tensor を独立した clone にする。その clone を
    各 parameter の現在 device へ書き戻す。``torch.no_grad()`` で行い、モデル自体は
    移動しない。``snapshot`` の Tensor は変更しない。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"snapshot は mapping である必要があります: {type(snapshot)!r}")
    named = dict(model.named_parameters())
    if set(snapshot) != set(named):
        missing = sorted(set(named) - set(snapshot))
        extra = sorted(set(snapshot) - set(named))
        raise ValueError(f"parameter 名が一致しません: missing={missing} extra={extra}")
    prepared: list[tuple[nn.Parameter, torch.Tensor]] = []
    for name, parameter in named.items():
        value = snapshot[name]
        if not torch.is_tensor(value):
            raise TypeError(f"{name} の snapshot は Tensor である必要があります: {type(value)!r}")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"{name} の shape が一致しません: snapshot={tuple(value.shape)} parameter={tuple(parameter.shape)}"
            )
        if value.dtype != parameter.dtype:
            raise ValueError(f"{name} の dtype が一致しません: snapshot={value.dtype} parameter={parameter.dtype}")
        prepared.append((parameter, value.detach().clone()))
    originals = [(parameter, parameter.detach().clone()) for parameter, _value in prepared]
    try:
        with torch.no_grad():
            for parameter, value in prepared:
                parameter.copy_(value.to(device=parameter.device))
    except Exception:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)
        raise
