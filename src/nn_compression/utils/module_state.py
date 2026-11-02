"""nn.Module の training 状態を、処理の前後で個別に保存・復元する。"""

from contextlib import contextmanager


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
