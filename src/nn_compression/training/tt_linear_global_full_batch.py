"""TT core を同時に更新する、全データ勾配の単純 GD。

隣接 2-core の結合、方向付き SVD、正準中心の移動、pair 巡回は行わない。
それらは ``tt_linear_scoped_update`` と ``tt_linear_full_batch_sweep`` の局所更新である。
このモジュールは、全 TT core とそれ以外の通常 parameter を、同じ全データ平均損失の
勾配で 1 歩だけ同時に動かす。core の shape は変えない。

minibatch は勾配を足し合わせるための分割であり、batch ごとに parameter は更新しない。
足す量は、batch 平均損失 ``L_b`` と batch の sample 数 ``n_b``、全 sample 数 ``N`` に対する
``(n_b / N) * L_b`` の勾配である。``L_b`` は平均なので、さらに ``N`` では割らない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .gpu_resident_batches import GpuResidentTensorBatches
from .tt_linear_full_batch_sweep import (
    NonFiniteLocalUpdate,
    _batch_on_device,
    _core_prefix,
    _full_batch_samples,
    _group_l2_norm,
    _mean_criterion,
    _relative_update,
    _require_finite_number,
    _restore_training_flags,
    _same_device,
    _tensor_is_finite,
    _tt_linear_layer,
)


def _same_stored_values(current: torch.Tensor, saved: torch.Tensor) -> bool:
    """NaN を含む保存値でも、値が変わっていなければ同じとみなす。

    ``torch.equal`` は NaN 同士を不一致にする。勾配集計が値を書いていないことの
    確認では、元から NaN だった parameter を変化と扱わない。
    """
    if current.shape != saved.shape or current.dtype != saved.dtype:
        return False
    current_nan = torch.isnan(current)
    saved_nan = torch.isnan(saved)
    if not torch.equal(current_nan, saved_nan):
        return False
    if not bool(current_nan.any()):
        return torch.equal(current, saved)
    filled_current = current.detach().clone()
    filled_saved = saved.detach().clone()
    filled_current[current_nan] = 0
    filled_saved[saved_nan] = 0
    return torch.equal(filled_current, filled_saved)


@dataclass(frozen=True)
class TTLinearGlobalFullBatchUpdate:
    """成功した 1 回の global 同時更新。失敗時はこのオブジェクトを返さない。

    ``gradient_norm`` と ``relative_update`` は、更新した全 parameter を一つの群にした値。
    core と通常 parameter の内訳は診断用で、別の更新ステップではない。
    ``relative_update`` は ``learning_rate * ||g|| / ||parameter||``。
    ``core_shapes`` は更新後の core shape であり、実効 weight の最小 TT-rank ではない。
    """

    full_batch_loss: float
    processed_samples: int
    parameter_norm: float
    gradient_norm: float
    relative_update: float
    core_parameter_norm: float
    core_gradient_norm: float
    core_relative_update: float
    dense_parameter_norm: float
    dense_gradient_norm: float
    dense_relative_update: float
    core_shapes: tuple[tuple[int, ...], ...]


def accumulate_tt_linear_global_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    batch_size: int | None,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[dict[str, torch.Tensor], float, int]:
    """全 sample の平均損失と、全 parameter の勾配を返す。

    対象は ``model.named_parameters()`` のすべてである。TT core も通常 parameter も、
    呼び出し前の ``requires_grad`` に関係なく、この集計のあいだだけ勾配を有効にする。
    結合テンソルは作らない。SVD は呼ばない。parameter の値は変えない。

    戻り値は ``(grads, full_batch_loss, processed_samples)``。``grads`` は parameter 名から
    ``detach().clone()`` した勾配への dict。``full_batch_loss`` は全 sample の平均損失。
    ``processed_samples`` は集計した sample 数で、``N`` と一致する。
    ``requires_grad`` と既存の ``.grad`` は戻す。
    """
    criterion = _mean_criterion(criterion)
    _tt_linear_layer(model, layer_name)
    if isinstance(source, GpuResidentTensorBatches) and not _same_device(source.device, device):
        raise ValueError(
            "GPU常駐sourceは事前配置したdeviceと一致する必要があります。"
            f" source={source.device}, requested={device}"
        )
    named = list(model.named_parameters())
    if not named:
        raise ValueError("更新する parameter がありません。")
    saved_requires_grad = {name: parameter.requires_grad for name, parameter in named}
    saved_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named
    }
    saved_values = {name: parameter.detach().clone() for name, parameter in named}
    flags = [(module, module.training) for module in model.modules()]
    try:
        for _, parameter in named:
            parameter.requires_grad_(True)
            parameter.grad = None
        dataset_size, batches = _full_batch_samples(source, batch_size)
        full_loss = None
        seen = 0
        for images, labels in batches:
            images, labels = _batch_on_device(images, labels, device)
            if images.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"images の batch 次元 {images.shape[0]} と labels {labels.shape[0]} が一致しません。"
                )
            n_b = int(labels.shape[0])
            logits = model(images)
            loss_b = criterion(logits, labels)
            if not _tensor_is_finite(loss_b) or not _tensor_is_finite(logits):
                raise NonFiniteLocalUpdate("batch 損失または logits が有限ではありません。")
            weighted = loss_b * (n_b / dataset_size)
            weighted.backward()
            detached = weighted.detach()
            full_loss = detached if full_loss is None else full_loss + detached
            seen += n_b
        if seen != dataset_size:
            raise ValueError(
                f"処理した sample 数 {seen} が全 sample 数 {dataset_size} と一致しません。"
            )
        if full_loss is None or not _tensor_is_finite(full_loss):
            raise NonFiniteLocalUpdate("全データの平均損失が有限ではありません。")
        grads: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"{name} の勾配がありません。")
            if not _tensor_is_finite(parameter.grad):
                raise NonFiniteLocalUpdate(f"{name} の勾配が有限ではありません。")
            grads[name] = parameter.grad.detach().clone()
        if set(grads) != set(saved_values):
            raise RuntimeError("勾配の parameter 名が集計前と一致しません。")
        return grads, float(full_loss.detach()), seen
    finally:
        _restore_training_flags(flags)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(saved_requires_grad[name])
            parameter.grad = saved_grads[name]
            if not _same_stored_values(parameter, saved_values[name]):
                raise RuntimeError(f"勾配集計中に parameter が変わりました: {name}")


def update_tt_linear_global_full_batch(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    learning_rate: float,
    batch_size: int | None,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> TTLinearGlobalFullBatchUpdate:
    """全 TT core と通常 parameter を、全データ平均勾配で同時に 1 歩更新する。

    更新は ``parameter - learning_rate * gradient``。momentum、Adam、weight decay、
    SVD、再正準化は使わない。候補がすべて有限で、shape が元と一致するときだけ書き戻す。
    非有限値のときは ``NonFiniteLocalUpdate`` を上げ、書き戻す前なので値は変えない。
    ``learning_rate`` は 0 以上の有限値。0 は値を変えない診断用の歩幅であり、凍結ではない。
    """
    learning_rate = _require_finite_number(learning_rate, "learning_rate")
    if learning_rate < 0:
        raise ValueError(f"learning_rate は 0 以上である必要があります: {learning_rate}")
    layer = _tt_linear_layer(model, layer_name)
    prefix = _core_prefix(layer_name)
    named = list(model.named_parameters())
    core_names = [name for name, _ in named if name.startswith(prefix)]
    dense_names = [name for name, _ in named if not name.startswith(prefix)]
    if not core_names:
        raise ValueError(f"{layer_name} の TT core が parameter にありません。")
    if not dense_names:
        raise ValueError(
            "通常 parameter がありません。全 core だけの更新は、この同時 GD の対象外です。"
        )
    before = {name: parameter.detach().clone() for name, parameter in named}
    shapes = {name: tuple(parameter.shape) for name, parameter in named}
    grads, full_loss, processed_samples = accumulate_tt_linear_global_full_batch_gradients(
        model,
        source,
        layer_name=layer_name,
        batch_size=batch_size,
        device=device,
        criterion=criterion,
    )
    candidates: dict[str, torch.Tensor] = {}
    for name, value in before.items():
        updated = value - learning_rate * grads[name]
        if not _tensor_is_finite(updated):
            raise NonFiniteLocalUpdate(f"{name} の更新候補が有限ではありません。")
        if tuple(updated.shape) != shapes[name]:
            raise ValueError(f"{name} の更新候補 shape が元と一致しません。")
        candidates[name] = updated

    def _norms(names: list[str]) -> tuple[float, float, float]:
        parameter_norm = _group_l2_norm([before[name] for name in names])
        gradient_norm = _group_l2_norm([grads[name] for name in names])
        relative = _relative_update(learning_rate, gradient_norm, parameter_norm)
        return parameter_norm, gradient_norm, relative

    parameter_norm, gradient_norm, relative_update = _norms(list(before))
    core_parameter_norm, core_gradient_norm, core_relative = _norms(core_names)
    dense_parameter_norm, dense_gradient_norm, dense_relative = _norms(dense_names)
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(candidates[name])
    except Exception:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(before[name])
        raise
    for name, parameter in model.named_parameters():
        if tuple(parameter.shape) != shapes[name]:
            with torch.no_grad():
                parameter.copy_(before[name])
            raise RuntimeError(f"{name} の shape が更新で変わりました。")
        if not _tensor_is_finite(parameter):
            with torch.no_grad():
                for saved_name, saved_parameter in model.named_parameters():
                    saved_parameter.copy_(before[saved_name])
            raise NonFiniteLocalUpdate(f"{name} の書き戻し結果が有限ではありません。")
    core_shapes = tuple(tuple(int(size) for size in core.shape) for core in layer.cores)
    if core_shapes != tuple(shapes[name] for name in core_names):
        raise RuntimeError("記録した core shape が更新前と一致しません。")
    if not math.isfinite(full_loss):
        raise NonFiniteLocalUpdate("全データの平均損失が有限ではありません。")
    return TTLinearGlobalFullBatchUpdate(
        full_batch_loss=full_loss,
        processed_samples=processed_samples,
        parameter_norm=parameter_norm,
        gradient_norm=gradient_norm,
        relative_update=relative_update,
        core_parameter_norm=core_parameter_norm,
        core_gradient_norm=core_gradient_norm,
        core_relative_update=core_relative,
        dense_parameter_norm=dense_parameter_norm,
        dense_gradient_norm=dense_gradient_norm,
        dense_relative_update=dense_relative,
        core_shapes=core_shapes,
    )
