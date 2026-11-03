"""全データ勾配による TTLinear の隣接2-core スイープ。

物理 DMRG（``dmrg.sweep``）の局所固有値問題ではない。隣接 TT pair を
結合したテンソルと、TT core 以外の通常 parameter を、全学習データの
平均分類損失の勾配で同時に1歩進め、pair だけを方向付き SVD で戻す。

minibatch は勾配を足し合わせるための分割であり、batch ごとに parameter
は更新しない。
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from ..compression.tt_linear import TTLinear
from ..compression.tt_linear_two_site import (
    assign_tt_linear_cores,
    module_forward_with_merged_tt_linear_pair,
)
from ..tt_mps.matrix_two_site import (
    BondRankDecision,
    merge_tt_matrix_cores,
    replace_tt_matrix_core_pair,
    split_merged_tt_matrix,
    split_merged_tt_matrix_by_discarded_weight,
)
from ..utils.modules import get_named_module
from .gpu_resident_batches import GpuResidentTensorBatches
from .loops import evaluate


class NonFiniteLocalUpdate(RuntimeError):
    """有限でない損失・勾配・候補・診断出力を検出した。

    この例外を上げる時点では、実モデルの parameter 値は書いていない。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class TTLinearLocalUpdate:
    """成功した1回の局所更新の記録。失敗時はこのオブジェクトを返さない。"""

    pair_index: int
    direction: str
    local_step: int
    center: int
    retained_rank: int
    discarded_squared_norm: float
    diagnostic_batch_loss: float
    tt_parameter_norm: float
    tt_gradient_norm: float
    tt_relative_update: float
    dense_parameter_norm: float
    dense_gradient_norm: float
    dense_relative_update: float
    bond_rank_decision: BondRankDecision | None = None


@dataclass(frozen=True)
class TTLinearFullBatchSweepResult:
    """往復学習の停止状態。``best_state`` と ``initial_state`` は独立した複製。"""

    stop_reason: str
    stop_detail: str
    completed_sweeps: int
    successful_local_updates: int
    incomplete_sweep_successes: int
    best_sweep: int
    best_validation_loss: float
    best_state: dict[str, torch.Tensor]
    initial_state: dict[str, torch.Tensor]
    center: int
    last_success_pair: int | None
    last_success_direction: str | None
    history: list[dict]
    local_updates: list[dict]
    stop_sweep: int | None
    stop_pair: int | None
    stop_direction: str | None
    stop_local_step: int | None


def _core_prefix(layer_name: str) -> str:
    return f"{layer_name}.cores."


def _group_l2_norm(tensors: Sequence[torch.Tensor]) -> float:
    """複数 tensor を連結したときの L2 norm。空の列は拒否する。"""
    if len(tensors) == 0:
        raise ValueError("L2 norm を取る tensor がありません。")
    squares = [torch.linalg.vector_norm(tensor.detach()).square() for tensor in tensors]
    value = float(torch.stack(squares).sum().sqrt())
    if not math.isfinite(value):
        raise ValueError(f"L2 norm が有限ではありません: {value}")
    return value


def _relative_update(learning_rate: float, gradient_norm: float, parameter_norm: float) -> float:
    """``learning_rate * ||g|| / ||parameter||``。分母 0 では inf を作らない。

    パラメータも勾配も 0 なら相対更新量は 0。パラメータが 0 で勾配が 0 でないときは
    定義できないので ``ValueError``。この値は診断専用で、更新の受理条件にはしない。
    """
    if parameter_norm == 0.0:
        if gradient_norm == 0.0:
            return 0.0
        raise ValueError(
            "parameter norm が 0 で勾配 norm が 0 でないため、相対更新量を定義できません。"
        )
    value = learning_rate * gradient_norm / parameter_norm
    if not math.isfinite(value):
        raise ValueError(f"相対更新量が有限ではありません: {value}")
    return value


_LOCAL_UPDATE_LOG_FIELDS = (
    "sweep",
    "direction",
    "pair_index",
    "local_step",
    "status",
    "retained_rank",
    "discarded_squared_norm",
    "center_after",
    "diagnostic_batch_loss",
    "tt_parameter_norm",
    "tt_gradient_norm",
    "tt_relative_update",
    "dense_parameter_norm",
    "dense_gradient_norm",
    "dense_relative_update",
    "message",
)


def _require_finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} は有限の実数である必要があります: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} は有限である必要があります: {value!r}")
    return number


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} は int である必要があります: {value!r}")
    if value < 1:
        raise ValueError(f"{name} は 1 以上である必要があります: {value}")
    return value


def _mean_criterion(criterion: nn.Module | None) -> nn.Module:
    if criterion is None:
        return nn.CrossEntropyLoss(reduction="mean")
    reduction = getattr(criterion, "reduction", None)
    if reduction != "mean":
        raise ValueError(
            "criterion.reduction は 'mean' である必要があります。"
            f" 平均損失を (n_b / N) 倍して足すためです: {reduction!r}"
        )
    return criterion


def _tt_linear_layer(model: nn.Module, layer_name: str) -> TTLinear:
    layer = get_named_module(model, layer_name)
    if not isinstance(layer, TTLinear):
        raise TypeError(
            f"{layer_name} は TTLinear である必要があります: {type(layer).__name__}"
        )
    if len(layer.cores) < 2:
        raise ValueError(f"{layer_name} の core 数は 2 以上である必要があります。")
    return layer


def _set_core_grads_disabled(model: nn.Module, layer_name: str) -> None:
    """TT core は凍結し、それ以外の parameter は勾配を有効にする。"""
    prefix = _core_prefix(layer_name)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(prefix))


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy(model.state_dict())


def _sync_module_values(destination: nn.Module, source: nn.Module, layer_name: str) -> None:
    """``source`` の値を ``destination`` へコピーする。storage は共有しない。"""
    source_layer = _tt_linear_layer(source, layer_name)
    destination_layer = _tt_linear_layer(destination, layer_name)
    assign_tt_linear_cores(
        destination_layer,
        [core.detach() for core in source_layer.cores],
        requires_grad=False,
    )
    source_parameters = dict(source.named_parameters())
    prefix = _core_prefix(layer_name)
    for name, parameter in destination.named_parameters():
        if name.startswith(prefix):
            continue
        value = source_parameters[name].detach()
        if parameter.shape != value.shape or parameter.dtype != value.dtype:
            raise ValueError(
                f"{name} の shape / dtype がコピー元と一致しません: "
                f"{tuple(parameter.shape)} {parameter.dtype} と "
                f"{tuple(value.shape)} {value.dtype}"
            )
        with torch.no_grad():
            parameter.copy_(value.to(parameter.device))
        parameter.requires_grad_(True)


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _restore_training_flags(flags: list[tuple[nn.Module, bool]]) -> None:
    for module, training in flags:
        module.training = training


def _canonical_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device)


def _same_device(left: torch.device, right: torch.device) -> bool:
    return _canonical_device(left) == _canonical_device(right)


def _batch_on_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device):
    """同じdeviceならsliceをそのまま返す。違うときだけ ``.to`` する。"""
    if not _same_device(images.device, device):
        images = images.to(device)
    if not _same_device(labels.device, device):
        labels = labels.to(device)
    return images, labels


def _full_batch_samples(source: DataLoader | GpuResidentTensorBatches, batch_size: int | None):
    """DataLoader または GPU 常駐 source から、全sample数と連続batchを返す。"""
    if isinstance(source, DataLoader):
        if source.drop_last:
            raise ValueError("勾配集計の loader は drop_last=False である必要があります。")
        if not isinstance(source.sampler, SequentialSampler):
            raise ValueError(
                "勾配集計の loader は shuffle=False（SequentialSampler）である必要があります。"
            )
        if batch_size is not None and source.batch_size != batch_size:
            raise ValueError("loader の batch_size と引数 batch_size が一致しません。")
        dataset_size = len(source.dataset)
        if dataset_size < 1:
            raise ValueError("空の dataset では全データ勾配を集計できません。")
        return int(dataset_size), source
    if isinstance(source, GpuResidentTensorBatches):
        if batch_size is None:
            raise ValueError("GPU常駐sourceには batch_size が必要です。")
        return source.n_samples, source.batches(batch_size)
    raise TypeError(
        "勾配集計の source は DataLoader または GpuResidentTensorBatches である必要があります: "
        f"{type(source).__name__}"
    )


def accumulate_tt_linear_pair_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    layer_name: str,
    pair_index: int,
    merged: torch.Tensor,
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """全学習データの平均 cross-entropy の勾配を、batch に分けて足す。

    batch の平均損失を ``L_b``、batch の sample 数を ``n_b``、
    全 sample 数を ``N`` とすると、足す量は ``(n_b / N) * L_b`` の勾配である。
    ``L_b`` は平均なので、さらに ``N`` で割らない。

    ``source`` は ``shuffle=False`` の ``DataLoader`` か、
    ``GpuResidentTensorBatches``。後者は ``batch_size`` を明示し、
    ``batches`` の連続sliceをそのまま使う。

    この関数は parameter も ``merged`` も更新しない。各 batch の計算グラフは
    ``backward`` のたびに捨てる。戻り値の勾配は ``detach().clone()`` した複製。

    戻り値:
        ``(grad_T, dense_grads)``。``dense_grads`` は ``requires_grad=True`` の
        parameter 名から勾配への dict。TT core は含めない。
    """
    if not merged.requires_grad or not merged.is_leaf:
        raise ValueError("merged は requires_grad=True の leaf である必要があります。")
    criterion = _mean_criterion(criterion)
    layer = _tt_linear_layer(model, layer_name)
    if pair_index < 0 or pair_index + 1 >= len(layer.cores):
        raise ValueError(f"pair_index={pair_index} は core 列の範囲外です。")
    if isinstance(source, GpuResidentTensorBatches) and not _same_device(source.device, device):
        raise ValueError(
            "GPU常駐sourceは事前配置したdeviceと一致する必要があります。"
            f" source={source.device}, requested={device}"
        )
    dataset_size, batches = _full_batch_samples(source, batch_size)

    merged.grad = None
    for parameter in model.parameters():
        parameter.grad = None

    seen = 0
    for images, labels in batches:
        images, labels = _batch_on_device(images, labels, device)
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"images の batch 次元 {images.shape[0]} と labels {labels.shape[0]} が一致しません。"
            )
        n_b = int(labels.shape[0])
        logits = module_forward_with_merged_tt_linear_pair(
            model,
            images,
            layer_name,
            pair_index,
            merged,
        )
        loss_b = criterion(logits, labels)
        if not _tensor_is_finite(loss_b) or not _tensor_is_finite(logits):
            raise NonFiniteLocalUpdate(
                f"pair={pair_index} の batch 損失または logits が有限ではありません。"
            )
        weighted = loss_b * (n_b / dataset_size)
        weighted.backward()
        seen += n_b

    if seen != dataset_size:
        raise ValueError(
            f"処理した sample 数 {seen} が全 sample 数 {dataset_size} と一致しません。"
            " drop_last や sampler の取りこぼしがないかを確認してください。"
        )
    if merged.grad is None:
        raise RuntimeError("結合テンソル T の勾配がありません。")
    if not _tensor_is_finite(merged.grad):
        raise NonFiniteLocalUpdate("結合テンソル T の勾配が有限ではありません。")

    dense_grads: dict[str, torch.Tensor] = {}
    prefix = _core_prefix(layer_name)
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            continue
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"{name} の勾配がありません。")
        if not _tensor_is_finite(parameter.grad):
            raise NonFiniteLocalUpdate(f"{name} の勾配が有限ではありません。")
        dense_grads[name] = parameter.grad.detach().clone()
    return merged.grad.detach().clone(), dense_grads


def measure_tt_pair_update_scales(
    model: nn.Module,
    loader: DataLoader,
    *,
    layer_name: str,
    pair_index: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float | int | bool]:
    """最初の更新をせずに、1 pair の全データ勾配と相対更新量だけを返す。

    相対更新量は ``learning_rate * ||gradient|| / ||parameter||`` である。
    TT 側の parameter は結合テンソル ``T``、通常 parameter 側は TT core 以外で
    勾配を有効にした tensor を連結した L2 norm である。

    parameter の値は変えない。``requires_grad`` と既存の ``.grad`` も戻す。
    この関数は SVD を呼ばない。
    """
    tt_lr = _require_finite_number(tt_pair_learning_rate, "tt_pair_learning_rate")
    dense_lr = _require_finite_number(
        dense_parameter_learning_rate, "dense_parameter_learning_rate"
    )
    layer = _tt_linear_layer(model, layer_name)
    if pair_index < 0 or pair_index + 1 >= len(layer.cores):
        raise ValueError(f"pair_index={pair_index} は範囲外です。")

    saved_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    saved_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    saved_values = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    try:
        _set_core_grads_disabled(model, layer_name)
        merged = (
            merge_tt_matrix_cores(layer.cores[pair_index], layer.cores[pair_index + 1])
            .detach()
            .clone()
            .requires_grad_(True)
        )
        grad_t, dense_grads = accumulate_tt_linear_pair_full_batch_gradients(
            model,
            loader,
            layer_name,
            pair_index,
            merged,
            device=device,
            criterion=criterion,
        )
        dense_names = [
            name
            for name, _parameter in model.named_parameters()
            if name in dense_grads
        ]
        tt_parameter_norm = _group_l2_norm([merged.detach()])
        tt_gradient_norm = _group_l2_norm([grad_t])
        dense_parameter_norm = _group_l2_norm(
            [saved_values[name] for name in dense_names]
        )
        dense_gradient_norm = _group_l2_norm([dense_grads[name] for name in dense_names])
        return {
            "pair_index": pair_index,
            "num_samples": len(loader.dataset),
            "tt_parameter_norm": tt_parameter_norm,
            "tt_gradient_norm": tt_gradient_norm,
            "tt_relative_update": _relative_update(
                tt_lr, tt_gradient_norm, tt_parameter_norm
            ),
            "dense_parameter_norm": dense_parameter_norm,
            "dense_gradient_norm": dense_gradient_norm,
            "dense_relative_update": _relative_update(
                dense_lr, dense_gradient_norm, dense_parameter_norm
            ),
            "gradients_finite": True,
        }
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(saved_requires_grad[name])
            parameter.grad = saved_grads[name]
            if not torch.equal(parameter, saved_values[name]):
                raise RuntimeError(f"診断中に parameter が変わりました: {name}")


def _diagnostic_forward(
    candidate: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    flags = [(module, module.training) for module in candidate.modules()]
    try:
        candidate.eval()
        with torch.no_grad():
            logits = candidate(images)
            loss = criterion(logits, labels)
    finally:
        _restore_training_flags(flags)
    return logits, loss


def update_tt_linear_two_core_pair(
    model: nn.Module,
    candidate: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    pair_index: int,
    direction: str,
    center: int,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    local_step: int = 0,
    criterion: nn.Module | None = None,
    observer: Callable[[dict], None] | None = None,
    batch_size: int | None = None,
    discarded_weight_threshold: float | None = None,
    min_bond_rank: int = 1,
) -> TTLinearLocalUpdate:
    """1 pair を全データ勾配で1歩進め、方向付き SVD のあと成功時だけ反映する。

    ``candidate`` は検証専用の複製である。実モデルと storage を共有してはならない。
    非有限値のときは ``NonFiniteLocalUpdate`` を上げ、``model`` の値は変えない。
    shape 不整合や SVD の ``LinAlgError`` は、非有限値へ読み替えない。

    直交中心 ``center`` は、この pair の2 site のどちらかにあることを前提にする。
    進入側である必要はない。pair の外にある場合は ``ValueError``。
    成功後の中心は、右向きなら ``pair_index + 1``、左向きなら ``pair_index``。
    ``discarded_weight_threshold`` が ``None`` のときは ``rank_cap`` 固定の分割である。
    値を渡すと、その相対 discarded weight で bond rank を選び、``rank_cap`` は上限になる。
    """
    criterion = _mean_criterion(criterion)
    rank_cap = _require_positive_int(rank_cap, "rank_cap")
    tt_lr = _require_finite_number(tt_pair_learning_rate, "tt_pair_learning_rate")
    dense_lr = _require_finite_number(
        dense_parameter_learning_rate, "dense_parameter_learning_rate"
    )
    local_step = _require_positive_int(local_step + 1, "local_step") - 1
    if direction not in {"right", "left"}:
        raise ValueError(f'direction は "right" または "left" である必要があります: {direction!r}')
    layer = _tt_linear_layer(model, layer_name)
    if pair_index < 0 or pair_index + 1 >= len(layer.cores):
        raise ValueError(f"pair_index={pair_index} は範囲外です。")
    if center not in {pair_index, pair_index + 1}:
        raise ValueError(
            f"center={center} は pair ({pair_index}, {pair_index + 1}) の外です。"
            " 全体の正準化を毎 pair でやり直さず、中心は pair 内に維持します。"
        )
    if diagnostic_images.shape[0] != diagnostic_labels.shape[0] or diagnostic_images.shape[0] < 1:
        raise ValueError("診断 batch は空でなく、images と labels の batch 次元が一致する必要があります。")
    if candidate is model:
        raise ValueError("candidate は model とは別のモジュールである必要があります。")
    for name, parameter in model.named_parameters():
        other = dict(candidate.named_parameters()).get(name)
        if other is None:
            raise ValueError(f"candidate に parameter がありません: {name}")
        if parameter.numel() > 0 and parameter.data_ptr() == other.data_ptr():
            raise ValueError(f"{name} が実モデルと候補モデルで storage を共有しています。")

    if observer is not None:
        observer(
            {
                "pair_index": pair_index,
                "direction": direction,
                "local_step": local_step,
                "center": center,
                "left_core": layer.cores[pair_index].detach().clone(),
                "right_core": layer.cores[pair_index + 1].detach().clone(),
            }
        )

    _set_core_grads_disabled(model, layer_name)
    merged = (
        merge_tt_matrix_cores(layer.cores[pair_index], layer.cores[pair_index + 1])
        .detach()
        .clone()
        .requires_grad_(True)
    )
    grad_t, dense_grads = accumulate_tt_linear_pair_full_batch_gradients(
        model,
        source,
        layer_name,
        pair_index,
        merged,
        device=device,
        criterion=criterion,
        batch_size=batch_size,
    )
    t_candidate = merged.detach() - tt_lr * grad_t
    if not _tensor_is_finite(t_candidate):
        raise NonFiniteLocalUpdate("T_candidate が有限ではありません。")

    dense_candidates: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.startswith(_core_prefix(layer_name)):
            continue
        if name not in dense_grads:
            raise RuntimeError(f"{name} の勾配が集計結果にありません。")
        updated = parameter.detach() - dense_lr * dense_grads[name]
        if not _tensor_is_finite(updated):
            raise NonFiniteLocalUpdate(f"{name} の更新候補が有限ではありません。")
        if updated.shape != parameter.shape:
            raise ValueError(f"{name} の更新候補 shape が元と一致しません。")
        dense_candidates[name] = updated

    # 診断値。受理条件には使わない。反映より前に計算し、分母 0 の inf は作らない。
    dense_names = list(dense_candidates)
    tt_parameter_norm = _group_l2_norm([merged.detach()])
    tt_gradient_norm = _group_l2_norm([grad_t])
    tt_relative_update = _relative_update(tt_lr, tt_gradient_norm, tt_parameter_norm)
    dense_parameter_norm = _group_l2_norm(
        [parameter.detach() for name, parameter in model.named_parameters() if name in dense_candidates]
    )
    dense_gradient_norm = _group_l2_norm([dense_grads[name] for name in dense_names])
    dense_relative_update = _relative_update(
        dense_lr, dense_gradient_norm, dense_parameter_norm
    )

    # SVD は学習グラフに入れない。T_candidate はすでに detach 済み。
    # threshold が None のときは従来の固定 rank_cap 分割のままである。
    decision = None
    if discarded_weight_threshold is None:
        split = split_merged_tt_matrix(
            t_candidate.detach(),
            max_bond=rank_cap,
            direction=direction,
        )
    else:
        split, decision = split_merged_tt_matrix_by_discarded_weight(
            t_candidate.detach(),
            threshold=discarded_weight_threshold,
            min_bond_rank=min_bond_rank,
            max_bond_rank=rank_cap,
            direction=direction,
        )
    if (
        not _tensor_is_finite(split.left_core)
        or not _tensor_is_finite(split.right_core)
        or not _tensor_is_finite(split.singular_values)
        or not _tensor_is_finite(split.discarded_squared_norm)
    ):
        raise NonFiniteLocalUpdate("SVD 結果が有限ではありません。")
    if split.left_core.dtype != layer.cores[pair_index].dtype:
        raise ValueError("SVD 後の core dtype が元の core と一致しません。")
    if split.left_core.device != layer.cores[pair_index].device:
        raise ValueError("SVD 後の core device が元の core と一致しません。")
    new_cores = replace_tt_matrix_core_pair(
        [core.detach() for core in layer.cores],
        pair_index,
        split.left_core,
        split.right_core,
    )

    _sync_module_values(candidate, model, layer_name)
    assign_tt_linear_cores(
        _tt_linear_layer(candidate, layer_name),
        new_cores,
        requires_grad=False,
    )
    candidate_parameters = dict(candidate.named_parameters())
    for name, value in dense_candidates.items():
        parameter = candidate_parameters[name]
        with torch.no_grad():
            parameter.copy_(value.to(parameter.device))

    images, labels = _batch_on_device(diagnostic_images, diagnostic_labels, device)
    logits, diagnostic_loss = _diagnostic_forward(candidate, images, labels, criterion)
    if not _tensor_is_finite(logits) or not _tensor_is_finite(diagnostic_loss):
        raise NonFiniteLocalUpdate(
            "固定診断 batch の候補 forward または loss が有限ではありません。"
            " これはその batch の確認であり、全学習データの有限性は保証しない。"
        )

    _sync_module_values(model, candidate, layer_name)
    new_center = pair_index + 1 if direction == "right" else pair_index
    return TTLinearLocalUpdate(
        pair_index=pair_index,
        direction=direction,
        local_step=local_step,
        center=new_center,
        retained_rank=int(split.retained_rank),
        discarded_squared_norm=float(split.discarded_squared_norm.detach()),
        diagnostic_batch_loss=float(diagnostic_loss.detach()),
        tt_parameter_norm=tt_parameter_norm,
        tt_gradient_norm=tt_gradient_norm,
        tt_relative_update=tt_relative_update,
        dense_parameter_norm=dense_parameter_norm,
        dense_gradient_norm=dense_gradient_norm,
        dense_relative_update=dense_relative_update,
        bond_rank_decision=decision,
    )


@dataclass
class _SweepCursor:
    center: int
    successes: int = 0
    cumulative: int = 0
    last_pair: int | None = None
    last_direction: str | None = None
    sweep_index: int = 0
    pair_index: int | None = None
    direction: str | None = None
    local_step: int | None = None


def _sweep_pairs(
    model: nn.Module,
    candidate: nn.Module,
    loader: DataLoader,
    pair_indices: Sequence[int],
    *,
    direction: str,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    local_steps_per_pair: int,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
    cursor: _SweepCursor,
    logs: list[dict],
    observer: Callable[[dict], None] | None,
) -> None:
    for pair_index in pair_indices:
        for local_step in range(local_steps_per_pair):
            cursor.sweep_index = cursor.sweep_index
            cursor.pair_index = pair_index
            cursor.direction = direction
            cursor.local_step = local_step
            result = update_tt_linear_two_core_pair(
                model,
                candidate,
                loader,
                layer_name=layer_name,
                pair_index=pair_index,
                direction=direction,
                center=cursor.center,
                rank_cap=rank_cap,
                tt_pair_learning_rate=tt_pair_learning_rate,
                dense_parameter_learning_rate=dense_parameter_learning_rate,
                diagnostic_images=diagnostic_images,
                diagnostic_labels=diagnostic_labels,
                device=device,
                local_step=local_step,
                criterion=criterion,
                observer=observer,
            )
            cursor.center = result.center
            cursor.successes += 1
            cursor.cumulative += 1
            cursor.last_pair = pair_index
            cursor.last_direction = direction
            logs.append(
                {
                    "sweep": cursor.sweep_index,
                    "direction": direction,
                    "pair_index": pair_index,
                    "local_step": local_step,
                    "status": "ok",
                    "retained_rank": result.retained_rank,
                    "discarded_squared_norm": result.discarded_squared_norm,
                    "center_after": result.center,
                    "diagnostic_batch_loss": result.diagnostic_batch_loss,
                    "tt_parameter_norm": result.tt_parameter_norm,
                    "tt_gradient_norm": result.tt_gradient_norm,
                    "tt_relative_update": result.tt_relative_update,
                    "dense_parameter_norm": result.dense_parameter_norm,
                    "dense_gradient_norm": result.dense_gradient_norm,
                    "dense_relative_update": result.dense_relative_update,
                    "message": "",
                }
            )


def sweep_tt_linear_pairs_left_to_right(
    model: nn.Module,
    candidate: nn.Module,
    loader: DataLoader,
    *,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    local_steps_per_pair: int,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    cursor: _SweepCursor,
    logs: list[dict],
    criterion: nn.Module | None = None,
    observer: Callable[[dict], None] | None = None,
) -> _SweepCursor:
    """pair ``(0,1), (1,2), ...`` を右向き分割で更新する。"""
    layer = _tt_linear_layer(model, layer_name)
    _sweep_pairs(
        model,
        candidate,
        loader,
        range(len(layer.cores) - 1),
        direction="right",
        layer_name=layer_name,
        rank_cap=rank_cap,
        tt_pair_learning_rate=tt_pair_learning_rate,
        dense_parameter_learning_rate=dense_parameter_learning_rate,
        local_steps_per_pair=local_steps_per_pair,
        diagnostic_images=diagnostic_images,
        diagnostic_labels=diagnostic_labels,
        device=device,
        criterion=_mean_criterion(criterion),
        cursor=cursor,
        logs=logs,
        observer=observer,
    )
    return cursor


def sweep_tt_linear_pairs_right_to_left(
    model: nn.Module,
    candidate: nn.Module,
    loader: DataLoader,
    *,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    local_steps_per_pair: int,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    cursor: _SweepCursor,
    logs: list[dict],
    criterion: nn.Module | None = None,
    observer: Callable[[dict], None] | None = None,
) -> _SweepCursor:
    """pair ``(d-2,d-1), ... , (0,1)`` を左向き分割で更新する。"""
    layer = _tt_linear_layer(model, layer_name)
    _sweep_pairs(
        model,
        candidate,
        loader,
        range(len(layer.cores) - 2, -1, -1),
        direction="left",
        layer_name=layer_name,
        rank_cap=rank_cap,
        tt_pair_learning_rate=tt_pair_learning_rate,
        dense_parameter_learning_rate=dense_parameter_learning_rate,
        local_steps_per_pair=local_steps_per_pair,
        diagnostic_images=diagnostic_images,
        diagnostic_labels=diagnostic_labels,
        device=device,
        criterion=_mean_criterion(criterion),
        cursor=cursor,
        logs=logs,
        observer=observer,
    )
    return cursor


def _advance_stopping(
    baseline: float,
    best_loss: float,
    stall_count: int,
    validation_loss: float,
    min_delta: float,
) -> tuple[float, float, int, bool, bool]:
    """early-stopping の基準と、それとは独立な best を1歩進める。

    改善は ``baseline - validation_loss > min_delta`` のときだけである。
    best は ``min_delta`` に関係なく、厳密に小さい validation loss で更新する。
    """
    improved = (baseline - validation_loss) > min_delta
    if improved:
        baseline = validation_loss
        stall_count = 0
    else:
        stall_count += 1
    best_updated = validation_loss < best_loss
    if best_updated:
        best_loss = validation_loss
    return baseline, best_loss, stall_count, improved, best_updated


def _evaluate_split(
    model: nn.Module,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
    validation_loss, validation_acc = evaluate(model, validation_loader, criterion, device)
    return train_loss, train_acc, validation_loss, validation_acc


def train_tt_linear_full_batch_sweeps(
    model: nn.Module,
    gradient_loader: DataLoader,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    max_sweeps: int,
    local_steps_per_pair: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    center: int = 0,
    criterion: nn.Module | None = None,
    observer: Callable[[dict], None] | None = None,
    early_stopping_enabled: bool = True,
) -> TTLinearFullBatchSweepResult:
    """左端から右へ、続けて右端から左へを1往復として繰り返す。

    呼び出し時点の ``model`` を sweep 0（初期モデル）として評価する。
    学習開始前の正準化はこの関数では行わない。``center`` は pair 更新の
    前提であり、往復の開始時は site 0 を想定する。

    1往復が正常に終わると中心は site 0 に戻る。途中で失敗した往復は
    完了数に入れず、最後に成功した局所更新までの ``model`` を残す。
    その途中状態は best にしない。

    ``early_stopping_enabled=False`` のときは、非改善回数では止まらない。
    best の更新は続ける。``patience`` を大きくして止める代わりにはしない。
    既定の ``True`` では、これまでどおり ``patience`` と ``min_delta`` で止まる。

    結果へ変換して返すのは ``NonFiniteLocalUpdate`` と
    ``torch.linalg.LinAlgError`` だけである。``ValueError``、
    ``AssertionError``、その他の例外は捕捉せず呼び出し元へ返す。
    """
    criterion = _mean_criterion(criterion)
    rank_cap = _require_positive_int(rank_cap, "rank_cap")
    tt_lr = _require_finite_number(tt_pair_learning_rate, "tt_pair_learning_rate")
    dense_lr = _require_finite_number(
        dense_parameter_learning_rate, "dense_parameter_learning_rate"
    )
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or max_sweeps < 0:
        raise ValueError(f"max_sweeps は 0 以上の int である必要があります: {max_sweeps!r}")
    local_steps = _require_positive_int(local_steps_per_pair, "local_steps_per_pair")
    patience = _require_positive_int(early_stopping_patience, "early_stopping_patience")
    min_delta = _require_finite_number(early_stopping_min_delta, "early_stopping_min_delta")
    if min_delta < 0:
        raise ValueError(f"early_stopping_min_delta は 0 以上である必要があります: {min_delta}")
    if not isinstance(early_stopping_enabled, bool):
        raise TypeError(
            "early_stopping_enabled は bool である必要があります: "
            f"{early_stopping_enabled!r}"
        )
    if isinstance(center, bool) or not isinstance(center, int):
        raise TypeError(f"center は int である必要があります: {center!r}")
    _tt_linear_layer(model, layer_name)
    for loader_name, loader in (
        ("gradient_loader", gradient_loader),
        ("train_eval_loader", train_eval_loader),
        ("validation_loader", validation_loader),
    ):
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError(f"{loader_name} は shuffle=False である必要があります。")
    _set_core_grads_disabled(model, layer_name)

    candidate = copy.deepcopy(model)
    initial_state = _clone_state_dict(model)
    history: list[dict] = []
    logs: list[dict] = []
    cursor = _SweepCursor(center=center)

    train_loss, train_acc, validation_loss, validation_acc = _evaluate_split(
        model, train_eval_loader, validation_loader, criterion, device
    )
    if not all(math.isfinite(value) for value in (train_loss, train_acc, validation_loss, validation_acc)):
        raise NonFiniteLocalUpdate("sweep 0 の評価指標が有限ではありません。")
    baseline = validation_loss
    best_loss = validation_loss
    best_sweep = 0
    best_state = _clone_state_dict(model)
    stall_count = 0
    history.append(
        {
            "sweep": 0,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "validation_loss": validation_loss,
            "validation_acc": validation_acc,
            "cumulative_successful_updates": 0,
            "center": cursor.center,
            "early_stop_baseline": baseline,
            "stall_count": stall_count,
            "is_best": True,
        }
    )

    stop_reason = "max_sweeps"
    if early_stopping_enabled:
        stop_detail = f"最大往復数 {max_sweeps} まで実行した。これは数学的収束の証明ではない。"
    else:
        stop_detail = (
            f"最大往復数 {max_sweeps} まで実行した。early stopping は無効。"
            "これは数学的収束の証明ではない。"
        )
    incomplete = 0
    completed = 0
    stop_sweep: int | None = None
    stop_pair: int | None = None
    stop_direction: str | None = None
    stop_local_step: int | None = None

    for sweep_index in range(1, max_sweeps + 1):
        cursor.sweep_index = sweep_index
        cursor.successes = 0
        failure: BaseException | None = None
        try:
            sweep_tt_linear_pairs_left_to_right(
                model,
                candidate,
                gradient_loader,
                layer_name=layer_name,
                rank_cap=rank_cap,
                tt_pair_learning_rate=tt_lr,
                dense_parameter_learning_rate=dense_lr,
                local_steps_per_pair=local_steps,
                diagnostic_images=diagnostic_images,
                diagnostic_labels=diagnostic_labels,
                device=device,
                cursor=cursor,
                logs=logs,
                criterion=criterion,
                observer=observer,
            )
            sweep_tt_linear_pairs_right_to_left(
                model,
                candidate,
                gradient_loader,
                layer_name=layer_name,
                rank_cap=rank_cap,
                tt_pair_learning_rate=tt_lr,
                dense_parameter_learning_rate=dense_lr,
                local_steps_per_pair=local_steps,
                diagnostic_images=diagnostic_images,
                diagnostic_labels=diagnostic_labels,
                device=device,
                cursor=cursor,
                logs=logs,
                criterion=criterion,
                observer=observer,
            )
        except NonFiniteLocalUpdate as exc:
            failure = exc
            stop_reason = "non_finite"
            stop_detail = str(exc)
        except torch.linalg.LinAlgError as exc:
            failure = exc
            stop_reason = "svd_failure"
            stop_detail = f"{type(exc).__name__}: {exc}"
        if failure is not None:
            incomplete = cursor.successes
            stop_sweep = sweep_index
            stop_pair = cursor.pair_index
            stop_direction = cursor.direction
            stop_local_step = cursor.local_step
            logs.append(
                {
                    "sweep": sweep_index,
                    "direction": cursor.direction,
                    "pair_index": cursor.pair_index,
                    "local_step": cursor.local_step,
                    "status": stop_reason,
                    "retained_rank": None,
                    "discarded_squared_norm": None,
                    "center_after": cursor.center,
                    "diagnostic_batch_loss": None,
                    "tt_parameter_norm": None,
                    "tt_gradient_norm": None,
                    "tt_relative_update": None,
                    "dense_parameter_norm": None,
                    "dense_gradient_norm": None,
                    "dense_relative_update": None,
                    "message": stop_detail,
                }
            )
            break

        train_loss, train_acc, validation_loss, validation_acc = _evaluate_split(
            model, train_eval_loader, validation_loader, criterion, device
        )
        if not all(
            math.isfinite(value) for value in (train_loss, train_acc, validation_loss, validation_acc)
        ):
            incomplete = cursor.successes
            stop_reason = "non_finite"
            stop_detail = f"sweep {sweep_index} の往復後評価が有限ではありません。"
            stop_sweep = sweep_index
            break
        completed += 1
        baseline, best_loss, stall_count, _, best_updated = _advance_stopping(
            baseline,
            best_loss,
            stall_count,
            validation_loss,
            min_delta,
        )
        if best_updated:
            best_sweep = sweep_index
            best_state = _clone_state_dict(model)
        history.append(
            {
                "sweep": sweep_index,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "validation_loss": validation_loss,
                "validation_acc": validation_acc,
                "cumulative_successful_updates": cursor.cumulative,
                "center": cursor.center,
                "early_stop_baseline": baseline,
                "stall_count": stall_count,
                "is_best": best_sweep == sweep_index,
            }
        )
        if early_stopping_enabled and stall_count >= patience:
            stop_reason = "early_stopping"
            stop_detail = (
                f"validation loss が min_delta={min_delta} を超えて改善しないまま "
                f"{patience} 往復続いた。数学的収束の証明ではない。"
            )
            break
    else:
        incomplete = 0

    if stop_reason == "max_sweeps":
        incomplete = 0

    return TTLinearFullBatchSweepResult(
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        completed_sweeps=completed,
        successful_local_updates=cursor.cumulative,
        incomplete_sweep_successes=incomplete,
        best_sweep=best_sweep,
        best_validation_loss=best_loss,
        best_state=best_state,
        initial_state=initial_state,
        center=cursor.center,
        last_success_pair=cursor.last_pair,
        last_success_direction=cursor.last_direction,
        history=history,
        local_updates=logs,
        stop_sweep=stop_sweep,
        stop_pair=stop_pair,
        stop_direction=stop_direction,
        stop_local_step=stop_local_step,
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"JSON にできない設定値です: {type(value)!r}")


def save_tt_linear_full_batch_sweep_run(
    directory: Path | str,
    model: nn.Module,
    result: TTLinearFullBatchSweepResult,
    config: Mapping,
) -> None:
    """初期・最終・best と、往復履歴・局所ログ・状態 JSON を書く。

    最終モデルは、いまの ``model``（最後に成功した局所更新まで）である。
    best はその state を上書きせず、別ファイルへ書く。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layer_name = config["layer_name"]
    payload_common = {
        "layer_name": layer_name,
        "out_modes": list(config["out_modes"]),
        "in_modes": list(config["in_modes"]),
    }
    torch.save(
        {**payload_common, "state_dict": result.initial_state},
        directory / "initial_model.pt",
    )
    torch.save(
        {**payload_common, "state_dict": model.state_dict()},
        directory / "final_model.pt",
    )
    torch.save(
        {**payload_common, "state_dict": result.best_state},
        directory / "best_model.pt",
    )

    history_fields = [
        "sweep",
        "train_loss",
        "train_acc",
        "validation_loss",
        "validation_acc",
        "cumulative_successful_updates",
        "center",
        "early_stop_baseline",
        "stall_count",
        "is_best",
    ]
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(result.history)

    log_fields = list(_LOCAL_UPDATE_LOG_FIELDS)
    with (directory / "local_updates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(result.local_updates)

    layer = _tt_linear_layer(model, layer_name)
    ranks = [int(layer.cores[0].shape[0])]
    ranks.extend(int(core.shape[-1]) for core in layer.cores)
    state = {
        "config": _json_ready(dict(config)),
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "completed_sweeps": result.completed_sweeps,
        "successful_local_updates": result.successful_local_updates,
        "incomplete_sweep_successes": result.incomplete_sweep_successes,
        "best_sweep": result.best_sweep,
        "best_validation_loss": result.best_validation_loss,
        "center": result.center,
        "last_success_pair": result.last_success_pair,
        "last_success_direction": result.last_success_direction,
        "stop_sweep": result.stop_sweep,
        "stop_pair": result.stop_pair,
        "stop_direction": result.stop_direction,
        "stop_local_step": result.stop_local_step,
        "final_ranks": ranks,
    }
    (directory / "run_state.json").write_text(
        json.dumps(_json_ready(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tt_linear_sweep_checkpoint(path: Path | str, model: nn.Module) -> nn.Module:
    """保存した state を ``model`` へ読み、通常 forward に使える状態で返す。

    core の bond が現在の ``model`` と違っても、``assign_tt_linear_cores`` で差し替える。
    それ以外の parameter は shape が一致する必要がある。
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    layer_name = payload["layer_name"]
    state = payload["state_dict"]
    prefix = _core_prefix(layer_name)
    core_keys = sorted(
        (key for key in state if key.startswith(prefix)),
        key=lambda key: int(key.rsplit(".", 1)[-1]),
    )
    layer = _tt_linear_layer(model, layer_name)
    # map_location="cpu" で読んだ core を、読み込み先 layer の device へ揃える。
    target_device = layer.cores[0].device
    cores = [state[key].to(device=target_device) for key in core_keys]
    assign_tt_linear_cores(layer, cores, requires_grad=False)
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    for key, value in state.items():
        if key.startswith(prefix):
            continue
        if key not in named:
            raise KeyError(f"読み込み先に parameter がありません: {key}")
        target = named[key]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(
                f"{key} の shape / dtype が一致しません: "
                f"{tuple(target.shape)} {target.dtype} と {tuple(value.shape)} {value.dtype}"
            )
        with torch.no_grad():
            target.copy_(value.to(target.device))
    return model


def _condition_rows_by_id(rows: Sequence[Mapping], label: str) -> dict[str, Mapping]:
    found: dict[str, Mapping] = {}
    for row in rows:
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or condition_id == "":
            raise ValueError(f"{label} に condition_id がありません。")
        if condition_id in found:
            raise ValueError(f"condition_id が重複しています: {condition_id}")
        found[condition_id] = row
    return found


def _setting_number(value):
    if isinstance(value, bool) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return value
    return number


def _settings_match(recorded: Mapping, expected: Mapping, fields: Sequence[str], label: str) -> None:
    for field in fields:
        if field not in recorded or field not in expected:
            raise ValueError(f"{label} の設定 {field} がありません。")
        left = _setting_number(recorded[field])
        right = _setting_number(expected[field])
        if isinstance(left, float) and isinstance(right, float):
            matched = math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)
        else:
            matched = left == right
        if not matched:
            raise ValueError(
                f"{label} の設定 {field} が一致しません: 記録={recorded[field]!r} 今回={expected[field]!r}"
            )


def load_condition_summary(path: Path | str) -> list[dict[str, str]]:
    """集計CSVを読む。ファイルが無ければ空リストを返す。

    ``condition_id`` の空欄と重複は ``ValueError`` にする。
    """
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _condition_rows_by_id(rows, str(path))
    return rows


def _write_condition_summary_atomic(
    path: Path,
    rows: Sequence[Mapping],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _recorded_condition_is_skippable(
    directory: Path,
    recorded: Mapping,
    expected: Mapping,
    setting_fields: Sequence[str],
    artifacts: Sequence[str],
) -> None:
    condition_id = expected["condition_id"]
    missing = [name for name in artifacts if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            f"{condition_id} は集計CSVにあります。成果物が足りないためスキップしません: {missing}"
        )
    _settings_match(recorded, expected, setting_fields, f"集計CSVの {condition_id}")
    payload = json.loads((directory / "run_state.json").read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{condition_id} の run_state.json に config がありません。")
    if config.get("condition_id") != condition_id:
        raise ValueError(
            f"{condition_id} の run_state.json の condition_id が一致しません: "
            f"{config.get('condition_id')!r}"
        )
    _settings_match(config, expected, setting_fields, f"run_state.json の {condition_id}")


def run_resumable_condition_grid(
    planned: Sequence[Mapping],
    *,
    summary_path: Path | str,
    models_dir: Path | str,
    fieldnames: Sequence[str],
    setting_fields: Sequence[str],
    run_one: Callable[[Mapping], Mapping],
    artifacts: Sequence[str] = (
        "initial_model.pt",
        "final_model.pt",
        "best_model.pt",
        "history.csv",
        "local_updates.csv",
        "run_state.json",
    ),
) -> list[dict[str, str]]:
    """記録済み条件を検証してスキップし、未実行条件の終了ごとに集計CSVを更新する。

    条件ディレクトリがあるのに集計行がない場合は、削除・上書き・スキップをせず
    ``RuntimeError`` で止まる。``run_one`` の例外は捕捉しない。それ以前に
    終了した条件の集計は、各条件の直後に保存してある。
    """
    summary_path = Path(summary_path)
    models_dir = Path(models_dir)
    recorded = load_condition_summary(summary_path)
    recorded_by_id = _condition_rows_by_id(recorded, str(summary_path))
    _condition_rows_by_id(planned, "実行予定")
    for spec in planned:
        condition_id = spec["condition_id"]
        directory = models_dir / condition_id
        if condition_id in recorded_by_id:
            _recorded_condition_is_skippable(
                directory,
                recorded_by_id[condition_id],
                spec,
                setting_fields,
                artifacts,
            )
            continue
        if directory.exists():
            raise RuntimeError(
                f"{condition_id} のディレクトリがありますが、集計CSVに記録がありません。"
                "不完全な実行の可能性があるため、削除・上書き・スキップはしません: "
                f"{directory}"
            )
        row = run_one(spec)
        if row.get("condition_id") != condition_id:
            raise RuntimeError(
                f"run_one が別の condition_id を返しました: {row.get('condition_id')!r}"
            )
        _settings_match(row, spec, setting_fields, f"今回の結果 {condition_id}")
        missing = [name for name in artifacts if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(
                f"{condition_id} の学習は終わったが、成果物がありません: {missing}。"
                "SAVE_RESULTS=False のままだと保存されません。"
                f" 保存先の想定: {directory}"
            )
        _recorded_condition_is_skippable(directory, row, spec, setting_fields, artifacts)
        recorded.append(dict(row))
        recorded_by_id[condition_id] = recorded[-1]
        _write_condition_summary_atomic(summary_path, recorded, fieldnames)
    return load_condition_summary(summary_path)
