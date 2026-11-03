"""TTLinear の 1 層を、discarded weight で bond rank を変えながら往復更新する。

保存、図、表は作らない。固定 rank の更新そのものは
``update_tt_linear_two_core_pair`` を使う。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..compression.tt_linear import TTLinear
from ..tt_mps import tt_matrix_ranks
from ..utils.modules import get_named_module
from .gpu_resident_batches import GpuResidentTensorBatches
from .loops import evaluate
from .tt_linear_full_batch_sweep import NonFiniteLocalUpdate, update_tt_linear_two_core_pair
from .validation_selection import (
    ValidationSelection,
    initial_validation_selection,
    update_validation_selection,
)


@dataclass(frozen=True)
class AdaptiveLocalUpdate:
    """1 回の局所更新で記録する rank と discarded weight。"""

    sweep: int
    pair_index: int
    direction: str
    local_step: int
    rank_before: int
    rank_after: int
    actual_ranks: tuple[int, ...]
    center_before: int
    center_after: int
    relative_discarded_weight: float
    discarded_squared_norm: float
    cumulative_updates: int
    diagnostic_batch_loss: float


@dataclass(frozen=True)
class AdaptiveSweepRecord:
    """往復が 1 回終わるごとの validation と選択状態。sweep 0 は更新前。"""

    sweep: int
    cumulative_local_updates: int
    validation_loss: float
    validation_accuracy: float
    center: int
    became_strict_best: bool
    strict_best_loss: float
    strict_best_step: int
    early_stop_triggered: bool


@dataclass(frozen=True)
class AdaptiveTTLinearSweepResult:
    """adaptive sweep の停止位置、最良状態、履歴。"""

    stop_reason: str
    stop_detail: str
    completed_sweeps: int
    total_local_updates: int
    center: int
    best_sweep: int | None
    best_state: dict[str, torch.Tensor]
    final_state: dict[str, torch.Tensor]
    history: tuple[AdaptiveSweepRecord, ...]
    local_updates: tuple[AdaptiveLocalUpdate, ...]
    selection: ValidationSelection | None


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} は 1 以上の int である必要があります: {value!r}")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} は 0 以上の int である必要があります: {value!r}")
    return value


def _require_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"discarded_weight_threshold は実数である必要があります: {value!r}")
    number = float(value)
    if not torch.isfinite(torch.tensor(number)) or number < 0.0:
        raise ValueError(f"discarded_weight_threshold は 0 以上の有限値である必要があります: {value!r}")
    return number


def _layer(model: nn.Module, layer_name: str) -> TTLinear:
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError(f"layer_name は空でない文字列である必要があります: {layer_name!r}")
    layer = get_named_module(model, layer_name)
    if not isinstance(layer, TTLinear):
        raise TypeError(f"{layer_name} は TTLinear である必要があります: {type(layer).__name__}")
    return layer


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _round_trip(n_cores: int) -> list[tuple[int, str]]:
    right = [(index, "right") for index in range(n_cores - 1)]
    left = [(index, "left") for index in range(n_cores - 2, -1, -1)]
    return right + left


def _bond_rank(layer: TTLinear, pair_index: int) -> int:
    return int(tt_matrix_ranks(list(layer.cores))[pair_index + 1])


def train_adaptive_tt_linear_full_batch_sweeps(
    model: nn.Module,
    gradient_source: DataLoader | GpuResidentTensorBatches,
    validation_loader: DataLoader,
    *,
    layer_name: str,
    discarded_weight_threshold: float,
    rank_cap: int,
    min_bond_rank: int = 1,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    max_sweeps: int,
    local_steps_per_pair: int,
    min_delta: float,
    patience: int,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    batch_size: int | None = None,
    criterion: nn.Module | None = None,
    center: int = 0,
) -> AdaptiveTTLinearSweepResult:
    """左から右、続けて右から左を 1 往復として ``max_sweeps`` 回まで更新する。

    各局所更新の bond rank は ``discarded_weight_threshold`` で選び、
    ``rank_cap`` を超えない。``layer_name`` で対象の TTLinear を指定する。
    モデル内の保存や図は作らない。

    失敗した局所更新は log に残さず、それまでの ``model`` を final とする。
    ``NonFiniteLocalUpdate`` は ``non_finite``、``torch.linalg.LinAlgError`` は
    ``svd_failure``、特異値を拒否する ``ValueError`` は ``invalid_singular_values``、
    ``torch.cuda.OutOfMemoryError`` は ``out_of_memory`` として返す。
    それ以外の例外は送出する。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not isinstance(gradient_source, (DataLoader, GpuResidentTensorBatches)):
        raise TypeError("gradient_source は DataLoader または GpuResidentTensorBatches です。")
    if not isinstance(validation_loader, DataLoader):
        raise TypeError("validation_loader は DataLoader である必要があります。")
    layer = _layer(model, layer_name)
    if len(layer.cores) < 2:
        raise ValueError("adaptive sweep には 2 site 以上の TTLinear が必要です。")
    threshold = _require_threshold(discarded_weight_threshold)
    cap = _require_positive_int(rank_cap, "rank_cap")
    minimum = _require_positive_int(min_bond_rank, "min_bond_rank")
    if minimum > cap:
        raise ValueError(f"min_bond_rank は rank_cap 以下である必要があります: {minimum} > {cap}")
    sweeps = _require_nonnegative_int(max_sweeps, "max_sweeps")
    local_steps = _require_positive_int(local_steps_per_pair, "local_steps_per_pair")
    if isinstance(center, bool) or not isinstance(center, int):
        raise TypeError(f"center は int である必要があります: {center!r}")
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    candidate = copy.deepcopy(model)
    visits = _round_trip(len(layer.cores))
    history: list[AdaptiveSweepRecord] = []
    updates: list[AdaptiveLocalUpdate] = []
    best_state = _state_copy(model)
    best_sweep = 0
    selection: ValidationSelection | None = None

    def finish(
        stop_reason: str,
        stop_detail: str,
        completed: int,
    ) -> AdaptiveTTLinearSweepResult:
        return AdaptiveTTLinearSweepResult(
            stop_reason=stop_reason,
            stop_detail=stop_detail,
            completed_sweeps=completed,
            total_local_updates=len(updates),
            center=center,
            best_sweep=best_sweep,
            best_state=best_state,
            final_state=_state_copy(model),
            history=tuple(history),
            local_updates=tuple(updates),
            selection=selection,
        )

    def record_validation(sweep: int, loss: float, accuracy: float, became_best: bool) -> None:
        assert selection is not None
        history.append(
            AdaptiveSweepRecord(
                sweep=sweep,
                cumulative_local_updates=len(updates),
                validation_loss=loss,
                validation_accuracy=accuracy,
                center=center,
                became_strict_best=became_best,
                strict_best_loss=selection.strict_best_loss,
                strict_best_step=selection.strict_best_step,
                early_stop_triggered=selection.early_stop_triggered,
            )
        )

    loss, accuracy = evaluate(model, validation_loader, criterion, device)
    if not torch.isfinite(torch.tensor([loss, accuracy])).all():
        return finish("non_finite", "sweep 0 の評価指標が有限ではありません。", 0)
    selection = initial_validation_selection(loss, 0)
    record_validation(0, float(loss), float(accuracy), True)

    for sweep in range(1, sweeps + 1):
        for pair_index, direction in visits:
            for local_step in range(local_steps):
                rank_before = _bond_rank(layer, pair_index)
                center_before = center
                try:
                    update = update_tt_linear_two_core_pair(
                        model,
                        candidate,
                        gradient_source,
                        layer_name=layer_name,
                        pair_index=pair_index,
                        direction=direction,
                        center=center,
                        rank_cap=cap,
                        tt_pair_learning_rate=tt_pair_learning_rate,
                        dense_parameter_learning_rate=dense_parameter_learning_rate,
                        diagnostic_images=diagnostic_images,
                        diagnostic_labels=diagnostic_labels,
                        device=device,
                        local_step=local_step,
                        criterion=criterion,
                        batch_size=batch_size,
                        discarded_weight_threshold=threshold,
                        min_bond_rank=minimum,
                    )
                except NonFiniteLocalUpdate as error:
                    return finish("non_finite", str(error), sweep - 1)
                except torch.linalg.LinAlgError as error:
                    return finish("svd_failure", f"{type(error).__name__}: {error}", sweep - 1)
                except ValueError as error:
                    if "特異値" in str(error):
                        return finish("invalid_singular_values", str(error), sweep - 1)
                    raise
                except torch.cuda.OutOfMemoryError as error:
                    return finish("out_of_memory", str(error), sweep - 1)
                center = int(update.center)
                decision = update.bond_rank_decision
                if decision is None:
                    raise RuntimeError("discarded weight を指定した更新なのに rank 選択結果がありません。")
                updates.append(
                    AdaptiveLocalUpdate(
                        sweep=sweep,
                        pair_index=pair_index,
                        direction=direction,
                        local_step=local_step,
                        rank_before=rank_before,
                        rank_after=int(update.retained_rank),
                        actual_ranks=tuple(int(rank) for rank in tt_matrix_ranks(list(layer.cores))),
                        center_before=center_before,
                        center_after=center,
                        relative_discarded_weight=float(decision.relative_discarded_weight),
                        discarded_squared_norm=float(update.discarded_squared_norm),
                        cumulative_updates=len(updates) + 1,
                        diagnostic_batch_loss=float(update.diagnostic_batch_loss),
                    )
                )
        loss, accuracy = evaluate(model, validation_loader, criterion, device)
        if not torch.isfinite(torch.tensor([loss, accuracy])).all():
            return finish("non_finite", f"sweep {sweep} の評価指標が有限ではありません。", sweep - 1)
        previous_best = selection.strict_best_step
        selection = update_validation_selection(
            selection,
            loss,
            len(updates),
            min_delta=min_delta,
            patience=patience,
        )
        became_best = (
            selection.strict_best_step == len(updates) and selection.strict_best_step != previous_best
        )
        if became_best:
            best_state = _state_copy(model)
            best_sweep = sweep
        record_validation(sweep, float(loss), float(accuracy), became_best)
        if selection.early_stop_triggered:
            return finish(
                "early_stopping",
                f"validation loss が min_delta={min_delta} 以上改善しないまま {patience} 回続いた。",
                sweep,
            )
    return finish("max_sweeps", f"最大往復数 {sweeps} まで実行した。", sweeps)
