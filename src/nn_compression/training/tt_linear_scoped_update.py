"""更新対象を明示した、全データ勾配の固定予算学習。

joint は結合した TT pair と通常 parameter を更新する。
tt_only は結合 pair だけを更新し、通常 parameter は勾配・更新候補・書き戻しから外す。
dense_only は通常 parameter だけを更新し、TT pair の結合、SVD、core 書き戻し、
中心移動、pair 巡回は行わない。

固定は学習率 0 では表さない。除外した側の学習率は ``None``。

GPU 常駐 batch は、呼び出すたびに先頭から slice を返す。1回の走査で
消費済みになる iterator は保持しない。sample 数や core 数は引数と
モデルから取り、特定のデータセットサイズは持たない。
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from ..compression.tt_linear_two_site import (
    assign_tt_linear_cores,
    module_forward_with_merged_tt_linear_pair,
)
from ..tt_mps.matrix_two_site import (
    merge_tt_matrix_cores,
    replace_tt_matrix_core_pair,
    split_merged_tt_matrix,
)
from .gpu_resident_batches import GpuResidentTensorBatches
from .loops import evaluate
from .tt_linear_full_batch_sweep import (
    NonFiniteLocalUpdate,
    _core_prefix,
    _group_l2_norm,
    _mean_criterion,
    _relative_update,
    _require_finite_number,
    _require_positive_int,
    _tensor_is_finite,
    _tt_linear_layer,
)


class TTUpdateScope:
    """``joint`` / ``tt_only`` / ``dense_only`` を区別する。"""

    JOINT = "joint"
    TT_ONLY = "tt_only"
    DENSE_ONLY = "dense_only"
    _VALUES = (JOINT, TT_ONLY, DENSE_ONLY)

    def __init__(self, value: str) -> None:
        if value not in self._VALUES:
            raise ValueError(
                f"更新 scope は {self._VALUES} のいずれかです。受け取った値: {value!r}"
            )
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TTUpdateScope):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"TTUpdateScope({self.value!r})"


@dataclass
class ScopedFullBatchBudgetResult:
    """固定更新予算の1条件の結果。best と final は別の state。"""

    scope: str
    stop_reason: str
    stop_detail: str
    completed_updates: int
    requested_updates: int
    history: list
    local_updates: list
    best_update_count: int
    best_validation_loss: float
    best_state: dict
    initial_state: dict
    center: int | None
    seconds_train: float
    seconds_eval: float
    svd_call_count: int
    dense_candidate_count: int
    core_writeback_count: int


def accumulate_scoped_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    batch_size: int,
    device: torch.device,
    pair_index: int | None = None,
    criterion: nn.Module | None = None,
) -> tuple:
    """全 sample の平均損失と、scope が更新する対象だけの勾配を返す。

    重みは ``(n_b / N) * L_b``。``L_b`` は batch 平均なので、さらに ``N`` では割らない。
    最後の batch が小さくても同じ定義。処理 sample 数は過不足なく ``N`` と一致する。

    戻り値は ``(grad_T, dense_grads, full_batch_loss)``。
    dense-only の ``grad_T`` は ``None``。tt-only の ``dense_grads`` は空。
    parameter は更新しない。
    """
    scope = _as_scope(scope)
    batch_size = _require_positive_int(batch_size, "batch_size")
    criterion = _mean_criterion(criterion)
    saved_flags = _prepare_gradient_targets(model, layer_name, scope)
    merged = None
    try:
        if scope.value == TTUpdateScope.DENSE_ONLY:
            if pair_index is not None:
                raise ValueError("dense_only は TT pair を結合しないため、pair_index は None です。")
        else:
            if pair_index is None:
                raise ValueError(f"{scope.value} には pair_index が必要です。")
            layer = _tt_linear_layer(model, layer_name)
            if pair_index < 0 or pair_index + 1 >= len(layer.cores):
                raise ValueError(f"pair_index={pair_index} は core 列の範囲外です。")
            merged = merge_tt_matrix_cores(
                layer.cores[pair_index],
                layer.cores[pair_index + 1],
            ).detach().clone().requires_grad_(True)

        for parameter in model.parameters():
            parameter.grad = None
        if merged is not None:
            merged.grad = None

        full_loss = None
        seen = 0
        n_samples = _source_size(source, batch_size)
        for images, labels in _iter_source_batches(source, batch_size, device):
            if images.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"images の batch 次元 {images.shape[0]} と labels {labels.shape[0]} が一致しません。"
                )
            n_b = int(labels.shape[0])
            if scope.value == TTUpdateScope.DENSE_ONLY:
                logits = model(images)
            else:
                logits = module_forward_with_merged_tt_linear_pair(
                    model,
                    images,
                    layer_name,
                    pair_index,
                    merged,
                )
            loss_b = criterion(logits, labels)
            if not _tensor_is_finite(loss_b) or not _tensor_is_finite(logits):
                raise NonFiniteLocalUpdate("batch 損失または logits が有限ではありません。")
            weighted = loss_b * (n_b / n_samples)
            weighted.backward()
            detached = weighted.detach()
            full_loss = detached if full_loss is None else full_loss + detached
            seen += n_b

        if seen != n_samples:
            raise ValueError(
                f"処理した sample 数 {seen} が全 sample 数 {n_samples} と一致しません。"
            )
        if full_loss is None or not _tensor_is_finite(full_loss):
            raise NonFiniteLocalUpdate("全データの平均損失が有限ではありません。")

        grad_t = None
        if merged is not None:
            if merged.grad is None:
                raise RuntimeError("結合テンソル T の勾配がありません。")
            if not _tensor_is_finite(merged.grad):
                raise NonFiniteLocalUpdate("結合テンソル T の勾配が有限ではありません。")
            grad_t = merged.grad.detach().clone()

        dense_grads: dict[str, torch.Tensor] = {}
        if scope.value != TTUpdateScope.TT_ONLY:
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
        return grad_t, dense_grads, float(full_loss.detach().cpu())
    finally:
        _restore_requires_grad(saved_flags)
        for parameter in model.parameters():
            parameter.grad = None


def train_scoped_full_batch_budget(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    batch_size: int,
    total_updates: int,
    evaluation_interval: int,
    rank_cap: int,
    tt_learning_rate: float | None,
    dense_learning_rate: float | None,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    local_steps_per_pair: int | None = None,
    max_sweeps: int | None = None,
    center: int = 0,
    criterion: nn.Module | None = None,
) -> ScopedFullBatchBudgetResult:
    """同じ更新回数の予算で、指定した scope だけを学習する。

    joint と tt_only は、左右1往復を評価間隔と一致させて繰り返す。
    dense_only の評価間隔は比較用の区切りであり、往復ではない。
    early stopping はしない。初期状態も best 候補に含める。
    loss が同値なら、累積更新数が小さい方を best にする。

    ``NonFiniteLocalUpdate`` は ``non_finite``、``torch.cuda.OutOfMemoryError`` は
    ``out_of_memory``。それ以外の例外はそのまま伝播し、停止理由へ変換しない。
    """
    scope = _as_scope(scope)
    total_updates = _require_positive_int(total_updates, "total_updates")
    evaluation_interval = _require_positive_int(evaluation_interval, "evaluation_interval")
    rank_cap = _require_positive_int(rank_cap, "rank_cap")
    if total_updates % evaluation_interval != 0:
        raise ValueError("total_updates は evaluation_interval の倍数である必要があります。")
    tt_rate, dense_rate = _rates_for_scope(scope, tt_learning_rate, dense_learning_rate)
    criterion = _mean_criterion(criterion)
    model.to(device)
    diagnostic_images = diagnostic_images.to(device)
    diagnostic_labels = diagnostic_labels.to(device)

    if scope.value == TTUpdateScope.DENSE_ONLY:
        if local_steps_per_pair is not None or max_sweeps is not None:
            raise ValueError(
                "dense_only は DMRG の往復ではありません。max_sweeps と local_steps_per_pair は None です。"
            )
        if center != 0:
            raise ValueError("dense_only は正準中心を移動しません。center は 0 のままです。")
        visits = None
    else:
        if local_steps_per_pair is None or max_sweeps is None:
            raise ValueError("joint と tt_only には local_steps_per_pair と max_sweeps が必要です。")
        local_steps = _require_positive_int(local_steps_per_pair, "local_steps_per_pair")
        sweeps = _require_positive_int(max_sweeps, "max_sweeps")
        n_cores = len(_tt_linear_layer(model, layer_name).cores)
        updates_per_sweep = 2 * (n_cores - 1) * local_steps
        if updates_per_sweep != evaluation_interval:
            raise ValueError(
                "joint と tt_only の評価間隔は、左右1往復の更新数と一致する必要があります。"
                f" 往復={updates_per_sweep}, evaluation_interval={evaluation_interval}"
            )
        if sweeps * updates_per_sweep != total_updates:
            raise ValueError(
                "max_sweeps と1往復の更新数の積が total_updates と一致しません。"
            )
        visits = _round_trip_visits(n_cores)
        local_steps_per_pair = local_steps

    initial_state = _clone_state_dict(model)
    initial_dense = _dense_cpu_state(model, layer_name)
    initial_cores = _core_cpu_state(model, layer_name)
    candidate = copy.deepcopy(model).to(device)
    history: list[dict] = []
    local_updates: list[dict] = []
    best_loss = math.inf
    best_updates = 10**18
    best_state = initial_state
    seconds_train = 0.0
    seconds_eval = 0.0
    svd_call_count = 0
    dense_candidate_count = 0
    core_writeback_count = 0
    completed = 0
    current_center: int | None = None if scope.value == TTUpdateScope.DENSE_ONLY else int(center)

    def evaluate_point(update_count: int) -> None:
        nonlocal best_loss, best_updates, best_state, seconds_eval
        _synchronize(device)
        started = _timer()
        train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
        val_loss, val_acc = evaluate(model, validation_loader, criterion, device)
        _synchronize(device)
        seconds_eval += _timer() - started
        if not all(math.isfinite(value) for value in (train_loss, train_acc, val_loss, val_acc)):
            raise NonFiniteLocalUpdate("評価指標が有限ではありません。")
        fixed_ok = _fixed_state_holds(model, layer_name, scope, initial_dense, initial_cores)
        if not fixed_ok:
            raise RuntimeError("固定対象の state が初期状態と一致しません。")
        if best_updates == 10**18:
            chosen = True
        else:
            chosen = _prefers_validation_checkpoint(
                val_loss,
                update_count,
                best_loss,
                best_updates,
            )
        if chosen:
            for row in history:
                row["is_best"] = False
            best_loss = float(val_loss)
            best_updates = int(update_count)
            best_state = _clone_state_dict(model)
        history.append(
            {
                "evaluation_block": update_count // evaluation_interval,
                "cumulative_updates": update_count,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "early_stopping_validation_loss": float(val_loss),
                "early_stopping_validation_accuracy": float(val_acc),
                "cumulative_train_seconds": seconds_train,
                "cumulative_eval_seconds": seconds_eval,
                "fixed_state_ok": True,
                "is_best": chosen,
            }
        )

    try:
        evaluate_point(0)
        while completed < total_updates:
            if scope.value == TTUpdateScope.DENSE_ONLY:
                _synchronize(device)
                started = _timer()
                step = _dense_step(
                    model,
                    candidate,
                    source,
                    scope=scope,
                    layer_name=layer_name,
                    batch_size=batch_size,
                    learning_rate=dense_rate,
                    device=device,
                    diagnostic_images=diagnostic_images,
                    diagnostic_labels=diagnostic_labels,
                    criterion=criterion,
                )
                _synchronize(device)
                seconds = _timer() - started
            else:
                visit_index = completed // local_steps_per_pair % len(visits)
                pair_index, direction = visits[visit_index]
                local_step = completed % local_steps_per_pair
                _synchronize(device)
                started = _timer()
                step = _pair_step(
                    model,
                    candidate,
                    source,
                    scope=scope,
                    layer_name=layer_name,
                    pair_index=pair_index,
                    direction=direction,
                    local_step=local_step,
                    center=current_center,
                    rank_cap=rank_cap,
                    tt_learning_rate=tt_rate,
                    dense_learning_rate=dense_rate,
                    batch_size=batch_size,
                    device=device,
                    diagnostic_images=diagnostic_images,
                    diagnostic_labels=diagnostic_labels,
                    criterion=criterion,
                )
                _synchronize(device)
                seconds = _timer() - started
                current_center = step["center_after"]
            seconds_train += seconds
            completed += 1
            svd_call_count += int(step["svd_called"])
            dense_candidate_count += int(step["dense_candidates_built"])
            core_writeback_count += int(step["cores_written"])
            local_updates.append(
                _log_row(
                    step,
                    scope=scope.value,
                    cumulative_updates=completed,
                    evaluation_block=(completed - 1) // evaluation_interval + 1,
                    seconds=seconds,
                )
            )
            if completed % evaluation_interval == 0:
                evaluate_point(completed)
    except NonFiniteLocalUpdate as exc:
        return _budget_result(
            scope=scope.value,
            stop_reason="non_finite",
            stop_detail=str(exc),
            completed=completed,
            requested=total_updates,
            history=history,
            local_updates=local_updates,
            best_updates=best_updates,
            best_loss=best_loss,
            best_state=best_state,
            initial_state=initial_state,
            center=current_center,
            seconds_train=seconds_train,
            seconds_eval=seconds_eval,
            svd_call_count=svd_call_count,
            dense_candidate_count=dense_candidate_count,
            core_writeback_count=core_writeback_count,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _budget_result(
            scope=scope.value,
            stop_reason="out_of_memory",
            stop_detail=str(exc),
            completed=completed,
            requested=total_updates,
            history=history,
            local_updates=local_updates,
            best_updates=best_updates,
            best_loss=best_loss,
            best_state=best_state,
            initial_state=initial_state,
            center=current_center,
            seconds_train=seconds_train,
            seconds_eval=seconds_eval,
            svd_call_count=svd_call_count,
            dense_candidate_count=dense_candidate_count,
            core_writeback_count=core_writeback_count,
        )

    return _budget_result(
        scope=scope.value,
        stop_reason="max_updates",
        stop_detail="",
        completed=completed,
        requested=total_updates,
        history=history,
        local_updates=local_updates,
        best_updates=best_updates,
        best_loss=best_loss,
        best_state=best_state,
        initial_state=initial_state,
        center=current_center,
        seconds_train=seconds_train,
        seconds_eval=seconds_eval,
        svd_call_count=svd_call_count,
        dense_candidate_count=dense_candidate_count,
        core_writeback_count=core_writeback_count,
    )


def save_scoped_full_batch_run(
    directory: Path | str,
    model: nn.Module,
    result: ScopedFullBatchBudgetResult,
    config: Mapping,
) -> None:
    """初期・最終・best と履歴、局所ログ、状態、設定を条件ディレクトリへ書く。

    既存ファイルは同じディレクトリの一時ファイルから置換する。
    ディレクトリ自体は削除しない。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layer_name = config["layer_name"]
    common = {
        "layer_name": layer_name,
        "out_modes": list(config["out_modes"]),
        "in_modes": list(config["in_modes"]),
    }
    _atomic_torch_save({**common, "state_dict": result.initial_state}, directory / "initial_model.pt")
    _atomic_torch_save({**common, "state_dict": model.state_dict()}, directory / "final_model.pt")
    _atomic_torch_save({**common, "state_dict": result.best_state}, directory / "best_model.pt")
    _atomic_csv(
        directory / "history.csv",
        [
            "evaluation_block",
            "cumulative_updates",
            "train_loss",
            "train_accuracy",
            "early_stopping_validation_loss",
            "early_stopping_validation_accuracy",
            "cumulative_train_seconds",
            "cumulative_eval_seconds",
            "fixed_state_ok",
            "is_best",
        ],
        result.history,
    )
    _atomic_csv(
        directory / "local_updates.csv",
        [
            "scope",
            "cumulative_updates",
            "evaluation_block",
            "status",
            "full_batch_loss",
            "update_seconds",
            "finite",
            "pair_index",
            "direction",
            "local_step",
            "center_before",
            "center_after",
            "tt_gradient_norm",
            "tt_parameter_norm",
            "tt_relative_update",
            "retained_rank",
            "discarded_squared_norm",
            "dense_gradient_norm",
            "dense_parameter_norm",
            "dense_relative_update",
        ],
        result.local_updates,
    )
    state = {
        "config": dict(config),
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "completed_updates": result.completed_updates,
        "requested_updates": result.requested_updates,
        "best_update_count": result.best_update_count,
        "best_validation_loss": result.best_validation_loss,
        "center": result.center,
        "seconds_train": result.seconds_train,
        "seconds_eval": result.seconds_eval,
        "svd_call_count": result.svd_call_count,
        "dense_candidate_count": result.dense_candidate_count,
        "core_writeback_count": result.core_writeback_count,
    }
    _atomic_text(
        directory / "run_state.json",
        json.dumps(_json_ready(state), ensure_ascii=False, indent=2),
    )
    _atomic_text(
        directory / "config.json",
        json.dumps(_json_ready(dict(config)), ensure_ascii=False, indent=2),
    )


def assert_scope_budget_completed(
    stop_reason: str,
    completed_updates: int,
    total_updates: int,
) -> None:
    """正常完了のときだけ返す。集計行を追加する前に呼ぶ。

    正常完了は ``stop_reason == "max_updates"`` かつ更新数が予算と一致すること。
    それ以外は診断用の成果物を残したあと、この例外で集計への追加を止める。
    """
    if not isinstance(stop_reason, str):
        raise TypeError(f"stop_reason は str である必要があります: {stop_reason!r}")
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError(f"completed_updates は int である必要があります: {completed_updates!r}")
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates < 1:
        raise TypeError(f"total_updates は 1 以上の int である必要があります: {total_updates!r}")
    if stop_reason == "max_updates" and completed_updates == total_updates:
        return
    raise RuntimeError(
        "正常完了していないため、condition_summary.csv へは記録しません。"
        f" stop_reason={stop_reason!r}"
        f" completed_updates={completed_updates} total_updates={total_updates}"
    )


def reject_incomplete_scope_rows(
    rows: Sequence[Mapping],
    scopes: Sequence[str],
    total_updates: int,
) -> None:
    """集計に残っている異常停止や更新数不足を、完了済みとしてスキップしない。

    行が無い条件は対象外。再開関数を呼ぶ前に使い、例外時はファイルを削除しない。
    """
    allowed = set(scopes)
    for row in rows:
        condition_id = row.get("condition_id")
        if condition_id not in allowed:
            continue
        completed = row.get("completed_updates")
        if isinstance(completed, str):
            completed = int(float(completed))
        assert_scope_budget_completed(str(row.get("stop_reason")), int(completed), total_updates)


def incomplete_scope_names(
    observations: Mapping,
    scopes: Sequence[str],
    total_updates: int,
) -> list[str]:
    """比較図を書いてよいときだけ空リストを返す。

    scope の観測が ``None`` なら、その条件は未完了として名前を返す。
    集計行があるのに設定不一致、history 欠落、正常完了でない、最終更新数が
    予算と違う場合は SKIP せず ``ValueError``。
    """
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates < 1:
        raise TypeError(f"total_updates は 1 以上の int である必要があります: {total_updates!r}")
    missing: list[str] = []
    for scope in scopes:
        item = observations.get(scope)
        if item is None:
            missing.append(str(scope))
            continue
        if item.get("settings_match") is not True:
            raise ValueError(f"{scope} の記録された設定が現在の設定と一致しません。")
        if item.get("history_exists") is not True:
            raise ValueError(f"{scope} は集計行がありますが history.csv がありません。")
        completed = item.get("completed_updates")
        final_updates = item.get("final_updates")
        if item.get("stop_reason") != "max_updates" or int(completed) != total_updates:
            raise ValueError(
                f"{scope} は正常完了の記録ではありません。"
                f" stop_reason={item.get('stop_reason')!r} completed_updates={completed!r}"
            )
        if int(final_updates) != total_updates:
            raise ValueError(
                f"{scope} の最終更新数が {total_updates} ではありません: {final_updates!r}"
            )
    return missing


def require_recorded_smoke_gate(
    report: Mapping,
    expected: Mapping,
    fields: Sequence[str],
) -> None:
    """保存済み smoke が期待と一致しないとき、主実験を始めずに例外にする。

    ファイルは削除も上書きもしない。呼び出し側は、この関数が返った後だけ主実験を始める。
    """
    missing = [field for field in fields if field not in report]
    if missing:
        raise RuntimeError(
            "smoke が未完了です。主実験は開始しません。不足: " + ", ".join(missing)
        )
    mismatches = []
    for field in fields:
        if report[field] != expected[field]:
            mismatches.append(
                f"{field}: recorded={report[field]!r} expected={expected[field]!r}"
            )
    if mismatches:
        raise RuntimeError(
            "smoke の記録が現在の設定と一致しないため、主実験は開始しません。\n"
            + "\n".join(mismatches)
        )


def _as_scope(scope: TTUpdateScope | str) -> TTUpdateScope:
    if isinstance(scope, TTUpdateScope):
        return scope
    if isinstance(scope, str):
        return TTUpdateScope(scope)
    raise TypeError(f"scope は TTUpdateScope または文字列です: {scope!r}")


def _rates_for_scope(
    scope: TTUpdateScope,
    tt_learning_rate: float | None,
    dense_learning_rate: float | None,
) -> tuple[float | None, float | None]:
    if scope.value == TTUpdateScope.JOINT:
        if tt_learning_rate is None or dense_learning_rate is None:
            raise ValueError("joint は TT 側と通常 parameter 側の両方の学習率が必要です。")
        return (
            _require_finite_number(tt_learning_rate, "tt_learning_rate"),
            _require_finite_number(dense_learning_rate, "dense_learning_rate"),
        )
    if scope.value == TTUpdateScope.TT_ONLY:
        if dense_learning_rate is not None:
            raise ValueError(
                "tt_only の通常 parameter は更新から除外します。学習率 0 ではなく None を渡してください。"
            )
        if tt_learning_rate is None:
            raise ValueError("tt_only には TT 側の学習率が必要です。")
        return _require_finite_number(tt_learning_rate, "tt_learning_rate"), None
    if dense_learning_rate is None:
        raise ValueError("dense_only には通常 parameter 側の学習率が必要です。")
    if tt_learning_rate is not None:
        raise ValueError(
            "dense_only の TT 側は更新から除外します。学習率 0 ではなく None を渡してください。"
        )
    return None, _require_finite_number(dense_learning_rate, "dense_learning_rate")


def _source_size(source: DataLoader | GpuResidentTensorBatches, batch_size: int) -> int:
    if isinstance(source, GpuResidentTensorBatches):
        return source.n_samples
    if isinstance(source, DataLoader):
        if source.drop_last:
            raise ValueError("勾配集計の loader は drop_last=False である必要があります。")
        if not isinstance(source.sampler, SequentialSampler):
            raise ValueError("勾配集計の loader は shuffle=False（SequentialSampler）である必要があります。")
        if source.batch_size != batch_size:
            raise ValueError("loader の batch_size と引数 batch_size が一致しません。")
        dataset_size = len(source.dataset)
        if dataset_size < 1:
            raise ValueError("空の dataset では全データ勾配を集計できません。")
        return int(dataset_size)
    raise TypeError("source は DataLoader または GpuResidentTensorBatches です。")


def _resolve_torch_device(device: torch.device) -> torch.device:
    """``cuda``（index なし）を現在の CUDA device に展開してから比較できるようにする。"""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _iter_source_batches(
    source: DataLoader | GpuResidentTensorBatches,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(source, GpuResidentTensorBatches):
        if _resolve_torch_device(source.device) != _resolve_torch_device(device):
            raise ValueError(
                "GPU 常駐 batch は事前配置した device と一致する必要があります。batch ごとの転送はしません。"
                f" source={source.device}, requested={device}"
            )
        yield from source.batches(batch_size)
        return
    for images, labels in source:
        yield images.to(device), labels.to(device)


def _prepare_gradient_targets(
    model: nn.Module,
    layer_name: str,
    scope: TTUpdateScope,
) -> list[tuple[nn.Parameter, bool]]:
    prefix = _core_prefix(layer_name)
    saved: list[tuple[nn.Parameter, bool]] = []
    for name, parameter in model.named_parameters():
        saved.append((parameter, bool(parameter.requires_grad)))
        if name.startswith(prefix):
            parameter.requires_grad_(False)
        elif scope.value == TTUpdateScope.TT_ONLY:
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
    return saved


def _restore_requires_grad(saved: Sequence[tuple[nn.Parameter, bool]]) -> None:
    for parameter, requires_grad in saved:
        parameter.requires_grad_(requires_grad)


def _dense_named_parameters(model: nn.Module, layer_name: str) -> list[tuple[str, nn.Parameter]]:
    prefix = _core_prefix(layer_name)
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith(prefix)
    ]


def _dense_update_candidates(
    named: Sequence[tuple[str, nn.Parameter]],
    grads: Mapping[str, torch.Tensor],
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    candidates = {}
    for name, parameter in named:
        candidates[name] = (parameter.detach() - learning_rate * grads[name]).detach()
    return candidates


def _pair_step(
    model: nn.Module,
    candidate: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    pair_index: int,
    direction: str,
    local_step: int,
    center: int,
    rank_cap: int,
    tt_learning_rate: float,
    dense_learning_rate: float | None,
    batch_size: int,
    device: torch.device,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    criterion: nn.Module,
) -> dict:
    if direction not in {"right", "left"}:
        raise ValueError(f"direction は right または left です: {direction!r}")
    if center not in {pair_index, pair_index + 1}:
        raise ValueError(
            f"center={center} は pair {pair_index} の隣接 site ではありません。"
        )
    outside_before = _outside_core_state(model, layer_name, pair_index)
    dense_before = _dense_cpu_state(model, layer_name)
    grad_t, dense_grads, full_loss = accumulate_scoped_full_batch_gradients(
        model,
        source,
        scope=scope,
        layer_name=layer_name,
        batch_size=batch_size,
        device=device,
        pair_index=pair_index,
        criterion=criterion,
    )
    if grad_t is None:
        raise RuntimeError("TT を更新する scope で結合テンソルの勾配がありません。")
    if scope.value == TTUpdateScope.TT_ONLY and dense_grads:
        raise RuntimeError("tt_only は通常 parameter の勾配を保持しません。")
    merged_now = _current_merged(model, layer_name, pair_index)
    tt_parameter_norm = _group_l2_norm([merged_now])
    tt_gradient_norm = _group_l2_norm([grad_t])
    tt_relative = _relative_update(tt_learning_rate, tt_gradient_norm, tt_parameter_norm)
    t_candidate = (merged_now - tt_learning_rate * grad_t).detach()

    dense_candidates = None
    dense_parameter_norm = None
    dense_gradient_norm = None
    dense_relative = None
    if scope.value == TTUpdateScope.JOINT:
        named = _dense_named_parameters(model, layer_name)
        dense_parameter_norm = _group_l2_norm([parameter.detach() for _, parameter in named])
        dense_gradient_norm = _group_l2_norm([dense_grads[name] for name, _ in named])
        dense_relative = _relative_update(
            dense_learning_rate,
            dense_gradient_norm,
            dense_parameter_norm,
        )
        dense_candidates = _dense_update_candidates(named, dense_grads, dense_learning_rate)

    split = split_merged_tt_matrix(
        t_candidate,
        max_bond=rank_cap,
        direction=direction,
    )
    _install_pair_on_candidate(
        model,
        candidate,
        layer_name,
        pair_index,
        split.left_core.detach(),
        split.right_core.detach(),
        dense_candidates,
    )
    diagnostic_loss = _diagnostic_loss(
        candidate,
        diagnostic_images,
        diagnostic_labels,
        criterion,
    )
    _copy_updated_pair_to_model(
        model,
        candidate,
        layer_name,
        pair_index,
        dense_candidates is not None,
    )
    outside_after = _outside_core_state(model, layer_name, pair_index)
    if not _states_equal(outside_before, outside_after):
        raise RuntimeError("対象 pair 以外の TT core が局所更新中に変化しました。")
    if scope.value == TTUpdateScope.TT_ONLY:
        if not _states_equal(dense_before, _dense_cpu_state(model, layer_name)):
            raise RuntimeError("tt_only で通常 parameter が変化しました。")
    center_after = pair_index + 1 if direction == "right" else pair_index
    return {
        "status": "ok",
        "full_batch_loss": full_loss,
        "finite": True,
        "pair_index": pair_index,
        "direction": direction,
        "local_step": local_step,
        "center_before": center,
        "center_after": center_after,
        "tt_gradient_norm": tt_gradient_norm,
        "tt_parameter_norm": tt_parameter_norm,
        "tt_relative_update": tt_relative,
        "dense_gradient_norm": dense_gradient_norm,
        "dense_parameter_norm": dense_parameter_norm,
        "dense_relative_update": dense_relative,
        "retained_rank": int(split.retained_rank),
        "discarded_squared_norm": float(split.discarded_squared_norm),
        "diagnostic_loss": diagnostic_loss,
        "svd_called": 1,
        "dense_candidates_built": int(dense_candidates is not None),
        "cores_written": 1,
    }


def _dense_step(
    model: nn.Module,
    candidate: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    criterion: nn.Module,
) -> dict:
    cores_before = _core_cpu_state(model, layer_name)
    grad_t, dense_grads, full_loss = accumulate_scoped_full_batch_gradients(
        model,
        source,
        scope=scope,
        layer_name=layer_name,
        batch_size=batch_size,
        device=device,
        pair_index=None,
        criterion=criterion,
    )
    if grad_t is not None:
        raise RuntimeError("dense_only は結合テンソルの勾配を持ちません。")
    named = _dense_named_parameters(model, layer_name)
    dense_parameter_norm = _group_l2_norm([parameter.detach() for _, parameter in named])
    dense_gradient_norm = _group_l2_norm([dense_grads[name] for name, _ in named])
    dense_relative = _relative_update(learning_rate, dense_gradient_norm, dense_parameter_norm)
    dense_candidates = _dense_update_candidates(named, dense_grads, learning_rate)
    _copy_dense_values(candidate, dense_candidates)
    _diagnostic_loss(candidate, diagnostic_images, diagnostic_labels, criterion)
    _copy_dense_values(model, dense_candidates)
    if not _states_equal(cores_before, _core_cpu_state(model, layer_name)):
        raise RuntimeError("dense_only で TT core が変化しました。")
    return {
        "status": "ok",
        "full_batch_loss": full_loss,
        "finite": True,
        "pair_index": None,
        "direction": None,
        "local_step": None,
        "center_before": None,
        "center_after": None,
        "tt_gradient_norm": None,
        "tt_parameter_norm": None,
        "tt_relative_update": None,
        "dense_gradient_norm": dense_gradient_norm,
        "dense_parameter_norm": dense_parameter_norm,
        "dense_relative_update": dense_relative,
        "retained_rank": None,
        "discarded_squared_norm": None,
        "diagnostic_loss": None,
        "svd_called": 0,
        "dense_candidates_built": 1,
        "cores_written": 0,
    }


def _current_merged(model: nn.Module, layer_name: str, pair_index: int) -> torch.Tensor:
    layer = _tt_linear_layer(model, layer_name)
    return merge_tt_matrix_cores(layer.cores[pair_index], layer.cores[pair_index + 1]).detach()


def _install_pair_on_candidate(
    model: nn.Module,
    candidate: nn.Module,
    layer_name: str,
    pair_index: int,
    left: torch.Tensor,
    right: torch.Tensor,
    dense_candidates: Mapping[str, torch.Tensor] | None,
) -> None:
    layer = _tt_linear_layer(model, layer_name)
    cores = [core.detach().clone() for core in layer.cores]
    updated = replace_tt_matrix_core_pair(cores, pair_index, left, right)
    assign_tt_linear_cores(
        _tt_linear_layer(candidate, layer_name),
        updated,
        requires_grad=False,
    )
    if dense_candidates is None:
        _copy_dense_from_module(candidate, model, layer_name)
    else:
        _copy_dense_values(candidate, dense_candidates)


def _copy_updated_pair_to_model(
    model: nn.Module,
    candidate: nn.Module,
    layer_name: str,
    pair_index: int,
    copy_dense: bool,
) -> None:
    candidate_layer = _tt_linear_layer(candidate, layer_name)
    model_layer = _tt_linear_layer(model, layer_name)
    cores = [core.detach().clone() for core in model_layer.cores]
    updated = replace_tt_matrix_core_pair(
        cores,
        pair_index,
        candidate_layer.cores[pair_index].detach(),
        candidate_layer.cores[pair_index + 1].detach(),
    )
    assign_tt_linear_cores(model_layer, updated, requires_grad=False)
    if copy_dense:
        _copy_dense_from_module(model, candidate, layer_name)


def _copy_dense_from_module(destination: nn.Module, source: nn.Module, layer_name: str) -> None:
    values = {
        name: parameter.detach()
        for name, parameter in _dense_named_parameters(source, layer_name)
    }
    _copy_dense_values(destination, values)


def _copy_dense_values(module: nn.Module, values: Mapping[str, torch.Tensor]) -> None:
    named = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            named[name].copy_(value.to(named[name].device))


def _diagnostic_loss(
    module: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> float:
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            logits = module(images)
            loss = criterion(logits, labels)
        if not _tensor_is_finite(loss) or not _tensor_is_finite(logits):
            raise NonFiniteLocalUpdate("診断 batch の損失または logits が有限ではありません。")
        return float(loss.detach().cpu())
    finally:
        module.train(was_training)


def _round_trip_visits(n_cores: int) -> list[tuple[int, str]]:
    if n_cores < 2:
        raise ValueError("隣接 pair を作るには core が2つ以上必要です。")
    right = [(index, "right") for index in range(n_cores - 1)]
    left = [(index, "left") for index in range(n_cores - 2, -1, -1)]
    return right + left


def _log_row(step: Mapping, *, scope: str, cumulative_updates: int, evaluation_block: int, seconds: float) -> dict:
    return {
        "scope": scope,
        "cumulative_updates": cumulative_updates,
        "evaluation_block": evaluation_block,
        "status": step["status"],
        "full_batch_loss": step["full_batch_loss"],
        "update_seconds": seconds,
        "finite": step["finite"],
        "pair_index": step["pair_index"],
        "direction": step["direction"],
        "local_step": step["local_step"],
        "center_before": step["center_before"],
        "center_after": step["center_after"],
        "tt_gradient_norm": step["tt_gradient_norm"],
        "tt_parameter_norm": step["tt_parameter_norm"],
        "tt_relative_update": step["tt_relative_update"],
        "retained_rank": step["retained_rank"],
        "discarded_squared_norm": step["discarded_squared_norm"],
        "dense_gradient_norm": step["dense_gradient_norm"],
        "dense_parameter_norm": step["dense_parameter_norm"],
        "dense_relative_update": step["dense_relative_update"],
    }


def _prefers_validation_checkpoint(
    loss: float,
    updates: int,
    best_loss: float,
    best_updates: int,
) -> bool:
    if not math.isfinite(loss) or not math.isfinite(best_loss):
        raise ValueError("best 選択の loss は有限である必要があります。")
    if loss < best_loss:
        return True
    if loss == best_loss and updates < best_updates:
        return True
    return False


def _fixed_state_holds(
    model: nn.Module,
    layer_name: str,
    scope: TTUpdateScope,
    initial_dense: dict[str, torch.Tensor],
    initial_cores: dict[str, torch.Tensor],
) -> bool:
    if scope.value == TTUpdateScope.TT_ONLY:
        return _states_equal(initial_dense, _dense_cpu_state(model, layer_name))
    if scope.value == TTUpdateScope.DENSE_ONLY:
        return _states_equal(initial_cores, _core_cpu_state(model, layer_name))
    return True


def _dense_cpu_state(model: nn.Module, layer_name: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _dense_named_parameters(model, layer_name)
    }


def _core_cpu_state(model: nn.Module, layer_name: str) -> dict[str, torch.Tensor]:
    prefix = _core_prefix(layer_name)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def _outside_core_state(model: nn.Module, layer_name: str, pair_index: int) -> dict[str, torch.Tensor]:
    layer = _tt_linear_layer(model, layer_name)
    kept = {}
    for index, core in enumerate(layer.cores):
        if index in {pair_index, pair_index + 1}:
            continue
        kept[str(index)] = core.detach().cpu().clone()
    return kept


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _budget_result(**kwargs) -> ScopedFullBatchBudgetResult:
    best_updates = kwargs["best_updates"]
    if best_updates == 10**18:
        raise RuntimeError("初期評価より前に best を確定できません。")
    return ScopedFullBatchBudgetResult(
        scope=kwargs["scope"],
        stop_reason=kwargs["stop_reason"],
        stop_detail=kwargs["stop_detail"],
        completed_updates=kwargs["completed"],
        requested_updates=kwargs["requested"],
        history=kwargs["history"],
        local_updates=kwargs["local_updates"],
        best_update_count=best_updates,
        best_validation_loss=kwargs["best_loss"],
        best_state=kwargs["best_state"],
        initial_state=kwargs["initial_state"],
        center=kwargs["center"],
        seconds_train=kwargs["seconds_train"],
        seconds_eval=kwargs["seconds_eval"],
        svd_call_count=kwargs["svd_call_count"],
        dense_candidate_count=kwargs["dense_candidate_count"],
        core_writeback_count=kwargs["core_writeback_count"],
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timer() -> float:
    return __import__("time").perf_counter()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    lines = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    lines.append(buffer.getvalue())
    _atomic_text(path, "".join(lines))


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
