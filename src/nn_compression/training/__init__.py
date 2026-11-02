"""学習・評価処理。"""

from .fit import fit_with_early_stopping, non_shuffling_loader
from .loops import evaluate, train_one_epoch
from .parameter_scope import (
    parameter_change_summary,
    set_trainable_parameter_prefixes,
    trainable_named_parameters,
)
from .tt_linear_full_batch_sweep import (
    NonFiniteLocalUpdate,
    TTLinearFullBatchSweepResult,
    TTLinearLocalUpdate,
    accumulate_tt_linear_pair_full_batch_gradients,
    load_condition_summary,
    load_tt_linear_sweep_checkpoint,
    measure_tt_pair_update_scales,
    run_resumable_condition_grid,
    save_tt_linear_full_batch_sweep_run,
    sweep_tt_linear_pairs_left_to_right,
    sweep_tt_linear_pairs_right_to_left,
    train_tt_linear_full_batch_sweeps,
    update_tt_linear_two_core_pair,
)
from .tt_linear_scoped_update import (
    GpuResidentTensorBatches,
    ScopedFullBatchBudgetResult,
    TTUpdateScope,
    accumulate_scoped_full_batch_gradients,
    assert_scope_budget_completed,
    incomplete_scope_names,
    reject_incomplete_scope_rows,
    require_recorded_smoke_gate,
    save_scoped_full_batch_run,
    train_scoped_full_batch_budget,
)

__all__ = [
    "set_trainable_parameter_prefixes",
    "trainable_named_parameters",
    "parameter_change_summary",
    "train_one_epoch",
    "evaluate",
    "fit_with_early_stopping",
    "non_shuffling_loader",
    "NonFiniteLocalUpdate",
    "TTLinearLocalUpdate",
    "TTLinearFullBatchSweepResult",
    "accumulate_tt_linear_pair_full_batch_gradients",
    "update_tt_linear_two_core_pair",
    "sweep_tt_linear_pairs_left_to_right",
    "sweep_tt_linear_pairs_right_to_left",
    "train_tt_linear_full_batch_sweeps",
    "save_tt_linear_full_batch_sweep_run",
    "load_condition_summary",
    "load_tt_linear_sweep_checkpoint",
    "measure_tt_pair_update_scales",
    "run_resumable_condition_grid",
    "TTUpdateScope",
    "GpuResidentTensorBatches",
    "ScopedFullBatchBudgetResult",
    "accumulate_scoped_full_batch_gradients",
    "train_scoped_full_batch_budget",
    "save_scoped_full_batch_run",
    "require_recorded_smoke_gate",
    "assert_scope_budget_completed",
    "reject_incomplete_scope_rows",
    "incomplete_scope_names",
]
