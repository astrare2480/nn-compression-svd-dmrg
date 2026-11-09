"""ALS の掃引の記録を表の行にする関数と、最終値に近づいた掃引の探索。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nn_compression.metrics.als_convergence import first_sweep_within, study_sweep_row


def _update(n_stages: int, outcome: str = "target_met", last_residual: float = 1e-9):
    stages = tuple(SimpleNamespace(residual=last_residual if i == 2 else 1e-3) for i in range(n_stages))
    return SimpleNamespace(stages=stages, outcome=outcome)


def _record(**overrides):
    base = dict(
        sweep=7,
        q_lambda=1.5,
        q_unregularized=1.4,
        relative_decrease=1e-6,
        delta_50=None,
        consecutive_small_windows=1,
        stationarity_max=3e-5,
        stationarity=(1e-5, 3e-5),
        updates=(_update(1), _update(2, "local_target_missed"), _update(3, last_residual=2e-9)),
        n_warnings=4,
        n_stage2=9,
        n_stage3=2,
        hvp_calls=120,
        n_rejected=0,
        seconds=1.25,
        peak_memory_bytes=None,
        stop_reason=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_study_sweep_row_counts_stages_and_warnings():
    row = study_sweep_row(_record())
    assert row["s_max"] == 3e-5
    assert row["stationarity"] == "1e-05;3e-05"
    assert row["n_updates"] == 3
    assert row["n_updates_stage2"] == 2
    assert row["n_updates_stage3"] == 1
    assert row["stage3_residuals"] == "2e-09"
    assert row["n_warnings_sweep"] == 1
    assert row["n_warnings"] == 4 and row["n_stage2"] == 9 and row["n_stage3"] == 2
    assert row["stop_reason"] is None


def test_first_sweep_within():
    history = [SimpleNamespace(sweep=s, q_lambda=q) for s, q in [(1, 2.0), (2, 1.2), (3, 1.0005), (4, 1.0)]]
    assert first_sweep_within(history, 1e-3) == 3
    assert first_sweep_within(history, 0.0) == 4
    assert first_sweep_within(history, 1.0) == 1
    assert first_sweep_within([], 1e-3) is None
    with pytest.raises(ValueError):
        first_sweep_within(history, -1.0)
