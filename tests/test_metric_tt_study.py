"""研究用 ALS の停止、3 段の解き直し、保存と再開。"""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import pytest
import torch

from nn_compression.compression.metric_low_rank import gram_matrix, regularize_gram
from nn_compression.tt_mps.matrix import dense_to_tt_matrix_cores
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt_solver import (
    MetricTTSolverConfig,
    StudyResolutionStopRule,
    decide_study_stop,
    run_metric_tt_als_study,
    study_resolution_delta,
)

OUT_MODES = (2, 3, 2)
IN_MODES = (2, 2, 3)
TIMING_FIELDS = {
    "seconds",
    "seconds_hvp",
    "seconds_preconditioner",
    "seconds_preconditioner_apply",
    "seconds_qr",
    "seconds_accept",
    "seconds_stationarity",
    "seconds_q_unregularized",
    "peak_memory_bytes",
}


def _problem(seed: int):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(math.prod(OUT_MODES), math.prod(IN_MODES), generator=generator, dtype=torch.float64)
    cores = dense_to_tt_matrix_cores(weight, OUT_MODES, IN_MODES, max_rank=3)
    activations = torch.randn(16, math.prod(IN_MODES), generator=generator, dtype=torch.float64)
    raw = gram_matrix(activations)
    return cores, weight, regularize_gram(raw, 1e-6), raw


def _config(**overrides) -> MetricTTSolverConfig:
    settings = {"device": "cpu", "precision": "float64", "max_cg_iterations": 80, "max_refinements": 4}
    settings.update(overrides)
    return MetricTTSolverConfig(**settings)


def _rule(**overrides) -> StudyResolutionStopRule:
    settings = {
        "d_ref": 1.0,
        "window_sweeps": 2,
        "resolution_tolerance": 1e-4,
        "consecutive_windows": 2,
        "max_sweeps": 4,
        "local_residual_tolerance": 1e-8,
        "escalation_refinements": (5, 8),
        "warn_residual_tolerance": 1e-8,
    }
    settings.update(overrides)
    return StudyResolutionStopRule(**settings)


def _without_timing(record) -> dict:
    body = {key: value for key, value in asdict(record).items() if key not in TIMING_FIELDS}
    updates = []
    for update in body["updates"]:
        copied = dict(update)
        stages = []
        for stage in update["stages"]:
            stage_copy = dict(stage)
            stage_copy.pop("seconds", None)
            stages.append(stage_copy)
        copied["stages"] = stages
        updates.append(copied)
    body["updates"] = updates
    return body


class _Interrupt(RuntimeError):
    pass


def test_study_stop_uses_two_windows_the_sweep_cap_or_an_unresolved_flag():
    rule = _rule(window_sweeps=1, max_sweeps=4, resolution_tolerance=1e-4, consecutive_windows=2)
    flat_tail = [100.0, 50.0, 50.0 - 1e-6, 50.0 - 2e-6]
    assert decide_study_stop(flat_tail[:2], rule)[0] is None
    assert decide_study_stop(flat_tail[:3], rule)[0] is None
    reason, delta, streak = decide_study_stop(flat_tail, rule)
    assert reason == "study_resolution_reached"
    assert streak == 2
    assert delta == pytest.approx(study_resolution_delta(flat_tail[2], flat_tail[3], rule.d_ref))

    still_moving = [10.0, 9.0, 8.0, 7.0, 6.0]
    assert decide_study_stop(still_moving, rule) == ("max_sweeps", pytest.approx(1.0), 0)
    assert decide_study_stop(still_moving[:2], rule, unresolved=True)[0] == "local_solver_unresolved"
    # 分解能が先に成り立つ掃引では、上限よりこちらを返す。
    assert decide_study_stop(flat_tail, replace(rule, max_sweeps=3))[0] == "study_resolution_reached"


def test_a_normal_local_solve_is_target_met_on_the_first_stage():
    cores, weight, gram, _raw = _problem(1)
    result = run_metric_tt_als_study(cores, weight, gram, OUT_MODES, IN_MODES, _config(), _rule(max_sweeps=1))
    assert result.stop_reason == "max_sweeps"
    assert result.n_warnings == 0
    assert result.n_stage2 == 0
    outcomes = [update.outcome for update in result.history[0].updates]
    assert outcomes == ["target_met"] * len(outcomes)
    assert all(len(update.stages) == 1 for update in result.history[0].updates)
    assert all(
        update.stages[0].cumulative_refinements == update.stages[0].refinements
        for update in result.history[0].updates
    )
    assert result.history[0].hvp_calls > 0


def test_escalation_reaches_the_warning_and_unresolved_classes(monkeypatch):
    import nn_compression.tt_mps.metric_tt_solver as solver

    cores, weight, gram, _raw = _problem(2)
    original = solver._solve_site

    def with_residual(residual: float):
        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            return (*output[:2], residual, *output[3:])

        return wrapped

    monkeypatch.setattr(solver, "_solve_site", with_residual(1e-7))
    warning = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        _config(max_refinements=1, max_cg_iterations=1),
        _rule(max_sweeps=1, escalation_refinements=(2, 3), warn_residual_tolerance=1e-6),
    )
    assert warning.stop_reason == "max_sweeps"
    assert warning.n_warnings == len(warning.history[0].updates)
    assert warning.n_stage3 == warning.n_warnings
    assert all(update.outcome == "local_target_missed" for update in warning.history[0].updates)
    assert all(len(update.stages) == 3 for update in warning.history[0].updates)
    assert warning.history[0].updates[0].confirmation_a is not None

    monkeypatch.setattr(solver, "_solve_site", with_residual(1.0))
    unresolved = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        _config(max_refinements=1, max_cg_iterations=1),
        _rule(max_sweeps=4, escalation_refinements=(2, 3), warn_residual_tolerance=1e-6),
    )
    assert unresolved.stop_reason == "local_solver_unresolved"
    assert unresolved.history == ()
    assert unresolved.failure is not None
    assert unresolved.failure.site == 0
    assert len(unresolved.failure.stages) == 3
    initial = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, 0)
    for got, want in zip(unresolved.cores, initial, strict=True):
        assert torch.equal(got, want)


def test_float64_nonpositive_curvature_is_unresolved_and_float32_is_only_counted(monkeypatch):
    import nn_compression.tt_mps.metric_tt_solver as solver

    cores, weight, gram, _raw = _problem(3)
    original = solver._solve_site

    def mark_curvature(*args, **kwargs):
        output = original(*args, **kwargs)
        kwargs["cg_statuses"].append("nonpositive_curvature")
        return (*output[:2], 0.0, *output[3:])

    monkeypatch.setattr(solver, "_solve_site", mark_curvature)
    float64_run = run_metric_tt_als_study(
        cores, weight, gram, OUT_MODES, IN_MODES, _config(precision="float64"), _rule(max_sweeps=2)
    )
    assert float64_run.stop_reason == "local_solver_unresolved"
    assert float64_run.failure is not None
    assert float64_run.failure.stages[0].nonpositive_curvature_count >= 1

    float32_run = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        _config(precision="float32_refine", max_refinements=1),
        _rule(max_sweeps=1, escalation_refinements=(2, 3)),
    )
    assert float32_run.stop_reason == "max_sweeps"
    assert all(update.outcome == "target_met" for update in float32_run.history[0].updates)
    assert all(update.stages[0].nonpositive_curvature_count >= 1 for update in float32_run.history[0].updates)


def test_second_stage_can_meet_the_target(monkeypatch):
    import nn_compression.tt_mps.metric_tt_solver as solver

    cores, weight, gram, _raw = _problem(4)
    original = solver._solve_site
    script = [1e-5, 0.0]

    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        residual = script.pop(0) if script else 0.0
        return (*output[:2], residual, *output[3:])

    monkeypatch.setattr(solver, "_solve_site", wrapped)
    result = run_metric_tt_als_study(
        cores, weight, gram, OUT_MODES, IN_MODES, _config(max_refinements=1), _rule(max_sweeps=1)
    )
    first = result.history[0].updates[0]
    assert first.outcome == "target_met"
    assert len(first.stages) == 2
    assert result.n_stage2 == 1
    assert result.n_stage3 == 0


def test_study_resume_reproduces_history_and_a_finished_checkpoint_returns(tmp_path):
    cores, weight, gram, raw = _problem(5)
    config = _config(precision="float32_refine", max_cg_iterations=30)
    rule = _rule(max_sweeps=4, window_sweeps=2, consecutive_windows=100)
    metadata = {"initialization": "M3"}
    whole = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        config,
        rule,
        report_gram=raw,
        checkpoint_path=tmp_path / "whole.pt",
        metadata=metadata,
        diagnostics=lambda _cores, sweep: {"sweep_seen": sweep},
    )
    assert [item["sweep"] for item in whole.diagnostics] == [2, 4]
    assert whole.stop_reason == "max_sweeps"

    path = tmp_path / "split.pt"

    def stop_at_three(record):
        if record.sweep == 3:
            raise _Interrupt

    with pytest.raises(_Interrupt):
        run_metric_tt_als_study(
            cores,
            weight,
            gram,
            OUT_MODES,
            IN_MODES,
            config,
            rule,
            report_gram=raw,
            checkpoint_path=path,
            metadata=metadata,
            progress=stop_at_three,
            diagnostics=lambda _cores, sweep: {"sweep_seen": sweep},
        )
    saved = torch.load(path, weights_only=True)
    assert saved["sweeps_completed"] == 2
    assert saved["stop_reason"] is None
    assert "delta_50" in saved

    resumed = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        config,
        rule,
        report_gram=raw,
        checkpoint_path=path,
        metadata=metadata,
        diagnostics=lambda _cores, sweep: {"sweep_seen": sweep},
    )
    assert resumed.resumed_from_sweep == 2
    assert [_without_timing(record) for record in resumed.history] == [
        _without_timing(record) for record in whole.history
    ]
    for got, want in zip(resumed.cores, whole.cores, strict=True):
        assert torch.equal(got, want)

    again = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        config,
        rule,
        report_gram=raw,
        checkpoint_path=path,
        metadata=metadata,
        progress=lambda record: (_ for _ in ()).throw(AssertionError("計算した")),
    )
    assert again.stop_reason == "max_sweeps"
    assert again.resumed_from_sweep == 4


def test_breakdown_is_recorded_and_does_not_stop_the_study(monkeypatch):
    import nn_compression.tt_mps.metric_tt_solver as solver

    cores, weight, gram, _raw = _problem(7)
    original = solver._solve_site

    def mark_breakdown(*args, **kwargs):
        output = original(*args, **kwargs)
        kwargs["cg_statuses"].append("breakdown")
        return (*output[:2], 0.0, *output[3:])

    monkeypatch.setattr(solver, "_solve_site", mark_breakdown)
    for precision in ("float64", "float32_refine"):
        result = run_metric_tt_als_study(
            cores,
            weight,
            gram,
            OUT_MODES,
            IN_MODES,
            _config(precision=precision, max_refinements=1, max_cg_iterations=20),
            _rule(max_sweeps=1, escalation_refinements=(2, 3)),
        )
        assert result.stop_reason == "max_sweeps"
        assert result.failure is None
        assert result.history[0].updates
        assert all(update.outcome == "target_met" for update in result.history[0].updates)
        assert all(
            "breakdown" in stage.cg_statuses
            for update in result.history[0].updates
            for stage in update.stages
        )


def test_checkpoint_without_timing_sync_or_idle_counts_still_resumes(tmp_path):
    cores, weight, gram, raw = _problem(8)
    config = _config(precision="float64", max_cg_iterations=20, max_refinements=1)
    rule = _rule(max_sweeps=1, window_sweeps=50, consecutive_windows=2, escalation_refinements=(2, 3))
    path = tmp_path / "old.pt"
    finished = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        config,
        rule,
        report_gram=raw,
        checkpoint_path=path,
        metadata={"initialization": "M3"},
    )
    assert finished.stop_reason == "max_sweeps"
    payload = torch.load(path, weights_only=True)
    payload["solver_config"].pop("timing_sync")
    for sweep in payload["history"]:
        for update in sweep["updates"]:
            for stage in update["stages"]:
                stage.pop("cg_idle_iterations")
    torch.save(payload, path)
    resumed = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        config,
        rule,
        report_gram=raw,
        checkpoint_path=path,
        metadata={"initialization": "M3"},
        progress=lambda _record: (_ for _ in ()).throw(AssertionError("計算した")),
    )
    assert resumed.stop_reason == "max_sweeps"
    assert resumed.resumed_from_sweep == finished.history[-1].sweep
    assert all(stage.cg_idle_iterations == 0 for update in resumed.history[0].updates for stage in update.stages)


def test_cumulative_escalation_caps_must_increase():
    with pytest.raises(ValueError):
        _rule(escalation_refinements=(4, 4))
    cores, weight, gram, _raw = _problem(9)
    with pytest.raises(ValueError):
        run_metric_tt_als_study(
            cores,
            weight,
            gram,
            OUT_MODES,
            IN_MODES,
            _config(max_refinements=5),
            _rule(escalation_refinements=(5, 8)),
        )


def test_later_stages_continue_from_the_previous_solution_and_stay_within_the_cap():
    cores, weight, gram, _raw = _problem(10)
    result = run_metric_tt_als_study(
        cores,
        weight,
        gram,
        OUT_MODES,
        IN_MODES,
        _config(precision="float32_refine", max_refinements=1, max_cg_iterations=1),
        _rule(max_sweeps=1, escalation_refinements=(2, 3), warn_residual_tolerance=1.0),
    )
    assert result.history
    continued = [update for update in result.history[0].updates if len(update.stages) >= 2]
    assert continued
    for update in continued:
        stages = update.stages
        assert stages[1].initial_residual == stages[0].residual
        if len(stages) == 3:
            assert stages[2].initial_residual == stages[1].residual
        assert stages[-1].cumulative_refinements <= 3
        assert stages[-1].cumulative_refinements == sum(stage.refinements for stage in stages)
        assert all(stage.refinements <= stage.max_refinements for stage in stages)
        assert update.outcome in {"target_met", "local_target_missed", "local_solver_unresolved"}
        if update.outcome == "local_solver_unresolved":
            assert update.accepted is False
        if update.outcome == "local_target_missed":
            assert update.confirmation_a is not None or len(stages) < 3
