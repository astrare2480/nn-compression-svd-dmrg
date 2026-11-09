"""長い ALS（停留性・停止の規則・保存と再開）の、小さい問題での検算。

設計：plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence/design.md §12
"""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import torch

from nn_compression.compression.metric_low_rank import gram_matrix, regularize_gram
from nn_compression.tt_mps.matrix import dense_to_tt_matrix_cores, tt_matrix_to_dense
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt_solver import (
    ConvergenceStopRule,
    MetricTTSolverConfig,
    metric_tt_stationarity_on_device,
    perturb_tt_matrix_site,
    run_metric_tt_als_on_device,
    run_metric_tt_als_until_converged,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA がない")

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


def _problem(seed: int, rank: int = 3, n_tokens: int = 20):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(math.prod(OUT_MODES), math.prod(IN_MODES), generator=generator, dtype=torch.float64)
    cores = dense_to_tt_matrix_cores(weight, OUT_MODES, IN_MODES, max_rank=rank)
    activations = torch.randn(n_tokens, math.prod(IN_MODES), generator=generator, dtype=torch.float64)
    raw = gram_matrix(activations)
    return cores, weight, regularize_gram(raw, 1e-6), raw


def _config(**overrides) -> MetricTTSolverConfig:
    settings = {"device": "cpu", "precision": "float64", "max_cg_iterations": 500}
    settings.update(overrides)
    return MetricTTSolverConfig(**settings)


def _without_timing(record) -> dict:
    return {key: value for key, value in asdict(record).items() if key not in TIMING_FIELDS}


def _explicit_stationarity(cores, weight, gram, site: int) -> float:
    """``P_k`` を列ごとに作り、``‖P_kᵀ vec((W − Ŵ)G)‖ / ‖P_kᵀ vec(W G)‖`` を直接計算する。"""
    canonical = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, site)
    shape = canonical[site].shape
    columns = []
    for index in range(canonical[site].numel()):
        unit = torch.zeros(canonical[site].numel(), dtype=torch.float64)
        unit[index] = 1.0
        replaced = list(canonical)
        replaced[site] = unit.reshape(shape)
        columns.append(tt_matrix_to_dense(replaced, OUT_MODES, IN_MODES).reshape(-1))
    projector = torch.stack(columns, dim=1)
    identity = torch.eye(projector.shape[1], dtype=torch.float64)
    assert torch.allclose(projector.T @ projector, identity, atol=1e-12)
    approximated = tt_matrix_to_dense(canonical, OUT_MODES, IN_MODES)
    numerator = projector.T @ ((weight - approximated) @ gram).reshape(-1)
    denominator = projector.T @ (weight @ gram).reshape(-1)
    return float(torch.linalg.vector_norm(numerator) / torch.linalg.vector_norm(denominator))


@pytest.mark.parametrize("center", [0, 1, 2])
def test_stationarity_matches_the_explicit_projection_at_every_site(center):
    cores, weight, gram, _raw = _problem(1)
    canonical = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, center)
    measured = metric_tt_stationarity_on_device(
        canonical, weight, gram, OUT_MODES, IN_MODES, center=center, device="cpu"
    )
    expected = [_explicit_stationarity(cores, weight, gram, site) for site in range(3)]
    assert measured == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert all(value > 0.0 for value in measured)


def test_stationarity_at_the_end_of_a_sweep_equals_the_next_pre_update_residual_at_site_0():
    cores, weight, gram, _raw = _problem(2)
    config = _config(sweep_tolerance=-1.0)
    after_one, _history = run_metric_tt_als_on_device(
        cores, weight, gram, OUT_MODES, IN_MODES, replace_sweeps(config, 1)
    )
    _fitted, history = run_metric_tt_als_on_device(
        cores, weight, gram, OUT_MODES, IN_MODES, replace_sweeps(config, 2)
    )
    measured = metric_tt_stationarity_on_device(
        after_one, weight, gram, OUT_MODES, IN_MODES, center=0, device="cpu"
    )
    assert measured[0] == pytest.approx(history[1].local_gradient_norms[0], rel=1e-12, abs=0.0)


def replace_sweeps(config: MetricTTSolverConfig, sweeps: int) -> MetricTTSolverConfig:
    from dataclasses import replace

    return replace(config, max_sweeps=sweeps)


def test_stationarity_leaves_the_inputs_unchanged():
    cores, weight, gram, _raw = _problem(3)
    canonical = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, 0)
    copies = [core.clone() for core in canonical]
    metric_tt_stationarity_on_device(canonical, weight, gram, OUT_MODES, IN_MODES, center=0, device="cpu")
    for before, after in zip(copies, canonical, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("precision", ["float64", "float32_refine"])
def test_runner_reproduces_the_existing_als_trajectory(precision):
    cores, weight, gram, raw = _problem(4)
    config = _config(precision=precision, sweep_tolerance=-1.0, max_sweeps=4)
    reference_cores, reference = run_metric_tt_als_on_device(cores, weight, gram, OUT_MODES, IN_MODES, config)
    rule = ConvergenceStopRule(relative_decrease_tolerance=1e-300, max_sweeps=4)
    result = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule, report_gram=raw
    )
    assert [record.q_lambda for record in result.history] == [record.q_lambda for record in reference]
    assert [record.cg_iterations for record in result.history] == [record.cg_iterations for record in reference]
    for got, want in zip(result.cores, reference_cores, strict=True):
        assert torch.equal(got, want)
    assert result.stop_reason == "max_sweeps"
    assert [record.stop_reason for record in result.history] == [None, None, None, "max_sweeps"]
    for record in result.history:
        assert record.update_sites == (0, 1, 2, 1, 0)
        assert record.monotone_ok
        assert record.q_unregularized is not None
        assert record.stationarity_max == max(record.stationarity)
        assert not any(record.retried)


def test_runner_returns_cpu_float64_and_leaves_the_inputs_unchanged():
    cores, weight, gram, raw = _problem(5)
    copies = [core.clone() for core in cores], weight.clone(), gram.clone(), raw.clone()
    result = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, _config(), ConvergenceStopRule(max_sweeps=2), report_gram=raw
    )
    for core in result.cores:
        assert core.dtype == torch.float64 and core.device.type == "cpu"
    for before, after in zip(copies[0], cores, strict=True):
        assert torch.equal(before, after)
    assert torch.equal(copies[1], weight) and torch.equal(copies[2], gram) and torch.equal(copies[3], raw)
    assert result.initial_q_lambda > result.history[-1].q_lambda
    assert result.resumed_from_sweep is None


def test_converged_needs_both_the_patience_and_the_stationarity_conditions():
    cores, weight, gram, _raw = _problem(6)
    # 相対減少は 1 未満なので毎掃引条件を満たす。s_max も 1e3 以下。→ 3 掃引目で止まる。
    loose = ConvergenceStopRule(relative_decrease_tolerance=1.0, patience_sweeps=3, stationarity_tolerance=1e3)
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, _config(), loose)
    assert [record.stop_reason for record in result.history] == [None, None, "converged"]
    assert [record.small_decrease_streak for record in result.history] == [1, 2, 3]
    # s_max の条件だけを満たさなくすると、上限まで回る。
    strict = ConvergenceStopRule(
        relative_decrease_tolerance=1.0, patience_sweeps=3, stationarity_tolerance=1e-300, max_sweeps=5
    )
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, _config(), strict)
    assert result.stop_reason == "max_sweeps"
    assert len(result.history) == 5


def test_small_decrease_streak_resets_when_the_decrease_is_large():
    cores, weight, gram, _raw = _problem(7)
    probe = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, _config(), ConvergenceStopRule(max_sweeps=8)
    )
    decreases = [record.relative_decrease for record in probe.history]
    threshold = sorted(decreases)[len(decreases) // 2]
    rule = ConvergenceStopRule(relative_decrease_tolerance=threshold, max_sweeps=8, stationarity_tolerance=1e-300)
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, _config(), rule)
    streak = 0
    for record in result.history:
        streak = streak + 1 if record.relative_decrease < threshold else 0
        assert record.small_decrease_streak == streak
    assert 0 < max(record.small_decrease_streak for record in result.history)


def test_local_solver_unresolved_stops_after_one_retry_and_keeps_the_initial_state():
    cores, weight, gram, _raw = _problem(8)
    config = _config(precision="float32_refine", max_cg_iterations=1, max_refinements=1)
    rule = ConvergenceStopRule(retry_max_refinements=1)
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, config, rule)
    assert result.stop_reason == "local_solver_unresolved"
    assert result.history == ()
    failure = result.failure
    assert failure is not None
    assert (failure.sweep, failure.site) == (1, 0)
    assert failure.update_sites == (0,)
    assert failure.update_retried == (True,)
    assert failure.update_accepted == (False,)
    assert failure.first_residual > rule.local_residual_tolerance
    assert failure.retry_residual > rule.local_residual_tolerance
    assert failure.refinements == 2
    # 完了した掃引がないので、最終状態は正準化した初期値のまま。
    initial = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, 0)
    for got, want in zip(result.cores, initial, strict=True):
        assert torch.equal(got, want)


def test_a_failure_in_a_later_sweep_discards_that_whole_sweep(monkeypatch, tmp_path):
    import nn_compression.tt_mps.metric_tt_solver as solver

    cores, weight, gram, raw = _problem(14)
    config = _config()
    rule = ConvergenceStopRule(max_sweeps=4, stationarity_tolerance=1e-300)
    reference = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, replace_rule_sweeps(rule, 1), report_gram=raw
    )
    original = solver._solve_site
    calls = {"n": 0}

    def failing(*args, **kwargs):
        output = original(*args, **kwargs)
        calls["n"] += 1
        # 1 掃引目の 5 回と、2 掃引目のサイト 0 は通す。2 掃引目のサイト 1 から失敗させる。
        if calls["n"] >= 7:
            return (output[0], output[1], 1.0, *output[3:])
        return output

    monkeypatch.setattr(solver, "_solve_site", failing)
    path = tmp_path / "run.pt"
    result = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule, report_gram=raw, checkpoint_path=path
    )
    assert result.stop_reason == "local_solver_unresolved"
    assert len(result.history) == 1
    assert result.history[0].stop_reason is None
    failure = result.failure
    assert (failure.sweep, failure.site) == (2, 1)
    assert failure.update_sites == (0, 1)
    assert failure.update_retried == (False, True)
    assert failure.update_accepted[1] is False
    assert failure.retry_residual == 1.0
    assert math.isfinite(failure.q_lambda_candidate) and math.isfinite(failure.q_lambda_before)
    # 2 掃引目のサイト 0 で受理した更新も捨て、1 掃引目の終わりの状態を最終状態にする。
    for got, want in zip(result.cores, reference.cores, strict=True):
        assert torch.equal(got, want)
    saved = torch.load(path, weights_only=True)
    assert saved["sweeps_completed"] == 1
    assert saved["stop_reason"] == "local_solver_unresolved"
    assert saved["failure"]["site"] == 1
    again = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule, report_gram=raw, checkpoint_path=path
    )
    assert again.failure == failure
    for got, want in zip(again.cores, result.cores, strict=True):
        assert torch.equal(got, want)


def replace_rule_sweeps(rule: ConvergenceStopRule, sweeps: int) -> ConvergenceStopRule:
    from dataclasses import replace

    return replace(rule, max_sweeps=sweeps)


@pytest.mark.parametrize("site", [0, 1, 2])
def test_perturbation_has_the_requested_relative_size_and_leaves_the_inputs_unchanged(site):
    cores, _weight, _gram, _raw = _problem(15)
    copies = [core.clone() for core in cores]
    perturbed, delta = perturb_tt_matrix_site(cores, OUT_MODES, IN_MODES, site=site, relative_size=1e-2, seed=0)
    assert delta == pytest.approx(1e-2, rel=1e-10)
    base = tt_matrix_to_dense(cores, OUT_MODES, IN_MODES)
    changed = tt_matrix_to_dense(perturbed, OUT_MODES, IN_MODES)
    measured = float(torch.linalg.matrix_norm(changed - base) / torch.linalg.matrix_norm(base))
    assert measured == pytest.approx(1e-2, rel=1e-8)
    again, _ = perturb_tt_matrix_site(cores, OUT_MODES, IN_MODES, site=site, relative_size=1e-2, seed=0)
    for left, right in zip(perturbed, again, strict=True):
        assert torch.equal(left, right)
    for before, after in zip(copies, cores, strict=True):
        assert torch.equal(before, after)


def test_a_site_0_perturbation_is_erased_by_the_first_update_but_a_site_1_one_is_not():
    cores, weight, gram, _raw = _problem(16, rank=2)
    config = _config(cg_relative_tolerance=1e-13)
    rule = ConvergenceStopRule(max_sweeps=1)

    def q_after_one_sweep(start):
        return run_metric_tt_als_until_converged(
            start, weight, gram, OUT_MODES, IN_MODES, config, rule
        ).history[0].q_lambda

    base = q_after_one_sweep(cores)
    site0, _ = perturb_tt_matrix_site(cores, OUT_MODES, IN_MODES, site=0, relative_size=1e-2, seed=0)
    site1, _ = perturb_tt_matrix_site(cores, OUT_MODES, IN_MODES, site=1, relative_size=1e-2, seed=0)
    assert abs(q_after_one_sweep(site0) - base) / base < 1e-10
    assert abs(q_after_one_sweep(site1) - base) / base > 1e-6


def test_a_successful_retry_lets_the_run_continue():
    cores, weight, gram, _raw = _problem(9)
    # 外側反復 1 回では 1e-4 程度までしか下がらないので、解き直しが必要になる。
    config = _config(precision="float32_refine", max_refinements=1)
    rule = ConvergenceStopRule(max_sweeps=2, retry_max_refinements=10)
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, config, rule)
    assert result.stop_reason == "max_sweeps"
    retried = [flag for record in result.history for flag in record.retried]
    assert any(retried)
    for record in result.history:
        assert all(residual <= rule.local_residual_tolerance for residual in record.cg_residuals)


def test_adaptive_tolerance_is_refused():
    cores, weight, gram, _raw = _problem(10)
    with pytest.raises(ValueError):
        run_metric_tt_als_until_converged(
            cores, weight, gram, OUT_MODES, IN_MODES, _config(cg_tolerance_mode="adaptive"), ConvergenceStopRule()
        )
    with pytest.raises(ValueError):
        run_metric_tt_als_until_converged(
            cores, weight, gram, OUT_MODES, IN_MODES, _config(cg_relative_tolerance=1e-6), ConvergenceStopRule()
        )


def _dense_norm(cores, sweep: int) -> dict:
    return {"dense_norm": float(torch.linalg.matrix_norm(tt_matrix_to_dense(cores, OUT_MODES, IN_MODES)))}


class _Interrupt(RuntimeError):
    pass


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    cores, weight, gram, raw = _problem(11)
    config = _config(precision="float32_refine")
    rule = ConvergenceStopRule(max_sweeps=7, checkpoint_every=2, stationarity_tolerance=1e-300)
    metadata = {"initialization": "M3", "inputs": {"weights": "abc"}}
    whole = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule,
        report_gram=raw, checkpoint_path=tmp_path / "whole.pt", metadata=metadata, diagnostics=_dense_norm,
    )
    assert [item["sweep"] for item in whole.diagnostics] == [2, 4, 6, 7]

    path = tmp_path / "split.pt"

    def stop_at_five(record):
        if record.sweep == 5:
            raise _Interrupt

    with pytest.raises(_Interrupt):
        run_metric_tt_als_until_converged(
            cores, weight, gram, OUT_MODES, IN_MODES, config, rule,
            report_gram=raw, checkpoint_path=path, metadata=metadata, progress=stop_at_five,
            diagnostics=_dense_norm,
        )
    saved = torch.load(path, weights_only=True)
    assert saved["sweeps_completed"] == 4 and saved["stop_reason"] is None

    resumed = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule,
        report_gram=raw, checkpoint_path=path, metadata=metadata, diagnostics=_dense_norm,
    )
    assert resumed.resumed_from_sweep == 4
    assert [_without_timing(record) for record in resumed.history] == [
        _without_timing(record) for record in whole.history
    ]
    for got, want in zip(resumed.cores, whole.cores, strict=True):
        assert torch.equal(got, want)
    assert resumed.diagnostics == whole.diagnostics
    assert resumed.initial_q_lambda == whole.initial_q_lambda

    # 止まった保存ファイルからは、計算せずにそのまま返す。
    again = run_metric_tt_als_until_converged(
        cores, weight, gram, OUT_MODES, IN_MODES, config, rule,
        report_gram=raw, checkpoint_path=path, metadata=metadata,
        progress=lambda record: (_ for _ in ()).throw(AssertionError("計算した")),
    )
    assert again.stop_reason == "max_sweeps"
    assert [_without_timing(record) for record in again.history] == [
        _without_timing(record) for record in whole.history
    ]


def test_resume_refuses_mismatched_settings_or_inputs(tmp_path):
    cores, weight, gram, _raw = _problem(12)
    config = _config()
    rule = ConvergenceStopRule(max_sweeps=4, checkpoint_every=1, stationarity_tolerance=1e-300)
    path = tmp_path / "run.pt"

    def stop_at_two(record):
        if record.sweep == 2:
            raise _Interrupt

    with pytest.raises(_Interrupt):
        run_metric_tt_als_until_converged(
            cores, weight, gram, OUT_MODES, IN_MODES, config, rule,
            checkpoint_path=path, metadata={"initialization": "M3"}, progress=stop_at_two,
        )
    attempts = [
        dict(metadata={"initialization": "M4_retruncated"}),
        dict(gram=gram * 2.0),
        dict(config=_config(max_refinements=4)),
        dict(rule=ConvergenceStopRule(max_sweeps=5, checkpoint_every=1, stationarity_tolerance=1e-300)),
    ]
    for overrides in attempts:
        arguments = dict(config=config, rule=rule, gram=gram, metadata={"initialization": "M3"})
        arguments.update(overrides)
        with pytest.raises(ValueError):
            run_metric_tt_als_until_converged(
                cores, weight, arguments["gram"], OUT_MODES, IN_MODES, arguments["config"], arguments["rule"],
                checkpoint_path=path, metadata=arguments["metadata"],
            )


@needs_cuda
def test_cuda_float32_refine_runner_matches_the_existing_cuda_als():
    cores, weight, gram, _raw = _problem(13)
    config = _config(device="cuda", precision="float32_refine", sweep_tolerance=-1.0, max_sweeps=3)
    _reference_cores, reference = run_metric_tt_als_on_device(cores, weight, gram, OUT_MODES, IN_MODES, config)
    rule = ConvergenceStopRule(relative_decrease_tolerance=1e-300, max_sweeps=3)
    result = run_metric_tt_als_until_converged(cores, weight, gram, OUT_MODES, IN_MODES, config, rule)
    assert [record.q_lambda for record in result.history] == pytest.approx(
        [record.q_lambda for record in reference], rel=1e-12
    )
    assert all(core.device.type == "cpu" for core in result.cores)
    assert result.history[0].peak_memory_bytes is not None
