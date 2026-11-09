"""本実験の仕事一覧、ロック、方式、記事の切り方、試運転の集計。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nn_compression.compression.metric_low_rank import gram_matrix, matrix_factor_parameter_count
from nn_compression.llm.qwen3_main_experiment import (
    ExperimentLayout,
    build_cell_methods,
    build_job_list,
    chunk_token_spans,
    claim_next_job,
    classify_bonds,
    clear_stale_locks,
    filter_jobs,
    finished_job_ids,
    format_lambda,
    jobs_for_mode,
    summarize_smoke,
    try_claim,
)
from nn_compression.llm.qwen3_main_run import article_units, run_job_loop
from nn_compression.tt_mps.matrix import dense_to_tt_matrix_cores
from nn_compression.tt_mps.metric_tt_solver import MetricTTSolverConfig
from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_csv


def test_job_list_counts_order_and_settings():
    jobs = build_job_list()
    assert len(jobs) == 112
    assert sum(job["kind"] == "M5" for job in jobs) == 42
    assert sum(job["kind"] == "M3F" for job in jobs) == 42
    assert sum(job["kind"] == "M5_M4r" for job in jobs) == 12
    assert sum(job["kind"] == "M5_lambda" for job in jobs) == 16
    assert jobs[0]["id"] == "L14_down_50_M5_M3init_lam1e-06"
    assert [job["layer"] for job in jobs[:28]] == [14] * 28
    assert jobs[0]["init"] == "M3"
    assert jobs[0]["lambda"] == 1e-6
    m4r = [job for job in jobs if job["kind"] == "M5_M4r"]
    assert {job["layer"] for job in m4r} == {14}
    assert "up" not in {job["projection"] for job in m4r}
    assert {job["init"] for job in m4r} == {"M4r"}
    lambdas = [job for job in jobs if job["kind"] == "M5_lambda"]
    assert {job["lambda"] for job in lambdas} == {1e-7, 1e-5}
    assert {job["projection"] for job in lambdas} == {"down", "gate", "q", "o"}
    assert format_lambda(1e-6) == "1e-06"
    layers = []
    for job in jobs:
        if job["layer"] not in layers:
            layers.append(job["layer"])
    assert layers == [14, 7, 21]
    smoke = jobs_for_mode(jobs, "smoke")
    assert sum(job["kind"] == "M5" for job in smoke) == 14
    assert sum(job["kind"] == "M5_M4r" for job in smoke) == 12
    assert sum(job["kind"] == "M3F" for job in smoke) == 1
    assert sum(job["kind"] == "M5_lambda" for job in smoke) == 1
    selected = filter_jobs(jobs, "L14_k_25_M5_M3init")
    assert [job["id"] for job in selected] == ["L14_k_25_M5_M3init_lam1e-06"]


def test_locks_do_not_double_claim_and_stale_locks_are_removed(tmp_path: Path):
    jobs = [{"id": "one"}, {"id": "two"}]
    first = claim_next_job(jobs, tmp_path, set(), "10")
    second = claim_next_job(jobs, tmp_path, set(), "11")
    third = claim_next_job(jobs, tmp_path, set(), "12")
    assert first["id"] == "one"
    assert second["id"] == "two"
    assert third is None
    assert not try_claim(tmp_path / "one.lock", "99")
    stale = tmp_path / "dead.lock"
    stale.write_text("99999999", encoding="utf-8")
    assert "dead.lock" in clear_stale_locks(tmp_path)
    assert not stale.exists()
    finished = tmp_path / "status.csv"
    atomic_write_csv(
        finished,
        [{"job_id": "one", "stop_reason": "max_sweeps"}, {"job_id": "two", "stop_reason": "resource_interrupted"}],
        ["job_id", "stop_reason"],
    )
    assert finished_job_ids(finished) == {"one"}


def test_chunk_articles_keep_the_short_tail():
    spans = chunk_token_spans(2500, 1024)
    assert spans == [(0, 1024), (1024, 2048), (2048, 2500)]
    units = article_units(["short", "abcdefghij"], lambda text: list(range(len(text) * 200)), limit=1)
    assert len(units) == 1
    assert units[0]["spans"][0] == (0, 1024)
    assert units[0]["spans"][-1][1] == units[0]["n_tokens"]
    assert units[0]["n_tokens"] > 1024


def test_small_method_checks_m2_parameters_and_saturation():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(16, 16, generator=generator, dtype=torch.float64)
    activations = torch.randn(32, 16, generator=generator, dtype=torch.float64)
    gram = gram_matrix(activations)
    _payload, row = build_cell_methods(
        weight,
        gram,
        32,
        layer=0,
        projection="k",
        budget_percent=100,
        check_design=False,
    )
    rank = row["parameters_m1"] // (16 + 16)
    assert row["parameters_m1"] == matrix_factor_parameter_count(16, 16, rank)
    assert row["parameters_m3"] > 0
    known = classify_bonds((1, 63, 63, 1), (1, 64, 128, 1))
    assert known["bond_class"] == "near_saturated_bond"
    assert classify_bonds((1, 44, 44, 1), (1, 64, 128, 1))["bond_class"] == "multi_cut"


def test_smoke_summary_from_fabricated_records():
    jobs = [
        {"id": "L14_k_50_M5_M3init_lam1e-06", "kind": "M5", "projection": "k", "budget_percent": 50, "group": "attn_in"},
        {"id": "L14_k_50_M5_M4r_M4rinit_lam1e-06", "kind": "M5_M4r", "projection": "k", "budget_percent": 50, "group": "attn_in"},
    ]
    seconds = {jobs[0]["id"]: [1.0, 3.0], jobs[1]["id"]: [2.0, 2.0]}
    counts = {jobs[0]["id"]: {"stage2": 3, "updates": 10}, jobs[1]["id"]: {"stage2": 0, "updates": 10}}
    under = summarize_smoke(jobs, seconds, counts, [1e-8], [40.0, 60.0], False)
    assert under["tau_warn_precheck"] == "1e-7 の条件を満たす"
    assert under["escalation_by_group"][0]["above_10_percent"]
    over = summarize_smoke(jobs, seconds, counts, [1e-6], [], True)
    assert over["tau_warn_precheck"] == "1e-8"
    assert over["gpu_contended"]
    assert over["expected_hours"] > 0


def _tiny_problem(root: Path) -> ExperimentLayout:
    data, models, results = root / "data", root / "models", root / "results"
    for path in (data / "weights", data / "calibration", models / "methods", results):
        path.mkdir(parents=True)
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(12, 12, generator=generator, dtype=torch.float64)
    activations = torch.randn(8, 12, generator=generator, dtype=torch.float64)
    gram = gram_matrix(activations)
    cores = dense_to_tt_matrix_cores(weight, (2, 3, 2), (2, 2, 3), max_rank=2)
    atomic_torch_save(data / "weights" / "W_L14_k.pt", {"W": weight})
    atomic_torch_save(data / "calibration" / "grams_L14.pt", {"attn_in": {"total": gram, "n_tokens": 8}})
    atomic_torch_save(
        models / "methods" / "L14_k_25.pt",
        {"m3_cores": cores, "out_modes": (2, 3, 2), "in_modes": (2, 2, 3)},
    )
    atomic_write_csv(
        results / "methods_summary.csv",
        [{"cell": "14_k_25", "q_m3": "1e30", "f_m3": "1e30", "d_ref_lam1e-06": "1e12", "d_f_ref": "1e12"}],
        ["cell", "q_m3", "f_m3", "d_ref_lam1e-06", "d_f_ref"],
    )
    return ExperimentLayout(data, models, results)


def test_interrupt_resume_matches_a_fresh_run(tmp_path: Path):
    layout = _tiny_problem(tmp_path / "resume")
    fresh_layout = _tiny_problem(tmp_path / "fresh")
    job = {
        "id": "tiny",
        "layer": 14,
        "projection": "k",
        "budget_percent": 25,
        "kind": "M5",
        "init": "M3",
        "lambda": 1e-6,
        "group": "attn_in",
    }
    config = MetricTTSolverConfig(device="cpu", precision="float64", max_cg_iterations=30, max_refinements=1)
    common = {
        "jobs": [job],
        "max_sweeps": 2,
        "tau_warn": 1e-7,
        "device": "cpu",
        "window_sweeps": 1,
        "config": config,
        "mode": "check",
    }
    with pytest.raises(InterruptedError):
        run_job_loop(layout=layout, lock_dir=tmp_path / "lock-a", owner="a", raise_after_sweep=2, **common)
    interrupted = torch.load(layout.als_dir("check") / "tiny.pt", map_location="cpu", weights_only=True)
    assert interrupted["sweeps_completed"] == 1
    assert interrupted["stop_reason"] is None
    run_job_loop(layout=layout, lock_dir=tmp_path / "lock-a", owner="a", **common)
    run_job_loop(layout=fresh_layout, lock_dir=tmp_path / "lock-b", owner="b", **common)
    resumed = torch.load(layout.als_dir("check") / "tiny.pt", map_location="cpu", weights_only=True)
    fresh = torch.load(fresh_layout.als_dir("check") / "tiny.pt", map_location="cpu", weights_only=True)
    assert resumed["stop_reason"] == fresh["stop_reason"]
    for left, right in zip(resumed["cores"], fresh["cores"], strict=True):
        assert torch.equal(left, right)
    assert resumed["history"][-1]["q_lambda"] == fresh["history"][-1]["q_lambda"]
    assert "tiny" in (layout.als_results("check") / "jobs_status.csv").read_text(encoding="utf-8")
