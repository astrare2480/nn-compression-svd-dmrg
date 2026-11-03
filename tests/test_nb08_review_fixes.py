"""Notebook 08 の import、初回保存、評価世代、実装 hash、正準形停止。"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
import torch

from nn_compression.training.gpu_resident_batches import GpuResidentTensorBatches
from nn_compression.training.tt_linear_full_batch_sweep import _batch_on_device, _full_batch_samples

NOTEBOOK = Path("notebooks/40_dmrg/10_fashion_mnist_mlp/08_adaptive_bond_rank_dmrg_like_pareto.ipynb")


def _namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "nb08_review"}
    for index in (1, 3):
        source = notebook.cells[index].source
        exec(compile(source, f"nb08_cell_{index}", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def nb08():
    return _namespace()


def test_fresh_namespace_has_numpy_and_rng_roundtrip(nb08):
    assert nb08["np"].random is not None
    state = nb08["capture_rng_state"]()
    nb08["np"].random.seed(123)
    nb08["restore_rng_state"](state)
    assert nb08["implementation_sha256"]() == nb08["implementation_sha256"]()
    assert "display_comparison" in nb08["LEARNING_HASH_EXCLUDED"]
    assert "train_adaptive" in nb08["LEARNING_HASH_INCLUDED"]
    assert "evaluate_saved_bests" in nb08["LEARNING_HASH_EXCLUDED"]


def test_stage_c_initial_canonicality_is_within_tolerance(nb08):
    shared = nb08["audit_initial_state"]()
    assert shared["initial_canonicality_error"] <= nb08["CANONICALITY_TOLERANCE"]
    assert shared["initial_ranks"] == (1, 8, 12, 12, 12, 12, 1)


def test_canonicality_failure_rejects_large_finite_error(nb08):
    assert nb08["canonicality_failure"](1e-3, 1e-4) == "tolerance"
    assert nb08["canonicality_failure"](float("nan"), 1e-4) == "non_finite"
    assert nb08["canonicality_failure"](1e-6, 1e-4) == ""


def test_first_running_publish_uses_rng_and_sweep_boundary(nb08, tmp_path):
    source = tmp_path / "initial.pt"
    torch.save({"marker": 1}, source)
    digest = nb08["file_sha256"](source)
    config = {"initial_model_sha256": digest, "implementation_sha256": "unit"}
    history = [{"sweep": 0, "cumulative_local_updates": 0, "max_canonicality_error": 0.0}]
    run_state = {
        "stop_reason": "running",
        "completed_epochs": None,
        "completed_sweeps": 0,
        "optimizer_steps": 0,
        "total_local_updates": 0,
        "history_rows": 1,
        "seconds_save": 0.0,
    }
    directory = tmp_path / "condition"
    nb08["publish_generation"](
        directory,
        config,
        {"torch": "test"},
        run_state,
        history,
        [],
        {"best": {"actual_ranks": [1]}, "final": {"actual_ranks": [1]}},
        None,
        True,
        {"rng": nb08["capture_rng_state"](), "center": 0, "completed_sweeps": 0, "total_local_updates": 0},
        source,
    )
    pointer = json.loads((directory / "active_generation.json").read_text(encoding="utf-8"))
    assert pointer["completed_sweeps"] == 0
    assert pointer["total_local_updates"] == 0
    assert pointer["stop_reason"] == "running"
    assert (directory / "initial_model.pt").is_file()


def test_implementation_hash_mismatch_does_not_rewrite_saved_config(nb08, tmp_path):
    source = tmp_path / "initial.pt"
    torch.save({"marker": 2}, source)
    current = nb08["implementation_sha256"]()
    config = nb08["adaptive_config"](
        {"split_sha256": nb08["NOTEBOOK06_SPLIT_INDEX_SHA256"]},
        1e-4,
        1,
    )
    config["initial_model_sha256"] = nb08["file_sha256"](source)
    config["implementation_sha256"] = current
    directory = tmp_path / "trained"
    run_state = {
        "stop_reason": "running",
        "completed_epochs": None,
        "completed_sweeps": 0,
        "optimizer_steps": 0,
        "total_local_updates": 0,
        "history_rows": 1,
        "center": 0,
        "seconds_save": 0.0,
    }
    nb08["publish_generation"](
        directory,
        config,
        {"torch": "test"},
        run_state,
        [{"sweep": 0}],
        [],
        {"best": {"actual_ranks": [1]}, "final": {"actual_ranks": [1]}},
        None,
        True,
        {"rng": nb08["capture_rng_state"](), "center": 0, "completed_sweeps": 0, "total_local_updates": 0},
        source,
    )
    config_path = directory / "generation_000001" / "config.json"
    before = config_path.read_bytes()
    original = nb08["implementation_sha256"]
    nb08["implementation_sha256"] = lambda: "0" * 64
    try:
        with pytest.raises(RuntimeError, match="hash"):
            nb08["directory_action"](directory, config)
    finally:
        nb08["implementation_sha256"] = original
    assert config_path.read_bytes() == before


def test_unresumable_stop_blocks_resume_without_replacing_pointer(nb08, tmp_path):
    source = tmp_path / "initial.pt"
    torch.save({"marker": 3}, source)
    config = nb08["adaptive_config"](
        {"split_sha256": nb08["NOTEBOOK06_SPLIT_INDEX_SHA256"]},
        1e-6,
        1,
    )
    config["initial_model_sha256"] = nb08["file_sha256"](source)
    config["implementation_sha256"] = nb08["implementation_sha256"]()
    directory = tmp_path / "stopped"
    run_state = {
        "stop_reason": "running",
        "completed_epochs": None,
        "completed_sweeps": 0,
        "optimizer_steps": 0,
        "total_local_updates": 0,
        "history_rows": 1,
        "center": 0,
        "seconds_save": 0.0,
    }
    nb08["publish_generation"](
        directory,
        config,
        {"torch": "test"},
        run_state,
        [{"sweep": 0}],
        [],
        {"best": {"actual_ranks": [1]}, "final": {"actual_ranks": [1]}},
        None,
        True,
        {"rng": nb08["capture_rng_state"](), "center": 0, "completed_sweeps": 0, "total_local_updates": 0},
        source,
    )
    pointer_before = (directory / "active_generation.json").read_bytes()
    nb08["atomic_json"](
        directory / "unresumable_stop.json",
        {"stop_reason": "canonicality_tolerance", "resumable": False, "center": 3, "cumulative_updates": 10},
    )
    with pytest.raises(RuntimeError, match="再開しません"):
        nb08["directory_action"](directory, config)
    assert (directory / "active_generation.json").read_bytes() == pointer_before


def test_evaluation_reuses_same_best_and_separates_a_new_best(nb08, tmp_path, monkeypatch):
    monkeypatch.setitem(nb08, "RESULTS_DIR", tmp_path / "results")
    settings = {"validation_batch_size": 512}
    metrics = {
        "threshold": 1e-4,
        "best_checkpoint_sha256": "aaa",
        "validation_error": 0.2,
        "latency_batch_1_seconds": "",
        "latency_batch_4096_seconds": "",
        "training_time_seconds": 3.0,
    }
    first = {
        "threshold": 1e-4,
        "best_checkpoint_sha256": "aaa",
        "evaluation_settings": settings,
        "latency_measured": False,
        "metrics": metrics,
    }
    nb08["write_evaluation_generation"](1e-4, first)
    active = nb08["load_active_evaluation"](1e-4)
    assert nb08["evaluation_decision"](active, "aaa", settings, False) == "reuse"
    assert nb08["evaluation_decision"](active, "aaa", settings, True) == "add_latency"
    assert nb08["evaluation_decision"](active, "bbb", settings, True) == "new_generation"
    before = (tmp_path / "results" / "evaluations" / "threshold_1e-04" / "evaluation_000001.json").read_bytes()
    followup = nb08["build_latency_followup"](active, {1: 0.01, 4096: 0.2}, "gpu-a")
    assert followup["metrics"]["validation_error"] == 0.2
    assert followup["metrics"]["best_checkpoint_sha256"] == "aaa"
    assert followup["metrics"]["latency_batch_1_seconds"] == 0.01
    nb08["write_evaluation_generation"](1e-4, followup)
    assert (tmp_path / "results" / "evaluations" / "threshold_1e-04" / "evaluation_000001.json").read_bytes() == before
    latest = nb08["load_active_evaluation"](1e-4)
    assert latest["latency_measured"] is True
    assert latest["best_checkpoint_sha256"] == "aaa"


def test_stale_evaluation_is_not_used_after_best_changes(nb08, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(nb08, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setitem(nb08, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setitem(nb08, "DISCARDED_WEIGHT_THRESHOLDS", (1e-4,))
    source = tmp_path / "initial.pt"
    torch.save({"marker": 4}, source)
    config = nb08["adaptive_config"](
        {"split_sha256": nb08["NOTEBOOK06_SPLIT_INDEX_SHA256"]},
        1e-4,
        1,
    )
    config["initial_model_sha256"] = nb08["file_sha256"](source)
    config["implementation_sha256"] = nb08["implementation_sha256"]()
    directory = nb08["condition_directory"](nb08["MODELS_DIR"], 1e-4)
    run_state = {
        "stop_reason": "early_stopping",
        "completed_epochs": None,
        "completed_sweeps": 0,
        "optimizer_steps": 0,
        "total_local_updates": 0,
        "history_rows": 1,
        "center": 0,
        "seconds_save": 0.0,
    }
    nb08["publish_generation"](
        directory,
        config,
        {"torch": "test"},
        run_state,
        [{"sweep": 0}],
        [],
        {"best": {"actual_ranks": [1], "marker": "new-best"}, "final": {"actual_ranks": [1]}},
        None,
        True,
        {"rng": nb08["capture_rng_state"](), "center": 0, "completed_sweeps": 0, "total_local_updates": 0},
        source,
    )
    best_path = nb08["load_active_generation"](directory)["path"] / "best_model.pt"
    best_bytes = best_path.read_bytes()
    settings = nb08["current_evaluation_settings"]()
    stale = {
        "threshold": 1e-4,
        "best_checkpoint_sha256": "stale-best",
        "evaluation_settings": settings,
        "latency_measured": True,
        "metrics": {
            "threshold": 1e-4,
            "best_checkpoint_sha256": "stale-best",
            "validation_error": 0.11,
            "training_time_seconds": 9.0,
            "latency_batch_1_seconds": 0.5,
        },
    }
    nb08["write_evaluation_generation"](1e-4, stale)
    evaluation_path = tmp_path / "results" / "evaluations" / "threshold_1e-04" / "evaluation_000001.json"
    evaluation_bytes = evaluation_path.read_bytes()
    rows, missing = nb08["load_adaptive_rows"]()
    assert rows == []
    assert missing == [{"threshold": 1e-4, "reason": "best更新後の評価が未実施"}]
    nb08["display_comparison"]()
    printed = capsys.readouterr().out
    assert "best更新後の評価が未実施" in printed
    assert "RUN_EVALUATE_BEST=True" in printed
    assert "0.11" not in printed
    mismatched = dict(stale)
    mismatched["best_checkpoint_sha256"] = nb08["file_sha256"](best_path)
    mismatched["metrics"] = dict(stale["metrics"])
    mismatched["metrics"]["best_checkpoint_sha256"] = mismatched["best_checkpoint_sha256"]
    mismatched["evaluation_settings"] = dict(settings)
    mismatched["evaluation_settings"]["validation_batch_size"] = 1
    nb08["write_evaluation_generation"](1e-4, mismatched)
    rows, missing = nb08["load_adaptive_rows"]()
    assert rows == []
    assert missing == [{"threshold": 1e-4, "reason": "評価設定が現在と違うため未使用"}]
    assert best_path.read_bytes() == best_bytes
    assert evaluation_path.read_bytes() == evaluation_bytes


def test_notebook_main_keeps_the_gpu_resident_slice_path():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = notebook.cells[3].source
    main = source[source.find("def execute_main") : source.find("def run_smoke")]
    adaptive = source[source.find("def train_adaptive") : source.find("def execute_main")]
    evaluate = source[source.find("def evaluate_early") : source.find("def evaluate_early") + 400]
    assert main.index("GpuResidentTensorBatches") < main.index("for threshold")
    assert "resident.n_samples != SPLIT_LENGTHS[0]" in main
    assert "batch_size=DMRG_BATCH_SIZE" in adaptive
    assert "discarded_weight_threshold=float(threshold)" in adaptive
    assert "make_loader" in evaluate
    assert "DataLoader(" not in adaptive[adaptive.find("for pair_index") :]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU常駐sliceは CUDA が必要です。")
def test_gpu_slices_do_not_copy_the_resident_storage():
    images = torch.randn(10, 1, 8, 8, device="cuda")
    labels = torch.arange(10, device="cuda")
    pointer = images.data_ptr()
    resident = GpuResidentTensorBatches(images, labels)
    count, batches = _full_batch_samples(resident, 4)
    seen = 0
    for batch_images, batch_labels in batches:
        moved_images, moved_labels = _batch_on_device(batch_images, batch_labels, torch.device("cuda"))
        assert moved_images.data_ptr() == batch_images.data_ptr()
        assert moved_labels.device.type == "cuda"
        seen += moved_images.shape[0]
    assert count == 10 and seen == 10
    assert images.data_ptr() == pointer
