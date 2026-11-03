"""Notebook 11 の学習 run ID、wall-clock、図の再生成条件を合成データなしで確認する。"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path("notebooks/40_dmrg/10_fashion_mnist_mlp/11_full_batch_global_vs_local_update_control.ipynb")


@pytest.fixture(scope="module")
def notebook():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["RUN_SMOKE"] is False
    assert namespace["RUN_MAIN_EXPERIMENT"] is False
    assert namespace["REGENERATE_DISPLAY_ARTIFACTS"] is False
    return namespace


def test_training_run_id_ignores_display_code_and_follows_contract(notebook):
    original_hash = notebook["notebook_code_sha256"]
    original_version = notebook["TRAINING_CONTRACT_VERSION"]
    original_rates = notebook["LEARNING_RATES"]
    try:
        first = notebook["training_run_identifier"](False)
        settings = notebook["training_settings"](False)
        assert "notebook_code_sha256" not in settings
        assert settings["training_contract_version"] == original_version
        assert notebook["canonical_settings"](False)["notebook_code_sha256"]

        def display_only(*_args, **_kwargs):
            return "display-only-hash"

        notebook["notebook_code_sha256"] = display_only
        assert notebook["training_run_identifier"](False) == first
        assert notebook["canonical_settings"](False)["notebook_code_sha256"] == "display-only-hash"

        notebook["TRAINING_CONTRACT_VERSION"] = original_version + 1
        assert notebook["training_run_identifier"](False) != first
        notebook["TRAINING_CONTRACT_VERSION"] = original_version

        notebook["LEARNING_RATES"] = (0.1, 0.2, 0.31)
        assert notebook["training_run_identifier"](False) != first
    finally:
        notebook["notebook_code_sha256"] = original_hash
        notebook["TRAINING_CONTRACT_VERSION"] = original_version
        notebook["LEARNING_RATES"] = original_rates


def test_resumed_wall_clock_is_monotonic(notebook):
    assert notebook["wall_clock_offset"]([]) == 0.0
    assert notebook["wall_clock_offset"]([{"wall_clock_seconds": 12.5}]) == 12.5
    resumed = notebook["resumed_wall_clock_seconds"](12.5, 100.0, 101.25)
    assert resumed == pytest.approx(13.75)
    notebook["require_monotonic_wall_clock"]([
        {"wall_clock_seconds": 12.5},
        {"wall_clock_seconds": resumed},
    ])
    with pytest.raises(RuntimeError, match="減少"):
        notebook["require_monotonic_wall_clock"]([
            {"wall_clock_seconds": 13.0},
            {"wall_clock_seconds": 12.0},
        ])


def test_redraw_stops_without_overwriting_when_manifest_disagrees(notebook, tmp_path):
    results = tmp_path / "results"
    figures = results / "figures"
    figures.mkdir(parents=True)
    sentinel = figures / "01_gradient_evaluations_vs_validation_loss.png"
    sentinel.write_bytes(b"keep")
    (results / "results_manifest.json").write_text(
        json.dumps({"training_run_id": "different-experiment", "conditions": []}),
        encoding="utf-8",
    )
    (results / "condition_summary.csv").write_text(
        "method,learning_rate,training_run_id\nglobal_gd,0.1,different-experiment\n",
        encoding="utf-8",
    )
    original_dirs = notebook["experiment_dirs"]
    original_save = notebook["save_figure"]
    calls = {"save": 0}

    def experiment_dirs(_smoke):
        return tmp_path / "models", results

    def save_figure(*_args, **_kwargs):
        calls["save"] += 1
        raise AssertionError("不一致のまま図を保存しました。")

    notebook["experiment_dirs"] = experiment_dirs
    notebook["save_figure"] = save_figure
    try:
        with pytest.raises(RuntimeError, match="上書きしません"):
            notebook["redraw_saved_figures"]()
    finally:
        notebook["experiment_dirs"] = original_dirs
        notebook["save_figure"] = original_save
    assert calls["save"] == 0
    assert sentinel.read_bytes() == b"keep"
