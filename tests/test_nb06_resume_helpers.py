"""Notebook 06 の再開 helper。Notebook 本体と smoke 成果物は実行しない。"""

import ast
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.utils import capture_rng_state, notebook_code_sha256, restore_rng_state

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "40_dmrg"
    / "10_fashion_mnist_mlp"
    / "06_dense_tt_dmrg_like_convergence_comparison.ipynb"
)
HELPERS = (
    "figure_temporary_path",
    "canonical_config_value",
    "comparable_config",
    "config_resume_decision",
    "capture_adam_training_state",
    "restore_adam_training_state",
)
SRC_HELPERS = (
    "atomic_torch_save",
    "atomic_write_csv",
    "atomic_write_json",
    "capture_rng_state",
    "file_sha256",
    "index_sha256",
    "notebook_code_sha256",
    "restore_rng_state",
)
REMOVED_DUPLICATES = (
    "file_sha256",
    "tensor_sha256",
    "index_sha256",
    "atomic_replace",
    "atomic_text",
    "normalize_notebook_source",
    "notebook_code_sha256",
    "capture_rng_state",
    "restore_rng_state",
)


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook):
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append(source)
    return cells


def _helpers():
    extracted = []
    for source in _code_cells(_notebook()):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
                extracted.append(ast.get_source_segment(source, node))
    assert [ast.parse(block).body[0].name for block in extracted] == list(HELPERS)
    namespace = {
        "json": json,
        "hashlib": __import__("hashlib"),
        "Path": Path,
        "random": random,
        "np": np,
        "torch": torch,
        "re": __import__("re"),
        "VOLATILE_CONFIG_KEYS": frozenset({"notebook_code_sha256", "implementation_sha256"}),
        "capture_rng_state": capture_rng_state,
        "restore_rng_state": restore_rng_state,
    }
    exec(compile("\n\n".join(extracted), str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def helpers():
    return _helpers()


def test_notebook_json_and_code_cells_parse():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    sources = _code_cells(notebook)
    assert sources
    for source in sources:
        ast.parse(source)
    joined = "\n".join(sources)
    assert "RUN_SMOKE = False" in joined
    assert "RUN_MAIN_EXPERIMENT = False" in joined
    assert "SAVE_RESULTS = True" in joined


def test_dmrg_initial_and_rank_audit_are_preserved_in_future_runs():
    """Fresh Run All で修復済み成果物の来歴と rank 判定を再び壊さない。"""
    sources = _code_cells(_notebook())
    joined = "\n".join(sources)
    continue_dmrg = None
    for source in sources:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "continue_dmrg":
                continue_dmrg = ast.get_source_segment(source, node)
    assert continue_dmrg is not None
    assert 'initial_path = NOTEBOOK05_JOINT / "initial_model.pt"' in continue_dmrg
    assert 'continuation_start_payload = checkpoint_payload(model)' in continue_dmrg
    assert 'continuation_start_after_3200_updates.pt' in joined
    assert "initial_payload = best_payload" not in continue_dmrg
    assert "path_fixed_actual_rank and best_checkpoint_fixed_actual_rank" in joined
    assert 'state["path_fixed_actual_rank"] = path_fixed_actual_rank' in joined
    assert 'state["best_checkpoint_fixed_actual_rank"] = best_checkpoint_fixed_actual_rank' in joined
    assert "state.update(metrics)" not in joined


def test_shared_helpers_come_from_src_and_are_not_redefined():
    defined = set()
    imported = set()
    for source in _code_cells(_notebook()):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "nn_compression.utils":
                imported.update(alias.name for alias in node.names)
    assert not defined & set(REMOVED_DUPLICATES)
    assert set(SRC_HELPERS) <= imported


def _hash_exclusions():
    for source in _code_cells(_notebook()):
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "NOTEBOOK_HASH_EXCLUDED_ASSIGNMENTS"
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("NOTEBOOK_HASH_EXCLUDED_ASSIGNMENTS がありません。")


def test_notebook_imports_resolve():
    seen = set()
    for source in _code_cells(_notebook()):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    seen.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                seen.add(node.module)
    for module_name in sorted(seen):
        __import__(module_name)


def test_figure_temporary_name_keeps_png_suffix(tmp_path, helpers):
    path = tmp_path / "validation_loss.png"
    temporary = helpers["figure_temporary_path"](path)
    assert temporary.name == "validation_loss.tmp.png"
    assert temporary.suffix == ".png"
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    figure.savefig(temporary, dpi=120, bbox_inches="tight", format="png")
    plt.close(figure)
    assert temporary.is_file()
    os.replace(temporary, path)
    assert path.is_file()
    assert not temporary.exists()


def _replace_code_line(notebook, old, new):
    mutated = copy.deepcopy(notebook)
    hits = 0
    for cell in mutated["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        count = source.count(old)
        hits += count
        source = source.replace(old, new)
        cell["source"] = source
    assert hits == 1
    return mutated


def test_notebook_code_hash_ignores_outputs_and_cap_lines():
    exclusions = _hash_exclusions()
    assert set(exclusions) == {
        "DENSE_MAX_EPOCHS",
        "STANDARD_TT_MAX_EPOCHS",
        "MAX_TOTAL_SWEEPS",
        "MAX_TOTAL_LOCAL_UPDATES",
    }

    def digest(notebook):
        return notebook_code_sha256(notebook, excluded_assignments=exclusions)

    notebook = _notebook()
    original = digest(notebook)
    outputs = copy.deepcopy(notebook)
    for cell in outputs["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = 123
            cell["outputs"] = [{"output_type": "stream", "text": "changed"}]
    assert digest(outputs) == original
    assert digest(_replace_code_line(notebook, "DENSE_MAX_EPOCHS = 50", "DENSE_MAX_EPOCHS = 80")) == original
    assert digest(_replace_code_line(notebook, "MAX_TOTAL_SWEEPS = 200", "MAX_TOTAL_SWEEPS = 240")) == original
    assert digest(_replace_code_line(notebook, "DENSE_LEARNING_RATE = 1e-3", "DENSE_LEARNING_RATE = 1e-2")) != original


def _adam_config(max_epochs=50, **overrides):
    config = {
        "learning_rate": 1e-3,
        "train_batch_size": 64,
        "rank_cap": 12,
        "selection_rule": "strict_best_and_stopping_reference",
        "max_epochs": max_epochs,
    }
    config.update(overrides)
    return config


def _dmrg_config(sweeps, updates_per_sweep=80, **overrides):
    config = {
        "tt_learning_rate": 0.3,
        "dense_parameter_learning_rate": 0.3,
        "batch_size": 4096,
        "local_steps_per_pair": 8,
        "updates_per_sweep": updates_per_sweep,
        "max_total_sweeps": sweeps,
        "max_total_local_updates": sweeps * updates_per_sweep,
    }
    config.update(overrides)
    return config


def test_config_resume_decision(helpers):
    decide = helpers["config_resume_decision"]
    base = _adam_config()
    assert decide(base, _adam_config(), "early_stopping", cap_fields=("max_epochs",), completed_amount=12) == "skip"
    assert decide(base, _adam_config(), "max_epochs", cap_fields=("max_epochs",), completed_amount=50) == "skip"
    assert decide(base, _adam_config(), "running", cap_fields=("max_epochs",), completed_amount=7) == "resume"
    assert (
        decide(base, _adam_config(max_epochs=60), "max_epochs", cap_fields=("max_epochs",), completed_amount=50)
        == "extend"
    )
    assert (
        decide(base, _adam_config(max_epochs=60), "running", cap_fields=("max_epochs",), completed_amount=7)
        == "resume"
    )
    with pytest.raises(ValueError, match="上限以外"):
        decide(base, _adam_config(learning_rate=1e-2), "running", cap_fields=("max_epochs",), completed_amount=7)
    with pytest.raises(ValueError, match="上限以外"):
        decide(base, _adam_config(train_batch_size=32), "max_epochs", cap_fields=("max_epochs",), completed_amount=50)
    with pytest.raises(ValueError, match="上限以外"):
        decide(base, _adam_config(selection_rule="other"), "running", cap_fields=("max_epochs",), completed_amount=1)
    with pytest.raises(ValueError, match="上限以外"):
        decide(base, _adam_config(rank_cap=8), "max_epochs", cap_fields=("max_epochs",), completed_amount=50)
    with pytest.raises(ValueError, match="再開しません"):
        decide(base, _adam_config(), "non_finite", cap_fields=("max_epochs",), completed_amount=3)
    with pytest.raises(ValueError, match="再開しません"):
        decide(base, _adam_config(), "out_of_memory", cap_fields=("max_epochs",), completed_amount=3)
    with pytest.raises(ValueError, match="early stopping"):
        decide(base, _adam_config(max_epochs=80), "early_stopping", cap_fields=("max_epochs",), completed_amount=12)
    with pytest.raises(ValueError, match="上限以上"):
        decide(base, _adam_config(), "running", cap_fields=("max_epochs",), completed_amount=50)
    with pytest.raises(ValueError, match="増やしたとき"):
        decide(base, _adam_config(max_epochs=40), "max_epochs", cap_fields=("max_epochs",), completed_amount=50)
    saved = _dmrg_config(41)
    extended = _dmrg_config(200)
    assert (
        decide(
            saved,
            extended,
            "max_sweeps",
            cap_fields=("max_total_sweeps", "max_total_local_updates"),
            completed_amount=41,
        )
        == "extend"
    )
    assert (
        decide(
            saved,
            _dmrg_config(41),
            "running",
            cap_fields=("max_total_sweeps", "max_total_local_updates"),
            completed_amount=40,
        )
        == "resume"
    )
    only_sweeps = _dmrg_config(200)
    only_sweeps["max_total_local_updates"] = saved["max_total_local_updates"]
    with pytest.raises(ValueError, match="揃って"):
        decide(
            saved,
            only_sweeps,
            "max_sweeps",
            cap_fields=("max_total_sweeps", "max_total_local_updates"),
            completed_amount=41,
        )
    broken = _dmrg_config(200)
    broken["max_total_local_updates"] = 16001
    with pytest.raises(ValueError, match="一意"):
        decide(
            saved,
            broken,
            "max_sweeps",
            cap_fields=("max_total_sweeps", "max_total_local_updates"),
            completed_amount=41,
        )
    with pytest.raises(ValueError, match="上限以外"):
        decide(
            saved,
            _dmrg_config(200, updates_per_sweep=40),
            "max_sweeps",
            cap_fields=("max_total_sweeps", "max_total_local_updates"),
            completed_amount=41,
        )
    with pytest.raises(ValueError, match="項目"):
        decide({"max_epochs": 50}, _adam_config(), "running", cap_fields=("max_epochs",), completed_amount=1)


def test_adam_state_resume_matches_continuous_run(tmp_path, helpers):
    capture = helpers["capture_adam_training_state"]
    restore = helpers["restore_adam_training_state"]
    features = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    dataset = TensorDataset(features, torch.arange(8))

    def make_model():
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(0.1)
            model.bias.zero_()
        return model

    def run_epoch(model, loader, optimizer):
        seen = []
        loss_fn = torch.nn.MSELoss()
        for batch_x, batch_y in loader:
            seen.append(batch_y.tolist())
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x).view(-1), batch_y.float())
            loss.backward()
            optimizer.step()
        return seen

    def fresh_run():
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        generator = torch.Generator()
        generator.manual_seed(7)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator, num_workers=0)
        return model, optimizer, generator, loader

    model_a, opt_a, gen_a, loader_a = fresh_run()
    first = run_epoch(model_a, loader_a, opt_a)
    second = run_epoch(model_a, loader_a, opt_a)
    python_a = random.random()
    numpy_a = float(np.random.rand())
    torch_a = torch.rand(3).clone()
    weights_a = copy.deepcopy(model_a.state_dict())
    optimizer_a = copy.deepcopy(opt_a.state_dict())

    model_b, opt_b, gen_b, loader_b = fresh_run()
    assert run_epoch(model_b, loader_b, opt_b) == first
    saved = capture(opt_b, gen_b, completed_epochs=1, optimizer_steps=2, include_cuda=False)
    path = tmp_path / "adam_training_state.pt"
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    torch.save(saved, temporary)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    model_c = make_model()
    model_c.load_state_dict(copy.deepcopy(model_b.state_dict()))
    opt_c = torch.optim.Adam(model_c.parameters(), lr=0.05)
    gen_c = torch.Generator()
    loader_c = DataLoader(dataset, batch_size=4, shuffle=True, generator=gen_c, num_workers=0)
    assert restore(loaded, opt_c, gen_c) == (1, 2)
    assert run_epoch(model_c, loader_c, opt_c) == second
    assert random.random() == python_a
    assert float(np.random.rand()) == numpy_a
    assert torch.equal(torch.rand(3), torch_a)
    for key, value in weights_a.items():
        assert torch.equal(model_c.state_dict()[key], value)
    for index, slot in optimizer_a["state"].items():
        restored_slot = opt_c.state_dict()["state"][index]
        for name, item in slot.items():
            other = restored_slot[name]
            if torch.is_tensor(item):
                assert torch.equal(other.cpu(), item.cpu())
            else:
                assert other == item
