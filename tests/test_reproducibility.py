"""utils.reproducibility の hash と乱数状態。"""

from __future__ import annotations

import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from nn_compression.utils import (
    capture_rng_state,
    index_sha256,
    notebook_code_sha256,
    restore_rng_state,
    tensor_sha256,
)


def test_tensor_hash_includes_dtype_shape_and_values():
    base = torch.arange(6, dtype=torch.float32)
    digest = tensor_sha256(base)
    assert tensor_sha256(base.clone()) == digest
    assert tensor_sha256(base.reshape(2, 3)) != digest
    assert tensor_sha256(base.to(torch.float64)) != digest
    changed = base.clone()
    changed[0] = 0.5
    assert tensor_sha256(changed) != digest
    assert digest != hashlib.sha256(base.numpy().tobytes()).hexdigest()


def test_tensor_hash_handles_views_grad_and_bfloat16_without_mutation():
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert tensor_sha256(matrix.t()) == tensor_sha256(matrix.t().contiguous())
    leaf = torch.ones(3, requires_grad=True)
    before = leaf.detach().clone()
    tensor_sha256(leaf)
    assert torch.equal(leaf.detach(), before)
    assert leaf.grad is None
    assert tensor_sha256(torch.ones(2, dtype=torch.bfloat16)) != tensor_sha256(
        torch.ones(2, dtype=torch.float16)
    )
    assert tensor_sha256(torch.tensor(3.0)) != tensor_sha256(torch.tensor([3.0]))


def test_tensor_hash_rejects_non_tensor_and_sparse():
    with pytest.raises(TypeError):
        tensor_sha256([1, 2])
    with pytest.raises(ValueError):
        tensor_sha256(torch.eye(2).to_sparse())


def test_index_hash_is_int64_bytes_and_accepts_subset():
    indices = [5, 0, 3]
    expected = hashlib.sha256(torch.tensor(indices, dtype=torch.int64).numpy().tobytes()).hexdigest()
    dataset = TensorDataset(torch.zeros(6, 1))
    assert index_sha256(indices) == expected
    assert index_sha256(Subset(dataset, indices)) == expected
    assert index_sha256(torch.tensor(indices, dtype=torch.int32)) == expected
    assert index_sha256(np.array(indices, dtype=np.int16)) == expected
    assert index_sha256([0, 3, 5]) != expected


@pytest.mark.parametrize(
    "bad",
    [[1, True], [1.0, 2.0], torch.tensor([1.0]), torch.tensor([[1, 2]]), "12", torch.tensor([True])],
)
def test_index_hash_rejects_non_integer_indices(bad):
    with pytest.raises((TypeError, ValueError)):
        index_sha256(bad)


def _notebook():
    return {
        "cells": [
            {"cell_type": "markdown", "id": "m1", "metadata": {}, "source": ["# title\n"]},
            {
                "cell_type": "code",
                "id": "c1",
                "execution_count": 3,
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": ["1\n"]}],
                "source": ["RUN = False\n", "MAX_EPOCHS = 50\n", "x = 1\n"],
            },
            {
                "cell_type": "code",
                "id": "c2",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": "def f():\n    MAX_EPOCHS = 3\n    return MAX_EPOCHS == 3\n",
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_notebook_hash_ignores_non_code_parts():
    notebook = _notebook()
    original = copy.deepcopy(notebook)
    digest = notebook_code_sha256(notebook)
    assert notebook == original

    changed = copy.deepcopy(notebook)
    changed["cells"][0]["source"] = ["# other\n"]
    changed["cells"][1]["outputs"] = []
    changed["cells"][1]["execution_count"] = 99
    changed["cells"][1]["id"] = "zzz"
    changed["cells"][2]["metadata"] = {"tags": ["x"]}
    changed["metadata"] = {"kernelspec": {"name": "other"}}
    changed["cells"][1]["source"] = ["RUN = False\r\n", "MAX_EPOCHS = 50   \r\n", "x = 1"]
    assert notebook_code_sha256(changed) == digest

    edited = copy.deepcopy(notebook)
    edited["cells"][1]["source"] = ["RUN = False\n", "MAX_EPOCHS = 50\n", "x = 2\n"]
    assert notebook_code_sha256(edited) != digest

    merged = copy.deepcopy(notebook)
    merged["cells"][1]["source"] = merged["cells"][1]["source"] + ["\n", merged["cells"][2]["source"]]
    del merged["cells"][2]
    assert notebook_code_sha256(merged) != digest


def test_notebook_hash_excluded_assignments_only_drop_assignments():
    notebook = _notebook()
    excluded = ("RUN", "MAX_EPOCHS")
    digest = notebook_code_sha256(notebook, excluded_assignments=excluded)
    assert digest != notebook_code_sha256(notebook)

    caps_changed = copy.deepcopy(notebook)
    caps_changed["cells"][1]["source"] = ["RUN = True\n", "MAX_EPOCHS: int = 80\n", "x = 1\n"]
    caps_changed["cells"][2]["source"] = "def f():\n    MAX_EPOCHS = 4\n    return MAX_EPOCHS == 3\n"
    assert notebook_code_sha256(caps_changed, excluded_assignments=excluded) == digest

    comparison_changed = copy.deepcopy(notebook)
    comparison_changed["cells"][2]["source"] = "def f():\n    MAX_EPOCHS = 3\n    return MAX_EPOCHS == 4\n"
    assert notebook_code_sha256(comparison_changed, excluded_assignments=excluded) != digest


def test_notebook_hash_reads_path(tmp_path):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(_notebook()), encoding="utf-8")
    assert notebook_code_sha256(path) == notebook_code_sha256(_notebook())
    assert notebook_code_sha256(str(path)) == notebook_code_sha256(_notebook())


@pytest.mark.parametrize(
    ("notebook", "kwargs", "error"),
    [
        ({"cells": "x"}, {}, ValueError),
        ([], {}, ValueError),
        ({"cells": [{"cell_type": "code", "source": 3}]}, {}, TypeError),
        ({"cells": []}, {"excluded_assignments": "RUN"}, TypeError),
        ({"cells": []}, {"excluded_assignments": ["not a name"]}, ValueError),
    ],
)
def test_notebook_hash_rejects_bad_input(notebook, kwargs, error):
    with pytest.raises(error):
        notebook_code_sha256(notebook, **kwargs)


def _draw():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_capture_and_restore_rng_state_round_trip():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = capture_rng_state(include_cuda=False)
    expected = _draw()
    _draw()
    restore_rng_state(state)
    assert _draw() == expected
    restore_rng_state(state)
    assert _draw() == expected
    assert "cuda" not in state


def test_captured_state_is_an_independent_copy():
    torch.manual_seed(0)
    np.random.seed(0)
    state = capture_rng_state(include_cuda=False)
    torch_before = state["torch"].clone()
    numpy_before = state["numpy"][1].copy()
    _draw()
    assert torch.equal(state["torch"], torch_before)
    assert np.array_equal(state["numpy"][1], numpy_before)


def test_restore_does_not_mutate_state():
    state = capture_rng_state(include_cuda=False)
    snapshot = copy.deepcopy(state)
    restore_rng_state(state)
    _draw()
    assert state["python"] == snapshot["python"]
    assert torch.equal(state["torch"], snapshot["torch"])
    assert np.array_equal(state["numpy"][1], snapshot["numpy"][1])


def test_restore_accepts_states_loaded_from_torch_save(tmp_path):
    state = capture_rng_state(include_cuda=False)
    expected = _draw()
    path = tmp_path / "rng.pt"
    torch.save({"rng": state}, path)
    loaded = torch.load(path, weights_only=False)["rng"]
    restore_rng_state(loaded)
    assert _draw() == expected


def test_invalid_torch_rng_length_leaves_every_generator_unchanged():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    state = capture_rng_state(include_cuda=False)
    saved_torch = state["torch"].clone()
    state["torch"] = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="長さ"):
        restore_rng_state(state)
    assert random.getstate() == python_before
    assert numpy_before[0] == np.random.get_state()[0]
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    assert numpy_before[2:] == np.random.get_state()[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.equal(state["torch"], torch.zeros(4, dtype=torch.uint8))
    assert torch.equal(saved_torch, torch_before)


def test_restore_rejects_missing_keys_without_partial_restore():
    state = capture_rng_state(include_cuda=False)
    broken = {key: value for key, value in state.items() if key != "torch"}
    random.seed(10)
    marker = random.getstate()
    with pytest.raises(KeyError):
        restore_rng_state(broken)
    assert random.getstate() == marker
    with pytest.raises(TypeError):
        restore_rng_state([1, 2, 3])


def test_include_cuda_true_without_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        capture_rng_state(include_cuda=True)
    assert "cuda" not in capture_rng_state()


def test_cuda_state_device_count_mismatch_is_rejected(monkeypatch):
    state = capture_rng_state(include_cuda=False)
    state["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="device 数"):
        restore_rng_state(state)


def test_cuda_states_are_restored_for_all_devices(monkeypatch):
    state = capture_rng_state(include_cuda=False)
    state["cuda"] = [torch.full((4,), 1, dtype=torch.uint8), torch.full((4,), 2, dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))
    restore_rng_state(state)
    assert [int(tensor[0]) for tensor in restored] == [1, 2]
    assert all(tensor is not original for tensor, original in zip(restored, state["cuda"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が必要")
def test_cuda_round_trip_on_real_device():
    state = capture_rng_state()
    assert len(state["cuda"]) == torch.cuda.device_count()
    expected = float(torch.rand(1, device="cuda"))
    torch.rand(1, device="cuda")
    restore_rng_state(state)
    assert float(torch.rand(1, device="cuda")) == expected
