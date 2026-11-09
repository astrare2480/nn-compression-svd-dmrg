"""CSV を行で読む関数と、JSON に書ける値へ直す関数。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_compression.utils.artifacts import atomic_write_csv, read_csv_rows, to_jsonable


def test_read_csv_rows(tmp_path):
    path = tmp_path / "rows.csv"
    atomic_write_csv(path, [{"a": 1, "b": "x"}, {"a": 2, "b": ""}], ["a", "b"])
    assert read_csv_rows(path) == [{"a": "1", "b": "x"}, {"a": "2", "b": ""}]
    assert read_csv_rows(tmp_path / "missing.csv", missing_ok=True) == []
    with pytest.raises(FileNotFoundError):
        read_csv_rows(tmp_path / "missing.csv")


def test_to_jsonable_handles_special_values():
    value = {
        1: (np.int64(3), np.float32(0.5), np.bool_(True)),
        "path": Path("a") / "b",
        "floats": [math.nan, math.inf, -math.inf, np.float64(-np.inf)],
        "tensor": torch.tensor(2.5),
        "none": None,
    }
    out = to_jsonable(value)
    assert out == {
        "1": [3, 0.5, True],
        "path": str(Path("a") / "b"),
        "floats": ["nan", "inf", "-inf", "-inf"],
        "tensor": 2.5,
        "none": None,
    }
    json.dumps(out, allow_nan=False)
    with pytest.raises(TypeError):
        to_jsonable(object())
    with pytest.raises(TypeError):
        to_jsonable(torch.ones(2))


def test_to_jsonable_refuses_keys_that_collide_after_str():
    with pytest.raises(ValueError):
        to_jsonable({1: "numeric", "1": "text"})
    with pytest.raises(ValueError):
        to_jsonable({"outer": {Path("a"): 1, "a": 2}})
    assert to_jsonable({1: "x", 2: "y"}) == {"1": "x", "2": "y"}
