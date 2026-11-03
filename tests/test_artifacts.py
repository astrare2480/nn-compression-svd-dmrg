"""utils.artifacts の atomic 保存とファイル hash。"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import pytest
import torch

from nn_compression.utils import (
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from nn_compression.utils import artifacts


def _leftovers(directory, keep):
    return sorted(path.name for path in directory.iterdir() if path.name not in keep)


def test_atomic_write_bytes_creates_parent_and_uses_same_directory(tmp_path):
    target = tmp_path / "nested" / "data.bin"
    seen = []

    def writer(temporary):
        seen.append(temporary)
        temporary.write_bytes(b"payload")

    assert atomic_write_bytes(target, writer) == target
    assert target.read_bytes() == b"payload"
    assert seen[0].parent == target.parent
    assert seen[0].suffix == ".bin"
    assert seen[0] != target
    assert _leftovers(target.parent, {"data.bin"}) == []


def test_writer_failure_keeps_existing_file_and_removes_temporary(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    def broken(temporary):
        temporary.write_text("half", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(target, broken)
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, {"state.json"}) == []


def test_writer_that_creates_nothing_is_rejected(tmp_path):
    target = tmp_path / "missing.bin"
    with pytest.raises(RuntimeError, match="一時ファイル"):
        atomic_write_bytes(target, lambda temporary: None)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_replace_failure_keeps_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "keep.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, {"keep.txt"}) == []


def test_transient_permission_error_is_retried(tmp_path, monkeypatch):
    target = tmp_path / "retry.txt"
    real_replace = os.replace
    calls = []

    def flaky_replace(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("locked")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", flaky_replace)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    atomic_write_text(target, "done")
    assert target.read_text(encoding="utf-8") == "done"
    assert len(calls) == 3


def test_persistent_permission_error_is_raised_without_leftovers(tmp_path, monkeypatch):
    target = tmp_path / "locked.txt"

    def locked(source, destination):
        raise PermissionError("locked")

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        atomic_write_text(target, "x")
    assert list(tmp_path.iterdir()) == []


def test_directory_target_is_rejected(tmp_path):
    with pytest.raises(IsADirectoryError):
        atomic_write_text(tmp_path, "x")


def test_atomic_write_text_uses_lf_by_default(tmp_path):
    target = tmp_path / "a.txt"
    atomic_write_text(target, "一\n二\n")
    assert target.read_bytes() == "一\n二\n".encode("utf-8")
    with pytest.raises(TypeError):
        atomic_write_text(target, b"bytes")


def test_atomic_write_json_format(tmp_path):
    target = tmp_path / "config.json"
    payload = {"b": 1, "a": "日本語"}
    atomic_write_json(target, payload)
    text = target.read_text(encoding="utf-8")
    assert text == json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert payload == {"b": 1, "a": "日本語"}


def test_atomic_write_csv_round_trip_and_extras(tmp_path):
    target = tmp_path / "rows.csv"
    rows = [{"x": 1, "y": None}, {"x": 0.1, "y": "s"}]
    atomic_write_csv(target, rows, ["x", "y"])
    with target.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"x": "1", "y": ""}, {"x": "0.1", "y": "s"}]
    assert rows == [{"x": 1, "y": None}, {"x": 0.1, "y": "s"}]

    with pytest.raises(ValueError):
        atomic_write_csv(target, [{"x": 1, "z": 2}], ["x"])
    with target.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert _leftovers(tmp_path, {"rows.csv"}) == []

    atomic_write_csv(target, [{"x": 1, "z": 2}], ["x"], extrasaction="ignore")
    with target.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"x": "1"}]


@pytest.mark.parametrize(
    ("fieldnames", "kwargs", "error"),
    [
        ("xy", {}, TypeError),
        ([], {}, ValueError),
        (["x", "x"], {}, ValueError),
        (["x"], {"extrasaction": "drop"}, ValueError),
    ],
)
def test_atomic_write_csv_rejects_bad_arguments(tmp_path, fieldnames, kwargs, error):
    with pytest.raises(error):
        atomic_write_csv(tmp_path / "bad.csv", [], fieldnames, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_atomic_torch_save_round_trip(tmp_path):
    target = tmp_path / "model.pt"
    payload = {"weight": torch.arange(4.0)}
    atomic_torch_save(target, payload)
    loaded = torch.load(target, weights_only=True)
    assert torch.equal(loaded["weight"], payload["weight"])
    assert _leftovers(tmp_path, {"model.pt"}) == []


def test_file_sha256_matches_hashlib(tmp_path):
    target = tmp_path / "large.bin"
    data = os.urandom((1 << 20) + 17)
    target.write_bytes(data)
    assert file_sha256(target) == hashlib.sha256(data).hexdigest()
    assert file_sha256(str(target)) == hashlib.sha256(data).hexdigest()
