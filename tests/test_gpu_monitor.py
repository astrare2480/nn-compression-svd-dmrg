"""nvidia-smi の出力の読み方と、一定間隔の記録（nvidia-smi の呼び出しは差し替える）。"""

from __future__ import annotations

import csv
import time

import pytest

from nn_compression.utils import gpu_monitor as gm


def _fake_smi(outputs):
    def run(*args, timeout=10.0):
        if args == gm.GPU_QUERY:
            return outputs.get("gpu", ("", "missing"))
        if args == gm.APPS_QUERY:
            return outputs.get("apps", ("", "missing"))
        raise AssertionError(args)

    return run


def test_query_gpu_reads_the_requested_card(monkeypatch):
    monkeypatch.setattr(gm, "run_nvidia_smi", _fake_smi({"gpu": ("71, 5000, 16000\n12, 100, 8000", None)}))
    assert gm.query_gpu() == {"utilization_gpu": 71, "memory_used_mib": 5000, "memory_total_mib": 16000}
    assert gm.query_gpu(1)["utilization_gpu"] == 12
    assert gm.query_gpu(2) is None


def test_failures_return_no_values(monkeypatch):
    monkeypatch.setattr(gm, "run_nvidia_smi", _fake_smi({"gpu": ("", "not found"), "apps": ("", "not found")}))
    assert gm.query_gpu() is None
    assert gm.compute_apps() == []
    snapshot = gm.nvidia_smi_snapshot()
    assert snapshot["gpu_error"] == "not found" and snapshot["apps_error"] == "not found"
    monkeypatch.setattr(gm, "run_nvidia_smi", _fake_smi({"gpu": ("[N/A], 1, 2", None)}))
    assert gm.query_gpu() is None


def test_run_nvidia_smi_without_the_command_does_not_raise(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gm.subprocess, "run", missing)
    text, error = gm.run_nvidia_smi(*gm.GPU_QUERY)
    assert text == "" and "FileNotFoundError" in error


def test_other_python_processes(monkeypatch):
    text = "10, C:\\python.exe, 500 MiB\n11, C:\\other.exe, 1 MiB\n12, /usr/bin/python3, 2 MiB"
    monkeypatch.setattr(gm, "run_nvidia_smi", _fake_smi({"apps": (text, None)}))
    apps = gm.compute_apps()
    assert [app["pid"] for app in apps] == [10, 11, 12]
    assert gm.other_python_processes(apps, {10}) == [12]
    assert gm.other_python_processes(apps, {10, 12}) == []


def test_sampler_records_rows_and_streams_csv(monkeypatch, tmp_path):
    values = iter([("50, 10, 100", None), ("", "busy"), ("70, 30, 100", None)])
    monkeypatch.setattr(gm, "run_nvidia_smi", lambda *a, **k: next(values, ("90, 40, 100", None)))
    path = tmp_path / "gpu.csv"
    with gm.GpuSampler(interval=0.01, csv_path=path) as sampler:
        deadline = time.perf_counter() + 5.0
        while len(sampler.rows) < 3 and time.perf_counter() < deadline:
            time.sleep(0.01)
    assert [row["utilization_gpu"] for row in sampler.rows[:3]] == [50, 70, 90]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(gm.SAMPLE_FIELDS)
    assert len(rows) == len(sampler.rows)
    summary = sampler.summary()
    assert summary["utilization_min"] == 50 and summary["memory_used_max_mib"] >= 30


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_sampler_rejects_non_finite_or_non_positive_intervals(interval):
    with pytest.raises(ValueError):
        gm.GpuSampler(interval=interval)


def test_sampler_thread_ends_after_the_block(monkeypatch):
    monkeypatch.setattr(gm, "run_nvidia_smi", lambda *a, **k: ("10, 1, 2", None))
    with gm.GpuSampler(interval=0.01) as sampler:
        time.sleep(0.05)
    assert not sampler._thread.is_alive()
    assert sampler.error is None


def test_unexpected_errors_stop_recording_but_not_the_block(monkeypatch, tmp_path):
    def broken(gpu_index=0):
        raise RuntimeError("driver crashed")

    monkeypatch.setattr(gm, "query_gpu", broken)
    with gm.GpuSampler(interval=0.01, csv_path=tmp_path / "gpu.csv") as sampler:
        deadline = time.perf_counter() + 5.0
        while sampler.error is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        value = "still running"
    assert value == "still running"
    assert sampler.error == "RuntimeError: driver crashed"
    assert not sampler._thread.is_alive()
    assert sampler.rows == []
