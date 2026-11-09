"""``nvidia-smi`` で GPU の使用率とメモリを読み、一定間隔で記録する。

``nvidia-smi`` がない、または失敗したときは例外にせず、値なし（``None`` や空の列）を返す。
記録は計算の邪魔をしないことを優先し、測定の失敗で実験を止めない。
"""

from __future__ import annotations

import csv
import math
import subprocess
import threading
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

GPU_QUERY = ("--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits")
APPS_QUERY = ("--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")
SAMPLE_FIELDS = ("seconds", "utilization_gpu", "memory_used_mib", "memory_total_mib")


def run_nvidia_smi(*args: str, timeout: float = 10.0) -> tuple[str, str | None]:
    """``nvidia-smi`` を実行し、(標準出力, 失敗の説明) を返す。成功なら説明は ``None``。"""
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "", f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        return completed.stdout.strip(), completed.stderr.strip() or f"returncode {completed.returncode}"
    return completed.stdout.strip(), None


def query_gpu(gpu_index: int = 0) -> dict[str, int] | None:
    """1 枚の GPU の使用率（%）と、使用中・全体のメモリ（MiB）。読めなければ ``None``。"""
    text, error = run_nvidia_smi(*GPU_QUERY)
    if error is not None:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if gpu_index >= len(lines):
        return None
    parts = [part.strip() for part in lines[gpu_index].split(",")]
    try:
        utilization, used, total = (int(part) for part in parts[:3])
    except ValueError:
        return None
    return {"utilization_gpu": utilization, "memory_used_mib": used, "memory_total_mib": total}


def compute_apps() -> list[dict[str, Any]]:
    """GPU で計算しているプロセスの pid・名前・使用メモリ。読めなければ空の列。"""
    text, error = run_nvidia_smi(*APPS_QUERY)
    if error is not None or not text:
        return []
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "process_name": parts[1], "used_gpu_memory": parts[2]})
    return rows


def other_python_processes(apps: Collection[dict[str, Any]], own_pids: Collection[int]) -> list[int]:
    """``apps`` のうち、``own_pids`` に含まれない Python プロセスの pid。"""
    own = set(own_pids)
    return [
        int(app["pid"])
        for app in apps
        if "python" in str(app["process_name"]).lower() and int(app["pid"]) not in own
    ]


def nvidia_smi_snapshot() -> dict[str, str]:
    """GPU と計算中のプロセスの、``nvidia-smi`` の出力そのもの（記録用）。失敗は ``*_error`` に入れる。"""
    recorded: dict[str, str] = {}
    for name, args in (("gpu", GPU_QUERY), ("apps", APPS_QUERY)):
        text, error = run_nvidia_smi(*args)
        recorded[name] = text
        if error is not None:
            recorded[f"{name}_error"] = error
    return recorded


class GpuSampler:
    """``with`` の間、別スレッドで ``interval`` 秒ごとに GPU を読む。

    ``csv_path`` を渡すと、読むたびに 1 行ずつ書き足して flush する（途中で止まっても残る）。
    行の列は ``SAMPLE_FIELDS``。``seconds`` は ``with`` に入ってからの経過秒。
    読めなかった回は行を作らない。
    記録の途中で予期しない例外が起きたら、記録だけを止めて ``error`` に説明を残す
    （``with`` の中の計算は止めない）。
    """

    def __init__(self, *, interval: float = 1.0, csv_path: Path | str | None = None, gpu_index: int = 0) -> None:
        interval = float(interval)
        if not (math.isfinite(interval) and interval > 0):
            raise ValueError(f"interval は有限の正の値です: {interval}")
        self.interval = interval
        self.csv_path = None if csv_path is None else Path(csv_path)
        self.gpu_index = gpu_index
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> GpuSampler:
        self._started = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 15.0)

    def _loop(self) -> None:
        try:
            self._record()
        except Exception as error:  # noqa: BLE001  測定の失敗で実験を止めない
            self.error = f"{type(error).__name__}: {error}"

    def _record(self) -> None:
        handle = None
        writer = None
        try:
            if self.csv_path is not None:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                handle = self.csv_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
                writer.writeheader()
                handle.flush()
            while True:
                sample = query_gpu(self.gpu_index)
                if sample is not None:
                    row = {"seconds": round(time.perf_counter() - self._started, 1), **sample}
                    self.rows.append(row)
                    if writer is not None:
                        writer.writerow(row)
                        handle.flush()
                if self._stop.wait(self.interval):
                    return
        finally:
            if handle is not None:
                handle.close()

    def summary(self) -> dict[str, float | int | None]:
        """使用率の回数・平均・最小・最大と、使用メモリの最大。"""
        values = [row["utilization_gpu"] for row in self.rows]
        memory = [row["memory_used_mib"] for row in self.rows]
        return {
            "n_samples": len(values),
            "utilization_mean": sum(values) / len(values) if values else None,
            "utilization_min": min(values) if values else None,
            "utilization_max": max(values) if values else None,
            "memory_used_max_mib": max(memory) if memory else None,
        }
