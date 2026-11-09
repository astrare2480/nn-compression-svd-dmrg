"""ALS の仕事を実行する。

本番:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_run_jobs.py --workers 2 --mode full

試運転:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_run_jobs.py --workers 2 --mode smoke

数分の確認（smoke の保存とは混ぜない）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_run_jobs.py --workers 2 --mode check --only L14_k_25_M5_M3init --max-sweeps 2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import torch.multiprocessing as mp

from nn_compression.llm.qwen3_main_experiment import (
    clear_stale_locks,
    filter_jobs,
    jobs_for_mode,
    layout_from_root,
    read_tau_warn,
)
from nn_compression.llm.qwen3_main_run import assert_production_data, ensure_jobs_file, worker_process
from nn_compression.utils.artifacts import atomic_write_json
from nn_compression.utils.paths import find_project_root


def _nvidia(*query: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *query],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _other_python(text: str, own_pids: set[int]) -> bool:
    for line in text.splitlines():
        lowered = line.lower()
        if "python" not in lowered:
            continue
        pid_text = line.split(",", 1)[0].strip()
        if pid_text.isdigit() and int(pid_text) not in own_pids:
            return True
    return False


def _watch_gpu(path: Path, stop: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seconds", "utilization_gpu", "memory_used_mib", "memory_total_mib"])
        started = time.perf_counter()
        while True:
            row = _nvidia("--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits")
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 3:
                writer.writerow([f"{time.perf_counter() - started:.1f}", *parts[:3]])
                handle.flush()
            if stop.wait(10):
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("smoke", "full", "check"), required=True)
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-sweeps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = find_project_root(Path(__file__))
    layout = layout_from_root(root)
    assert_production_data(layout, args.mode)
    jobs = filter_jobs(jobs_for_mode(ensure_jobs_file(layout), args.mode), args.only)
    if args.mode == "smoke":
        max_sweeps = 5 if args.max_sweeps is None else args.max_sweeps
    elif args.mode == "full":
        max_sweeps = 1000 if args.max_sweeps is None else args.max_sweeps
    else:
        if args.max_sweeps is None:
            raise SystemExit("check では --max-sweeps が必要です。")
        max_sweeps = args.max_sweeps
    tau = read_tau_warn(layout.settings, mode=args.mode)
    lock_dir = layout.results / "locks" / args.mode
    lock_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_locks(lock_dir)
    apps = _nvidia("--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")
    atomic_write_json(
        layout.results / f"gpu_before_{args.mode}.json",
        {"other_python": _other_python(apps, {os.getpid()}), "apps": apps},
    )
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_gpu,
        args=(layout.results / f"gpu_utilization_{args.mode}.csv", stop),
        daemon=True,
    )
    watcher.start()
    spec = {
        "jobs": jobs,
        "data": str(layout.data),
        "models": str(layout.models),
        "results": str(layout.results),
        "mode": args.mode,
        "max_sweeps": max_sweeps,
        "tau_warn": tau,
        "device": args.device,
        "lock_dir": str(lock_dir),
    }
    if args.workers == 1:
        worker_process(spec)
    else:
        context = mp.get_context("spawn")
        processes = [context.Process(target=worker_process, args=(spec,)) for _ in range(args.workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                raise SystemExit(process.exitcode)
    stop.set()
    watcher.join(timeout=2)
    print(json.dumps({"jobs": len(jobs), "mode": args.mode, "max_sweeps": max_sweeps}), flush=True)


if __name__ == "__main__":
    main()
