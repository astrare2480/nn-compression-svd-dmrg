"""試運転の記録から時間・解き直し・τ_warn の事前確認・GPU をまとめる。

  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_smoke_summary.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from nn_compression.llm.qwen3_main_experiment import (
    build_job_list,
    jobs_for_mode,
    layout_from_root,
    summarize_smoke,
)
from nn_compression.llm.qwen3_main_run import read_history_timing, read_update_counts
from nn_compression.utils.artifacts import atomic_write_json
from nn_compression.utils.paths import find_project_root


def main() -> None:
    layout = layout_from_root(find_project_root(Path(__file__)))
    jobs = jobs_for_mode(build_job_list(), "smoke")
    seconds = {}
    counts = {}
    residuals = []
    for job in jobs:
        path = layout.als_results("smoke") / f"{job['id']}_history.csv"
        seconds[job["id"]] = read_history_timing(path)
        info = read_update_counts(path)
        counts[job["id"]] = info
        residuals.extend(info["residuals"])
    utilizations = []
    gpu_path = layout.results / "gpu_utilization_smoke.csv"
    if gpu_path.is_file():
        with gpu_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("utilization_gpu"):
                    utilizations.append(float(row["utilization_gpu"]))
    before = layout.results / "gpu_before_smoke.json"
    contended = False
    if before.is_file():
        contended = bool(json.loads(before.read_text(encoding="utf-8")).get("other_python"))
    payload = summarize_smoke(jobs, seconds, counts, residuals, utilizations, contended)
    atomic_write_json(layout.results / "smoke_summary.json", payload)
    print(payload["tau_warn_precheck"], payload["exceeds_48h"], flush=True)


if __name__ == "__main__":
    main()
