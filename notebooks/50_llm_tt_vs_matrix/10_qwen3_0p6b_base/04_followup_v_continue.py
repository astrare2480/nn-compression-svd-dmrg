"""04 の追加分析：v_proj 25% の M5 を、保存済みの解から解き足す。

問い：同じ較正の目的のまま解き進めると、先頭のトークンの value の誤差は減るか。
（減れば「最適化の不足」、目的が下がっても横ばいなら「目的の重みづけ」を支持する。）

- 初期値：04 の M5（M3 から始めた本）の最終コア。
- 規則：04 と同じソルバー設定・τ_warn・λ。ただし「研究上の分解能」で止めない
  （consecutive_windows を非常に大きくする）。止まるのは掃引の上限か、局所問題が解けないときだけ。
- 窓（既定 100 掃引）ごとに、e_wt2・e_c4 を計算し、その時点のコアを保存する。
  先頭のトークンの誤差は、保存したコアから後で計算する（モデルを動かすのは後）。
- 途中で止めても、同じコマンドで窓の区切りから再開する。

使い方:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_v_continue.py --extra-sweeps 4000
  数分の確認（保存は check に分ける）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_v_continue.py --check
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from nn_compression.compression.metric_low_rank import regularize_gram
from nn_compression.llm.qwen3_main_experiment import (
    HISTORY_FIELDS,
    file_sha256,
    history_row,
    layout_from_root,
    read_tau_warn,
    solver_config,
)
from nn_compression.llm.qwen3_main_run import _summary_map, _validation_errors, d_ref_for
from nn_compression.tt_mps.metric_tt_solver import StudyResolutionStopRule, run_metric_tt_als_study
from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_csv, atomic_write_json
from nn_compression.utils.paths import find_project_root

CELLS = ("L7_v_25", "L14_v_25", "L21_v_25")
NO_RESOLUTION_STOP = 10**9


def _job_for(layout, cell: str) -> dict:
    jobs = json.loads((layout.results / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    job_id = f"{cell}_M5_M3init_lam1e-06"
    matches = [job for job in jobs if job["id"] == job_id]
    if len(matches) != 1:
        raise RuntimeError(f"{job_id} が jobs.json に 1 件ありません。")
    return matches[0]


def run_cell(spec: dict) -> None:
    root = Path(spec["root"])
    layout = layout_from_root(root)
    cell = spec["cell"]
    job = _job_for(layout, cell)
    tag = spec["tag"]
    out_models = layout.models / "followup" / tag / cell
    out_results = layout.results / "followup_analysis" / tag
    out_models.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    start_path = layout.als_dir("full") / f"{job['id']}.pt"
    start = torch.load(start_path, map_location="cpu", weights_only=True)
    if start["stop_reason"] is None:
        raise RuntimeError(f"{start_path} は終わっていない保存です。")
    method = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
    weight_path = layout.weights / f"W_L{job['layer']}_{job['projection']}.pt"
    weight = torch.load(weight_path, map_location="cpu", weights_only=True)["W"]
    grams_path = layout.calibration / f"grams_L{job['layer']}.pt"
    raw = torch.load(grams_path, map_location="cpu", weights_only=True)[job["group"]]["total"]
    gram = regularize_gram(raw, float(job["lambda"]))
    out_modes, in_modes = method["out_modes"], method["in_modes"]

    summary = _summary_map(layout.results / "methods_summary.csv")[f"{job['layer']}_{job['projection']}_{job['budget_percent']}"]
    rule = StudyResolutionStopRule(
        d_ref=d_ref_for(job, summary),
        window_sweeps=spec["window"],
        resolution_tolerance=1e-4,
        consecutive_windows=NO_RESOLUTION_STOP,
        max_sweeps=spec["extra_sweeps"],
        local_residual_tolerance=1e-8,
        escalation_refinements=(10, 50),
        warn_residual_tolerance=spec["tau_warn"],
    )
    metadata = {
        "job_id": f"{job['id']}_continue",
        "start_checkpoint": str(start_path.relative_to(root)),
        "start_checkpoint_sha256": file_sha256(start_path),
        "start_sweeps": int(start["sweeps_completed"]),
        "start_stop_reason": str(start["stop_reason"]),
        "weight_sha256": file_sha256(weight_path),
        "gram_sha256": file_sha256(grams_path),
        "method_sha256": file_sha256(layout.methods / f"{cell}.pt"),
    }

    def diagnostics(cores, sweep):
        values = dict(_validation_errors(layout, job, cores, out_modes, in_modes, weight))
        atomic_torch_save(
            out_models / f"cores_s{sweep:05d}.pt",
            {"cores": [core.to(torch.float64).contiguous() for core in cores], "sweep_after_start": sweep,
             "start_sweeps": int(start["sweeps_completed"]), "out_modes": out_modes, "in_modes": in_modes},
        )
        return values

    started = time.perf_counter()

    def progress(record):
        if record.sweep % 50 == 0 or record.stop_reason:
            print(
                f"{cell} +{record.sweep} Q={record.q_lambda:.10e} s_max={record.stationarity_max:.3e} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    result = run_metric_tt_als_study(
        [core.to(torch.float64) for core in start["cores"]],
        weight,
        gram,
        out_modes,
        in_modes,
        solver_config(spec["device"]),
        rule,
        report_gram=raw,
        checkpoint_path=out_models / "checkpoint.pt",
        metadata=metadata,
        diagnostics=diagnostics,
        progress=progress,
    )
    errors = {int(item["sweep"]): item for item in result.diagnostics}
    rows = [
        history_row(record, {"e_wt2": errors.get(record.sweep, {}).get("e_wt2"),
                             "e_c4": errors.get(record.sweep, {}).get("e_c4")})
        for record in result.history
    ]
    atomic_write_csv(out_results / f"{cell}_history.csv", rows, HISTORY_FIELDS)
    atomic_write_json(
        out_results / f"{cell}_summary.json",
        {
            **metadata,
            "stop_reason": result.stop_reason,
            "extra_sweeps_done": result.history[-1].sweep if result.history else 0,
            "initial_q_lambda": result.initial_q_lambda,
            "start_q_lambda_recorded": float(start["q_lambda_last"]),
            "final_q_lambda": result.history[-1].q_lambda if result.history else None,
            "final_s_max": result.history[-1].stationarity_max if result.history else None,
            "seconds_total": result.seconds_total,
            "window": spec["window"],
            "max_extra_sweeps": spec["extra_sweeps"],
        },
    )
    print(f"{cell} done stop={result.stop_reason}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-sweeps", type=int, default=4000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="2 掃引だけ。保存先は v_continue_check")
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = find_project_root(Path(__file__))
    layout = layout_from_root(root)
    tau = read_tau_warn(layout.settings, mode="full")
    cells = [cell for cell in args.cells.split(",") if cell]
    base = {
        "root": str(root),
        "tag": "v_continue_check" if args.check else "v_continue",
        "window": 1 if args.check else args.window,
        "extra_sweeps": 2 if args.check else args.extra_sweeps,
        "tau_warn": tau,
        "device": args.device,
    }
    if os.name == "nt":
        # 実行中だけ Windows の自動スリープを止める（設定は変えない。プロセス終了で元に戻る）
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    context = mp.get_context("spawn")
    processes = [context.Process(target=run_cell, args=({**base, "cell": cell},)) for cell in cells]
    for process in processes:
        process.start()
    failed = []
    for cell, process in zip(cells, processes):
        process.join()
        if process.exitcode != 0:
            failed.append(cell)
    if failed:
        raise SystemExit(f"失敗: {failed}")


if __name__ == "__main__":
    main()
