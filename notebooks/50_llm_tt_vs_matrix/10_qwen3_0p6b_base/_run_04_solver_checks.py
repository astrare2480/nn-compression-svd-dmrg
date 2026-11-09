"""04 本実験のソルバーの確認（Claude が回す）。

設計：plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md の §6.2a・§3.2
1. down_proj・T2・50%・M3 を run_metric_tt_als_study で 50 掃引（実データで動くか、1 掃引の時間）
2. up_proj・T2・50%・M3 を同じ関数で 50 掃引し、03 の history.csv と Q_λ を比べる
3. G：03 の 50%・M4_retruncated の 426 掃引目の終わりから、仮の τ_warn = 1e-7 で 5 掃引

使い方：python _run_04_solver_checks.py [down|up|g|all]
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT = Path(r"D:\dev\nn-compression-svd-dmrg")
sys.path.insert(0, str(PROJECT / "src"))

from nn_compression.compression.metric_low_rank import regularize_gram  # noqa: E402
from nn_compression.tt_mps.metric_tt_solver import (  # noqa: E402
    MetricTTSolverConfig,
    StudyResolutionStopRule,
    run_metric_tt_als_study,
)

PILOT_DATA = PROJECT / "data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot"
PILOT_MODELS = PROJECT / "models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot"
DATA04 = PROJECT / "data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment"
MODELS = PROJECT / "models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev"
RESULTS = PROJECT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev"
R03 = PROJECT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence"
M03 = PROJECT / "models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence"

CONFIG = MetricTTSolverConfig(
    device="cuda",
    precision="float32_refine",
    preconditioner="jacobi",
    cg_tolerance_mode="fixed",
    cg_relative_tolerance=1e-8,
    max_cg_iterations=500,
    max_refinements=5,
    cg_check_every=1,
    timing_sync=False,
)
LAMBDA = 1e-6


class GpuSampler:
    """nvidia-smi で GPU の使用率を 2 秒ごとに記録する。"""

    def __init__(self) -> None:
        self.rows: list[tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().split(",")
                self.rows.append((time.time(), int(out[0]), int(out[1])))
            except Exception:
                pass
            self._stop.wait(2.0)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        values = [row[1] for row in self.rows]
        return {
            "n_samples": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }


def gpu_snapshot() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()


def d_refs() -> dict:
    return json.loads((RESULTS / "data_checks.json").read_text(encoding="utf-8"))


def progress_printer(label: str):
    def show(record) -> None:
        print(
            f"{label} sweep {record.sweep} Q {record.q_lambda:.10e} {record.seconds:.2f}s "
            f"hvp {record.hvp_calls} stage2 {record.n_stage2} stage3 {record.n_stage3} warn {record.n_warnings}",
            flush=True,
        )
    return show


def sweep_rows(label: str, result) -> list[dict]:
    rows = []
    for record in result.history:
        rows.append({
            "run": label,
            "sweep": record.sweep,
            "q_lambda": repr(record.q_lambda),
            "relative_decrease": record.relative_decrease,
            "stationarity_max": record.stationarity_max,
            "delta_50": record.delta_50,
            "n_stage2": record.n_stage2,
            "n_stage3": record.n_stage3,
            "n_warnings": record.n_warnings,
            "hvp_calls": record.hvp_calls,
            "cg_iterations": ";".join(str(v) for v in record.cg_iterations),
            "cg_residuals": ";".join(f"{v:.3e}" for v in record.cg_residuals),
            "n_rejected": record.n_rejected,
            "seconds": record.seconds,
            "seconds_hvp": record.seconds_hvp,
            "seconds_stationarity": record.seconds_stationarity,
            "peak_memory_bytes": record.peak_memory_bytes,
        })
    return rows


def escalated_updates(result) -> list[dict]:
    items = []
    for record in result.history:
        for index, update in enumerate(record.updates):
            if len(update.stages) > 1:
                item = asdict(update)
                item["sweep"] = record.sweep
                item["update_index"] = index
                items.append(item)
    return items


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_down() -> dict:
    payload = torch.load(MODELS / "m3_down_T2_50.pt", weights_only=True)
    weight = torch.load(DATA04 / "w_down_float64.pt", weights_only=True)["W_down"]
    grams = torch.load(DATA04 / "calibration_grams.pt", weights_only=True)
    gram = grams["down_proj_total"]
    rule = StudyResolutionStopRule(d_ref=float(d_refs()["d_ref_down"]), max_sweeps=50)
    before = gpu_snapshot()
    with GpuSampler() as sampler:
        result = run_metric_tt_als_study(
            payload["cores"], weight, regularize_gram(gram, LAMBDA), payload["out_modes"], payload["in_modes"],
            CONFIG, rule, report_gram=gram,
            checkpoint_path=MODELS / "claude_check_down_T2_50_M3.pt",
            metadata={"check": "down_50_sweeps", "projection": "layers.14.mlp.down_proj", "budget": 50, "initial": "M3"},
            progress=progress_printer("down"),
        )
    rows = sweep_rows("down_50_M3", result)
    write_csv(RESULTS / "claude_check_down_history.csv", rows)
    seconds = [record.seconds for record in result.history]
    return {
        "stop_reason": result.stop_reason,
        "sweeps": len(result.history),
        "initial_q_lambda": result.initial_q_lambda,
        "q_lambda_at_50": result.history[-1].q_lambda if result.history else None,
        "relative_drop_0_to_50": (result.initial_q_lambda - result.history[-1].q_lambda) / result.initial_q_lambda,
        "d_ref": rule.d_ref,
        "delta_50_at_50": result.history[-1].delta_50 if result.history else None,
        "seconds_per_sweep_median": sorted(seconds)[len(seconds) // 2] if seconds else None,
        "seconds_per_sweep_last10_mean": sum(seconds[-10:]) / len(seconds[-10:]) if seconds else None,
        "seconds_total": result.seconds_total,
        "n_stage2": result.n_stage2,
        "n_stage3": result.n_stage3,
        "n_warnings": result.n_warnings,
        "failure": None if result.failure is None else asdict(result.failure),
        "escalated_updates": escalated_updates(result),
        "peak_memory_bytes_max": max((r.peak_memory_bytes or 0) for r in result.history) if result.history else None,
        "gpu_before": before,
        "gpu_utilization": sampler.summary(),
    }


def run_up() -> dict:
    payload = torch.load(PILOT_MODELS / "m3_T2_50.pt", weights_only=True)
    weight = torch.load(PILOT_DATA / "weights_float64.pt", weights_only=True)["W"]
    gram = torch.load(DATA04 / "calibration_grams.pt", weights_only=True)["up_proj_total"]
    rule = StudyResolutionStopRule(d_ref=float(d_refs()["d_ref_up"]), max_sweeps=50)
    before = gpu_snapshot()
    with GpuSampler() as sampler:
        result = run_metric_tt_als_study(
            payload["cores"], weight, regularize_gram(gram, LAMBDA), payload["out_modes"], payload["in_modes"],
            CONFIG, rule, report_gram=gram,
            checkpoint_path=MODELS / "claude_check_up_T2_50_M3.pt",
            metadata={"check": "up_50_sweeps_vs_03", "projection": "layers.14.mlp.up_proj", "budget": 50, "initial": "M3"},
            progress=progress_printer("up"),
        )
    write_csv(RESULTS / "claude_check_up_history.csv", sweep_rows("up_50_M3", result))
    reference: dict[int, float] = {}
    with (R03 / "history.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["budget"] == "50" and row["initial"] == "M3" and int(row["sweep"]) <= 50:
                reference[int(row["sweep"])] = float(row["q_lambda"])
    diffs = [
        abs(record.q_lambda - reference[record.sweep]) / reference[record.sweep]
        for record in result.history if record.sweep in reference
    ]
    equal = [record.q_lambda == reference[record.sweep] for record in result.history if record.sweep in reference]
    seconds = [record.seconds for record in result.history]
    return {
        "stop_reason": result.stop_reason,
        "sweeps": len(result.history),
        "q_lambda_at_50": result.history[-1].q_lambda if result.history else None,
        "q_lambda_03_at_50": reference.get(50),
        "max_relative_difference_vs_03": max(diffs) if diffs else None,
        "abs_difference_at_50_over_d_ref": (
            abs(result.history[-1].q_lambda - reference[50]) / rule.d_ref if result.history and 50 in reference else None
        ),
        "n_sweeps_bitwise_equal_vs_03": sum(equal),
        "n_sweeps_compared": len(equal),
        "seconds_per_sweep_median": sorted(seconds)[len(seconds) // 2] if seconds else None,
        "n_stage2": result.n_stage2,
        "n_stage3": result.n_stage3,
        "n_warnings": result.n_warnings,
        "escalated_updates": escalated_updates(result),
        "gpu_before": before,
        "gpu_utilization": sampler.summary(),
    }


def run_g() -> dict:
    state = torch.load(M03 / "long_als_T2_50_M4_retruncated.pt", map_location="cpu", weights_only=True)
    cores = state["cores"]
    weight = torch.load(PILOT_DATA / "weights_float64.pt", weights_only=True)["W"]
    gram = torch.load(DATA04 / "calibration_grams.pt", weights_only=True)["up_proj_total"]
    rule = StudyResolutionStopRule(d_ref=float(d_refs()["d_ref_up"]), max_sweeps=5, warn_residual_tolerance=1e-7)
    before = gpu_snapshot()
    with GpuSampler() as sampler:
        result = run_metric_tt_als_study(
            cores, weight, regularize_gram(gram, LAMBDA), (2, 16, 96), (64, 8, 2),
            CONFIG, rule, report_gram=gram,
            checkpoint_path=MODELS / "claude_check_tau_warn_from_03_M4r_426_continue.pt",
            metadata={"check": "tau_warn_precheck_1", "start": "03 50% M4_retruncated sweep 426", "tau_warn": 1e-7},
            progress=progress_printer("g"),
        )
    failures_03 = json.loads((R03 / "failures.json").read_text(encoding="utf-8"))
    return {
        "stop_reason": result.stop_reason,
        "sweeps": len(result.history),
        "q_lambda_start_03_sweep_426": state["history"][-1]["q_lambda"] if state.get("history") else None,
        "q_lambda_each_sweep": [record.q_lambda for record in result.history],
        "seconds_each_sweep": [record.seconds for record in result.history],
        "n_updates_each_sweep": [len(record.updates) for record in result.history],
        "n_stage2": result.n_stage2,
        "n_stage3": result.n_stage3,
        "n_warnings": result.n_warnings,
        "escalated_updates": escalated_updates(result),
        "failure": None if result.failure is None else asdict(result.failure),
        "failure_03": failures_03,
        "gpu_before": before,
        "gpu_utilization": sampler.summary(),
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out_path = RESULTS / "claude_checks.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results["config"] = {key: (str(value) if key == "device" else value) for key, value in asdict(CONFIG).items()}
    results["torch"] = torch.__version__
    for name, function in (("down", run_down), ("up", run_up), ("g", run_g), ("g_continue", run_g)):
        if which == name or (which == "all" and name != "g_continue"):
            print(f"=== {name} start {time.strftime('%H:%M:%S')}", flush=True)
            results[name] = function()
            out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            print(f"=== {name} done {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
