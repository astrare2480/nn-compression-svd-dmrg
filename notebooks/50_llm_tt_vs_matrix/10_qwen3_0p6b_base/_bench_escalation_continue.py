"""解き直しを前の段から続けたときの、1 掃引だけの確認。依頼 01c。"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import subprocess
from pathlib import Path

import torch

from nn_compression.compression.metric_low_rank import regularize_gram
from nn_compression.tt_mps.metric_tt_solver import (
    MetricTTSolverConfig,
    StudyResolutionStopRule,
    run_metric_tt_als_study,
)
from nn_compression.utils.artifacts import atomic_write_json
from nn_compression.utils.paths import find_project_root


def _nvidia_smi() -> dict[str, str]:
    queries = {
        "gpu": ["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        "apps": ["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    }
    recorded: dict[str, str] = {}
    for name, args in queries.items():
        completed = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False)
        recorded[name] = completed.stdout.strip()
        if completed.returncode != 0:
            recorded[f"{name}_error"] = completed.stderr.strip()
    return recorded


def main() -> None:
    root = find_project_root(Path(__file__))
    rel = Path("50_llm_tt_vs_matrix") / "10_qwen3_0p6b_base"
    gpu_before = _nvidia_smi()
    als = torch.load(
        root / "models" / rel / "03_long_als_convergence" / "long_als_T2_50_M4_retruncated.pt",
        map_location="cpu",
        weights_only=True,
    )
    weights = torch.load(
        root / "data" / rel / "01_single_layer_pilot" / "weights_float64.pt",
        map_location="cpu",
        weights_only=True,
    )
    grams = torch.load(
        root / "data" / rel / "04_main_experiment" / "calibration_grams.pt",
        map_location="cpu",
        weights_only=True,
    )
    checks = json.loads(
        (root / "results" / rel / "04_main_experiment" / "solver_dev" / "data_checks.json").read_text(
            encoding="utf-8"
        )
    )
    out_modes = tuple(int(value) for value in als["input_hashes"]["out_modes"])
    in_modes = tuple(int(value) for value in als["input_hashes"]["in_modes"])
    config = MetricTTSolverConfig(
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
    rule = StudyResolutionStopRule(
        d_ref=float(checks["d_ref_up"]),
        max_sweeps=1,
        warn_residual_tolerance=1e-7,
    )
    result = run_metric_tt_als_study(
        als["cores"],
        weights["W"],
        regularize_gram(grams["up_proj_total"], 1e-6),
        out_modes,
        in_modes,
        config,
        rule,
    )
    sweep = result.history[0]
    middle = []
    for update in sweep.updates:
        if update.site != 1:
            continue
        middle.append(
            {
                "outcome": update.outcome,
                "accepted": update.accepted,
                "confirmation_a": update.confirmation_a,
                "stages": [
                    {
                        "stage": stage.stage,
                        "initial_residual": stage.initial_residual,
                        "residual": stage.residual,
                        "refinements": stage.refinements,
                        "cumulative_refinements": stage.cumulative_refinements,
                        "max_refinements": stage.max_refinements,
                        "seconds": stage.seconds,
                        "cg_iterations": stage.cg_iterations,
                    }
                    for stage in update.stages
                ],
            }
        )
    payload = {
        "gpu_before": gpu_before,
        "stop_reason": result.stop_reason,
        "q_lambda": sweep.q_lambda,
        "sweep_seconds": sweep.seconds,
        "n_stage2": sweep.n_stage2,
        "n_stage3": sweep.n_stage3,
        "n_warnings": sweep.n_warnings,
        "middle_updates": middle,
        "previous_restart": {
            "note": "Claude が最初から解き直した 1 掃引目",
            "stage1_residuals": [1.148e-7, 1.322e-7],
            "stage3_residuals": [6.4e-9, 8.1e-9],
            "stage3_refinements": [12, 11],
            "sweep_seconds": 26.3,
        },
    }
    atomic_write_json(
        root / "results" / rel / "04_main_experiment" / "solver_dev" / "escalation_continue_check.json",
        payload,
    )


if __name__ == "__main__":
    main()
