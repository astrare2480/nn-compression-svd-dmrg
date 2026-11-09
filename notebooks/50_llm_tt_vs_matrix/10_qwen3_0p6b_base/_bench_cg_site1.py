"""up_proj サイト 1 の CG を 4 通りだけ測る。依頼 01b。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import subprocess
import time
from pathlib import Path

import torch

from nn_compression.compression.metric_low_rank import regularize_gram
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt_solver import MetricTTSolverConfig, solve_metric_tt_site_on_device
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
        root / "models" / rel / "03_long_als_convergence" / "long_als_T2_50_M3.pt",
        map_location="cpu",
        weights_only=True,
    )
    modes = torch.load(
        root / "models" / rel / "01_single_layer_pilot" / "m3_T2_50.pt",
        map_location="cpu",
        weights_only=True,
    )
    weights = torch.load(
        root / "data" / rel / "01_single_layer_pilot" / "weights_float64.pt",
        map_location="cpu",
        weights_only=True,
    )
    # per_document 付きの 1GB は読まない。04 の up_proj_total は pilot と相対 1e-15 以下で一致済み。
    grams = torch.load(
        root / "data" / rel / "04_main_experiment" / "calibration_grams.pt",
        map_location="cpu",
        weights_only=True,
    )
    out_modes = tuple(int(value) for value in modes["out_modes"])
    in_modes = tuple(int(value) for value in modes["in_modes"])
    cores = canonicalize_tt_matrix_cores(als["cores"], out_modes, in_modes, 1)
    weight = weights["W"]
    gram = regularize_gram(grams["up_proj_total"], 1e-6)
    cases = (
        {"cg_check_every": 1, "timing_sync": True},
        {"cg_check_every": 1, "timing_sync": False},
        {"cg_check_every": 10, "timing_sync": True},
        {"cg_check_every": 10, "timing_sync": False},
    )
    measured = []
    for spec in cases:
        config = MetricTTSolverConfig(
            device="cuda",
            precision="float32_refine",
            preconditioner="jacobi",
            cg_tolerance_mode="fixed",
            cg_relative_tolerance=1e-8,
            max_cg_iterations=500,
            max_refinements=5,
            cg_check_every=spec["cg_check_every"],
            timing_sync=spec["timing_sync"],
        )
        solve_metric_tt_site_on_device(cores, 1, weight, gram, out_modes, in_modes, config)
        repeats = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            record = solve_metric_tt_site_on_device(cores, 1, weight, gram, out_modes, in_modes, config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            per_iteration = None if record.iterations == 0 else record.seconds / record.iterations
            repeats.append(
                {
                    "seconds": record.seconds,
                    "wall_seconds": elapsed,
                    "cg_iterations": record.iterations,
                    "refinements": record.refinements,
                    "final_relative_residual": record.final_relative_residual,
                    "seconds_per_cg_iteration": per_iteration,
                }
            )
        measured.append({**spec, "repeats": repeats})
    destination = (
        root / "results" / rel / "04_main_experiment" / "solver_dev" / "bench_cg_site1.json"
    )
    atomic_write_json(destination, {"gpu_before": gpu_before, "cases": measured})


if __name__ == "__main__":
    main()
