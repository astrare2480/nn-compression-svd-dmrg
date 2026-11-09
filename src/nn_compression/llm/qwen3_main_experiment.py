"""Qwen3 本実験のデータの準備、方式、ALS の仕事一覧。

ソルバーの計算は呼ばない。ALS は ``run_metric_tt_als_study`` に渡すだけ。
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nn_compression.compression.metric_low_rank import (
    matrix_factor_parameter_count,
    max_matrix_rank_within_budget,
    regularize_gram,
    squared_output_error_from_gram,
    truncated_svd_weight,
    whitened_truncated_svd,
)
from nn_compression.tt_mps.matrix import (
    dense_to_tt_matrix_cores,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from nn_compression.tt_mps.metric_tt import (
    fit_diagonal_metric_tt,
    prepare_m5_from_diagonal_tt,
    select_balanced_tt_matrix_modes,
    select_uniform_tt_rank,
    structural_tt_matrix_ranks,
    tt_parameter_count,
)
from nn_compression.tt_mps.metric_tt_solver import (
    MetricTTSolverConfig,
    StudyResolutionStopRule,
    run_metric_tt_als_study,
)
from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_csv, atomic_write_json, file_sha256
from nn_compression.utils.paths import find_project_root, get_experiment_dirs

LAYERS = (14, 7, 21)
PROJECTION_ORDER = ("down", "gate", "up", "q", "k", "v", "o")
BUDGETS = (50, 25)
LAMBDA_MAIN = 1e-6
LAMBDA_SWEEP = (1e-7, 1e-6, 1e-5)
M4R_PROJECTIONS = ("down", "gate", "q", "k", "v", "o")
LAMBDA_PROJECTIONS = ("down", "gate", "q", "o")
INPUT_GROUPS = {
    "q": "attn_in",
    "k": "attn_in",
    "v": "attn_in",
    "o": "o_in",
    "gate": "mlp_in",
    "up": "mlp_in",
    "down": "down_in",
}
SHAPES = {
    "q": (2048, 1024),
    "k": (1024, 1024),
    "v": (1024, 1024),
    "o": (1024, 2048),
    "gate": (3072, 1024),
    "up": (3072, 1024),
    "down": (1024, 3072),
}
GROUP_MODULES = {
    "attn_in": "self_attn.q_proj",
    "o_in": "self_attn.o_proj",
    "mlp_in": "mlp.up_proj",
    "down_in": "mlp.down_proj",
}
EXPECTED_MODES_AND_RANKS = {
    ("q", 50): ((2, 16, 64), (64, 8, 2), (1, 89, 89, 1)),
    ("q", 25): ((2, 16, 64), (64, 8, 2), (1, 63, 63, 1)),
    ("k", 50): ((2, 8, 64), (32, 16, 2), (1, 63, 63, 1)),
    ("k", 25): ((2, 8, 64), (32, 16, 2), (1, 44, 44, 1)),
    ("v", 50): ((2, 8, 64), (32, 16, 2), (1, 63, 63, 1)),
    ("v", 25): ((2, 8, 64), (32, 16, 2), (1, 44, 44, 1)),
    ("o", 50): ((2, 8, 64), (64, 16, 2), (1, 89, 89, 1)),
    ("o", 25): ((2, 8, 64), (64, 16, 2), (1, 63, 63, 1)),
    ("gate", 50): ((2, 16, 96), (64, 8, 2), (1, 109, 109, 1)),
    ("gate", 25): ((2, 16, 96), (64, 8, 2), (1, 77, 77, 1)),
    ("up", 50): ((2, 16, 96), (64, 8, 2), (1, 109, 109, 1)),
    ("up", 25): ((2, 16, 96), (64, 8, 2), (1, 77, 77, 1)),
    ("down", 50): ((2, 8, 64), (64, 16, 3), (1, 109, 109, 1)),
    ("down", 25): ((2, 8, 64), (64, 16, 3), (1, 77, 77, 1)),
}
PARALLEL_SLOWDOWN = 1.27
EXPECTED_SWEEPS = {"M5": 400, "M3F": 400, "M5_lambda": 400, "M5_M4r": 1000}
NLL_HOURS = 4.0
PYTHIA_HOURS = 4.0
SMOKE_TAU_WARN = 1e-7

SUMMARY_FIELDS = (
    "layer",
    "projection",
    "budget_percent",
    "group",
    "out_modes",
    "in_modes",
    "ranks",
    "bond_ratios",
    "bond_class",
    "parameters_m1",
    "parameters_m2",
    "parameters_m3",
    "parameters_m4",
    "parameters_m4r",
    "q_m1",
    "q_m1_unregularized",
    "q_m3",
    "q_m3_unregularized",
    "q_m4",
    "q_m4_unregularized",
    "q_m4r",
    "q_m4r_unregularized",
    "q_m3_lam1e-07",
    "q_m3_lam1e-05",
    "f_m1",
    "f_m3",
    "d_f_ref",
    "small_baseline_gap_f",
    "q_m2_lam1e-07",
    "q_m2_unregularized_lam1e-07",
    "d_ref_lam1e-07",
    "small_baseline_gap_lam1e-07",
    "q_m2_lam1e-06",
    "q_m2_unregularized_lam1e-06",
    "d_ref_lam1e-06",
    "small_baseline_gap_lam1e-06",
    "q_m2_lam1e-05",
    "q_m2_unregularized_lam1e-05",
    "d_ref_lam1e-05",
    "small_baseline_gap_lam1e-05",
    "seconds",
)

STATUS_FIELDS = (
    "job_id",
    "kind",
    "layer",
    "projection",
    "budget_percent",
    "stop_reason",
    "sweeps",
    "q_lambda",
    "delta_50",
    "s_max",
    "n_warnings",
    "n_stage2",
    "n_stage3",
    "seconds",
    "peak_memory_bytes",
    "check_ok",
    "check_relative_gap",
    "check_detail",
)

HISTORY_FIELDS = (
    "sweep",
    "q_lambda",
    "q_unregularized",
    "relative_decrease",
    "delta_50",
    "consecutive_small_windows",
    "s_max",
    "stationarity",
    "n_updates",
    "n_updates_stage2",
    "n_updates_stage3",
    "n_warnings_sweep",
    "n_warnings",
    "n_stage2",
    "n_stage3",
    "stage3_residuals",
    "hvp_calls",
    "n_rejected",
    "seconds",
    "peak_memory_bytes",
    "stop_reason",
    "e_wt2",
    "e_c4",
)


def experiment_dirs(project_root: Path | str) -> tuple[Path, Path, Path]:
    """本実験の data・models・results を返す。

    ``get_experiment_dirs`` の data はプロジェクトの ``data`` 直下なので、
    実験名のディレクトリはここで作る。
    """
    data_root, models, results = get_experiment_dirs(
        project_root,
        "50_llm_tt_vs_matrix",
        "10_qwen3_0p6b_base",
        "04_main_experiment",
    )
    data = data_root / "50_llm_tt_vs_matrix" / "10_qwen3_0p6b_base" / "04_main_experiment"
    data.mkdir(parents=True, exist_ok=True)
    return data, models, results


def format_lambda(value: float) -> str:
    """``1e-06`` のような仕事 id 用の表記。"""
    return f"{value:.0e}"


def chunk_token_spans(n_tokens: int, length: int = 1024) -> list[tuple[int, int]]:
    """重なりなしで ``length`` ずつ切る。最後の端は短くてよい。"""
    if length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {length}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens は 0 以上である必要があります: {n_tokens}")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + length, n_tokens)
        spans.append((start, end))
        start = end
    return spans


def classify_bonds(ranks: Sequence[int], structural: Sequence[int]) -> dict[str, object]:
    """内部結合の r / r_max。どれかが 0.9 を超えたら ``near_saturated_bond``。"""
    if len(ranks) != len(structural) or len(ranks) < 3:
        raise ValueError(f"rank 列の長さが違います: {ranks} / {structural}")
    ratios = [float(rank) / float(limit) for rank, limit in zip(ranks[1:-1], structural[1:-1], strict=True)]
    kind = "near_saturated_bond" if any(ratio > 0.9 for ratio in ratios) else "multi_cut"
    return {"ratios": ratios, "bond_class": kind}


def reference_denominator(primary: float, baseline: float) -> tuple[float, bool]:
    """D_ref と、差が primary の 5% 未満かどうか。"""
    gap = primary - baseline
    small = gap < 0.05 * primary
    return max(gap, 0.05 * primary), small


def build_job_list() -> list[dict[str, Any]]:
    """設計の順番で 112 本の仕事を返す。"""
    jobs: list[dict[str, Any]] = []
    for layer in LAYERS:
        for kind in ("M5", "M3F", "M5_M4r", "M5_lambda"):
            projections = PROJECTION_ORDER
            lambdas: tuple[float, ...] = (LAMBDA_MAIN,)
            init = "M3"
            if kind == "M5_M4r":
                if layer != 14:
                    continue
                projections = M4R_PROJECTIONS
                init = "M4r"
            elif kind == "M5_lambda":
                if layer != 14:
                    continue
                projections = LAMBDA_PROJECTIONS
                lambdas = (1e-7, 1e-5)
            elif kind == "M3F":
                init = "M3"
            for projection in projections:
                for budget in BUDGETS:
                    for lam in lambdas:
                        jobs.append(
                            {
                                "id": (
                                    f"L{layer}_{projection}_{budget}_{kind}_{init}init_lam{format_lambda(lam)}"
                                ),
                                "layer": layer,
                                "projection": projection,
                                "budget_percent": budget,
                                "kind": kind,
                                "init": init,
                                "lambda": lam,
                                "group": INPUT_GROUPS[projection],
                            }
                        )
    return jobs


def jobs_for_mode(jobs: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    """``smoke`` は layer 14 の M5・M4r 全部と、M3F・λ を 1 本ずつ。"""
    if mode in {"full", "check"}:
        return [dict(job) for job in jobs]
    if mode != "smoke":
        raise ValueError(f"mode は smoke / full / check です: {mode}")
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for job in jobs:
        if int(job["layer"]) != 14:
            continue
        kind = str(job["kind"])
        if kind in {"M5", "M5_M4r"}:
            selected.append(dict(job))
        elif kind in {"M3F", "M5_lambda"} and kind not in seen:
            seen.add(kind)
            selected.append(dict(job))
    return selected


def filter_jobs(jobs: Sequence[Mapping[str, Any]], only: str | None) -> list[dict[str, Any]]:
    if not only:
        return [dict(job) for job in jobs]
    return [dict(job) for job in jobs if only in str(job["id"])]


@dataclass(frozen=True)
class ExperimentLayout:
    data: Path
    models: Path
    results: Path

    @property
    def calibration(self) -> Path:
        return self.data / "calibration"

    @property
    def validation(self) -> Path:
        return self.data / "validation"

    @property
    def weights(self) -> Path:
        return self.data / "weights"

    @property
    def methods(self) -> Path:
        return self.models / "methods"

    def als_dir(self, mode: str) -> Path:
        return self.models / "als" / mode

    def als_results(self, mode: str) -> Path:
        return self.results / "als" / mode

    @property
    def settings(self) -> Path:
        return self.results / "settings" / "tau_warn.json"

    @property
    def manifest(self) -> Path:
        return self.calibration / "manifest.json"


def layout_from_root(project_root: Path | str) -> ExperimentLayout:
    data, models, results = experiment_dirs(project_root)
    return ExperimentLayout(data, models, results)


def read_tau_warn(path: Path, *, mode: str) -> float:
    """設定ファイルの τ_warn。smoke / check でファイルが無いときだけ仮の 1e-7。"""
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["tau_warn"])
    if mode in {"smoke", "check"}:
        return SMOKE_TAU_WARN
    raise FileNotFoundError(f"τ_warn の設定がありません: {path}")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def clear_stale_locks(lock_dir: Path) -> list[str]:
    """終わったプロセスが残したロックを消す。消したファイル名を返す。"""
    removed: list[str] = []
    if not lock_dir.is_dir():
        return removed
    for path in lock_dir.glob("*.lock"):
        try:
            pid = int(path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            pid = 0
        if not pid_alive(pid):
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def try_claim(lock_path: Path, owner: str) -> bool:
    """排他的にロックを作れたら True。中に所有者を書く。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(owner)
    return True


@contextmanager
def exclusive_lock(lock_path: Path):
    """共有ファイルを更新するあいだのロック。"""
    while True:
        if lock_path.exists():
            try:
                owner = int(lock_path.read_text(encoding="utf-8").strip() or "0")
            except ValueError:
                owner = 0
            if not pid_alive(owner):
                lock_path.unlink(missing_ok=True)
        if try_claim(lock_path, str(os.getpid())):
            break
        time.sleep(0.05)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def claim_next_job(
    jobs: Sequence[Mapping[str, Any]],
    lock_dir: Path,
    done_ids: set[str],
    owner: str,
) -> dict[str, Any] | None:
    """まだ終わっておらず、取られていない最初の仕事を取る。"""
    for job in jobs:
        job_id = str(job["id"])
        if job_id in done_ids:
            continue
        if try_claim(lock_dir / f"{job_id}.lock", owner):
            return dict(job)
    return None


def _squared_frobenius(delta: torch.Tensor) -> float:
    return float(torch.sum(delta * delta).item())


def _relative_error(left: float, right: float) -> float:
    scale = max(abs(right), 1e-30)
    return abs(left - right) / scale


def build_cell_methods(
    weight: torch.Tensor,
    gram: torch.Tensor,
    n_tokens: int,
    *,
    layer: int,
    projection: str,
    budget_percent: int,
    check_design: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """1 セルの M1〜M4r を作り、保存用の辞書と要約の 1 行を返す。密行列は入れない。"""
    started = time.perf_counter()
    out_features, in_features = tuple(weight.shape)
    if check_design and (out_features, in_features) != SHAPES[projection]:
        raise ValueError(f"L{layer} {projection} の shape が違います: {tuple(weight.shape)}")
    budget = (out_features * in_features * budget_percent) // 100
    modes = select_balanced_tt_matrix_modes(out_features, in_features, 3)
    out_modes = tuple(int(mode) for mode in modes["out_modes"])
    in_modes = tuple(int(mode) for mode in modes["in_modes"])
    selected = select_uniform_tt_rank(out_modes, in_modes, budget)
    ranks = tuple(int(rank) for rank in selected["ranks"])
    if check_design:
        expected = EXPECTED_MODES_AND_RANKS[(projection, budget_percent)]
        if (out_modes, in_modes, ranks) != expected:
            raise ValueError(
                f"L{layer} {projection} {budget_percent}% の形が設計と違います: "
                f"{(out_modes, in_modes, ranks)} != {expected}"
            )
    structural = structural_tt_matrix_ranks(out_modes, in_modes)
    bonds = classify_bonds(ranks, structural)
    matrix_rank = max_matrix_rank_within_budget(out_features, in_features, budget)
    reproduced = tt_matrix_to_dense(
        dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=None),
        out_modes,
        in_modes,
    )
    reproduction = torch.linalg.matrix_norm(weight - reproduced) / torch.linalg.matrix_norm(weight)
    if float(reproduction) > 1e-10:
        raise ValueError(f"切り詰めなし TT-SVD の相対誤差が 1e-10 を超えます: {float(reproduction)}")

    u_factor, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    m1 = truncated_svd_weight(weight, matrix_rank)
    m3_cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=int(selected["rank_cap"]))
    if tt_matrix_ranks(m3_cores) != ranks:
        raise ValueError(f"M3 の rank が選択と違います: {tt_matrix_ranks(m3_cores)} != {ranks}")
    if tt_parameter_count(m3_cores) != tt_matrix_num_parameters(m3_cores):
        raise ValueError("TT のパラメータ数が式と一致しません。")
    m3 = tt_matrix_to_dense(m3_cores, out_modes, in_modes)
    diagonal = fit_diagonal_metric_tt(weight, gram, n_tokens, out_modes, in_modes, int(selected["rank_cap"]))
    m4 = tt_matrix_to_dense(diagonal.absorbed_cores, out_modes, in_modes)
    m4r_cores = prepare_m5_from_diagonal_tt(diagonal, out_modes, in_modes, ranks)
    m4r = tt_matrix_to_dense(m4r_cores, out_modes, in_modes)
    gram_main = regularize_gram(gram, LAMBDA_MAIN)
    row: dict[str, Any] = {
        "layer": layer,
        "projection": projection,
        "budget_percent": budget_percent,
        "group": INPUT_GROUPS[projection],
        "out_modes": list(out_modes),
        "in_modes": list(in_modes),
        "ranks": list(ranks),
        "bond_ratios": bonds["ratios"],
        "bond_class": bonds["bond_class"],
        "parameters_m1": matrix_factor_parameter_count(out_features, in_features, matrix_rank),
        "parameters_m2": matrix_factor_parameter_count(out_features, in_features, matrix_rank),
        "parameters_m3": tt_parameter_count(m3_cores),
        "parameters_m4": tt_parameter_count(diagonal.absorbed_cores) + int(diagonal.scales.numel()),
        "parameters_m4r": tt_parameter_count(m4r_cores),
        "q_m1": float(squared_output_error_from_gram(weight - m1, gram_main)),
        "q_m1_unregularized": float(squared_output_error_from_gram(weight - m1, gram)),
        "q_m3": float(squared_output_error_from_gram(weight - m3, gram_main)),
        "q_m3_unregularized": float(squared_output_error_from_gram(weight - m3, gram)),
        "q_m4": float(squared_output_error_from_gram(weight - m4, gram_main)),
        "q_m4_unregularized": float(squared_output_error_from_gram(weight - m4, gram)),
        "q_m4r": float(squared_output_error_from_gram(weight - m4r, gram_main)),
        "q_m4r_unregularized": float(squared_output_error_from_gram(weight - m4r, gram)),
        "f_m1": _squared_frobenius(weight - m1),
        "f_m3": _squared_frobenius(weight - m3),
    }
    d_f, small_f = reference_denominator(row["f_m3"], row["f_m1"])
    row["d_f_ref"] = d_f
    row["small_baseline_gap_f"] = small_f
    factors: dict[str, Any] = {}
    for lam in LAMBDA_SWEEP:
        gram_lambda = regularize_gram(gram, lam)
        fitted = whitened_truncated_svd(weight, gram_lambda, matrix_rank)
        q_value = float(fitted.q_lambda)
        discarded = float(fitted.discarded_singular_value_energy)
        if _relative_error(q_value, discarded) > 1e-12:
            raise ValueError(f"M2 の Q_λ と捨てた特異値エネルギーが相対 1e-12 を超えます: λ={lam}")
        label = format_lambda(lam)
        q_m3_lambda = float(squared_output_error_from_gram(weight - m3, gram_lambda))
        row[f"q_m2_lam{label}"] = q_value
        row[f"q_m2_unregularized_lam{label}"] = float(
            squared_output_error_from_gram(weight - fitted.weight, gram)
        )
        row[f"q_m3_lam{label}"] = q_m3_lambda
        d_ref, small = reference_denominator(q_m3_lambda, q_value)
        row[f"d_ref_lam{label}"] = d_ref
        row[f"small_baseline_gap_lam{label}"] = small
        transformed = weight @ _gram_sqrt(gram_lambda)
        u_m2, s_m2, vh_m2 = torch.linalg.svd(transformed, full_matrices=False)
        factors[f"m2_u_{label}"] = u_m2[:, :matrix_rank].contiguous()
        factors[f"m2_s_{label}"] = s_m2[:matrix_rank].contiguous()
        factors[f"m2_vh_{label}"] = vh_m2[:matrix_rank, :].contiguous()
    row["seconds"] = time.perf_counter() - started
    payload = {
        "layer": layer,
        "projection": projection,
        "budget_percent": budget_percent,
        "out_modes": out_modes,
        "in_modes": in_modes,
        "ranks": ranks,
        "matrix_rank": matrix_rank,
        "m1_u": u_factor[:, :matrix_rank].contiguous(),
        "m1_s": singular[:matrix_rank].contiguous(),
        "m1_vh": vh[:matrix_rank, :].contiguous(),
        "m3_cores": m3_cores,
        "m4_cores": diagonal.absorbed_cores,
        "m4_scales": diagonal.scales,
        "m4r_cores": m4r_cores,
        **factors,
    }
    return payload, row


def _gram_sqrt(gram_lambda: torch.Tensor) -> torch.Tensor:
    from nn_compression.compression.metric_low_rank import gram_sqrt_and_inv_sqrt

    factor, _inverse, _values = gram_sqrt_and_inv_sqrt(gram_lambda)
    return factor


def upsert_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str], key: str) -> None:
    """``key`` が同じ行を置き換えて、CSV を書き直す。"""
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing.extend(csv.DictReader(handle))
    by_key = {row[key]: row for row in existing}
    for row in rows:
        by_key[str(row[key])] = {name: row.get(name) for name in fieldnames}
    atomic_write_csv(path, list(by_key.values()), fieldnames)


def _cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['layer']}_{row['projection']}_{row['budget_percent']}"


def write_methods_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    prepared = []
    for row in rows:
        item = {name: row.get(name) for name in SUMMARY_FIELDS}
        item["cell"] = _cell_key(row)
        prepared.append(item)
    fields = ("cell", *SUMMARY_FIELDS)
    upsert_csv(path, prepared, fields, "cell")


def validation_relative_errors(
    weight: torch.Tensor,
    approximated: torch.Tensor,
    grams: Mapping[str, torch.Tensor],
) -> dict[str, float | None]:
    """検証 Gram の合計から e = Q / ||W G^{1/2}||^2 を返す。"""
    values: dict[str, float | None] = {}
    delta = weight - approximated
    for name, gram in grams.items():
        numerator = float(squared_output_error_from_gram(delta, gram))
        denominator = float(squared_output_error_from_gram(weight, gram))
        values[name] = None if denominator == 0.0 else numerator / denominator
    return values


def history_row(record: Any, errors: Mapping[str, float | None]) -> dict[str, Any]:
    stage2 = sum(len(update.stages) >= 2 for update in record.updates)
    stage3 = [update.stages[2].residual for update in record.updates if len(update.stages) >= 3]
    warnings = sum(update.outcome == "local_target_missed" for update in record.updates)
    return {
        "sweep": record.sweep,
        "q_lambda": record.q_lambda,
        "q_unregularized": record.q_unregularized,
        "relative_decrease": record.relative_decrease,
        "delta_50": record.delta_50,
        "consecutive_small_windows": record.consecutive_small_windows,
        "s_max": record.stationarity_max,
        "stationarity": ";".join(str(value) for value in record.stationarity),
        "n_updates": len(record.updates),
        "n_updates_stage2": stage2,
        "n_updates_stage3": len(stage3),
        "n_warnings_sweep": warnings,
        "n_warnings": record.n_warnings,
        "n_stage2": record.n_stage2,
        "n_stage3": record.n_stage3,
        "stage3_residuals": ";".join(str(value) for value in stage3),
        "hvp_calls": record.hvp_calls,
        "n_rejected": record.n_rejected,
        "seconds": record.seconds,
        "peak_memory_bytes": record.peak_memory_bytes,
        "stop_reason": record.stop_reason,
        "e_wt2": errors.get("e_wt2"),
        "e_c4": errors.get("e_c4"),
    }


def load_status_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finished_job_ids(status_path: Path) -> set[str]:
    """数値の失敗と、停止まで終わった仕事を飛ばす。中断は飛ばさない。"""
    done: set[str] = set()
    for row in load_status_rows(status_path):
        reason = row.get("stop_reason") or ""
        if reason and reason != "resource_interrupted":
            done.add(row["job_id"])
    return done


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def summarize_smoke(
    jobs: Sequence[Mapping[str, Any]],
    sweep_seconds: Mapping[str, Sequence[float]],
    update_counts: Mapping[str, Mapping[str, int]],
    stage3_residuals: Sequence[float],
    gpu_utilizations: Sequence[float],
    gpu_contended: bool,
) -> dict[str, Any]:
    """試運転の記録から、時間・解き直し・τ_warn・GPU をまとめる。"""
    by_kind_proj: dict[tuple[str, str, int], list[float]] = {}
    by_kind: dict[str, list[float]] = {}
    for job in jobs:
        samples = list(sweep_seconds.get(str(job["id"]), ()))
        if not samples:
            continue
        by_kind_proj.setdefault((str(job["kind"]), str(job["projection"]), int(job["budget_percent"])), []).extend(
            samples
        )
        by_kind.setdefault(str(job["kind"]), []).extend(samples)
    catalog = build_job_list()
    estimates = []
    missing = []
    expected_seconds = 0.0
    conservative_seconds = 0.0
    for job in catalog:
        key = (str(job["kind"]), str(job["projection"]), int(job["budget_percent"]))
        layer14 = (str(job["kind"]), str(job["projection"]), int(job["budget_percent"]))
        samples = by_kind_proj.get(layer14) or by_kind.get(str(job["kind"]), [])
        sweep_median = median(samples)
        if sweep_median is None:
            missing.append(job["id"])
            continue
        expected_n = EXPECTED_SWEEPS[str(job["kind"])]
        estimates.append(
            {
                "id": job["id"],
                "median_sweep_seconds": sweep_median,
                "expected_sweeps": expected_n,
                "expected_seconds": sweep_median * expected_n,
                "conservative_seconds": sweep_median * 1000,
            }
        )
        expected_seconds += sweep_median * expected_n
        conservative_seconds += sweep_median * 1000
    expected_hours = expected_seconds / PARALLEL_SLOWDOWN / 3600.0 + NLL_HOURS + PYTHIA_HOURS
    conservative_hours = conservative_seconds / PARALLEL_SLOWDOWN / 3600.0 + NLL_HOURS + PYTHIA_HOURS
    group_updates: dict[str, list[int]] = {}
    for job in jobs:
        counts = update_counts.get(str(job["id"]))
        if not counts:
            continue
        bucket = group_updates.setdefault(str(job["group"]), [0, 0])
        bucket[0] += int(counts.get("stage2", 0))
        bucket[1] += int(counts.get("updates", 0))
    groups = []
    for name, (stage2, updates) in sorted(group_updates.items()):
        ratio = None if updates == 0 else stage2 / updates
        groups.append(
            {
                "group": name,
                "stage2_updates": stage2,
                "updates": updates,
                "ratio": ratio,
                "above_10_percent": bool(ratio is not None and ratio > 0.10),
            }
        )
    finite_stage3 = [value for value in stage3_residuals if math.isfinite(value)]
    tau = "1e-8" if any(value > 1e-7 for value in finite_stage3) else "1e-7 の条件を満たす"
    if not finite_stage3:
        tau = "3 段目の更新がない"
    return {
        "estimates": estimates,
        "missing_timing": missing,
        "expected_hours": expected_hours,
        "conservative_hours": conservative_hours,
        "parallel_divisor": PARALLEL_SLOWDOWN,
        "nll_hours": NLL_HOURS,
        "pythia_hours": PYTHIA_HOURS,
        "exceeds_48h": expected_hours > 48.0,
        "escalation_by_group": groups,
        "tau_warn_precheck": tau,
        "stage3_residuals": list(finite_stage3),
        "gpu_utilization_mean": None if not gpu_utilizations else sum(gpu_utilizations) / len(gpu_utilizations),
        "gpu_contended": gpu_contended,
    }


def solver_config(device: str = "cuda") -> MetricTTSolverConfig:
    return MetricTTSolverConfig(
        device=device,
        precision="float32_refine",
        preconditioner="jacobi",
        cg_tolerance_mode="fixed",
        cg_relative_tolerance=1e-8,
        max_cg_iterations=500,
        max_refinements=5,
        cg_check_every=1,
        timing_sync=False,
    )


def study_rule(
    d_ref: float,
    *,
    max_sweeps: int,
    tau_warn: float,
    window_sweeps: int = 50,
) -> StudyResolutionStopRule:
    return StudyResolutionStopRule(
        d_ref=d_ref,
        window_sweeps=window_sweeps,
        resolution_tolerance=1e-4,
        consecutive_windows=2,
        max_sweeps=max_sweeps,
        local_residual_tolerance=1e-8,
        escalation_refinements=(10, 50),
        warn_residual_tolerance=tau_warn,
    )


def reconstruct_matrix(payload: Mapping[str, Any], init: str) -> torch.Tensor:
    out_modes = payload["out_modes"]
    in_modes = payload["in_modes"]
    if init == "M3":
        return tt_matrix_to_dense(payload["m3_cores"], out_modes, in_modes)
    if init == "M4r":
        return tt_matrix_to_dense(payload["m4r_cores"], out_modes, in_modes)
    raise ValueError(f"初期値が不明です: {init}")


def initial_cores(payload: Mapping[str, Any], init: str) -> list[torch.Tensor]:
    if init == "M3":
        return list(payload["m3_cores"])
    if init == "M4r":
        return list(payload["m4r_cores"])
    raise ValueError(f"初期値が不明です: {init}")


def check_finished_objective(
    *,
    kind: str,
    weight: torch.Tensor,
    gram: torch.Tensor,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    recorded_q: float,
    reference_q: float,
) -> dict[str, Any]:
    """最終コアから Q（M3F は F）を計算し直し、初期値の誤差以下かも見る。"""
    approximated = tt_matrix_to_dense(cores, out_modes, in_modes)
    if kind == "M3F":
        recomputed = _squared_frobenius(weight - approximated)
    else:
        recomputed = float(squared_output_error_from_gram(weight - approximated, gram))
    gap = _relative_error(recomputed, recorded_q)
    monotone = recomputed <= reference_q * (1.0 + 1e-8)
    return {
        "check_ok": gap <= 1e-10 and monotone,
        "check_relative_gap": gap,
        "check_detail": f"recomputed={recomputed}; reference={reference_q}; monotone={monotone}",
    }


def run_one_study(
    job: Mapping[str, Any],
    *,
    weight: torch.Tensor,
    gram: torch.Tensor,
    report_gram: torch.Tensor | None,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
    rule: StudyResolutionStopRule,
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    diagnostics: Callable[..., Mapping[str, Any]] | None = None,
    progress: Callable[..., None] | None = None,
) -> Any:
    return run_metric_tt_als_study(
        cores,
        weight,
        gram,
        out_modes,
        in_modes,
        config,
        rule,
        report_gram=report_gram,
        checkpoint_path=checkpoint_path,
        metadata=dict(metadata),
        diagnostics=diagnostics,
        progress=progress,
    )


def project_root_from(start: Path | None = None) -> Path:
    return find_project_root(start or Path(__file__))


def dump_jobs(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_json(path, {"jobs": list(jobs)})


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["jobs"])


def sha256_of(path: Path) -> str:
    return file_sha256(path) if path.is_file() else ""


def as_jsonable(record: Any) -> dict[str, Any]:
    return asdict(record)
