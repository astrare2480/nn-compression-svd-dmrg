"""04 の追加分析 1：較正 Gram 行列の固有方向ごとに、M2 と M5 が残した誤差を測る。

各セルで残差 R = W − Ŵ（Ŵ は M2 または M5）を作り、較正 Gram 行列 G_cal の固有ベクトル u_j
（固有値の大きい順）ごとに ||R u_j||^2 を求める。
その向きの重みとして、較正・WikiText-2・C4 の Gram 行列の対角成分 u_j^T G u_j を使う。

- 較正では G_cal は u_j で対角なので、Σ_j λ_j ||R u_j||^2 は Q（正則化なし）に一致する。
- 検証の Gram 行列は u_j で対角ではないので、Σ_j (u_j^T G u_j) ||R u_j||^2 は近似。
  近似が正確な値の何割を捉えるかも出す。

出力：セルごとの CSV（帯ごとの値）と、検算（e が cells.csv と一致するか）。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_eigen_directions.py
結果: results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "eigen_directions"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from nn_compression.compression.metric_low_rank import regularize_gram, whitened_truncated_svd  # noqa: E402
from nn_compression.llm.qwen3_main_experiment import INPUT_GROUPS, layout_from_root  # noqa: E402
from nn_compression.tt_mps.matrix import tt_matrix_to_dense  # noqa: E402

torch.set_grad_enabled(False)
layout = layout_from_root(ROOT)
cells_csv = {r["cell"]: r for r in csv.DictReader(open(layout.results / "evaluation" / "cells.csv", encoding="utf-8"))}

BANDS = [(0, 16), (16, 64), (64, 256), (256, None)]
BAND_NAMES = ["1-16", "17-64", "65-256", "257-"]


def load_grams(layer: int):
    cal = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=False)
    val = {d: torch.load(layout.validation / f"grams_{d}_L{layer}.pt", map_location="cpu", weights_only=False) for d in ("wt2", "c4")}
    return cal, val


def quad(R: torch.Tensor, G: torch.Tensor) -> float:
    return float(torch.sum((R @ G) * R))


rows = []
checks = []
for layer in (7, 14, 21):
    cal, val = load_grams(layer)
    eig_cache = {}
    for cell, info in sorted(cells_csv.items()):
        if int(info["layer"]) != layer:
            continue
        projection, budget = info["projection"], int(info["budget_percent"])
        group = INPUT_GROUPS[projection]
        G_cal = cal[group]["total"]
        if group not in eig_cache:
            values, vectors = torch.linalg.eigh(G_cal)
            order = torch.argsort(values, descending=True)
            eig_cache[group] = (values[order].clamp_min(0.0), vectors[:, order])
        lam, U = eig_cache[group]
        payload = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
        W = torch.load(layout.weights / f"W_L{layer}_{projection}.pt", map_location="cpu", weights_only=True)["W"]
        out_modes, in_modes = payload["out_modes"], payload["in_modes"]
        M2 = whitened_truncated_svd(W, regularize_gram(G_cal, 1e-6), int(payload["matrix_rank"])).weight
        M3 = tt_matrix_to_dense(payload["m3_cores"], out_modes, in_modes)
        saved = torch.load(layout.als_dir("full") / f"{cell}_M5_M3init_lam1e-06.pt", map_location="cpu", weights_only=False)
        M5 = tt_matrix_to_dense(saved["cores"], out_modes, in_modes)
        R = {"M2": W - M2, "M3": W - M3, "M5": W - M5}
        # 向きごとの残差エネルギー ||R u_j||^2
        energy = {m: torch.sum((r @ U) ** 2, dim=0) for m, r in R.items()}
        weights = {"cal": lam}
        exact_e = {}
        for dom in ("wt2", "c4"):
            G = val[dom][group]["total"]
            weights[dom] = torch.einsum("ij,ik,kj->j", U, G, U)  # u_j^T G u_j
            base = quad(W, G)
            exact_e[dom] = {m: quad(r, G) / base for m, r in R.items()}
            for m in ("M2", "M3", "M5"):
                ref = float(info[f"e_{m}_{dom}"])
                checks.append({"cell": cell, "domain": dom, "method": m, "ours": exact_e[dom][m], "cells_csv": ref,
                               "rel_gap": abs(exact_e[dom][m] - ref) / ref})
        base_cal = quad(W, G_cal)
        for src in ("cal", "wt2", "c4"):
            w = weights[src]
            base = base_cal if src == "cal" else quad(W, val[src][group]["total"])
            excess = (energy["M5"] - energy["M2"]) * w / base          # M5 が M2 より余計に残した誤差（向きごと）
            gap_m3 = (energy["M3"] - energy["M2"]) * w / base          # 分母（溝）の向きごと
            row = {"cell": cell, "layer": layer, "projection": projection, "budget": budget, "main_set": info["main_set"],
                   "weighting": src,
                   "excess_total_diag": float(excess.sum()), "gap_total_diag": float(gap_m3.sum())}
            if src != "cal":
                row["excess_total_exact"] = exact_e[src]["M5"] - exact_e[src]["M2"]
                row["gap_total_exact"] = exact_e[src]["M3"] - exact_e[src]["M2"]
            else:
                row["excess_total_exact"] = (quad(R["M5"], G_cal) - quad(R["M2"], G_cal)) / base_cal
                row["gap_total_exact"] = (quad(R["M3"], G_cal) - quad(R["M2"], G_cal)) / base_cal
            for (a, b), name in zip(BANDS, BAND_NAMES):
                row[f"excess_{name}"] = float(excess[a:b].sum())
                row[f"gap_{name}"] = float(gap_m3[a:b].sum())
            # 向きの重みの比（検証 ÷ 較正、トークンあたり）を帯ごとに（中央値）
            if src != "cal":
                n_cal = cal[group]["n_tokens"]
                n_val = val[src][group]["n_tokens"]
                ratio = (weights[src] / n_val) / (lam / n_cal).clamp_min(1e-300)
                for (a, b), name in zip(BANDS, BAND_NAMES):
                    row[f"weight_ratio_{name}"] = float(ratio[a:b].median())
            rows.append(row)
        print(cell, "done", flush=True)

fields = list(dict.fromkeys(k for r in rows for k in r))  # 最初に出た順（毎回同じ）
with open(OUT / "eigen_bands.csv", "w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
worst = max(c["rel_gap"] for c in checks)
json.dump({"max_rel_gap_vs_cells_csv": worst, "n_checks": len(checks)}, open(OUT / "checks.json", "w"), indent=2)
print("max rel gap vs cells.csv:", worst)
