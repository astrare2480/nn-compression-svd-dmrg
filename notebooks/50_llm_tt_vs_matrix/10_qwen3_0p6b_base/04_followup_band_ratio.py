"""04 の追加分析 4：固有方向の帯ごとに、M5 が M2 の何倍の誤差を残したか。

分析 1（04_followup_eigen_directions.py）で「M5 が M2 より余計に残した誤差の 68% は 257 位以降」と分かった。
それが「257 位以降で M5 が M2 の何倍も落とす」からなのか、「257 位以降はもともと誤差が大きい帯だから」なのかを分けるため、
帯ごとに M5 と M2 の誤差の比を直接測る。主な 34 セルが対象。

    比（較正の重み）= Σ_{j∈帯} λ_j ||R_M5 u_j||^2 / Σ_{j∈帯} λ_j ||R_M2 u_j||^2
    比（重みなし）  = Σ_{j∈帯} ||R_M5 u_j||^2 / Σ_{j∈帯} ||R_M2 u_j||^2

R = W − Ŵ、u_j と λ_j は較正 Gram 行列の固有ベクトルと固有値（大きい順）。
あわせて、各帯に較正データの入力の量（Σ λ_j）がどれだけ乗っているかの割合も出す。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_band_ratio.py
結果: results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/band_ratio/
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "band_ratio"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from nn_compression.compression.metric_low_rank import regularize_gram, whitened_truncated_svd  # noqa: E402
from nn_compression.llm.qwen3_main_experiment import INPUT_GROUPS, layout_from_root  # noqa: E402
from nn_compression.tt_mps.matrix import tt_matrix_to_dense  # noqa: E402

torch.set_grad_enabled(False)
layout = layout_from_root(ROOT)
cells = {r["cell"]: r for r in csv.DictReader(open(layout.results / "evaluation" / "cells.csv", encoding="utf-8"))}
BANDS = [(0, 16), (16, 64), (64, 256), (256, None)]
NAMES = ["1-16", "17-64", "65-256", "257-"]

rows = []
energy_rows = []
for layer in (7, 14, 21):
    cal = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=False)
    eig = {}
    for cell, info in sorted(cells.items()):
        if int(info["layer"]) != layer or info["main_set"] not in ("True", "true", "1"):
            continue
        group = INPUT_GROUPS[info["projection"]]
        G = cal[group]["total"]
        if group not in eig:
            values, vectors = torch.linalg.eigh(G)
            order = torch.argsort(values, descending=True)
            eig[group] = (values[order].clamp_min(0.0), vectors[:, order])
            lam = eig[group][0]
            energy_rows.append({"layer": layer, "group": group, "dim": int(lam.numel()),
                                **{f"input_share_{n}": float(lam[a:b].sum() / lam.sum()) for (a, b), n in zip(BANDS, NAMES)}})
        lam, U = eig[group]
        payload = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
        W = torch.load(layout.weights / f"W_L{layer}_{info['projection']}.pt", map_location="cpu", weights_only=True)["W"]
        M2 = whitened_truncated_svd(W, regularize_gram(G, 1e-6), int(payload["matrix_rank"])).weight
        saved = torch.load(layout.als_dir("full") / f"{cell}_M5_M3init_lam1e-06.pt", map_location="cpu", weights_only=False)
        M5 = tt_matrix_to_dense(saved["cores"], payload["out_modes"], payload["in_modes"])
        e2 = torch.sum(((W - M2) @ U) ** 2, dim=0)
        e5 = torch.sum(((W - M5) @ U) ** 2, dim=0)
        row = {"cell": cell, "layer": layer, "projection": info["projection"], "budget": int(info["budget_percent"])}
        for (a, b), n in zip(BANDS, NAMES):
            row[f"ratio_weighted_{n}"] = float((lam[a:b] * e5[a:b]).sum() / (lam[a:b] * e2[a:b]).sum())
            row[f"ratio_unweighted_{n}"] = float(e5[a:b].sum() / e2[a:b].sum())
        rows.append(row)
        print(cell, flush=True)

for path, data in ((OUT / "band_ratio.csv", rows), (OUT / "input_share.csv", energy_rows)):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

print("\n帯ごとの M5 ÷ M2（主な 34 セルの中央値 [最小, 最大]）")
for kind in ("weighted", "unweighted"):
    print(" ", kind)
    for n in NAMES:
        values = [r[f"ratio_{kind}_{n}"] for r in rows]
        print(f"    {n:7s} {statistics.median(values):6.2f}  [{min(values):.2f}, {max(values):.2f}]")
