"""本実験の評価（設計 §7）。

GPU の計算（層ごとに保存。途中で止まっても、終わった層は飛ばして再開する）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py --stage gpu

集計（ブートストラップ・分類・W/B/F・表）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py --stage summarize

小さい確認（2 セル・数単位。結果は evaluation_smoke に書く）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py --stage gpu --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nn_compression.llm import qwen3_main_evaluation as ev
from nn_compression.llm import qwen3_main_statistics as st
from nn_compression.llm.hf_loading import load_causal_lm_and_tokenizer, load_download_record, load_wikitext
from nn_compression.llm.perplexity import sliding_window_perplexity
from nn_compression.llm.qwen3_main_experiment import (
    BUDGETS,
    GROUP_MODULES,
    INPUT_GROUPS,
    LAMBDA_PROJECTIONS,
    LAYERS,
    PROJECTION_ORDER,
    layout_from_root,
)
from nn_compression.llm.qwen3_main_run import (
    CALIBRATION_LIMIT,
    _encode,
    article_units,
    official_sequences,
)
from nn_compression.llm.text_documents import read_c4_texts, restore_wikitext_articles
from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_csv, atomic_write_json
from nn_compression.utils.paths import find_project_root

DEVICE = "cuda"
SMOKE_CELLS = ("L14_down_25", "L14_q_25", "L14_up_25")
SMOKE_UNITS = {"wt2": 3, "c4": 4}
FULL_CELL_COUNT = len(LAYERS) * len(PROJECTION_ORDER) * len(BUDGETS)  # 42
FINGERPRINT_NAME = "fingerprint.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_csv_map(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def stop_if_failed(checks: list[dict[str, Any]], where: str) -> None:
    failed = [item for item in checks if not item["ok"]]
    if failed:
        raise RuntimeError(f"{where} の検算に失敗しました: {failed[:5]}")


# ---------------------------------------------------------------- 実行条件の記録（再開時の照合）


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fingerprint(layout, record: dict, unit_hashes: dict[str, list[str]], *, smoke: bool) -> dict[str, Any]:
    """raw の結果が依存する条件（モデル・単位・方式の表）をまとめる。

    途中から再開するとき、これが前回と同じでなければ止める。条件の違う層の結果が
    混ざると、同じブートストラップの番号が別の記事を指してしまうため。
    """
    files = {
        "units.json": layout.validation / "units.json",
        "methods_summary.csv": layout.results / "methods_summary.csv",
        "jobs_status.csv": layout.als_results("full") / "jobs_status.csv",
    }
    fingerprint = {
        "model": record["model"],
        "smoke": smoke,
        "unit_hashes": unit_hashes,
        "files_sha256": {name: _sha256(path) for name, path in files.items()},
    }
    return json.loads(json.dumps(fingerprint))  # JSON に書いて読んだときと同じ形にそろえる


def check_or_write_fingerprint(raw_dir: Path, fingerprint: dict[str, Any]) -> None:
    """raw にある結果が、今の条件で作られたものかを確かめる。空なら記録を書く。"""
    path = raw_dir / FINGERPRINT_NAME
    existing = sorted(item.name for item in raw_dir.glob("*.pt"))
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != fingerprint:
            differs = sorted(key for key in set(saved) | set(fingerprint) if saved.get(key) != fingerprint.get(key))
            raise RuntimeError(f"{raw_dir} の結果は今と違う条件で作られています（違う項目: {differs}）。raw を退避してからやり直してください。")
        return
    if existing:
        raise RuntimeError(f"{raw_dir} に結果 {existing} があるのに {FINGERPRINT_NAME} がありません。条件を照合できないので止めます。")
    atomic_write_json(path, fingerprint)


# ---------------------------------------------------------------- データ


def validation_units(record: dict, tokenizer, units_json: dict) -> dict[str, list[dict[str, Any]]]:
    """保存した units.json と同じ単位を、トークン列つきで作り直す（hash・長さ・区間を検算）。"""
    encode = _encode(tokenizer)
    wt2 = load_wikitext(record, "wikitext-2-raw-v1", "validation")
    wt2_units = article_units(restore_wikitext_articles(row["text"] for row in wt2), encode, limit=None)
    c4_sequences = official_sequences(read_c4_texts(record["c4"]["path"]), encode, source="c4-validation", limit=CALIBRATION_LIMIT)
    c4_units = [
        {"hash": item["document_hash"], "n_tokens": len(item["token_ids"]), "spans": [[0, len(item["token_ids"])]], "token_ids": item["token_ids"]}
        for item in c4_sequences
    ]
    out = {"wt2": wt2_units, "c4": c4_units}
    for domain, units in out.items():
        saved = units_json[domain]
        if len(saved) != len(units):
            raise RuntimeError(f"{domain} の単位の数が違います: {len(units)} != {len(saved)}")
        for left, right in zip(units, saved, strict=True):
            spans = [list(span) for span in left["spans"]]
            if left["hash"] != right["hash"] or int(left["n_tokens"]) != int(right["n_tokens"]) or spans != [list(s) for s in right["spans"]]:
                raise RuntimeError(f"{domain} の単位が units.json と一致しません: {right['hash']}")
    return out


def chunks_of(units: list[dict[str, Any]]) -> list[tuple[int, torch.Tensor]]:
    chunks = []
    for index, unit in enumerate(units):
        ids = unit["token_ids"]
        for start, end in unit["spans"]:
            chunks.append((index, torch.tensor(ids[start:end], dtype=torch.long)))
    return chunks


# ---------------------------------------------------------------- GPU


def original_nll(model, articles, out_path: Path) -> dict[str, Any]:
    if out_path.is_file():
        return torch.load(out_path, map_location="cpu", weights_only=False)
    started = time.perf_counter()
    sums, tokens = ev.full_model_article_nll(model, articles, device=DEVICE)
    reference = sliding_window_perplexity(model, torch.tensor(articles[0]), max_length=1024, stride=512, use_cache=False)
    gap = ev.relative_gap(sums[0], reference["mean_nll"] * reference["n_tokens"])
    checks = [
        {"check": "original_nll_matches_sliding_window_perplexity_article0", "ok": gap <= 1e-5 and int(tokens[0]) == reference["n_tokens"], "gap": gap}
    ]
    stop_if_failed(checks, "元のモデルの NLL")
    payload = {"sums": sums, "tokens": tokens, "checks": checks, "seconds": time.perf_counter() - started}
    atomic_torch_save(out_path, payload)
    log(f"original nll={ev.mean_nll(sums, tokens):.6f} ppl={ev.perplexity(sums, tokens):.4f} ({payload['seconds']:.0f}s)")
    return payload


def evaluate_layer(model, layout, layer: int, cell_filter, units, original: dict[str, Any], out_path: Path) -> None:
    started = time.perf_counter()
    summary = read_csv_map(layout.results / "methods_summary.csv", "cell")
    status = read_csv_map(layout.als_results("full") / "jobs_status.csv", "job_id")
    grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
    cells: list[ev.CellMethods] = []
    checks: list[dict[str, Any]] = []
    for projection in PROJECTION_ORDER:
        for budget in BUDGETS:
            name = f"L{layer}_{projection}_{budget}"
            if cell_filter is not None and name not in cell_filter:
                continue
            cell = ev.build_cell_matrices(
                layout,
                layer=layer,
                projection=projection,
                budget=budget,
                calibration_gram=grams[INPUT_GROUPS[projection]]["total"],
                summary=summary[f"{layer}_{projection}_{budget}"],
                status=status,
            )
            cells.append(cell)
            checks.extend(cell.checks)
    stop_if_failed(checks, f"L{layer} の方式")
    log(f"L{layer}: methods ready ({len(cells)} cells, {time.perf_counter() - started:.0f}s)")

    # 単位ごとの二次形式 <Δ^T Δ, G_u>
    entries: dict[str, list[tuple[str, str]]] = {group: [] for group in GROUP_MODULES}
    products: dict[str, list[torch.Tensor]] = {group: [] for group in GROUP_MODULES}
    seen_weights: set[str] = set()
    for cell in cells:
        group = INPUT_GROUPS[cell.projection]
        weight = cell.weight.to(DEVICE)
        if cell.projection not in seen_weights:
            entries[group].append((cell.projection, "W"))
            products[group].append(weight.transpose(0, 1) @ weight)
            seen_weights.add(cell.projection)
        for name, matrix in cell.matrices.items():
            delta = weight - matrix.to(DEVICE)
            entries[group].append((cell.cell, name))
            products[group].append(delta.transpose(0, 1) @ delta)
    stacks = {group: torch.stack(items) for group, items in products.items() if items}
    del products
    torch.cuda.empty_cache()
    unit_stats: dict[str, Any] = {}
    sublayer_stats: dict[str, Any] = {}
    for domain in ("wt2", "c4"):
        domain_started = time.perf_counter()
        domain_units = units[domain]
        results, totals, caches = ev.unit_quadratic_pass(
            model, layer, chunks_of(domain_units), len(domain_units), stacks, device=DEVICE
        )
        if cell_filter is None:
            saved = torch.load(layout.validation / f"grams_{domain}_L{layer}.pt", map_location="cpu", weights_only=True)
            for group, total in totals.items():
                reference = saved[group]["total"]
                gap = float(torch.max(torch.abs(total - reference)) / torch.max(torch.abs(reference)))
                checks.append({"check": f"L{layer}:{domain}:{group}:unit_grams_sum_to_saved_total", "ok": gap <= 1e-10, "gap": gap})
        per_cell: dict[str, dict[str, np.ndarray]] = {}
        dens: dict[str, np.ndarray] = {}
        for group, items in entries.items():
            for column, (owner, name) in enumerate(items):
                values = results[group][:, column].copy()
                if name == "W":
                    dens[owner] = values
                else:
                    per_cell.setdefault(owner, {})[name] = values
        unit_stats[domain] = {"num": per_cell, "den": {cell.cell: dens[cell.projection] for cell in cells}}
        for cell in cells:
            job = f"{cell.cell}_M5_M3init_lam1e-06"
            history = layout.als_results("full") / f"{job}_history.csv"
            with history.open(encoding="utf-8", newline="") as handle:
                last = list(csv.DictReader(handle))[-1]
            if cell_filter is None and last.get(f"e_{domain}"):
                e_here = per_cell[cell.cell]["M5"].sum() / unit_stats[domain]["den"][cell.cell].sum()
                gap = ev.relative_gap(float(e_here), float(last[f"e_{domain}"]))
                checks.append({"check": f"{cell.cell}:{domain}:M5_e_matches_als_history", "ok": gap <= 1e-8, "gap": gap})
        log(f"L{layer} {domain}: unit errors done ({time.perf_counter() - domain_started:.0f}s)")
        sub_started = time.perf_counter()
        nums, sub_dens, gated, sub_checks = ev.sublayer_pass(model, layer, caches, len(domain_units), cells, device=DEVICE)
        checks.extend(sub_checks)
        sublayer_stats[domain] = {"num": nums, "den": sub_dens, "gated": gated}
        del caches
        torch.cuda.empty_cache()
        log(f"L{layer} {domain}: sublayer done ({time.perf_counter() - sub_started:.0f}s)")
    del stacks
    torch.cuda.empty_cache()
    stop_if_failed(checks, f"L{layer} の単位・サブ層")

    # NLL（WikiText-2 の記事ごと）
    nll_started = time.perf_counter()
    articles = [unit["token_ids"] for unit in units["wt2"]]
    windows = ev.cache_window_inputs(model, layer, articles, device=DEVICE)
    base_sums, base_tokens = ev.nll_from_layer(model, layer, windows, len(articles))
    gap = float(np.max(np.abs(base_sums - original["sums"][: len(articles)]) / np.abs(original["sums"][: len(articles)])))
    checks.append({"check": f"L{layer}:nll_from_layer_equals_full_model", "ok": gap <= 1e-9, "gap": gap})
    checks.append({"check": f"L{layer}:nll_tokens_equal", "ok": bool(np.array_equal(base_tokens, original["tokens"][: len(articles)]))})
    first = cells[0]
    module = ev.module_at(model.model.layers[layer], ev.PROJECTION_PATHS[first.projection])
    with ev.swapped_weight(module, first.weight.to(device=DEVICE, dtype=torch.float32)):
        identity_sums, _ = ev.nll_from_layer(model, layer, windows, len(articles))
    checks.append({"check": f"L{layer}:nll_identity_with_original_weight", "ok": bool(np.array_equal(identity_sums, base_sums))})
    stop_if_failed(checks, f"L{layer} の NLL")
    nll = ev.nll_pass(model, layer, windows, len(articles), cells, device=DEVICE, progress=log)
    for cell_name, values in nll.items():
        ev.finite_or_raise(values, f"{cell_name}:nll")
    del windows
    torch.cuda.empty_cache()
    log(f"L{layer}: nll done ({time.perf_counter() - nll_started:.0f}s)")
    payload = {
        "layer": layer,
        "cells": [cell.cell for cell in cells],
        "stop_reasons": {cell.cell: cell.stop_reasons for cell in cells},
        "unit_stats": unit_stats,
        "sublayer_stats": sublayer_stats,
        "nll": nll,
        "nll_tokens": base_tokens,
        "checks": checks,
        "seconds": time.perf_counter() - started,
    }
    atomic_torch_save(out_path, payload)
    log(f"L{layer}: saved ({payload['seconds']:.0f}s)")


def run_gpu(project_root: Path, *, smoke: bool, layers: list[int]) -> None:
    layout = layout_from_root(project_root)
    out = layout.results / ("evaluation_smoke" if smoke else "evaluation")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    record = load_download_record(project_root / "data" / "hf_cache" / "download_record.json")
    model, tokenizer = load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
    model.to(dtype=torch.float32, device=DEVICE)
    model.eval()
    ev.assert_uniform_attention(model)
    units_json = json.loads((layout.validation / "units.json").read_text(encoding="utf-8"))
    units = validation_units(record, tokenizer, units_json)
    if smoke:
        units = {domain: items[: SMOKE_UNITS[domain]] for domain, items in units.items()}
    log(f"units wt2={len(units['wt2'])} c4={len(units['c4'])}")
    unit_hashes = {domain: [u["hash"] for u in items] for domain, items in units.items()}
    check_or_write_fingerprint(out / "raw", run_fingerprint(layout, record, unit_hashes, smoke=smoke))
    original = original_nll(model, [unit["token_ids"] for unit in units["wt2"]], out / "raw" / "original_nll.pt")
    atomic_write_json(out / "units_used.json", unit_hashes)
    for layer in layers:
        path = out / "raw" / f"L{layer}.pt"
        if path.is_file():
            log(f"L{layer}: already done, skip")
            continue
        cell_filter = set(SMOKE_CELLS) if smoke else None
        if smoke and not any(name.startswith(f"L{layer}_") for name in SMOKE_CELLS):
            continue
        evaluate_layer(model, layout, layer, cell_filter, units, original, path)


# ---------------------------------------------------------------- 集計


def _flatten(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{key}_{prefix}": value for key, value in row.items()}


def summarize(project_root: Path, *, smoke: bool) -> None:
    layout = layout_from_root(project_root)
    out = layout.results / ("evaluation_smoke" if smoke else "evaluation")
    summary = read_csv_map(layout.results / "methods_summary.csv", "cell")
    status = read_csv_map(layout.als_results("full") / "jobs_status.csv", "job_id")
    raw_dir = out / "raw"

    # raw が今の条件で作られたか（B-3）
    record = load_download_record(project_root / "data" / "hf_cache" / "download_record.json")
    unit_hashes = json.loads((out / "units_used.json").read_text(encoding="utf-8"))
    fingerprint_path = raw_dir / FINGERPRINT_NAME
    if not fingerprint_path.is_file():
        raise RuntimeError(f"{fingerprint_path} がありません。条件を照合できないので集計しません。")
    if json.loads(fingerprint_path.read_text(encoding="utf-8")) != run_fingerprint(layout, record, unit_hashes, smoke=smoke):
        raise RuntimeError(f"{raw_dir} の結果は今と違う条件で作られています。集計しません。")
    n_units = {domain: len(hashes) for domain, hashes in unit_hashes.items()}

    # 層とセルがそろっているか（B-1）。欠けていたら部分的な集計を書かずに止める。
    if smoke:
        expected_cells = set(SMOKE_CELLS)
    else:
        expected_cells = {f"L{layer}_{projection}_{budget}" for layer in LAYERS for projection in PROJECTION_ORDER for budget in BUDGETS}
    expected_layers = sorted({int(name[1:].split("_")[0]) for name in expected_cells})
    missing_layers = [layer for layer in expected_layers if not (raw_dir / f"L{layer}.pt").is_file()]
    if missing_layers:
        raise RuntimeError(f"層 {missing_layers} の結果がありません。--stage gpu を終えてから集計してください。")
    original = torch.load(raw_dir / "original_nll.pt", map_location="cpu", weights_only=False)
    raws = [torch.load(raw_dir / f"L{layer}.pt", map_location="cpu", weights_only=False) for layer in expected_layers]
    found_cells = [cell for raw in raws for cell in raw["cells"]]
    if len(found_cells) != len(set(found_cells)) or set(found_cells) != expected_cells:
        raise RuntimeError(
            f"セルがそろっていません: 足りない {sorted(expected_cells - set(found_cells))}、余分 {sorted(set(found_cells) - expected_cells)}"
        )
    if not smoke and len(found_cells) != FULL_CELL_COUNT:
        raise RuntimeError(f"セルの数が {len(found_cells)} です（{FULL_CELL_COUNT} のはず）。")
    if len(original["sums"]) != n_units["wt2"]:
        raise RuntimeError(f"元のモデルの NLL の記事数 {len(original['sums'])} が単位の数 {n_units['wt2']} と違います。")
    for raw in raws:
        if len(raw["nll_tokens"]) != n_units["wt2"]:
            raise RuntimeError(f"L{raw['layer']} の NLL の記事数が単位の数と違います。")
        for domain in n_units:
            for cell in raw["cells"]:
                if len(raw["unit_stats"][domain]["den"][cell]) != n_units[domain]:
                    raise RuntimeError(f"{cell} の {domain} の単位の数が {n_units[domain]} と違います。")

    counts = {domain: st.bootstrap_counts(n) for domain, n in n_units.items()}
    expected_digest = {domain: st.counts_digest(value) for domain, value in counts.items()}
    used_digests: dict[str, list[str]] = {domain: [] for domain in counts}

    def evaluate(numerators, denominator, domain: str, **kwargs) -> dict[str, Any]:
        """evaluate_cell を呼び、実際に使った再標本の sha256 を控えて結果から外す（B-8）。"""
        result = st.evaluate_cell(numerators, denominator, counts[domain], **kwargs)
        used_digests[domain].append(result.pop("counts_digest"))
        return result

    checks: list[dict[str, Any]] = list(original["checks"])
    cell_rows, sub_rows, gated_rows, nll_rows, lambda_rows, init_rows = [], [], [], [], [], []
    for raw in raws:
        checks.extend(raw["checks"])
        for cell in raw["cells"]:
            layer, projection, budget = cell[1:].split("_")
            meta_row = summary[f"{layer}_{projection}_{budget}"]
            m5_job = status[f"{cell}_M5_M3init_lam1e-06"]
            stop = raw["stop_reasons"][cell]
            unresolved = stop.get("M5") == "local_solver_unresolved"
            confirmation = cell not in ("L14_up_50", "L14_up_25")
            meta = {
                "cell": cell,
                "layer": int(layer),
                "projection": projection,
                "budget_percent": int(budget),
                "group": INPUT_GROUPS[projection],
                "bond_class": meta_row["bond_class"],
                "confirmation": confirmation,
                "main_set": confirmation and meta_row["bond_class"] == "multi_cut",
                "small_baseline_gap": meta_row["small_baseline_gap_lam1e-06"],
                "m5_stop_reason": stop.get("M5"),
                "m5_sweeps": int(m5_job["sweeps"]),
                "optimization_censored": stop.get("M5") == "max_sweeps",
                "m3f_stop_reason": stop.get("M3F"),
                "rho_calibration": st.calibration_rho(
                    float(meta_row["q_m2_lam1e-06"]), float(meta_row["q_m3"]), float(m5_job["q_lambda"])
                ),
            }
            row = dict(meta)
            for domain in ("wt2", "c4"):
                stats = raw["unit_stats"][domain]
                result = evaluate(stats["num"][cell], stats["den"][cell], domain, solver_unresolved=unresolved)
                row.update(_flatten(domain, result))
                row[f"strong_calibration_validation_reversal_{domain}"] = meta["rho_calibration"] >= 0.5 and result["class"] == "low"
                for lam in ("1e-07", "1e-05"):
                    alt = {"M2": stats["num"][cell][f"M2_lam{lam}"], "M3": stats["num"][cell]["M3"], "M5": stats["num"][cell]["M5"]}
                    row[f"class_M2lam{lam}_M5main_{domain}"] = evaluate(
                        alt, stats["den"][cell], domain, m3f=None, solver_unresolved=unresolved
                    )["class"]
                if "M5_M4r" in stats["num"][cell]:
                    init = evaluate(
                        {"M2": stats["num"][cell]["M2"], "M3": stats["num"][cell]["M3"], "M5": stats["num"][cell]["M5_M4r"]},
                        stats["den"][cell],
                        domain,
                        m3f=None,
                        solver_unresolved=stop.get("M5_M4r") == "local_solver_unresolved",
                    )
                    init_rows.append(
                        {
                            "cell": cell,
                            "domain": domain,
                            "m5_m4r_stop_reason": stop.get("M5_M4r"),
                            "e_M5": result["e_M5"],
                            "e_M5_M4r": init["e_M5"],
                            "rho_M5": result["rho"],
                            "rho_M5_M4r": init["rho"],
                            "class_M5": result["class"],
                            "class_M5_M4r": init["class"],
                        }
                    )
                if f"M5_lam1e-07" in stats["num"][cell]:
                    classes = {"1e-06": result["class"]}
                    for lam in ("1e-07", "1e-05"):
                        lam_numerators = {"M2": stats["num"][cell][f"M2_lam{lam}"], "M3": stats["num"][cell]["M3"], "M5": stats["num"][cell][f"M5_lam{lam}"]}
                        lam_unresolved = stop.get(f"M5_lam{lam}") == "local_solver_unresolved"
                        lam_result = evaluate(lam_numerators, stats["den"][cell], domain, m3f=None, solver_unresolved=lam_unresolved)
                        classes[lam] = lam_result["class"]
                        lambda_rows.append({"cell": cell, "domain": domain, "lambda": lam, "class": lam_result["class"], "rho": lam_result["rho"], "e_M5": lam_result["e_M5"], "e_M2": lam_result["e_M2"]})
                    lambda_rows.append({"cell": cell, "domain": domain, "lambda": "1e-06", "class": result["class"], "rho": result["rho"], "e_M5": result["e_M5"], "e_M2": result["e_M2"]})
                    # 解けなかった λ（not_classified）は比べる相手にしない。比べられる λ が 1 つ以下なら判定しない。
                    resolved = [name for name in classes.values() if name != "not_classified"]
                    row[f"lambda_sensitive_{domain}"] = len(set(resolved)) > 1 if len(resolved) >= 2 else None
                    row[f"lambda_not_classified_{domain}"] = [lam for lam, name in classes.items() if name == "not_classified"]
                else:
                    row[f"lambda_sensitive_{domain}"] = None
                    row[f"lambda_not_classified_{domain}"] = None
                sub = raw["sublayer_stats"][domain]
                kind = "attn" if projection in ev.ATTENTION_PROJECTIONS else "mlp"
                sub_result = evaluate(sub["num"][cell], sub["den"][kind], domain, solver_unresolved=unresolved)
                sub_rows.append({**meta, "domain": domain, "sublayer": kind, **sub_result})
                if cell in sub["gated"]:
                    gated = sub["gated"][cell]
                    gated_nums = {key[4:]: value for key, value in gated.items() if key.startswith("num_")}
                    gated_rows.append({**meta, "domain": domain, **evaluate(gated_nums, gated["den"], domain, solver_unresolved=unresolved)})
            cell_rows.append(row)
            sums = {"orig": original["sums"], **raw["nll"][cell]}
            pairs = [(name, "orig") for name in ev.NLL_METHODS] + [("M5", "M3"), ("M5", "M2"), ("M5", "M1")]
            nll_result = st.nll_report(sums, raw["nll_tokens"], counts["wt2"], pairs=pairs)
            used_digests["wt2"].append(nll_result.pop("counts_digest"))
            nll_rows.append({**meta, **nll_result})
    for domain, digests in used_digests.items():
        checks.append(
            {
                "check": f"bootstrap_same_counts_for_all_methods:{domain}",
                "ok": bool(digests) and set(digests) == {expected_digest[domain]},
                "n_evaluations": len(digests),
                "digest": expected_digest[domain],
                "distinct_digests_used": sorted(set(digests)),
            }
        )
    main_rows = [row for row in cell_rows if row["main_set"]]
    aggregate: dict[str, Any] = {}
    if not smoke:
        aggregate["main"] = st.aggregate_statements(main_rows)  # 34 でなければ ValueError で止まる
    else:
        aggregate["main"] = {"skipped": f"smoke（main_set のセルが {len(main_rows)}）"}
    for label, selector in (
        ("near_saturated_bond", lambda r: r["confirmation"] and r["bond_class"] == "near_saturated_bond"),
        ("development_cells", lambda r: not r["confirmation"]),
    ):
        aggregate[label] = [
            {"cell": r["cell"], "class_wt2": r["class_wt2"], "class_c4": r["class_c4"], "rho_wt2": r["rho_wt2"], "rho_c4": r["rho_c4"]}
            for r in cell_rows
            if selector(r)
        ]
    for domain in ("wt2", "c4"):
        aggregate[f"sublayer_class_counts_main_{domain}"] = {
            name: sum(1 for r in sub_rows if r["main_set"] and r["domain"] == domain and r["class"] == name) for name in st.CLASS_ORDER
        }
    aggregate["flags"] = {
        "optimization_censored": [r["cell"] for r in cell_rows if r["optimization_censored"]],
        "not_classified": [r["cell"] for r in cell_rows if r["class_wt2"] == "not_classified"],
        "lambda_sensitive": [r["cell"] for r in cell_rows if r.get("lambda_sensitive_wt2") or r.get("lambda_sensitive_c4")],
        "lambda_not_classified": [r["cell"] for r in cell_rows if r.get("lambda_not_classified_wt2")],
        "strong_calibration_validation_reversal": [
            r["cell"] for r in cell_rows if r["strong_calibration_validation_reversal_wt2"] or r["strong_calibration_validation_reversal_c4"]
        ],
        "attained_residual_positive_wt2": [r["cell"] for r in cell_rows if r["attained_residual_positive_wt2"]],
        "small_baseline_gap": [r["cell"] for r in cell_rows if str(r["small_baseline_gap"]) == "True"],
    }
    failed = [item for item in checks if not item["ok"]]
    aggregate["checks_passed"] = not failed
    aggregate["n_checks"] = len(checks)
    for name, rows in (
        ("cells.csv", cell_rows),
        ("cells_sublayer.csv", sub_rows),
        ("cells_e_gated_up.csv", gated_rows),
        ("nll.csv", nll_rows),
        ("lambda.csv", lambda_rows),
        ("init_m4r.csv", init_rows),
    ):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            atomic_write_csv(out / name, rows, fields)
    atomic_write_json(out / "checks.json", {"failed": failed, "checks": checks})
    atomic_write_json(out / "aggregate.json", aggregate)
    log(f"summary: cells={len(cell_rows)} checks={len(checks)} failed={len(failed)}")
    log(json.dumps(aggregate["main"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gpu", "summarize", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in LAYERS))
    args = parser.parse_args()
    root = find_project_root(Path(__file__))
    layers = [int(item) for item in args.layers.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage in ("gpu", "all"):
        run_gpu(root, smoke=args.smoke, layers=layers)
    if args.stage in ("summarize", "all"):
        summarize(root, smoke=args.smoke)


if __name__ == "__main__":
    main()
