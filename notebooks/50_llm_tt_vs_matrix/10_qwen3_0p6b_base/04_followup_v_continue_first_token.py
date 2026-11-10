"""04 の追加分析 6 の続き：解き足した M5（v_proj 25%）で、先頭のトークンの誤差はどう動いたか。

v の続き計算（04_followup_v_continue.py）が 100 掃引ごとに保存したコアと、開始点（04 の M5）について、
次を測る（WikiText-2 と C4 の検証区間）。

1. 射影の出力の段階：||R x_0||^2 / ||W x_0||^2（先頭）と、先頭以外での同じ比（R = W − Ŵ）。
2. head の出力の段階・o の後の段階（分析 3 の sink test と同じ定義）：
   - sink：v の出力を先頭の位置だけ置き換えたときの誤差 / 元の値（全位置で和）
   - full：全位置で置き換えたときの誤差 / 元の値
   q・k は元のままなので、attention の重みは変わらない。差は Δheads = probs @ ΔV で直接計算する。
比較のため M2・M3 の値も同じ方法で出す。新しい ALS はない。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_v_continue_first_token.py
  数分の確認（コア 3 個・区間 20 個、保存は別ファイル）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_v_continue_first_token.py --check
結果: results/.../followup_analysis/v_continue/first_token_errors.json と first_token_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "followup_attention_stages", Path(__file__).resolve().with_name("04_followup_attention_stages.py")
)
base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base_mod)

from nn_compression.tt_mps.matrix import tt_matrix_to_dense  # noqa: E402

ROOT = base_mod.ROOT
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "v_continue"
DEVICE = base_mod.DEVICE
LAYERS = (7, 14, 21)
GROUP = 26  # 一度に GPU で処理する行列の数
torch.set_grad_enabled(False)


def candidate_matrices(layout, layer, cm, check):
    """[(名前, 掃引数, 行列 float64 CPU)]。M2・M3、04 の M5（掃引 0）、保存したコアの順。"""
    cell = f"L{layer}_v_25"
    payload = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
    start = torch.load(layout.als_dir("full") / f"{cell}_M5_M3init_lam1e-06.pt", map_location="cpu", weights_only=True)
    start_dense = tt_matrix_to_dense([c.to(torch.float64) for c in start["cores"]], payload["out_modes"], payload["in_modes"])
    gap = float((start_dense - cm.matrices["M5"].to(torch.float64)).abs().max())
    items = [("M2", None, cm.matrices["M2"].to(torch.float64)), ("M3", None, cm.matrices["M3"].to(torch.float64)),
             ("M5", int(start["sweeps_completed"]), start_dense)]
    folder = layout.models / "followup" / "v_continue" / cell
    files = sorted(folder.glob("cores_s*.pt"))
    if check:
        files = files[:2]
    for path in files:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        dense = tt_matrix_to_dense([c.to(torch.float64) for c in saved["cores"]], saved["out_modes"], saved["in_modes"])
        items.append(("M5", int(saved["start_sweeps"]) + int(saved["sweep_after_start"]), dense))
    return items, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "first_token_errors_check" if args.check else "first_token_errors"

    layout = base_mod.layout_from_root(ROOT)
    record = base_mod.load_download_record(ROOT / "data" / "hf_cache" / "download_record.json")
    model, tokenizer = base_mod.load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
    model.to(dtype=torch.float32, device=DEVICE)
    model.eval()
    units_json = json.loads((layout.validation / "units.json").read_text(encoding="utf-8"))
    units = base_mod.script.validation_units(record, tokenizer, units_json)
    chunks = {dom: base_mod.script.chunks_of(items) for dom, items in units.items()}
    if args.check:
        chunks = {dom: c[:20] for dom, c in chunks.items()}
    summary = base_mod.script.read_csv_map(layout.results / "methods_summary.csv", "cell")
    status = base_mod.script.read_csv_map(layout.als_results("full") / "jobs_status.csv", "job_id")

    rows = []
    checks = {}
    for layer in LAYERS:
        attn = model.model.layers[layer].self_attn
        w_q, w_k = attn.q_proj.weight.detach(), attn.k_proj.weight.detach()
        w_v32 = attn.v_proj.weight.detach()
        w_o = attn.o_proj.weight.detach()
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        cm = base_mod.build_cell_matrices(layout, layer=layer, projection="v", budget=25,
                                          calibration_gram=grams["attn_in"]["total"],
                                          summary=summary[f"{layer}_v_25"], status=status)
        bad = [c for c in cm.checks if not c["ok"]]
        if bad:
            raise RuntimeError(f"検算に失敗: {bad[:3]}")
        W = torch.load(layout.weights / f"W_L{layer}_v.pt", map_location="cpu", weights_only=True)["W"].to(torch.float64)
        items, gap = candidate_matrices(layout, layer, cm, args.check)
        checks[f"L{layer}_start_vs_04_M5_max_abs"] = gap
        checks[f"L{layer}_W_file_vs_model_max_abs"] = float((W - w_v32.double().cpu()).abs().max())
        n = len(items)
        R64 = torch.stack([W - m for _, _, m in items]).to(DEVICE)          # (n, out, in) float64
        R32 = R64.to(torch.float32)
        W64 = W.to(DEVICE)
        d = attn.head_dim

        for dom, dom_chunks in chunks.items():
            t0 = time.perf_counter()
            inputs = base_mod.capture_attention_inputs(model, layer, dom_chunks)
            acc = {k: torch.zeros(n, dtype=torch.float64, device=DEVICE)
                   for k in ("p_first", "p_rest", "h_sink", "h_full", "o_sink", "o_full")}
            den = {k: 0.0 for k in ("p_first", "p_rest", "h", "o")}
            for x_cpu, cos_cpu, sin_cpu in inputs:
                x = x_cpu.to(DEVICE)                                             # (1, T, in) float32
                cos, sin = cos_cpu.to(DEVICE), sin_cpu.to(DEVICE)
                b, t, _ = x.shape
                x64 = x[0].double()
                # 射影の段階（float64）
                wx = x64 @ W64.T
                den["p_first"] += float((wx[0] ** 2).sum())
                den["p_rest"] += float((wx[1:] ** 2).sum())
                # 元の attention（float32、sink test と同じ計算）
                q = attn.q_norm((x @ w_q.T).view(b, t, -1, d)).transpose(1, 2)
                k = attn.k_norm((x @ w_k.T).view(b, t, -1, d)).transpose(1, 2)
                v = (x @ w_v32.T).view(b, t, -1, d).transpose(1, 2)
                q_rot, k_rot = base_mod.apply_rotary_pos_emb(q, k, cos, sin)
                groups = q.shape[1] // k.shape[1]
                scores = (q_rot @ k_rot.repeat_interleave(groups, dim=1).transpose(-1, -2)) * attn.scaling
                mask = torch.ones(t, t, dtype=torch.bool, device=DEVICE).tril()
                probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1, dtype=torch.float32)[0]  # (H, T, T)
                heads = probs @ v.repeat_interleave(groups, dim=1)[0]               # (H, T, d)
                den["h"] += float((heads.double() ** 2).sum())
                den["o"] += float((heads.transpose(0, 1).reshape(t, -1) @ w_o.T).double().pow(2).sum())
                H = probs.shape[0]
                for s in range(0, n, GROUP):
                    sl = slice(s, min(n, s + GROUP))
                    rx64 = torch.einsum("ti,noi->nto", x64, R64[sl])               # (g, T, out) = R x
                    acc["p_first"][sl] += (rx64[:, 0] ** 2).sum(dim=-1)
                    acc["p_rest"][sl] += (rx64[:, 1:] ** 2).sum(dim=(1, 2))
                    dv = torch.einsum("ti,noi->nto", x[0], R32[sl])                # ΔV = −(Ŵ − W)x の符号違い。二乗なので同じ
                    g = dv.shape[0]
                    dv = dv.view(g, t, -1, d).transpose(1, 2).repeat_interleave(groups, dim=1)  # (g, H, T, d)
                    dh_full = probs.unsqueeze(0) @ dv                               # (g, H, T, d)
                    dh_sink = probs[:, :, 0].view(1, H, t, 1) * dv[:, :, 0:1, :]    # 先頭の位置だけ置き換え
                    for name, dh in (("full", dh_full), ("sink", dh_sink)):
                        acc[f"h_{name}"][sl] += (dh.double() ** 2).sum(dim=(1, 2, 3))
                        do = dh.transpose(1, 2).reshape(g, t, -1) @ w_o.T
                        acc[f"o_{name}"][sl] += (do.double() ** 2).sum(dim=(1, 2))
                del x, cos, sin, probs, heads
            for i, (method, sweep, _m) in enumerate(items):
                rows.append({
                    "layer": layer, "domain": dom, "method": method, "sweep": "" if sweep is None else sweep,
                    "proj_first": float(acc["p_first"][i]) / den["p_first"],
                    "proj_rest": float(acc["p_rest"][i]) / den["p_rest"],
                    "heads_sink": float(acc["h_sink"][i]) / den["h"],
                    "heads_full": float(acc["h_full"][i]) / den["h"],
                    "out_sink": float(acc["o_sink"][i]) / den["o"],
                    "out_full": float(acc["o_full"][i]) / den["o"],
                })
            last = rows[-1]
            print(f"L{layer} {dom}: {len(inputs)} chunks {n} matrices {time.perf_counter() - t0:.0f}s "
                  f"last sweep {last['sweep']} proj_first={last['proj_first']:.3f} heads_sink={last['heads_sink']:.3f}",
                  flush=True)
            del inputs
            torch.cuda.empty_cache()
        del R64, R32

    # 分析 3（sink test）の値との突き合わせ（開始点と M2・M3）
    sink_path = OUT.parent / "sink_test" / "sink_test.json"
    if sink_path.is_file() and not args.check:
        sink = json.loads(sink_path.read_text(encoding="utf-8"))
        worst = 0.0
        start_sweeps = {r["layer"]: r["sweep"] for r in rows if r["method"] == "M5" and r["sweep"] is not None
                        and r["sweep"] == min(x["sweep"] for x in rows if x["layer"] == r["layer"] and x["method"] == "M5")}
        for r in rows:
            if r["method"] == "M5" and r["sweep"] != start_sweeps[r["layer"]]:
                continue  # 比べるのは開始点（04 の M5）と M2・M3 だけ
            for where in ("sink", "full"):
                a = sink[f"L{r['layer']}_v_25|{r['method']}|{r['domain']}|{where}"]
                for stage, key in (("heads", f"heads_{where}"), ("out", f"out_{where}")):
                    ref = a[f"{stage}_num"] / a[f"{stage}_den"]
                    worst = max(worst, abs(r[key] - ref) / ref)
        checks["max_rel_gap_vs_sink_test"] = worst

    fields = ["layer", "domain", "method", "sweep", "proj_first", "proj_rest", "heads_sink", "heads_full", "out_sink", "out_full"]
    with open(OUT / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"{stem}.json").write_text(json.dumps({"checks": checks, "rows": rows}, indent=1), encoding="utf-8")
    print("checks:", checks)


if __name__ == "__main__":
    main()
