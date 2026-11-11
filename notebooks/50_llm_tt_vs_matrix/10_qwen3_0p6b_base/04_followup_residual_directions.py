"""04 の追加分析 7：同じ較正の目的で、M2 と M5 は誤差をどの向きに置いたか（v_proj 25%、L7・L14・L21）。

方式：M2、M3、M5 の開始点（04 の M5、M3 から）、M5 の最終（v の続き計算の +10,000 掃引後）。
較正 Gram 行列 G（正則化なし）の固有ベクトル u_i・固有値 γ_i（大きい順）について、R = W − Ŵ として次を出す。

1. 重みなしの誤差 a_i = ||R u_i||^2 と、元の重みでの量 ||W u_i||^2。
2. 較正の目的への寄与 w_i = γ_i a_i。Σ_i w_i = tr(R G R^T) が厳密に成り立つ（G と P_b は可換なので、
   帯ごとの和 Σ_{i∈b} w_i = tr(R G P_b R^T) にも交差項はない）。÷ tr(W G W^T) で相対誤差にする。
3. 先頭のトークンでの誤差の向きごとの分解 c_i = Σ (R x_0)·(R u_i)(u_i·x_0)（Σ_i c_i = Σ ||R x_0||^2）。
4. 順位 k 以降の累積 C(k) = Σ_{i≥k}（較正の寄与・先頭の誤差・先頭の入力の量）。帯の区切りに頼らない見方。
5. まれな向き（257 位以降）での残差 R P_tail の右特異ベクトル v_j・特異値 s_j と、先頭の入力との重なり
   q_j = 平均 (v_j·x_0)^2 / ||x_0||^2。||R P_tail x_0||^2 = Σ_j s_j^2 (v_j·x_0)^2。

モデルは attention の入力を取り出すところまでしか動かさない。新しい ALS はない。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_residual_directions.py
  数分の確認（区間 20 個・保存は別ファイル）: ... --check
結果: results/.../followup_analysis/residual_directions/residual_directions.json
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "residual_directions"
DEVICE = base_mod.DEVICE
LAYERS = (7, 14, 21)
BANDS = [(0, 16), (16, 64), (64, 256), (256, None)]
BAND_NAMES = ["1-16", "17-64", "65-256", "257-"]
CUTS = (17, 33, 65, 129, 257, 513)   # C(k) を表に出す順位（1 始まり）
TAIL_START = 256                     # 0 始まりの添字。257 位以降
TOP_SINGULAR = 20
torch.set_grad_enabled(False)


def dense_from(path, out_modes, in_modes):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return tt_matrix_to_dense([c.to(torch.float64) for c in saved["cores"]], out_modes, in_modes)


def method_matrices(layout, layer, cm):
    cell = f"L{layer}_v_25"
    payload = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
    om, im = payload["out_modes"], payload["in_modes"]
    final_dir = layout.models / "followup" / "v_continue" / cell
    final_path = sorted(final_dir.glob("cores_s*.pt"))[-1]
    final_sweep = int(torch.load(final_path, map_location="cpu", weights_only=True)["sweep_after_start"])
    return {
        "M2": cm.matrices["M2"].to(torch.float64),
        "M3": cm.matrices["M3"].to(torch.float64),
        "M5_start": dense_from(layout.als_dir("full") / f"{cell}_M5_M3init_lam1e-06.pt", om, im),
        "M5_final": dense_from(final_path, om, im),
    }, final_sweep


def band_sums(vec, dim):
    return [float(vec[a:(dim if b is None else b)].sum()) for a, b in BANDS]


def tail_cum(vec):
    """C(k) = Σ_{i≥k} vec_i / Σ vec（k は 1 始まり）。"""
    total = float(vec.sum())
    return {str(k): float(vec[k - 1:].sum()) / total for k in CUTS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "residual_directions_check" if args.check else "residual_directions"

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

    results = {"bands": BAND_NAMES, "cuts": list(CUTS), "layers": {}}
    checks = {"max_cal_sum_gap": 0.0, "max_first_sum_gap": 0.0, "max_tail_svd_gap": 0.0}
    for layer in LAYERS:
        t0 = time.perf_counter()
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        G = grams["attn_in"]["total"].to(DEVICE, torch.float64)
        cm = base_mod.build_cell_matrices(layout, layer=layer, projection="v", budget=25,
                                          calibration_gram=grams["attn_in"]["total"],
                                          summary=summary[f"{layer}_v_25"], status=status)
        bad = [c for c in cm.checks if not c["ok"]]
        if bad:
            raise RuntimeError(f"検算に失敗: {bad[:3]}")
        W = torch.load(layout.weights / f"W_L{layer}_v.pt", map_location="cpu", weights_only=True)["W"].to(DEVICE, torch.float64)
        mats, final_sweep = method_matrices(layout, layer, cm)
        values, vectors = torch.linalg.eigh(G)
        order = torch.argsort(values, descending=True)
        gamma = values[order].clamp_min(0.0)
        U = vectors[:, order].contiguous()
        dim = U.shape[0]
        WU = W @ U
        w_norm = (WU ** 2).sum(dim=0)                     # ||W u_i||^2
        cal_den = float((gamma * w_norm).sum())           # tr(W G W^T)
        R = {m: W - A.to(DEVICE) for m, A in mats.items()}
        RU = {m: r @ U for m, r in R.items()}

        # 先頭と先頭以外の入力（固有基底の成分）
        coef_first = {}
        energy_rest = {}
        for dom, dom_chunks in chunks.items():
            inputs = base_mod.capture_attention_inputs(model, layer, dom_chunks)
            firsts, rest = [], torch.zeros(dim, dtype=torch.float64, device=DEVICE)
            rest_gram = torch.zeros(dim, dim, dtype=torch.float64, device=DEVICE)   # 固有基底での先頭以外の Gram
            for x_cpu, _c, _s in inputs:
                c = x_cpu[0].to(DEVICE, torch.float64) @ U
                firsts.append(c[0])
                rest += (c[1:] ** 2).sum(dim=0)
                rest_gram += c[1:].T @ c[1:]
            coef_first[dom] = torch.stack(firsts)          # (n, d)
            energy_rest[dom] = (rest, rest_gram)
            del inputs
            torch.cuda.empty_cache()

        layer_out = {"final_sweep_after_start": final_sweep, "cal_den": cal_den, "methods": {}, "inputs": {}}
        for dom in chunks:
            e0 = (coef_first[dom] ** 2).sum(dim=0)
            layer_out["inputs"][dom] = {
                "first_energy_tail_cum": tail_cum(e0),
                "rest_energy_tail_cum": tail_cum(energy_rest[dom][0]),
            }
        per_rank = {"gamma": gamma.cpu().tolist(), "w_norm": w_norm.cpu().tolist(), "methods": {}}
        for m, ru in RU.items():
            a = (ru ** 2).sum(dim=0)                        # ||R u_i||^2
            w = gamma * a                                   # 較正の目的への寄与
            direct = float(torch.trace(R[m] @ G @ R[m].T))
            checks["max_cal_sum_gap"] = max(checks["max_cal_sum_gap"], abs(float(w.sum()) - direct) / direct)
            rel_unw = a / w_norm.clamp_min(1e-300)          # 向きごとの相対誤差（重みなし）
            mo = {
                "cal_relative_error": float(w.sum()) / cal_den,
                "cal_band_relative": [v / cal_den for v in band_sums(w, dim)],
                "cal_band_share": [v / float(w.sum()) for v in band_sums(w, dim)],
                "cal_tail_cum": tail_cum(w),
                "unweighted_band_share": [v / float(a.sum()) for v in band_sums(a, dim)],
                "unweighted_rel_band_median": [float(rel_unw[s0:(dim if s1 is None else s1)].median()) for s0, s1 in BANDS],
                "unweighted_rel_band_ratio": [band_sums(a, dim)[i] / band_sums(w_norm, dim)[i] for i in range(len(BANDS))],
                "domains": {},
            }
            # まれな向きでの残差の特異方向
            Rt = ru[:, TAIL_START:]                         # R P_tail を固有基底の tail 成分で表したもの
            _, s, Vh = torch.linalg.svd(Rt, full_matrices=False)
            checks["max_tail_svd_gap"] = max(checks["max_tail_svd_gap"],
                                             abs(float((s ** 2).sum()) - float(a[TAIL_START:].sum())) / float(a[TAIL_START:].sum()))
            for dom in chunks:
                cf = coef_first[dom]
                rx0 = cf @ ru.T                             # (n, out)
                c_i = ((rx0 @ ru) * cf).sum(dim=0)          # 向きごとの寄与
                total = float((rx0 ** 2).sum())
                den0 = float(((cf @ WU.T) ** 2).sum())
                checks["max_first_sum_gap"] = max(checks["max_first_sum_gap"], abs(float(c_i.sum()) - total) / total)
                proj = cf[:, TAIL_START:] @ Vh.T            # (n, k) v_j·x_0
                # 重なり q_j = Σ (v_j·x)^2 / Σ ||x||^2（入力の量の何割が v_j に乗るか）
                q = (proj ** 2).sum(dim=0) / float((cf ** 2).sum())
                contrib = (s ** 2) * (proj ** 2).sum(dim=0) # ||R P_tail x_0||^2 への寄与
                tail_total = float(contrib.sum())
                rest_e, rest_gram = energy_rest[dom]
                q_rest = ((Vh @ rest_gram[TAIL_START:, TAIL_START:]) * Vh).sum(dim=1) / float(rest_e.sum())
                # 入力の長さあたりの出力・誤差の大きさ（先頭と先頭以外）
                x0_sq = float((cf ** 2).sum())
                rest_sq = float(rest_e.sum())
                gain = {
                    "first_W": den0 / x0_sq,
                    "first_R": total / x0_sq,
                    "rest_W": float(torch.trace(WU @ rest_gram @ WU.T)) / rest_sq,
                    "rest_R": float(torch.trace(ru @ rest_gram @ ru.T)) / rest_sq,
                }
                q_cal = (Vh ** 2) @ gamma[TAIL_START:] / float(gamma.sum())
                mo["domains"][dom] = {
                    "gain": gain,
                    "first_relative_error": total / den0,
                    "first_band_share": [v / total for v in band_sums(c_i, dim)],
                    "first_tail_cum": tail_cum(c_i),
                    "first_tail_part_relative": float(((cf[:, TAIL_START:] @ Rt.T) ** 2).sum()) / den0,
                    "tail_top_singular": [
                        {"j": j + 1, "s2_share": float(s[j] ** 2 / (s ** 2).sum()), "q_first": float(q[j]),
                         "q_rest": float(q_rest[j]), "q_cal": float(q_cal[j]), "share_of_first_tail_error": float(contrib[j]) / tail_total}
                        for j in range(TOP_SINGULAR)
                    ],
                    "tail_top_k_share_of_first_tail_error": {str(k): float(contrib[:k].sum()) / tail_total for k in (1, 5, 10, 20)},
                    "tail_top_k_q_first": {str(k): float(q[:k].sum()) for k in (1, 5, 10, 20)},
                    "tail_top_k_q_rest": {str(k): float(q_rest[:k].sum()) for k in (1, 5, 10, 20)},
                    "tail_top_k_q_cal": {str(k): float(q_cal[:k].sum()) for k in (1, 5, 10, 20)},
                }
                if dom == "wt2":
                    per_rank["methods"].setdefault(m, {})["first_contrib_wt2"] = (c_i / den0).cpu().tolist()
            per_rank["methods"].setdefault(m, {}).update({"a": a.cpu().tolist(), "w": (w / cal_den).cpu().tolist()})
            layer_out["methods"][m] = mo
            d = mo["domains"]["wt2"]
            print(f"L{layer} {m:9s} cal={mo['cal_relative_error']:.4f} cal257+={mo['cal_band_share'][3]:.3f} "
                  f"unw257+={mo['unweighted_band_share'][3]:.3f} unw_rel257+={mo['unweighted_rel_band_ratio'][3]:.3f} "
                  f"first={d['first_relative_error']:.3f} top1/5/10 of first tail err="
                  f"{d['tail_top_k_share_of_first_tail_error']['1']:.2f}/{d['tail_top_k_share_of_first_tail_error']['5']:.2f}/"
                  f"{d['tail_top_k_share_of_first_tail_error']['10']:.2f} q_top5 first/rest/cal={d['tail_top_k_q_first']['5']:.3f}/"
                  f"{d['tail_top_k_q_rest']['5']:.4f}/{d['tail_top_k_q_cal']['5']:.5f} "
                  f"gain W first/rest={d['gain']['first_W']:.4f}/{d['gain']['rest_W']:.4f} "
                  f"R first/rest={d['gain']['first_R']:.4f}/{d['gain']['rest_R']:.4f}", flush=True)
        layer_out["per_rank"] = per_rank
        results["layers"][str(layer)] = layer_out
        print(f"L{layer} done {time.perf_counter() - t0:.0f}s", flush=True)
    results["checks"] = checks
    (OUT / f"{stem}.json").write_text(json.dumps(results), encoding="utf-8")
    print("checks:", checks)


if __name__ == "__main__":
    main()
