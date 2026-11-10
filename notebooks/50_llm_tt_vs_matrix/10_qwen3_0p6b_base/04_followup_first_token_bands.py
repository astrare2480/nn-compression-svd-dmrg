"""04 の追加分析 5：先頭のトークンの入力は、較正 Gram 行列のどの固有方向の帯に乗っているか。

問い：v で M5 だけが先頭のトークンの値を外すのは、「先頭の入力が較正でまれな向き（下位の帯）にある」からか。

各層の attention の入力（input_layernorm の後。q・k・v で共通）について、
較正 Gram 行列 G_cal = Σ x x^T の固有ベクトル u_j（固有値 λ_j の大きい順）を使い、次を測る。

1. 入力の量の帯ごとの割合：Σ_{j∈帯} (u_j·x)^2 / Σ_j (u_j·x)^2。先頭の位置と、先頭以外で別々に。
2. 向きごとの「まれさ」：先頭の 1 トークンあたりの (u_j·x_0)^2 を、較正の 1 トークンあたりの λ_j/n_cal で割った比。
   帯ごとの中央値。1 より大きいと、その向きに較正の平均より多く乗っている。
3. 先頭の入力どうしの向きの近さ：区間ごとの x_0 の平均の cos 類似度、平均の向きが各帯に乗る割合。
4. 先頭のトークンでの射影の出力の誤差 ||R x_0||^2 / ||W x_0||^2（R = W − Ŵ）を、帯ごとに分ける。
   分け方は c_b = (R x_0)·(R P_b x_0)（P_b は帯 b の固有ベクトルへの射影）。Σ_b c_b = ||R x_0||^2 が厳密に成り立つ
   （帯の間の交差は、右側の帯に割り当てている。負の値もありうる）。
   k・v の 25%、M2・M3・M5（M3 から）。L14 は M4r から始めた M5 も。先頭以外の位置でも同じ量を出す（比較用）。

モデルは入力を取り出すところまでしか動かさない（各層で止める）。新しい ALS はない。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_first_token_bands.py
結果: results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/first_token_bands/
"""

from __future__ import annotations

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
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "first_token_bands"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = base_mod.DEVICE
BANDS = [(0, 16), (16, 64), (64, 256), (256, None)]
BAND_NAMES = ["1-16", "17-64", "65-256", "257-"]
PROJECTIONS = ("v", "k")
METHODS = ("M2", "M3", "M5")
torch.set_grad_enabled(False)


def band_slices(dim):
    return [slice(a, dim if b is None else b) for a, b in BANDS]


def main():
    layout = base_mod.layout_from_root(ROOT)
    record = base_mod.load_download_record(ROOT / "data" / "hf_cache" / "download_record.json")
    model, tokenizer = base_mod.load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
    model.to(dtype=torch.float32, device=DEVICE)
    model.eval()
    units_json = json.loads((layout.validation / "units.json").read_text(encoding="utf-8"))
    units = base_mod.script.validation_units(record, tokenizer, units_json)
    chunks = {dom: base_mod.script.chunks_of(items) for dom, items in units.items()}
    summary = base_mod.script.read_csv_map(layout.results / "methods_summary.csv", "cell")
    status = base_mod.script.read_csv_map(layout.als_results("full") / "jobs_status.csv", "job_id")
    results: dict = {"bands": BAND_NAMES, "layers": {}}
    checks = {"max_band_sum_gap": 0.0, "max_error_sum_gap": 0.0}
    for layer in (7, 14, 21):
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        g_cal = grams["attn_in"]["total"].to(DEVICE, torch.float64)
        n_cal = int(grams["attn_in"]["n_tokens"])
        values, vectors = torch.linalg.eigh(g_cal)
        order = torch.argsort(values, descending=True)
        lam = values[order].clamp_min(0.0)
        U = vectors[:, order].contiguous()
        dim = U.shape[0]
        slices = band_slices(dim)
        per_token_cal = lam / n_cal

        # 方式の行列（検算つきで作り直す）と残差 R
        residuals = {}
        weights_full = {}
        for proj in PROJECTIONS:
            cm = base_mod.build_cell_matrices(layout, layer=layer, projection=proj, budget=25,
                                              calibration_gram=grams["attn_in"]["total"],
                                              summary=summary[f"{layer}_{proj}_25"], status=status)
            bad = [c for c in cm.checks if not c["ok"]]
            if bad:
                raise RuntimeError(f"検算に失敗: {bad[:3]}")
            W = torch.load(layout.weights / f"W_L{layer}_{proj}.pt", map_location="cpu", weights_only=True)["W"].to(DEVICE)
            weights_full[proj] = W
            for m in METHODS:
                residuals[(proj, m)] = W - cm.matrices[m].to(DEVICE, torch.float64)
            m4r_path = layout.als_dir("full") / f"L{layer}_{proj}_25_M5_M4r_M4rinit_lam1e-06.pt"
            if m4r_path.is_file():
                payload = torch.load(layout.methods / f"L{layer}_{proj}_25.pt", map_location="cpu", weights_only=True)
                saved = torch.load(m4r_path, map_location="cpu", weights_only=True)
                approx = tt_matrix_to_dense(saved["cores"], payload["out_modes"], payload["in_modes"]).to(DEVICE, torch.float64)
                residuals[(proj, "M5_M4rinit")] = W - approx
        # R U は帯ごとの分解で何度も使うので先に作る
        RU = {key: R @ U for key, R in residuals.items()}
        WU = {proj: W @ U for proj, W in weights_full.items()}

        layer_out = {}
        for dom, dom_chunks in chunks.items():
            t0 = time.perf_counter()
            inputs = base_mod.capture_attention_inputs(model, layer, dom_chunks)
            energy0 = torch.zeros(dim, dtype=torch.float64, device=DEVICE)   # Σ (u_j·x0)^2
            energy_rest = torch.zeros(dim, dtype=torch.float64, device=DEVICE)
            n_rest = 0
            first_dirs = []
            err = {where: {key: {"num": 0.0, "den": 0.0, "band": [0.0] * len(BANDS)} for key in residuals}
                   for where in ("first", "rest")}
            for x_cpu, _cos, _sin in inputs:
                X = x_cpu[0].to(DEVICE, torch.float64)            # (T, d)
                coef = X @ U                                      # (T, d) 固有基底での成分
                energy0 += coef[0] ** 2
                energy_rest += (coef[1:] ** 2).sum(dim=0)
                n_rest += X.shape[0] - 1
                first_dirs.append(X[0] / X[0].norm())
                for where, c in (("first", coef[:1]), ("rest", coef[1:])):
                    for key, ru in RU.items():
                        proj = key[0]
                        full = c @ ru.T                            # (n, out) = R x
                        ref = c @ WU[proj].T                       # W x
                        e = err[where][key]
                        e["num"] += float((full ** 2).sum())
                        e["den"] += float((ref ** 2).sum())
                        for i, s in enumerate(slices):
                            part = c[:, s] @ ru[:, s].T            # R P_b x
                            e["band"][i] += float((full * part).sum())
            n_first = len(inputs)
            share0 = [float(energy0[s].sum() / energy0.sum()) for s in slices]
            share_rest = [float(energy_rest[s].sum() / energy_rest.sum()) for s in slices]
            checks["max_band_sum_gap"] = max(checks["max_band_sum_gap"], abs(sum(share0) - 1.0), abs(sum(share_rest) - 1.0))
            rarity0 = (energy0 / n_first) / per_token_cal.clamp_min(1e-300)
            rarity_rest = (energy_rest / n_rest) / per_token_cal.clamp_min(1e-300)
            dirs = torch.stack(first_dirs)
            mean_dir = dirs.mean(dim=0)
            cos_to_mean = (dirs @ (mean_dir / mean_dir.norm()))
            mean_coef = (mean_dir / mean_dir.norm()) @ U
            errors_out = {}
            for where in ("first", "rest"):
                for key, e in err[where].items():
                    label = f"{key[0]}|{key[1]}|{where}"
                    total = e["num"]
                    checks["max_error_sum_gap"] = max(checks["max_error_sum_gap"], abs(sum(e["band"]) - total) / total)
                    errors_out[label] = {
                        "relative_error": total / e["den"],
                        "band_share": [b / total for b in e["band"]],
                        "band_relative": [b / e["den"] for b in e["band"]],
                    }
            layer_out[dom] = {
                "n_first": n_first,
                "n_rest": n_rest,
                "input_share_first": share0,
                "input_share_rest": share_rest,
                "rarity_first_median": [float(rarity0[s].median()) for s in slices],
                "rarity_rest_median": [float(rarity_rest[s].median()) for s in slices],
                "rarity_first_mean": [float(rarity0[s].mean()) for s in slices],
                "top_directions_first": [
                    {"rank": int(j) + 1, "share": float(energy0[j] / energy0.sum()), "rarity": float(rarity0[j])}
                    for j in torch.argsort(energy0, descending=True)[:10].tolist()
                ],
                "first_cos_to_mean_median": float(cos_to_mean.median()),
                "first_cos_to_mean_min": float(cos_to_mean.min()),
                "mean_first_direction_share": [float((mean_coef[s] ** 2).sum()) for s in slices],
                "errors": errors_out,
            }
            print(f"L{layer} {dom}: {n_first} chunks {time.perf_counter() - t0:.0f}s "
                  f"first share={['%.3f' % v for v in share0]} rest share={['%.3f' % v for v in share_rest]}", flush=True)
            del inputs
            torch.cuda.empty_cache()
        results["layers"][str(layer)] = layer_out
    results["checks"] = checks
    (OUT / "first_token_bands.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("checks:", checks)


if __name__ == "__main__":
    main()
