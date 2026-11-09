"""04 の追加分析 3：attention sink（先頭のトークン）の仮説を確かめる。

置き換えた射影（k・v・q）の出力を、
    full   全部の位置で使う（通常の評価と同じ）
    sink   先頭の位置（区間の 0 番目）だけで使い、ほかは元の値
    rest   先頭以外で使い、先頭は元の値
の 3 通りにして、softmax 後の重み・head ごとの出力・o の後の誤差を測る。
あわせて、先頭の位置への注目の重みの平均と、先頭の入力の大きさ（ほかの位置との比）を記録する。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_sink_test.py
結果: results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import torch

# 分析 2 のスクリプトの関数（データの用意・attention の入力の取り出し）を使う。
_spec = importlib.util.spec_from_file_location(
    "followup_attention_stages", Path(__file__).resolve().with_name("04_followup_attention_stages.py")
)
base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base_mod)

ROOT = base_mod.ROOT
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "sink_test"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = base_mod.DEVICE
METHODS = ("M2", "M3", "M5")
PROJECTIONS = ("k", "v", "q")


def stages(attn, x, cos, sin, weights, replace, proj, where):
    """``replace`` の射影の出力を ``where``（"full"・"sink"・"rest"）の位置だけ置き換える。"""
    b, t, _ = x.shape
    d = attn.head_dim
    raw = {p: (x @ weights[p].T).view(b, t, -1, d) for p in ("q", "k", "v")}
    if where != "full":
        new = (x @ replace.T).view(b, t, -1, d)
        pos = torch.zeros(t, dtype=torch.bool, device=x.device)
        pos[0] = True
        use = pos if where == "sink" else ~pos
        raw[proj] = torch.where(use.view(1, t, 1, 1), new, raw[proj])
    else:
        raw[proj] = (x @ replace.T).view(b, t, -1, d)
    q = attn.q_norm(raw["q"]).transpose(1, 2)
    k = attn.k_norm(raw["k"]).transpose(1, 2)
    v = raw["v"].transpose(1, 2)
    q_rot, k_rot = base_mod.apply_rotary_pos_emb(q, k, cos, sin)
    groups = q.shape[1] // k.shape[1]
    k_rep = k_rot.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)
    scores = (q_rot @ k_rep.transpose(-1, -2)) * attn.scaling
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1, dtype=torch.float32)
    heads = probs @ v_rep
    out = attn.o_proj(heads.transpose(1, 2).reshape(b, t, -1))
    return {"weights": probs, "heads": heads, "out": out}


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
    results = {}
    for layer in (7, 14, 21):
        attn = model.model.layers[layer].self_attn
        base = {p: getattr(attn, f"{p}_proj").weight.detach().clone() for p in ("q", "k", "v")}
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        cell_weights = {}
        for proj in PROJECTIONS:
            cm = base_mod.build_cell_matrices(layout, layer=layer, projection=proj, budget=25,
                                              calibration_gram=grams["attn_in"]["total"],
                                              summary=summary[f"{layer}_{proj}_25"], status=status)
            cell_weights[proj] = {m: cm.matrices[m].to(device=DEVICE, dtype=torch.float32) for m in METHODS}
        for dom, dom_chunks in chunks.items():
            t0 = time.perf_counter()
            inputs = base_mod.capture_attention_inputs(model, layer, dom_chunks)
            acc = {}
            info = {"sink_mass": 0.0, "n_rows": 0, "norm_ratio": []}
            with torch.no_grad():
                for x_cpu, cos_cpu, sin_cpu in inputs:
                    x, cos, sin = x_cpu.to(DEVICE), cos_cpu.to(DEVICE), sin_cpu.to(DEVICE)
                    ref = stages(attn, x, cos, sin, base, base["q"], "q", "full")
                    info["sink_mass"] += float(ref["weights"][..., 0].double().sum())
                    info["n_rows"] += ref["weights"][..., 0].numel()
                    norms = x[0].norm(dim=-1)
                    info["norm_ratio"].append(float(norms[0] / norms[1:].median()))
                    for proj in PROJECTIONS:
                        for m in METHODS:
                            for where in ("full", "sink", "rest"):
                                new = stages(attn, x, cos, sin, base, cell_weights[proj][m], proj, where)
                                a = acc.setdefault(f"L{layer}_{proj}_25|{m}|{dom}|{where}", {})
                                for stage in ("weights", "heads", "out"):
                                    a[f"{stage}_num"] = a.get(f"{stage}_num", 0.0) + float(((new[stage] - ref[stage]).double() ** 2).sum())
                                    a[f"{stage}_den"] = a.get(f"{stage}_den", 0.0) + float((ref[stage].double() ** 2).sum())
                    del x, cos, sin, ref
            results.update(acc)
            ratios = sorted(info["norm_ratio"])
            results[f"L{layer}|{dom}|info"] = {
                "sink_mass_mean": info["sink_mass"] / info["n_rows"],
                "first_token_norm_ratio_median": ratios[len(ratios) // 2],
            }
            print(f"L{layer} {dom}: {len(inputs)} chunks {time.perf_counter() - t0:.0f}s  sink mass {info['sink_mass'] / info['n_rows']:.3f}", flush=True)
            del inputs
            torch.cuda.empty_cache()
    json.dump(results, open(OUT / "sink_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
