"""04 の追加分析 2：k・v（対照に q）を置き換えたとき、attention のどの段階で M2 と M5 の差が開くか。

Qwen3 の attention を段階ごとに自分で計算する（元の重みで、モジュールの出力と一致することを検算する）。

    q = q_norm(q_proj x), k = k_norm(k_proj x), v = v_proj x   （head ごと）
    q, k に RoPE（位置の回転）
    S = q k^T * scaling + 因果 mask,  A = softmax(S),  Z = A v,  O = o_proj(Z)

段階：
    proj    射影の出力（head ごと）
    normed  q_norm / k_norm の後（q・k だけ）
    scores  注目の点数 S（因果の範囲だけ）
    weights softmax の後の重み A
    heads   head ごとの出力 Z（o_proj の前）
    out     o_proj の後（サブ層の出力。評価の cells_sublayer.csv と同じ量）

点数の誤差 ΔS は、各 query の行で「全部の key に同じだけずれる成分（shift）」と、
「key の間の差を変える成分（contrast）」に分ける。softmax は shift を無視する。

使い方（リポジトリのどこからでも）:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_attention_stages.py
結果: results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
OUT = ROOT / "results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis" / "attention_stages"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from nn_compression.llm.hf_loading import load_causal_lm_and_tokenizer, load_download_record  # noqa: E402
from nn_compression.llm.qwen3_main_evaluation import build_cell_matrices  # noqa: E402
from nn_compression.llm.qwen3_main_experiment import layout_from_root  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "evaluate_script", ROOT / "notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py"
)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

DEVICE = "cuda"
METHODS = ("M2", "M3", "M5")
PROJECTIONS = ("k", "v", "q")
BUDGET = 25
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class _Stop(Exception):
    pass


def capture_attention_inputs(model, layer_index, chunks):
    """各区間について、attention の入力（input_layernorm の後）と位置の cos・sin を CPU に取る。"""
    attn = model.model.layers[layer_index].self_attn
    store = []

    def pre_hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cos, sin = kwargs["position_embeddings"]
        store.append((hidden.detach().cpu(), cos.detach().cpu(), sin.detach().cpu()))
        raise _Stop

    handle = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for _unit, ids in chunks:
                try:
                    model(ids.reshape(1, -1).to(DEVICE), use_cache=False)
                except _Stop:
                    pass
    finally:
        handle.remove()
    return store


def attention_stages(attn, x, cos, sin, weights):
    """weights は {"q": W_q, "k": W_k, "v": W_v}（GPU・float32）。段階ごとの値を返す。"""
    b, t, _ = x.shape
    d = attn.head_dim
    q_raw = (x @ weights["q"].T).view(b, t, -1, d)
    k_raw = (x @ weights["k"].T).view(b, t, -1, d)
    v_raw = (x @ weights["v"].T).view(b, t, -1, d)
    q = attn.q_norm(q_raw).transpose(1, 2)
    k = attn.k_norm(k_raw).transpose(1, 2)
    v = v_raw.transpose(1, 2)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    groups = q.shape[1] // k.shape[1]
    k_rep = k_rot.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)
    scores = (q_rot @ k_rep.transpose(-1, -2)) * attn.scaling
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1, dtype=torch.float32).to(q.dtype)
    heads = probs @ v_rep  # (b, Hq, t, d)
    out = attn.o_proj(heads.transpose(1, 2).reshape(b, t, -1))
    return {
        "proj_q": q_raw, "proj_k": k_raw, "proj_v": v_raw,
        "normed_q": q, "normed_k": k,
        "scores": scores, "mask": mask, "weights": probs, "heads": heads, "out": out,
    }


def sq_per_head(diff, head_dim_axis):
    return (diff.double() ** 2).sum(dim=[i for i in range(diff.ndim) if i != head_dim_axis]).cpu()


def main():
    layout = layout_from_root(ROOT)
    record = load_download_record(ROOT / "data" / "hf_cache" / "download_record.json")
    model, tokenizer = load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
    model.to(dtype=torch.float32, device=DEVICE)
    model.eval()
    units_json = json.loads((layout.validation / "units.json").read_text(encoding="utf-8"))
    units = script.validation_units(record, tokenizer, units_json)
    chunks = {dom: script.chunks_of(items) for dom, items in units.items()}
    summary = script.read_csv_map(layout.results / "methods_summary.csv", "cell")
    status = script.read_csv_map(layout.als_results("full") / "jobs_status.csv", "job_id")
    results = {}
    max_identity_gap = 0.0
    for layer in (7, 14, 21):
        attn = model.model.layers[layer].self_attn
        base = {"q": attn.q_proj.weight.detach().clone(), "k": attn.k_proj.weight.detach().clone(), "v": attn.v_proj.weight.detach().clone()}
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        cell_weights = {}
        for proj in PROJECTIONS:
            cm = build_cell_matrices(layout, layer=layer, projection=proj, budget=BUDGET,
                                     calibration_gram=grams["attn_in"]["total"], summary=summary[f"{layer}_{proj}_{BUDGET}"],
                                     status=status)
            bad = [c for c in cm.checks if not c["ok"]]
            if bad:
                raise RuntimeError(f"検算に失敗: {bad[:3]}")
            cell_weights[proj] = {m: cm.matrices[m].to(device=DEVICE, dtype=torch.float32) for m in METHODS}
        for dom, dom_chunks in chunks.items():
            t0 = time.perf_counter()
            inputs = capture_attention_inputs(model, layer, dom_chunks)
            acc = {}
            with torch.no_grad():
                for index, (x_cpu, cos_cpu, sin_cpu) in enumerate(inputs):
                    x, cos, sin = x_cpu.to(DEVICE), cos_cpu.to(DEVICE), sin_cpu.to(DEVICE)
                    ref = attention_stages(attn, x, cos, sin, base)
                    if index == 0:
                        module_out = attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=None)[0]
                        gap = float((module_out - ref["out"]).abs().max() / ref["out"].abs().max())
                        max_identity_gap = max(max_identity_gap, gap)
                    mask = ref["mask"]
                    counts = mask.sum(dim=-1).double()  # 各 query が見る key の数
                    for proj in PROJECTIONS:
                        for m in METHODS:
                            w = dict(base)
                            w[proj] = cell_weights[proj][m]
                            new = attention_stages(attn, x, cos, sin, w)
                            key = f"L{layer}_{proj}_{BUDGET}|{m}|{dom}"
                            a = acc.setdefault(key, {})
                            pairs = {"proj": (new[f"proj_{proj}"], ref[f"proj_{proj}"], 2), "heads": (new["heads"], ref["heads"], 1),
                                     "out": (new["out"], ref["out"], None)}
                            if proj in ("q", "k"):
                                pairs["normed"] = (new[f"normed_{proj}"], ref[f"normed_{proj}"], 1)
                                ds = torch.where(mask, new["scores"] - ref["scores"], 0.0)
                                s_ref = torch.where(mask, ref["scores"], 0.0)
                                pairs["scores"] = (ds + s_ref, s_ref, 1)
                                pairs["weights"] = (new["weights"], ref["weights"], 1)
                                row_mean = ds.double().sum(dim=-1) / counts  # (b, H, t)
                                shift = (row_mean ** 2 * counts).sum(dim=[0, 2]).cpu()
                                total = (ds.double() ** 2).sum(dim=[0, 2, 3]).cpu()
                                a["shift"] = a.get("shift", 0) + shift
                                a["score_total"] = a.get("score_total", 0) + total
                            for stage, (n, r, axis) in pairs.items():
                                if axis is None:
                                    num = torch.tensor([float(((n - r).double() ** 2).sum())])
                                    den = torch.tensor([float((r.double() ** 2).sum())])
                                else:
                                    num = sq_per_head(n - r, axis)
                                    den = sq_per_head(r, axis)
                                a[f"{stage}_num"] = a.get(f"{stage}_num", 0) + num
                                a[f"{stage}_den"] = a.get(f"{stage}_den", 0) + den
            for key, a in acc.items():
                results[key] = {k: v.tolist() for k, v in a.items()}
            print(f"L{layer} {dom}: {len(inputs)} chunks {time.perf_counter() - t0:.0f}s", flush=True)
            del inputs
            torch.cuda.empty_cache()
    json.dump({"results": results, "max_identity_gap": max_identity_gap}, open(OUT / "attention_stages.json", "w"), indent=1)
    print("max identity gap (manual vs module, relative):", max_identity_gap)


if __name__ == "__main__":
    main()
