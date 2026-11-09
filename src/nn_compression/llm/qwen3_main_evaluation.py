"""本実験の評価のうち、モデルを動かす部分（設計 §7.2・§7.6・§7.8）。

1 層ずつ、次の順で計算する。

1. 方式の行列を作り（CPU・float64）、較正の値と検算する。
2. 検証の文章を元のモデルに通し、置き換える層の入力から、単位（記事・文書）ごとに
   誤差の二乗和 :math:`\\langle \\Delta^\\top \\Delta, G_u \\rangle` を計算する。
   :math:`G_u` は単位 u の入力の Gram 行列。同時に、その層の入力と MLP の入力を保存する。
3. 保存した入力から 1 つの射影だけを置き換えてサブ層を計算し直し、残差を足す前の出力の誤差を測る。
4. WikiText-2 の記事ごとの滑り窓で、置き換えた層から最後までだけを計算し直して NLL を測る。
   置き換えた層より前は、どの方式でも元のモデルと同じなので計算しない。

モデルは GPU の float32（予備実験の §209 と同じ）。方式の重みは float64 から float32 に直して入れる。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nn_compression.compression.metric_low_rank import (
    regularize_gram,
    squared_output_error_from_gram,
    whitened_truncated_svd,
)
from nn_compression.llm.qwen3_main_experiment import (
    EXPECTED_MODES_AND_RANKS,
    GROUP_MODULES,
    INPUT_GROUPS,
    LAMBDA_SWEEP,
    ExperimentLayout,
    format_lambda,
)
from nn_compression.tt_mps.matrix import tt_matrix_ranks, tt_matrix_to_dense
from nn_compression.tt_mps.metric_tt import tt_parameter_count

PROJECTION_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
ATTENTION_PROJECTIONS = ("q", "k", "v", "o")
MLP_PROJECTIONS = ("gate", "up", "down")
SUBLAYER_METHODS = ("M1", "M2", "M3", "M3F", "M4", "M5")
NLL_METHODS = ("M1", "M2", "M3", "M4", "M5")
NLL_MAX_LENGTH = 1024
NLL_STRIDE = 512
CHECK_RELATIVE = 1e-9


class _Stop(Exception):
    """順伝播を途中で止めるための例外。"""


def module_at(root: nn.Module, dotted: str) -> nn.Module:
    current = root
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def relative_gap(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-300)


@contextmanager
def removing(handles: Sequence[Any]) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def swapped_weight(module: nn.Linear, weight: torch.Tensor) -> Iterator[None]:
    """``module.weight`` の中身を一時的に ``weight`` にする（同じ device・dtype の Tensor）。"""
    original = module.weight.data.clone()
    module.weight.data.copy_(weight)
    try:
        yield
    finally:
        module.weight.data.copy_(original)


# ---------------------------------------------------------------- 方式の行列


@dataclass
class CellMethods:
    """1 セルの元の重みと、方式ごとの近似（CPU・float64）。"""

    cell: str
    layer: int
    projection: str
    budget: int
    weight: torch.Tensor
    matrices: dict[str, torch.Tensor]
    stop_reasons: dict[str, str]
    checks: list[dict[str, Any]] = field(default_factory=list)


def als_job_ids(cell: str) -> dict[str, str]:
    """方式名 → ALS の仕事 id。ファイルがない方式は呼び出し側で落とす。"""
    ids = {
        "M5": f"{cell}_M5_M3init_lam1e-06",
        "M3F": f"{cell}_M3F_M3init_lam1e-06",
        "M5_M4r": f"{cell}_M5_M4r_M4rinit_lam1e-06",
    }
    for lam in LAMBDA_SWEEP:
        if lam != 1e-6:
            label = format_lambda(lam)
            ids[f"M5_lam{label}"] = f"{cell}_M5_lambda_M3init_lam{label}"
    return ids


def _check(checks: list[dict[str, Any]], name: str, ok: bool, **detail: Any) -> None:
    checks.append({"check": name, "ok": bool(ok), **detail})


def build_cell_matrices(
    layout: ExperimentLayout,
    *,
    layer: int,
    projection: str,
    budget: int,
    calibration_gram: torch.Tensor,
    summary: Mapping[str, str],
    status: Mapping[str, Mapping[str, str]],
    mode: str = "full",
) -> CellMethods:
    """方式の行列を作り、較正の値・ランク・パラメータ数を検算する（設計 §7.8 の 1・2）。"""
    cell = f"L{layer}_{projection}_{budget}"
    payload = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
    weight = torch.load(layout.weights / f"W_L{layer}_{projection}.pt", map_location="cpu", weights_only=True)["W"]
    out_modes, in_modes = payload["out_modes"], payload["in_modes"]
    checks: list[dict[str, Any]] = []
    gram_main = regularize_gram(calibration_gram, 1e-6)
    matrices: dict[str, torch.Tensor] = {
        "M1": (payload["m1_u"] * payload["m1_s"]) @ payload["m1_vh"],
        "M3": tt_matrix_to_dense(payload["m3_cores"], out_modes, in_modes),
        "M4": tt_matrix_to_dense(payload["m4_cores"], out_modes, in_modes),
        "M4r": tt_matrix_to_dense(payload["m4r_cores"], out_modes, in_modes),
    }
    for lam in LAMBDA_SWEEP:
        label = format_lambda(lam)
        fitted = whitened_truncated_svd(weight, regularize_gram(calibration_gram, lam), int(payload["matrix_rank"]))
        name = "M2" if lam == 1e-6 else f"M2_lam{label}"
        matrices[name] = fitted.weight
        expected = float(summary[f"q_m2_lam{label}"])
        gap = relative_gap(float(fitted.q_lambda), expected)
        _check(checks, f"{cell}:{name}:q_lambda", gap <= CHECK_RELATIVE, gap=gap)
    q_m3 = float(squared_output_error_from_gram(weight - matrices["M3"], gram_main))
    gap = relative_gap(q_m3, float(summary["q_m3"]))
    _check(checks, f"{cell}:M3:q_lambda", gap <= CHECK_RELATIVE, gap=gap)
    q_m1 = float(squared_output_error_from_gram(weight - matrices["M1"], gram_main))
    gap = relative_gap(q_m1, float(summary["q_m1"]))
    _check(checks, f"{cell}:M1:q_lambda", gap <= CHECK_RELATIVE, gap=gap)
    # M5_M4r は M4r から始めるので、M3 ではなく開始点 M4r と比べる。
    q_m4r = float(squared_output_error_from_gram(weight - matrices["M4r"], gram_main))
    expected_shape = EXPECTED_MODES_AND_RANKS[(projection, budget)]
    stop_reasons: dict[str, str] = {}
    for name, job_id in als_job_ids(cell).items():
        path = layout.als_dir(mode) / f"{job_id}.pt"
        if not path.is_file():
            continue
        if job_id not in status:
            raise RuntimeError(f"{job_id} が jobs_status.csv にありません。")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        cores = saved["cores"]
        matrices[name] = tt_matrix_to_dense(cores, out_modes, in_modes)
        stop_reasons[name] = str(saved["stop_reason"])
        ranks = tuple(int(r) for r in tt_matrix_ranks(cores))
        _check(
            checks,
            f"{cell}:{name}:ranks_and_parameters",
            ranks == expected_shape[2] and tt_parameter_count(cores) == int(summary["parameters_m3"]),
            ranks=list(ranks),
            parameters=tt_parameter_count(cores),
        )
        delta = weight - matrices[name]
        if name == "M3F":
            value = float(torch.sum(delta * delta))
            reference = float(summary["f_m3"])
        else:
            lam = 1e-6 if name in ("M5", "M5_M4r") else float(name.split("lam")[1])
            value = float(squared_output_error_from_gram(delta, regularize_gram(calibration_gram, lam)))
            if name == "M5_M4r":
                reference = q_m4r
            else:
                reference = float(summary["q_m3"] if lam == 1e-6 else summary[f"q_m3_lam{format_lambda(lam)}"])
        recorded = float(status[job_id]["q_lambda"])
        gap = relative_gap(value, recorded)
        _check(
            checks,
            f"{cell}:{name}:objective",
            gap <= CHECK_RELATIVE and value <= reference * (1.0 + 1e-12),
            gap=gap,
            value=value,
            start_reference=reference,
            start="M4r" if name == "M5_M4r" else "M3",
        )
    return CellMethods(cell, layer, projection, budget, weight, matrices, stop_reasons, checks)


# ---------------------------------------------------------------- 順伝播の部品


@dataclass
class LayerCaches:
    """1 ドメイン・1 層の、区間（chunk）ごとの保存。"""

    unit_of_chunk: list[int]
    hidden: list[torch.Tensor]  # 層の入力（CPU）
    mlp_in: list[torch.Tensor]  # MLP の入力（post_attention_layernorm の出力。CPU）
    kwargs_by_length: dict[int, dict[str, Any]]
    first_attn_output: torch.Tensor | None = None
    first_mlp_output: torch.Tensor | None = None


def _detach_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        elif isinstance(value, tuple) and all(isinstance(item, torch.Tensor) for item in value):
            out[key] = tuple(item.detach().clone() for item in value)
        else:
            out[key] = value
    return out


def _same_kwargs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)):
                return False
        elif isinstance(a, tuple) and a and isinstance(a[0], torch.Tensor):
            if not (isinstance(b, tuple) and len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))):
                return False
        elif a is not b and a != b:
            return False
    return True


def _record_kwargs(store: dict[int, dict[str, Any]], length: int, kwargs: Mapping[str, Any]) -> None:
    current = _detach_kwargs({key: value for key, value in kwargs.items() if key != "hidden_states"})
    if length not in store:
        store[length] = current
    elif not _same_kwargs(store[length], current):
        raise RuntimeError(f"長さ {length} の層の引数が区間によって違います（位置・mask が想定と違う）。")


def assert_uniform_attention(model: nn.Module) -> None:
    """全層が同じ注意の型であること（保存した mask を後ろの層にも使うため）。"""
    types = set(getattr(model.config, "layer_types", None) or ["full_attention"])
    if types != {"full_attention"}:
        raise RuntimeError(f"層の注意の型が一様ではありません: {types}")


def unit_quadratic_pass(
    model: nn.Module,
    layer_index: int,
    chunks: Sequence[tuple[int, torch.Tensor]],
    n_units: int,
    group_products: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str,
    keep_caches: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], LayerCaches]:
    """元のモデルで層 ``layer_index`` まで計算し、単位ごとの二次形式を返す。

    ``chunks`` は (単位の番号, トークン列) の並び。同じ単位の区間は続けて並べる。
    ``group_products[group]`` は形 ``(k, n, n)`` の float64 の行列の束 :math:`P_j = \\Delta_j^\\top \\Delta_j`。
    戻り値の 1 つ目は group → ``(n_units, k)`` の :math:`\\langle P_j, G_u \\rangle`。
    2 つ目は group → 全区間の Gram 行列の合計（保存済みの検証 Gram との検算用）。
    """
    layer = model.model.layers[layer_index]
    modules = {group: module_at(layer, path) for group, path in GROUP_MODULES.items()}
    flat = {group: products.reshape(products.shape[0], -1) for group, products in group_products.items()}
    unit_grams: dict[str, torch.Tensor | None] = {group: None for group in modules}
    totals: dict[str, torch.Tensor | None] = {group: None for group in modules}
    results = {group: np.zeros((n_units, flat[group].shape[0]), dtype=np.float64) for group in flat}
    caches = LayerCaches([], [], [], {})

    def gram_hook(group: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            features = inputs[0].detach()[0].to(torch.float64)
            gram = features.transpose(0, 1) @ features
            current = unit_grams[group]
            unit_grams[group] = gram if current is None else current + gram

        return hook

    def layer_pre_hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        if keep_caches:
            caches.hidden.append(hidden.detach().to("cpu"))
            _record_kwargs(caches.kwargs_by_length, int(hidden.shape[1]), kwargs)
        return None

    def mlp_pre_hook(_module, inputs):
        if keep_caches:
            caches.mlp_in.append(inputs[0].detach().to("cpu"))

    def attn_hook(_module, _inputs, output):
        if keep_caches and caches.first_attn_output is None:
            caches.first_attn_output = output[0].detach().to("cpu")

    def mlp_hook(_module, _inputs, output):
        if keep_caches and caches.first_mlp_output is None:
            caches.first_mlp_output = output.detach().to("cpu")

    def stop_hook(_module, _inputs, _output):
        raise _Stop

    def flush(unit: int) -> None:
        for group, gram in unit_grams.items():
            if gram is None:
                raise RuntimeError(f"単位 {unit} で {group} の hook が呼ばれていません。")
            symmetric = (gram + gram.transpose(0, 1)) * 0.5
            totals[group] = symmetric if totals[group] is None else totals[group] + symmetric
            if group in flat:
                results[group][unit] = (flat[group] @ symmetric.reshape(-1)).cpu().numpy()
            unit_grams[group] = None

    handles = [module.register_forward_pre_hook(gram_hook(group)) for group, module in modules.items()]
    handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))
    handles.append(layer.mlp.register_forward_pre_hook(mlp_pre_hook))
    handles.append(layer.self_attn.register_forward_hook(attn_hook))
    handles.append(layer.mlp.register_forward_hook(mlp_hook))
    handles.append(layer.register_forward_hook(stop_hook))
    model.eval()
    current_unit = None
    with removing(handles), torch.no_grad():
        for unit, token_ids in chunks:
            if current_unit is not None and unit != current_unit:
                flush(current_unit)
            current_unit = unit
            if keep_caches:
                caches.unit_of_chunk.append(unit)
            try:
                model(token_ids.reshape(1, -1).to(device), use_cache=False)
            except _Stop:
                pass
            else:
                raise RuntimeError("層の後ろで止まりませんでした。")
        if current_unit is not None:
            flush(current_unit)
    return results, {group: total.cpu() for group, total in totals.items() if total is not None}, caches


def run_attention(layer: nn.Module, hidden: torch.Tensor, kwargs: Mapping[str, Any]) -> torch.Tensor:
    """層の入力から、残差を足す前の attention の出力（o_proj のあと）を返す。"""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output[0])
        raise _Stop

    handle = layer.self_attn.register_forward_hook(hook)
    try:
        layer(hidden, **kwargs)
    except _Stop:
        pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("attention の出力を取れませんでした。")
    return captured[0]


def sublayer_pass(
    model: nn.Module,
    layer_index: int,
    caches: LayerCaches,
    n_units: int,
    cells: Sequence[CellMethods],
    *,
    device: torch.device | str,
    methods: Sequence[str] = SUBLAYER_METHODS,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """1 つの射影だけを置き換え、サブ層の出力の誤差を単位ごとに足す（設計 §7.6）。

    戻り値：cell → 方式 → 単位ごとの誤差の二乗和、サブ層（"attn"・"mlp"）→ 元の出力の二乗和、
    up のセル → {"num_M?", "den"} の E_gated、検算のリスト。
    検算：元の重みそのものを入れた出力が元の出力とビット単位で一致すること（§7.8 の 4）、
    保存した入力から計算し直した元の出力が、最初の順伝播で取った出力と一致すること。
    """
    layer = model.model.layers[layer_index]
    act = layer.mlp.act_fn
    gpu_weights: dict[str, dict[str, torch.Tensor]] = {}
    for cell in cells:
        names = [name for name in methods if name in cell.matrices]
        gpu_weights[cell.cell] = {name: cell.matrices[name].to(device=device, dtype=torch.float32) for name in names}
        gpu_weights[cell.cell]["W"] = cell.weight.to(device=device, dtype=torch.float32)
    nums = {cell.cell: {name: np.zeros(n_units) for name in gpu_weights[cell.cell] if name != "W"} for cell in cells}
    dens = {"attn": np.zeros(n_units), "mlp": np.zeros(n_units)}
    gated = {
        cell.cell: {**{f"num_{name}": np.zeros(n_units) for name in gpu_weights[cell.cell] if name != "W"}, "den": np.zeros(n_units)}
        for cell in cells
        if cell.projection == "up"
    }
    checks: list[dict[str, Any]] = []
    identity_ok = True
    model.eval()

    def sq(x: torch.Tensor) -> float:
        return float(torch.sum(x.to(torch.float64) ** 2))

    with torch.no_grad():
        for index, unit in enumerate(caches.unit_of_chunk):
            hidden = caches.hidden[index].to(device)
            kwargs = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in caches.kwargs_by_length[int(hidden.shape[1])].items()
            }
            mlp_in = caches.mlp_in[index].to(device)
            y_attn = run_attention(layer, hidden, kwargs)
            y_mlp = layer.mlp(mlp_in)
            if index == 0:
                same_attn = torch.equal(y_attn.cpu(), caches.first_attn_output)
                same_mlp = torch.equal(y_mlp.cpu(), caches.first_mlp_output)
                _check(checks, f"L{layer_index}:replay_equals_forward", same_attn and same_mlp)
            dens["attn"][unit] += sq(y_attn)
            dens["mlp"][unit] += sq(y_mlp)
            gate_out = act(layer.mlp.gate_proj(mlp_in))
            up_out = layer.mlp.up_proj(mlp_in)
            for cell in cells:
                module = module_at(layer, PROJECTION_PATHS[cell.projection])
                for name, weight in gpu_weights[cell.cell].items():
                    with swapped_weight(module, weight):
                        if cell.projection in ATTENTION_PROJECTIONS:
                            y = run_attention(layer, hidden, kwargs)
                            reference = y_attn
                        else:
                            y = layer.mlp(mlp_in)
                            reference = y_mlp
                    if name == "W":
                        identity_ok = identity_ok and torch.equal(y, reference)
                        continue
                    nums[cell.cell][name][unit] += sq(y - reference)
                    if cell.projection == "up":
                        delta = (weight - gpu_weights[cell.cell]["W"]).to(torch.float32)
                        gated[cell.cell][f"num_{name}"][unit] += sq(gate_out * (mlp_in @ delta.transpose(0, 1)))
                if cell.projection == "up":
                    gated[cell.cell]["den"][unit] += sq(gate_out * up_out)
    _check(checks, f"L{layer_index}:sublayer_identity_with_original_weight", identity_ok)
    return nums, dens, gated, checks


# ---------------------------------------------------------------- NLL


def article_windows(n_tokens: int, max_length: int = NLL_MAX_LENGTH, stride: int = NLL_STRIDE) -> list[tuple[int, int, int]]:
    """``sliding_window_perplexity`` と同じ窓（begin, end, 評価する新しいトークンの数）。"""
    windows = []
    previous_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_length, n_tokens)
        windows.append((begin, end, end - previous_end))
        previous_end = end
        if end == n_tokens:
            break
    return windows


def window_labels(chunk: torch.Tensor, target_length: int) -> torch.Tensor:
    labels = chunk.clone()
    if target_length < chunk.shape[1]:
        labels[:, :-target_length] = -100
    return labels


def window_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """1 トークンずらした交差エントロピーの合計（float64 で足す）と、数えたトークン数。"""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    scored = int(mask.sum())
    if scored == 0:
        raise RuntimeError("評価対象のトークンが 0 です。")
    losses = nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    )
    return float(losses.to(torch.float64).sum()), scored


@dataclass
class WindowCaches:
    article_of_window: list[int]
    hidden: list[torch.Tensor]  # 層の入力（GPU）
    labels: list[torch.Tensor]  # GPU
    kwargs_by_length: dict[int, dict[str, Any]]


def full_model_article_nll(
    model: nn.Module,
    articles: Sequence[Sequence[int]],
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """元のモデル全体で、記事ごとの NLL の合計とトークン数。"""
    sums = np.zeros(len(articles))
    tokens = np.zeros(len(articles))
    model.eval()
    with torch.no_grad():
        for index, ids in enumerate(articles):
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=device).reshape(1, -1)
            for begin, end, target in article_windows(len(ids)):
                chunk = ids_tensor[:, begin:end]
                logits = model(chunk, use_cache=False).logits
                value, scored = window_nll_sum(logits, window_labels(chunk, target))
                sums[index] += value
                tokens[index] += scored
    return sums, tokens


def cache_window_inputs(
    model: nn.Module,
    layer_index: int,
    articles: Sequence[Sequence[int]],
    *,
    device: torch.device | str,
) -> WindowCaches:
    """各窓について、層 ``layer_index`` の入力を保存する（その層の手前で止める）。"""
    layer = model.model.layers[layer_index]
    caches = WindowCaches([], [], [], {})

    def pre_hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        caches.hidden.append(hidden.detach().clone())
        _record_kwargs(caches.kwargs_by_length, int(hidden.shape[1]), kwargs)
        raise _Stop

    model.eval()
    with removing([layer.register_forward_pre_hook(pre_hook, with_kwargs=True)]), torch.no_grad():
        for index, ids in enumerate(articles):
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=device).reshape(1, -1)
            for begin, end, target in article_windows(len(ids)):
                chunk = ids_tensor[:, begin:end]
                try:
                    model(chunk, use_cache=False)
                except _Stop:
                    pass
                caches.article_of_window.append(index)
                caches.labels.append(window_labels(chunk, target))
    return caches


def nll_from_layer(
    model: nn.Module,
    layer_index: int,
    caches: WindowCaches,
    n_articles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """保存した入力から、層 ``layer_index`` 以降だけを計算して記事ごとの NLL の合計を返す。"""
    layers = model.model.layers
    sums = np.zeros(n_articles)
    tokens = np.zeros(n_articles)
    with torch.no_grad():
        for hidden, labels, article in zip(caches.hidden, caches.labels, caches.article_of_window, strict=True):
            kwargs = caches.kwargs_by_length[int(hidden.shape[1])]
            state = hidden
            for index in range(layer_index, len(layers)):
                state = layers[index](state, **kwargs)
            logits = model.lm_head(model.model.norm(state))
            value, scored = window_nll_sum(logits, labels)
            sums[article] += value
            tokens[article] += scored
    return sums, tokens


def nll_pass(
    model: nn.Module,
    layer_index: int,
    caches: WindowCaches,
    n_articles: int,
    cells: Sequence[CellMethods],
    *,
    device: torch.device | str,
    methods: Sequence[str] = NLL_METHODS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """セル × 方式ごとに、記事ごとの NLL の合計を返す。"""
    layer = model.model.layers[layer_index]
    out: dict[str, dict[str, np.ndarray]] = {}
    for cell in cells:
        module = module_at(layer, PROJECTION_PATHS[cell.projection])
        out[cell.cell] = {}
        for name in methods:
            started = time.perf_counter()
            weight = cell.matrices[name].to(device=device, dtype=torch.float32)
            with swapped_weight(module, weight):
                sums, _tokens = nll_from_layer(model, layer_index, caches, n_articles)
            out[cell.cell][name] = sums
            if progress is not None:
                progress(f"{cell.cell} {name} nll={sums.sum():.6e} {time.perf_counter() - started:.1f}s")
    return out


def finite_or_raise(values: Mapping[str, np.ndarray], label: str) -> None:
    for name, array in values.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{label}:{name} に有限でない値があります。")


def mean_nll(sums: np.ndarray, tokens: np.ndarray) -> float:
    return float(sums.sum() / tokens.sum())


def perplexity(sums: np.ndarray, tokens: np.ndarray) -> float:
    return math.exp(mean_nll(sums, tokens))


def group_of(projection: str) -> str:
    return INPUT_GROUPS[projection]


def output_dir(layout: ExperimentLayout) -> Path:
    return layout.results / "evaluation"
