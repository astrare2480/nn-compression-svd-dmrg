"""本実験の準備と、ALS の仕事を 1 本ずつ実行する処理。"""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from nn_compression.compression.metric_low_rank import regularize_gram
from nn_compression.llm.activation_capture import accumulate_input_grams
from nn_compression.llm.hf_loading import load_causal_lm_and_tokenizer, load_download_record, load_wikitext
from nn_compression.llm.qwen3_main_experiment import (
    BUDGETS,
    GROUP_MODULES,
    INPUT_GROUPS,
    LAYERS,
    PROJECTION_ORDER,
    SHAPES,
    STATUS_FIELDS,
    ExperimentLayout,
    build_cell_methods,
    build_job_list,
    check_finished_objective,
    dump_jobs,
    exclusive_lock,
    file_sha256,
    finished_job_ids,
    format_lambda,
    history_row,
    initial_cores,
    layout_from_root,
    load_status_rows,
    read_tau_warn,
    solver_config,
    study_rule,
    upsert_csv,
    write_methods_summary,
)
from nn_compression.llm.text_documents import (
    document_sha256,
    hash_overlaps,
    read_c4_texts,
    restore_wikitext_articles,
    select_token_prefix_sequences,
)
from nn_compression.tt_mps.matrix import tt_matrix_to_dense
from nn_compression.tt_mps.metric_tt_solver import MetricTTSolverConfig, StudyResolutionStopRule
from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_json

SEQUENCE_LENGTH = 1024
CALIBRATION_LIMIT = 128
SEED = 0


def _encode(tokenizer):
    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    return encode


def select_prefix_early(documents, encode, *, source: str, limit: int):
    """公式の抽出と同じ順で、``limit`` 本そろった時点で止める。

    先頭 ``limit`` 本の hash とトークンは、公式関数を最後まで走らせたときの
    先頭 ``limit`` 本と同じである。
    """
    import numpy as np

    order = np.random.default_rng(SEED).permutation(len(documents))
    sequences = []
    for position in order:
        text = documents[int(position)]
        token_ids = tuple(int(token) for token in encode(text))
        if len(token_ids) < SEQUENCE_LENGTH:
            continue
        sequences.append(
            {
                "document_hash": document_sha256(text),
                "source": source,
                "token_ids": token_ids[:SEQUENCE_LENGTH],
                "n_tokens_in_document": len(token_ids),
            }
        )
        if len(sequences) >= limit:
            break
    return sequences


def official_sequences(documents, encode, *, source: str, limit: int) -> list[dict[str, Any]]:
    selected = select_token_prefix_sequences(
        documents,
        encode,
        source=source,
        min_tokens=SEQUENCE_LENGTH,
        length=SEQUENCE_LENGTH,
        limit=limit,
        seed=SEED,
    )
    return [
        {
            "document_hash": item.document_hash,
            "source": item.source,
            "token_ids": item.token_ids,
            "n_tokens_in_document": item.n_tokens_in_document,
        }
        for item in selected.sequences
    ]


def article_units(articles: Sequence[str], encode, *, limit: int | None) -> list[dict[str, Any]]:
    """1024 トークン以上の記事を、重なりなしの区間に切る。"""
    from nn_compression.llm.qwen3_main_experiment import chunk_token_spans

    units = []
    for article in articles:
        token_ids = tuple(int(token) for token in encode(article))
        if len(token_ids) < SEQUENCE_LENGTH:
            continue
        spans = chunk_token_spans(len(token_ids), SEQUENCE_LENGTH)
        units.append(
            {
                "id": document_sha256(article),
                "hash": document_sha256(article),
                "n_tokens": len(token_ids),
                "spans": spans,
                "token_ids": token_ids,
            }
        )
        if limit is not None and len(units) >= limit:
            break
    return units


def _module(layer, dotted: str):
    current = layer
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def _layer_modules(model, layer_index: int) -> dict[str, torch.nn.Module]:
    layer = model.model.layers[layer_index]
    return {name: _module(layer, path) for name, path in GROUP_MODULES.items()}


def _assert_shared_inputs(model, token_ids: torch.Tensor, device: str) -> None:
    for layer_index in LAYERS:
        layer = model.model.layers[layer_index]
        names = {
            "q": layer.self_attn.q_proj,
            "k": layer.self_attn.k_proj,
            "v": layer.self_attn.v_proj,
            "gate": layer.mlp.gate_proj,
            "up": layer.mlp.up_proj,
        }
        captured: dict[str, torch.Tensor] = {}
        handles = []
        for name, module in names.items():
            def hook(_module, inputs, name=name):
                captured[name] = inputs[0].detach()

            handles.append(module.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                model(token_ids.reshape(1, -1).to(device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if not (torch.equal(captured["q"], captured["k"]) and torch.equal(captured["q"], captured["v"])):
            raise RuntimeError(f"layer {layer_index} の q・k・v の入力が一致しません。")
        if not torch.equal(captured["gate"], captured["up"]):
            raise RuntimeError(f"layer {layer_index} の gate・up の入力が一致しません。")


def _save_weights(model, weights_dir: Path) -> list[dict[str, Any]]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer_index in LAYERS:
        layer = model.model.layers[layer_index]
        for projection in PROJECTION_ORDER:
            module = _module(layer, _projection_path(projection))
            if module.bias is not None:
                raise RuntimeError(f"L{layer_index} {projection} に bias があります。")
            weight = module.weight.detach().to(dtype=torch.float64, device="cpu").contiguous()
            if tuple(weight.shape) != SHAPES[projection]:
                raise RuntimeError(f"L{layer_index} {projection} の shape が違います: {tuple(weight.shape)}")
            path = weights_dir / f"W_L{layer_index}_{projection}.pt"
            atomic_torch_save(path, {"W": weight, "shape": list(weight.shape)})
            rows.append({"layer": layer_index, "projection": projection, "shape": list(weight.shape)})
    return rows


def _projection_path(projection: str) -> str:
    return {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }[projection]


def _sequences_to_tensors(sequences: Sequence[Mapping[str, Any]], key: str = "token_ids") -> list[torch.Tensor]:
    return [torch.tensor(item[key], dtype=torch.long) for item in sequences]


def _chunk_tensors(units: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
    tensors = []
    for unit in units:
        ids = unit["token_ids"]
        for start, end in unit["spans"]:
            tensors.append(torch.tensor(ids[start:end], dtype=torch.long))
    return tensors


def _compare_existing_grams(saved: Mapping[str, Any], existing_path: Path) -> None:
    if not existing_path.is_file():
        raise FileNotFoundError(existing_path)
    existing = torch.load(existing_path, map_location="cpu", weights_only=True)
    pairs = (
        ("mlp_in", "up_proj_total"),
        ("down_in", "down_proj_total"),
    )
    for group, key in pairs:
        left = saved[group]["total"]
        right = existing[key]
        if tuple(left.shape) != tuple(right.shape) or not torch.equal(left, right):
            gap = float(torch.max(torch.abs(left - right))) if tuple(left.shape) == tuple(right.shape) else float("inf")
            raise RuntimeError(f"layer 14 の {group} が {key} とビット単位で一致しません。最大差={gap}")


def prepare_experiment_data(project_root: Path, *, limit: int | None, device: str = "cuda") -> dict[str, Any]:
    """較正・検証の Gram と、float64 の重みを書く。``limit`` があると本数を絞る。"""
    layout = layout_from_root(project_root)
    record = load_download_record(project_root / "data" / "hf_cache" / "download_record.json")
    model, tokenizer = load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
    if len(model.model.layers) != 28:
        raise RuntimeError(f"層数が {len(model.model.layers)} です。")
    shapes = _save_weights(model, layout.weights)
    model.to(dtype=torch.float32, device=device)
    encode = _encode(tokenizer)
    train = load_wikitext(record, "wikitext-103-raw-v1", "train")
    documents = restore_wikitext_articles(row["text"] for row in train)
    if limit is None:
        calibration = official_sequences(documents, encode, source="wikitext-103-train", limit=CALIBRATION_LIMIT)
        pilot_path = (
            project_root
            / "results"
            / "50_llm_tt_vs_matrix"
            / "10_qwen3_0p6b_base"
            / "01_single_layer_pilot"
            / "data_preparation.json"
        )
        expected = json.loads(pilot_path.read_text(encoding="utf-8"))["splits"]["calibration"]["document_hashes"]
        hashes = [item["document_hash"] for item in calibration]
        if hashes != expected:
            raise RuntimeError("較正 128 本の文書 hash がパイロットと一致しません。")
    else:
        calibration = select_prefix_early(documents, encode, source="wikitext-103-train", limit=limit)
    first = _sequences_to_tensors(calibration[:1])[0]
    _assert_shared_inputs(model, first, device)
    modules = {}
    for layer_index in LAYERS:
        for name, module in _layer_modules(model, layer_index).items():
            modules[f"L{layer_index}_{name}"] = module
    grams = accumulate_input_grams(model, modules, _sequences_to_tensors(calibration), device=device)
    layout.calibration.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, dict[str, Any]] = {}
    for layer_index in LAYERS:
        payload = {}
        for name in GROUP_MODULES:
            key = f"L{layer_index}_{name}"
            payload[name] = {"total": grams[key]["total"], "n_tokens": grams[key]["n_tokens"]}
        by_layer[layer_index] = payload
        atomic_torch_save(layout.calibration / f"grams_L{layer_index}.pt", payload)
    if limit is None:
        _compare_existing_grams(by_layer[14], layout.data / "calibration_grams.pt")
    atomic_write_json(
        layout.calibration / "document_hashes.json",
        {"hashes": [item["document_hash"] for item in calibration], "limited": limit is not None},
    )
    wt2 = load_wikitext(record, "wikitext-2-raw-v1", "validation")
    wt2_units = article_units(restore_wikitext_articles(row["text"] for row in wt2), encode, limit=limit)
    c4_documents = read_c4_texts(record["c4"]["path"])
    if limit is None:
        c4_sequences = official_sequences(c4_documents, encode, source="c4-validation", limit=CALIBRATION_LIMIT)
    else:
        c4_sequences = select_prefix_early(c4_documents, encode, source="c4-validation", limit=limit)
    c4_units = [
        {
            "id": item["document_hash"],
            "hash": item["document_hash"],
            "n_tokens": SEQUENCE_LENGTH,
            "spans": [[0, SEQUENCE_LENGTH]],
            "token_ids": item["token_ids"],
        }
        for item in c4_sequences
    ]
    overlaps = hash_overlaps(
        {
            "calibration": [item["document_hash"] for item in calibration],
            "wt2": [item["hash"] for item in wt2_units],
            "c4": [item["hash"] for item in c4_units],
        }
    )
    if overlaps:
        raise RuntimeError(f"較正と検証の文書 hash が重なっています: {overlaps}")
    public_units = {
        "wt2": [{key: unit[key] for key in ("id", "hash", "n_tokens", "spans")} for unit in wt2_units],
        "c4": [{key: unit[key] for key in ("id", "hash", "n_tokens", "spans")} for unit in c4_units],
    }
    layout.validation.mkdir(parents=True, exist_ok=True)
    atomic_write_json(layout.validation / "units.json", public_units)
    for domain, tensors in (("wt2", _chunk_tensors(wt2_units)), ("c4", _chunk_tensors(c4_units))):
        domain_grams = accumulate_input_grams(model, modules, tensors, device=device)
        for layer_index in LAYERS:
            payload = {}
            for name in GROUP_MODULES:
                key = f"L{layer_index}_{name}"
                payload[name] = {"total": domain_grams[key]["total"], "n_tokens": domain_grams[key]["n_tokens"]}
            atomic_torch_save(layout.validation / f"grams_{domain}_L{layer_index}.pt", payload)
    manifest = {
        "limited": limit is not None,
        "limit": limit,
        "n_calibration": len(calibration),
        "n_wt2": len(wt2_units),
        "n_c4": len(c4_units),
        "model_revision": record["model"]["revision"],
        "weight_shapes": shapes,
    }
    atomic_write_json(layout.manifest, manifest)
    return manifest


def build_requested_methods(project_root: Path, *, only: str | None = None) -> list[dict[str, Any]]:
    """方式を作る。``only`` は ``L14_k_25`` のようなセル指定。"""
    layout = layout_from_root(project_root)
    layout.methods.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in LAYERS:
        grams = torch.load(layout.calibration / f"grams_L{layer}.pt", map_location="cpu", weights_only=True)
        for projection in PROJECTION_ORDER:
            for budget in BUDGETS:
                cell = f"L{layer}_{projection}_{budget}"
                if only and only not in cell:
                    continue
                weight = torch.load(
                    layout.weights / f"W_L{layer}_{projection}.pt",
                    map_location="cpu",
                    weights_only=True,
                )["W"]
                group = INPUT_GROUPS[projection]
                payload, row = build_cell_methods(
                    weight,
                    grams[group]["total"],
                    int(grams[group]["n_tokens"]),
                    layer=layer,
                    projection=projection,
                    budget_percent=budget,
                )
                atomic_torch_save(layout.methods / f"{cell}.pt", payload)
                rows.append(row)
                print(f"{cell} {row['bond_class']} {row['seconds']:.1f}s", flush=True)
    if rows:
        write_methods_summary(layout.results / "methods_summary.csv", rows)
    return rows


def _summary_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["cell"]: row for row in csv.DictReader(handle)}


def _validation_errors(layout: ExperimentLayout, job: dict[str, Any], cores, out_modes, in_modes, weight):
    errors: dict[str, float | None] = {"e_wt2": None, "e_c4": None}
    approximated = None
    for domain in ("wt2", "c4"):
        path = layout.validation / f"grams_{domain}_L{job['layer']}.pt"
        if not path.is_file():
            continue
        if approximated is None:
            approximated = tt_matrix_to_dense(cores, out_modes, in_modes)
        gram = torch.load(path, map_location="cpu", weights_only=True)[job["group"]]["total"]
        numerator = float(torch.sum((weight - approximated) * ((weight - approximated) @ gram)))
        denominator = float(torch.sum(weight * (weight @ gram)))
        errors[f"e_{domain}"] = None if denominator == 0.0 else numerator / denominator
    return errors


def execute_study_job(
    job: dict[str, Any],
    *,
    layout: ExperimentLayout,
    config: MetricTTSolverConfig,
    rule: StudyResolutionStopRule,
    checkpoint_path: Path,
    progress: Callable[..., None] | None = None,
) -> Any:
    """1 本の ALS。終わった保存があればソルバーが計算せず戻る。"""
    cell = f"L{job['layer']}_{job['projection']}_{job['budget_percent']}"
    method = torch.load(layout.methods / f"{cell}.pt", map_location="cpu", weights_only=True)
    weight = torch.load(
        layout.weights / f"W_L{job['layer']}_{job['projection']}.pt",
        map_location="cpu",
        weights_only=True,
    )["W"]
    grams = torch.load(
        layout.calibration / f"grams_L{job['layer']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    raw = grams[job["group"]]["total"]
    if job["kind"] == "M3F":
        gram = torch.eye(weight.shape[1], dtype=torch.float64)
        report = raw
    else:
        gram = regularize_gram(raw, float(job["lambda"]))
        report = raw
    record_path = None
    for parent in (layout.data, *layout.data.parents):
        candidate = parent / "hf_cache" / "download_record.json"
        if candidate.is_file():
            record_path = candidate
            break
    revision = ""
    if record_path is not None:
        revision = json.loads(record_path.read_text(encoding="utf-8"))["model"]["revision"]
    metadata = {
        "job_id": job["id"],
        "model_revision": revision,
        "weight_sha256": file_sha256(layout.weights / f"W_L{job['layer']}_{job['projection']}.pt"),
        "gram_sha256": file_sha256(layout.calibration / f"grams_L{job['layer']}.pt"),
        "method_sha256": file_sha256(layout.methods / f"{cell}.pt"),
    }
    out_modes = method["out_modes"]
    in_modes = method["in_modes"]

    def diagnostics(cores, sweep):
        values = _validation_errors(layout, job, cores, out_modes, in_modes, weight)
        values["sweep"] = sweep
        return values

    from nn_compression.llm.qwen3_main_experiment import run_one_study

    return run_one_study(
        job,
        weight=weight,
        gram=gram,
        report_gram=report,
        cores=initial_cores(method, job["init"]),
        out_modes=out_modes,
        in_modes=in_modes,
        config=config,
        rule=rule,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        diagnostics=diagnostics,
        progress=progress,
    )


def _status_row(job: dict[str, Any], result, check: dict[str, Any]) -> dict[str, Any]:
    last = result.history[-1] if result.history else None
    return {
        "job_id": job["id"],
        "kind": job["kind"],
        "layer": job["layer"],
        "projection": job["projection"],
        "budget_percent": job["budget_percent"],
        "stop_reason": result.stop_reason,
        "sweeps": 0 if last is None else last.sweep,
        "q_lambda": None if last is None else last.q_lambda,
        "delta_50": None if last is None else last.delta_50,
        "s_max": None if last is None else last.stationarity_max,
        "n_warnings": result.n_warnings,
        "n_stage2": result.n_stage2,
        "n_stage3": result.n_stage3,
        "seconds": result.seconds_total,
        "peak_memory_bytes": None if last is None else last.peak_memory_bytes,
        "check_ok": check.get("check_ok"),
        "check_relative_gap": check.get("check_relative_gap"),
        "check_detail": check.get("check_detail"),
    }


def write_job_outputs(job, result, layout: ExperimentLayout, mode: str, summary: dict[str, str]) -> dict[str, Any]:
    errors_by_sweep = {int(item["sweep"]): item for item in result.diagnostics}
    rows = []
    for record in result.history:
        saved = errors_by_sweep.get(record.sweep, {})
        rows.append(history_row(record, {"e_wt2": saved.get("e_wt2"), "e_c4": saved.get("e_c4")}))
    history_path = layout.als_results(mode) / f"{job['id']}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    from nn_compression.llm.qwen3_main_experiment import HISTORY_FIELDS
    from nn_compression.utils.artifacts import atomic_write_csv

    atomic_write_csv(history_path, rows, HISTORY_FIELDS)
    lam = format_lambda(float(job["lambda"]))
    if job["kind"] == "M3F":
        reference = float(summary["f_m3"])
    else:
        reference = float(summary["q_m3"] if float(job["lambda"]) == 1e-6 else summary[f"q_m3_lam{lam}"])
    if not result.history:
        check = {"check_ok": False, "check_relative_gap": None, "check_detail": "掃引がありません"}
    else:
        method = torch.load(
            layout.methods / f"L{job['layer']}_{job['projection']}_{job['budget_percent']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        weight = torch.load(
            layout.weights / f"W_L{job['layer']}_{job['projection']}.pt",
            map_location="cpu",
            weights_only=True,
        )["W"]
        if job["kind"] == "M3F":
            gram_for_check = torch.eye(weight.shape[1], dtype=torch.float64)
        else:
            raw = torch.load(
                layout.calibration / f"grams_L{job['layer']}.pt",
                map_location="cpu",
                weights_only=True,
            )[job["group"]]["total"]
            gram_for_check = regularize_gram(raw, float(job["lambda"]))
        recorded = float(result.history[-1].q_lambda)
        check = check_finished_objective(
            kind=job["kind"],
            weight=weight,
            gram=gram_for_check,
            cores=result.cores,
            out_modes=method["out_modes"],
            in_modes=method["in_modes"],
            recorded_q=recorded,
            reference_q=reference,
        )
        del method, weight
    status = _status_row(job, result, check)
    status_path = layout.als_results(mode) / "jobs_status.csv"
    with exclusive_lock(layout.als_results(mode) / "jobs_status.lock"):
        upsert_csv(status_path, [status], STATUS_FIELDS, "job_id")
    return status


def d_ref_for(job: dict[str, Any], summary: dict[str, str]) -> float:
    if job["kind"] == "M3F":
        return float(summary["d_f_ref"])
    return float(summary[f"d_ref_lam{format_lambda(float(job['lambda']))}"])


def assert_production_data(layout: ExperimentLayout, mode: str) -> None:
    if mode == "check":
        return
    if not layout.manifest.is_file():
        raise FileNotFoundError(f"データの準備がありません: {layout.manifest}")
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    if manifest.get("limited"):
        raise RuntimeError("較正が --limit のままです。本数を絞らない準備を先に実行してください。")


def run_job_loop(
    jobs: Sequence[dict[str, Any]],
    *,
    layout: ExperimentLayout,
    mode: str,
    max_sweeps: int,
    tau_warn: float,
    device: str,
    lock_dir: Path,
    owner: str,
    raise_after_sweep: int | None = None,
    window_sweeps: int = 50,
    config: MetricTTSolverConfig | None = None,
) -> None:
    """取れる仕事がなくなるまで実行する。数値の失敗は記録して次へ進む。"""
    config = solver_config(device) if config is None else config
    summaries = _summary_map(layout.results / "methods_summary.csv")
    while True:
        done = finished_job_ids(layout.als_results(mode) / "jobs_status.csv")
        from nn_compression.llm.qwen3_main_experiment import claim_next_job

        job = claim_next_job(jobs, lock_dir, done, owner)
        if job is None:
            return
        cell = f"{job['layer']}_{job['projection']}_{job['budget_percent']}"
        summary = summaries[cell]
        rule = study_rule(
            d_ref_for(job, summary),
            max_sweeps=max_sweeps,
            tau_warn=tau_warn,
            window_sweeps=window_sweeps,
        )
        checkpoint = layout.als_dir(mode) / f"{job['id']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        def progress(record, job_id=job["id"]):
            if record.sweep % 10 == 0 or record.stop_reason:
                print(
                    f"{job_id} sweep={record.sweep} Q={record.q_lambda} "
                    f"delta50={record.delta_50} stage2={record.n_stage2} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if raise_after_sweep is not None and record.sweep >= raise_after_sweep:
                raise InterruptedError(job_id)

        try:
            result = execute_study_job(
                job,
                layout=layout,
                config=config,
                rule=rule,
                checkpoint_path=checkpoint,
                progress=progress,
            )
        except (InterruptedError, KeyboardInterrupt):
            status = {
                "job_id": job["id"],
                "kind": job["kind"],
                "layer": job["layer"],
                "projection": job["projection"],
                "budget_percent": job["budget_percent"],
                "stop_reason": "resource_interrupted",
                "sweeps": "",
                "q_lambda": "",
                "delta_50": "",
                "s_max": "",
                "n_warnings": "",
                "n_stage2": "",
                "n_stage3": "",
                "seconds": "",
                "peak_memory_bytes": "",
                "check_ok": "",
                "check_relative_gap": "",
                "check_detail": "resource_interrupted",
            }
            with exclusive_lock(layout.als_results(mode) / "jobs_status.lock"):
                upsert_csv(layout.als_results(mode) / "jobs_status.csv", [status], STATUS_FIELDS, "job_id")
            (lock_dir / f"{job['id']}.lock").unlink(missing_ok=True)
            raise
        except Exception as exc:
            status = {
                "job_id": job["id"],
                "kind": job["kind"],
                "layer": job["layer"],
                "projection": job["projection"],
                "budget_percent": job["budget_percent"],
                "stop_reason": "check_failed",
                "sweeps": "",
                "q_lambda": "",
                "delta_50": "",
                "s_max": "",
                "n_warnings": "",
                "n_stage2": "",
                "n_stage3": "",
                "seconds": "",
                "peak_memory_bytes": "",
                "check_ok": False,
                "check_relative_gap": "",
                "check_detail": f"{type(exc).__name__}: {exc}",
            }
            with exclusive_lock(layout.als_results(mode) / "jobs_status.lock"):
                upsert_csv(layout.als_results(mode) / "jobs_status.csv", [status], STATUS_FIELDS, "job_id")
            (lock_dir / f"{job['id']}.lock").unlink(missing_ok=True)
            print(f"{job['id']} を記録して次へ進みます: {exc}", flush=True)
            continue
        try:
            write_job_outputs(job, result, layout, mode, summary)
        finally:
            (lock_dir / f"{job['id']}.lock").unlink(missing_ok=True)


def ensure_jobs_file(layout: ExperimentLayout) -> list[dict[str, Any]]:
    jobs = build_job_list()
    dump_jobs(layout.results / "jobs.json", jobs)
    return jobs


def worker_process(spec: dict[str, Any]) -> None:
    run_job_loop(
        spec["jobs"],
        layout=ExperimentLayout(Path(spec["data"]), Path(spec["models"]), Path(spec["results"])),
        mode=spec["mode"],
        max_sweeps=spec["max_sweeps"],
        tau_warn=spec["tau_warn"],
        device=spec["device"],
        lock_dir=Path(spec["lock_dir"]),
        owner=str(os.getpid()),
    )


def read_history_timing(path: Path) -> list[float]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        sweep = int(row["sweep"])
        if 2 <= sweep <= 5 and row["seconds"]:
            values.append(float(row["seconds"]))
    return values


def read_update_counts(path: Path) -> dict[str, int]:
    stage2 = 0
    updates = 0
    residuals: list[float] = []
    if not path.is_file():
        return {"stage2": 0, "updates": 0, "residuals": residuals}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stage2 += int(float(row["n_updates_stage2"] or 0))
            updates += int(float(row["n_updates"] or 0))
            if row["stage3_residuals"]:
                residuals.extend(float(item) for item in row["stage3_residuals"].split(";") if item)
    return {"stage2": stage2, "updates": updates, "residuals": residuals}


def load_status_for_tests(path: Path) -> list[dict[str, str]]:
    return load_status_rows(path)
