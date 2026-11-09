# 本実行と動作確認の共通手順。RUN_FULL_EXPERIMENT が False でも、このセルは関数を定義するだけ。

import csv

from nn_compression.compression.metric_low_rank import gated_error_parts_per_document
from nn_compression.metrics.pilot_decision import RhoEvaluation, evaluate_rho
from nn_compression.tt_mps.metric_tt import DiagonalMetricTT

SUMMARY_COLUMNS = [
    "row_id",
    "calibration_set",
    "tensorization",
    "budget",
    "method",
    "initial",
    "adopted",
    "n_parameters",
    "macs",
    "q",
    "q_lambda",
    "e_calibration",
    "E_rel_calibration",
    "E_gated_calibration",
    "e_validation_same",
    "E_rel_validation_same",
    "E_gated_validation_same",
    "e_validation_other",
    "E_rel_validation_other",
    "E_gated_validation_other",
    "perplexity",
    "mean_nll",
    "n_tokens",
    "seconds",
    "fit_source",
]
HISTORY_COLUMNS = [
    "calibration_set",
    "tensorization",
    "budget",
    "initial",
    "sweep",
    "q_lambda",
    "relative_decrease",
    "cg_iterations",
    "cg_residuals",
    "n_rejected",
    "seconds",
    "stop_reason",
]
INITIAL_COLUMNS = [
    "calibration_set",
    "tensorization",
    "budget",
    "initial",
    "q_lambda_before_als",
    "q_lambda_m4",
    "q_lambda_minus_m4",
]
BOOTSTRAP_COLUMNS = [
    "calibration_set",
    "tensorization",
    "budget",
    "domain",
    "statistic",
    "contrast",
    "estimate",
    "low",
    "high",
    "n_resamples",
]
RHO_COLUMNS = [
    "calibration_set",
    "tensorization",
    "budget",
    "domain",
    "e_m2",
    "e_m3",
    "e_m5",
    "g",
    "g_ci_low",
    "g_ci_high",
    "rho",
    "rho_status",
    "delta_tt",
    "r_m5",
    "kappa",
    "rho_gated",
    "rho_gated_status",
]
PERPLEXITY_COLUMNS = [
    "row_id",
    "calibration_set",
    "tensorization",
    "budget",
    "method",
    "initial",
    "perplexity",
    "mean_nll",
    "n_tokens",
    "n_windows",
    "seconds",
]
EVALUATION_DOMAINS = ("calibration", "validation_same", "validation_other")
BOOTSTRAP_PAIRS = (("M3", "M2"), ("M3", "M5"), ("M5", "M2"))
SPLIT_CACHE = {}
PREPARED_CORES = {}
M4_REFIT = {}
PILOT_MODEL = None
PILOT_UP = None


def empty_state():
    return {
        "summary": {},
        "history": {},
        "initial": {},
        "perplexity": {},
        "completed_m5": [],
        "completed_metrics": [],
        "completed_perplexity": [],
    }


def state_paths(suffix):
    return {
        "json": RESULTS_DIR / f"pilot_state{suffix}.json",
        "parts": RESULTS_DIR / f"error_parts{suffix}.pt",
        "summary": RESULTS_DIR / f"summary{suffix}.csv",
        "history": RESULTS_DIR / f"m5_history{suffix}.csv",
        "initial": RESULTS_DIR / f"m5_initial_values{suffix}.csv",
        "bootstrap": RESULTS_DIR / f"bootstrap{suffix}.csv",
        "rho": RESULTS_DIR / f"rho{suffix}.csv",
        "perplexity": RESULTS_DIR / f"perplexity{suffix}.csv",
    }


def load_run(suffix):
    paths = state_paths(suffix)
    if paths["json"].is_file():
        state = json.loads(paths["json"].read_text(encoding="utf-8"))
        for key, value in empty_state().items():
            state.setdefault(key, value if not isinstance(value, dict) else {})
    else:
        state = empty_state()
    parts = {}
    if paths["parts"].is_file():
        parts = torch.load(paths["parts"], weights_only=False)
    return state, parts


def row_id(calibration_set, tensorization, budget, method, initial):
    return "|".join([calibration_set, tensorization, budget, method, initial])


def m5_key(calibration_set, tensorization, budget, initial):
    return "|".join([calibration_set, tensorization, budget, initial])


def close_enough(left, right):
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1e-5 * scale


def load_split(name):
    if name not in SPLIT_CACHE:
        activations = torch.load(EXPERIMENT_DATA / f"{name}_activations.pt", weights_only=False)
        grams = torch.load(EXPERIMENT_DATA / f"{name}_grams.pt", weights_only=False)
        SPLIT_CACHE[name] = {
            "activation": activations["activation"],
            "per_document": grams["per_document"],
            "total": grams["total"],
            "n_tokens": int(grams["n_tokens"]),
        }
    return SPLIT_CACHE[name]


def calibration_bundle(calibration_set):
    saved = load_split("calibration")
    if calibration_set == "full":
        gram = saved["total"]
        return saved["activation"], saved["per_document"], gram, regularize_gram(gram, LAMBDA), saved["n_tokens"]
    if calibration_set != "sensitivity_16":
        raise ValueError(f"calibration_set が不明です: {calibration_set}")
    activation = saved["activation"][:SENSITIVITY_LIMIT]
    per_document = saved["per_document"][:SENSITIVITY_LIMIT]
    gram = (per_document.sum(dim=0) + per_document.sum(dim=0).transpose(0, 1)) * 0.5
    n_tokens = int(activation.shape[0] * activation.shape[1])
    return activation, per_document, gram, regularize_gram(gram, LAMBDA), n_tokens


def m3_core_list(label, budget):
    key = (label, budget)
    if key in M3_CORES:
        return M3_CORES[key]
    payload = torch.load(MODELS_DIR / f"m3_{label}_{budget}.pt", weights_only=False)
    return payload["cores"]


def m4_fit_for(label, budget, gram, n_tokens, out_modes, in_modes, rank_cap, calibration_set):
    if calibration_set == "full" and (label, budget) in M4_FITS:
        return M4_FITS[(label, budget)], "section_4_4"
    if calibration_set == "full":
        payload = torch.load(MODELS_DIR / f"m4_{label}_{budget}.pt", weights_only=False)
        fitted = DiagonalMetricTT(
            scaled_cores=payload["scaled_cores"],
            input_scale_inv=payload["input_scale_inv"],
            absorbed_cores=payload["absorbed_cores"],
            scales=payload["scales"],
            max_rank=payload["max_rank"],
        )
        return fitted, "section_4_4"
    cache_key = (calibration_set, label, budget)
    if cache_key not in M4_REFIT:
        M4_REFIT[cache_key] = fit_m4(W, gram, n_tokens, out_modes, in_modes, rank_cap)
    return M4_REFIT[cache_key], "this_cell"


def prepared_initial(calibration_set, label, budget, initial, gram, n_tokens):
    cache_key = (calibration_set, label, budget, initial)
    if cache_key in PREPARED_CORES:
        return PREPARED_CORES[cache_key]
    spec = TENSORIZATIONS[label]
    out_modes = tuple(spec["out_modes"])
    in_modes = tuple(spec["in_modes"])
    rank_cap = spec["budgets"][budget]["M3"]["rank_cap"]
    target_ranks = tuple(tt_matrix_ranks(m3_core_list(label, budget)))
    if initial == "M3":
        cores = m3_core_list(label, budget)
        source = "section_4_4"
    elif initial == "M4_retruncated":
        fitted, source = m4_fit_for(
            label, budget, gram, n_tokens, out_modes, in_modes, rank_cap, calibration_set
        )
        cores = prepare_m5_from_diagonal_tt(fitted, out_modes, in_modes, target_ranks)
        source = source if calibration_set == "full" else "this_cell"
    else:
        raise ValueError(f"初期値の名前が不明です: {initial}")
    prepared = canonicalize_tt_matrix_cores(cores, out_modes, in_modes, 0)
    PREPARED_CORES[cache_key] = (prepared, out_modes, in_modes, source)
    return PREPARED_CORES[cache_key]


def core_path(suffix, calibration_set, label, budget, initial):
    return MODELS_DIR / f"m5_{calibration_set}_{label}_{budget}_{initial}{suffix}.pt"


def blank_summary(identifier, calibration_set, tensorization, budget, method, initial):
    row = {name: "" for name in SUMMARY_COLUMNS}
    row.update(
        {
            "row_id": identifier,
            "calibration_set": calibration_set,
            "tensorization": tensorization,
            "budget": budget,
            "method": method,
            "initial": initial,
            "adopted": "",
        }
    )
    return row


def attach_metrics(state, parts, identifier, approximated, calibration_set, gram_lambda):
    if identifier in state["completed_metrics"]:
        return
    activation, per_document, gram, _gram_lambda, _n_tokens = calibration_bundle(calibration_set)
    numerator, denominator = gram_ratio_parts(W, approximated, per_document)
    gated_numerator, gated_denominator = gated_error_parts_per_document(
        W, approximated, W_GATE, list(activation.unbind(0))
    )
    bundle = metric_bundle(
        W,
        approximated,
        gram,
        W_GATE,
        activation.reshape(-1, IN_FEATURES),
        "gram",
    )
    ratio = float((numerator.sum() / denominator.sum()).item())
    gated_ratio = float((gated_numerator.sum() / gated_denominator.sum()).item())
    if not close_enough(ratio, bundle["e"]) or not close_enough(gated_ratio, bundle["E_gated"]):
        raise RuntimeError(f"{identifier} の較正誤差が、文書合計と metric_bundle で一致しません。")
    stored = {
        "calibration": {
            "num": numerator,
            "den": denominator,
            "gnum": gated_numerator,
            "gden": gated_denominator,
        }
    }
    scalars = {
        "q": float(squared_output_error_from_gram(W - approximated, gram).item()),
        "q_lambda": calibration_q_lambda(W, approximated, gram_lambda),
        "e_calibration": ratio,
        "E_rel_calibration": math.sqrt(ratio),
        "E_gated_calibration": gated_ratio,
    }
    for split in ("validation_same", "validation_other"):
        saved = load_split(split)
        split_activation = saved["activation"]
        numerator, denominator = activation_ratio_parts(W, approximated, list(split_activation.unbind(0)))
        gated_numerator, gated_denominator = gated_error_parts_per_document(
            W, approximated, W_GATE, list(split_activation.unbind(0))
        )
        flat = split_activation.reshape(-1, IN_FEATURES)
        bundle = metric_bundle(W, approximated, flat, W_GATE, flat, "activations")
        ratio = float((numerator.sum() / denominator.sum()).item())
        gated_ratio = float((gated_numerator.sum() / gated_denominator.sum()).item())
        if not close_enough(ratio, bundle["e"]) or not close_enough(gated_ratio, bundle["E_gated"]):
            raise RuntimeError(f"{identifier} の {split} 誤差が、文書合計と metric_bundle で一致しません。")
        stored[split] = {
            "num": numerator,
            "den": denominator,
            "gnum": gated_numerator,
            "gden": gated_denominator,
        }
        scalars[f"e_{split}"] = ratio
        scalars[f"E_rel_{split}"] = math.sqrt(ratio)
        scalars[f"E_gated_{split}"] = gated_ratio
    parts[identifier] = stored
    state["summary"][identifier].update(scalars)
    state["completed_metrics"].append(identifier)


def mark_adopted(state):
    groups = {}
    for row in state["summary"].values():
        if row["method"] != "M5":
            row["adopted"] = ""
            continue
        key = (row["calibration_set"], row["tensorization"], row["budget"])
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        best = min(rows, key=lambda item: (float(item["q_lambda"]), 0 if item["initial"] == "M3" else 1))
        for item in rows:
            item["adopted"] = "yes" if item["initial"] == best["initial"] else "no"


def adopted_initial(state, tensorization, budget):
    found = [
        row["initial"]
        for row in state["summary"].values()
        if row["calibration_set"] == "full"
        and row["tensorization"] == tensorization
        and row["budget"] == budget
        and row["method"] == "M5"
        and row["adopted"] == "yes"
    ]
    if len(found) != 1:
        raise RuntimeError(f"主実験の採用初期値が1つに定まっていません: {tensorization} {budget} {found}")
    return found[0]


def matrix_weight(method, budget, gram_lambda):
    rank = EXPECTED_MATRIX_RANKS[budget]
    if method == "M1":
        return truncated_svd_weight(W, rank)
    if method == "M2":
        return whitened_truncated_svd(W, gram_lambda, rank).weight
    raise ValueError(method)


def tt_dense(cores, out_modes, in_modes):
    return tt_matrix_to_dense(cores, out_modes, in_modes)


def parameter_record(method, budget, cores, out_modes, in_modes, extra):
    if method in {"M1", "M2"}:
        rank = EXPECTED_MATRIX_RANKS[budget]
        return (
            matrix_factor_parameter_count(OUT_FEATURES, IN_FEATURES, rank),
            compressed_linear_macs(IN_FEATURES, OUT_FEATURES, rank),
        )
    ranks = tuple(tt_matrix_ranks(cores))
    parameters = tt_parameter_count(cores) + extra
    macs = int(sum(tt_linear_direct_contraction_macs(out_modes, in_modes, ranks))) + extra
    return parameters, macs


def stop_reason(history, max_sweeps, sweep_tolerance):
    if not history:
        return "no_sweep"
    if history[-1].relative_decrease < sweep_tolerance:
        return "relative_decrease"
    if len(history) >= max_sweeps:
        return "max_sweeps"
    return "stopped"


def ensure_model():
    global PILOT_MODEL, PILOT_UP
    if PILOT_MODEL is None:
        model, _tokenizer = load_causal_lm_and_tokenizer(RECORD, dtype=torch.bfloat16, device="cpu")
        model = model.to(dtype=torch.float32, device="cuda")
        model.eval()
        module = model.model.layers[LAYER_INDEX].mlp.up_proj
        if not torch.equal(module.weight.detach().cpu(), W.to(dtype=torch.float32)):
            raise RuntimeError("再読込した up_proj が、bf16 から float64 を経由した重みと一致しません。")
        PILOT_MODEL = model
        PILOT_UP = module
    return PILOT_MODEL, PILOT_UP


def run_one_perplexity(state, suffix, identifier, meta, weight):
    if identifier in state["completed_perplexity"]:
        return
    model, module = ensure_model()
    before = module.weight.detach().clone()
    started = time.perf_counter()
    if weight is None:
        result = sliding_window_perplexity(
            model, PERPLEXITY_IDS, max_length=1024, stride=512, use_cache=False
        )
    else:
        result = layer_perplexity(model, module, weight, PERPLEXITY_IDS)
    elapsed = time.perf_counter() - started
    if not torch.equal(module.weight.detach(), before):
        raise RuntimeError("perplexity の後に up_proj の重みが戻っていません。")
    state["perplexity"][identifier] = {
        "row_id": identifier,
        "calibration_set": meta["calibration_set"],
        "tensorization": meta["tensorization"],
        "budget": meta["budget"],
        "method": meta["method"],
        "initial": meta["initial"],
        "perplexity": result["perplexity"],
        "mean_nll": result["mean_nll"],
        "n_tokens": result["n_tokens"],
        "n_windows": result["n_windows"],
        "seconds": elapsed,
    }
    state["completed_perplexity"].append(identifier)
    if identifier in state["summary"]:
        state["summary"][identifier]["perplexity"] = result["perplexity"]
        state["summary"][identifier]["mean_nll"] = result["mean_nll"]
        state["summary"][identifier]["n_tokens"] = result["n_tokens"]
    save_run(suffix, state, CURRENT_PARTS)
    print(meta["method"], meta["tensorization"], meta["budget"], result["perplexity"], f"{elapsed:.1f}s", flush=True)


def csv_value(value):
    if value is None:
        return ""
    return value


def write_tables(suffix, state, parts):
    paths = state_paths(suffix)
    summary_rows = [state["summary"][key] for key in sorted(state["summary"])]
    atomic_write_csv(paths["summary"], summary_rows, SUMMARY_COLUMNS)
    history_rows = [state["history"][key] for key in sorted(state["history"])]
    atomic_write_csv(paths["history"], history_rows, HISTORY_COLUMNS)
    initial_rows = [state["initial"][key] for key in sorted(state["initial"])]
    atomic_write_csv(paths["initial"], initial_rows, INITIAL_COLUMNS)
    perplexity_rows = [state["perplexity"][key] for key in sorted(state["perplexity"])]
    atomic_write_csv(paths["perplexity"], perplexity_rows, PERPLEXITY_COLUMNS)
    bootstrap_rows = []
    rho_rows = []
    groups = {
        (row["calibration_set"], row["tensorization"], row["budget"])
        for row in state["summary"].values()
        if row["method"] == "M5" and row["adopted"] == "yes"
    }
    for calibration_set, tensorization, budget in sorted(groups):
        adopted = [
            row["initial"]
            for row in state["summary"].values()
            if row["calibration_set"] == calibration_set
            and row["tensorization"] == tensorization
            and row["budget"] == budget
            and row["method"] == "M5"
            and row["adopted"] == "yes"
        ][0]
        method_ids = {
            "M2": row_id(calibration_set, "matrix", budget, "M2", ""),
            "M3": row_id(calibration_set, tensorization, budget, "M3", ""),
            "M5": row_id(calibration_set, tensorization, budget, "M5", adopted),
        }
        if any(identifier not in parts for identifier in method_ids.values()):
            continue
        for domain in EVALUATION_DOMAINS:
            field = "e_calibration" if domain == "calibration" else f"e_{domain}"
            gated_field = "E_gated_calibration" if domain == "calibration" else f"E_gated_{domain}"
            for statistic, num_name, den_name, value_field in (
                ("e", "num", "den", field),
                ("E_gated", "gnum", "gden", gated_field),
            ):
                bundle = {
                    name: (parts[identifier][domain][num_name], parts[identifier][domain][den_name])
                    for name, identifier in method_ids.items()
                }
                intervals = difference_intervals(bundle, BOOTSTRAP_PAIRS)
                for left, right in BOOTSTRAP_PAIRS:
                    estimate = intervals[(left, right)]
                    bootstrap_rows.append(
                        {
                            "calibration_set": calibration_set,
                            "tensorization": tensorization,
                            "budget": budget,
                            "domain": domain,
                            "statistic": statistic,
                            "contrast": f"{left}-{right}",
                            "estimate": estimate["estimate"],
                            "low": estimate["low"],
                            "high": estimate["high"],
                            "n_resamples": estimate["n_resamples"],
                        }
                    )
                if statistic == "e":
                    point_interval = intervals[("M3", "M2")]
                    evaluation = evaluate_rho(
                        budget=budget,
                        tensorization=tensorization,
                        domain=domain,
                        e_m2=float(state["summary"][method_ids["M2"]][value_field]),
                        e_m3=float(state["summary"][method_ids["M3"]][value_field]),
                        e_m5=float(state["summary"][method_ids["M5"]][value_field]),
                        g_ci_low=point_interval["low"],
                        g_ci_high=point_interval["high"],
                        min_relative_g=MIN_RELATIVE_G,
                    )
                    gated_interval = None
                else:
                    gated_interval = intervals[("M3", "M2")]
                    gated_evaluation = evaluate_rho(
                        budget=budget,
                        tensorization=tensorization,
                        domain=domain,
                        e_m2=float(state["summary"][method_ids["M2"]][value_field]),
                        e_m3=float(state["summary"][method_ids["M3"]][value_field]),
                        e_m5=float(state["summary"][method_ids["M5"]][value_field]),
                        g_ci_low=gated_interval["low"],
                        g_ci_high=gated_interval["high"],
                        min_relative_g=MIN_RELATIVE_G,
                    )
                    rho_rows.append(
                        {
                            "calibration_set": calibration_set,
                            "tensorization": tensorization,
                            "budget": budget,
                            "domain": domain,
                            "e_m2": evaluation.e_m2,
                            "e_m3": evaluation.e_m3,
                            "e_m5": evaluation.e_m5,
                            "g": evaluation.g,
                            "g_ci_low": evaluation.g_ci_low,
                            "g_ci_high": evaluation.g_ci_high,
                            "rho": csv_value(evaluation.rho),
                            "rho_status": evaluation.rho_status,
                            "delta_tt": evaluation.delta_tt,
                            "r_m5": evaluation.r_m5,
                            "kappa": csv_value(evaluation.kappa),
                            "rho_gated": csv_value(gated_evaluation.rho),
                            "rho_gated_status": gated_evaluation.rho_status,
                        }
                    )
    atomic_write_csv(paths["bootstrap"], bootstrap_rows, BOOTSTRAP_COLUMNS)
    atomic_write_csv(paths["rho"], rho_rows, RHO_COLUMNS)


def save_run(suffix, state, parts):
    paths = state_paths(suffix)
    atomic_write_json(paths["json"], json_ready(state))
    atomic_torch_save(paths["parts"], parts)
    write_tables(suffix, state, parts)


def resolve_initials(plan, state, label, budget):
    requested = plan["m5_initials"]
    if requested == "adopted_from_full":
        return (adopted_initial(state, label, budget),)
    return tuple(requested)


def execute_plan(plan):
    global CURRENT_PARTS
    suffix = plan["file_suffix"]
    state, parts = load_run(suffix)
    CURRENT_PARTS = parts
    als_ran = []
    als_skipped = []
    for budget in plan["budgets"]:
        _activation, _documents, gram, gram_lambda, n_tokens = calibration_bundle(plan["calibration_set"])
        for method in ("M1", "M2"):
            if method not in plan["methods"]:
                continue
            identifier = row_id(plan["calibration_set"], "matrix", budget, method, "")
            if identifier not in state["completed_metrics"]:
                print("fit", identifier, flush=True)
                started = time.perf_counter()
                approximated = matrix_weight(method, budget, gram_lambda)
                parameters, macs = parameter_record(method, budget, None, None, None, 0)
                row = blank_summary(identifier, plan["calibration_set"], "matrix", budget, method, "")
                row["n_parameters"] = parameters
                row["macs"] = macs
                row["seconds"] = time.perf_counter() - started
                row["fit_source"] = "this_cell"
                state["summary"][identifier] = row
                attach_metrics(state, parts, identifier, approximated, plan["calibration_set"], gram_lambda)
                save_run(suffix, state, parts)
        for label in plan["tensorizations"]:
            spec = TENSORIZATIONS[label]
            out_modes = tuple(spec["out_modes"])
            in_modes = tuple(spec["in_modes"])
            rank_cap = spec["budgets"][budget]["M3"]["rank_cap"]
            fitted, m4_source = m4_fit_for(
                label,
                budget,
                gram,
                n_tokens,
                out_modes,
                in_modes,
                rank_cap,
                plan["calibration_set"],
            )
            missing_initials = [
                initial
                for initial in plan["record_initials"]
                if "|".join([plan["calibration_set"], label, budget, initial]) not in state["initial"]
            ]
            if missing_initials:
                q_m4 = calibration_q_lambda(W, dense_m4(fitted, out_modes, in_modes), gram_lambda)
            for initial in missing_initials:
                initial_id = "|".join([plan["calibration_set"], label, budget, initial])
                prepared, _out, _inn, _source = prepared_initial(
                    plan["calibration_set"], label, budget, initial, gram, n_tokens
                )
                q_before = float(q_lambda_of_cores(prepared, W, gram_lambda, out_modes, in_modes).item())
                state["initial"][initial_id] = {
                    "calibration_set": plan["calibration_set"],
                    "tensorization": label,
                    "budget": budget,
                    "initial": initial,
                    "q_lambda_before_als": q_before,
                    "q_lambda_m4": q_m4,
                    "q_lambda_minus_m4": q_before - q_m4,
                }
                save_run(suffix, state, parts)
            if "M3" in plan["methods"]:
                identifier = row_id(plan["calibration_set"], label, budget, "M3", "")
                if identifier not in state["completed_metrics"]:
                    print("fit", identifier, flush=True)
                    started = time.perf_counter()
                    cores = m3_core_list(label, budget)
                    approximated = tt_dense(cores, out_modes, in_modes)
                    parameters, macs = parameter_record("M3", budget, cores, out_modes, in_modes, 0)
                    row = blank_summary(identifier, plan["calibration_set"], label, budget, "M3", "")
                    row["n_parameters"] = parameters
                    row["macs"] = macs
                    row["seconds"] = time.perf_counter() - started
                    row["fit_source"] = "section_4_4"
                    state["summary"][identifier] = row
                    attach_metrics(state, parts, identifier, approximated, plan["calibration_set"], gram_lambda)
                    save_run(suffix, state, parts)
            if "M4" in plan["methods"]:
                identifier = row_id(plan["calibration_set"], label, budget, "M4", "")
                if identifier not in state["completed_metrics"]:
                    print("fit", identifier, flush=True)
                    started = time.perf_counter()
                    approximated = dense_m4(fitted, out_modes, in_modes)
                    parameters, macs = parameter_record(
                        "M4", budget, fitted.scaled_cores, out_modes, in_modes, IN_FEATURES
                    )
                    row = blank_summary(identifier, plan["calibration_set"], label, budget, "M4", "")
                    row["n_parameters"] = parameters
                    row["macs"] = macs
                    row["seconds"] = time.perf_counter() - started
                    row["fit_source"] = m4_source
                    state["summary"][identifier] = row
                    attach_metrics(state, parts, identifier, approximated, plan["calibration_set"], gram_lambda)
                    save_run(suffix, state, parts)
            for initial in resolve_initials(plan, state, label, budget):
                key = m5_key(plan["calibration_set"], label, budget, initial)
                identifier = row_id(plan["calibration_set"], label, budget, "M5", initial)
                path = core_path(suffix, plan["calibration_set"], label, budget, initial)
                final_cores = None
                if key in state["completed_m5"]:
                    als_skipped.append(key)
                    print("skip M5", key, flush=True)
                    if identifier not in state["completed_metrics"]:
                        payload = torch.load(path, weights_only=False)
                        final_cores = payload["cores"]
                else:
                    als_ran.append(key)
                    print("ALS", key, flush=True)
                    prepared, out_modes, in_modes, fit_source = prepared_initial(
                        plan["calibration_set"], label, budget, initial, gram, n_tokens
                    )
                    started = time.perf_counter()
                    if plan["use_experiment_stop"]:
                        final_cores, history = run_m5(prepared, W, gram_lambda, out_modes, in_modes)
                    else:
                        final_cores, history = run_metric_tt_als(
                            prepared,
                            W,
                            gram_lambda,
                            out_modes,
                            in_modes,
                            relative_tolerance=plan["relative_tolerance"],
                            max_cg_iterations=plan["max_cg_iterations"],
                            sweep_tolerance=plan["sweep_tolerance"],
                            max_sweeps=plan["max_sweeps"],
                        )
                    reason = stop_reason(history, plan["max_sweeps"], plan["sweep_tolerance"])
                    atomic_torch_save(
                        path,
                        {
                            "cores": final_cores,
                            "out_modes": list(out_modes),
                            "in_modes": list(in_modes),
                            "initial": initial,
                        },
                    )
                    for record in history:
                        history_id = f"{key}|{record.sweep:03d}"
                        state["history"][history_id] = {
                            "calibration_set": plan["calibration_set"],
                            "tensorization": label,
                            "budget": budget,
                            "initial": initial,
                            "sweep": record.sweep,
                            "q_lambda": record.q_lambda,
                            "relative_decrease": record.relative_decrease,
                            "cg_iterations": ";".join(str(item) for item in record.cg_iterations),
                            "cg_residuals": ";".join(f"{item:.6e}" for item in record.cg_residuals),
                            "n_rejected": record.n_rejected,
                            "seconds": record.seconds,
                            "stop_reason": reason,
                        }
                    parameters, macs = parameter_record("M5", budget, final_cores, out_modes, in_modes, 0)
                    row = blank_summary(
                        identifier, plan["calibration_set"], label, budget, "M5", initial
                    )
                    row["n_parameters"] = parameters
                    row["macs"] = macs
                    row["seconds"] = time.perf_counter() - started
                    row["fit_source"] = fit_source
                    state["summary"][identifier] = row
                    state["completed_m5"].append(key)
                    save_run(suffix, state, parts)
                if identifier not in state["completed_metrics"]:
                    if final_cores is None:
                        payload = torch.load(path, weights_only=False)
                        final_cores = payload["cores"]
                    approximated = tt_dense(final_cores, out_modes, in_modes)
                    attach_metrics(state, parts, identifier, approximated, plan["calibration_set"], gram_lambda)
                    matching = [
                        item
                        for item in state["history"].values()
                        if item["calibration_set"] == plan["calibration_set"]
                        and item["tensorization"] == label
                        and item["budget"] == budget
                        and item["initial"] == initial
                    ]
                    if matching:
                        last_q = max(matching, key=lambda item: int(item["sweep"]))["q_lambda"]
                        if not close_enough(state["summary"][identifier]["q_lambda"], last_q):
                            raise RuntimeError(f"{identifier} の Q_λ が掃引履歴と一致しません。")
                    save_run(suffix, state, parts)
    mark_adopted(state)
    if plan["compute_perplexity"]:
        run_one_perplexity(
            state,
            suffix,
            "full_model",
            {
                "calibration_set": "full",
                "tensorization": "",
                "budget": "",
                "method": "full_model",
                "initial": "",
            },
            None,
        )
        for row in list(state["summary"].values()):
            if row["calibration_set"] != plan["calibration_set"]:
                continue
            if row["tensorization"] not in plan["tensorizations"] and row["tensorization"] != "matrix":
                continue
            if row["budget"] not in plan["budgets"]:
                continue
            if row["method"] == "M5" and row["adopted"] != "yes":
                continue
            if row["row_id"] in state["completed_perplexity"]:
                continue
            weight = materialize_row(plan, state, row)
            run_one_perplexity(state, suffix, row["row_id"], row, weight)
    save_run(suffix, state, parts)
    return {"als_ran": als_ran, "als_skipped": als_skipped, "state": state}


def materialize_row(plan, state, row):
    _activation, _documents, gram, gram_lambda, n_tokens = calibration_bundle(row["calibration_set"])
    if row["method"] in {"M1", "M2"}:
        return matrix_weight(row["method"], row["budget"], gram_lambda)
    spec = TENSORIZATIONS[row["tensorization"]]
    out_modes = tuple(spec["out_modes"])
    in_modes = tuple(spec["in_modes"])
    if row["method"] == "M3":
        return tt_dense(m3_core_list(row["tensorization"], row["budget"]), out_modes, in_modes)
    if row["method"] == "M4":
        rank_cap = spec["budgets"][row["budget"]]["M3"]["rank_cap"]
        fitted, _source = m4_fit_for(
            row["tensorization"],
            row["budget"],
            gram,
            n_tokens,
            out_modes,
            in_modes,
            rank_cap,
            row["calibration_set"],
        )
        return dense_m4(fitted, out_modes, in_modes)
    if row["method"] == "M5":
        payload = torch.load(
            core_path(plan["file_suffix"], row["calibration_set"], row["tensorization"], row["budget"], row["initial"]),
            weights_only=False,
        )
        return tt_dense(payload["cores"], out_modes, in_modes)
    raise ValueError(row["method"])


CURRENT_PARTS = {}
