"""Notebook 06 の保存済み成果物を CPU だけで照合し、条件を満たすときだけ修復する。

学習・評価・latency 測定・CUDA API は呼ばない。Notebook のセルは実行しない。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "40_dmrg" / "10_fashion_mnist_mlp"
RESULTS = ROOT / "results" / "40_dmrg" / "10_fashion_mnist_mlp"
NB06 = ROOT / "notebooks" / "40_dmrg" / "10_fashion_mnist_mlp" / "06_dense_tt_dmrg_like_convergence_comparison.ipynb"
M06 = MODELS / "06_dense_tt_dmrg_like_convergence_comparison"
M05 = MODELS / "05_update_scope_ablation_gpu_resident" / "joint"
STAGE_C = MODELS / "01_tt_initialization_scale_diagnostics" / "initial_stage_c.pt"
SUMMARY = RESULTS / "06_dense_tt_dmrg_like_convergence_comparison" / "comparison_summary.csv"
LATENCY = RESULTS / "06_dense_tt_dmrg_like_convergence_comparison" / "latency.csv"
REPAIR_DIR = RESULTS / "06_dense_tt_dmrg_like_convergence_comparison" / "rank_initial_repair"
BACKUP_DIR = REPAIR_DIR / "backup"
CONTINUATION_NAME = "continuation_start_after_3200_updates.pt"

EXPECTED_RANKS = (1, 8, 12, 12, 12, 12, 1)
EXPECTED_RANK_TEXT = "1|8|12|12|12|12|1"
OUT_MODES = (4, 2, 2, 2, 4, 4)
IN_MODES = (2, 2, 2, 2, 7, 7)
EXPECTED_CORE_ELEMENTS = 5968
EXPECTED_TT_PARAMETERS = 140378
EXPECTED_DENSE_PARAMETERS = 535818
NOTEBOOK05_UPDATES = 3200
PROTECTED_UNCHANGED = (
    M06 / "dmrg_like" / "best_model.pt",
    M06 / "dmrg_like" / "final_model.pt",
    M06 / "dmrg_like" / "history.csv",
    M06 / "dmrg_like" / "local_updates.csv",
    M06 / "dmrg_like" / "run_state.json",
    M06 / "dmrg_like" / "adam_training_state.pt",
    M06 / "standard_tt" / "initial_model.pt",
    M06 / "standard_tt" / "best_model.pt",
    M06 / "standard_tt" / "final_model.pt",
    M06 / "standard_tt" / "history.csv",
    M06 / "standard_tt" / "run_state.json",
    M06 / "standard_tt" / "adam_training_state.pt",
    M06 / "dense" / "initial_model.pt",
    M06 / "dense" / "best_model.pt",
    M06 / "dense" / "final_model.pt",
    M06 / "dense" / "history.csv",
    M06 / "dense" / "run_state.json",
    M06 / "dense" / "adam_training_state.pt",
    SUMMARY,
    LATENCY,
    NB06,
    M05 / "initial_model.pt",
    M05 / "best_model.pt",
    M05 / "final_model.pt",
    M05 / "history.csv",
    M05 / "local_updates.csv",
    STAGE_C,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cpu(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError(f"state_dict を持つ checkpoint ではありません: {type(payload).__name__}")
    return payload["state_dict"]


def core_tensors(state):
    keys = sorted(
        (key for key in state if key.startswith("fc1.cores.")),
        key=lambda key: int(key.rsplit(".", 1)[1]),
    )
    cores = [state[key] for key in keys]
    if len(cores) != 6:
        raise ValueError(f"fc1 core 数が 6 ではありません: {keys}")
    for core in cores:
        if not torch.is_tensor(core) or core.device.type != "cpu":
            raise ValueError("core が CPU tensor ではありません")
    return keys, cores


def ranks_of(cores) -> tuple[int, ...]:
    return tuple(int(core.shape[0]) for core in cores) + (int(cores[-1].shape[-1]),)


def shapes_of(cores) -> list[tuple[int, ...]]:
    return [tuple(int(size) for size in core.shape) for core in cores]


def expected_shapes() -> list[tuple[int, ...]]:
    shapes = []
    for index, (out_mode, in_mode) in enumerate(zip(OUT_MODES, IN_MODES, strict=True)):
        shapes.append((EXPECTED_RANKS[index], out_mode, in_mode, EXPECTED_RANKS[index + 1]))
    return shapes


def tensor_differences(left, right) -> list[str]:
    left_state = state_dict_of(left)
    right_state = state_dict_of(right)
    differences = []
    if set(left_state) != set(right_state):
        differences.append("state_dict のキー集合が違います")
    for key in sorted(set(left_state) & set(right_state)):
        left_value = left_state[key]
        right_value = right_state[key]
        if not torch.is_tensor(left_value) or not torch.is_tensor(right_value):
            differences.append(f"{key}: tensor ではありません")
            continue
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            differences.append(f"{key}: shape または dtype が違います")
            continue
        if not torch.equal(left_value.cpu(), right_value.cpu()):
            differences.append(f"{key}: 値が違います")
    return differences


def metadata_view(payload) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if key != "state_dict" and not torch.is_tensor(value)
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV ヘッダがありません: {path}")
        return list(reader.fieldnames), list(reader)


def checkpoint_facts(path: Path) -> dict:
    payload = load_cpu(path)
    state = state_dict_of(payload)
    keys, cores = core_tensors(state)
    ranks = ranks_of(cores)
    elements = sum(int(core.numel()) for core in cores)
    parameters = sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))
    return {
        "path": str(path.relative_to(ROOT)),
        "metadata_keys": [key for key in payload if key != "state_dict"],
        "metadata": metadata_view(payload),
        "core_keys": keys,
        "shapes": shapes_of(cores),
        "ranks": ranks,
        "core_elements": elements,
        "parameters": parameters,
        "matches_expected_shapes": shapes_of(cores) == expected_shapes(),
        "matches_expected_counts": elements == EXPECTED_CORE_ELEMENTS and parameters == EXPECTED_TT_PARAMETERS,
    }


def contiguous_updates(rows: list[dict[str, str]]) -> dict:
    updates = [int(row["cumulative_updates"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    duplicates = sorted({value for value in updates if updates.count(value) > 1})
    return {
        "rows": len(rows),
        "first": updates[0] if updates else None,
        "last": updates[-1] if updates else None,
        "contiguous_from_one": updates == expected,
        "duplicate_count": len(duplicates),
        "duplicate_examples": duplicates[:10],
    }


def retained_rank_mismatches(rows: list[dict[str, str]], limit: int = 10) -> list[dict]:
    mismatches = []
    for row_number, row in enumerate(rows, start=1):
        pair = int(row["pair_index"])
        retained = int(row["retained_rank"])
        if pair < 0 or pair + 1 >= len(EXPECTED_RANKS):
            expected = None
        else:
            expected = EXPECTED_RANKS[pair + 1]
        if retained != expected or row["status"] != "ok":
            mismatches.append(
                {
                    "row": row_number,
                    "cumulative_updates": row["cumulative_updates"],
                    "pair_index": pair,
                    "retained_rank": retained,
                    "expected_bond_rank": expected,
                    "status": row["status"],
                }
            )
            if len(mismatches) >= limit:
                break
    return mismatches


def history_rank_values(rows: list[dict[str, str]]) -> dict:
    ranks = sorted({row["actual_ranks"] for row in rows})
    flags = sorted({row["fixed_actual_rank"] for row in rows})
    elements = sorted({row["tt_core_elements"] for row in rows})
    parameters = sorted({row["total_parameters"] for row in rows})
    return {
        "rows": len(rows),
        "actual_ranks": ranks,
        "fixed_actual_rank": flags,
        "tt_core_elements": elements,
        "total_parameters": parameters,
        "all_expected": ranks == [EXPECTED_RANK_TEXT] and flags == ["True"] and elements == ["5968"] and parameters == ["140378"],
    }


def prefix_matches(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]], fields: list[str]) -> dict:
    mismatches = []
    for index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True), start=1):
        for field in fields:
            if left[field] != right[field]:
                mismatches.append({"row": index, "field": field, "notebook06": left[field], "notebook05": right[field]})
                if len(mismatches) >= 5:
                    return {"matches": False, "examples": mismatches}
    return {"matches": not mismatches, "examples": mismatches}


def notebook_summary_mentions() -> dict:
    notebook = json.loads(NB06.read_text(encoding="utf-8"))
    html_parts = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            html = data.get("text/html")
            if isinstance(html, list):
                html_parts.append("".join(html))
            elif isinstance(html, str):
                html_parts.append(html)
    combined = "\n".join(html_parts)
    return {
        "embedded_html_contains_fixed_actual_rank": "fixed_actual_rank" in combined,
        "embedded_true_count_for_rank_label": combined.count(">fixed_actual_rank<"),
        "embedded_true_count": combined.count(">True<"),
    }


def build_audit() -> dict:
    dmrg_initial = load_cpu(M06 / "dmrg_like" / "initial_model.pt")
    dmrg_best = checkpoint_facts(M06 / "dmrg_like" / "best_model.pt")
    dmrg_final = checkpoint_facts(M06 / "dmrg_like" / "final_model.pt")
    dmrg_initial_facts = checkpoint_facts(M06 / "dmrg_like" / "initial_model.pt")
    standard_initial = checkpoint_facts(M06 / "standard_tt" / "initial_model.pt")
    standard_best = checkpoint_facts(M06 / "standard_tt" / "best_model.pt")
    standard_final = checkpoint_facts(M06 / "standard_tt" / "final_model.pt")
    stage_c = load_cpu(STAGE_C)
    initial_05 = load_cpu(M05 / "initial_model.pt")
    final_05 = load_cpu(M05 / "final_model.pt")

    log_fields, log_rows = read_csv(M06 / "dmrg_like" / "local_updates.csv")
    source_fields, source_rows = read_csv(M05 / "local_updates.csv")
    history_fields, history_rows = read_csv(M06 / "dmrg_like" / "history.csv")
    standard_fields, standard_rows = read_csv(M06 / "standard_tt" / "history.csv")
    run_state = json.loads((M06 / "dmrg_like" / "run_state.json").read_text(encoding="utf-8"))
    standard_state = json.loads((M06 / "standard_tt" / "run_state.json").read_text(encoding="utf-8"))
    dense_state = json.loads((M06 / "dense" / "run_state.json").read_text(encoding="utf-8"))
    dense_initial = load_cpu(M06 / "dense" / "initial_model.pt")

    log_continuity = contiguous_updates(log_rows)
    source_continuity = contiguous_updates(source_rows)
    rank_mismatches = retained_rank_mismatches(log_rows)
    history_ranks = history_rank_values(history_rows)
    standard_ranks = history_rank_values(standard_rows)
    prefix = {
        "fieldnames_equal": log_fields == source_fields,
        "source_rows": len(source_rows),
        "comparison": None,
    }
    if log_fields == source_fields and len(source_rows) == NOTEBOOK05_UPDATES and len(log_rows) >= NOTEBOOK05_UPDATES:
        prefix["comparison"] = prefix_matches(log_rows[:NOTEBOOK05_UPDATES], source_rows, log_fields)
    else:
        prefix["comparison"] = {"matches": False, "examples": [{"reason": "行数またはヘッダが前提と違います"}]}

    initial_vs_final = tensor_differences(dmrg_initial, final_05)
    initial05_vs_stage = tensor_differences(initial_05, stage_c)
    standard_vs_stage = tensor_differences(load_cpu(M06 / "standard_tt" / "initial_model.pt"), stage_c)
    metadata_same_keys = set(metadata_view(initial_05)) == set(metadata_view(dmrg_initial))
    stage_extra_keys = sorted(set(metadata_view(stage_c)) - set(metadata_view(initial_05)))

    dmrg_checkpoints_ok = all(
        facts["matches_expected_shapes"] and facts["matches_expected_counts"]
        for facts in (dmrg_initial_facts, dmrg_best, dmrg_final)
    )
    standard_checkpoints_ok = all(
        facts["matches_expected_shapes"] and facts["matches_expected_counts"]
        for facts in (standard_initial, standard_best, standard_final)
    )
    log_ok = (
        log_continuity["contiguous_from_one"]
        and log_continuity["duplicate_count"] == 0
        and log_continuity["rows"] == int(run_state["total_local_updates"])
        and log_continuity["rows"] == int(run_state["optimizer_steps"])
        and not rank_mismatches
        and prefix["comparison"]["matches"]
        and source_continuity["contiguous_from_one"]
        and source_continuity["rows"] == NOTEBOOK05_UPDATES
    )
    history_ok = (
        history_ranks["all_expected"]
        and history_ranks["rows"] == int(run_state["history_rows"])
        and int(history_rows[-1]["cumulative_local_updates"]) == int(run_state["total_local_updates"])
        and int(history_rows[0]["cumulative_local_updates"]) == 0
        and "actual_ranks" in history_fields
    )
    dmrg_path_confirmed = dmrg_checkpoints_ok and log_ok and history_ok
    dmrg_best_confirmed = dmrg_best["matches_expected_shapes"] and dmrg_best["matches_expected_counts"]
    standard_recorded_confirmed = standard_checkpoints_ok and standard_ranks["all_expected"]
    standard_best_confirmed = standard_best["matches_expected_shapes"] and standard_best["matches_expected_counts"]
    dense_parameters = sum(int(value.numel()) for value in state_dict_of(dense_initial).values())
    dense_has_cores = any(key.startswith("fc1.cores.") for key in state_dict_of(dense_initial))

    initial_identity_ok = (
        not initial_vs_final
        and not initial05_vs_stage
        and dmrg_initial_facts["metadata"]["layer_name"] == "fc1"
        and tuple(dmrg_initial_facts["metadata"]["out_modes"]) == OUT_MODES
        and tuple(dmrg_initial_facts["metadata"]["in_modes"]) == IN_MODES
        and tuple(metadata_view(initial_05)["out_modes"]) == OUT_MODES
        and tuple(metadata_view(initial_05)["in_modes"]) == IN_MODES
    )

    return {
        "dmrg_like": {
            "initial": dmrg_initial_facts,
            "best": dmrg_best,
            "final": dmrg_final,
            "log_continuity": log_continuity,
            "source_log_continuity": source_continuity,
            "retained_rank_mismatch_examples": rank_mismatches,
            "prefix_3200": prefix,
            "history": history_ranks,
            "run_state_fixed_actual_rank_not_used_as_evidence": run_state.get("fixed_actual_rank"),
            "run_state_actual_ranks_not_used_as_evidence": run_state.get("actual_ranks"),
            "path_fixed_rank": "PASS" if dmrg_path_confirmed else "FAIL",
            "best_checkpoint_fixed_rank": "PASS" if dmrg_best_confirmed else "FAIL",
            "fixed_rank_and": "PASS" if dmrg_path_confirmed and dmrg_best_confirmed else "FAIL",
        },
        "standard_tt": {
            "initial": standard_initial,
            "best": standard_best,
            "final": standard_final,
            "history": standard_ranks,
            "history_fields_include_actual_ranks": "actual_ranks" in standard_fields,
            "local_update_log": "未確認",
            "local_update_log_reason": "通常 Adam TT の step ごとの local update log は保存されていない。epoch 境界の history と initial/best/final だけを確認した。",
            "code_does_not_resize_cores": "Adam 更新は既存 core tensor の値を更新し、core の shape を組み替える処理をこの保存結果からは実行していない。全 step の shape は未確認。",
            "initial_matches_stage_c_tensors": not standard_vs_stage,
            "recorded_checkpoints_and_history": "PASS" if standard_recorded_confirmed else "FAIL",
            "best_checkpoint_fixed_rank": "PASS" if standard_best_confirmed else "FAIL",
            "full_step_path": "未確認",
            "stored_fixed_actual_rank_not_used_as_evidence": standard_state.get("fixed_actual_rank"),
        },
        "dense": {
            "tt_rank_path": "未確認",
            "reason": "Dense には TT core がない。固定 rank の学習経路は対象外。",
            "has_tt_cores": dense_has_cores,
            "initial_parameters": dense_parameters,
            "parameters_match_expected": dense_parameters == EXPECTED_DENSE_PARAMETERS,
            "stored_fixed_actual_rank_not_used_as_evidence": dense_state.get("fixed_actual_rank"),
            "stored_actual_ranks": dense_state.get("actual_ranks"),
        },
        "initial_checkpoint": {
            "current_06_initial_equals_05_final_tensors": not initial_vs_final,
            "tensor_differences_06_initial_vs_05_final": initial_vs_final,
            "05_initial_equals_stage_c_tensors": not initial05_vs_stage,
            "tensor_differences_05_initial_vs_stage_c": initial05_vs_stage,
            "06_and_05_initial_metadata_key_sets_equal": metadata_same_keys,
            "stage_c_metadata_keys_absent_from_05_initial": stage_extra_keys,
            "identity_ok": initial_identity_ok,
        },
        "notebook_embedded_summary": notebook_summary_mentions(),
    }


def write_audit_table(audit: dict) -> None:
    fields, rows = read_csv(SUMMARY)
    extra = [
        "path_rank_check",
        "best_checkpoint_rank_check",
        "full_step_log_check",
        "fixed_rank_and",
        "audit_note",
    ]
    notes = {
        "dense": (
            "未確認",
            "未確認",
            "未確認",
            "未確認",
            "Dense には TT rank 経路がない。保存済みの性能値と固定 rank 列は変更していない。",
        ),
        "standard_tt": (
            audit["standard_tt"]["recorded_checkpoints_and_history"],
            audit["standard_tt"]["best_checkpoint_fixed_rank"],
            "未確認",
            "未確認",
            "記録済みの initial/best/final と history は期待 rank。step ごとの log は存在しないため、全 step の固定 rank は確認済みとしない。保存値は変更していない。",
        ),
        "dmrg_like": (
            audit["dmrg_like"]["path_fixed_rank"],
            audit["dmrg_like"]["best_checkpoint_fixed_rank"],
            audit["dmrg_like"]["path_fixed_rank"],
            audit["dmrg_like"]["fixed_rank_and"],
            "学習経路の local update log と best checkpoint の両方で actual ranks (1, 8, 12, 12, 12, 12, 1) を確認した。主表の数値変更は不要。",
        ),
    }
    destination = REPAIR_DIR / "comparison_summary_rank_audit.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields) + extra)
        writer.writeheader()
        for row in rows:
            condition = row["condition"]
            path_check, best_check, step_check, combined, note = notes[condition]
            copied = dict(row)
            copied.update(
                {
                    "path_rank_check": path_check,
                    "best_checkpoint_rank_check": best_check,
                    "full_step_log_check": step_check,
                    "fixed_rank_and": combined,
                    "audit_note": note,
                }
            )
            writer.writerow(copied)


def repair_initial(before_hashes: dict[str, str]) -> dict:
    source = M05 / "initial_model.pt"
    target = M06 / "dmrg_like" / "initial_model.pt"
    continuation = M06 / "dmrg_like" / CONTINUATION_NAME
    if continuation.exists():
        raise FileExistsError(f"退避先が既にあります: {continuation}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=False)
    backup = BACKUP_DIR / "dmrg_like_initial_model.pt"
    shutil.copy2(target, backup)
    if sha256(backup) != before_hashes[str(target.relative_to(ROOT))]:
        raise RuntimeError("バックアップの SHA-256 が変更前と一致しません")
    shutil.copy2(target, continuation)
    temporary = target.with_name(target.name + ".repair.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    repaired = sha256(target)
    if repaired != sha256(source):
        raise RuntimeError("修復後の initial_model.pt が 05 joint initial と一致しません")
    if sha256(continuation) != before_hashes[str(target.relative_to(ROOT))]:
        raise RuntimeError("継続開始 checkpoint が変更前の 06 initial と一致しません")
    return {
        "backup": str(backup.relative_to(ROOT)),
        "continuation_start": str(continuation.relative_to(ROOT)),
        "repaired_initial": str(target.relative_to(ROOT)),
        "repaired_initial_sha256": repaired,
        "continuation_sha256": sha256(continuation),
        "reason": "06 の initial_model.pt は 3,200 更新後の 05 final だった。履歴の 0 更新に対応する 05 joint initial へ戻し、元の中身は継続開始 checkpoint として残した。",
    }


def main() -> None:
    existing = [path for path in PROTECTED_UNCHANGED if path.exists()]
    missing_optional = [str(path.relative_to(ROOT)) for path in PROTECTED_UNCHANGED if not path.exists()]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in existing}
    before[str((M06 / "dmrg_like" / "initial_model.pt").relative_to(ROOT))] = sha256(M06 / "dmrg_like" / "initial_model.pt")
    audit = build_audit()
    dmrg_ok = audit["dmrg_like"]["fixed_rank_and"] == "PASS"
    initial_ok = audit["initial_checkpoint"]["identity_ok"]
    standard_recorded_ok = audit["standard_tt"]["recorded_checkpoints_and_history"] == "PASS"
    if not (dmrg_ok and initial_ok and standard_recorded_ok):
        REPAIR_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "action": "stopped_without_model_changes",
            "audit": audit,
            "sha256_before": before,
            "missing_paths": missing_optional,
        }
        (REPAIR_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("照合が揃わなかったため、checkpoint は変更していません。")

    repair = repair_initial(before)
    write_audit_table(audit)
    after_targets = {
        str((M06 / "dmrg_like" / "initial_model.pt").relative_to(ROOT)): sha256(M06 / "dmrg_like" / "initial_model.pt"),
        str((M06 / "dmrg_like" / CONTINUATION_NAME).relative_to(ROOT)): sha256(M06 / "dmrg_like" / CONTINUATION_NAME),
    }
    unchanged = {
        relative: sha256(ROOT / relative) == digest
        for relative, digest in before.items()
        if relative != str((M06 / "dmrg_like" / "initial_model.pt").relative_to(ROOT))
    }
    report = {
        "action": "repaired_initial_checkpoint_only",
        "cuda_api_called": False,
        "training_or_evaluation_executed": False,
        "notebook_executed": False,
        "audit": audit,
        "repair": repair,
        "sha256_before": before,
        "sha256_after_changed": after_targets,
        "unchanged_hashes_match": unchanged,
        "missing_paths": missing_optional,
        "comparison_summary_modified": False,
        "run_state_modified": False,
        "notebook_modified": False,
        "reason_table_unchanged": "DMRG-like の固定 rank 判定は学習経路と best checkpoint の両方で True だった。主表の数値と固定 rank 列は保存値のまま。",
    }
    (REPAIR_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPAIR_DIR / "sha256_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repair": repair, "unchanged": all(unchanged.values()), "dmrg_and": audit["dmrg_like"]["fixed_rank_and"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
