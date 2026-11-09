"""01_single_layer_pilot.ipynb に本実行用セルを足す。"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NOTEBOOK = Path(__file__).with_name("01_single_layer_pilot.ipynb")
EXPERIMENT = Path(__file__).with_name("_experiment_cell.py")
MARKER = "# 本実行と動作確認の共通手順"


def replace_once(source, old, new, label):
    if old not in source:
        raise SystemExit(f"置き換え対象がありません: {label}")
    return source.replace(old, new, 1)


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    methods = notebook.cells[8]
    methods.source = replace_once(
        methods.source,
        "def initial_m5_from_m4(weight, gram, n_tokens, out_modes, in_modes, rank_cap, target_ranks):\n",
        "def initial_m5_from_m4(weight, gram, n_tokens, out_modes, in_modes, rank_cap, target_ranks):\n"
        "    # M4 の密行列を TT-SVD で作り直したコアを、M3 の rank まで広げる。\n",
        "initial_m5",
    )
    methods.source = replace_once(
        methods.source,
        '    """候補は (名前, Q_λ, コア, 履歴)。較正の Q_λ が小さい方を採る。同点は M3。"""\n',
        '    """候補は (名前, Q_λ, コア, 履歴)。名前は M3 または M4_retruncated。\n'
        '    較正の Q_λ が小さい方を採る。同点は M3。\n'
        '    """\n',
        "adopt",
    )
    evaluation = notebook.cells[9]
    evaluation.source = evaluation.source.replace(
        "- 入力: 重み、近似、Gram または活性化、ゲート重み、perplexity のトークン列\n",
        "- 入力: 重み、近似、Gram または活性化、`statistic_kind`、ゲート重み、perplexity のトークン列\n",
    )
    metrics = notebook.cells[10]
    metrics.source = replace_once(
        metrics.source,
        "def metric_bundle(weight, approximated, gram_or_activations, gate_weight, activations):\n"
        "    from nn_compression.compression.metric_low_rank import gated_error_parts, output_error_ratios\n"
        "\n"
        "    ratios = output_error_ratios(weight, approximated, gram_or_activations)\n",
        "def metric_bundle(weight, approximated, statistic, gate_weight, activations, statistic_kind):\n"
        "    from nn_compression.compression.metric_low_rank import (\n"
        "        gated_error_parts,\n"
        "        output_error_ratios_from_activations,\n"
        "        output_error_ratios_from_gram,\n"
        "    )\n"
        "\n"
        "    if statistic_kind == \"gram\":\n"
        "        ratios = output_error_ratios_from_gram(weight, approximated, statistic)\n"
        "    elif statistic_kind == \"activations\":\n"
        "        ratios = output_error_ratios_from_activations(weight, approximated, statistic)\n"
        "    else:\n"
        "        raise ValueError(f\"statistic_kind は gram または activations です: {statistic_kind}\")\n",
        "metric_bundle",
    )
    kept = []
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith(MARKER):
            break
        if cell.cell_type == "markdown" and cell.source.startswith("## 5. 本実行の手順"):
            break
        kept.append(cell)
    notebook.cells = kept
    experiment = EXPERIMENT.read_text(encoding="utf-8").strip("\n")
    compile(experiment, "<experiment>", "exec")
    notebook.cells.extend(
        [
            new_markdown_cell(
                "\n".join(
                    [
                        "## 5. 本実行の手順",
                        "",
                        "- 入力: 4.4 のコア、4.5 の関数、保存済みの活性化と Gram",
                        "- このステップで行うこと: 主実験、感度確認、判定、動作確認が共有する手順を定義する。",
                        "- 行わないこと: M5 を走らせない。ファイルを書かない。",
                        "- 出力: 関数 `execute_plan`",
                        "- 完了条件: 関数が定義され、`RUN_FULL_EXPERIMENT` は False のまま。",
                        "- 次への接続: 5.1 以降が `execute_plan` に条件を渡す。",
                    ]
                )
            ),
            new_code_cell(experiment),
            new_markdown_cell(
                "\n".join(
                    [
                        "## 5.1 主実験",
                        "",
                        "- 入力: `execute_plan`、`RUN_FULL_EXPERIMENT`",
                        "- このステップで行うこと: フラグが真のとき、T1・T2、予算 50%・25%、M1〜M4 と M5 の初期値 M3・M4_retruncated を計算する。",
                        "- 行わないこと: フラグが偽のときは何も計算しない。λ と停止条件は変えない。",
                        "- 出力: `summary.csv`、`m5_history.csv`、`m5_initial_values.csv`、`bootstrap.csv`、`rho.csv`、`perplexity.csv`。コアは `models/`。",
                        "- 完了条件: フラグが偽なら、新しい結果ファイルを作らずに終わる。",
                        "- 次への接続: 5.2 は、ここで採用された初期値を読む。",
                    ]
                )
            ),
            new_code_cell(
                "\n".join(
                    [
                        "if not RUN_FULL_EXPERIMENT:",
                        '    print("RUN_FULL_EXPERIMENT が False のため、主実験はしない。")',
                        "else:",
                        "    execute_plan(",
                        "        {",
                        '            "calibration_set": "full",',
                        '            "tensorizations": ("T1", "T2"),',
                        '            "budgets": ("50", "25"),',
                        '            "methods": ("M1", "M2", "M3", "M4", "M5"),',
                        '            "m5_initials": ("M3", "M4_retruncated"),',
                        '            "record_initials": ("M3", "M4_retruncated"),',
                        '            "max_sweeps": SWEEP_MAX,',
                        '            "max_cg_iterations": CG_MAX_ITERATIONS,',
                        '            "relative_tolerance": CG_RELATIVE_TOLERANCE,',
                        '            "sweep_tolerance": SWEEP_RELATIVE_DECREASE,',
                        '            "file_suffix": "",',
                        '            "compute_perplexity": True,',
                        '            "use_experiment_stop": True,',
                        "        }",
                        "    )",
                    ]
                )
            ),
            new_markdown_cell(
                "\n".join(
                    [
                        "## 5.2 較正量の感度確認",
                        "",
                        "- 入力: 主実験で採用された初期値、較正系列の先頭 16",
                        "- このステップで行うこと: フラグが真のとき、T2 の予算 50%・25% で G を 16 系列から作り直し、M2 と、採用された初期値の M5 を計算する。ρ の分母に使う M3 は、主実験と同じ TT-SVD をこの 16 系列で測り直す。",
                        "- 行わないこと: フラグが偽のときは何もしない。判定には使わない。perplexity は測らない。",
                        "- 出力: 5.1 と同じ CSV に、`calibration_set=sensitivity_16` の行を足す。",
                        "- 完了条件: フラグが偽ならファイルを増やさない。",
                        "- 次への接続: 5.3 は `calibration_set=full` の行だけを判定に使う。",
                    ]
                )
            ),
            new_code_cell(
                "\n".join(
                    [
                        "if not RUN_FULL_EXPERIMENT:",
                        '    print("RUN_FULL_EXPERIMENT が False のため、感度確認はしない。")',
                        "else:",
                        "    execute_plan(",
                        "        {",
                        '            "calibration_set": "sensitivity_16",',
                        '            "tensorizations": ("T2",),',
                        '            "budgets": ("50", "25"),',
                        '            "methods": ("M2", "M3", "M5"),',
                        '            "m5_initials": "adopted_from_full",',
                        '            "record_initials": ("M3", "M4_retruncated"),',
                        '            "max_sweeps": SWEEP_MAX,',
                        '            "max_cg_iterations": CG_MAX_ITERATIONS,',
                        '            "relative_tolerance": CG_RELATIVE_TOLERANCE,',
                        '            "sweep_tolerance": SWEEP_RELATIVE_DECREASE,',
                        '            "file_suffix": "",',
                        '            "compute_perplexity": False,',
                        '            "use_experiment_stop": True,',
                        "        }",
                        "    )",
                    ]
                )
            ),
            new_markdown_cell(
                "\n".join(
                    [
                        "## 5.3 判定",
                        "",
                        "- 入力: `rho.csv` の `calibration_set=full` の行",
                        "- このステップで行うこと: フラグが真のとき、`pilot_decision` で判定し、使ったセルと一緒に `decision.json` へ書く。",
                        "- 行わないこと: フラグが偽のときは判定しない。感度確認の行は使わない。",
                        "- 出力: `decision.json`",
                        "- 完了条件: フラグが偽なら `decision.json` を作らない。",
                        "- 次への接続: 動作確認は別ファイル名で、この判定を呼ばない。",
                    ]
                )
            ),
            new_code_cell(
                "\n".join(
                    [
                        "if not RUN_FULL_EXPERIMENT:",
                        '    print("RUN_FULL_EXPERIMENT が False のため、判定は出さない。")',
                        "else:",
                        "    rho_path = RESULTS_DIR / \"rho.csv\"",
                        "    with rho_path.open(encoding=\"utf-8\", newline=\"\") as handle:",
                        "        rho_rows = list(csv.DictReader(handle))",
                        "    cells = []",
                        "    recorded = []",
                        "    for row in rho_rows:",
                        "        if row[\"calibration_set\"] != \"full\":",
                        "            continue",
                        "        rho = None if row[\"rho\"] == \"\" else float(row[\"rho\"])",
                        "        kappa = None if row[\"kappa\"] == \"\" else float(row[\"kappa\"])",
                        "        cells.append(",
                        "            RhoEvaluation(",
                        "                budget=row[\"budget\"],",
                        "                tensorization=row[\"tensorization\"],",
                        "                domain=row[\"domain\"],",
                        "                e_m2=float(row[\"e_m2\"]),",
                        "                e_m3=float(row[\"e_m3\"]),",
                        "                e_m5=float(row[\"e_m5\"]),",
                        "                g_ci_low=float(row[\"g_ci_low\"]),",
                        "                g_ci_high=float(row[\"g_ci_high\"]),",
                        "                g=float(row[\"g\"]),",
                        "                rho=rho,",
                        "                rho_status=row[\"rho_status\"],",
                        "                delta_tt=float(row[\"delta_tt\"]),",
                        "                r_m5=float(row[\"r_m5\"]),",
                        "                kappa=kappa,",
                        "            )",
                        "        )",
                        "        recorded.append(",
                        "            {",
                        "                \"budget\": row[\"budget\"],",
                        "                \"tensorization\": row[\"tensorization\"],",
                        "                \"domain\": row[\"domain\"],",
                        "                \"e_m2\": float(row[\"e_m2\"]),",
                        "                \"e_m3\": float(row[\"e_m3\"]),",
                        "                \"e_m5\": float(row[\"e_m5\"]),",
                        "                \"g\": float(row[\"g\"]),",
                        "                \"g_ci_low\": float(row[\"g_ci_low\"]),",
                        "                \"g_ci_high\": float(row[\"g_ci_high\"]),",
                        "                \"rho\": rho,",
                        "                \"rho_status\": row[\"rho_status\"],",
                        "                \"delta_tt\": float(row[\"delta_tt\"]),",
                        "                \"r_m5\": float(row[\"r_m5\"]),",
                        "                \"kappa\": kappa,",
                        "            }",
                        "        )",
                        "    outcome = pilot_decision(cells)",
                        "    atomic_write_json(",
                        "        RESULTS_DIR / \"decision.json\",",
                        "        json_ready({\"outcome\": outcome, \"cells\": recorded}),",
                        "    )",
                        "    print(outcome)",
                    ]
                )
            ),
            new_markdown_cell(
                "\n".join(
                    [
                        "## 5.4 動作確認",
                        "",
                        "- 入力: 5. の `execute_plan`",
                        "- このステップで行うこと: T2、予算 25%、M5 の初期値は M3 だけ、掃引 1 回、CG 10 反復で最後まで書く。終わった状態でもう一度実行し、その M5 を飛ばす。続けて 16 系列の感度確認を 1 本だけ走らせる。",
                        "- 行わないこと: 本実行の停止条件では走らせない。`decision.json` は出さない。ファイル名には `_smoke` を付ける。",
                        "- 出力: `summary_smoke.csv` ほか `_smoke` の付いた CSV と、`models/` の `*_smoke.pt`",
                        "- 完了条件: 列名が定義どおり。2 回目は M5 を 1 本も走らせない。重みの差し替えが元に戻る。判定ファイルが無い。",
                        "- 次への接続: 結果を見てから、5.1 のフラグを真にして本実行する。",
                    ]
                )
            ),
            new_code_cell(
                "\n".join(
                    [
                        "def remove_smoke_outputs():",
                        "    for path in RESULTS_DIR.glob(\"*_smoke.csv\"):",
                        "        path.unlink()",
                        "    for name in (\"pilot_state_smoke.json\", \"error_parts_smoke.pt\"):",
                        "        (RESULTS_DIR / name).unlink(missing_ok=True)",
                        "    for path in MODELS_DIR.glob(\"*_smoke.pt\"):",
                        "        path.unlink()",
                        "",
                        "",
                        "def assert_header(path, columns):",
                        "    header = path.read_text(encoding=\"utf-8\").splitlines()[0]",
                        "    if header.split(\",\") != list(columns):",
                        "        raise RuntimeError(f\"{path.name} の列が違います: {header}\")",
                        "",
                        "",
                        "SMOKE_MAIN = {",
                        '    "calibration_set": "full",',
                        '    "tensorizations": ("T2",),',
                        '    "budgets": ("25",),',
                        '    "methods": ("M1", "M2", "M3", "M4", "M5"),',
                        '    "m5_initials": ("M3",),',
                        '    "record_initials": ("M3", "M4_retruncated"),',
                        '    "max_sweeps": 1,',
                        '    "max_cg_iterations": 10,',
                        '    "relative_tolerance": CG_RELATIVE_TOLERANCE,',
                        '    "sweep_tolerance": SWEEP_RELATIVE_DECREASE,',
                        '    "file_suffix": "_smoke",',
                        '    "compute_perplexity": True,',
                        '    "use_experiment_stop": False,',
                        "}",
                        "SMOKE_SENSITIVITY = {",
                        '    "calibration_set": "sensitivity_16",',
                        '    "tensorizations": ("T2",),',
                        '    "budgets": ("25",),',
                        '    "methods": ("M2", "M3", "M5"),',
                        '    "m5_initials": "adopted_from_full",',
                        '    "record_initials": ("M3", "M4_retruncated"),',
                        '    "max_sweeps": 1,',
                        '    "max_cg_iterations": 10,',
                        '    "relative_tolerance": CG_RELATIVE_TOLERANCE,',
                        '    "sweep_tolerance": SWEEP_RELATIVE_DECREASE,',
                        '    "file_suffix": "_smoke",',
                        '    "compute_perplexity": False,',
                        '    "use_experiment_stop": False,',
                        "}",
                        "smoke_started = time.perf_counter()",
                        "remove_smoke_outputs()",
                        "first = execute_plan(SMOKE_MAIN)",
                        "summary_text = (RESULTS_DIR / \"summary_smoke.csv\").read_text(encoding=\"utf-8\")",
                        "second = execute_plan(SMOKE_MAIN)",
                        "if (RESULTS_DIR / \"summary_smoke.csv\").read_text(encoding=\"utf-8\") != summary_text:",
                        "    raise RuntimeError(\"再開の再実行で summary_smoke.csv の内容が変わりました。\")",
                        "if first[\"als_ran\"] != [\"full|T2|25|M3\"]:",
                        "    raise RuntimeError(f\"動作確認の M5 が想定と違います: {first['als_ran']}\")",
                        "if second[\"als_ran\"] or second[\"als_skipped\"] != [\"full|T2|25|M3\"]:",
                        "    raise RuntimeError(",
                        "        f\"再開で M5 を飛ばせていません: ran={second['als_ran']} skipped={second['als_skipped']}\"",
                        "    )",
                        "third = execute_plan(SMOKE_SENSITIVITY)",
                        "if third[\"als_ran\"] != [\"sensitivity_16|T2|25|M3\"]:",
                        "    raise RuntimeError(f\"感度確認の M5 が想定と違います: {third['als_ran']}\")",
                        "if (RESULTS_DIR / \"decision.json\").exists():",
                        "    raise RuntimeError(\"動作確認が decision.json を作りました。\")",
                        "assert_header(RESULTS_DIR / \"summary_smoke.csv\", SUMMARY_COLUMNS)",
                        "assert_header(RESULTS_DIR / \"m5_history_smoke.csv\", HISTORY_COLUMNS)",
                        "assert_header(RESULTS_DIR / \"m5_initial_values_smoke.csv\", INITIAL_COLUMNS)",
                        "assert_header(RESULTS_DIR / \"bootstrap_smoke.csv\", BOOTSTRAP_COLUMNS)",
                        "assert_header(RESULTS_DIR / \"rho_smoke.csv\", RHO_COLUMNS)",
                        "assert_header(RESULTS_DIR / \"perplexity_smoke.csv\", PERPLEXITY_COLUMNS)",
                        "for name in (",
                        "    \"summary_smoke.csv\",",
                        "    \"m5_history_smoke.csv\",",
                        "    \"m5_initial_values_smoke.csv\",",
                        "    \"bootstrap_smoke.csv\",",
                        "    \"rho_smoke.csv\",",
                        "    \"perplexity_smoke.csv\",",
                        "):",
                        "    lines = (RESULTS_DIR / name).read_text(encoding=\"utf-8\").splitlines()",
                        "    print(f\"--- {name} ({len(lines) - 1} 行) ---\")",
                        "    print(\"\\n\".join(lines[:4]))",
                        "initial = third[\"state\"][\"initial\"][\"full|T2|25|M4_retruncated\"]",
                        "print(",
                        "    \"T2 25% M4_retruncated Q_lambda\",",
                        "    initial[\"q_lambda_before_als\"],",
                        "    \"M4 Q_lambda\",",
                        "    initial[\"q_lambda_m4\"],",
                        "    \"差\",",
                        "    initial[\"q_lambda_minus_m4\"],",
                        ")",
                        "print(f\"動作確認 {time.perf_counter() - smoke_started:.1f}s\")",
                        "print(\"ここで止める。本実行と判定には進まない。\")",
                    ]
                )
            ),
        ]
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{index}", "exec")
    nbformat.write(notebook, NOTEBOOK)
    print(f"cells {len(notebook.cells)}")


if __name__ == "__main__":
    main()
