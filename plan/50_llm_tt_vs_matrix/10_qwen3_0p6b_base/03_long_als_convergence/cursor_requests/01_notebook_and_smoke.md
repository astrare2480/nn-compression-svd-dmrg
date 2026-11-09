# Cursor への依頼：長い ALS の Notebook と、実データでの試運転

作成：2026-11-08　依頼者：レビュー担当（Claude）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence/design.md`（以下「設計」。§ はこの文書の節）

## 0. 背景と範囲

- 予備実験（`01_single_layer_pilot`）の M5 の ALS は 50 掃引で打ち切られ、T2 では初期値（M3 と M4_retruncated）によって Q_λ に約 6% の差が残った。この実験は、停止の規則を決めて長く回し、その差が消えるか残るかを確かめる（設計 §1）。
- **ソルバーと判定関数は実装済み**で、小さい問題の単体テストは通っている（設計 §12）。
  - `src/nn_compression/tt_mps/metric_tt_solver.py`：`ConvergenceStopRule`、`run_metric_tt_als_until_converged`、`ConvergenceRunResult`、`ConvergenceSweepRecord`、`LocalSolverFailure`、`metric_tt_stationarity_on_device`、`perturb_tt_matrix_site`
  - `src/nn_compression/metrics/als_convergence.py`：`classify_als_initialization_dependence`
  - 使い方は各 docstring と `tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py` を見ること。
- 今回お願いすること：
  1. Notebook `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence.ipynb` を作る（試運転・本実行・集計・判定のセルをすべて書く）。
  2. **試運転（§5）だけを実行**して、結果を報告する（§9）。
- **本実行（4 本）はまだ実行しない**。試運転の結果を Claude がレビューしてから回す。
- `src/` は原則として変更しない。ソルバーや判定関数に不具合や足りない機能を見つけたら、直さずに、再現の手順をつけて報告する。
- 予備実験とベンチマーク（`01_single_layer_pilot`・`02_metric_tt_solver_benchmark`）の data・models・results・Notebook には書き込まない（読むだけ）。

## 1. 入力（読むだけ）

`get_experiment_dirs(PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "01_single_layer_pilot", create=False)` の data・models・results を使う。

| 量 | ファイル | 使い方 |
|---|---|---|
| W、W_gate | data の `weights_float64.pt`（キー `W`・`W_gate`、(3072, 1024) float64） | 目的と E_gated |
| 較正の G | data の `calibration_grams.pt` の `total` | `G_λ = regularize_gram(G, 1e-6)` を目的に、G を `report_gram` に使う。`per_document` は使わない。`calibration_activations.pt` は読まない |
| 検証の Gram | data の `validation_same_grams.pt`・`validation_other_grams.pt` の `total` | e と距離 |
| 検証の活性化 | data の `validation_same_activations.pt`・`validation_other_activations.pt` の `activation`（文書 × 1024 × 1024） | E_gated |
| M3 | models の `m3_T2_{50,25}.pt`（`cores`・`out_modes`・`in_modes`） | 初期値。mode もここから取る |
| M4 | models の `m4_T2_{50,25}.pt` | `DiagonalMetricTT(scaled_cores=…, input_scale_inv=…, absorbed_cores=…, scales=…, max_rank=…)` を作り、`prepare_m5_from_diagonal_tt(fitted, out_modes, in_modes, tt_matrix_ranks(M3 のコア))` で **M4_retruncated** を作り直す（予備実験の Notebook の `prepared_initial` と同じ手順） |
| 比較用 | results の `m5_initial_values.csv`（`calibration_set=full`、`tensorization=T2`）、`m5_history.csv`（同、`sweep=50`）、`rho.csv`（同、`calibration_set=full`） | 検算 1、50 掃引目の比較、ρ の e_M2・e_M3 |
| 比較用 | `02_metric_tt_solver_benchmark` の results の `als_trajectory.csv`（`tensorization=T2`、`budget=50`、`solver=cuda_float32`） | 検算 2 |

- ALS には、M3 のコアを**保存されたまま**（正準化せずに）渡す。正準化はソルバーの中で行う。ベンチマークも同じ渡し方なので、検算 2 はビット単位で一致すると予想している。
- 各入力ファイルの SHA-256（`utils.file_sha256`）を `metadata` に入れて、再開のときの照合に使う。

## 2. ソルバーの設定と停止の規則（設計 §3・§5。変えない）

```python
CONFIG = MetricTTSolverConfig(device="cuda", precision="float32_refine", preconditioner="jacobi",
                              cg_tolerance_mode="fixed", cg_relative_tolerance=1e-8,
                              max_cg_iterations=500, max_refinements=5)
FULL_RULE = ConvergenceStopRule()                                 # 既定値が設計 §5 の値
SMOKE_RULE = ConvergenceStopRule(max_sweeps=5, checkpoint_every=2)
EQUIVALENCE_TOLERANCE = 1e-3                                      # 設計 §9 の 0.1%
```

- `metadata` には少なくとも、`budget`、`initial`（`"M3"`・`"M4_retruncated"`・`"M3_perturbed_site1"`）、`run_kind`（`"full"` か、試運転の中の区別）、入力ファイルの SHA-256 を入れる。
- 保存ファイルの置き場：`get_experiment_dirs(PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "03_long_als_convergence")` の models に、`long_als_T2_{budget}_{initial}.pt`（本実行）、`long_als_T2_{budget}_{initial}_smoke.pt`（試運転）、`long_als_T2_50_M3_resume_smoke.pt`（再開の確認）。

## 3. 診断の関数（設計 §7。停止の判定には使わない）

`run_metric_tt_als_until_converged(..., diagnostics=…)` に、次を返す関数を渡す（保存のたびに呼ばれる。本実行では 50 掃引ごとと停止時）。

- `e_validation_same`・`e_validation_other`：`output_error_ratios_from_gram(W, Ŵ, Gram)["e"]`（Ŵ は `tt_matrix_to_dense(cores, out_modes, in_modes)`）
- `E_gated_validation_same`・`E_gated_validation_other`：`gated_error_parts_per_document(W, Ŵ, W_gate, list(activation.unbind(0)))` の分子の和 / 分母の和
- `e_calibration`：`output_error_ratios_from_gram(W, Ŵ, G)["e"]`（G は正則化なし）

これらの関数は CPU の float64 を要求する。診断と保存にかかった時間は `result.save_timings` に入る。

## 4. Notebook の構成

1. 目的と設計へのリンク（markdown）。「この Notebook は予備実験のファイルを読むだけ」と明記する。
2. 設定：`RUN_SMOKE = True`、`RUN_FULL = False`、パス、§2 の設定値、環境（GPU 名、torch・CUDA の版）。
3. 入力の読み込みと SHA-256。
4. 初期値の準備：4 本（T2 × 50・25 × M3・M4_retruncated）。**検算 1**（§5 の 1）。
5. 診断の関数（§3）と、記録を CSV の行に直す関数。
6. 試運転（`RUN_SMOKE` のときだけ）：§5。
7. 本実行（`RUN_FULL` のときだけ）：4 本を 50%・M3 → 50%・M4_retruncated → 25%・M3 → 25%・M4_retruncated の順に回す。保存ファイルがあれば続きから回る（ソルバーが照合する）。進み具合を 10 掃引ごとに表示する（掃引、Q_λ、相対減少、s_max、解き直しの有無、経過時間）。
8. 検算 3・4（本実行の 4 本について）。
9. 集計（§6）と判定（§7）。
10. 摂動の追加の 1 本（§7 の条件を満たした予算だけ）。
11. 保存。

## 5. 試運転（設計 §6.1。今回実行するのはここまで）

ファイル名にはすべて `_smoke` を付ける。どれかが通らなければ、その場で止め、直そうとせずに報告する。

- 試運転のセルの最初で、`03_long_als_convergence` の models にある `*_smoke.pt` だけを削除する。ソルバーは保存ファイルがあると続きから回る（止まっていれば計算せずに返す）ので、前回の試運転のファイルが残っていると、再開の確認（4）が成り立たない。本実行のファイル（`_smoke` の付かないもの）は消さない。

1. **検算 1**：4 本の初期値について、`result.initial_q_lambda` が `m5_initial_values.csv` の `q_lambda_before_als` と相対 1e-10 で一致する。
2. **検算 2**：T2・50%・M3 を `SMOKE_RULE` で 5 掃引回し、各掃引の Q_λ を `als_trajectory.csv` の `cuda_float32` の 5 行と比べる。相対差の最大値を記録し、1e-6 以下であることを確かめる（ビット単位で一致すると予想しているので、一致したかどうかも記録する）。
3. **4 本の 5 掃引**：残りの 3 本も `SMOKE_RULE` で 5 掃引回す（1 本目は 2 の結果を使う）。各本の掃引あたりの時間、s_max、解き直しの有無、`monotone_ok`、`failure` を記録する。
4. **再開の確認**：T2・50%・M3 について、
   - 2 と同じ規則・同じ `metadata` で、別の保存ファイル（`long_als_T2_50_M3_resume_smoke.pt`）を使って回し、`progress` の中で 3 掃引目の記録を受け取ったときに例外を投げて中断する（2 掃引目の保存は済んでいる）。
   - 同じ呼び出しをもう一度行い、保存した状態から 5 掃引まで再開する。`resumed_from_sweep == 2` を確かめる。
   - 2 の中断なしの本と、**時間以外の記録すべて**（`ConvergenceSweepRecord` の `seconds*` と `peak_memory_bytes` を除く項目）、**最終のコア**（`torch.equal`）、**診断値**がビット単位で一致することを確かめる。一致しなければ、違う項目と、コアの最大の絶対差・相対差を記録して止める（GPU の計算の決定性は保証していないため。設計 §6.1）。
5. **時間の内訳**：本ごとに、ALS 本体（`seconds − seconds_stationarity − seconds_q_unregularized`）、s_k の計算、正則化なしの Q、診断、保存の時間を分けて出す。
6. **本実行の時間の見積もり**：掃引 2〜5 の平均時間 × 1,000 ＋ 診断と保存の時間 × 20 を、4 本について足す。**5 時間を超える見込み**なら、報告で目立つように書く。
7. 検算 3・4 を、試運転の 4 本についても行う（§7 の 1・2）。

## 6. 集計（本実行の結果があるときだけ動くように書く。今回は実行しない）

予算ごとに 2 本（M3・M4_retruncated）を比べる。

- **δ_Q**：`relative_objective_difference`（判定関数が返す値）
- **2 つの解の距離**（設計 §4。どれも相対距離で、二乗ではない）：ΔŴ = Ŵ_M3 − Ŵ_M4r として
  - Frobenius：‖ΔŴ‖_F / ‖W‖_F
  - Gram ごと：sqrt(tr(ΔŴ A ΔŴᵀ) / tr(W A Wᵀ))。A は G_λ、G、WT-2 の Gram、C4 の Gram の 4 種類
- **最終の Q_λ の 0.1% 以内に初めて入った掃引数**（本ごと）：Q_λ,s ≤ Q_λ,final × (1 + 1e-3) となる最初の s
- **50 掃引目との比較**（本ごと）：予備実験の 50 掃引目（`m5_history.csv`）と、この実験の 50 掃引目の Q_λ の相対差。合否の条件にはしない（設計 §6 の最後の段落）
- **ρ**（本ごと、領域ごと、50 掃引目と最終）：ρ = (e_M3 − e_M5) / (e_M3 − e_M2)。e_M2・e_M3 は予備実験の `rho.csv`（`calibration_set=full`、`tensorization=T2`、同じ予算・領域）の値。e_M5 は、較正は `e_calibration`、検証は `e_validation_same`・`e_validation_other`。点推定だけ（区間はつけない）
- E_gated の 50 掃引目と最終の値も並べる（診断）

## 7. 検算 3・4、判定、摂動（設計 §6・§9）

1. **検算 3**：全掃引で `monotone_ok` が真。
2. **検算 4**：最終のコアから CPU の float64 で Q_λ を計算し直し（`tt_matrix_to_dense` と `G_λ` で `tr(ΔW G_λ ΔWᵀ)`）、`history` の最後の記録の `q_lambda` と相対 1e-10 で一致する。最終のコアは常に最後に完了した掃引の終わりのものなので、`local_solver_unresolved` で止まった本も同じ比べ方でよい。`history` が空（1 掃引目で失敗）なら比べず、「完了した掃引なし」と記録する。
3. **判定**：予算ごとに `classify_als_initialization_dependence(stop_M3, stop_M4r, Q_M3, Q_M4r, equivalence_tolerance=EQUIVALENCE_TOLERANCE)`。結果を見てから規則や幅を変えない。
4. **摂動の 1 本**：判定が `converged_initialization_dependent` になった予算だけ、`perturb_tt_matrix_site(M3 の保存されたままのコア, out_modes, in_modes, site=1, relative_size=1e-2, seed=0)` で初期値を作り、`FULL_RULE` で 1 本回す（`initial="M3_perturbed_site1"`）。
   - 返ってきた δ_W（約 1e-2 のはず）を記録する。
   - M3 の本・M4_retruncated の本それぞれとの δ_Q と距離を出す。参考として `classify_als_initialization_dependence` を (M3, 摂動) の組にも当てはめて記録するが、**§9 の判定は変えない**。

## 8. 出力（`03_long_als_convergence` の results。試運転は `_smoke` を付ける）

| ファイル | 中身 |
|---|---|
| `settings.json` | §2 の設定値、規則、入力ファイルの SHA-256、環境、`RUN_SMOKE`・`RUN_FULL` |
| `checks.json` | 検算 1〜4 の値と合否、再開の確認（一致したか、違った項目、コアの最大の差） |
| `history.csv` | 1 行 = 1 本の 1 掃引。予算、初期値、`ConvergenceSweepRecord` の全項目（タプルは `;` でつなぐ。s_k は `s_0`・`s_1`・`s_2` の列に分ける） |
| `diagnostics.csv` | 1 行 = 1 本の 1 回の保存。予算、初期値、掃引、§3 の値、診断と保存の時間 |
| `failures.json` | `local_solver_unresolved` で止まった本の `LocalSolverFailure`（なければ空のリスト） |
| `timing.json` | §5 の 5・6 の内訳と見積もり |
| `comparison.csv` | 予算ごと：2 本の停止理由・完了した掃引の数・最終の Q_λ、δ_Q、判定、5 種類の距離、0.1% 以内に入った掃引数、50 掃引目の比較 |
| `rho.csv` | 予算、初期値、領域、時点（50・final）、e_M5、e_M2、e_M3、ρ、E_gated |
| `decision.json` | 予算ごとの判定と δ_Q、使った等価幅 |
| `perturbation.csv` | 摂動の 1 本を回したときだけ |

試運転では `settings_smoke.json`・`checks_smoke.json`・`history_smoke.csv`・`diagnostics_smoke.csv`・`failures_smoke.json`・`timing_smoke.json` を出す（`comparison`・`rho`・`decision` は出さない）。

## 9. 実行のしかたと報告

- 実行は、`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/_execute_notebook.py` に、実行する Notebook のパスを引数で受け取れるようにする変更を入れて使う（引数がなければ今までどおり `01_single_layer_pilot.ipynb`）。出力のログを残す。
- 実行の前後で `tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py`・`tests/test_metric_tt_solver.py` が通ることを確かめる。
- 報告してほしいこと：
  1. §5 の 1〜7 の結果（検算の数値、ビット単位で一致したか、各本の 5 掃引の Q_λ と s_max、解き直しや失敗の有無）
  2. 時間の内訳と、本実行の見積もり
  3. 設計や依頼と違うことをした点と、その理由
  4. 気づいた不具合や、設計の曖昧な点（直さずに）

## 10. やらないこと

- 本実行（`RUN_FULL = True`）、摂動の本の実行
- `src/` の変更（不具合は報告だけ）、ソルバーの設定・停止の規則・判定の幅の変更
- T1、別の層、λ の変更、perplexity の評価、adaptive の許容誤差、Kronecker 前処理（設計 §10）
- 予備実験・ベンチマークのファイルへの書き込み
- 検証データを停止の判定に使うこと
