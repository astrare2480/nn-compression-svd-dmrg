# Cursor への依頼 01c：解き直しを「前の段の続きから」にする

作成：2026-11-08　依頼者：レビュー担当（Claude）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md`（以下「設計」。§3.1 と §6.2b）
前の依頼：`01_solver_gpu_and_escalation.md`、`01b_blocked_cg_rework.md`

## 0. 背景と範囲

- `run_metric_tt_als_study` の 3 段の解き直しは、今は段ごとに**コアの初期値から最初から**解き直している（2 段目は外側の反復 10 回、3 段目は 50 回）。
- Claude の確認（設計 §6.2b の 3）：03 の 50%・M4r の 426 掃引目の終わりから 5 掃引回すと、真ん中のコアの更新 10 回すべてが 3 段目まで進み、3 段目では外側の反復 11〜18 回で目標の 1e-8 に届いた。1 掃引に 26〜34 秒かかった（ふだんの 11〜15 倍）。1・2 段目の計算（外側の反復 15 回分）を捨てていることが無駄になっている。
- 今回お願いすること：解き直しを、**前の段で得た解の続きから**行うようにする。目標の残差（1e-8）、τ_warn、区分の決め方、受理の判定は変えない。
- 実装、単体テスト、1〜2 分で終わる確認まで（§4）。長い計算は Claude が回す（§6）。

## 1. 変更の内容

- `StudyResolutionStopRule.escalation_refinements` の意味を、**外側の反復の累計の上限**にする。既定値 (10, 50) はそのまま：
  - 1 段目：`config.max_refinements`（5）回まで
  - 2 段目：1 段目の解から続けて、累計 10 回まで（つまり最大 5 回足す）
  - 3 段目：2 段目の解から続けて、累計 50 回まで（つまり最大 40 回足す）
  - 累計の上限は、前の段の上限より大きいこと（`5 < 10 < 50`）を検証する。
- 実装の案（同じ働きなら別の書き方でよい）：段 k では、`current` のサイトのコアを前の段の解に置き換えた列を `_solve_site` に渡し、`max_refinements` を「累計の上限 − これまでに使った外側の反復の数」にする。`_solve_site` は渡されたコアを初期値にして、float64 で残差を測り直してから続けるので、右辺・ヘッセ行列・前処理は変わらない（中心以外のコアと G だけで決まる）。
- 記録（`StudyStageRecord`）：各段の残差、その段で使った CG の反復数・外側の反復数・時間に加えて、外側の反復の累計を記録する。確認の値 a = (Q_λ(2 段目の解) − Q_λ(3 段目の解)) / d_ref も今までどおり記録する。
- 旧来の「最初から」の動作は残さなくてよい。残す場合は、規則に `escalation_mode`（`"continue"` が既定、`"restart"` は今の動作）を足す。
- 保存の形式の名前は変える（規則の中身が変わるので、古い保存ファイルから再開しないようにする）。

## 2. 変えないもの

- `cg_check_every=1` の計算、`timing_sync`、`run_metric_tt_als_until_converged`・`run_metric_tt_als_on_device`・`solve_metric_tt_site_on_device` の既定の動作。
- 1 段目で目標に届いた更新の計算（ビット単位で今と同じ）。
- 区分（`target_met`・`local_target_missed`・`local_solver_unresolved`）、曲率と `breakdown` の扱い、停止の規則。

## 3. テスト

- 1 段目で届く場合、今とビット単位で同じ。
- 2・3 段目に進む場合（`max_refinements` を小さくするなどで段を進ませる）：
  - 2 段目が 1 段目の解から始まっている（2 段目の最初の残差が 1 段目の最後の残差と一致する）。
  - 外側の反復の累計が上限を超えない。
  - 区分と受理の判定が規則どおり。
- 既存のテスト（`tests/test_metric_tt_solver.py`・`tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py`・`tests/test_qwen3_metric_pilot_checks.py`・`tests/test_metric_tt_study.py`・`tests/test_accumulate_input_grams.py`）がすべて通る。最後にまとめて 1 回流し、件数を報告する。

## 4. 1〜2 分で終わる確認（Cursor が行う）

- 03 の `models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence/long_als_T2_50_M4_retruncated.pt` の `cores` から、up_proj・T2・50%（W は予備実験の `weights_float64.pt` の `W`、G は `data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/calibration_grams.pt` の `up_proj_total`、`regularize_gram(G, 1e-6)`、d_ref は `results/.../04_main_experiment/solver_dev/data_checks.json` の `d_ref_up`）で、**1 掃引だけ**回す。設定は `cg_check_every=1`、`timing_sync=False`、`warn_residual_tolerance=1e-7`、`max_sweeps=1`。保存ファイルは使わない。
- 記録する：真ん中のコアの 2 回の更新それぞれの各段の残差・外側の反復数・時間、1 掃引の時間、Q_λ。Claude の前回の結果（1 掃引目：1 段目 1.148e-7・1.322e-7、3 段目まで進んで 6.4e-9・8.1e-9、外側の反復は 3 段目だけで 12・11 回、1 掃引 26.3 秒）と並べる。
- 結果は `results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev/escalation_continue_check.json` に保存する。計測の前に `nvidia-smi` の使用率を記録する。

## 5. 報告してほしいこと

1. 変えた関数・引数・保存の形式
2. テストの結果（まとめて 1 回流した件数）
3. §4 の結果と、前回との比較
4. 依頼と違うことをした点と、その理由
5. 気づいた不具合や、設計の曖昧な点（直さずに）

## 6. やらないこと（Claude が回す）

- 5 掃引以上の計算、試運転、本実験の Notebook
- 目標の残差・τ_warn・区分・停止の規則の変更、局所の許容誤差を緩める方法（設計 §6.4）
- 区切りの CG の変更、CUDA Graph、複数のプロセス
- 予備実験・ベンチマーク・03 のファイルへの書き込み
