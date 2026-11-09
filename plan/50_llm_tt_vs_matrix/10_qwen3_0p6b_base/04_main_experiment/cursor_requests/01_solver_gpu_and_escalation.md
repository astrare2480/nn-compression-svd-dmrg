# Cursor への依頼：本実験のソルバー（GPU の同期削減・解き直しの 3 段・新しい停止の規則）と、その確認

作成：2026-11-08　依頼者：レビュー担当（Claude）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md`（以下「設計」。§ はこの文書の節）

## 0. 背景と範囲

- 本実験は、Qwen3-0.6B-Base の 3 層 × 7 射影 × 2 予算などで、計量つきの TT（M5）を 1 サイトの ALS で求める（設計 §2）。そのためのソルバーの変更を、実験の Notebook より先に行う。
- 今回お願いすること（A〜G）：
  - A. プロファイル（設計 §6.2 の 1）
  - B. CG の中で CPU に値を取り出す回数を減らす（設計 §6.2 の 2）
  - C. CG の終わり方の状態を返す
  - D. 3 段の解き直し（設計 §3.1）と、新しい停止の規則（設計 §4.1）を持つ ALS の関数
  - E. layer 14 の `down_proj` の較正の Gram 行列・M2・M3 を作る（F で使う。Gram 行列を作る関数は本実験でも使う）
  - F. 同等性の確認（設計 §6.2 の合格の条件）
  - G. τ_warn の事前の確認の 1 つ目（設計 §3.2 の 1）
- **03 の結果を変えない**：既存の `run_metric_tt_als_until_converged`・`ConvergenceStopRule`・`run_metric_tt_als_on_device`・`solve_metric_tt_site_on_device` の計算は、既定の設定でビット単位で今と同じにする。新しい機能は、新しい引数・新しい関数で足す。
- 本実験の Notebook、試運転、ほかの層・射影のデータは**次の依頼**で行う。今回はやらない（§10）。
- 予備実験・ベンチマーク・03 の data・models・results・Notebook には書き込まない（読むだけ）。
- 不具合や設計の曖昧な点を見つけたら、直さずに、再現の手順をつけて報告する（今回の範囲の `src/` の変更は除く）。
- 計測（A・F）は、GPU を実験以外で使っていないときに行い、取り合いの有無を記録する（設計 §6.5）。

## 1. 入力（読むだけ）

| 量 | ファイル |
|---|---|
| up_proj の W・G | `data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/` の `weights_float64.pt`（`W`）、`calibration_grams.pt`（`total`） |
| up_proj の M3（T2） | `models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/m3_T2_50.pt` |
| up_proj の Q_λ(M3) | `results/.../01_single_layer_pilot/m5_initial_values.csv`（`calibration_set=full, tensorization=T2, budget=50, initial=M3` の `q_lambda_before_als`） |
| 予備実験の較正データの選び方 | `results/.../01_single_layer_pilot/data_preparation.json`（`splits.calibration`）と、予備実験の Notebook `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot.ipynb` のデータ準備のセル |
| 03 の保存状態 | `models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence/long_als_T2_50_M3.pt`（A で使う）、`long_als_T2_50_M4_retruncated.pt`（G で使う。426 掃引目の終わり、`stop_reason=local_solver_unresolved`）、`results/.../03_long_als_convergence/failures.json`、`history_smoke.csv` |
| モデル | `data/hf_cache/`（予備実験の `download_record.json` の revision。オフラインで読む） |

## 2. A：プロファイル（設計 §6.2 の 1）

- 03 の `long_als_T2_50_M3.pt` の `cores`（1,000 掃引目の終わり、中心 0）を初期値にして、up_proj・T2・50% で、03 と同じ設定（設計 §3 の冒頭）で 1 掃引回す。
- `torch.profiler`（CPU と CUDA）で、サイト 1 の 1 回の更新について、次の時間を分ける：HVP（密行列を作る部分、G を掛ける行列積、随伴の縮約）、前処理、dot・norm、`.item()` などで CPU が GPU を待つ時間、カーネルの間の空白、QR、受理の判定。
- 20 掃引回す間、`nvidia-smi` で GPU の使用率を 1 秒ごとに記録する。
- 「HVP 以外の約 1.35 秒/掃引の主な原因は `.item()` の同期」という見立て（設計 §6.1。コードを読んだだけ）が合っているかを、数字で報告する。

## 3. B：CG の同期を減らす（設計 §6.2 の 2）

- `MetricTTSolverConfig` に `cg_check_every: int = 1` を足す。
  - **1 は今と同じ計算**（ビット単位で同じ）。
  - k > 1 のとき：CG を k 回ずつまとめて回し、相対残差を CPU に取り出して収束を確かめるのは、k 回ごとの区切りだけにする。
  - 曲率 ≤ 0 と、値が有限でないことは、GPU の上のフラグにためて、区切りで確かめる。フラグが立った反復の更新とそれ以降の更新は、解 x に入れない（`torch.where` などで止める）。x は最後の正常な反復のものを返す。
  - 区切りの途中で許容誤差に届いても、区切りの終わりまで回してよい（最大 k − 1 回余分）。`max_cg_iterations` は超えない。
  - 返す反復の回数は、実際に回した回数。
  - 外側の反復ごとの `.item()`（float64 の残差を測るところなど）は今のままでよい。
- 本実験では k = 10 を使う。
- CUDA Graph、batched CG、複数のプロセスは、今回はやらない（設計 §6.2 の 3 以降）。

## 4. C：CG の終わり方の状態を返す

- `_conjugate_gradient` が、終わり方の状態（`converged`、`max_iterations`、`nonpositive_curvature`、`nonfinite`）を返すようにする。`cg_check_every=1` でも状態を返す（計算は変えない）。
- 局所の解き方（`_solve_site` など）は、外側の反復ごとの状態と、曲率 ≤ 0 が起きた回数を集めて返す。
- **扱い（設計 §3.1 の修正。§11 を参照）**：
  - 値が有限でない → 「解けなかった」（`local_solver_unresolved`）。
  - float32 の内側の CG での曲率 ≤ 0 → 今と同じく、その内側の CG を終えて外側の反復（float64 で残差を測り直す）へ進む。**それだけでは「解けなかった」にしない。** 回数を記録する。
  - float64 の CG（`precision="float64"` のとき）での曲率 ≤ 0 → 「解けなかった」。

## 5. D：3 段の解き直しと、新しい停止の規則を持つ ALS の関数

### 5.1 新しい停止の規則

`StudyResolutionStopRule`（名前は変えてよい）を足す。既定値は設計の値にする。

| 引数 | 既定値 | 意味 |
|---|---|---|
| `d_ref` | （必須、正） | 設計 §4.1 の D_ref = max(Q_λ(M3) − Q_λ(M2), 0.05·Q_λ(M3))。呼び出し側が計算して渡す |
| `window_sweeps` | 50 | 窓の長さ。止める判断と保存は、この倍数の掃引でだけ行う |
| `resolution_tolerance` | 1e-4 | Δ₅₀ のしきい値 |
| `consecutive_windows` | 2 | 続けて満たす回数 |
| `max_sweeps` | 1000 | 上限 |
| `local_residual_tolerance` | 1e-8 | 局所の目標 |
| `escalation_refinements` | (10, 50) | 2 段目・3 段目の `max_refinements` |
| `warn_residual_tolerance` | 1e-8 | τ_warn。1e-8（＝目標と同じ）なら警告の区分はない |

- Δ₅₀(s) = (Q_λ(s − 50) − Q_λ(s)) / d_ref。Q_λ(0) は、渡された初期値（正準化の前）の Q_λ（`initial_q_lambda`）。
- 停止の理由は `study_resolution_reached`（Δ₅₀(s) < 1e-4 かつ Δ₅₀(s − 50) < 1e-4）、`max_sweeps`、`local_solver_unresolved` の 3 つ。
- s_max（掃引の終わりの停留性）は 03 と同じく毎掃引記録するが、止める判断には使わない。1 掃引ごとの相対減少も使わない。

### 5.2 3 段の解き直し（設計 §3.1）

- 1 段目は `config.max_refinements`（5）。1e-8 に届かなければ 2 段目（10 で最初から）、まだ届かなければ 3 段目（50 で最初から）。
- 区分：
  - どれかの段で残差 ≤ 1e-8 → 目標に届いた。ふだんの受理の判定（Q_λ が増えなければ受理、増えれば棄却）へ。
  - 3 段目の残差 ≤ τ_warn、値が有限 → 警告 `local_target_missed`。3 段目の解について、ふだんの受理の判定へ（Q_λ が増えれば、ふだんどおり棄却して続ける）。
  - 3 段目の残差 > τ_warn、または値が有限でない（§4）→ `local_solver_unresolved`。扱いは 03 と同じ（失敗した掃引は捨て、最後に完了した掃引の終わりを最終状態にし、診断を別に保存する）。
- 更新ごとに記録する：何段目まで進んだか、各段の残差・CG の反復数・外側の反復数・時間・CG の状態、曲率 ≤ 0 の回数、区分。3 段目まで進んだ更新では、2 段目と 3 段目の候補の Q_λ と、確認の値 a = (Q_λ(2 段目の候補) − Q_λ(3 段目の候補)) / d_ref も記録する。

### 5.3 関数

- `run_metric_tt_als_study`（名前は変えてよい）を足す。掃引の順、正準形の扱い、受理の規則、保存と再開（設定・入力のハッシュ・`metadata` の照合、止まった保存ファイルからはそのまま返す）、`diagnostics`・`progress` の呼び方は、`run_metric_tt_als_until_converged` と同じにする。保存ファイルの形式の名前は新しくする。
- 保存のたびに、Δ₅₀ の値、続けて満たした回数、警告の数、2・3 段目に進んだ数も記録する。
- 既存の関数と共通の部分をまとめ直してもよいが、§0 の「03 の結果を変えない」を守ること。

## 6. E：layer 14 の `down_proj` の較正の Gram 行列・M2・M3

- **Gram 行列を作る関数を `src/nn_compression/llm/` に足す**：1 回の順伝播で、複数のモジュールの入力（forward pre-hook）から、Gram 行列（平均を引かない二次モーメント、float64）を系列ごとに足し合わせる。活性化はためない（`down_proj` の入力は 3072 次元で、ためると大きいため）。戻り値は、モジュールの名前ごとの `total` と `n_tokens`。本実験の次の依頼でも、全層・全射影に使う。小さいモデルでの単体テストを書く（既存の `gram_matrix` を活性化に当てた結果と一致すること）。
- 較正データは予備実験と**同じ選び方**で作る（WikiText-103 train、`select_token_prefix_sequences`、seed 0、128 系列 × 1,024 トークン。引数は予備実験の Notebook のとおり）。
  - 選ばれた 128 文書の hash が、`data_preparation.json` の `splits.calibration.document_hashes` と**順番まで一致する**ことを確かめる。
- モデルの読み込みと順伝播の dtype・device は、予備実験の Notebook と同じにする。
- 同じ順伝播で `layers[14].mlp.up_proj` と `layers[14].mlp.down_proj` の入力の Gram 行列を作る。
  - **検算**：up_proj の Gram 行列が、予備実験の `calibration_grams.pt` の `total` と一致する（相対 Frobenius 差 ≤ 1e-12。ビット単位で一致したかも記録する）。合わなければ止めて報告する。
- `down_proj` の重み W_down（bf16 → float64）を保存する。
- `down_proj`・T2・50% の M3 と M2：
  - mode とランクは `select_balanced_tt_matrix_modes(1024, 3072, 3)` と `select_uniform_tt_rank` で作り、設計 §2.2 の表（出力 (2,8,64)、入力 (64,16,3)、ランク (109,109)）と一致することを確かめる。M3 は `dense_to_tt_matrix_cores`。
  - M2 は `whitened_truncated_svd`、rank は `max_matrix_rank_within_budget(1024, 3072, 1024·3072/2)`。M2 の Q_λ が、捨てた白色化の特異値の二乗和と一致することを確かめる（予備実験の検算 2 と同じ）。
  - G_λ = `regularize_gram(G, 1e-6)`。
- D_ref を、up_proj・T2・50%（Q_λ(M3) は §1 の CSV、Q_λ(M2) は予備実験のデータから計算し直す）と、down_proj・T2・50% について計算する。
- 保存先：`data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/`（git の管理対象外）に、Gram 行列（射影ごと）、W_down、選んだ文書の hash。M3 のコアは `models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev/`。

## 7. F：同等性の確認（設計 §6.2 の合格の条件。前もって固定）

1. **既存の関数が変わっていないこと**：`run_metric_tt_als_until_converged` を 03 と同じ設定・既定の規則で、up_proj・T2・50%・M3 から 5 掃引回し、03 の `history_smoke.csv`（50%・M3 の 5 行）と、時間以外の項目がビット単位で一致することを確かめる。
2. **CG の同期を減らす前と後**：新しい関数（§5.3）で、up_proj・T2・50%・M3 と down_proj・T2・50%・M3 を、それぞれ `cg_check_every=1` と `cg_check_every=10` で 50 掃引ずつ回す（`max_sweeps=50`、τ_warn = 1e-8、d_ref は §6 の値）。
   - **合格の条件**：
     - (a) すべての更新で、区分（目標に届いた・警告・解けなかった）が同じ。何段目まで進んだかの違いは記録する。
     - (b) 50 掃引後の |Q_λ(k=10) − Q_λ(k=1)| / d_ref ≤ 1e-6。
   - 記録する：受理・棄却の判定の違い、CG の反復の回数、1 掃引の時間、1 秒あたりの HVP の回数、GPU の使用率（1 秒ごと）、曲率 ≤ 0 の回数。
   - 合格しなければ止めて報告する。k の値を変えて試し直さない。

## 8. G：τ_warn の事前の確認の 1 つ目（設計 §3.2 の 1）

- 03 の `long_als_T2_50_M4_retruncated.pt` の `cores`（426 掃引目の終わり、中心 0）を `torch.load` で直接読み、新しい関数で、up_proj・T2・50%、`cg_check_every=10`、仮の τ_warn = 1e-7、d_ref は up_proj・50% の値、`max_sweeps=5` で回す（保存ファイルは別の名前にする）。
- 03 で止まった局所問題（427 掃引目のサイト 1 にあたる、1 掃引目のサイト 1）について、各段の残差、3 段目の残差が 1e-7 以下に届いたか、確認の値 a、時間を記録する。03 の `failures.json` の値（最初の残差 1.07e-7、解き直したあとの残差 1.19e-8）と並べる。
- 5 掃引の間に 2 段目以上に進んだ更新すべてを同じように記録する。
- **τ_warn の値はここでは決めない**。試運転（次の依頼）の結果とあわせて、設計 §3.2 の規則で決める。

## 9. 置き場所と出力

- Notebook：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_solver_dev.ipynb`（A・E・F・G をこの順に。実行は既存の `_execute_notebook.py` に Notebook のパスを渡す）
- results：`results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev/`
  - `settings.json`（設定、入力ファイルの SHA-256、環境、GPU の取り合いの有無）
  - `profile.json`（A の内訳）、`gpu_utilization.csv`
  - `data_checks.json`（E の hash の一致、up_proj の Gram の一致、mode・ランク、M2 の検算、D_ref）
  - `equivalence.json`・`equivalence_history.csv`（F の 1・2）
  - `tau_warn_precheck.json`（G）
- models：`models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev/`（M3 のコア、F・G の保存ファイル）
- 大きい trace ファイルは results に入れない。
- `docs/90_src設計/05_Core_API_v1/Internal_API.md` に、足した関数・引数の行を足す。

## 10. テストと報告

- 新しいテスト（小さい問題）：
  - `cg_check_every=1` が今とビット単位で同じ。
  - `cg_check_every=10` で同じ許容誤差に届く。曲率 ≤ 0・値が有限でないフラグが立ったとき、x がそれ以降の更新を含まない。
  - CG の状態が返る。
  - 3 段の解き直しの各区分（目標に届いた・警告・解けなかった）。`max_refinements` を小さくして段を進ませるなどでよい。
  - 新しい停止の規則の各分岐（Δ₅₀ の 2 回続けて、上限、解けなかった）。判定の部分を純粋な関数に分けて、作った Q_λ の列で試してもよい。
  - 新しい関数の保存と再開（再開の前後で履歴と結果が一致）。
  - Gram 行列を作る関数。
- 既存のテスト（`tests/test_metric_tt_solver.py`・`tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py`・`tests/test_qwen3_metric_pilot_checks.py`）がすべて通ること。
- 報告してほしいこと：
  1. A の内訳と、`.item()` の同期の見立てが合っていたか
  2. E の検算の結果と D_ref の値
  3. F の 1・2 の結果（合否、数値、時間、GPU の使用率、曲率 ≤ 0 の回数）
  4. G の結果
  5. 設計や依頼と違うことをした点と、その理由
  6. 気づいた不具合や、設計の曖昧な点（直さずに）

## 11. 設計からの修正（Claude。依頼文の作成時）

設計 §3.1 の表では「CG で曲率が 0 以下になった → 解けなかった」としていたが、次のように直す（設計にも反映済み）。

- 局所ヘッセ行列は正定値（最小固有値 ≥ G_λ の最小固有値 > 0。予備実験の設計 §6.2）なので、正確な計算なら曲率は必ず正である。
- 一方、内側の CG は float32 で、G の条件数は約 3e14 と大きい。float32 の丸め（相対で約 6e-8）のため、収束の近くで曲率が 0 以下と計算されることは、数値の誤差としてありうる。これは外側の反復（float64 で残差を測り直して直す）で正される。
- そのため、float32 の内側の CG での曲率 ≤ 0 だけでは「解けなかった」にせず、記録にとどめる（§4）。最終的に解けたかどうかは、float64 で測った残差（3 段目の残差と τ_warn）と、値が有限かどうかで決める。

## 12. やらないこと

- 本実験の Notebook、試運転、ほかの層・射影のデータ、検証データの準備、NLL・サブ層の出力の評価（次の依頼）
- τ_warn の値を決めること
- k（`cg_check_every`）や合格の条件を、結果を見て変えること
- CUDA Graph、batched CG、複数のプロセス、TF32・bf16、局所の許容誤差を緩める方法（設計 §6.4）
- 既存の関数の既定の動作の変更
- 予備実験・ベンチマーク・03 のファイルへの書き込み
