# Cursor への依頼 02：本実験のデータ準備・方式の作成・ALS の実行の仕組み（試運転の準備まで）

作成：2026-11-08　依頼者：レビュー担当（Claude）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md`（以下「設計」。§ はこの文書の節）
前の依頼：`01_solver_gpu_and_escalation.md`・`01b_blocked_cg_rework.md`・`01c_escalation_continue.md`（ソルバーは完成。設計 §6.2a・§6.2b）

## 0. 背景と範囲

- 本実験（設計 §2）の Qwen3 の部分：層 7・14・21 × 射影 q・k・v・o・gate・up・down × 予算 50%・25% ＝ 42 セル。
- 今回お願いすること（A〜E）：
  - A. データの準備：全 3 層の較正の Gram 行列、重み、検証データの選び方と検証の Gram 行列
  - B. 方式の作成：M1・M2・M3・M4・M4r と、D_ref などの量、検算
  - C. ALS の仕事の一覧（M5・M3F・λ の確認）と、それを **2 つのプロセスで同時に**、途中から再開できるように回す仕組み
  - D. 試運転のモードと、その集計（時間の見積もり、解き直しの割合、τ_warn の事前の確認の 2 つ目）
  - E. 単体テスト
- **作業の分け方**：Cursor は実装・単体テスト・数分で終わる確認まで。データの準備の本番、試運転、本実行は **Claude が回す**（§7）。
- **今回やらないこと**：評価（検証データでの単位ごとの誤差、ブートストラップ、H_c による分類、全体の結論、サブ層の出力の誤差、NLL）は次の依頼 03 で行う。Pythia も別の依頼にする。
- ソルバー（`src/nn_compression/tt_mps/metric_tt_solver.py`）の計算は変えない。不具合や足りない機能を見つけたら、直さずに報告する。
- 予備実験・ベンチマーク・03 のファイルには書き込まない（読むだけ）。

## 1. 共通の設定（設計 §3・§4・§6.2a。変えない）

```python
CONFIG = MetricTTSolverConfig(device="cuda", precision="float32_refine", preconditioner="jacobi",
                              cg_tolerance_mode="fixed", cg_relative_tolerance=1e-8,
                              max_cg_iterations=500, max_refinements=5,
                              cg_check_every=1, timing_sync=False)
LAMBDA = 1e-6                       # G_λ = regularize_gram(G, LAMBDA)
# 停止の規則：StudyResolutionStopRule(d_ref=..., window_sweeps=50, resolution_tolerance=1e-4,
#             consecutive_windows=2, max_sweeps=1000, local_residual_tolerance=1e-8,
#             escalation_refinements=(10, 50), warn_residual_tolerance=TAU_WARN)
```

- **TAU_WARN は設定ファイル**（`results/.../04_main_experiment/settings/tau_warn.json`）から読む。値は試運転のあとで Claude が決めて書く（設計 §3.2）。試運転では仮に 1e-7 を使う。コードに埋め込まない。
- パスはすべて `get_experiment_dirs(PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "04_main_experiment")` から作る（data・models・results）。

## 2. A：データの準備

### 2.1 対象のモジュールと入力のグループ

| 入力のグループ | Gram 行列を取るモジュール | その入力を使う射影 |
|---|---|---|
| `attn_in`（1024 次元） | `layers[L].self_attn.q_proj` | q・k・v |
| `o_in`（2048 次元） | `layers[L].self_attn.o_proj` | o |
| `mlp_in`（1024 次元） | `layers[L].mlp.up_proj` | gate・up |
| `down_in`（3072 次元） | `layers[L].mlp.down_proj` | down |

- L は 7・14・21。q・k・v（gate・up）の入力が同じテンソルであることは、最初の 1 系列で確かめる（hook で取った入力が一致すること）。
- 依頼 01 の `accumulate_input_grams` を使い、**1 回の順伝播で 12 個のモジュールの Gram 行列**を作る。活性化はためない。

### 2.2 較正

- 予備実験と同じ選び方（WikiText-103 train、`select_token_prefix_sequences`、seed 0、128 系列 × 1,024 トークン）。選んだ 128 文書の hash が、予備実験の `data_preparation.json` の `splits.calibration.document_hashes` と順番まで一致することを確かめる（依頼 01 の E と同じ）。
- 検算：layer 14 の `mlp_in` の Gram 行列が、依頼 01 の E で作った `data/.../04_main_experiment/calibration_grams.pt` の `up_proj_total` とビット単位で一致する（同じ計算なので一致するはず。しなければ最大の差を記録して止める）。`down_in` も同様に `down_proj_total` と比べる。
- 保存：`data/.../04_main_experiment/calibration/grams_L{L}.pt`（グループごとの `total`、`n_tokens`）、選んだ文書の hash。

### 2.3 検証データ（設計 §7.2）

| データ | 選び方 | 単位 |
|---|---|---|
| `wt2` | WikiText-2 validation を `restore_wikitext_articles` で記事に分け、1,024 トークン以上の記事を**すべて、全文で**使う（予備実験の記録では 60 記事中 59 記事） | 記事 |
| `c4` | 予備実験と同じ C4 validation のファイルから、1,024 トークン以上の文書を `select_token_prefix_sequences` と同じ並べ替え（seed 0）で **128 文書**、各先頭 1,024 トークン | 文書 |

- 記事の全文は 1,024 トークンより長いことがある。全文を 1 回で流すと系列長が長くなるので、**記事の中で 1,024 トークンずつに切って**流し（重なりなし、最後の端は短くてよい）、記事の単位で足し合わせる。切り方は記録する。
- 較正と検証の文書の hash が重ならないことを確かめる（予備実験と同じ）。
- 単位ごとの情報（単位の id・hash・トークン数・切った区間）を `data/.../04_main_experiment/validation/units.json` に保存する。
- 検証の Gram 行列（ドメインごと・層ごと・グループごとの**合計だけ**）を作って保存する（`validation/grams_{wt2,c4}_L{L}.pt`）。ALS の途中の診断（§4.4）に使う。単位ごとの値は次の依頼 03 で作るので、今回は作らない。

### 2.4 重み

- 3 層 × 7 射影の重み（bf16 → float64）を `data/.../04_main_experiment/weights/W_L{L}_{proj}.pt` に保存する。shape を記録する（q (2048,1024)、k・v (1024,1024)、o (1024,2048)、gate・up (3072,1024)、down (1024,3072)）。バイアスがないことを確かめる。

## 3. B：方式の作成（設計 §7.1）

セル（層・射影・予算）ごとに、ALS を使わない方式を作る。すべて CPU の float64 でよい（M4 の 3072 次元も含め、時間を記録する）。

- **形とランク**：`select_balanced_tt_matrix_modes(m, n, 3)` と `select_uniform_tt_rank`（予算 = m·n の 50%・25%）。設計 §2.2 の表と一致することを確かめる。結合ごとの r_b / r_b,max と、`multi_cut`・`near_saturated_bond`（> 0.9）の分類を記録する（設計 §2.3）。
- **M1**：`truncated_svd_weight`、rank は `max_matrix_rank_within_budget(m, n, 予算)`。
- **M2**：`whitened_truncated_svd(W, G_λ, rank)`。λ = 1e-7・1e-6・1e-5 の 3 通り（設計 §7.7）。M2 の Q_λ が、捨てた白色化の特異値の二乗和と相対 1e-12 で一致することを確かめる。
- **M3**：`dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=...)`。
- **M4**：`fit_diagonal_metric_tt`（予備実験の手順。`n_tokens` は較正のトークン数）。
- **M4r**：`prepare_m5_from_diagonal_tt(M4, out_modes, in_modes, tt_matrix_ranks(M3))`。M4r の初期値は **layer 14 の up 以外の 6 射影だけ**に使うが、作るのは全セルでよい。
- **保存**：M1・M2 は低ランクの因子（U・V など）で、M3・M4・M4r は TT のコアで保存する（密行列は保存しない）。`models/.../04_main_experiment/methods/L{L}_{proj}_{budget}.pt`。
- **量**（`results/.../04_main_experiment/methods_summary.csv`、1 行 = 1 セル）：実際のパラメータ数（M1・M2 は r(m+n)、TT は Σ r_{k−1} m_k n_k r_k、M4 は対角の分を足す）、較正での Q_λ（M1・M2（3 つの λ）・M3・M4・M4r。正則化なしの Q も）、Frobenius 誤差 F = ‖W − Ŵ‖²（M1・M3）、
  - $D_{\mathrm{ref}} = \max\bigl(Q_\lambda(M_3) - Q_\lambda(M_2),\ 0.05\,Q_\lambda(M_3)\bigr)$（λ ごと）と、`small_baseline_gap` の印
  - $D^{F}_{\mathrm{ref}} = \max\bigl(F(M_3) - F(M_1),\ 0.05\,F(M_3)\bigr)$（M3F 用）
- 検算：切り詰めなしの TT-SVD が W を相対 1e-10 で再現する（各射影で 1 回）。パラメータ数が式どおり。

## 4. C：ALS の仕事の一覧と、実行の仕組み

### 4.1 仕事の一覧

1 つの仕事 = 1 本の ALS。`results/.../04_main_experiment/jobs.json` に、決まった順番で並べる。

| 種類 | 対象 | 初期値 | 目的 | 本数 |
|---|---|---|---|---|
| `M5` | 42 セル | M3 | G_λ（λ = 1e-6） | 42 |
| `M5_M4r` | layer 14 の q・k・v・o・gate・down × 2 予算 | M4r | G_λ（λ = 1e-6） | 12 |
| `M3F` | 42 セル | M3 | Frobenius（G = 単位行列。D^F_ref を使う） | 42 |
| `M5_lambda` | layer 14 の q・o・gate・down × 2 予算 × λ = 1e-7・1e-5 | M3 | G_λ（その λ。D_ref もその λ） | 16 |

- 順番：layer 14（down から。設計 §2.1）→ layer 7 → layer 21。同じ層の中は、M5 → M3F → M5_M4r → M5_lambda。
- 仕事の id の例：`L14_down_50_M5_M3init_lam1e-06`。
- M3F は G = 単位行列なので、`run_metric_tt_als_study` に単位行列を `gram` として渡せばよい（局所の方程式はほぼ 1 回で解ける）。`report_gram` には正則化なしの G を渡し、計量つきの誤差も記録する。
- 各仕事の `metadata` に、セル・種類・初期値・λ・入力ファイルの SHA-256・モデルの revision を入れる。

### 4.2 実行の仕組み

- スクリプト `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_run_jobs.py` を作る。引数で `--workers 2`、`--mode {smoke,full}`、`--only <id の一部>` を受け取る。
- **2 つのプロセスで同時に回す**（設計 §6.2a）。各プロセスは、一覧の先頭から「まだ終わっていない、だれも取っていない」仕事を 1 つずつ取る。取ったことは、ロックのファイルを排他的に作る（`os.open(..., O_CREAT | O_EXCL)`）ことで示し、中にプロセスの番号を書く。起動のときに、終わったプロセスが残したロックは消す。
- 各仕事は `checkpoint_path`（`models/.../04_main_experiment/als/{mode}/{job_id}.pt`）つきで回す。中断（設計 §6.6 の `resource_interrupted`）しても、もう一度起動すれば 50 掃引ごとの保存から続く。止まった仕事は計算せずに飛ばす。
- 数値の失敗（`local_solver_unresolved`、検算の失敗）は記録して次の仕事へ進み、回し直さない。
- 仕事が終わるたびに、`results/.../04_main_experiment/als/{mode}/{job_id}_history.csv`（掃引ごとの記録。依頼 01 の `claude_check_*_history.csv` と同じ列に、更新ごとの段の数・解き直しの数・警告の数を足す）と、`jobs_status.csv`（1 行 = 1 仕事：停止の理由、掃引の数、最終の Q_λ、最後の Δ₅₀、s_max、警告・2 段目・3 段目の数、時間、ピークのメモリ、検算の結果）を更新する。
- 進み具合をログに出す（10 掃引ごと：仕事の id、掃引、Q_λ、Δ₅₀、解き直し、経過時間）。
- 実行の間、`nvidia-smi` で GPU の使用率を 10 秒ごとに記録する（`gpu_utilization_{mode}.csv`）。開始の前に、ほかに GPU を使う Python がないかを記録する。

### 4.3 仕事ごとの検算（設計 §7.8）

- M5：最終の Q_λ が M3 の Q_λ 以下。M3F：最終の F が M3 の F 以下。
- 最終のコアから CPU の float64 で Q_λ（M3F は F）を計算し直し、記録と相対 1e-10 で一致する。

### 4.4 途中の診断（停止には使わない）

保存のたび（50 掃引ごと）に、§2.3 の検証の Gram 行列（合計）を使って e（WT-2・C4）を計算し、記録する（`diagnostics`）。サブ層の出力の誤差は途中では計算しない（§9）。

## 5. D：試運転のモードと集計（設計 §6.5・§3.2・§3.3）

- `--mode smoke`：layer 14 の仕事だけを、`max_sweeps=5` で回す（M5 の 14 本、M5_M4r の 12 本。M3F と M5_lambda も各 1 本ずつ動作確認として入れる）。保存は `smoke` のフォルダに分ける。τ_warn は仮に 1e-7。
- 集計のスクリプト `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_smoke_summary.py` を作る。出力は `results/.../04_main_experiment/smoke_summary.json`：
  1. **時間の見積もり**：仕事ごとに、掃引 2〜5 の時間の中央値。
     - 期待値：M5（M3 から）・M3F・M5_lambda は 400 掃引、M5_M4r は 1,000 掃引で止まると仮定する。
     - 保守的な値：すべて 1,000 掃引。
     - 層 7・21 は、同じ射影・予算の layer 14 の時間を使って見積もる。
     - 2 本同時の効果は、Claude の計測の 1.27 倍（設計 §6.2a）で割る。
     - 評価の時間（設計 §7.9 の NLL 約 4 時間など）は定数として足す。
     - 期待値が 48 時間を超えるかを出す。
  2. **解き直しの割合**：入力のグループごとに、2 段目以上に進んだ更新の数 ÷ 局所の更新の数。10% を超えるグループに印をつける（設計 §3.3）。
  3. **τ_warn の事前の確認の 2 つ目**：3 段目まで進んだ更新すべての 3 段目の残差。すべて 1e-7 以下なら「τ_warn = 1e-7 の条件を満たす」、1 つでも超えれば「1e-8」と出す（設計 §3.2。決定は Claude）。
  4. GPU の使用率の平均と、取り合いの有無。

## 6. E：テスト

- 仕事の一覧の作り方（本数・順番・id・初期値・λ）
- ロックによる仕事の取り合い（2 つのプロセスが同じ仕事を取らない、残ったロックの片付け）。小さい問題でよい
- 中断と再開（小さい問題で、途中で止めて起動し直すと続きから回り、結果が中断なしと一致する）
- 方式の作成（小さい行列で、M2 の検算、パラメータ数、飽和の分類）
- 検証データの記事の切り方
- 試運転の集計（作った記録から、見積もり・割合・τ_warn の判定が正しく出る）
- 既存のテスト（`tests/test_metric_tt_solver.py`・`tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py`・`tests/test_qwen3_metric_pilot_checks.py`・`tests/test_metric_tt_study.py`・`tests/test_accumulate_input_grams.py`）と新しいテストを、最後にまとめて 1 回流して件数を報告する。

## 7. 数分で終わる確認（Cursor が行う）と、Claude が回すもの

- **Cursor が行う**：
  - データの準備を、**較正 2 系列・検証 2 単位だけ**で動かす（`--limit` のような引数）。12 個のモジュールの Gram 行列ができ、q・k・v の入力が一致すること。
  - 方式の作成を **1 セルだけ**（layer 14 の k・25%）で動かし、検算が通ること。
  - 仕事の実行を、その 1 セルの M5 を `max_sweeps=2` で、`--workers 2` で動かす（もう 1 つのプロセスは仕事がなくて終わる）。
  - どれも数分で終わる見込み。これより長いものは回さない。
- **Claude が回す**：データの準備の本番（全 3 層・全データ）、方式の作成の本番（42 セル）、試運転、本実行。

## 8. 報告してほしいこと

1. 作ったスクリプト・関数と、その使い方（Claude が本番を回すのに必要なコマンド）
2. テストの結果（まとめて 1 回流した件数）
3. §7 の確認の結果
4. 依頼と違うことをした点と、その理由
5. 気づいた不具合や、設計の曖昧な点（直さずに）

## 9. 設計からの修正（Claude。依頼文の作成時。設計にも反映済み）

- 設計 §4.5 では、保存のたびに検証データでの e とサブ層の出力の誤差を記録するとしていた。サブ層の出力の誤差は、モデル全体を流す必要があり重いので、途中では計算せず、最終の状態についてだけ次の依頼 03 で計算する。途中の診断は、検証の Gram 行列（合計）から計算できる e だけにする。

## 10. やらないこと

- データの準備・方式の作成の本番、試運転、本実行（Claude が回す）
- 評価（単位ごとの誤差、ブートストラップ、H_c、全体の結論、サブ層の出力、NLL）と Pythia（次の依頼）
- ソルバーの計算、停止の規則、目標の残差、τ_warn の値、予算・tensorization・対象の変更
- 予備実験・ベンチマーク・03 のファイルへの書き込み
