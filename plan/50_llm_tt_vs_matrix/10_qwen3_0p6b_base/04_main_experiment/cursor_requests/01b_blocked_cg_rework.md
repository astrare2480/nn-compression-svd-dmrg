# Cursor への依頼 01b：区切りの CG（`cg_check_every > 1`）の作り直し

作成：2026-11-08　依頼者：レビュー担当（Claude）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md`（以下「設計」）
前の依頼：`01_solver_gpu_and_escalation.md`（以下「依頼 01」）

## 0. 背景と範囲

- 依頼 01 の実装は、設計どおりで、単体テスト 87 件も通った（Claude が確認）。ただし、**区切りの CG（`cg_check_every=10`）が、元の CG（`cg_check_every=1`）より遅い**。
- Claude の計測（GPU を実験以外で使っていない状態。up_proj・T2・50%、03 の `long_als_T2_50_M3.pt` の最終状態を中心 1 に移したコア、サイト 1 の局所問題を 1 回解く。CG 1,000 回、外側の反復 2 回、3 回ずつ）：

| 方式 | CG 1 回あたり |
|---|---|
| `cg_check_every=1`（今のまま） | 1.0〜1.2 ms |
| `cg_check_every=1`、`_Accumulator` の同期を外した場合 | 約 0.92 ms |
| `cg_check_every=10`（同期の有無によらず） | 約 1.7 ms |
| HVP 1 回だけ（float32） | GPU 0.49 ms、CPU で発行する時間 0.28 ms |

- Claude の見立て（コードを読んだことと上の数値から。プロファイルでは確かめていない）：`_conjugate_gradient_blocked` は、反復ごとに全要素の `torch.isfinite(...).all()` を 3 回、全要素の `torch.where` を 3 回、0 次元の小さい演算をたくさん足していて、1 回あたりのカーネルの発行が多すぎる。そのため、GPU より先に CPU の発行が追いつかなくなっている（Windows ではカーネルの発行が比較的重い）。
- 今回お願いすること：区切りの CG を作り直し、CG 1 回あたり **0.6 ms 以下**（同じ条件。HVP の 0.49 ms に近づける）を目指す。あわせて、レビューで見つけた 2 点を直す（§2 の 3・4）。
- **数分で終わる確認まで**にする（§4）。同等性の確認（依頼 01 の F）と τ_warn の事前の確認（依頼 01 の G）は、Claude が回す（§6）。

## 1. 変えないもの

- `cg_check_every=1` の計算（ビット単位で今と同じ）。
- `run_metric_tt_als_until_converged`・`run_metric_tt_als_on_device`・`solve_metric_tt_site_on_device` の既定の動作。
- `run_metric_tt_als_study` の規則（3 段の解き直し、区分、停止の規則）。CG の状態の名前を 1 つ足す場合（§2 の 3）は、区分の決め方を §2 の 3 のとおりにする。

## 2. 作り直し

### 2.1 区切りの CG の反復を軽くする

- 反復の中では、HVP、前処理、内積、ベクトルの更新以外の**全要素の演算をしない**。全要素の `isfinite(...).all()` と、全要素の `torch.where` を反復の中から外す。
- 曲率が 0 以下・値が有限でない場合の扱いは、**0 次元のスカラーだけ**で行う。たとえば次のようにする（同じ働きなら別の書き方でよい）：
  - 0 次元の真偽値 `active`（まだ正常に進んでいるか）を GPU の上に持つ。
  - `ok = active & (curvature > 0)`（curvature が NaN なら比較は偽になる）。
  - `alpha = torch.where(ok, residual_dot / curvature, 0)`（0 次元）。x と残差の更新は `x = x + alpha * direction` のまま。alpha が 0 なら x は変わらない。
  - `active = ok` のように、一度止まったらその区切りの残りでは進まないようにする。β と方向の更新も同じ考え方で、0 次元のスカラーで止める。
  - 印（曲率 ≤ 0、値が有限でない、内積が 0）は 0 次元のテンソルにため、区切りの終わりにまとめて CPU に取り出す。取り出す回数は区切りごとに 1 回にする（`torch.stack` などでまとめる）。
- 値が有限かどうかは、区切りの終わりの相対残差（すでに取り出している値）と、スカラーの印で判断する。全要素を確かめる必要があれば、区切りの終わりに 1 回だけにする。
- 印が立ったあとの反復は、区切りの終わりまで HVP を回してよい（そこで止めるには CPU との同期が要るため）。ただし、返す反復回数には**実際に x を進めた回数**を入れ、空回りの回数は別に返す（記録用）。

### 2.2 時間を測る仕組みの同期

- `_Accumulator` は、`__enter__`・`__exit__` のたびに `torch.cuda.synchronize` を呼んでいる。HVP と前処理を 1 回呼ぶごとに 2 回ずつ同期する。
- `MetricTTSolverConfig` に `timing_sync: bool = True` を足す。`True` は今と同じ。`False` のときは、`_Accumulator` が同期せず、CPU から見た時間だけを足す（GPU の時間ではないことを docstring に書く）。掃引全体の時間（`seconds`）の測り方は変えない。
- 本実験では `timing_sync=False` を使う。

### 2.3 内積が 0 の場合を、値が有限でない場合と分ける

- 今は、内積（`residual_dot`）が 0 になったことも `nonfinite` の印に入れている。内積が 0 になるのは、前処理つきの残差が 0 になった（ちょうど解けた）場合で、値が有限でないこととは違う。設計では `nonfinite` は「解けなかった」として本を止めるので、誤って止めるおそれがある。
- 内積が 0 の場合は、新しい状態 `breakdown` にする。ただし区切りの終わりの残差が許容誤差以下なら `converged` にする（今と同じ）。
- `run_metric_tt_als_study` では、`breakdown` は float32 の内側の CG の曲率 ≤ 0 と同じく、**記録だけにする**（それだけでは「解けなかった」にしない）。float64 の CG の `breakdown` も、記録だけにする（正確に解けた場合にしか起きないため）。

### 2.4 そのほか

- CUDA Graph はまだ使わない。上の作り直しで 0.6 ms に届かなかったときの次の手として、報告で見込みを書いてくれればよい。
- `record_function` は残してよい（プロファイラーが動いていないときの負担は小さい）。ただし計測で負担が目立つなら報告する。

## 3. テスト

- `cg_check_every=1` がビット単位で今と同じ（今のテストを流用）。
- `cg_check_every=10` で、同じ許容誤差に届く。
- 曲率 ≤ 0、値が有限でない、内積が 0 の各場合に、状態と、x が印のあとに進んでいないことを確かめる（小さい問題で、HVP や前処理を細工してよい）。
- `timing_sync=False` で、計算結果が `True` のときとビット単位で同じ。
- `run_metric_tt_als_study` で、`breakdown` が「解けなかった」にならない。
- 既存のテスト（`tests/test_metric_tt_solver.py`・`tests/test_metric_tt_convergence.py`・`tests/test_als_convergence.py`・`tests/test_qwen3_metric_pilot_checks.py`・`tests/test_metric_tt_study.py`・`tests/test_accumulate_input_grams.py`）がすべて通る。最後に**まとめて 1 回**流して、件数を報告する。

## 4. 数分で終わる確認（Cursor が行う）

- 計測用の小さいスクリプト `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/_bench_cg_site1.py` を作る。§0 の条件（up_proj・T2・50%、03 の `long_als_T2_50_M3.pt` の `cores` を `canonicalize_tt_matrix_cores(..., 1)` で中心 1 に移す、`solve_metric_tt_site_on_device(..., site=1)`、`precision="float32_refine"`、Jacobi、許容誤差 1e-8、`max_cg_iterations=500`、`max_refinements=5`）で、次の 4 通りを、温め 1 回のあと 3 回ずつ測る：
  - `cg_check_every=1`、`timing_sync=True`
  - `cg_check_every=1`、`timing_sync=False`
  - `cg_check_every=10`、`timing_sync=True`
  - `cg_check_every=10`、`timing_sync=False`
- 記録する：時間、CG の回数、外側の反復の回数、最終の残差、CG 1 回あたりの時間。結果は `results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/solver_dev/bench_cg_site1.json` に保存する。
- 計測の前に `nvidia-smi` で GPU の使用率を記録し、ほかのプロセスが GPU を使っていたら書く。
- 1 分ほどで終わる見込み。これより長い計算は回さない。

## 5. 報告してほしいこと

1. 変えた関数と引数
2. テストの結果（まとめて 1 回流した件数）
3. §4 の計測の結果（4 通りの CG 1 回あたりの時間と、最終の残差）。0.6 ms に届いたか。届かなければ、残りの時間が何に使われているかの見込みと、次の手（CUDA Graph など）の見込み
4. 依頼と違うことをした点と、その理由
5. 気づいた不具合や、設計の曖昧な点（直さずに）

## 6. やらないこと（Claude が回す）

- 依頼 01 の F（同等性の確認：up・down の 50 掃引）と G（τ_warn の事前の確認）
- 長い計測、プロファイル、本実験の Notebook や試運転
- CUDA Graph、batched CG、複数のプロセス、TF32・bf16、局所の許容誤差を緩める方法
- 既存の関数の既定の動作の変更
- 予備実験・ベンチマーク・03 のファイルへの書き込み
