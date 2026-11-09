# Cursor への依頼：GPU 版ソルバーのレビュー指摘の修正

作成：2026-11-08　依頼者：レビュー担当（Claude）
前回の依頼：`01_gpu_metric_tt_solver.md`

> **実施の記録（2026-11-08）**：この依頼は Cursor には渡さず、ユーザーの指示で Claude が実装し、ベンチマークの Notebook も実行した（§7 の「Notebook を実行しない」はユーザーの指示で解除）。コミットはしていない。
> - 1〜6 は依頼のとおりに実施した。`tests/test_metric_tt_solver.py` は 18 件すべて成功（7 件を追加）。T1 の site 2 の形での `_environment_factor` の追加メモリは 4.3MB（修正前は約 71GB の確保で停止）。
> - 依頼と違う点：Kronecker の固有値分解は `_kronecker_eigensystem` に分け、呼び出し回数のテストはこの関数を数えた。修正前後の一致は、テストの中で修正前の式に差し替えて ALS の `Q_λ` の推移を比べた（相対 1e-10 で一致）。`Internal_API.md` にあった存在しない名前 `output_error_ratios` は、2 つの関数に置き換えた。
> - docs-sync のテスト 2 件とテスト全体の 5 件の失敗・3 件のエラーは、今回触っていないファイル（`utils`・`metrics` などの以前の関数のドキュメント漏れ、Notebook 06 の補助関数、パッケージの公開範囲）のもので、変更前から起きていた。
> - ベンチマークの結果は `design.md`（`01_single_layer_pilot`）の §12.4 の 2・3 に記録した。

## 0. 背景と範囲

- 前回の実装（`src/nn_compression/tt_mps/metric_tt_solver.py`）をレビューした。
  - HVP・右辺・Jacobi 対角・Kronecker 近似・反復改良・許容誤差の決め方・停止・TF32 の扱いは、仕様どおりで正しい。
  - 前回の報告で、自分で決めた点（関数の公開、`float32_matmul_precision` の保存と復元、停止理由の優先順）もこのままでよい。
- 直すのは、下の 1〜2（ソルバー）、3〜4（ベンチマークの Notebook）、5（ドキュメント）。
- **予備実験の本実行は終わった**。前回の「Notebook を実行しない」「テスト全体を回さない」という制限はなくなった。
  ただし、今回もベンチマークの Notebook は**実行しない**（修正をレビューしてから実行する）。
- `metric_tt.py`、予備実験（`01_single_layer_pilot`）の Notebook・結果・モデル・データ・設計書は、引き続き変更しない。

## 1. 【不具合】Kronecker 前処理の環境の縮約がメモリ不足になる

- `_environment_factor` は `einsum("amn,bmp->anbp")` で、shape `(r, n, r, n)` の中間の配列を作っている。
  T1・50% の site 2 では、左のコア群が `r = 382`、`n_left = 256`、`m_left = 4` なので、中間の配列は約 9.6e9 要素（float64 で約 77GB）になる。
  ベンチマークの §2（T1 の site 2）と §3（全サイトの更新）は、ここで止まる。
- 次の順番で計算すること（大きい中間の配列を作らない）。

```python
weighted = torch.einsum("amn,np->amp", group, metric_factor)   # (r, m, n)
return torch.einsum("amp,bmp->ab", weighted, group)            # (r, r)
```

- `kronecker_preconditioner_matrix` も同じ関数を使っているので、同時に直る。
- テストを 1 件追加する：小さい問題で、新しい計算が古い式（`(r, n, r, n)` を作る式。テストの中に参照用として書く）と相対 1e-12 で一致すること。

## 2. 【効率】前処理を CG の反復ごとに作り直している

- `_solve_site` の `precondition64` / `precondition32` は、呼ばれるたびに次を計算し直している。
  - Jacobi：`metric_tt_jacobi_diagonal_on_device`（CPU 版は局所問題ごとに 1 回だけ）
  - Kronecker：左右のコア群の縮約、`K_L`・`K_R`、3 つの小さい行列の `eigh`
- **局所問題ごとに 1 回だけ作り、その局所問題の中で使い回す**形にする。
  - Jacobi：対角（`_safe_jacobi` を通したもの）を、解き始める前に 1 回計算する。
  - Kronecker：`K_L`・`G_j`・`K_R` の固有値分解（固有値と固有ベクトル）を、解き始める前に 1 回計算する。`_apply_kronecker_inverse` は、分解済みのものを受け取って掛けるだけにする。
  - `float32_refine`：float32 版の前処理（対角、または固有値分解を float32 にしたもの）も、局所問題ごとに 1 回だけ作り、外側の反復の間で使い回す。コアの中心は外側の反復で変わるが、前処理が依存するのは中心以外のコアと `G` だけなので、作り直す必要はない。
- 前処理を作る時間は `seconds_preconditioner` に入れ、掛ける時間は別の項目 `seconds_preconditioner_apply` として `DeviceSweepRecord` に足す。`LocalSolveRecord` にも `seconds_preconditioner_setup` と `seconds_preconditioner_apply` を足す。
- 数値の結果は変わらないはずなので、既存のテストがそのまま通ることを確かめる。

## 3. ベンチマーク §2：測るサイトを変える

- 今は `n_cores // 2`（T1 は site 2、T2 は site 1）だけを測っている。
- 予備実験で CG の反復が多かったのは T1 の **site 1**（M3 から始めた最初の掃引で 406 回・362 回。site 2 は 240 回）。
- **両端以外のすべてのサイト**を測る：T1 は site 1 と 2、T2 は site 1。
  - 正準中心は、site 0 で正準化してから、そのサイトまで移す（今と同じ方法）。
  - 行数の確認：3 サイト × 4 設定 × 4 閾値 = 48 行。
- 各行に、局所問題の大きさ `K_k` の次元（`r_{k-1}·n_k·r_k`）と、前処理を作る時間・掛ける時間を足す。

## 4. ベンチマーク §3：設定を 1 つずつ変える

- 今は「CPU 版」「CUDA の既定」「float32_refine＋Kronecker＋adaptive」の 3 つで、最後のものは 3 つを同時に変えているため、どれが効いたか分からない。
- 次の 6 つにする（どれも 5 掃引、`sweep_tolerance=-1`、T1・50% と T2・50% の M3 から開始）。

| 名前 | 計算機 | 精度 | 前処理 | 許容誤差 |
|---|---|---|---|---|
| `cpu_reference` | CPU（`run_metric_tt_als`） | float64 | Jacobi | fixed 1e-8 |
| `cuda_default` | CUDA | float64 | Jacobi | fixed 1e-8 |
| `cuda_kronecker` | CUDA | float64 | Kronecker | fixed 1e-8 |
| `cuda_adaptive` | CUDA | float64 | Jacobi | adaptive |
| `cuda_float32` | CUDA | float32_refine | Jacobi | fixed 1e-8 |
| `cuda_all` | CUDA | float32_refine | Kronecker | adaptive |

- adaptive の係数は既定値（`adaptive_eta=0.1`、上限 1e-2、下限 1e-8）のままにする。
- 行数の確認：2 条件 × 6 設定 × 5 掃引 = 60 行。
- `als_trajectory.csv` に、掃引ごとの `seconds_hvp`、`seconds_preconditioner`、`seconds_preconditioner_apply`、`seconds_qr`、`seconds_accept`、`peak_memory_bytes`、CG の反復数の合計、`max(local_gradient_norms)` を足す（CPU 版は空欄でよい）。
- §4 の表に、設定ごとの「5 掃引後の `Q_λ`」「`cpu_reference` との `Q_λ` の相対差」「合計時間」「`cpu_reference` に対する速さの比」を出す。
- 冒頭の注意書き「予備実験の本実行中は実行しない」は、「予備実験は 2026-11-08 に終わった。`01_single_layer_pilot` のファイルは読むだけにする」に書き換える。

## 5. ドキュメント

- `docs/90_src設計/05_Core_API_v1/Internal_API.md` に、`output_error_ratios_from_gram` と `output_error_ratios_from_activations`（`compression/metric_low_rank.py`）を、既存の書き方に合わせて足す。
- 2 で増えた項目を、ソルバーの節に反映する。
- docs-sync のテストが通ることを確かめる。

## 6. テスト

- 実行するもの：
  - `pytest tests/test_metric_tt_solver.py`
  - docs-sync のテスト
  - テスト全体（`pytest`）。前回は止めていたが、今回は回してよい。
- 1 の追加テストに加えて、次を足す。
  - 前処理の使い回し：1 回の局所問題の中で、Jacobi 対角の計算と Kronecker の固有値分解がそれぞれ 1 回しか呼ばれないこと（`unittest.mock` で呼び出し回数を数える、などの方法でよい）。
  - 実際の大きさに近い形でメモリ不足にならないこと：T1 の site 2 と同じ形（左のコア群 `r=382`、`m_left=4`、`n_left=256`）の乱数のコア群で `_environment_factor` を呼び、CUDA があれば `torch.cuda.max_memory_allocated` が 1GB 未満であること。CUDA がなければ CPU で、結果の shape だけを確かめる。

## 7. やらないこと

- ベンチマークの Notebook を実行しない。
- `metric_tt.py`、予備実験のファイル、`design.md` を変更しない。
- 2 サイト更新、`G_λ` の低ランク近似、複数の設定の同時実行は入れない（前回と同じ）。
- コミットしない。

## 8. 報告してほしいこと

1. 変更したファイルと、それぞれの変更の概要
2. 1 の修正後、T1 の site 2 の形で `_environment_factor` が使うメモリ（測った値）
3. 2 の修正の前後で、小さい問題の結果（`Q_λ` の推移）が一致したか
4. `pytest tests/test_metric_tt_solver.py`、docs-sync のテスト、テスト全体の結果（失敗やスキップがあれば理由も）
5. 仕様どおりにできなかった点や、判断に迷って自分で決めた点
