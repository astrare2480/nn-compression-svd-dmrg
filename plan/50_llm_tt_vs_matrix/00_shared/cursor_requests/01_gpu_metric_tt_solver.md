# Cursor への依頼：計量つき TT の 1 サイト ALS を GPU で動かす版（実装のみ）

作成：2026-11-08　依頼者：レビュー担当（Claude）

## 0. 背景と範囲

- `src/nn_compression/tt_mps/metric_tt.py` の 1 サイト ALS（M5）は、CPU の float64 だけで動く。
  Qwen3 の 1 層の予備実験では、1 本に 17〜93 分かかり、どの本も 50 掃引の上限で止まった（収束の基準 1e-6 には未達）。
- 次の実験では、層・射影・予算が増える。今の作りのままだと数日かかる。
- **今回は実装とテストだけ**を行う。ベンチマークの Notebook は用意するが、**実行しない**。
  初期値（精度・前処理・許容誤差の決め方）は、後でベンチマークを見て決める。
  そのため、選択肢はすべて引数で切り替えられるようにし、どれかを決め打ちしないこと。
- **予備実験の本実行が、いま CPU で動いている**（Cursor の Notebook）。次のものには触らないこと。
  - `src/nn_compression/tt_mps/metric_tt.py`（読むだけ。import して再利用するのはよい）
  - `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot.ipynb` と同じフォルダの補助スクリプト
  - `results/`・`models/`・`data/` の `01_single_layer_pilot/` の中身（読むのもベンチマークの Notebook を実行するときだけ）
  - `plan/.../01_single_layer_pilot/design.md`

## 1. 問題の形（今の実装と同じ）

- 重み `W ∈ ℝ^{m×n}`、コア `G^{(k)} ∈ ℝ^{r_{k-1}×m_k×n_k×r_k}`、計量 `G_λ ∈ ℝ^{n×n}`（対称正定値）。
- 目的 `Q_λ = tr(ΔW G_λ ΔWᵀ)`。中心 `k` の局所問題は `H x = b`。
  - `H = Pᵀ(G_λ ⊗ I)P`、`b = Pᵀ vec(W G_λ)`。
- 局所ヘッセ行列は、出力 mode について単位行列になる：`H = K_k ⊗ I_{m_k}`。
  - `K_k[(a,j,b),(a',j',b')] = Σ A[a,nL; a',nL'] · G_λ[(nL,j,nR),(nL',j',nR')] · B[b,nR; b',nR']`
  - `A[a,nL; a',nL'] = Σ_{mL} L[a,mL,nL] L[a',mL,nL']`（左のコア群の、出力 mode についての Gram）。`B` は右側について同様。
  - `K_k` の次元は `r_{k-1}·n_k·r_k`。予備実験の中央のサイトでは 5 万〜20 万なので、`K_k` 自体は作らない。
- 実測の参考値：`cond(G_λ) ≈ 4.1e8`。CPU の float64 で、HVP 1 回（`3072×1024` の密行列を作って `G_λ` を掛ける）が約 33 ms。

## 2. 作るもの

新しいモジュール `src/nn_compression/tt_mps/metric_tt_solver.py` を作る。`metric_tt.py` の関数は、CPU・float64 に限らないもの（`_contract_left_group` など）を import して使ってよい。CPU に限る検査がある関数は、device と dtype を問わない版をこのモジュールに作る（コピーしてよい）。

### 2.1 設定のデータクラス

```python
@dataclass(frozen=True)
class MetricTTSolverConfig:
    device: str | torch.device = "cuda"
    precision: Literal["float64", "float32_refine"] = "float64"
    preconditioner: Literal["jacobi", "kronecker"] = "jacobi"
    cg_tolerance_mode: Literal["fixed", "adaptive"] = "fixed"
    cg_relative_tolerance: float = 1e-8       # fixed のときの値。adaptive のときは下限
    cg_tolerance_max: float = 1e-2            # adaptive の上限
    adaptive_eta: float = 0.1                 # adaptive の係数（2.4 節）
    max_cg_iterations: int = 500
    max_refinements: int = 5                  # float32_refine の外側の反復回数の上限
    sweep_tolerance: float = 1e-6
    max_sweeps: int = 50
    gradient_tolerance: float | None = None   # 相対勾配ノルムで止める（None なら使わない）
    time_limit_seconds: float | None = None   # 掃引の切れ目で打ち切る（None なら使わない）
```

- 既定値は、今の CPU 版と同じ設定（float64・Jacobi・1e-8 で固定・500 回・1e-6・50 掃引）にする。
  device だけが違う。後でベンチマークを見て、実験ごとに引数で変える。
- `torch.backends.cuda.matmul.allow_tf32` と `torch.backends.cudnn.allow_tf32` を、計算の間だけ False にし、終わったら元に戻す（context manager）。

### 2.2 精度

- `float64`：計算はすべて device 上の float64。
- `float32_refine`：HVP と CG を float32 で行い、外側で float64 の反復改良を行う。
  1. float64 で `r = b − H x` を計算する。
  2. float32 の PCG で `H d = r` を解く。許容誤差は内側用（2.4 節で決まる値と 1e-4 の大きい方）。
  3. `x ← x + d`（float64）。float64 の相対残差が目標以下になるか、`max_refinements` 回に達したら止める。
- どちらの場合も、`Q_λ` の計算、更新を受理するかの判定、正準化（QR）は float64 で行う。
- `G_λ` は、device 上に float64 版と（必要なら）float32 版の両方を 1 回だけ置いて使い回す。
  `W G_λ` も 1 回だけ計算して使い回す。

### 2.3 前処理

- `jacobi`：今の `metric_tt_jacobi_diagonal` と同じ定義。device と dtype を問わない版にする。
- `kronecker`：`G_λ ≈ G_L ⊗ G_j ⊗ G_R` という Kronecker 積の近似から作る前処理。
  - 入力 mode の分け方（`nL`, `n_k`, `nR`）はサイトごとに違うので、近似はサイトごとに作る。同じ実行の中ではキャッシュする。
  - 近似は Van Loan–Pitsianis の方法を 2 段階で使う（`G ≈ G_L ⊗ G_rest`、そのあと `G_rest ≈ G_j ⊗ G_R`）。最大の特異値の組は、べき乗法か `torch.linalg.svd` で求める。各因子は対称化し、符号をそろえて正定値にする。最小固有値は、最大固有値の 1e-12 倍で下を切る。
  - `K_L[a,a'] = Σ A[a,nL;a',nL'] G_L[nL,nL']`、`K_R[b,b'] = Σ B[b,nR;b',nR'] G_R[nR,nR']` とし、`M = K_L ⊗ G_j ⊗ K_R` を前処理に使う。
  - `M⁻¹` は、3 つの小さい行列（`r_{k-1}`、`n_k`、`r_k` 次）の固有値分解から適用する。出力 mode `m_k` 方向には同じものを掛ける。
  - 近似の当てはまり（相対 Frobenius 誤差 `‖G − G_L⊗G_j⊗G_R‖/‖G‖`）を、サイトごとに記録する。

### 2.4 CG の許容誤差

- `fixed`：すべての局所問題で `cg_relative_tolerance` を使う（今と同じ）。
- `adaptive`：サイト `k` を解く前に、初期残差の相対値 `g_k = ‖b − H x_0‖ / ‖b‖` を計算する。これは局所勾配と同じ量で、追加の費用はない。
  許容誤差は `clip(adaptive_eta · g_k, cg_relative_tolerance, cg_tolerance_max)` とする。
- どちらの場合も、`g_k` はサイトごとに記録する（2.6 節）。

### 2.5 掃引と停止

- 掃引の順番、正準中心の動かし方、受理・棄却の規則は、`run_metric_tt_als` と同じにする（1 掃引で `2d−1` 回の更新）。
- 停止の条件は次のどれか（最初に満たしたもの）。止まった理由を記録する。
  - `sweep_tolerance`：掃引ごとの `Q_λ` の相対減少がこれ未満
  - `gradient_tolerance`（None でないとき）：その掃引の `max_k g_k` がこれ未満
  - `max_sweeps`
  - `time_limit_seconds`（None でないとき）：掃引の切れ目で経過時間を見る
- 公開する関数：

```python
def run_metric_tt_als_on_device(
    cores, weight, gram, out_modes, in_modes, config: MetricTTSolverConfig,
    *, progress: Callable[[DeviceSweepRecord], None] | None = None,
) -> tuple[list[torch.Tensor], tuple[DeviceSweepRecord, ...]]
```

- 入力は CPU の float64 でも device 上でもよい。入力は変更しない。
- 戻り値のコアは **CPU の float64** にする（今の保存形式と比較のコードをそのまま使うため）。
- `progress` は掃引ごとに呼ぶ。Notebook から途中経過を出すのに使う。

### 2.6 記録（`DeviceSweepRecord`）

`SweepRecord` の項目（sweep、q_lambda、relative_decrease、cg_iterations、cg_residuals、n_rejected、seconds）に、次を加える。

- `local_gradient_norms`：サイト更新ごとの `g_k`（tuple）
- `cg_tolerances`：実際に使った CG の許容誤差（tuple）
- `refinements`：`float32_refine` の外側の反復回数（tuple。float64 のときはすべて 0）
- `seconds_hvp`、`seconds_preconditioner`、`seconds_qr`、`seconds_accept`：掃引の中の時間の内訳（device で同期してから測る）
- `peak_memory_bytes`：CUDA のときは `torch.cuda.max_memory_allocated`。CPU のときは None
- `stop_reason`：最後の掃引だけに入れる。それ以外は None

## 3. テスト（`tests/test_metric_tt_solver.py`）

- 小さい問題だけを使う（例：`m=n=64` 前後、3〜4 コア、rank 4〜8、seed 固定）。
  CUDA がないときは、CUDA が必要なテストを skip する。CPU でも動くテストは CPU で回す。
- 確かめること：
  1. device と dtype を問わない HVP・右辺・Jacobi 対角が、`metric_tt.py` の CPU 版と一致する（float64 で相対 1e-12）。
  2. `H = K ⊗ I_{m_k}` が成り立つ。小さい問題で `K` を明示的に作り、HVP と一致することを確かめる。
  3. Kronecker 前処理：`G` がちょうど Kronecker 積のときは、`M = K` になる（相対 1e-10）。一般の `G` でも、`M` は対称正定値になる。
  4. 既定の設定（float64・Jacobi・fixed）で device 上の ALS を 3 掃引回したとき、`Q_λ` の推移が CPU 版の `run_metric_tt_als` と一致する（相対 1e-9）。
  5. `float32_refine` で解いた局所問題の float64 の相対残差が、目標の 1e-8 以下になる。条件数が `1e6` 程度になるよう作った `G` で確かめる。
  6. `adaptive` のとき、使った許容誤差が上限と下限の間に入る。そのうえで `Q_λ` が単調に減る（棄却を含む）。
  7. どの設定でも、戻り値が CPU の float64 で、入力のコアが変更されていない。
  8. 停止の理由が正しく記録される（`max_sweeps` と、`gradient_tolerance`・`time_limit_seconds` のそれぞれ）。
  9. TF32 の設定が、実行後に元に戻っている。
- 実行するのは `pytest tests/test_metric_tt_solver.py` だけにすること。予備実験が CPU を使っているので、**テスト全体は今は回さない**。

## 4. ベンチマークの Notebook（作るだけで、実行しない）

`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/02_metric_tt_solver_benchmark.ipynb`

- 予備実験の保存物を読むだけにする。書き込みはしない。
  - `data/.../01_single_layer_pilot/weights_float64.pt`、`calibration_grams.pt`
  - `models/.../01_single_layer_pilot/` の M3 と M5 のコア
- 出力先：`results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/02_metric_tt_solver_benchmark/`
- 測ること：
  1. **HVP 1 回の時間**。T1・T2 の各サイトについて、CPU の float64、CUDA の float64、CUDA の float32（TF32 なし）の 3 通り。ウォームアップの後、20 回の中央値をとる。
  2. **局所問題の収束**。T1・T2 の中央のサイト（M3 のコアから始め、正準中心をそのサイトに移す）について、`jacobi` と `kronecker` のそれぞれで、相対残差が 1e-2, 1e-4, 1e-6, 1e-8 に届くまでの CG の反復回数と時間を測る。`float64` と `float32_refine` の両方で測る。
  3. **短い ALS**。T1・50% と T2・50% について、M3 から 5 掃引だけ回す。設定ごとに `Q_λ` の推移と時間を比べる。設定は少なくとも次の 3 つ。
     - 今と同じ設定（CPU 版の関数）
     - 既定の設定を device 上で回したもの
     - `float32_refine`・`kronecker`・`adaptive`
- 結果は CSV と、Notebook 内の表で出す。実行時の GPU 名、torch と CUDA のバージョンも記録する。

## 5. ドキュメント

- 新しく公開する関数とデータクラスを、`docs/90_src設計/05_Core_API_v1/Internal_API.md` に、既存の書き方に合わせて追記する。docs-sync のテストが通る形にする。
- この Markdown の中で、`metric_tt.py`（CPU の参照実装）と `metric_tt_solver.py`（device を選べる版）の役割の違いを 2〜3 行で書く。

## 6. やらないこと

- `metric_tt.py`、予備実験の Notebook・補助スクリプト・結果・設計書を変更しない。
- ベンチマークの Notebook を実行しない。テスト全体を回さない。
- 2 サイト更新（2 つのコアをまとめて解いてから SVD で切る）は入れない。研究の問いそのものに関わるので、別途設計する。
- `G_λ` を低ランク近似して計算を減らす方法は入れない（目的関数が変わるため）。
- 複数の設定を GPU で同時に回す仕組みは入れない（ベンチマークの後で検討する）。
- コミットしない。

## 7. 報告してほしいこと

1. 作った・変えたファイルの一覧（新しいファイルと、Internal_API.md の差分の行数）
2. 公開した関数とデータクラスのシグネチャ
3. `pytest tests/test_metric_tt_solver.py` の結果（skip したテストがあれば理由も）
4. Kronecker 前処理で、正定値にするためにクリップが起きた条件があれば、その内容
5. 仕様どおりにできなかった点や、判断に迷って自分で決めた点
