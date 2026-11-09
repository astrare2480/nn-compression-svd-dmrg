# Cursor への実装依頼：Qwen3-0.6B-Base 1層の予備実験（データを考慮した TT 対 行列低ランク）

- 作成：2026年11月7日（Claude）
- 設計の正本：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/design.md`（作成時のパスは plan/gpt2_tt_vs_matrix_pilot_design.md）。**この依頼文と設計が食い違う場合は、実装せずに質問すること。**
- 今回の範囲：src の共通関数とテスト、Notebook の実装、検算、計算時間の事前計測まで。**本実行（§4 の全条件）は、事前計測の結果をユーザーが確認してから行う。**

---

## 0. 守ること

- リポジトリのルール（`.cursor/rules/`）に従う。特に次の2つ。
  - 既存の `src/nn_compression/` の関数を、Notebook の中で再実装しない。
  - 既存の Notebook を上書きしない。
- **再利用できる処理は `src/nn_compression/` に置く。** Notebook には、実験の判断（条件の組み合わせ、どの結果を採用するかの規則、ファイル名、表、図）だけを書く。
- 設計で事前に固定した値（λ = 1e-6、CG と掃引の停止条件、しきい値、seed）を変えない。合わない場合は、止めて報告する。
- 検算（§5）が1つでも合わなければ、そこで止めて報告する。
- WikiText-2 test は読み込まない。
- コミットはしない。

## 1. 環境（準備済み）

- Python：`C:\Users\astra\anaconda3\envs\pytorch-svd\python.exe`（Python 3.12、torch 2.11.0+cu128、transformers 5.22.2、datasets 5.1.0）
- 環境変数：`HF_HOME=D:\dev\nn-compression-svd-dmrg\data\hf_cache`、`HF_HUB_OFFLINE=1`
- 取得した版（revision）と保存先は `data/hf_cache/download_record.json` にある。**モデルとデータは、この revision を指定して読み込む。**
  - モデル：`Qwen/Qwen3-0.6B-Base`（transformers 5 では `from_pretrained(..., dtype=...)`。`torch_dtype` ではない）
  - WikiText：`Salesforce/wikitext`、設定名 `wikitext-103-raw-v1` と `wikitext-2-raw-v1`
  - C4：`allenai/c4` の `en/c4-validation.00000-of-00008.json.gz`（1ファイルだけ。ローカルのパスから読む）
- 確認済みの事実：`model.model.layers[14].mlp.up_proj.weight` は (3072, 1024)、bf16、バイアスなし。WikiText-2 と WikiText-103 の validation・test は同じ文章。
- 計算機：RTX 5070 Ti（16GB）、RAM 64GB、Ryzen 7 9800X3D（8コア）。float64 の計算は CPU で行い、GPU では行わない。

## 2. 置き場所（`get_experiment_dirs` の規約に従う）

- `get_experiment_dirs(project_root, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "01_single_layer_pilot")`
- Notebook：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot.ipynb`
- 結果（git で管理。小さい CSV・JSON だけ）：`results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/`
- 大きいファイル（保存した活性化など）：`data/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/`（git の管理対象外）
- TT のコアなど：`models/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/01_single_layer_pilot/`

## 3. src に置く共通関数

既存の API を先に確認し、使えるものは使う。主なもの：
- `tt_mps/matrix.py`：`dense_to_tt_matrix_cores`、`tt_matrix_to_dense`、`tt_matrix_num_parameters`
- `tt_mps/modes.py`：`ordered_factorizations`、`tt_matrix_shape_imbalance`、`tt_matrix_num_parameters_from_ranks`
- `tt_mps/matrix_two_site.py`：`canonicalize_tt_matrix_cores`、`tt_matrix_canonicality_errors`
- `utils/paths.py`：`get_experiment_dirs`
- コアの形の規約：G^{(k)} ∈ ℝ^{r_{k-1} × m_k × n_k × r_k}（m は出力、n は入力）、W ∈ ℝ^{m × n}。

新しく作るモジュールの案を下に示す。名前や分け方は、既存の構成に合わせて変えてよい。**変えた場合は報告すること。** どの関数にも docstring を付け、入力と出力の形・dtype を明記する。

### 3.1 `src/nn_compression/llm/`（新規パッケージ）

| モジュール | 内容 |
|---|---|
| `hf_loading.py` | `download_record.json` を読み、記録された revision・ローカルのパスでモデルと tokenizer をオフラインで読み込む。dtype と device を引数で指定できるようにする |
| `text_documents.py` | (a) WikiText の行から記事を復元する。記事の始まりは、見出しの行 ` = 題名 = `（`=` が1つ。` = = 小見出し = = ` は記事の始まりではない）。(b) C4 の json.gz から文書を読む。(c) 文書を seed 固定で無作為に並べ替え、条件を満たす文書から系列を作る（§4.2）。(d) 文書の hash（正規化した本文の SHA-256）を計算し、データの組の間で重複を検査する |
| `activation_capture.py` | 指定した `nn.Linear` に forward pre-hook を付けて、実際に渡された入力を取り出す。系列ごと・文書 id つきで返す。hook は必ず外す（context manager にする） |
| `perplexity.py` | 決まった評価方法で perplexity を計算する（§4.5）。指定した層の重みを一時的に差し替える context manager も置く（終わったら元の重みに戻す） |

### 3.2 `src/nn_compression/compression/metric_low_rank.py`（新規）

| 関数 | 内容 |
|---|---|
| Gram 行列 | G = Σ_t x_t x_tᵀ を float64 で計算する（平均は引かない）。文書ごとの G と合計を返せるようにする |
| 正則化 | G_λ = G + λ · tr(G)/n · I |
| 診断 | G と G_λ の固有値（最小・最大）、条件数、effective rank、正則化の項と最小固有値の比、チャネルごとの RMS の最大・中央値・上位の分位点 |
| M1 | 通常の SVD で rank r に切る |
| M2 | 白色化した SVD：W G_λ^{1/2} を SVD で rank r に切り、G_λ^{-1/2} を掛けて戻す。G_λ^{±1/2} は固有値分解で作る。捨てた特異値の二乗和も返す |
| 誤差 | ΔW = W − W̃ に対して、(i) Gram 行列から Q = tr(ΔW G ΔWᵀ)、(ii) 活性化 X から直接 ‖ΔW X‖_F²、(iii) E_gated の分子と分母。すべて文書ごとに返せるようにする（ブートストラップ用） |

E_gated² = ‖SiLU(W_gate X) ⊙ (ΔW X)‖_F² / ‖SiLU(W_gate X) ⊙ (W X)‖_F²（W_gate は圧縮しない元の値）

### 3.3 `src/nn_compression/tt_mps/metric_tt.py`（新規）

| 関数 | 内容 |
|---|---|
| tensorization の選択 | 出力 3072・入力 1024 を、それぞれ `ordered_factorizations` で d 個に分ける。全組み合わせのうち `tt_matrix_shape_imbalance` が最小のものを選ぶ。同点の場合は、(出力の mode, 入力の mode) を辞書順に並べて最初のものにする（規則を docstring に書く） |
| rank の選択 | 一様な上限 r にそろえ、両端の bond は構造上の最大値で頭打ちにする。パラメータ数が予算以下になる最大の r を返す。追加のパラメータ数（M4 の対角スケール 1024 個など）を引数で受け取れるようにする |
| M4 | 対角スケール D = diag(s) で W D を TT-SVD し、D⁻¹ をコアの入力側に吸収した TT と、(TT, D⁻¹) の組の両方を返せるようにする |
| bond の拡張 | ゼロで埋めて bond の次元を広げる（W は変わらない）。M5 を M4 から始めるときに使う（§4.4） |
| HVP | 中心のコア c_k に対して、v ↦ H_k v を計算する。H_k は Q_λ(W̃) = tr((W − W̃) G_λ (W − W̃)ᵀ) の、コア k に関するヘッセ行列。最初は「v から密な ΔW を作り、G_λ を掛けて、環境で縮約して戻す」素直な実装でよい。右辺 b_k も同じ方法で作る |
| Jacobi 前処理 | H_k の対角成分を計算する |
| PCG | 行列を作らない前処理つき共役勾配法。直前の値から開始する。相対残差 1e-8 か 500 反復で止める。反復回数と最終の残差を返す |
| 1サイト ALS | 正準形を保ちながら左右に掃引する。各更新の後に較正データの Q_λ を計算し直す。減っていれば受理し、増えていれば棄却して元のコアに戻す（棄却も記録する）。掃引の停止は、Q_λ の相対減少が 1e-6 未満か、50 回。掃引ごとの履歴（Q_λ、CG の反復数、残差、棄却数、時間）を返す |

- **K_k ⊗ I の構造を使ったブロック CG は、今回は実装しない。** 事前計測で重すぎると分かった場合にだけ、相談のうえで追加する。
- 計算はすべて CPU の float64 で行う。

### 3.4 `src/nn_compression/metrics/bootstrap.py`（新規。既存に同じ機能があれば、それを使う）

- 文書単位の再標本化（1,000回、seed 0）。比の量 e = Σ_d 分子_d / Σ_d 分母_d を、方式ごとに同じ再標本で計算する。差（例：g = e(M3) − e(M2)）の95%区間を返す。

### 3.5 テスト（`tests/` に pytest で追加）

設計 §8 の検算のうち、モデルを使わないものは、小さいランダムな問題でテストにする。
1. 切り詰めなしの TT-SVD が W を再現する（float64、相対誤差 1e-10 以下）
2. M2 の Q_λ が、捨てた白色化特異値の二乗和と一致し、M1 の Q_λ 以下である
3. HVP が、明示的に作ったヘッセ行列（小さい問題で、HVP を基底ベクトルに掛けて作るのではなく、P を明示的に組んで Pᵀ(G_λ ⊗ I)P として作る）と一致する
4. 正準形での局所ヘッセ行列の最小固有値が λ_min(G_λ) 以上である
5. PCG を直前の値から開始すると、局所の Q_λ が反復ごとに増えない
6. 実際のパラメータ数が `tt_matrix_num_parameters_from_ranks` と一致する
7. bond をゼロで広げても W が変わらない
8. WikiText の記事の復元が、見出しの規則どおりになる（小さい人工データで）

## 4. Notebook で行うこと（順番どおり）

各ステップの最初に、入力・出力・完了条件をマークダウンで書く（`learning-notebook-todo-contracts.mdc` に準じる）。

### 4.1 設定
- seed、λ、停止条件、予算、しきい値、パスを1つのセルにまとめ、`settings.json` に保存する。ライブラリのバージョンと、モデル・データの revision も保存する。

### 4.2 データの準備
| 用途 | 作り方 |
|---|---|
| 較正 | WikiText-103 train の記事を seed 0 で並べ替え、1,024 トークン以上の記事から、記事ごとに先頭の 1,024 トークンを1系列だけ取る。128系列（128記事） |
| 較正（感度確認用） | 上の 128系列の先頭 16系列 |
| 同じドメインの検証 | WikiText-2 validation の記事から、同じ規則で 32系列 |
| 別ドメインの検証 | C4 の文書から、同じ規則で 32系列（1,024 トークン以上の文書に限るので、長い文書に偏る。このことを記録する） |

- トークン化では特殊トークンを付けない（`add_special_tokens=False`）。設定を記録する。
- 記録すること：系列数、有効トークン数、文書数、条件を満たした文書の数、除外したトークン数。
- 較正と検証の文書の hash が重複していないことを確認する（検算8）。

### 4.3 活性化の取り出し
- モデルは bf16 の重みを **float32 に変換して** GPU に載せ、`model.eval()`、`torch.no_grad()` で順伝播する（perplexity の評価と同じ精度にそろえるため）。
- `model.model.layers[14].mlp.up_proj` の入力を pre-hook で取り出し、float64 に変換して `data/...` に保存する（系列ごと・文書 id つき）。
- 文書ごとの Gram 行列と、合計の G を作る。G の診断を `metric_spectrum.csv` に保存する。
- W と W_gate は bf16 の値を float64 に変換して使う。

### 4.4 方式（設計 §4）
- tensorization：T1（4コア）、T2（3コア）。選んだ mode を記録する。
- 予算：50%（1,572,864）、25%（786,432）。行列の rank はそれぞれ 384、192。
- M1、M2、M3、M4、M5 を実装する。M5 は M3 から始めたものと M4 から始めたものの2通りを実行し、較正データの Q_λ が小さい方を採用する。
  - M4 から始めるときは、D⁻¹ をコアに吸収してから、M5 の rank（M3 と同じ）までゼロで bond を広げ、正準形にしてから始める。
- **M4b（AWT-Std）**：arXiv 2610.10085 の §4、式(5)・(6) を原典で確認できた場合だけ実装する。確認できない場合は推測で実装せず、未実装として報告する。
- 較正量の感度確認（16系列）は、M2 と M5 だけでよい。

### 4.5 評価（設計 §5）
- 較正・同じドメインの検証・別ドメインの検証のそれぞれで、e、E_rel、E_gated を計算する。較正は、正則化なしの値と Q_λ の両方を出す。
- ブートストラップで、g = e(M3) − e(M2) などの差の95%区間を出す。
- 1層置換の perplexity：WikiText-2 validation の全文（`"\n\n".join(text)`）、系列長 1,024、stride 512、モデル全体を float32 で評価する。元のモデルも同じ条件で測る。設定を記録する。
- 判定（設計 §9）の計算は、関数として用意だけしておく。**本実行の前には判定を出さない。**

### 4.6 検算（モデルを使うもの）
- 検算6：up_proj を元の W のまま差し替えたとき（bf16 → float64 → float32 の経路）、float32 の元のモデルと logits・perplexity が一致する。
- 検算8：文書の hash が重複していない。
- 結果を `checks.json` に保存する。pytest の結果もあわせて記録する。

### 4.7 計算時間の事前計測（設計 §6.5）
- 条件：T2、予算 25%、M5（M3 から開始）、1コアの更新だけ、CG 10反復、半分の掃引1回。
- HVP 1回の時間、CG 1反復の時間、ピークメモリを測り、§4.4 の全条件（M5 の最悪の場合：CG 500反復、掃引50回）にかかる時間を見積もる。
- **ここで止めて報告する。** 本実行には進まない。

## 5. 報告してほしいこと

1. 作ったり変えたりした src のモジュール・関数の一覧と、その役割
2. pytest の結果（新しく追加したテストと、既存のテストが壊れていないか）
3. 検算1〜8の結果（数値つき）
4. データの準備の記録（系列数、トークン数、文書数、除外数）
5. G の診断（条件数、最小固有値、effective rank、外れ値のチャネル）
6. 選んだ tensorization と、各予算での rank と実際のパラメータ数
7. 計算時間の事前計測の結果と、全体の見積もり
8. M4b を実装したか。しなかった場合は、その理由
9. 設計から外れたところ、判断に迷ったところ

## 6. やらないこと

- 本実行（全条件の M5）と判定
- λ、停止条件、しきい値、予算の変更
- WikiText-2 test の使用
- 既存の Notebook・src の関数の挙動の変更（新しい関数の追加はよい。既存の関数を変える必要がある場合は、先に相談する）
- コミット
