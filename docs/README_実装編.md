---
title: NN圧縮実装編 目次
aliases:
  - NN圧縮実装ロードマップ
  - SVD実装編
  - Tucker実装編
  - HOOI実装編
tags:
  - PyTorch
  - SVD
  - Tucker
  - HOSVD
  - HOOI
  - NN圧縮
  - 実装
---

# NN圧縮実装編 目次

## サマリー

現在の再利用可能な実装の正本は、旧 `code/` やNotebook内の関数ではなく、

```text
src/nn_compression/
```

である。

現在はSVDだけでなく、**Tensor基本演算、Tucker / HOSVD、generic / partial HOOI、Conv2d Tucker-2までsrc化済み**。

Notebookは、

```text
学習過程・自作実装
実験条件
rank候補
選択規則
結果の解釈
```

を担当し、再利用可能な処理と守るべきcontractを `src` に置く。

---

# 1. 現在のディレクトリ構成

```text
src/nn_compression/
├─ tensor/
│  └─ operations.py
├─ compression/
│  ├─ svd.py
│  ├─ linear_svd.py
│  ├─ conv_svd.py
│  ├─ tucker.py
│  ├─ hooi.py
│  ├─ conv_tucker.py
│  ├─ mlp_svd.py
│  ├─ named_layers.py
│  └─ rank_sweep.py
├─ metrics/
│  ├─ model_comparison.py
│  ├─ macs.py
│  ├─ mlp_macs.py
│  ├─ cnn_macs.py
│  └─ tensor_approximation.py
├─ models/
├─ datasets/
├─ selection/
├─ training/
└─ utils/
```

責務は大きく、

```text
Tensor基本演算
圧縮・分解
評価
モデル
データ・分割
rank選択
学習
共通utility
```

へ分ける。

---

# 2. `tensor/`

## `operations.py`

Tucker / HOOIから共通利用するTensor演算を置く。

```text
unfold
fold
mode_dot
```

`unfold(X, mode)` は対象modeを行方向へ置き、残りをflattenする。

```text
X: I0 × ... × In × ... × IN-1
↓
unfold_n(X): In × Π(m≠n) Im
```

`fold` はその逆操作、`mode_dot` は

$$
\mathcal Y=\mathcal X\times_nA
$$

を実装する。

数式・shapeは [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]] を参照。

---

# 3. `compression/` — SVD

## `svd.py`

モデル非依存のtruncated SVD、rank validation、retained energy等を扱う。

rankは数学的最大rank以下を要求し、不正rankを黙ってclipしない。

## `linear_svd.py`

```text
Linear(D_in → D_out)
↓
Linear(D_in → r, bias=False)
Linear(r → D_out, bias=元bias)
```

へ置換する。

## `conv_svd.py`

`groups=1` のConv2dを、

```text
Conv2d(C_in → C_out, K×K)
↓
Conv2d(C_in → r, K×K, bias=False)
Conv2d(r → C_out, 1×1, bias=元bias)
```

へ置換する。

空間Conv側が元の `stride / padding / dilation / padding_mode` を引き継ぎ、biasは最終層へ置く。

## `mlp_svd.py` / `named_layers.py` / `rank_sweep.py`

MLP圧縮、named module置換、rank sweepを共通化する。

rank sweepでは `include_model=False` を基本とし、全candidate modelを結果表へ保持せず、Fine-tuning対象だけrankから再構築する。

---

# 4. `compression/` — Tucker / HOSVD

## `tucker.py`

主な責務：

```text
hosvd
reconstruct_tucker
```

HOSVDではfactorを各対象modeについて**元のTensorから独立に**求める。

```text
X → mode 0 unfold → SVD → U0
X → mode 1 unfold → SVD → U1
...
```

factorを求める途中でXを逐次projectしてはいけない。それは標準的な1-pass HOSVDとは別処理になる。

coreはfactor転置で射影し、再構成ではfactorを逆向きに掛ける。

---

# 5. `compression/hooi.py`

layer固有処理を持たないgeneric HOOIを実装する。

公開API：

```text
has_converged
hooi_sweep
core_from_factors
hooi
```

`ranks` のkeyを更新対象modeとして扱う。

```python
ranks = {0: r0, 1: r1, 2: r2}
```

なら3-mode HOOI、

```python
ranks = {0: rank_out, 1: rank_in}
```

なら4階Conv weightのmode 0 / 1だけを更新するpartial HOOIになる。

1 sweep内では、先に更新したfactorを後続mode更新で使うGauss-Seidel型。入力 `factors` はcloneし、呼び出し元を破壊しない。

```text
HOSVD初期化
→ initial error
→ hooi_sweep
→ core再計算
→ reconstruction
→ relative error
→ convergence判定
```

を繰り返し、error historyはpandasに依存しない `list[float]` として返す。

---

# 6. `compression/conv_tucker.py`

Conv2d固有のTucker-2を担当する。

主なAPI：

```text
tucker2_decompose_conv_weight
build_tucker2_conv
build_tucker2_conv_from_components
tucker2_hooi
tucker2_hooi_sweep
tucker2_effective_weight
```

Conv weight

```text
(C_out, C_in, K_h, K_w)
```

のmode 0 / 1を圧縮し、

```text
C_in
→ 1x1 / U_in^T
→ R_in
→ kxk / core
→ R_out
→ 1x1 / U_out
→ C_out
```

へ置換する。

現在は `groups=1` の通常Conv2dを対象とし、中央core Convが元の `stride / padding / dilation / padding_mode` を継承する。元biasは最後の1x1へ置く。

HOSVDとHOOIは分解法だけを変え、3層構築処理はcomponents builderで共有する。

---

# 7. `metrics/`

SVD時代のmodel比較・MACs・latencyに加え、Tensor近似用に

```text
tensor_approximation.py
```

を追加している。

Tucker/HOOIでは主に

$$
\frac{\|X-\hat X\|_F}{\|X\|_F}
$$

を共通関数で計算し、自作HOOIとTensorLyの最終factor/coreも同じ評価式で比較する。

```text
parameters reduction
MACs reduction
weight relative error
validation / test accuracy
decomposition time
Fine-tuning後accuracy
```

は別指標として扱う。

---

# 8. `models/` / `training/` / `datasets/` / `selection/` / `utils/`

SVDで作った実験基盤をTuckerでも再利用する。

重要なcontract：

```text
baselineとcompressedでParameterを共有しない
device / dtype / requires_gradを維持する
train用shuffle loaderを評価で余分に回さない
split RNGとtraining RNGを分ける
benchmarkではsame input batchを使う
Pareto / kneeの数式処理と実験上の選択規則を分ける
nested moduleをstrictに置換する
```

これらは分解法に依存しない。

---

# 9. Notebookとsrcの役割分担

```text
Notebook
→ 学習過程
→ 自作アルゴリズム
→ 実験条件
→ rank候補
→ 結果・考察

src
→ reusable implementation
→ validation
→ type / docstring
→ API contract
→ regression test対象
```

そのためTucker編でも、

```text
02_hooi.ipynb
04_hooi_tucker2.ipynb
```

の自作実装は学習履歴として残し、最終比較Notebookではsrcを利用する。

---

# 10. canonical Notebook

## SVD

```text
MNIST
notebooks/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected.ipynb

Fashion-MNIST MLP
notebooks/10_svd/20_fashion_mnist_mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb

Fashion-MNIST CNN
notebooks/10_svd/30_fashion_mnist_cnn/04_cnn_conv_svd_corrected.ipynb
notebooks/10_svd/30_fashion_mnist_cnn/05_cnn_conv_linear_svd_corrected.ipynb

CIFAR-10
notebooks/10_svd/40_cifar10_cnn/02_svd_global_compression_using_src_corrected.ipynb
```

## Tucker / HOOI

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb

notebooks/20_tucker/10_cifar10_cnn/
├─ 01_tucker2_conv.ipynb
├─ 02_rank_sweep.ipynb
├─ 03_finetuning.ipynb
├─ 04_hooi_tucker2.ipynb
└─ 05_hosvd_vs_hooi_finetuning.ipynb
```

`*_before_src.ipynb` やoriginalはhistorical snapshotとして残し、正式な値はcanonical Notebook / resultsを優先する。

---

# 11. tests

SVD実装だけのreview時点では `49 passed` だった。

Tucker / HOOI src化・テスト整理後の**ローカル全pytest**では、

```text
99 passed
0 failed
```

を確認している。

これはローカル実行結果であり、GitHub CIによる独立確認ではない。

主なTucker/HOOI追加確認：

```text
3-mode HOOI
4階Conv weightのpartial HOOI
HOOI errorがHOSVD初期値より悪化しない
factor / core shape
hooi_sweepが入力factorを破壊しない
error history
invalid mode / rank / factor shape
Tucker-2 components build
bias / spatial config
device / dtype / requires_grad
```

---

# 12. 旧 `code/` の位置付け

```text
code/
= historical

src/nn_compression/
= current reusable implementation
```

旧Notebookやartifactは学習履歴として残すが、新しい実装はsrcを正本とする。

---

# 13. 数式・検証との対応

- [[00_基礎理論/00_数式導出監査]]
- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[05_SVD基礎実装検証/README]]
- [[06_Tucker基礎実装検証/README]]

理論式、Notebookの学習実装、srcの最終API、testsの4層を対応させて読む。

---

# 14. 次の実装

次は **TT / MPS**。

SVDで学んだtruncationと、Tucker/HOOIで学んだTensor mode・rank・sweep・収束を土台に、

```text
tensorization
→ TT / MPS core
→ TT rank / bond dimension
→ TT-SVD
→ reconstruction / error
→ NNへの適用
→ DMRGへの局所sweep接続
```

へ進む。

先回りして巨大なTensor Network frameworkを作らず、学習Notebookで必要な最小単位を理解してからsrc化する方針を継続する。

---

# 関連

- [[README]]
- [[00_基礎理論/README]]
- [[05_SVD基礎実装検証/README]]
- [[06_Tucker基礎実装検証/README]]
- [[30_CIFAR10_CNN/README]]
- [[40_TensorTrain_MPS/README]]
