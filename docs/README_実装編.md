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

現在はSVDだけでなく、**Tensor基本演算、Tucker / HOSVD、generic / partial HOOI、Conv2d Tucker-2、入力validation、学習・評価状態管理までsrc化済み**。

Notebookは、

```text
学習過程・自作実装
実験条件
rank候補
選択規則
結果の解釈
```

を担当し、再利用可能な処理と守るべきcontractを `src` に置く。

src全体のアーキテクチャ、各moduleの責務、処理フロー、Public APIの責務・引数・戻り値・使用場面は [[07_src設計/README]] を正本とする。

---

# 1. 現在のディレクトリ構成

```text
src/nn_compression/
├─ tensor/
│  ├─ operations.py
│  └─ validation.py
├─ compression/
│  ├─ svd.py
│  ├─ linear_svd.py
│  ├─ conv_svd.py
│  ├─ tucker.py
│  ├─ tucker_validation.py
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
入力contract
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

## `validation.py`

Tensor-level public APIの共通contractを持つ。

```text
Tensorは2階以上
各dimensionは正
modeはboolではない整数
modeは0 <= mode < ndim
foldのtarget shapeは2次元以上かつ正
```

`fold` はさらにunfolded行列のrow / column / numel整合まで確認し、誤ったtransposeを黙ってreshapeしない。

この層は `compression` に依存しない。Tensor基本演算をTucker以外からも再利用できる依存方向を保つ。

---

# 3. `compression/` — SVD

## `svd.py`

モデル非依存のtruncated SVD、rank validation、retained energy等を扱う。

rankは数学的最大rank以下を要求し、不正rankを黙ってclipしない。bool/Tensor scalarは整数rankとして扱わない。

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
tucker_parameter_count
parameter_ratio
compression_factor
```

HOSVDではfactorを各対象modeについて**元のTensorから独立に**求める。

```text
X → mode 0 unfold → SVD → U0
X → mode 1 unfold → SVD → U1
...
```

factorを求める途中でXを逐次projectしてはいけない。それは標準的な1-pass HOSVDとは別処理になる。

coreはfactor転置で射影し、再構成ではfactorを逆向きに掛ける。

## `tucker_validation.py`

Tucker/HOOI固有のpublic contractをまとめる。

```text
ranksはMapping
bool rankを拒否
mode-n unfolding上の最大rank
実数浮動小数点dtype限定
abs_tol / rel_tolは有限かつ0以上
max_iterはbool/Tensor scalarではない0以上の整数
```

rank上限は単なるmode dimensionではなく、

$$
\min\left(I_n,\prod_{m\ne n}I_m\right)
$$

というmode-n unfoldingのSVD最大rankで統一する。

現行HOSVD/HOOIはfactor射影に `U.T` を用いるため**実数浮動小数点Tensor限定**。複素Tensorは共役転置が必要で、integer/bool Tensorも線形代数分解の正式対象外なので、Public API入口で `TypeError` とする。

`tucker.py` と `tucker_validation.py` を分けることで、中心数式と入力contractを分離する。

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
→ factor / rank feasibility確認
→ initial error
→ hooi_sweep
→ core再計算
→ reconstruction
→ relative error
→ convergence判定
```

を繰り返し、error historyはpandasに依存しない `list[float]` として返す。

HOOIのfeasibilityはloop前に検証するため、`max_iter=0` でも不可能なrankを受理しない。

zero Tensorはrelative Frobenius errorの分母が0になるため、HOOIでは明示的に拒否する。

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

### Autograd境界

```text
tucker2_hooi
→ Tensor-level decomposition
→ 入力weightをdetachしない

build_tucker2_conv
→ Module construction / initialization
→ 元weightをdetachして新しいleaf Parameterへcopy
```

元Convと置換後3層はParameter storageを共有しない。device / dtype / requires_gradは維持する。

`tucker2_effective_weight()` はfine-tuning前後の3層を等価な4階weightへ戻してweight errorを評価するためのhelperで、評価用に各layer weightをdetachして再構成する。

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

## `macs.py`

圧縮MACs helperは、実際の分解で生成できないrankに対して値だけ返さない。

```text
bool / Tensor scalar rankを拒否
1 <= rank <= 最大rank
compressed Convはgroups=1
out_h / out_wはbool/Tensor scalar以外の正整数scalar
```

を要求する。

`metrics` が `compression` のvalidation helperを直接importすると依存方向によって循環importを作るため、小さな整数validationはmetrics側にも局所的に持つ。

## `model_comparison.py`

```text
count_parameters
parameters_reduction
accuracy_drop
agreement
logits_rmse
benchmark_inference
collect_compression_metrics
```

等を持つ。

`benchmark_inference()` の `warmup` は0以上、`repeats` は1以上の整数を要求し、bool/Tensor scalarは拒否する。同一比較ではsame input batchを使う。

`agreement` / `logits_rmse` は `len(loader)` に依存せず、実走査後にempty loaderを検出するため `IterableDataset` に対応する。

`collect_compression_metrics()` は複数指標で同じloaderを再走査するため、現状は再走査可能なloaderを要求する。

---

# 8. `training/`

## `loops.py`

`train_one_epoch()` / `evaluate()` を持つ。

`train_one_epoch()` は空loaderを `ZeroDivisionError` に落とさず、実走査後に明示的な `ValueError` とする。`len(loader)` を仮定しない。

`evaluate()` は一時的に `model.eval()` へ切り替えるが、呼び出し前後でrootだけでなく**全submoduleの `.training` 状態を個別に保存・復元**する。

これは、

```text
root model = train
一部BatchNorm = eval
```

のようなfrozen submodule構成を `model.train(old_state)` で一括上書きしないためである。正常終了時だけでなく例外時も復元する。

`metrics.model_comparison` の状態保護も同じhelperへ委譲する。

---

# 9. `models/` / `datasets/` / `selection/` / `utils/`

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

# 10. Notebookとsrcの役割分担

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

# 11. canonical Notebook

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

# 12. tests

SVD実装だけのreview時点では `49 passed`、Tucker/HOOI src化直後は `99 passed` だった。

その後、src全体のpublic API contract reviewを複数回行い、baseline `49836bc` ではローカル全pytestで、

```text
252 passed
0 failed
```

を確認している。

Core API v1 self reviewでは、追加で `tests/test_core_api_v1_contract.py` を作成し、境界validationを補強した。この追加差分はGitHub CIが無いため、最終freeze前にローカルpytest確認を行う。

主な確認：

```text
Tensor ndim / positive shape
mode bool拒否 / 範囲
Tucker ranks Mapping限定
bool/Tensor scalar rank拒否
mode-n unfolding最大rank
HOOI projected rank feasibility
max_iter=0でもfeasibility検証
complex / integer / bool dtype拒否
Tucker-2 autograd境界
leaf Parameter / storage非共有
empty loader / IterableDataset
全submodule train/eval状態復元
benchmark warmup / repeats validation
CUDA / device contract
```

---

# 13. 現行public APIで意図的に残している制約

以下はCritical/Majorではなく、後続の設計判断として残している。

```text
NumPy scalar受理方針は全APIで完全統一していない
collect_compression_metricsは再走査可能なloaderを前提とする
validation helperに小規模な重複がある
HOSVD/HOOIは複素・integer・bool Tensor未対応
```

validation helperの重複は、単にまとめればよいとは限らない。`compression.hooi` が `metrics` を利用する現在の依存関係では、`metrics` から `compression` のhelperをimportすると循環importを作るためである。

TT/MPSで同種のvalidationが増えた段階で、`tensor` / `compression` / `metrics` の依存方向を維持できる共通validation層を新設するか再検討する。

---

# 14. 旧 `code/` の位置付け

```text
code/
= historical

src/nn_compression/
= current reusable implementation
```

旧Notebookやartifactは学習履歴として残すが、新しい実装はsrcを正本とする。

---

# 15. 数式・検証との対応

- [[00_基礎理論/00_数式導出監査]]
- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[05_SVD基礎実装検証/README]]
- [[06_Tucker基礎実装検証/README]]
- [[07_src設計/README]]

理論式、Notebookの学習実装、srcの最終API、tests、設計書を対応させて読む。

---

# 16. 次の実装

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

Tucker src reviewで追加した次の原則もTT/MPSへ引き継ぐ。

```text
shape / rankを入口で検証する
不可能なrankを黙って補正しない
Tensor-level APIとModule constructionのautograd境界を分ける
評価前後のmodel/submodule状態を壊さない
empty / Iterable loaderを意識する
理論計算量と実測latencyを分ける
```

先回りして巨大なTensor Network frameworkを作らず、学習Notebookで必要な最小単位を理解してからsrc化する方針を継続する。

---

# 関連

- [[README]]
- [[00_基礎理論/README]]
- [[05_SVD基礎実装検証/README]]
- [[06_Tucker基礎実装検証/README]]
- [[07_src設計/README]]
- [[30_CIFAR10_CNN/README]]
- [[40_TensorTrain_MPS/README]]
