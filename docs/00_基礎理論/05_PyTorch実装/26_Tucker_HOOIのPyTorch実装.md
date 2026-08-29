---
title: Tucker・HOOIのPyTorch実装
aliases:
  - Tucker src実装
  - HOOI src実装
  - Conv Tucker PyTorch
tags:
  - Tucker
  - HOOI
  - PyTorch
  - src
  - NN圧縮
---

# Tucker・HOOIのPyTorch実装

## サマリー

Tucker/HOOI編では、Notebookで自作して理解した処理を、最終的に `src/nn_compression/` へ責務分離した。

```text
Tensor演算
→ tensor/operations.py

Tensor入力contract
→ tensor/validation.py

HOSVD / Tucker再構成
→ compression/tucker.py

Tucker / HOOI固有contract
→ compression/tucker_validation.py

汎用HOOI
→ compression/hooi.py

Conv2d Tucker-2
→ compression/conv_tucker.py

Tensor近似誤差
→ metrics/tensor_approximation.py
```

学習Notebookは自作実装を残し、再利用用srcは汎用APIとして整理する。

---

## 1. Tensor演算

```text
src/nn_compression/tensor/operations.py
src/nn_compression/tensor/validation.py
```

主な関数：

```python
unfold(X, mode)
fold(unfolded, mode, shape)
mode_dot(X, matrix, mode)
```

`mode_dot` は

$$
\mathcal{Y}=\mathcal{X}\times_n A
$$

を実装し、HOSVD/HOOI双方から使う。

現行public contractでは、Tensor APIへ渡すshapeは2次元以上で各dimensionが正であることを要求する。`mode` はboolではない整数で、

$$
0\le n < \operatorname{ndim}(X)
$$

を満たす必要がある。

`fold` は単に総要素数が一致すればreshapeするのではなく、unfoldedの行数・列数が指定modeと元shapeに一致するかを確認する。転置されたunfoldedを黙って受理しない。

---

## 2. HOSVD / Tucker

```text
src/nn_compression/compression/tucker.py
src/nn_compression/compression/tucker_validation.py
```

主な関数：

```python
hosvd(X, ranks)
reconstruct_tucker(core, factors)
tucker_parameter_count(shape, ranks)
parameter_ratio(shape, ranks)
compression_factor(shape, ranks)
```

`hosvd` の重要な実装契約は、factorを各modeについて**元のXから独立に**求めること。

```text
X → mode0 unfold → SVD → U0
X → mode1 unfold → SVD → U1
...
```

factor計算中にXを順次projectしてはいけない。それをすると標準的な一回のHOSVDとは異なる処理になる。

### rank contract

`ranks` は `{mode: rank}` 形式の `Mapping` とする。空 `{}` はHOSVD/Tucker parameter計算ではidentity指定として扱えるが、HOOIでは更新対象が無いため拒否する。

mode-n unfolding

$$
X_{(n)}\in\mathbb R^{I_n\times\prod_{m\ne n}I_m}
$$

に対して取りうるSVD rank上限は、

$$
\boxed{
R_n
\le
\min\left(I_n,\prod_{m\ne n}I_m\right)
}
$$

である。現行srcではHOSVD、parameter count、ratio、compression factorでこの上限を共通contractとして使う。

boolはPython上 `int` のsubclassだが、rankやmodeとしては明示的に拒否する。

### dtype contract

現行HOSVD/HOOIはfactor射影に `U.T` を使う**実数Tensor向け実装**である。

複素Tensorでは本来共役転置 `U.mH` が必要であり、`torch.linalg.svd` 自体は複素dtypeを処理できてしまう。silent failureを避けるため、現行srcでは複素Tensorを入口で `TypeError` として拒否する。

---

## 3. 汎用HOOI

```text
src/nn_compression/compression/hooi.py
```

公開API：

```python
has_converged
hooi_sweep
core_from_factors
hooi
```

### `hooi_sweep`

`ranks` のkeyを更新対象modeとして扱う。

```python
for target_mode in ranks:
    projected = X
    for other_mode in ranks:
        if other_mode == target_mode:
            continue
        projected = mode_dot(
            projected,
            updated_factors[other_mode].T,
            other_mode,
        )
```

これにより、

```python
ranks = {0: 3, 1: 2, 2: 2}
```

なら3-mode HOOI、

```python
ranks = {0: rank_out, 1: rank_in}
```

なら4階Conv weightのmode 0 / 1だけを更新するpartial HOOIになる。

入力 `factors` はcloneし、破壊的に変更しない。1 sweep内では先に更新したfactorを後続mode更新に使うGauss-Seidel型である。

### HOOI入力validation

反復前に、

- `ranks` がMappingで空でない
- factor keyとrank keyが一致
- factorが2次元
- factor shapeが `(X.shape[mode], rank)`
- factorのdevice / dtypeがXと一致
- rankが元unfoldingだけでなく、他factorで射影した後のprojected unfoldingでも実現可能

を確認する。

このfeasibility確認は反復loopより前に行うため、`max_iter=0` でも不可能なrankを受理しない。

### `core_from_factors`

$$
G
=
X
\times_0U_0^{\mathsf T}
\times_1U_1^{\mathsf T}
\cdots
$$

を現在factorから計算する。

### `hooi`

```text
HOSVD初期化
→ factor / rank feasibility確認
→ initial errorをhistory[0]へ保存
→ hooi_sweep
→ core再計算
→ reconstruction
→ relative error
→ has_converged
```

を繰り返す。

historyはpandasに依存しない `list[float]` とし、Notebook側でDataFrame化する。

HOOIではrelative Frobenius errorを使うため、分母が0になるzero Tensorは明示的に拒否する。

`max_iter` はboolではない0以上の整数、`abs_tol / rel_tol` は有限かつ0以上を要求する。`has_converged()` 単体でも同じtolerance contractを検証する。

---

## 4. Conv Tucker-2

```text
src/nn_compression/compression/conv_tucker.py
```

主なAPI：

```python
tucker2_decompose_conv_weight
build_tucker2_conv
build_tucker2_conv_from_components
tucker2_hooi
tucker2_hooi_sweep
tucker2_effective_weight
```

### `build_tucker2_conv_from_components`

事前に計算した

```text
core
U_out
U_in
```

から

```text
C_in
→ 1x1 / U_in.T
→ R_in
→ kxk / core
→ R_out
→ 1x1 / U_out
→ C_out
```

を作る。

HOSVDとHOOIで分解法が違っても、3層を作る処理は共通化できる。

現行実装は `groups=1` の通常 `nn.Conv2d` を対象とする。中央core Convが元Convの `stride / padding / dilation / padding_mode` を継承し、元biasは最後の1x1 Convへ置く。

`rank_out / rank_in` はboolを拒否し、

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

を入口で要求する。

### `tucker2_effective_weight`

fine-tuning後の3層から

$$
\hat W
=
G\times_0U_{\mathrm{out}}\times_1U_{\mathrm{in}}
$$

を再構成し、元の1つのConv weightと比較できるようにする。

---

## 5. Autograd境界：`detach()` と `no_grad()`

現行srcでは、**Tensor-level decomposition** と **Module構築** でautograd境界を分ける。

### Tensor-level HOOI

```python
tucker2_hooi(weight, ...)
```

は低レベルTensor APIなので、内部で入力 `weight` を `detach()` しない。入力が `requires_grad=True` なら、分解計算のgraphを不要に切らない。

### Module構築

```python
build_tucker2_conv(conv, ...)
```

は学習済みConvから新しい3層Moduleを作る**初期化処理**なので、ここで元 `conv.weight` を `detach()` して分解する。

新しい層への値コピーは、

```python
with torch.no_grad():
    input_layer.weight.copy_(...)
    core_layer.weight.copy_(...)
    output_layer.weight.copy_(...)
```

とする。

これはコピー操作のgradient historyを不要にするためであり、コピー後のParameterを学習不能にするものではない。`requires_grad` は元Convから引き継ぎ、fine-tuningでは通常どおり

```python
loss.backward()
optimizer.step()
```

で更新できる。

置換後の3層Parameterはleafであり、元ConvのParameter storageを共有しない。

### effective weight評価

`tucker2_effective_weight()` はfine-tuning前後の3層を評価用の1つの4階weightへ戻すhelperなので、各layer weightをdetachして再構成する。

---

## 6. validationの責務

現行srcでは、中心アルゴリズムと入力contractを分ける。

```text
tensor/validation.py
→ Tensor ndim / positive shape / mode

compression/tucker_validation.py
→ Tucker shape / rank Mapping / unfolding最大rank
→ 実数dtype
→ tolerance / max_iter

conv_tucker.py内部validation
→ Conv2d 4階shape
→ rank_out / rank_in
→ component shape / dtype / device
→ groups=1 / Sequential構造
```

似た整数・shape validationが一部に重複しているが、現時点では依存方向を壊す大規模refactorを避けている。特に `compression.hooi` が `metrics` を利用するため、metrics側からcompression helperを安易にimportすると循環importを作る。

NumPy scalarをどのAPIまで受理するかは全体で完全統一しておらず、後続のAPI設計課題として残している。

---

## 7. 評価・学習側で固定したcontract

Tucker/HOOIの分解実装だけでなく、比較実験の共通基盤もsrc reviewで固定した。

- `evaluate()` は処理前後でroot + 全submoduleの `.training` 状態を個別に復元
- `agreement()` / `logits_rmse()` / `benchmark_inference()` / `collect_compression_metrics()` も同じ状態復元contractを利用
- `train_one_epoch()` / `evaluate()` はempty loaderを明示的に拒否
- `agreement()` / `logits_rmse()` は `len(loader)` に依存せず `IterableDataset` を扱える
- `benchmark_inference()` の `warmup` は0以上、`repeats` は1以上のboolではない整数
- baseline / compressedのlatency比較では同じ `input_batch` を使う

`collect_compression_metrics()` は複数指標でloaderを再走査するため、現状はone-shot iteratorではなく再走査可能なloaderを前提とする。

---

## 8. Notebookとsrcの役割分担

```text
Notebook
→ 学習過程
→ 自作アルゴリズムを残す
→ なぜそうなるかを確認する

src
→ 最終的な再利用実装
→ validation / docstring / 型を整理
→ 実験Notebookから呼び出す
```

そのため、

```text
02_hooi.ipynb
04_hooi_tucker2.ipynb
```

は自作実装を学習記録として残し、

```text
05_hosvd_vs_hooi_finetuning.ipynb
```

はsrc化後、重複HOOI処理をsrc呼び出しへ置換した。

---

## 9. 回帰テスト

Tucker/HOOI src化直後はローカル `99 passed` だったが、その後src全体のcontract reviewを複数回行った。

現行rv6では、

```text
252 passed
0 failed
```

をローカル全pytestで確認している。これはTucker専用テスト数ではなくリポジトリ全体のtest suiteであり、GitHub CIによる独立確認を意味しない。

主な確認項目：

- 3階Tensor HOOI / 4階Conv weightのpartial HOOI
- core / factor shape
- `hooi_sweep` が入力factorを破壊しない
- HOOI error history / convergence
- HOSVD buildとcomponents buildのforward一致
- Tensor ndim / positive shape / fold shape
- mode / rankのbool拒否
- Tucker ranks Mapping contract
- mode-n unfolding最大rank
- projected HOOI rank feasibility
- `max_iter=0` でもfeasibility検証
- 複素dtype拒否
- device / dtype / requires_grad
- Tucker-2 Tensor-level autograd保持
- Module構築後Parameterのleaf性・storage非共有
- bias / spatial config / groups contract
- empty loader / IterableDataset
- root + 全submoduleのtrain/eval状態復元
- benchmark `warmup / repeats` validation
- CUDA / device contract

検証詳細：[[06_Tucker基礎実装検証/01_Tucker_HOSVD基礎実装の確認結果]]
