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

HOSVD / Tucker再構成
→ compression/tucker.py

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
```

主な関数：

```python
unfold(X, mode)
fold(matrix, mode, shape)
mode_dot(X, matrix, mode)
```

`mode_dot` は

$$
\mathcal{Y}=\mathcal{X}\times_n A
$$

を実装し、HOSVD/HOOI双方から使う。

---

## 2. HOSVD / Tucker

```text
src/nn_compression/compression/tucker.py
```

主な関数：

```python
hosvd(X, ranks)
reconstruct_tucker(core, factors)
```

`hosvd` の重要な実装契約は、factorを各modeについて**元のXから独立に**求めること。

```text
X → mode0 unfold → SVD → U0
X → mode1 unfold → SVD → U1
...
```

factor計算中にXを順次projectしてはいけない。それをすると標準的な一回のHOSVDとは異なる処理になる。

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

入力 `factors` はcloneし、破壊的に変更しない。

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
→ initial errorをhistory[0]へ保存
→ hooi_sweep
→ core再計算
→ reconstruction
→ relative error
→ has_converged
```

を繰り返す。

historyはpandasに依存しない `list[float]` とし、Notebook側でDataFrame化する。

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

### `tucker2_effective_weight`

fine-tuning後の3層から

$$
\hat W
=
G\times_0U_{\mathrm{out}}\times_1U_{\mathrm{in}}
$$

を再構成し、元の1つのConv weightと比較できるようにする。

---

## 5. `detach()` と `no_grad()`

### 分解対象

```python
weight = conv.weight.detach()
```

分解は既存学習グラフのbackpropagationを目的にしないので、学習済みParameterから切り離して扱う。

### 新しい層へのコピー

```python
with torch.no_grad():
    input_layer.weight.copy_(...)
    core_layer.weight.copy_(...)
    output_layer.weight.copy_(...)
```

これは「分解結果を新しいParameterの初期値としてセットする」処理であり、コピー自体のgradient historyは不要。

`no_grad()` はその後の

```python
loss.backward()
optimizer.step()
```

を禁止しない。`requires_grad` を元Convから引き継げば、fine-tuningで通常どおり更新できる。

---

## 6. validation

srcでは最低限、

- mode範囲
- rank範囲
- factor key
- factor shape
- Conv2d weightの4階shape
- `groups=1`
- component shape

を確認する。

HOOIのfactor shapeは

$$
U^{(n)}\in\mathbb{R}^{I_n\times R_n}
$$

Tucker-2 coreは

$$
G\in\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times K_h\times K_w}
$$

を契約とする。

---

## 7. Notebookとsrcの役割分担

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

## 8. 回帰テスト

整理後は、HOOI系を `tests/test_hooi.py`、Conv構築系を `tests/test_conv_tucker.py` に責務分離した。

全pytestのローカル実行では

```text
99 passed
0 failed
```

を確認している。

主な確認項目：

- HOOIがHOSVD初期誤差より悪化しない
- 3階Tensor HOOI
- 4階Conv weightのpartial HOOI
- core / factor shape
- `hooi_sweep` が入力factorを破壊しない
- error historyが非増加
- HOSVD buildとcomponents buildのforward一致
- device / dtype / requires_grad
- bias / spatial config
- invalid rank / mode

検証詳細：[[06_Tucker基礎実装検証/01_Tucker_HOSVD基礎実装の確認結果]]
