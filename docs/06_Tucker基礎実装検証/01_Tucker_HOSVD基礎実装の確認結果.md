---
title: Tucker・HOSVD基礎実装の確認結果
tags: [Tucker, HOSVD, HOOI, 実装検証, PyTorch]
---

# Tucker・HOSVD基礎実装の確認結果

## 対象

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb
```

現行srcの主な対応先：

```text
src/nn_compression/
├─ tensor/
│  ├─ operations.py
│  └─ validation.py
├─ compression/
│  ├─ tucker.py
│  ├─ tucker_validation.py
│  └─ hooi.py
└─ metrics/
   └─ tensor_approximation.py
```

数式導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]] を参照する。

## 1. unfold / fold

$$
X_{(n)}\in\mathbb R^{I_n\times\prod_{m\ne n}I_m}
$$

であり、

$$
\operatorname{fold}_n(\operatorname{unfold}_n(X))=X
$$

を確認する。

現行 `tensor/operations.py` の `unfold` / `fold` / `mode_dot` は共通のmode規約を使う。public contractは `tensor/validation.py` に分離され、Tensorは2階以上、各dimensionは正、modeはboolではない有効な整数を要求する。`fold` は行数・列数・総要素数まで検証し、転置されたunfold結果を黙ってreshapeしない。

## 2. random 3階Tensor

seed 0、

$$
X\in\mathbb R^{6\times5\times4},
\qquad
(R_0,R_1,R_2)=(3,2,2).
$$

結果：

| 指標 | 値 |
| --- | ---: |
| HOSVD error | 0.865671 |
| HOOI error | 0.768581 |
| 保存された改善量 | 0.097089 |
| core | `(3,2,2)` |
| sweep | 10 |

表示値だけを引くと、

$$
0.865671-0.768581=0.097090.
$$

一方、実行時の未丸めraw値から保存された改善量は `0.097089`。末尾差は各errorの表示丸めによる。

このrandom tensorではHOOIがHOSVD初期値から誤差を下げる方向に動いた。改善幅そのものを一般化しない。

## 3. Conv-like 4階weightのpartial HOOI

$$
W\in\mathbb R^{64\times32\times3\times3},
\qquad
(R_{out},R_{in})=(32,16).
$$

mode 2/3を保持するのでcoreは

$$
\begin{aligned}
(64,32,3,3)
&\xrightarrow{\times_0U_{out}^T}
(32,32,3,3)\\
&\xrightarrow{\times_1U_{in}^T}
(32,16,3,3).
\end{aligned}
$$

結果：

| 指標 | 値 |
| --- | ---: |
| HOSVD error | 0.755302 |
| HOOI error | 0.727775 |
| 保存された改善量 | 0.027528 |
| core | `(32,16,3,3)` |
| `U_out` | `(64,32)` |
| `U_in` | `(32,16)` |
| sweep | 10 |

表示値からは

$$
0.755302-0.727775=0.027527
$$

だが、未丸めraw値から保存された改善量は `0.027528`。ここも表示丸め差として扱う。

## 4. Conv構築一致

同じHOSVD componentsから `build_tucker2_conv` と `build_tucker2_conv_from_components` を作り、forward差のrelative errorが

$$
0.00\times10^0=0
$$

になった。

これは分解値が同じなら共通builderが同じforwardを作ることのsanity check。

## 5. 現行srcで固定したTucker contract

### shape / mode / ranks

`hosvd()` / `tucker_parameter_count()` / `parameter_ratio()` / `compression_factor()` は、Tucker rankを `Mapping` として受け取る。空 `{}` はHOSVD側では「圧縮しないidentity指定」として許可されるが、HOOIでは更新対象modeが必要なので空rankは拒否する。

rankの上限は単純な `shape[mode]` ではなく、mode-n unfolding

$$
I_n\times\prod_{m\ne n}I_m
$$

に対するSVDの最大rank

$$
\min\left(I_n,\prod_{m\ne n}I_m\right)
$$

で統一する。このため、細長いTensorでdimension自体は大きくてもunfolding上実現できないrankは拒否される。

### 実数dtype限定

現行HOSVD/HOOIはfactor射影に `U.T` を使う実数Tensor向け実装として仕様を固定している。複素Tensorでは共役転置 `U.mH` が必要になるため、複素dtypeをsilentに処理せず `TypeError` で明示的に拒否する。

対象は `hosvd()`、`hooi()`、`hooi_sweep()`、`core_from_factors()` と、それらを内部利用するTucker-2 helper。

### HOOI feasibility

`hooi()` はHOSVD初期化後、反復loopへ入る前にfactor shape / dtype / device / projected unfolding上のrank feasibilityを検証する。したがって `max_iter=0` でも不可能なrankを見逃さない。

`has_converged()` は

$$
|e_t-e_{t-1}|\leq \mathrm{abs\_tol}+\mathrm{rel\_tol}|e_{t-1}|
$$

を使い、`abs_tol` / `rel_tol` はboolを拒否し、有限かつ0以上を要求する。

## 6. Autograd境界

低レベルTensor APIの `tucker2_hooi()` は入力weightをdetachしない。したがって、分解計算そのものでは `requires_grad` を不必要に切らない。

一方、学習済み `nn.Conv2d` を新しい3層moduleへ置換する `build_tucker2_conv()` は「初期化としての分解」なので元weightをdetachし、新たに構築した層のParameterへ値をcopyする。新しいParameterは元Convとstorageを共有しないleaf Parameterとなる。

この境界は、

```text
Tensor-level decomposition
→ autogradを勝手に切らない

Module construction
→ 元modelから独立した新しいParameterを作る
```

という責務分離として固定する。

## 7. 回帰テスト

src review / contract整理後のローカル全pytest：

```text
252 passed
0 failed
```

主な確認：

```text
unfold / fold / mode_dot のndim・shape・mode contract
Tucker ranksのMapping限定 / bool拒否
mode-n unfolding上の最大rank
3階HOOI / 4階partial HOOI
factor / core shape
入力factor非破壊
収束history / tolerance validation
max_iterに依存しないHOOI feasibility
実数dtype限定・複素dtype拒否
Tucker-2のdevice / dtype / requires_grad / autograd境界
```

この `252 passed` はリポジトリ全体のローカルtest suiteの結果であり、Tucker専用テスト数ではない。またGitHub CIによる独立実行結果を意味しない。

## 結論

Tensorのmode演算→HOSVD→HOOIを同じ基本演算上に構成し、3階full Tuckerと4階partial Tucker-2の両方でshapeと誤差改善方向を確認した。

その後のsrc reviewでは、数学的アルゴリズム本体を変えず、shape / mode / rank / dtype / tolerance / autograd境界をpublic contractとして明示化した。現行実装は**実数Tensorを対象とするHOSVD/HOOI**として扱う。
