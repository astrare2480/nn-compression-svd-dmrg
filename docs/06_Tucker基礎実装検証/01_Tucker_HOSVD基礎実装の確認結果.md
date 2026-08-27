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

最終src：`tensor/operations.py`、`compression/tucker.py`、`compression/hooi.py`、`metrics/tensor_approximation.py`。

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

## 5. 回帰テスト

src整理後の全pytest：

```text
99 passed
0 failed
```

主な確認：3階HOOI、partial HOOI、factor/core shape、入力factor非破壊、収束history、invalid mode/rank、Conv bias/spatial config/device/dtype/requires_grad。

## 結論

Tensorのmode演算→HOSVD→HOOIを同じ基本演算上に構成し、3階full Tuckerと4階partial Tucker-2の両方でshapeと誤差改善方向を確認した。
