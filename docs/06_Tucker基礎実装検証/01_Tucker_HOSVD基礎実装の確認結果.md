---
title: Tucker・HOSVD基礎実装の確認結果
tags:
  - Tucker
  - HOSVD
  - 実装検証
  - PyTorch
---

# Tucker・HOSVD基礎実装の確認結果

## 1. 対象

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb
```

最終src：

```text
src/nn_compression/tensor/operations.py
src/nn_compression/compression/tucker.py
src/nn_compression/compression/hooi.py
src/nn_compression/metrics/tensor_approximation.py
```

---

## 2. 基礎実装で確認した契約

### unfold / fold

mode $n$について

$$
X_{(n)}\in\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

となり、

$$
\operatorname{fold}_n(\operatorname{unfold}_n(X))=X
$$

を満たすこと。

### mode product

$$
Y=X\times_n A
$$

で、行列 $A\in\mathbb{R}^{J\times I_n}$ を掛けたmodeだけ

$$
I_n\rightarrow J
$$

へ変わること。

### HOSVD

factorを各対象modeについて元のXから独立に求め、

$$
G=X\times_0U_0^T\times_1U_1^T\cdots
$$

からcoreを得ること。

### reconstruction

$$
\hat X=G\times_0U_0\times_1U_1\cdots
$$

で元shapeへ戻ること。

---

## 3. src化後のHOOI random tensor数値確認

seed 0、

$$
X\in\mathbb{R}^{6\times5\times4}
$$

rank

$$
(3,2,2)
$$

で確認した。

| 指標 | 値 |
| --- | ---: |
| HOSVD relative error | 0.865671 |
| HOOI final relative error | 0.768581 |
| 改善量 | 0.097089 |
| core shape | `(3, 2, 2)` |
| 実行したsweep | 10 |

この数値確認では、HOOIがHOSVD初期値より明確に誤差を下げる方向に動作した。

これは特定random tensorに対するsanity checkであり、一般的な改善幅を意味しない。

---

## 4. Conv-like random weightのpartial HOOI確認

seed 0、

$$
W\in\mathbb{R}^{64\times32\times3\times3}
$$

rank

$$
(R_{out},R_{in})=(32,16)
$$

で確認した。

| 指標 | 値 |
| --- | ---: |
| HOSVD relative error | 0.755302 |
| HOOI final relative error | 0.727775 |
| 改善量 | 0.027528 |
| core shape | `(32, 16, 3, 3)` |
| $U_{out}$ | `(64, 32)` |
| $U_{in}$ | `(32, 16)` |
| 実行したsweep | 10 |

mode 2 / 3を圧縮しないpartial HOOIでも、core shapeが

$$
(R_{out},R_{in},K_h,K_w)
$$

になることを確認できた。

---

## 5. Conv構築の一致

同じHOSVD componentsから

```text
build_tucker2_conv
build_tucker2_conv_from_components
```

で構築したモデルのforward出力relative errorは

```text
0.00e+00
```

となり一致した。

これにより、分解ロジックと3層構築ロジックを分離しても同じforwardを再現できることを確認した。

---

## 6. 回帰テスト

src整理後、HOOIとConv Tucker-2の重複テストを統合し、全pytestを実行した。

```text
99 passed
0 failed
```

主な確認項目：

- 3階HOOI
- Conv weight partial HOOI
- HOOI errorがHOSVD初期値より悪化しない
- factor/core shape
- `hooi_sweep` が入力factorを破壊しない
- error historyが許容誤差内で非増加
- invalid mode / rank / factor shape validation
- HOSVD構築とcomponents構築のforward一致
- Convのbias / stride / padding / dilation
- device / dtype / requires_grad

---

## 7. 結論

基礎Tensor操作、HOSVD、Tucker再構成、partial HOOIを同じmode演算の上に構成でき、random tensorと回帰テストの両面で実装契約を確認した。

次のCNN実験では、この基礎実装を実際の学習済み `conv2.weight` に適用する。
