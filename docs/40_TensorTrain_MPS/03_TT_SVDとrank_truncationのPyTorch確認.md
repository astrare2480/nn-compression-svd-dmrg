---
title: TT-SVDとrank truncationのPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - TT-SVD
  - truncation
---

# TT-SVDとrank truncationのPyTorch確認

## このノートの位置づけ

以下の学習Notebookで確認した結果を、TT/MPS基礎の数値検証としてまとめる。

- `notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb`
- `notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb`
- `notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb`

理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/30_TT_MPSの定義]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/31_TT-rankとunfolding]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/32_TT-SVD]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/33_TT-SVDの打ち切りと誤差]] を参照する。

## 00: 3階TensorのTT-SVD

3階Tensor

$$
X\in\mathbb R^{2\times3\times4}
$$

に対してTT-SVDを行い、得られたcore shapeは

$$
G_1:(1,2,2),\qquad
G_2:(2,3,4),\qquad
G_3:(4,4,1)
$$

となった。

TT coreから再構成したTensorと元Tensorとの差は

$$
\|X-\widehat X\|_F
\approx
5.96\times10^{-15}
$$

で、float64の丸め誤差水準だった。

したがって、rankを切り捨てないTT-SVDでは、元Tensorを数値的にほぼ完全再構成できることを確認した。

## 01: TT-rankとunfolding rank

4階TensorのTT core rankは

$$
(r_1,r_2,r_3)=(2,4,2)
$$

だった。

各cutに対応するunfolding rankを直接計算すると、

$$
(2,4,2)
$$

となり、

$$
\boxed{
r_k
=
\operatorname{rank}
\left(
X^{\langle k\rangle}
\right)
}
$$

が数値的に一致した。

再構成のrelative Frobenius errorは

$$
6.414\times10^{-16}
$$

だった。

この例では、

- dense parameters: 24
- TT parameters: 48

となり、**TT表現は常に圧縮になるわけではない**ことも確認した。

小さいTensorや高いbond rankでは、TTのparameter数がdense表現を上回ることがある。

## 02: Rank truncationと誤差のtrade-off

同じTensorに対し最大bond rankを変えてTT-SVDを行った結果は次の通り。

| max rank | bond ranks | TT parameters | parameter ratio | relative error |
| ---: | --- | ---: | ---: | ---: |
| 1 | [1, 1, 1] | 16 | 0.0625 | 0.942147 |
| 2 | [2, 2, 2] | 48 | 0.1875 | 0.850899 |
| 4 | [4, 4, 4] | 160 | 0.6250 | 0.614882 |
| 8 | [4, 8, 4] | 288 | 1.1250 | 0.294857 |

rankを大きくすると誤差は減少した一方、parameter数は増加した。

$$
\boxed{
\text{小さいbond rank}
\Rightarrow
\text{高圧縮・大誤差}
}
$$

$$
\boxed{
\text{大きいbond rank}
\Rightarrow
\text{低圧縮・小誤差}
}
$$

という基本的なtrade-offを確認した。

また、この例ではmax rankを8まで許すとparameter ratioが1を超えており、圧縮率と近似精度を別々に評価する必要がある。

## 確認できたこと

1. exact TT-SVDでは元Tensorを丸め誤差水準で再構成できる。
2. TT-rankは対応するunfolding rankと一致する。
3. TT表現が必ずdense表現より小さくなるわけではない。
4. bond rankを制限するとparameter数と近似誤差のtrade-offが生じる。
5. 以後のcanonical formやbond truncationでは、このrankが内部bond dimensionとして扱われる。
