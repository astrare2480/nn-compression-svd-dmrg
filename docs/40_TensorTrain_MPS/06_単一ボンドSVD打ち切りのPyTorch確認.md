---
title: 単一ボンドSVD打ち切りのPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - SVD
  - truncation
  - bond-rank
---

# 単一ボンドSVD打ち切りのPyTorch確認

## このノートの位置づけ

`notebooks/30_tt_mps/00_fundamentals/09_truncated_svd_and_bond_rank_truncation.ipynb` の数値確認結果をまとめる。

理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/45_TT_MPSの単一ボンドSVD打ち切り]] を参照する。SVD center moveとSchmidt形は [[40_TensorTrain_MPS/05_SVD中心移動とSchmidt形のPyTorch確認]] を参照する。

## Setup

第2サイトを中心とするmixed-canonical 3階TT/MPSを用い、中心行列の特異値を教材用に

$$
(\sigma_1,\sigma_2,\sigma_3)
=
(6,2,0.25)
$$

へ設定した。

中心コアの左展開は、

$$
A
=
G_2^{[C]\langle L\rangle}
\in
\mathbb R^{6\times3}
$$

である。

exact rankは、

$$
\rho=3
$$

とし、保持rankを、

$$
k=2
$$

とした。

したがって第3Schmidt成分だけを削除する例である。

## Reduced SVDとshape

実際の出力は、

- `A.shape = (6, 3)`
- `U.shape = (6, 3)`
- `S.shape = (3,)`
- `Vh.shape = (3, 3)`

だった。

truncation後は、

- `U_k.shape = (6, 2)`
- `S_k.shape = (2,)`
- `Vh_k.shape = (2, 3)`

となった。

特異値は数値的にも、

$$
(6, 2, 0.25)
$$

を再現した。

## 特異値二乗への寄与

各特異値の二乗は、

$$
(36, 4, 0.0625)
$$

だった。

したがって、

$$
\|X\|_F^2
=
40.0625,
$$

保持する成分は、

$$
36+4
=
40,
$$

捨てる成分は、

$$
0.25^2
=
0.0625
$$

である。

保持率は、

$$
\frac{40}{40.0625}
\approx
0.99844
$$

だった。

したがって、テンソル自体は変化するが、捨てるSchmidt成分のFrobeniusノルム二乗への寄与は小さい教材例になっている。

## Bond rankとcore shape

truncation後のshapeは、

$$
G_{2,\mathrm{truncated}}
:
(2,3,2),
$$

$$
T_k
=
\Sigma_kV_k^T
:
(2,3),
$$

$$
G_{3,\mathrm{truncated}}
:
(2,5,1)
$$

だった。

つまり第2–第3 bondのdimensionが、

$$
\boxed{
3\rightarrow2
}
$$

へ縮んだ。

物理次元は変化せず、内部bond dimensionだけが減っている。

## Truncation error

元Tensorとtruncated Tensorのshapeはいずれも、

$$
(4,3,5)
$$

である。

truncation errorは、

$$
\|X-\widetilde X\|_F
=
0.25000000000000006
$$

だった。

理論値は、

$$
\sqrt{
\sum_{\beta>k}\sigma_\beta^2
}
=
\sqrt{0.25^2}
=
0.25
$$

なので一致した。

二乗誤差では、

$$
\|X-\widetilde X\|_F^2
=
0.06250000000000003,
$$

$$
\sum_{\beta>k}\sigma_\beta^2
=
0.0625000000000003
$$

となった。

両者の差は、

$$
2.78\times10^{-16}
$$

だった。

したがって、

$$
\boxed{
\|X-\widetilde X\|_F^2
=
\sum_{\beta=k+1}^{\rho}
\sigma_\beta^2
}
$$

をfloat64の丸め誤差内で確認した。

## Pythagoras型のノルム分解

数値結果は、

$$
\|X\|_F^2
=
40.062500000000014,
$$

$$
\|\widetilde X\|_F^2
=
39.999999999999986,
$$

$$
\|X-\widetilde X\|_F^2
=
0.06250000000000003
$$

だった。

したがって、

$$
\boxed{
\|X\|_F^2
=
\|\widetilde X\|_F^2
+
\|X-\widetilde X\|_F^2
}
$$

のgapは、

$$
2.84\times10^{-14}
$$

で丸め誤差水準だった。

## 確認できたこと

1. truncated SVDではbond rankが実際に$3\to2$へ縮む。
2. truncation後のTensorは元Tensorと同一ではない。
3. 捨てたSchmidt成分のノルム二乗寄与は$\sigma_3^2=0.0625$である。
4. 全Tensorの二乗誤差と捨てた特異値二乗和が一致する。
5. retained partとdiscarded partは直交し、Pythagoras型のノルム分解が成り立つ。
6. ここでは「このtruncated SVDが全rank-2候補の中で最適」というEckart–Young–Mirskyの最適性までは扱わない。
