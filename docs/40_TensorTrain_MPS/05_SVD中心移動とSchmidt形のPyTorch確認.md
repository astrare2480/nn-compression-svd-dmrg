---
title: SVD中心移動とSchmidt形のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - SVD
  - Schmidt
  - canonical
---

# SVD中心移動とSchmidt形のPyTorch確認

## このノートの位置づけ

`notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb` の数値確認結果をまとめる。

理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/43_TT_MPSのSVD中心移動とSchmidt形]] を参照する。QRによる中心移動は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]]、mixed-canonical環境の等長性は [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]] を参照する。

## Setup

第2サイトを中心とするmixed-canonical form

$$
X
=
G_1^{[L]}
G_2^{[C]}
G_3^{[R]}
$$

を使った。

core shapeは、

$$
G_1^{[L]}:(1,4,2),\qquad
G_2^{[C]}:(2,3,3),\qquad
G_3^{[R]}:(3,5,1)
$$

だった。

元Tensorからmixed-canonical形への再構成誤差は、

$$
8.43\times10^{-15}
$$

だった。

## 第2中心コアのSVD

左展開

$$
A
=
G_2^{[C]\langle L\rangle}
\in
\mathbb R^{6\times3}
$$

へreduced SVDを適用した。

得られたshapeは、

$$
U:(6,3),\qquad
S:(3,),\qquad
V^T:(3,3)
$$

だった。

$U$を第2coreへ戻し、$\Sigma V^T$を第3coreへ吸収したSVD center move後のshapeは、

$$
G_2^{[L]}:(2,3,3),
$$

$$
G_3^{[C]}:(3,5,1)
$$

となった。

再構成誤差は、

$$
1.58\times10^{-14}
$$

だった。

すべての特異値を保持するexact SVDなので、全Tensorは丸め誤差水準で不変だった。

## SVD後の直交性と中心ノルム

第2coreの左直交性誤差は、

$$
9.04\times10^{-16}
$$

だった。

centerを第3サイトへ移した後、

$$
\left|
\|X\|_F
-
\|G_3^{[C]}\|_F
\right|
\approx
3.55\times10^{-15}
$$

となり、center normの局所化を確認した。

## Sigmaをbond上へ明示する

右coreを$V^T$でbond基底回転した行列は、

$$
\widetilde R
\in
\mathbb R^{3\times5}
$$

となった。

右直交性誤差は、

$$
\|\widetilde R\widetilde R^T-I\|_F
\approx
8.28\times10^{-16}
$$

だった。

$$
X
=
G_1^{[L]}
G_2^{[L]}
\Sigma
\widetilde G_3^{[R]}
$$

というSigmaをbond上に残した形での再構成誤差は、

$$
1.39\times10^{-14}
$$

だった。

## Schmidt states

左Schmidt blockは、

$$
L\in\mathbb R^{12\times3},
$$

右Schmidt blockは、

$$
R\in\mathbb R^{3\times5}
$$

となった。

直交性誤差は、

$$
\|L^TL-I\|_F
\approx
8.24\times10^{-16},
$$

$$
\|RR^T-I\|_F
\approx
8.28\times10^{-16}
$$

だった。

したがって、

$$
|X\rangle
=
\sum_{\beta=1}^{\rho}
\sigma_\beta
|L_\beta\rangle
\otimes
|R_\beta\rangle
$$

というSchmidt形の左右block statesが数値的に正規直交していることを確認した。

## Norm identity

数値結果は、

$$
\|X\|_F^2
=
714.0546134934193,
$$

$$
\|G_2^{[C]}\|_F^2
=
714.0546134934189,
$$

$$
\sum_\beta\sigma_\beta^2
=
714.0546134934187
$$

だった。

差は、

$$
\left|
\|X\|_F^2
-
\|G_2^{[C]}\|_F^2
\right|
\approx
4.55\times10^{-13},
$$

$$
\left|
\|X\|_F^2
-
\sum_\beta\sigma_\beta^2
\right|
\approx
5.68\times10^{-13}
$$

で、float64の丸め誤差水準だった。

## 確認できたこと

1. 全特異値を保持するSVD center moveはexactである。
2. $U$を左coreへ戻すことで左直交性が得られる。
3. $V^T$によるbond基底回転後も右直交性が保たれる。
4. $\Sigma$をbond上へ明示するとSchmidt係数を直接読める。
5. 左右Schmidt statesは正規直交する。
6. $\|X\|_F^2=\sum_\beta\sigma_\beta^2$ を数値確認した。
