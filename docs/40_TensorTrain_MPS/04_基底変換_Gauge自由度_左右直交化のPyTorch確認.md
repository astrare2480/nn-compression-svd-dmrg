---
title: 基底変換・Gauge自由度・左右直交化のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - gauge
  - QR
  - orthogonalization
---

# 基底変換・Gauge自由度・左右直交化のPyTorch確認

## このノートの位置づけ

以下の学習Notebookで確認した結果をまとめる。

- `notebooks/30_tt_mps/00_fundamentals/03_basis_transform_rank_invariance.ipynb`
- `notebooks/30_tt_mps/00_fundamentals/04_gauge_freedom_qr_L2.ipynb`
- `notebooks/30_tt_mps/00_fundamentals/05_right_orthogonalization_right_block.ipynb`

理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/34_基底変換とTT-rank不変性]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/35_TT_MPSの等長写像と射影]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/36_TT_MPSのGauge自由度と左QR直交化]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/37_TT_MPSの左ブロックと直交性の導出]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/38_TT_MPSの右QR直交化と右ブロック]] を参照する。

## 03: 基底変換とrank不変性

3階Tensor

$$
X\in\mathbb R^{2\times3\times4}
$$

について、第1cutでは

$$
X^{\langle1\rangle}
\in
\mathbb R^{2\times12}
$$

を直交行列で基底変換した。

直交性誤差は

$$
\|U^TU-I\|_F
\approx
1.57\times10^{-16}
$$

だった。

第2cutでも左側へ直交変換を掛けた関係

$$
X^{\langle2\rangle}
=
L_2B^{\langle2\rangle}
$$

について、

$$
\|X^{\langle2\rangle}-L_2B^{\langle2\rangle}\|_F
\approx
8.00\times10^{-16}
$$

を確認した。

rankは、

$$
\operatorname{rank}(X^{\langle2\rangle})
=
\operatorname{rank}(B^{\langle2\rangle})
=
4
$$

で一致した。

一方、内部rankを意図的に1まで落とした別Tensorでは対応rankが

$$
3
$$

となり、元のrank 4から変化した。

したがって、**可逆・直交な基底変換そのものではrankは変わらないが、内部表現を実際に低rank化すればunfolding rankも変化する**ことを確認した。

## 04: Gauge自由度と左QR直交化

core shapeは、

$$
G_1:(1,4,2),\qquad
G_2:(2,3,3),\qquad
G_3:(3,5,1)
$$

とした。

bond上へ可逆行列と逆行列を挿入するgauge変換後のrelative reconstruction errorは

$$
9.95\times10^{-17}
$$

だった。

第1coreをQR分解して、

$$
A_1=Q_1R_1
$$

とし、$R_1$を第2coreへ吸収した後の再構成誤差は

$$
1.53\times10^{-16}
$$

だった。

左直交性は、

$$
\|Q_1^TQ_1-I\|_F
\approx
1.72\times10^{-16}
$$

だった。

さらに第2coreも左QR直交化し、左ブロック行列

$$
L_2\in\mathbb R^{12\times3}
$$

について、

$$
\|L_2^TL_2-I\|_F
\approx
7.19\times10^{-16}
$$

を確認した。

つまり、QRによるgauge fixingで全Tensorを保ったまま左側のblock statesを正規直交化できる。

## 05: 右QR直交化と右ブロック

第3coreをright unfoldingして、

$$
A_3\in\mathbb R^{3\times5}
$$

とし、

$$
A_3^T=QR
$$

を用いて右直交coreを作った。

右直交性は、

$$
\left\|
A_3^{[R]}
\left(A_3^{[R]}\right)^T
-I
\right\|_F
\approx
4.04\times10^{-16}
$$

だった。

$R^T$を左隣coreへ吸収したとき、

- local absorption error: $2.44\times10^{-15}$
- global reconstruction error: $6.21\times10^{-15}$

となり、局所積と全Tensorが丸め誤差水準で保存された。

右ブロック行列についても、

$$
\|R_2R_2^T-I\|_F
\approx
4.04\times10^{-16}
$$

だった。

## 確認できたこと

1. 直交基底変換は対応するunfolding rankを変えない。
2. gauge変換はcoreを変えても全Tensorを変えない。
3. 左QRでは残差を右隣へ吸収することで左直交化できる。
4. 右QRでは転置QRを使い、残差を左隣へ吸収することで右直交化できる。
5. 左右のblock Gramが単位行列になることをPyTorchで確認した。
6. これらがmixed-canonical formとorthogonality centerの土台になる。
