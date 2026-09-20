---
title: Schmidt打ち切りとボンド状態の物理的意味
tags:
  - MPS
  - Schmidt
  - truncation
---

# Schmidt打ち切りとボンド状態の物理的意味

[[43_TT_MPSのSVD中心移動とSchmidt形]] のexact SVDは状態を変えず、切断のSchmidt係数を露出させる。ここでは、非零係数を実際に捨てると何が変わるかを、左ブロック $A=(i_1,i_2)$ と右ブロック $B=i_3$ の一つの切断で考える。数学的な局所誤差と複数cutのTT-SVD誤差は [[33_TT-SVDの打ち切りと誤差]] に置く。

## 1. 何を一本のボンド状態と呼ぶか

規格化された純粋状態のSchmidt分解を

$$
|\psi\rangle
=\sum_{\beta=1}^{\rho}\sigma_\beta
|L_\beta\rangle_A\otimes|R_\beta\rangle_B,
\qquad
\sum_{\beta=1}^{\rho}\sigma_\beta^2=1
$$

とする。左右の状態群はそれぞれ正規直交し、$\rho$ は非零係数の本数である。$\beta$ は単独の左サイトや右サイトの配置ではなく、切断をまたぐ対応づけられた一組のブロック状態を数える。係数を $\sigma_\beta$ から0へ変えると、その組の項全体が波動関数から消える。

$$
|\widetilde\psi_k\rangle
=\sum_{\beta=1}^{k}\sigma_\beta
|L_\beta\rangle_A\otimes|R_\beta\rangle_B,
\qquad k<\rho.
$$

$\sigma_{k+1}>0$ だから $|\widetilde\psi_k\rangle\ne|\psi\rangle$ である。exact SVDによる中心移動と異なり、これは単なるコア表示の変更ではない。ボンド次元を $\rho$ から $k$ へ下げると、保持できるSchmidtチャネルが $k$ 本になる。

## 2. 左ブロックへの射影を添字で確認する

残す左状態が張る部分空間への直交射影を

$$
P_k=\sum_{\gamma=1}^{k}|L_\gamma\rangle\langle L_\gamma|
$$

と定義する。右側には恒等写像を作用させる。途中の内積を省略せずに計算すると

$$
\begin{aligned}
(P_k\otimes I_B)|\psi\rangle
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}
\sigma_\beta
|L_\gamma\rangle
\langle L_\gamma|L_\beta\rangle
\otimes|R_\beta\rangle\\
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}
\sigma_\beta\delta_{\gamma\beta}
|L_\gamma\rangle\otimes|R_\beta\rangle\\
&=|\widetilde\psi_k\rangle.
\end{aligned}
$$

落としたSchmidt状態は射影先の空間にない。射影後の状態を規格化し直しても、それらの成分は戻らない。

## 3. 特異値の二乗は何の重みか

右ブロックをトレースアウトした縮約密度行列は、右状態の正規直交性を用いて

$$
\begin{aligned}
\rho_A
&=\operatorname{Tr}_B(|\psi\rangle\langle\psi|)\\
&=\sum_{\beta,\gamma}\sigma_\beta\sigma_\gamma
|L_\beta\rangle\langle L_\gamma|
\underbrace{\langle R_\gamma|R_\beta\rangle}_{\delta_{\gamma\beta}}\\
&=\sum_{\beta=1}^{\rho}\sigma_\beta^2
|L_\beta\rangle\langle L_\beta|.
\end{aligned}
$$

従って $\rho_A|L_\beta\rangle=\sigma_\beta^2|L_\beta\rangle$ である。$\sigma_\beta^2$ は全状態のノルム二乗に対する直交成分の重みであり、左Schmidt状態の縮約密度行列における重みでもある。ただし、これを「任意の物理現象をその割合で再現できる」という保証とは読まない。

捨てた重みを

$$
w_{\mathrm{discard}}:=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2
$$

と書く。直交性から、単一cutで一回だけSchmidt打ち切りをした非規格化状態について

$$
\begin{aligned}
\|\psi-\widetilde\psi_k\|^2
&=\left\|\sum_{\beta=k+1}^{\rho}\sigma_\beta
|L_\beta\rangle\otimes|R_\beta\rangle\right\|^2\\
&=\sum_{\beta,\gamma>k}\sigma_\beta\sigma_\gamma
\langle L_\gamma|L_\beta\rangle
\langle R_\gamma|R_\beta\rangle\\
&=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2
=w_{\mathrm{discard}}.
\end{aligned}
$$

また $\|\widetilde\psi_k\|^2=1-w_{\mathrm{discard}}$ である。再規格化した状態は $|\widetilde\psi_k^{\mathrm{norm}}\rangle=|\widetilde\psi_k\rangle/\sqrt{1-w_{\mathrm{discard}}}$ で、元状態との二乗距離は一般に $w_{\mathrm{discard}}$ とは異なる。この等式を、複数cutを順に打ち切るTT-SVD全体へそのまま流用しない。

## 4. 二つのスピン状態を全要素で見る

基底順を $|\uparrow\uparrow\rangle,|\uparrow\downarrow\rangle,|\downarrow\uparrow\rangle,|\downarrow\downarrow\rangle$ とする。具体例として

$$
|\psi\rangle
=\sqrt{0.99}|\uparrow\downarrow\rangle
+\sqrt{0.01}|\downarrow\uparrow\rangle
$$

を取る。係数行列と状態ベクトルの全要素は

$$
C=\begin{pmatrix}0&\sqrt{0.99}\\\sqrt{0.01}&0\end{pmatrix},
\qquad
[\psi]=\begin{pmatrix}0\\\sqrt{0.99}\\\sqrt{0.01}\\0\end{pmatrix}.
$$

右側を消去すると

$$
\rho_A=CC^\dagger
=\begin{pmatrix}0.99&0\\0&0.01\end{pmatrix}.
$$

rank 1に打ち切ると

$$
[\widetilde\psi_1]
=\begin{pmatrix}0\\\sqrt{0.99}\\0\\0\end{pmatrix},
\qquad
[\psi-\widetilde\psi_1]
=\begin{pmatrix}0\\0\\\sqrt{0.01}\\0\end{pmatrix},
\qquad
\|\psi-\widetilde\psi_1\|^2=0.01.
$$

再規格化後は $|\uparrow\downarrow\rangle$ という積状態になる。元の状態にあった $|\downarrow\uparrow\rangle$ との相関した組はなくなり、この切断のSchmidt rankは2から1へ変わる。「小さいから物理的に無意味」ではなく、**この状態のこの切断でノルム二乗の重みが小さい**という判断である。多数の係数が無視できない状態では、小さなボンド次元で重要な相関を失い得る。
