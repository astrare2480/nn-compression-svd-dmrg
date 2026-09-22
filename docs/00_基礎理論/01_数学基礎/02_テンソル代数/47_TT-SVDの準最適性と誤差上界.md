---
title: TT-SVDの準最適性と誤差上界
tags:
  - TT
  - MPS
  - TT-SVD
  - tensor-train
  - SVD
  - EckartYoungMirsky
  - Frobenius
  - quasi-optimality
  - unfolding
  - orthogonality
  - isometry
---

# TT-SVDの準最適性と誤差上界

このノートは、dense tensorから左→右にTTを構成する標準 **TT-SVD** の誤差評価と準最適性を導出する。

[[46_Eckart_Young_Mirsky定理とTT_MPS単一ボンド打ち切りの最適性|Eckart–Young–Mirsky定理と単一bond打ち切り]]から出発し、複数cutにまたがるTT-SVDではなぜ大域最適性が直ちには出ないのかを整理する。

元テンソルを $A$、TT-SVDの出力を $T$、元の第 $k$ unfoldingの最良rank-$r_k$近似誤差を $\varepsilon_k$ とする。まず、TT-SVDの基本誤差上界

$$
\|A-T\|_F
\le
\left(
\sum_{k=1}^{d-1}\varepsilon_k^2
\right)^{1/2}
$$

を導出する。次に、指定TT-rank以下の大域最適近似を $A_\ast$ として、

$$
\left(
\sum_{k=1}^{d-1}\varepsilon_k^2
\right)^{1/2}
\le
\sqrt{d-1}\,\|A-A_\ast\|_F
$$

を示す。これら二つの不等式を接続することで、TT-SVDの準最適性が得られる。その際、間違えやすい論点も明示する。

PyTorchによる数値検証は [[40_TensorTrain_MPS/07_TT-SVDの準最適性と誤差上界のPyTorch確認]]と [Notebook 11](../../../../notebooks/30_tt_mps/00_fundamentals/11_tt_svd_quasi_optimality_and_error_bound.ipynb) に分ける。

---

## 0. このノートの到達点

対象は、既存のTTを圧縮する **TT-rounding** ではなく、dense tensor

$$
A\in\mathbb R^{n_1\times\cdots\times n_d}
$$

からTTを新規構築する、標準の **左→右TT-SVD** である。

目標TT-rankを

$$
(r_1,\ldots,r_{d-1})
$$

とし、TT-SVDの出力を $T$ とする。各元のunfoldingの最良行列近似誤差を適切に $\varepsilon_k$ と定義すると、次が成り立つ。

$$
\boxed{
\|A-T\|_F^2
\le
\sum_{k=1}^{d-1}\varepsilon_k^2
}
\tag{TT-SVD基本誤差上界}
$$

さらに、TT-rank $\le(r_1,\ldots,r_{d-1})$ の集合における大域最適近似を $A_\ast$ とすれば、

$$
\boxed{
\|A-T\|_F
\le
\sqrt{d-1}\,\|A-A_\ast\|_F
}
\tag{TT-SVD準最適性}
$$

である。

この不等式は、TT-SVDが大域最適解 $A_\ast$ を常に返すという意味ではない。TT-SVDは、最良TT近似の誤差に対して最大 $\sqrt{d-1}$ 倍以内に入ることが保証される、**準最適（quasi-optimal）**な逐次アルゴリズムである。

---

## 1. 出発点：単一bondでは何が最適か

前段の学習では、mixed-canonical TT/MPSにおける単一bondのSVD打ち切りを扱った。

行列

$$
M=U\Sigma V^T
=
\sum_{\alpha=1}^{\rho}
\sigma_\alpha u_\alpha v_\alpha^T,
\qquad
\sigma_1\ge\cdots\ge\sigma_\rho>0
$$

に対して、rank $r$ に打ち切るとは

$$
M_r
=
\sum_{\alpha=1}^{r}
\sigma_\alpha u_\alpha v_\alpha^T
$$

を取ることである。Eckart–Young–Mirsky（EYM）定理より、

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|M-B\|_F
=
\|M-M_r\|_F
=
\left(
\sum_{\alpha>r}\sigma_\alpha^2
\right)^{1/2}
}
$$

であり、打ち切りSVDはFrobeniusノルムで最良のrank-$r$近似である。

### 1.1 なぜ残差が特異値の二乗和か

残差は

$$
M-M_r
=
\sum_{\alpha=r+1}^{\rho}
\sigma_\alpha u_\alpha v_\alpha^T.
$$

SVDのrank-1成分はFrobenius内積で直交する。

$$
\begin{aligned}
\left\langle
u_\alpha v_\alpha^T,
 u_\beta v_\beta^T
\right\rangle_F
&=
\operatorname{tr}
\left((u_\alpha v_\alpha^T)^T(u_\beta v_\beta^T)\right)\\
&=
(u_\alpha^Tu_\beta)(v_\alpha^Tv_\beta)\\
&=
\delta_{\alpha\beta}.
\end{aligned}
$$

また

$$
\|u_\alpha v_\alpha^T\|_F=1.
$$

従ってPythagorasにより、

$$
\boxed{
\|M-M_r\|_F^2
=
\sum_{\alpha>r}\sigma_\alpha^2.
}
$$

ここで重要なのは、これはまず「**truncated SVDを選んだときの実際の誤差**」を計算していることだという点である。EYM定理はさらに、この値が任意のrank $\le r$ 候補より小さいことを保証する。

---

## 2. TT-SVDへ進んだときの難しさ

単一行列・単一bondなら、EYMにより最良近似を一発で得られる。

しかし $d$ 階テンソルには

$$
1,2,\ldots,d-1
$$

という複数のcutがある。TT-SVDでは、1つ目のcutでSVDしてrankを制限したあと、その圧縮済みの残りを次のcutでSVDする。

したがって、次の推論は誤りである。

> 各局所SVDは、その時点での最良rank近似である。したがって最終的なTTも、全TT-rank制約の下で大域最適である。

第1段階の選択は第2段階以降の問題そのものを変える。TT-SVDは全コアを同時に最適化する方法ではなく、順番に部分空間を固定する貪欲な構成である。

それでも、誤差は制御できる。そのために、**元のテンソルの各unfoldingで定義される誤差**と、帰納法・直交性・等長性を使う。

---

## 3. 記号と対象の固定

以下、混乱を避けるため対象を標準の左→右TT-SVDに固定する。

$$
A\in\mathbb R^{n_1\times\cdots\times n_d}.
$$

### 3.1 cut $k$ のunfolding

第 $k$ cutは、添字群

$$
(i_1,\ldots,i_k)
\quad\big|\quad
(i_{k+1},\ldots,i_d)
$$

でテンソルを行列にする操作である。

$$
A^{\langle k\rangle}
\in
\mathbb R^{(n_1\cdots n_k)\times(n_{k+1}\cdots n_d)}.
$$

unfoldingは全要素の並べ替えにすぎないので、Frobeniusノルムを保存する。

$$
\|A\|_F=\|A^{\langle k\rangle}\|_F.
$$

また線形なので、

$$
(A-C)^{\langle k\rangle}
=
A^{\langle k\rangle}-C^{\langle k\rangle}.
$$

したがって、

$$
\|A^{\langle k\rangle}-C^{\langle k\rangle}\|_F
=
\|A-C\|_F.
$$

### 3.2 元のunfoldingに対する最良誤差 $\varepsilon_k$

目標TT-rankを

$$
(r_1,\ldots,r_{d-1})
$$

とする。今回の理論で使う $\varepsilon_k$ は、**元のテンソル $A$** の第 $k$ unfoldingについて定義する。

$$
\boxed{
\varepsilon_k
:=
\min_{\operatorname{rank}(B)\le r_k}
\left\|
A^{\langle k\rangle}-B
\right\|_F.
}
\tag{1}
$$

EYMにより、

$$
\boxed{
\varepsilon_k^2
=
\sum_{\ell>r_k}
\sigma_\ell\left(A^{\langle k\rangle}\right)^2.
}
\tag{2}
$$

ここで

- $k$：どのcutかを表す番号
- $\ell$：そのunfolding行列の $\ell$ 番目の特異値を表す番号
- $\sigma_\ell(A^{\langle k\rangle})$：第 $k$ unfoldingの $\ell$ 番目に大きい特異値

である。

### 3.3 最重要の区別

このノートでは、$\varepsilon_k$ を必ず (1) の意味で使う。

$$
\boxed{
\varepsilon_k
\text{ は、元の }A^{\langle k\rangle}\text{ の最良rank-}r_k\text{行列近似誤差。}
}
$$

TT-SVDの第 $k$ 段階で実際にSVDする中間行列を $M_k$、その行列で実際に捨てた特異値の尾部ノルムを $δ_k$ とする。

$$
\delta_k
:=
\min_{\operatorname{rank}(Z)\le r_k}
\|M_k-Z\|_F
=
\left(
\sum_{\ell>r_k}\sigma_\ell(M_k)^2
\right)^{1/2}.
$$

$\delta_k$ は**実際の局所打ち切り誤差**、$\varepsilon_k$ は**元の $A^{\langle k\rangle}$ の最良rank-$r_k$誤差**である。定義対象が異なるため、一般に $\delta_k=\varepsilon_k$ とは限らない。ただし、このノートで扱う標準の左から右へのTT-SVDでは、後に

$$
\delta_1=\varepsilon_1,
\qquad
\delta_k\le\varepsilon_k
\quad(2\le k\le d-1)
$$

を示す。

この区別を曖昧にしたことが、途中の混乱の主因だった。

---

## 4. 標準TT-SVDの左→右構成

### 4.1 第1ステップ

第1 unfoldingを作る。

$$
A^{\langle1\rangle}
\in
\mathbb R^{n_1\times(n_2\cdots n_d)}.
$$

SVDを取る。

$$
A^{\langle1\rangle}=U\Sigma V^T.
$$

上位 $r_1$ 成分を残す。残す左特異ベクトルを

$$
U_1=U(:,1:r_1)
\in
\mathbb R^{n_1\times r_1}
$$

とし、

$$
\Sigma_1=
\Sigma(1:r_1,1:r_1),
\qquad
V_1^T=V^T(1:r_1,:)
$$

とする。

残した部分を次段階へ送る行列を

$$
\widehat A^{\langle1\rangle}
:=
\Sigma_1V_1^T
\in
\mathbb R^{r_1\times(n_2\cdots n_d)}
$$

と定義する。

捨てた部分を

$$
E_1^{\langle1\rangle}
:=
U_-\Sigma_-V_-^T
$$

とすれば、SVDを残す部分と捨てる部分に分けた恒等式として

$$
\boxed{
A^{\langle1\rangle}
=
U_1\widehat A^{\langle1\rangle}
+
E_1^{\langle1\rangle}.
}
\tag{3}
$$

が成り立つ。

この式は新しい定理ではない。SVDの和

$$
A^{\langle1\rangle}
=
\sum_{\ell=1}^{\rho}
\sigma_\ell u_\ell v_\ell^T
$$

を、$\ell\le r_1$ と $\ell>r_1$ に分け、それぞれに名前をつけただけである。

### 4.2 $B$ と縮約テンソル $\widehat A$

第1段階で保持したテンソルを $B$ と書くと、

$$
B^{\langle1\rangle}
=
U_1\widehat A^{\langle1\rangle}.
\tag{4}
$$

元の添字に戻すと、

$$
\boxed{
B(i_1,i_2,\ldots,i_d)
=
\sum_{\alpha_1=1}^{r_1}
U_1(i_1,\alpha_1)
\widehat A(\alpha_1,i_2,\ldots,i_d).
}
\tag{5}
$$

ここで $\alpha_1$ は

$$
\alpha_1=1,\ldots,r_1
$$

を取る新しい内部bond添字である。

式 (5) は、行列積 (4) の成分表示そのものである。第1unfoldingで列添字としてまとめられていた

$$
\mathbf j=(i_2,\ldots,i_d)
$$

を、元の複数添字に戻しただけである。

$$
B^{\langle1\rangle}(i_1,\mathbf j)
=
\sum_{\alpha_1=1}^{r_1}
U_1(i_1,\alpha_1)
\widehat A^{\langle1\rangle}(\alpha_1,\mathbf j).
$$

### 4.3 次段階と複合添字

$\widehat A$ は、軸を省略せずに書けば

$$
\widehat A
\in
\mathbb R^{r_1\times n_2\times\cdots\times n_d}
$$

という $d$ 本の軸を持つ配列である。ただし、次のTT-SVD問題では左bond添字 $\alpha_1$ と物理添字 $i_2$ を一つの複合添字

$$
j_2=(\alpha_1,i_2)
$$

にまとめる。そのため、再帰問題に渡すテンソルは

$$
\widehat A_2
\in
\mathbb R^{(r_1n_2)\times n_3\times\cdots\times n_d}
$$

とみなせる $(d-1)$ 階テンソルである。

次に

$$
M_2
:=
\operatorname{reshape}
\left(
\widehat A^{\langle1\rangle},
(r_1n_2)\times(n_3\cdots n_d)
\right)
$$

とreshapeしてSVDを取り、rank $r_2$ で打ち切る。reshapeは値と格納順を変えず、行と列にまとめる添字群だけを変える。同じ操作を左から右へ繰り返して、TTコア

$$
G_1,G_2,\ldots,G_d
$$

を作る。

TTコアのshapeは

$$
r_0=r_d=1,
\qquad
G_k\in\mathbb R^{r_{k-1}\times n_k\times r_k}.
$$

すなわち、

$$
G_1\in\mathbb R^{1\times n_1\times r_1},
$$

$$
G_k\in\mathbb R^{r_{k-1}\times n_k\times r_k}
\quad(2\le k\le d-1),
$$

$$
G_d\in\mathbb R^{r_{d-1}\times n_d\times1}.
$$

再構成は

$$
\widetilde A(i_1,\ldots,i_d)
=
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G_1(1,i_1,\alpha_1)
G_2(\alpha_1,i_2,\alpha_2)
\cdots
G_d(\alpha_{d-1},i_d,1).
$$

rank制限なしで各SVDの全成分を保持すれば、理論上情報は失われず、丸め誤差を除いて

$$
\|A-\widetilde A\|_F\approx0
$$

となる。

---

## 5. 第1打ち切り誤差と後続誤差の直交性

TT-SVD誤差上界の核心は、最初に捨てた成分と、その後の近似で生じる成分が直交することである。

### 5.1 捨てた残差は捨てた左特異空間にある

SVDの構造から、

$$
E_1^{\langle1\rangle}
=
U_-\Sigma_-V_-^T.
$$

一方、残した左特異ベクトルは $U_1$ であり、SVDの直交性より

$$
\boxed{
U_1^TU_-=0.
}
\tag{6}
$$

従って

$$
U_1^TE_1^{\langle1\rangle}
=
U_1^TU_-\Sigma_-V_-^T
=0.
\tag{7}
$$

つまり、$E_1$ の各列は捨てた左特異空間に入り、$\operatorname{col}(U_1)$ と直交する。

### 5.2 後続近似は残した空間の中にある

縮約テンソル $\widehat A$ を残りのTT-SVDで近似し、その結果を $\widehat T$ とする。

元の第1modeへ埋め戻した最終近似 $T$ は、

$$
T^{\langle1\rangle}
=
U_1\widehat T^{\langle1\rangle}
\tag{8}
$$

と作る。

保持テンソル $B$ は (4) より

$$
B^{\langle1\rangle}
=
U_1\widehat A^{\langle1\rangle}.
$$

よって、

$$
\boxed{
B^{\langle1\rangle}-T^{\langle1\rangle}
=
U_1
\left(
\widehat A^{\langle1\rangle}
-
\widehat T^{\langle1\rangle}
\right).
}
\tag{9}
$$

従って $B-T$ の各列は $U_1$ の列の線形結合であり、$B-T$ は $\operatorname{col}(U_1)$ の中にある。

### 5.3 Frobenius内積を実際に計算する

Frobenius内積は

$$
\langle X,Y\rangle_F
:=
\operatorname{tr}(X^TY).
$$

$$
A-B=E_1,
\qquad
B-T=U_1C
$$

と書く。ここで

$$
C:=
\widehat A^{\langle1\rangle}
-
\widehat T^{\langle1\rangle}.
$$

すると、

$$
\begin{aligned}
\langle A-B,B-T\rangle_F
&=
\langle E_1,U_1C\rangle_F\\
&=
\operatorname{tr}(E_1^TU_1C).
\end{aligned}
$$

(7) の転置より

$$
E_1^TU_1=(U_1^TE_1)^T=0.
$$

よって、

$$
\boxed{
\langle A-B,B-T\rangle_F=0.
}
\tag{10}
$$

### 5.4 Pythagoras型分解

単なる足し引きの恒等式として

$$
A-T=(A-B)+(B-T).
$$

一般に、

$$
\|P+Q\|_F^2
=
\|P\|_F^2+
\|Q\|_F^2+2\langle P,Q\rangle_F.
$$

これに

$$
P=A-B,
\qquad
Q=B-T
$$

を代入し、(10)を使うと、

$$
\boxed{
\|A-T\|_F^2
=
\|A-B\|_F^2+
\|B-T\|_F^2.
}
\tag{11}
$$

第1cutでの最良rank-$r_1$誤差の定義から

$$
\|A-B\|_F=\varepsilon_1.
$$

したがって、

$$
\boxed{
\|A-T\|_F^2
=
\varepsilon_1^2+
\|B-T\|_F^2.
}
\tag{12}
$$

これは、最初に捨てた誤差と、残りの小さいテンソルの近似誤差を二乗和として分ける式である。

---

## 6. 列直交行列による埋め込みと縮約

この回で最も重要な線形代数の区別の一つが、列直交行列 $U$ を左から掛ける場合と、$U^T$ を左から掛ける場合の違いである。

$$
U\in\mathbb R^{m\times r},
\qquad
U^TU=I_r.
$$

### 6.1 $U$ を左から掛ける：等長埋め込み

任意の

$$
Z\in\mathbb R^{r\times p}
$$

に対して、

$$
\begin{aligned}
\|UZ\|_F^2
&=
\operatorname{tr}((UZ)^T(UZ))\\
&=
\operatorname{tr}(Z^TU^TUZ)\\
&=
\operatorname{tr}(Z^TZ)\\
&=
\|Z\|_F^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|UZ\|_F=\|Z\|_F.
}
\tag{13}
$$

これは $U$ が $r$ 次元空間を $m$ 次元空間へ等長に埋め込む写像だからである。

式 (9) に適用すると、

$$
\|B-T\|_F
=
\|\widehat A-\widehat T\|_F.
\tag{14}
$$

従って (12) は

$$
\boxed{
\|A-T\|_F^2
=
\varepsilon_1^2
+
\|\widehat A-\widehat T\|_F^2.
}
\tag{15}
$$

と書ける。

### 6.2 $U^T$ を左から掛ける：縮約であり、ノルムは非増大

任意の

$$
Y\in\mathbb R^{m\times p}
$$

に対して、一般には

$$
\boxed{
\|U^TY\|_F\le\|Y\|_F.
}
\tag{16}
$$

これは $U^T$ が、$m$ 次元空間の情報を $\operatorname{col}(U)$ の $r$ 次元座標へ縮約するためである。

$$
P:=UU^T
$$

とおくと、$P$ は $\operatorname{col}(U)$ への直交射影である。

$$
P^T=P,
\qquad
P^2=U(U^TU)U^T=UU^T=P.
$$

また、

$$
\begin{aligned}
\|U^TY\|_F^2
&=
\operatorname{tr}((U^TY)^T(U^TY))\\
&=
\operatorname{tr}(Y^TUU^TY)\\
&=
\|UU^TY\|_F^2\\
&=
\|PY\|_F^2\\
&\le
\|Y\|_F^2.
\end{aligned}
$$

最後の不等号は、直交射影が直交補空間の成分を落とし、ノルムを増やさないためである。

ベクトル $x$ なら、

$$
x=Px+(I-P)x,
\qquad
Px\perp(I-P)x,
$$

なので

$$
\|x\|_2^2
=
\|Px\|_2^2+
\|(I-P)x\|_2^2
\ge
\|Px\|_2^2.
$$

行列についても各列にこの議論を適用してFrobeniusノルムの二乗を足せばよい。

### 6.3 等号になる条件

$$
\|U^TY\|_F=\|Y\|_F
$$

となるのは、$Y$ の各列がすべて $\operatorname{col}(U)$ に入るとき、すなわち

$$
UU^TY=Y
$$

のときに限る。

一般には $Y$ が $\operatorname{col}(U)$ と直交する成分を含むので、その成分は $U^T$ により失われ、厳密不等号になりうる。

小さい例として

$$
U=
\begin{pmatrix}
1\\0
\end{pmatrix},
\qquad
U^T=
\begin{pmatrix}
1&0
\end{pmatrix},
\qquad
x=
\begin{pmatrix}
3\\4
\end{pmatrix}
$$

では、

$$
\|x\|_2=5,
\qquad
\|U^Tx\|_2=3.
$$

第2成分は $\operatorname{col}(U)$ に直交するため、縮約で消える。

---

## 7. 縮約後の残りcutの最良誤差はなぜ増えないか

帰納法を成立させる核心は、縮約後の $\widehat A$ の残りcutにおける最良誤差を、元の $A$ の対応cutの最良誤差で上から抑えることである。

### 7.1 cut番号の対応

第1modeを縮約した後のテンソルは

$$
\widehat A
\in
\mathbb R^{r_1\times n_2\times\cdots\times n_d}
$$

であり、モードは

$$
(\alpha_1,i_2,\ldots,i_d)
$$

である。

次段階では $(\alpha_1,i_2)$ を一つの先頭modeとみなす。そのため、元のテンソル $A$ のcut $k$ に対応するcutは、$\widehat A$ 内部では $k-1$ 番目のcutになる。その行添字は

$$
(\alpha_1,i_2,\ldots,i_k)
$$

である。

以下では、元の $A$ のcut番号との対応を保つため、$k=2,\ldots,d-1$ に対して、縮約テンソルの内部cut $k-1$ での最良rank-$r_k$誤差を $\widehat\varepsilon_k$ と書く。

### 7.2 最初に出た疑問

次のように言いたくなる。

> $U_1^T$ はノルムを増やさない。だから $\widehat\varepsilon_k\le\varepsilon_k$ である。

しかし、これはそのままでは二つの意味で飛躍である。

1. $k\ge2$ では、$U_1^T$ だけでは複合行添字 $(i_1,\ldots,i_k)$ に作用できない。
2. 最良誤差 $\widehat\varepsilon_k$ と縮約後の誤差を比較するには、**縮約後の問題で許されるrank-$r_k$候補を実際に一つ作る**必要がある。

### 7.3 複合添字に作用する縮約行列

$k=2,\ldots,d-1$ に対して

$$
N_{2:k}:=\prod_{j=2}^{k}n_j
$$

と置く。元のunfoldingと縮約後のunfoldingのshapeは

$$
A^{\langle k\rangle}
\in
\mathbb R^{(n_1N_{2:k})\times(n_{k+1}\cdots n_d)},
$$

$$
\widehat A^{\langle k-1\rangle}
\in
\mathbb R^{(r_1N_{2:k})\times(n_{k+1}\cdots n_d)}
$$

である。$U_1^T\in\mathbb R^{r_1\times n_1}$ は $i_1$ に作用し、$i_2,\ldots,i_k$ には作用しない。そこで、縮約行列を

$$
\boxed{
C_k
:=
U_1^T\otimes I_{N_{2:k}}
\in
\mathbb R^{(r_1N_{2:k})\times(n_1N_{2:k})}
}
\tag{17}
$$

と定義する。PyTorchのC-orderと同じく、最後の添字が速く変化する複合添字順では、

$$
\boxed{
\widehat A^{\langle k-1\rangle}
=
C_kA^{\langle k\rangle}
}
\tag{18}
$$

である。$U_1^TU_1=I_{r_1}$ だから、

$$
\begin{aligned}
C_kC_k^T
&=
(U_1^T\otimes I_{N_{2:k}})
(U_1\otimes I_{N_{2:k}})\\
&=
(U_1^TU_1)\otimes I_{N_{2:k}}\\
&=
I_{r_1N_{2:k}}.
\end{aligned}
$$

したがって $C_k$ は行直交であり、任意の適合する行列 $Y$ に対して

$$
\|C_kY\|_F\le\|Y\|_F
$$

が成り立つ。

### 7.4 $d=3$、$n_1=n_2=n_3=2$ の全要素表示

$r_1=1$ とし、第1段階で残した列特異ベクトルを

$$
U_1=
\begin{pmatrix}
u_1\\
u_2
\end{pmatrix},
\qquad
u_1^2+u_2^2=1
$$

とする。行添字を

$$
(i_1,i_2)
=
(1,1),(1,2),(2,1),(2,2)
$$

の順に並べると、

$$
C_2
=
U_1^T\otimes I_2
=
\begin{pmatrix}
u_1&0&u_2&0\\
0&u_1&0&u_2
\end{pmatrix}.
$$

元のcut 2 unfoldingを全要素で書けば、

$$
A^{\langle2\rangle}
=
\begin{pmatrix}
A_{111}&A_{112}\\
A_{121}&A_{122}\\
A_{211}&A_{212}\\
A_{221}&A_{222}
\end{pmatrix}.
$$

したがって、縮約後のcut 1 unfoldingは

$$
\begin{aligned}
\widehat A^{\langle1\rangle}
&=
C_2A^{\langle2\rangle}\\
&=
\begin{pmatrix}
u_1A_{111}+u_2A_{211}
&
u_1A_{112}+u_2A_{212}\\
u_1A_{121}+u_2A_{221}
&
u_1A_{122}+u_2A_{222}
\end{pmatrix}.
\end{aligned}
$$

第1行は $(\alpha_1,i_2)=(1,1)$、第2行は $(\alpha_1,i_2)=(1,2)$ に対応する。$U_1^T$ が $i_1$ 方向だけを縮約し、$I_2$ が $i_2$ の並びを保つことが、行列の全要素から確認できる。

### 7.5 元の最良近似から候補を作る

元の $A$ のcut $k$ に対する最良rank-$r_k$近似を

$$
A^{\langle k\rangle}=B_k+E_k
$$

とする。

$$
\operatorname{rank}(B_k)\le r_k,
\qquad
\|E_k\|_F=\varepsilon_k.
\tag{19}
$$

式 (18) に代入すると、

$$
\widehat A^{\langle k-1\rangle}
=
C_kB_k+C_kE_k.
\tag{20}
$$

左から行列を掛けてもrankは増えない。

$$
\operatorname{rank}(C_kB_k)
\le
\operatorname{rank}(B_k)
\le r_k.
\tag{21}
$$

したがって

$$
C_kB_k
$$

は、$\widehat A^{\langle k-1\rangle}$ のrank-$r_k$近似として許される候補である。

### 7.6 二つの不等号を分ける

縮約後テンソルの最良誤差を

$$
\widehat\varepsilon_k
=
\min_{\operatorname{rank}(\widehat B)\le r_k}
\left\|
\widehat A^{\langle k-1\rangle}
-
\widehat B
\right\|_F
$$

とする。

候補 $\widehat B=C_kB_k$ を代入すると、

$$
\begin{aligned}
\widehat\varepsilon_k
&\le
\left\|
\widehat A^{\langle k-1\rangle}
-C_kB_k
\right\|_F\\
&=
\|C_kE_k\|_F\\
&\le
\|E_k\|_F\\
&=
\varepsilon_k.
\end{aligned}
$$

よって、

$$
\boxed{
\widehat\varepsilon_k\le\varepsilon_k,
\qquad
k=2,\ldots,d-1.
}
\tag{22}
$$

この鎖には性質の異なる二つの不等号がある。

1. 最初の不等式

   $$
   \widehat\varepsilon_k
   \le
   \left\|
   \widehat A^{\langle k-1\rangle}
   -C_kB_k
   \right\|_F
   $$

   は、**最良誤差 $\le$ 許される候補の誤差**という最小化問題の定義から従う。

2. 最後の不等式

   $$
   \|C_kE_k\|_F\le\|E_k\|_F
   $$

   は、$C_k=U_1^T\otimes I_{N_{2:k}}$ が行直交な縮約であり、Frobeniusノルムを増やさないことから従う。

この二段階を分けることが重要である。

---

## 8. 帰納法によるTT-SVD基本誤差上界

### 8.1 基底：$d=2$

$d=2$ ではcutは1つだけであり、TT-SVDは行列 $A^{\langle1\rangle}$ のtruncated SVDそのものである。したがってEYM定理から、

$$
\|A-T\|_F^2
=
\delta_1^2
=
\varepsilon_1^2.
$$

これが帰納法の基底である。

### 8.2 帰納ステップ

式 (3) で得た $\widehat A$ の $(\alpha_1,i_2)$ を複合添字にまとめ、$(d-1)$ 階テンソルとして標準TT-SVDを再帰的に適用する。目標rankは

$$
(r_2,\ldots,r_{d-1})
$$

である。再帰的なTT-SVDの出力を $\widehat T$、各段階で実際に捨てる局所誤差を $\delta_2,\ldots,\delta_{d-1}$ とする。帰納法の仮定により、

$$
\|\widehat A-\widehat T\|_F^2
=
\sum_{k=2}^{d-1}\delta_k^2
\le
\sum_{k=2}^{d-1}\widehat\varepsilon_k^2.
\tag{23}
$$

また、式 (22) より $\widehat\varepsilon_k\le\varepsilon_k$ なので、

$$
\sum_{k=2}^{d-1}
\widehat\varepsilon_k^2
\le
\sum_{k=2}^{d-1}
\varepsilon_k^2.
\tag{24}
$$

### 8.3 元のテンソルへ戻す

$\widehat T$ を使って最終TT近似 $T$ を

$$
T^{\langle1\rangle}
=
U_1\widehat T^{\langle1\rangle}
$$

と構成する。

ここで式 (15)、$\delta_1=\varepsilon_1$、帰納法の仮定を使うと、

$$
\begin{aligned}
\|A-T\|_F^2
&=
\delta_1^2+
\|\widehat A-\widehat T\|_F^2\\
&=
\sum_{k=1}^{d-1}\delta_k^2\\
&\le
\varepsilon_1^2+
\sum_{k=2}^{d-1}\widehat\varepsilon_k^2\\
&\le
\varepsilon_1^2+
\sum_{k=2}^{d-1}\varepsilon_k^2\\
&=
\sum_{k=1}^{d-1}\varepsilon_k^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|A-T\|_F^2
\le
\sum_{k=1}^{d-1}\varepsilon_k^2.
}
\tag{25}
$$

平方根を取れば、

$$
\boxed{
\|A-T\|_F
\le
\left(
\sum_{k=1}^{d-1}\varepsilon_k^2
\right)^{1/2}.
}
\tag{26}
$$

ここで、等号

$$
\boxed{
\|A-T\|_F^2
=
\sum_{k=1}^{d-1}\delta_k^2
}
$$

は、実際の局所打ち切り誤差が直交するために得られる厳密な誤差合成則である。その上で、$\delta_k\le\varepsilon_k$ を使った不等式が標準TT-SVDの基本誤差上界である。

### 8.4 一般の第 $k$ 段階で $\delta_k\le\varepsilon_k$ となる理由

第 $k-1$ コアまでを縮約した左ブロック行列を

$$
Q_{k-1}
\in
\mathbb R^{(n_1\cdots n_{k-1})\times r_{k-1}},
\qquad
Q_{k-1}^TQ_{k-1}=I_{r_{k-1}}
$$

とする。第 $k$ 段階の左展開行列に作用する等長埋め込みと縮約を

$$
F_k:=Q_{k-1}\otimes I_{n_k},
\qquad
F_k^T=Q_{k-1}^T\otimes I_{n_k}
$$

と置くと、TT-SVDが第 $k$ 段階でSVDする行列は

$$
M_k=F_k^TA^{\langle k\rangle}
$$

と書ける。元の $A^{\langle k\rangle}$ の最良rank-$r_k$近似を $B_k^\ast$ とすると、$F_k^TB_k^\ast$ は $M_k$ に対するrank-$r_k$以下の許される候補である。したがって、

$$
\begin{aligned}
\delta_k
&=
\min_{\operatorname{rank}(Z)\le r_k}
\|M_k-Z\|_F\\
&\le
\|M_k-F_k^TB_k^\ast\|_F\\
&=
\|F_k^T(A^{\langle k\rangle}-B_k^\ast)\|_F\\
&\le
\|A^{\langle k\rangle}-B_k^\ast\|_F\\
&=
\varepsilon_k.
\end{aligned}
$$

$k=1$ では $F_1=I_{n_1}$ なので $\delta_1=\varepsilon_1$ である。$k\ge2$ では縮約によりノルムが小さくなりうるため、一般には等号とは限らず $\delta_k\le\varepsilon_k$ である。

### 8.5 この証明で実際に使った構造

各段階で同じことを繰り返している。

1. 最初のunfoldingをSVDし、rank制限して保持部分と捨てた残差へ分ける
2. 捨てた残差は、保持した左特異空間と直交する
3. 保持部分の内部テンソルを、次元を1つ減らして近似する
4. 後続近似を保持空間に埋め戻す
5. 直交性により前段の誤差と後続誤差を二乗和として加える
6. 縮約後の各最良誤差は、元の対応cutの最良誤差以下である

この構造により、複数cutにわたる逐次構成であっても、元のunfoldingの尾部特異値から全体誤差を評価できる。

---

## 9. 大域最適TT近似との比較

TT-rank $\le(r_1,\ldots,r_{d-1})$ のテンソル全体の中で、$A$ に最も近いものを $A_\ast$ とする。

$$
\|A-A_\ast\|_F
=
\min_{\operatorname{TT-rank}(Y)\le(r_1,\ldots,r_{d-1})}
\|A-Y\|_F.
$$

以後、

$$
e_\ast:=\|A-A_\ast\|_F
$$

と書く。

### 9.1 なぜ $\varepsilon_k\le e_\ast$ か

$A_\ast$ は指定TT-rank以下である。TT-rankの定義から、各cut $k$ で

$$
\operatorname{rank}
\left(A_\ast^{\langle k\rangle}\right)
\le r_k.
$$

従って $A_\ast^{\langle k\rangle}$ は、$\varepsilon_k$ を定義する行列最適化問題

$$
\min_{\operatorname{rank}(B)\le r_k}
\left\|A^{\langle k\rangle}-B\right\|_F
$$

の許される候補の一つである。

最小値は任意の候補での値以下なので、

$$
\begin{aligned}
\varepsilon_k
&=
\min_{\operatorname{rank}(B)\le r_k}
\left\|A^{\langle k\rangle}-B\right\|_F\\
&\le
\left\|A^{\langle k\rangle}
-A_\ast^{\langle k\rangle}\right\|_F\\
&=
\|A-A_\ast\|_F\\
&=e_\ast.
\end{aligned}
$$

すなわち、

$$
\boxed{
\varepsilon_k\le e_\ast
\qquad(k=1,\ldots,d-1).
}
\tag{27}
$$

ここで不等号の向きは $\le$ で正しい。

理由は、$\varepsilon_k$ の最小化では「第 $k$ cutでrank $\le r_k$」という一つの行列ランク制約だけを課すのに対し、$A_\ast$ を求める最適化では全cutのTT-rank制約を同時に満たさなければならないからである。単一cutの行列近似の候補集合の方が広く、最小値はより小さくなれる。

### 9.2 準最適性

(25) と (27) を結ぶ。

$$
\begin{aligned}
\|A-T\|_F^2
&\le
\sum_{k=1}^{d-1}\varepsilon_k^2\\
&\le
\sum_{k=1}^{d-1}e_\ast^2\\
&=
(d-1)e_\ast^2.
\end{aligned}
$$

平方根を取ると、

$$
\boxed{
\|A-T\|_F
\le
\sqrt{d-1}\,e_\ast
=
\sqrt{d-1}\,\|A-A_\ast\|_F.
}
\tag{28}
$$

これがTT-SVDの準最適性である。

### 9.3 なぜ係数が $\sqrt{d-1}$ か

各cutについて

$$
\varepsilon_k\le e_\ast
$$

なので、二乗して全 $d-1$ 個を足すと、

$$
\sum_{k=1}^{d-1}\varepsilon_k^2
\le
(d-1)e_\ast^2.
$$

その平方根が

$$
\sqrt{d-1}\,e_\ast
$$

である。

$\sqrt{d-1}$ は典型的な実測比を表す値ではなく、最悪の場合にも超えないことを保証する係数である。すべての $\varepsilon_k$ が同時に $e_\ast$ と等しいという特殊な場合に上限へ張り付く。

### 9.4 ここで主張していないこと

次の主張はしていない。

$$
T=A_\ast.
$$

また、

> TT-SVDで得たTTは、任意の別構成TTより常に誤差が小さい。

とも主張していない。

主張はあくまで、

$$
\|A-T\|_F
\le
\sqrt{d-1}\,\|A-A_\ast\|_F
$$

という保証である。

---

## 10. よくある混同と、採用しない議論

この節では、似た式を同一視したときに生じる誤りを整理する。

### 10.1 混同してはいけない誤差量

次の三者は異なる。

| 記号・量 | 見ている対象 | 意味 |
|---|---|---|
| $\varepsilon_k$ | 元の $A^{\langle k\rangle}$ | 最良rank-$r_k$行列近似誤差 |
| $\delta_k$ | TT-SVDの中間行列 $M_k$ | 第 $k$ 段階で実際に捨てた特異値の尾部ノルム |
| $\|A-T\|_F$ | 元テンソルと最終TT | TT-SVDの最終実測誤差 |

このため、次を無条件に書いてはいけない。

$$
\delta_k=\varepsilon_k.
$$

一方、このノートで扱う標準の左から右へのTT-SVDでは、第8.4節で示したように

$$
\boxed{
\delta_1=\varepsilon_1,
\qquad
\delta_k\le\varepsilon_k
\quad(2\le k\le d-1)
}
$$

が成り立つ。したがって、「定義対象が異なるので等しいとは限らない」ことと、「標準TT-SVDでは大小関係を証明できる」ことを分ける必要がある。TT-roundingや異なる掃き方向のアルゴリズムへ、この不等式を証明なしに流用してはいけない。

### 10.2 誤った射影列の考え方

元の各unfoldingの上位特異空間への射影を $P_k$ と作り、それらを順にかけたテンソル $Y$ について、

$$
\|A-T\|_F\le\|A-Y\|_F
$$

と主張するのは誤りである。

TT-SVDは大域最適法ではないため、「TT-SVDは任意の別構成より必ず良い」とは言えない。また、元の各cutから独立に作った射影は一般に可換でも入れ子でもなく、順にかけたときの誤差を単純に $\sum_k\varepsilon_k^2$ で評価することもできない。

### 10.3 使ってはいけない因数分解

前段の打ち切り後に、元のunfoldingを

$$
A^{\langle k\rangle}
=
(Q_{k-1}\otimes I_{n_k})M_k
$$

と完全に復元できるかのように書き、元の $A^{\langle k\rangle}$ と途中行列 $M_k$ の特異値が同じだと扱うのは誤りである。左辺には、前段階で捨てた直交補空間成分も含まれるからである。

正しい関係は逆向きの縮約

$$
M_k
=
(Q_{k-1}^T\otimes I_{n_k})A^{\langle k\rangle}
$$

である。この縮約はノルムを増やさない。そのため、元の最良rank-$r_k$近似を縮約後の候補として使うことで、$\delta_k\le\varepsilon_k$ を導く。

### 10.4 再利用できる正しい議論

準最適性の証明は、次の線形代数の部品を順番に接続している。

1. SVD打ち切りの「残す部分＋捨てる部分」分解
   $$
   M=U_+\Sigma_+V_+^T+U_-\Sigma_-V_-^T.
   $$

2. 捨てた左特異空間と、保持空間へ埋め込まれた後続誤差の直交性

3. 列直交行列 $Q$ による埋め込みのノルム保存
   $$
   \|QZ\|_F=\|Z\|_F.
   $$

4. 転置 $Q^T$ による縮約のノルム非増大
   $$
   \|Q^TY\|_F\le\|Y\|_F.
   $$

5. 「最良値 $\le$ 許される候補の値」という最小化の基本原理

6. 大域最適TT近似を各unfoldingの行列問題の候補として使う議論

これらを正しい順番で接続したものが、上のTT-SVD準最適性証明である。

---

## 11. $d=3$ での見取り図

$d=3$ ではcutは2つである。

$$
A\in\mathbb R^{n_1\times n_2\times n_3}.
$$

目標rankは $(r_1,r_2)$ である。

第1unfoldingは

$$
A^{\langle1\rangle}
\in
\mathbb R^{n_1\times(n_2n_3)}.
$$

第1段階のSVD打ち切りにより

$$
A^{\langle1\rangle}
=
U_1\widehat A^{\langle1\rangle}+E_1^{\langle1\rangle}.
$$

$\widehat A$ は

$$
\widehat A\in\mathbb R^{r_1\times n_2\times n_3}
$$

であり、複合添字 $(\alpha_1,i_2)$ を行にした

$$
M_2
\in
\mathbb R^{(r_1n_2)\times n_3}
$$

にreshapeして第2段階のSVDを行う。第7.4節の $n_1=n_2=n_3=2$、$r_1=1$ の例では、

$$
M_2
=
(U_1^T\otimes I_2)A^{\langle2\rangle}
\in
\mathbb R^{2\times2}
$$

であり、その4要素は第7.4節で書いた $u_1A_{1i_2i_3}+u_2A_{2i_2i_3}$ である。

後続近似の誤差は $U_1$ の列空間に埋め込まれ、第1段階で捨てた $E_1$ は直交補空間にある。そのため

$$
\|A-T\|_F^2
=
\delta_1^2+
\delta_2^2.
$$

ここで、

$$
\delta_1=\varepsilon_1,
\qquad
\delta_2\le\varepsilon_2
$$

であるから、

$$
\|A-T\|_F^2
\le
\varepsilon_1^2+
\varepsilon_2^2.
$$

大域最適TT近似 $A_\ast$ と比較すれば

$$
\|A-T\|_F
\le
\sqrt2\,\|A-A_\ast\|_F.
$$

---

## 12. 数値検証との対応

PyTorchの操作、コード、shape、実行値は [[40_TensorTrain_MPS/07_TT-SVDの準最適性と誤差上界のPyTorch確認]] に分離する。手を動かす演習は [Notebook 11](../../../../notebooks/30_tt_mps/00_fundamentals/11_tt_svd_quasi_optimality_and_error_bound.ipynb) を参照する。

数値検証では、次を確認した。

1. 指定特異値 $(5,2,0.5,0.1)$ を持つ行列をrank 2へ打ち切ると、二乗誤差は $0.5^2+0.1^2=0.26$ となる。
2. rank制限なしTT-SVDの再構成誤差は $3.47\times10^{-14}$ で、float64の丸め誤差水準だった。
3. 固定TT-rank $(2,4,2)$ では、実測誤差 $13.6271$ が、元のunfoldingから求めた上界 $18.1552$ 以下だった。
4. 列直交行列による埋め込みはノルムを保存し、その転置による縮約はノルムを増やさないことを確認した。

3は基本誤差上界の検証である。大域最適TT近似 $A_\ast$ を数値的に求めていないため、$\sqrt{d-1}$ 準最適性そのものを直接検証したものではない。

---

## 13. 最終まとめ

### 行列のEYM

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|M-B\|_F^2
=
\sum_{\ell>r}\sigma_\ell(M)^2.
}
$$

### TT-SVDで使う局所量

$$
\boxed{
\varepsilon_k
=
\min_{\operatorname{rank}(B)\le r_k}
\|A^{\langle k\rangle}-B\|_F.
}
$$

これは元の $A$ の第 $k$ unfoldingに対する最良行列近似誤差である。一方、標準TT-SVDの実際の局所打ち切り誤差 $\delta_k$ との関係は

$$
\boxed{
\delta_1=\varepsilon_1,
\qquad
\delta_k\le\varepsilon_k
\quad(2\le k\le d-1)
}
$$

である。

### TT-SVD基本誤差上界

$$
\boxed{
\|A-T\|_F^2
=
\sum_{k=1}^{d-1}\delta_k^2
\le
\sum_{k=1}^{d-1}\varepsilon_k^2.
}
$$

### 大域最適TT近似との比較

$$
\boxed{
\varepsilon_k
\le
\|A-A_\ast\|_F.
}
$$

### 準最適性

$$
\boxed{
\|A-T\|_F
\le
\sqrt{d-1}\,\|A-A_\ast\|_F.
}
$$
