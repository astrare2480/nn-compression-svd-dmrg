---
title: TT・MPSの単一ボンドSVD打ち切り
tags:
  - TT
  - MPS
  - SVD
  - Schmidt
  - truncation
---

# TT・MPSの単一ボンドSVD打ち切り

[[43_TT_MPSのSVD中心移動とSchmidt形]] で得た、3階TT/MPSの第2サイトから第3サイトへの**打ち切りなし**の中心移動を出発点にする。このノートでは切断 $(i_1,i_2)\mid i_3$ の**一つのボンドを一回だけ**打ち切る。複数の切断を順に処理するTT-SVD全体の誤差は [[33_TT-SVDの打ち切りと誤差]]、有効ブロック状態の物理的意味は [[00_基礎理論/07_物理基礎/44_Schmidt打ち切りとボンド状態の物理的意味]] に置く。

## 1. 出発点：中心行列と全テンソルの切断行列

第2サイトが直交中心である実数の3階TTを考える。

$$
X(i_1,i_2,i_3)
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[C]}(\alpha_1,i_2,\alpha_2)
G_3^{[R]}(\alpha_2,i_3,1).
$$

第1コアの左展開を $H_{i_1,\alpha_1}=G_1^{[L]}(1,i_1,\alpha_1)$、第3コアの右展開を $R_{\alpha_2,i_3}=G_3^{[R]}(\alpha_2,i_3,1)$ と置く。混合正準条件は

$$
H\in\mathbb R^{n_1\times r_1},\quad H^TH=I_{r_1},
\qquad
R\in\mathbb R^{r_2\times n_3},\quad RR^T=I_{r_2}
$$

である。中心の左展開は

$$
A_{(\alpha_1,i_2),\alpha_2}
:=G_2^{[C]}(\alpha_1,i_2,\alpha_2),
\qquad
A\in\mathbb R^{(r_1n_2)\times r_2}.
$$

行の順を $p=(\alpha_1-1)n_2+i_2$ と固定する。同じ順に全テンソルをcut unfoldingすると、$q=(i_1-1)n_2+i_2$ が行位置であり、

$$
\begin{aligned}
X^{\langle2\rangle}_{(i_1,i_2),i_3}
&=\sum_{\alpha_1,\alpha_2}
H_{i_1,\alpha_1}
A_{(\alpha_1,i_2),\alpha_2}
R_{\alpha_2,i_3},\\
X^{\langle2\rangle}
&=(H\otimes I_{n_2})AR.
\end{aligned}
$$

ここで $A$ の行は $(\alpha_1,i_2)$、$X^{\langle2\rangle}$ の行は $(i_1,i_2)$ である。両者を同じ行列とみなしてはいけない。$T:=H\otimes I_{n_2}$ は $T^TT=I_{r_1n_2}$ を満たす。

## 2. exact SVD と truncated SVD

$A\ne0$、$\rho=\operatorname{rank}(A)\le\min(r_1n_2,r_2)$ とし、非零特異値だけを使うcompact SVDを

$$
\begin{aligned}
A&=U\Sigma V^T
=\sum_{\beta=1}^{\rho}\sigma_\beta u_\beta v_\beta^T,\\
U&\in\mathbb R^{(r_1n_2)\times\rho},\quad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_\rho)
\in\mathbb R^{\rho\times\rho},\\
V^T&\in\mathbb R^{\rho\times r_2},\quad
U^TU=V^TV=I_\rho,\quad
\sigma_1\ge\cdots\ge\sigma_\rho>0
\end{aligned}
$$

と書く。すべての非零特異値を残すので $U\Sigma V^T=A$ は厳密な等式である。$\rho<r_2$ で零特異方向を省いても誤差は0である。この場合はボンドの表示サイズが縮むが、同じサイズの可逆行列を挿入する意味でのGauge変換ではない。

$1\le k<\rho$ として先頭 $k$ 本だけを残すと、

$$
\begin{aligned}
U_k&=[u_1,\ldots,u_k]
\in\mathbb R^{(r_1n_2)\times k},\\
\Sigma_k&=\operatorname{diag}(\sigma_1,\ldots,\sigma_k)
\in\mathbb R^{k\times k},\\
V_k^T&=\begin{pmatrix}v_1^T\\ \vdots\\ v_k^T\end{pmatrix}
\in\mathbb R^{k\times r_2},\\
\widetilde A_k&=U_k\Sigma_kV_k^T
=\sum_{\beta=1}^{k}\sigma_\beta u_\beta v_\beta^T.
\end{aligned}
$$

同じ行列を元のshapeのまま見るなら、$\Sigma$ の後ろの非零成分を0へ置き換えた

$$
\Sigma_k^{(0)}
:=\operatorname{diag}(\sigma_1,\ldots,\sigma_k,0,\ldots,0)
\in\mathbb R^{\rho\times\rho},
\qquad
U\Sigma_k^{(0)}V^T=U_k\Sigma_kV_k^T
$$

である。**特異値を0にする**のは捨てる成分を見せる表記、**列・行を削って $k$ 次元へ縮める**のはコアへ戻すときのcompactな表記で、どちらも同じ近似行列を表す。

捨てた行列を引き算で明示すると

$$
A-\widetilde A_k
=\sum_{\beta=1}^{\rho}\sigma_\beta u_\beta v_\beta^T
-\sum_{\beta=1}^{k}\sigma_\beta u_\beta v_\beta^T
=\sum_{\beta=k+1}^{\rho}\sigma_\beta u_\beta v_\beta^T.
$$

$\sigma_{k+1}>0$ なので $\widetilde A_k\ne A$ である。単に$\Sigma$の零成分を表記から除くexact SVDと、非零成分を削除するtruncated SVDの違いはここにある。

## 3. ボンドrankとコアの全添字

exact SVDによる中心移動では

$$
\begin{aligned}
G_2^{[L]}(\alpha_1,i_2,\beta)
&=U_{(\alpha_1,i_2),\beta},\\
G_3^{[C]}(\beta,i_3,1)
&=\sum_{\alpha_2=1}^{r_2}
(\Sigma V^T)_{\beta,\alpha_2}
G_3^{[R]}(\alpha_2,i_3,1)
\end{aligned}
$$

とし、$G_2^{[L]}$ のshapeは $(r_1,n_2,\rho)$、$G_3^{[C]}$ のshapeは $(\rho,n_3,1)$ になる。truncationでは同じ式の $\beta$ を $1,\ldots,k$ に制限する。

$$
\begin{aligned}
\widetilde G_2^{[L]}(\alpha_1,i_2,\beta)
&=(U_k)_{(\alpha_1,i_2),\beta},
&&\widetilde G_2^{[L]}\in\mathbb R^{r_1\times n_2\times k},\\
\widetilde G_3^{[C]}(\beta,i_3,1)
&=\sum_{\alpha_2=1}^{r_2}
(\Sigma_kV_k^T)_{\beta,\alpha_2}
G_3^{[R]}(\alpha_2,i_3,1),
&&\widetilde G_3^{[C]}\in\mathbb R^{k\times n_3\times1}.
\end{aligned}
$$

第1コアは $(1,n_1,r_1)$ のままで、今回の一回の操作では第1–第2ボンドの $r_1$ は変わらない。全要素の式では

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1=1}^{r_1}\sum_{\beta=1}^{\rho}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta)
G_3^{[C]}(\beta,i_3,1),\\
\widetilde X_k(i_1,i_2,i_3)
&=\sum_{\alpha_1=1}^{r_1}\sum_{\beta=1}^{k}
G_1^{[L]}(1,i_1,\alpha_1)
\widetilde G_2^{[L]}(\alpha_1,i_2,\beta)
\widetilde G_3^{[C]}(\beta,i_3,1).
\end{aligned}
$$

旧ボンド $\alpha_2$ は第3コアを作る縮約の中で消え、新しいボンド $\beta$ の範囲が $\rho\to k$ になる。$\rho$ は**compactにしたexact表現のSchmidt rank**である。元のコアの表示ボンド $r_2$ は、冗長な場合には $\rho$ より大きくてもよい。

## 4. Schmidt状態をどのように削るか

第2コアのSVDから、全テンソルの左右の係数を

$$
\begin{aligned}
L_\beta(i_1,i_2)
&=\sum_{\alpha_1=1}^{r_1}
H_{i_1,\alpha_1}U_{(\alpha_1,i_2),\beta},\\
\mathcal R_\beta(i_3)
&=\sum_{\alpha_2=1}^{r_2}
(V^T)_{\beta,\alpha_2}R_{\alpha_2,i_3}
\end{aligned}
$$

と定める。$L=(H\otimes I_{n_2})U$ は列直交、$\mathcal R=V^TR$ は行直交である。従って

$$
X^{\langle2\rangle}
=L\Sigma\mathcal R
=\sum_{\beta=1}^{\rho}\sigma_\beta L_{:,\beta}\mathcal R_{\beta,:},
\qquad
\widetilde X_k^{\langle2\rangle}
=L_k\Sigma_k\mathcal R_k
=\sum_{\beta=1}^{k}\sigma_\beta L_{:,\beta}\mathcal R_{\beta,:}.
$$

物理基底で $|L_\beta\rangle=\sum_{i_1,i_2}L_\beta(i_1,i_2)|i_1,i_2\rangle$、$|R_\beta\rangle=\sum_{i_3}\mathcal R_\beta(i_3)|i_3\rangle$ と置けば

$$
|X\rangle
=\sum_{\beta=1}^{\rho}\sigma_\beta
|L_\beta\rangle\otimes|R_\beta\rangle,
\qquad
|\widetilde X_k\rangle
=\sum_{\beta=1}^{k}\sigma_\beta
|L_\beta\rangle\otimes|R_\beta\rangle.
$$

これが一本のボンドを通る独立なSchmidt成分を $\rho$ 本から $k$ 本に減らす意味である。$|L_\beta\rangle$ や $|R_\beta\rangle$ はそれぞれ一つのサイトの固定配置ではなく、各ブロック上の正規直交状態である。$\beta$ ごとに左右が一対一になるのは、係数行列が $\Sigma_{\beta\gamma}=\sigma_\beta\delta_{\beta\gamma}$ と対角化されているからである。スカラーの掛け算順序を指しているのではない。

## 5. なぜ小さい特異値を候補にするか

$|S_\beta\rangle:=|L_\beta\rangle\otimes|R_\beta\rangle$ と略記する。積状態の内積を成分へ展開すると

$$
\begin{aligned}
\langle S_\beta|S_\gamma\rangle
&=\sum_{i_1,i_2,i_3}
L_\beta(i_1,i_2)L_\gamma(i_1,i_2)
\mathcal R_\beta(i_3)\mathcal R_\gamma(i_3)\\
&=\left[\sum_{i_1,i_2}L_\beta(i_1,i_2)L_\gamma(i_1,i_2)\right]
\left[\sum_{i_3}\mathcal R_\beta(i_3)\mathcal R_\gamma(i_3)\right]\\
&=\delta_{\beta\gamma}\delta_{\beta\gamma}
=\delta_{\beta\gamma}.
\end{aligned}
$$

従ってSchmidt展開は直交展開であり、

$$
\begin{aligned}
\|X\|_F^2
&=\left\langle\sum_{\beta=1}^{\rho}\sigma_\beta S_\beta
\middle|\sum_{\gamma=1}^{\rho}\sigma_\gamma S_\gamma\right\rangle\\
&=\sum_{\beta=1}^{\rho}\sum_{\gamma=1}^{\rho}
\sigma_\beta\sigma_\gamma\delta_{\beta\gamma}
=\sum_{\beta=1}^{\rho}\sigma_\beta^2.
\end{aligned}
$$

各成分のノルム二乗は $\|\sigma_\beta S_\beta\|^2=\sigma_\beta^2$。特異値が降順なので、小さい成分から削れば、この**固定されたSchmidt直交展開の中で**削るノルム二乗が小さい。例えば

$$
(\sigma_1,\sigma_2,\sigma_3,\sigma_4)=(10,3,0.2,0.01)
\quad\Longrightarrow\quad
(\sigma_1^2,\sigma_2^2,\sigma_3^2,\sigma_4^2)
=(100,9,0.04,0.0001).
$$

第4成分を一つ削る損失は $0.0001$、第1成分なら $100$ である。これはFrobeniusノルムで測った大きさの比較であり、個別の観測量や長距離相関への影響を同じ比率で保証するものではない。**すべてのrank-$k$行列の中での最適性**を証明するEckart–Young–Mirsky定理は、このノートの学習範囲に含めない。

## 6. 不変でなくなる理由と単一cutの誤差

元の状態と近似状態を引くと、消えた項がそのまま残る。

$$
\begin{aligned}
|X\rangle-|\widetilde X_k\rangle
&=\sum_{\beta=1}^{\rho}\sigma_\beta|S_\beta\rangle
-\sum_{\beta=1}^{k}\sigma_\beta|S_\beta\rangle\\
&=\sum_{\beta=k+1}^{\rho}\sigma_\beta|S_\beta\rangle.
\end{aligned}
$$

$k<\rho$ なら少なくとも $\sigma_{k+1}>0$ であり、直交する非零成分が残るため $X\ne\widetilde X_k$ である。左Schmidt状態の残す部分空間への射影を

$$
P_k:=\sum_{\gamma=1}^{k}|L_\gamma\rangle\langle L_\gamma|
$$

と定義すれば、途中のKronecker deltaを含めて

$$
\begin{aligned}
(P_k\otimes I_B)|X\rangle
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}\sigma_\beta
|L_\gamma\rangle
\langle L_\gamma|L_\beta\rangle
\otimes|R_\beta\rangle\\
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}\sigma_\beta
\delta_{\gamma\beta}|L_\gamma\rangle\otimes|R_\beta\rangle\\
&=\sum_{\beta=1}^{k}\sigma_\beta
|L_\beta\rangle\otimes|R_\beta\rangle
=|\widetilde X_k\rangle.
\end{aligned}
$$

誤差のノルム二乗は、交差項を省略せずに

$$
\begin{aligned}
\|X-\widetilde X_k\|_F^2
&=\left\langle
\sum_{\beta=k+1}^{\rho}\sigma_\beta S_\beta
\middle|
\sum_{\gamma=k+1}^{\rho}\sigma_\gamma S_\gamma
\right\rangle\\
&=\sum_{\beta=k+1}^{\rho}\sum_{\gamma=k+1}^{\rho}
\sigma_\beta\sigma_\gamma
\langle S_\beta|S_\gamma\rangle\\
&=\sum_{\beta=k+1}^{\rho}\sum_{\gamma=k+1}^{\rho}
\sigma_\beta\sigma_\gamma\delta_{\beta\gamma}\\
&=\boxed{\sum_{\beta=k+1}^{\rho}\sigma_\beta^2}.
\end{aligned}
$$

同じ結論を中心行列からも確認できる。$\widetilde X_k^{\langle2\rangle}=T\widetilde A_kR$ だから

$$
\begin{aligned}
\|X-\widetilde X_k\|_F^2
&=\|T(A-\widetilde A_k)R\|_F^2\\
&=\operatorname{tr}\!\left[
R^T(A-\widetilde A_k)^T T^T T(A-\widetilde A_k)R
\right]\\
&=\operatorname{tr}\!\left[
(A-\widetilde A_k)^T(A-\widetilde A_k)RR^T
\right]\\
&=\|A-\widetilde A_k\|_F^2
=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2.
\end{aligned}
$$

第2行から第3行では $T^TT=I$ とtraceの巡回性を使い、第4行では $RR^T=I$ を使った。これで局所行列の誤差が全テンソルの**同じ一つのcut**の誤差になる理由も分かる。複数cutを順に打ち切った最終TTの誤差に、この一回の等式をそのまま代入してはいけない。

規格化された $|X\rangle$ なら、捨てた重みを $w=\sum_{\beta>k}\sigma_\beta^2$ とすると

$$
\|\widetilde X_k\|_F^2=1-w,
\qquad
\|X-\widetilde X_k\|_F^2=w.
$$

後で $|\widehat X_k\rangle=|\widetilde X_k\rangle/\sqrt{1-w}$ と再規格化した場合、捨てた成分は戻らず、元状態との二乗距離は一般に $w$ ではない。

## 7. 行位置と8要素を最後まで追う例

既存の43番ノートとは別の小さい例を使う。$n_1=n_2=n_3=r_1=r_2=2$、$H=R=I_2$ とし、中心行列のcompact SVDを

$$
U=\frac1{\sqrt2}
\begin{pmatrix}
1&0\\0&1\\0&-1\\1&0
\end{pmatrix},\qquad
\Sigma=\begin{pmatrix}3&0\\0&1\end{pmatrix},\qquad
V^T=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}
$$

とする。$U^TU=V^TV=I_2$ である。$A=U\Sigma V^T$ は

$$
A=\frac12
\begin{pmatrix}
3&3\\
1&-1\\
-1&1\\
3&3
\end{pmatrix}.
$$

4行は順に $(\alpha_1,i_2)=(1,1),(1,2),(2,1),(2,2)$、2列は $\alpha_2=1,2$ に対応する。例えば第3行第2列は

$$
A_{(2,1),2}
=U_{(2,1),1}\,3\,(V^T)_{1,2}
+U_{(2,1),2}\,1\,(V^T)_{2,2}
=0+\left(-\frac1{\sqrt2}\right)
\left(-\frac1{\sqrt2}\right)=\frac12.
$$

$H=R=I_2$ なので、$X^{\langle2\rangle}=A$ であり、8要素は二つの$i_2$ sliceとして

$$
X(:,1,:)=
\begin{pmatrix}3/2&3/2\\-1/2&1/2\end{pmatrix},
\qquad
X(:,2,:)=
\begin{pmatrix}1/2&-1/2\\3/2&3/2\end{pmatrix}.
$$

例えば $X(2,1,2)=A_{(2,1),2}=1/2$ である。$k=1$ へ打ち切ると、

$$
\widetilde A_1
=U_1\Sigma_1V_1^T
=\frac12
\begin{pmatrix}
3&3\\0&0\\0&0\\3&3
\end{pmatrix},
\quad
A-\widetilde A_1
=\frac12
\begin{pmatrix}
0&0\\1&-1\\-1&1\\0&0
\end{pmatrix}.
$$

新しいコアは、$\widetilde G_2^{[L]}\in\mathbb R^{2\times2\times1}$、$\widetilde G_3^{[C]}\in\mathbb R^{1\times2\times1}$ となり、具体的には

$$
\widetilde G_2^{[L]}(:,1,1)
=\frac1{\sqrt2}\begin{pmatrix}1\\0\end{pmatrix},
\qquad
\widetilde G_2^{[L]}(:,2,1)
=\frac1{\sqrt2}\begin{pmatrix}0\\1\end{pmatrix},
\qquad
\widetilde G_3^{[C]}(1,:,1)
=\frac3{\sqrt2}\begin{pmatrix}1&1\end{pmatrix}.
$$

近似テンソルの8要素も

$$
\widetilde X_1(:,1,:)
=\begin{pmatrix}3/2&3/2\\0&0\end{pmatrix},
\qquad
\widetilde X_1(:,2,:)
=\begin{pmatrix}0&0\\3/2&3/2\end{pmatrix}.
$$

例えば $\widetilde X_1(2,1,2)=0$ なので、元の $X(2,1,2)=1/2$ は変わった。全要素から二乗誤差を足せば、非零残差は $1/2,-1/2,-1/2,1/2$ の4個で

$$
\|X-\widetilde X_1\|_F^2
=4\left(\frac12\right)^2
=1
=\sigma_2^2.
$$

元のノルム二乗も $\|X\|_F^2=4(3/2)^2+4(1/2)^2=10=3^2+1^2$。保持後は $\|\widetilde X_1\|_F^2=9$、捨てた比率は $1/10$ である。この例は**一つのcutの打ち切り**だけを検証している。
