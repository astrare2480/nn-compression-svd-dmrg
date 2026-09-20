---
title: TT・MPSのSVD中心移動とSchmidt形
tags:
  - TT
  - MPS
  - SVD
  - Schmidt
---

# TT・MPSのSVD中心移動とSchmidt形

[[41_TT_MPSの直交中心の移動]] では、QRの直交因子を旧中心に置き、残差因子を隣へ吸収した。ここでは第2サイト中心の3階TTを出発点に、同じ移動を**打ち切りなしのSVD**で行う。SVDでは左右の直交性に加えて、切断 $12\mid3$ の係数を非負の対角行列として明示できる。物理状態としての用語は [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]] と対応する。

## 1. 出発点とSVDの対象

実数の3階TTを第2サイト中心の混合正準形で

$$
X(i_1,i_2,i_3)
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[C]}(\alpha_1,i_2,\alpha_2)
G_3^{[R]}(\alpha_2,i_3,1)
$$

と書く。$G_1^{[L]}$ の左展開は列直交、$G_3^{[R]}$ の右展開は行直交である。中心 $G_2^{[C]}\in\mathbb R^{r_1\times n_2\times r_2}$ には直交条件を課さない。

まず中心だけを左展開する。$p=(\alpha_1,i_2)$ と置いて

$$
A_{(\alpha_1,i_2),\alpha_2}
:=G_2^{[C]}(\alpha_1,i_2,\alpha_2),
\qquad
A\in\mathbb R^{(r_1n_2)\times r_2}.
$$

これはまだ**全テンソルの**cut unfoldingではない。全テンソルの行には $i_1$ も含まれる。両者が同じ非零特異値を持つ理由は後で導く。

$A\ne0$ とし、数学的なexact rankを $\rho=\operatorname{rank}(A)$ とする。非零特異値だけを残すcompact SVDは

$$
A=U\Sigma V^T,\qquad
U\in\mathbb R^{(r_1n_2)\times\rho},\quad
\Sigma\in\mathbb R^{\rho\times\rho},\quad
V^T\in\mathbb R^{\rho\times r_2},
$$

$$
U^TU=I_\rho,\qquad V^TV=I_\rho,\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_\rho),
\quad \sigma_1\ge\cdots\ge\sigma_\rho>0
$$

である。零特異値を表記から除くのは圧縮rankを選んだことではない。非零成分はすべて保持するので $A=U\Sigma V^T$ は厳密な等式である。$\rho=0$ の零テンソルは係数がすべて0の自明な場合として分ける。

## 2. $U$ を左直交コアに戻し、$\Sigma V^T$ を右へ渡す

$$
G_2^{[L]}(\alpha_1,i_2,\beta)
:=U_{(\alpha_1,i_2),\beta},
\qquad
G_2^{[L]}\in\mathbb R^{r_1\times n_2\times\rho}.
$$

$U^TU=I_\rho$ を成分へ戻すと

$$
\sum_{\alpha_1=1}^{r_1}\sum_{i_2=1}^{n_2}
G_2^{[L]}(\alpha_1,i_2,\beta)
G_2^{[L]}(\alpha_1,i_2,\gamma)
=\delta_{\beta\gamma}.
$$

従って新しい第2コアは左直交である。一方、$U$ だけでは元の $A$ にならない。残りを

$$
B:=\Sigma V^T\in\mathbb R^{\rho\times r_2},
\qquad
B_{\beta,\alpha_2}=\sigma_\beta(V^T)_{\beta,\alpha_2}
$$

と置く。SVDの積を添字で書けば

$$
G_2^{[C]}(\alpha_1,i_2,\alpha_2)
=\sum_{\beta=1}^{\rho}
G_2^{[L]}(\alpha_1,i_2,\beta)B_{\beta,\alpha_2}.
$$

これを第3コアへ吸収して

$$
G_3^{[C]}(\beta,i_3,1)
:=\sum_{\alpha_2=1}^{r_2}
B_{\beta,\alpha_2}G_3^{[R]}(\alpha_2,i_3,1),
\qquad
G_3^{[C]}\in\mathbb R^{\rho\times n_3\times1}
$$

と定義する。旧ボンド $\alpha_2$ は和で消え、新ボンド $\beta$ が残る。元の式に代入する順序まで示すと

$$
\begin{aligned}
X_{\mathrm{after}}(i_1,i_2,i_3)
&=\sum_{\alpha_1,\beta}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta)
G_3^{[C]}(\beta,i_3,1)\\
&=\sum_{\alpha_1,\beta,\alpha_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta)
B_{\beta,\alpha_2}G_3^{[R]}(\alpha_2,i_3,1)\\
&=\sum_{\alpha_1,\alpha_2}
G_1^{[L]}(1,i_1,\alpha_1)
\left[\sum_\beta G_2^{[L]}(\alpha_1,i_2,\beta)B_{\beta,\alpha_2}\right]
G_3^{[R]}(\alpha_2,i_3,1)\\
&=X_{\mathrm{before}}(i_1,i_2,i_3).
\end{aligned}
$$

従って全要素が不変で、中心は第3サイトへ移る。$\rho<r_2$ ならボンドの**表示サイズ**は減るが、零特異方向だけを除くので近似誤差は0である。ただしこの場合、同サイズの可逆行列を挿入する意味でのGauge変換とは呼ばない。非零特異値をさらに捨てる一回の打ち切りは [[45_TT_MPSの単一ボンドSVD打ち切り]] の別操作である。

ここで行っているのは、既存の鎖状TTの隣接ボンドをまたぐ**局所的な中心移動**である。[[22_Tucker分解とHOSVD]] のように、各物理modeの因子行列と一つの密な3階coreを新たに作るTucker分解ではない。$\Sigma$ もTucker coreではなく、切断 $(i_1,i_2)\mid i_3$ の左右ブロックを結ぶ一本のボンド上の対角行列である。また、全テンソルから左から順にコアを構成するTT-SVDそのものとも区別する。

## 3. $\Sigma$ をボンド上に残す

$B$ をまとめて吸収せず、右側の基底回転と対角係数に分ける。$R(\alpha_2,i_3):=G_3^{[R]}(\alpha_2,i_3,1)$ とし、

$$
\widetilde R(\beta,i_3)
:=\sum_{\alpha_2=1}^{r_2}(V^T)_{\beta,\alpha_2}R(\alpha_2,i_3),
\qquad
\widetilde R=V^TR\in\mathbb R^{\rho\times n_3}
$$

と置く。元の右直交性は $RR^T=I_{r_2}$ なので、行列積を省略せずに書けば

$$
\begin{aligned}
\widetilde R\widetilde R^T
&=(V^TR)(V^TR)^T\\
&=V^T(RR^T)V\\
&=V^TI_{r_2}V
=I_\rho.
\end{aligned}
$$

この式は $\rho<r_2$ でも成立する。$\widetilde G_3^{[R]}(\beta,i_3,1):=\widetilde R(\beta,i_3)$ とすれば

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1,\beta,\gamma}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta)
\Sigma_{\beta,\gamma}
\widetilde G_3^{[R]}(\gamma,i_3,1)\\
&=\sum_{\alpha_1,\beta}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta)
\sigma_\beta\widetilde G_3^{[R]}(\beta,i_3,1).
\end{aligned}
$$

第2行は $\Sigma_{\beta,\gamma}=\sigma_\beta\delta_{\beta\gamma}$ により $\gamma$ の和を消した結果である。$\Sigma$ の対角性は、QRの一般に非対角な三角因子からは得られない。

## 4. 全テンソルのcut unfoldingとSchmidt形

左ブロックと右ブロックの係数を

$$
L_\beta(i_1,i_2)
:=\sum_{\alpha_1=1}^{r_1}
G_1^{[L]}(1,i_1,\alpha_1)G_2^{[L]}(\alpha_1,i_2,\beta),
\qquad
R_\beta(i_3):=\widetilde G_3^{[R]}(\beta,i_3,1)
$$

と定義する。$L$ の行は $(i_1,i_2)$、列は $\beta$、$R$ の行は $\beta$、列は $i_3$ であり、

$$
L\in\mathbb R^{(n_1n_2)\times\rho},\qquad
R\in\mathbb R^{\rho\times n_3}.
$$

第1コアの左展開を $H(i_1,\alpha_1):=G_1^{[L]}(1,i_1,\alpha_1)$ とすれば、同じ複合添字順で

$$
L=(H\otimes I_{n_2})U.
$$

$H^TH=I_{r_1}$ だから、Kronecker積の行列積公式より

$$
\begin{aligned}
L^TL
&=U^T(H^T\otimes I_{n_2})(H\otimes I_{n_2})U\\
&=U^T(H^TH\otimes I_{n_2})U\\
&=U^TU=I_\rho.
\end{aligned}
$$

成分でも、第1コアのGramを先に縮約すると

$$
\begin{aligned}
\sum_{i_1,i_2}L_\beta(i_1,i_2)L_\gamma(i_1,i_2)
&=\sum_{i_1,i_2}
\left[\sum_{\alpha_1}H(i_1,\alpha_1)
U_{(\alpha_1,i_2),\beta}\right]
\left[\sum_{\alpha_1'}H(i_1,\alpha_1')
U_{(\alpha_1',i_2),\gamma}\right]\\
&=\sum_{\alpha_1,\alpha_1',i_2}
U_{(\alpha_1,i_2),\beta}U_{(\alpha_1',i_2),\gamma}
\underbrace{\sum_{i_1}H(i_1,\alpha_1)H(i_1,\alpha_1')}_{\delta_{\alpha_1\alpha_1'}}\\
&=\sum_{\alpha_1,i_2}
U_{(\alpha_1,i_2),\beta}U_{(\alpha_1,i_2),\gamma}
=\delta_{\beta\gamma}.
\end{aligned}
$$

右側は前節の $\widetilde R\widetilde R^T=I_\rho$ である。成分でも旧右コアの行直交性を先に使えば

$$
\begin{aligned}
\sum_{i_3}R_\beta(i_3)R_\gamma(i_3)
&=\sum_{i_3}
\left[\sum_{\alpha_2}(V^T)_{\beta,\alpha_2}R(\alpha_2,i_3)\right]
\left[\sum_{\alpha_2'}(V^T)_{\gamma,\alpha_2'}R(\alpha_2',i_3)\right]\\
&=\sum_{\alpha_2,\alpha_2'}
(V^T)_{\beta,\alpha_2}(V^T)_{\gamma,\alpha_2'}
\underbrace{\sum_{i_3}R(\alpha_2,i_3)R(\alpha_2',i_3)}
_{\delta_{\alpha_2\alpha_2'}}\\
&=\sum_{\alpha_2}
(V^T)_{\beta,\alpha_2}(V^T)_{\gamma,\alpha_2}
=\delta_{\beta\gamma}.
\end{aligned}
$$

最後の等号は $V^TV=I_\rho$、すなわち $V^T$ の行同士の直交性である。これも $\rho<r_2$ の場合に成立する。従って全テンソルの切断 $12\mid3$ について

$$
X^{\langle2\rangle}_{(i_1,i_2),i_3}
=X(i_1,i_2,i_3)
=\sum_{\beta=1}^{\rho}
L_\beta(i_1,i_2)\sigma_\beta R_\beta(i_3),
\qquad
X^{\langle2\rangle}=L\Sigma R.
$$

ここで初めて、中心だけの $A$ の特異値が全テンソルのcut unfoldingの特異値であることが示された。$L$ は列直交、$R$ は行直交なので $L\Sigma R$ 自体がcompact SVDである。$\sigma_\beta$ は個々のコア要素や第3コアだけの「残す割合」ではない。二つのブロックをまとめた後の、正規直交なrank-1行列 $L_{:,\beta}R_{\beta,:}$ 全体の係数である。対角行列の二つの添字を残すと

$$
\sum_{\beta,\gamma}L_{:,\beta}\Sigma_{\beta,\gamma}R_{\gamma,:}
=\sum_{\beta}\sigma_\beta L_{:,\beta}R_{\beta,:}.
$$

「一対一」とはスカラー積の順序ではなく、$\beta\ne\gamma$ の混合係数が0という意味である。

各 $L_{:,\beta}R_{\beta,:}$ が単位Frobeniusノルムのrank-1行列であることも、行要素の内積から確かめられる。

$$
\begin{aligned}
\langle L_{:,\beta}R_{\beta,:},
L_{:,\gamma}R_{\gamma,:}\rangle_F
&=\sum_{i_1,i_2,i_3}
L_\beta(i_1,i_2)R_\beta(i_3)
L_\gamma(i_1,i_2)R_\gamma(i_3)\\
&=\left[\sum_{i_1,i_2}L_\beta(i_1,i_2)L_\gamma(i_1,i_2)\right]
\left[\sum_{i_3}R_\beta(i_3)R_\gamma(i_3)\right]\\
&=\delta_{\beta\gamma}.
\end{aligned}
$$

$\beta=\gamma$ ならノルムは1で、$\beta\ne\gamma$ なら二つのrank-1行列は直交する。従って $\|\sigma_\beta L_{:,\beta}R_{\beta,:}\|_F=\sigma_\beta$。$\sigma_\beta$ は一つのコア要素の「残す割合」ではなく、**切断全体の一つの直交rank-1成分の大きさ**である。

物理基底から $|L_\beta\rangle:=\sum_{i_1,i_2}L_\beta(i_1,i_2)|i_1\rangle\otimes|i_2\rangle$、$|R_\beta\rangle:=\sum_{i_3}R_\beta(i_3)|i_3\rangle$ と定めると、有限和の順序を入れ替えて

$$
\begin{aligned}
|X\rangle
&=\sum_{i_1,i_2,i_3}X(i_1,i_2,i_3)
|i_1\rangle\otimes|i_2\rangle\otimes|i_3\rangle\\
&=\sum_\beta\sigma_\beta
\left[\sum_{i_1,i_2}L_\beta(i_1,i_2)|i_1\rangle\otimes|i_2\rangle\right]
\otimes
\left[\sum_{i_3}R_\beta(i_3)|i_3\rangle\right]\\
&=\sum_\beta\sigma_\beta|L_\beta\rangle\otimes|R_\beta\rangle.
\end{aligned}
$$

$L^TL=I_\rho$、$RR^T=I_\rho$、$\sigma_\beta>0$ だから、これは単なる積状態の和ではなくSchmidt分解である。複素係数の場合は転置を随伴に替え、右Schmidt ketの係数とbraの共役を区別する。

## 5. ノルムとテンソル積の内積

積基底の内積を成分で確かめる。$a=(i_1,i_2)$、$b=i_3$ とし、$\langle a|a'\rangle=\delta_{aa'}$、$\langle b|b'\rangle=\delta_{bb'}$ とすれば

$$
\begin{aligned}
\langle L_\beta\otimes R_\beta
\mid L_\gamma\otimes R_\gamma\rangle
&=\sum_{a,b,a',b'}
L_\beta(a)R_\beta(b)L_\gamma(a')R_\gamma(b')
\delta_{aa'}\delta_{bb'}\\
&=\left[\sum_a L_\beta(a)L_\gamma(a)\right]
\left[\sum_b R_\beta(b)R_\gamma(b)\right]\\
&=\delta_{\beta\gamma}\delta_{\beta\gamma}
=\delta_{\beta\gamma}.
\end{aligned}
$$

独立な物理添字の和が二つの内積の積に分かれるためである。同じ事実は $(l\otimes r)^T(l'\otimes r')=(l^Tl')(r^Tr')$ とも書ける。従って交差項を消すと

$$
\begin{aligned}
\|X\|_F^2
&=\langle X|X\rangle
=\sum_{\beta,\gamma}\sigma_\beta\sigma_\gamma
\langle L_\beta\otimes R_\beta
\mid L_\gamma\otimes R_\gamma\rangle\\
&=\sum_{\beta,\gamma}\sigma_\beta\sigma_\gamma\delta_{\beta\gamma}
=\sum_{\beta=1}^{\rho}\sigma_\beta^2.
\end{aligned}
$$

別経路でも、中心ノルムの局所化とSVDの等長性から

$$
\|X\|_F^2=\|G_2^{[C]}\|_F^2
=\|A\|_F^2=\|\Sigma\|_F^2
=\sum_\beta\sigma_\beta^2
$$

となる。$X$ が規格化された状態なら $\sum_\beta\sigma_\beta^2=1$。これはこの段階では規格化条件であり、特異値の打ち切りやエンタングルメントエントロピーの導入ではない。

## 6. 一つの全要素例

新しい切断を明瞭にするため、$n_1=n_2=n_3=r_1=r_2=2$ とし、第1・第3コアの境界を除いた行列を $H=R=I_2$ にする。第2中心コアの左展開を次の4行2列に取る。行順は $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ である。

$$
\begin{aligned}
U&=\begin{pmatrix}1&0\\0&0\\0&0\\0&1\end{pmatrix},&
\Sigma&=\begin{pmatrix}2&0\\0&1\end{pmatrix},&
V^T&=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\\
A=U\Sigma V^T
&=\frac1{\sqrt2}\begin{pmatrix}
2&2\\0&0\\0&0\\1&-1
\end{pmatrix}.&&
\end{aligned}
$$

二つの物理sliceへ戻すと

$$
G_2^{[C]}(:,1,:)=\begin{pmatrix}\sqrt2&\sqrt2\\0&0\end{pmatrix},
\qquad
G_2^{[C]}(:,2,:)=\begin{pmatrix}0&0\\1/\sqrt2&-1/\sqrt2\end{pmatrix}.
$$

$H=R=I_2$ なので、全テンソルの8要素も

$$
X(:,1,:)=\begin{pmatrix}\sqrt2&\sqrt2\\0&0\end{pmatrix},
\qquad
X(:,2,:)=\begin{pmatrix}0&0\\1/\sqrt2&-1/\sqrt2\end{pmatrix}
$$

である。SVD中心移動後の第2コアと第3中心コアは

$$
G_2^{[L]}(:,1,:)=\begin{pmatrix}1&0\\0&0\end{pmatrix},
\quad
G_2^{[L]}(:,2,:)=\begin{pmatrix}0&0\\0&1\end{pmatrix},
\quad
G_3^{[C]}(:,:,1)=\Sigma V^T
=\frac1{\sqrt2}\begin{pmatrix}2&2\\1&-1\end{pmatrix}.
$$

例えば $X(2,2,2)=1\cdot1\cdot(-1/\sqrt2)=-1/\sqrt2$、$X(1,1,1)=1\cdot1\cdot\sqrt2=\sqrt2$ と再構成できる。対角行列を残すなら右コアは $\widetilde R=V^T$ で、

$$
|X\rangle
=2\,|1,1\rangle\otimes\frac{|1\rangle+|2\rangle}{\sqrt2}
+|2,2\rangle\otimes\frac{|1\rangle-|2\rangle}{\sqrt2}.
$$

左状態二本も右状態二本も正規直交し、Schmidt係数は $2,1$ である。8要素から直接計算したノルムも

$$
\|X\|_F^2
=2+2+\frac12+\frac12
=5
=2^2+1^2.
$$

量子状態として規格化するなら $|X\rangle/\sqrt5$ とし、Schmidt係数は $2/\sqrt5,1/\sqrt5$ になる。QRによる中心移動だけでも左右の直交性は作れるが、一般に残差因子は非対角なので、この二つのSchmidt係数をそのまま読めるわけではない。

## 7. QRとSVDで右へ渡す因子を比べる

同じ中心行列 $A\in\mathbb R^{(r_1n_2)\times r_2}$ を右へ動かす場合、QRでは $A=Q_{\mathrm{QR}}R_{\mathrm{QR}}$ として、左コアへ $Q_{\mathrm{QR}}$、右コアへ $R_{\mathrm{QR}}G_3^{[R]\langle L\rangle}$ を入れる。SVDでは $A=U\Sigma V^T$ として、左コアへ $U$、右コアへ **$\Sigma V^TG_3^{[R]\langle L\rangle}$ 全体**を入れる。$\Sigma$ だけを右へ渡すと $V^T$ が欠け、一般に元の $A$ を再構成できない。

$$
\begin{aligned}
A_{(\alpha_1,i_2),\alpha_2}
&=\sum_{\beta}
(Q_{\mathrm{QR}})_{(\alpha_1,i_2),\beta}
(R_{\mathrm{QR}})_{\beta,\alpha_2}\\
&=\sum_{\beta=1}^{\rho}
U_{(\alpha_1,i_2),\beta}
\sigma_\beta(V^T)_{\beta,\alpha_2}.
\end{aligned}
$$

どちらも左因子は列直交で、残りの因子を右へ運べば $X$ は不変である。QRの残差 $R_{\mathrm{QR}}$ は、行列のshapeに応じて上三角または上台形で、正方形の場合は例えば

$$
R_{\mathrm{QR}}=
\begin{pmatrix}
r_{11}&r_{12}\\
0&r_{22}
\end{pmatrix}
$$

なら $r_{12}$ が二つの右ボンド方向を混ぜる。これをそのままボンド上のSchmidt係数とは読めない。SVDは左右の直交基底の間の係数を $\Sigma_{\beta\gamma}=\sigma_\beta\delta_{\beta\gamma}$ へ対角化する。特異値がどこから来るかは

$$
A^TA\,v_\beta=\sigma_\beta^2v_\beta,
\qquad
Av_\beta=\sigma_\beta u_\beta
$$

でも確かめられる。$\sigma_\beta$ は $A$ が右特異方向 $v_\beta$ をどれだけ伸ばすかを表す。QRの直交化だけでは、この固有方向と伸縮率は露出しない。

ここでQRの「thin」分解の列数を、無条件に $\rho=\operatorname{rank}(A)$ としてはいけない。標準的なthin QRは $q=\min(r_1n_2,r_2)$ 列で、rank落ちした場合にも余分な直交方向を含み得る。compact SVDは非零特異値の本数 $\rho$ 列だけを取る。どちらも**全因子を保持**すればexactだが、$\rho<r_2$ のSVDを同サイズ可逆Gauge変換と同一視しない。

## 8. 一般の積状態和からSchmidt形まで

左右の正規直交基底 $|a_m\rangle_A$ と $|b_n\rangle_B$ を任意に選んだだけなら、係数行列 $C$ は一般に非対角である。2次元ずつなら全要素は

$$
\begin{aligned}
|X\rangle
&=c_{11}|a_1\rangle|b_1\rangle
+c_{12}|a_1\rangle|b_2\rangle\\
&\quad+c_{21}|a_2\rangle|b_1\rangle
+c_{22}|a_2\rangle|b_2\rangle,
\qquad
C=\begin{pmatrix}c_{11}&c_{12}\\c_{21}&c_{22}\end{pmatrix}.
\end{aligned}
$$

$|a_1\rangle|b_2\rangle$ などの交差係数があるので、これはまだSchmidt形とは限らない。$C=U_C\Sigma_CV_C^T$ で左右の基底を同時に更新すると

$$
|X\rangle
=\sum_{\beta=1}^{\operatorname{rank}(C)}
(\Sigma_C)_{\beta\beta}
|L_\beta\rangle_A\otimes|R_\beta\rangle_B
$$

となる。この形をSchmidt分解と呼ぶ条件は、左右がそれぞれ正規直交し、係数が非負で、**同じ $\beta$ 同士だけが結合する**ことである。一般の積状態の和や、左側だけをQRで直交化した表現とは区別する。

今回のTT/MPSでは $C$ に対応するのは**全テンソルの** $X^{\langle2\rangle}$ である。中心行列 $A$ の $V$ だけが物理的な右Schmidt状態になるわけではない。右状態の係数は $V^TR$、左状態の係数は $(H\otimes I_{n_2})U$ である。左右の等長性により $A$ と $X^{\langle2\rangle}$ の非零特異値は一致する。

## 9. テンソル積の内積を4要素から確かめる

第5節では積基底のdeltaから内積を分解した。2成分ずつを全要素で書くと、実数ベクトル $l=(a,b)^T$、$r=(c,d)^T$ に対して

$$
l\otimes r
=\begin{pmatrix}ac\\ad\\bc\\bd\end{pmatrix},
\qquad
l'\otimes r'
=\begin{pmatrix}a'c'\\a'd'\\b'c'\\b'd'\end{pmatrix}.
$$

従って積の4項は

$$
\begin{aligned}
(l\otimes r)^T(l'\otimes r')
&=aa'cc'+aa'dd'+bb'cc'+bb'dd'\\
&=(aa'+bb')(cc'+dd')\\
&=(l^Tl')(r^Tr').
\end{aligned}
$$

独立な左添字と右添字の和を別々に括れることが本質である。一般次元でも

$$
\begin{aligned}
\langle x\otimes y|x'\otimes y'\rangle
&=\sum_{a,b,a',b'}x(a)y(b)x'(a')y'(b')
\delta_{aa'}\delta_{bb'}\\
&=\sum_{a,b}x(a)x'(a)y(b)y'(b)\\
&=\left[\sum_a x(a)x'(a)\right]
\left[\sum_b y(b)y'(b)\right].
\end{aligned}
$$

同じ事実は $(l^T\otimes r^T)(l'\otimes r')=(l^Tl')\otimes(r^Tr')=(l^Tl')(r^Tr')$ とも表せる。特に $\|x\otimes y\|=\|x\|\,\|y\|$ であり、どちらか一方の内積が0なら積状態も直交する。左右Schmidt状態がそれぞれ正規直交するため、異なるSchmidt成分の交差項は0となり、$\|X\|_F^2=\sum_\beta\sigma_\beta^2$ が従う。
