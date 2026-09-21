---
title: TT・MPSのSVD中心移動とSchmidt形
tags:
  - TT
  - MPS
  - SVD
  - Schmidt
---

# TT・MPSのSVD中心移動とSchmidt形

[[41_TT_MPSの直交中心の移動]] では、QRの直交因子を旧中心に置き、残差因子を隣へ吸収した。出発点となる中心ノルムと、左右環境を固定した中心摂動の等長性は [[40_TT_MPSの中心ノルムと内積の導出]] にある。ここでは第2サイト中心の3階TTを出発点に、同じ移動を**打ち切りなしのSVD**で行う。SVDでは左右の直交性に加えて、切断 $12\mid3$ の係数を非負の対角行列として明示できる。物理状態としての用語は [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]] と対応する。

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

旧中心の左右にあるブロック状態を

$$
|L_{\alpha_1}\rangle
:=\sum_{i_1}G_1^{[L]}(1,i_1,\alpha_1)|i_1\rangle,
\qquad
|R_{\alpha_2}\rangle
:=\sum_{i_3}G_3^{[R]}(\alpha_2,i_3,1)|i_3\rangle
$$

と定義する。コアの直交条件から $\langle L_{\alpha_1}|L_{\gamma_1}\rangle=\delta_{\alpha_1\gamma_1}$、$\langle R_{\alpha_2}|R_{\gamma_2}\rangle=\delta_{\alpha_2\gamma_2}$ である。従って同じ状態は

$$
|X\rangle
=\sum_{\alpha_1,i_2,\alpha_2}
G_2^{[C]}(\alpha_1,i_2,\alpha_2)
|L_{\alpha_1}\rangle\otimes|i_2\rangle\otimes|R_{\alpha_2}\rangle
$$

とも書ける。この段階の $G_2^{[C]}$ は、直交する旧ブロック基底間の係数だが、$(\alpha_1,i_2)$ と $\alpha_2$ の間でまだ対角化されていない。

まず中心だけを左展開する。$p=(\alpha_1,i_2)$ と置いて

$$
A_{(\alpha_1,i_2),\alpha_2}
:=G_2^{[C]}(\alpha_1,i_2,\alpha_2),
\qquad
A\in\mathbb R^{(r_1n_2)\times r_2}.
$$

これはまだ**全テンソルの**cut unfoldingではない。全テンソルの行には $i_1$ も含まれる。両者が同じ非零特異値を持つ理由は後で導く。また、同じ第2コアのright unfolding $C_{\mathrm{right}}\in\mathbb R^{r_1\times(n_2r_2)}$ は $\alpha_1$ を行、$(i_2,\alpha_2)$ を列にまとめる。こちらは中心を第1サイト側へ動かす際の行列で、今回の $2\to3$ に使う $A\in\mathbb R^{(r_1n_2)\times r_2}$ と区別する。

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

$\Sigma$ は対角**行列**、$\sigma_\beta$ はその対角成分である非負の**スカラー**である。同じ特異値が複数あるときも値の組は定まるが、その縮退部分空間内の特異ベクトル、従って個々のSchmidt基底は一意ではない。同じ値の成分を左右で対応させて回転しても $X$ は変わらない。

ここでの $\rho$ は数学的なrankである。浮動小数点計算で零特異値が微小な非零値として現れる場合の数値rank判定は別の問題であり、このexact中心移動から打ち切り閾値を決めたことにはならない。

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

$\Sigma$ を常に独立したコアとして保存する必要はない。第2節のように左展開で $G_3^{[C]\langle L\rangle}=\Sigma\widetilde G_3^{[R]\langle L\rangle}$ と吸収しても、SVDで選んだ左右の基底と特異値の意味は失われない。ボンド上のSchmidt係数を明示したいときに、二つの因子へ分けて表示する。

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

一つの行要素と列要素に固定して、消える和も書く。

$$
\begin{aligned}
X^{\langle2\rangle}_{(i_1,i_2),i_3}
&=\sum_{\beta=1}^{\rho}\sum_{\gamma=1}^{\rho}
L_{(i_1,i_2),\beta}\Sigma_{\beta,\gamma}R_{\gamma,i_3}\\
&=\sum_{\beta=1}^{\rho}\sum_{\gamma=1}^{\rho}
L_{(i_1,i_2),\beta}\sigma_\beta
\delta_{\beta\gamma}R_{\gamma,i_3}\\
&=\sum_{\beta=1}^{\rho}
L_{(i_1,i_2),\beta}\sigma_\beta R_{\beta,i_3}.
\end{aligned}
$$

一般の係数行列 $C$ なら $\sum_{\beta,\gamma}L_{(i_1,i_2),\beta}C_{\beta\gamma}R_{\gamma,i_3}$ に $\beta\ne\gamma$ の交差係数が残り得る。「一対一」とはスカラー積の順序ではなく、$\Sigma$ の非対角係数が0という意味である。

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

中心を第3サイトへ移した後も、同じノルムは第3中心コアだけで計算できる。左ブロックのGramを先に縮約すると

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{i_1,i_2,i_3}
\left[\sum_\beta L_\beta(i_1,i_2)G_3^{[C]}(\beta,i_3,1)\right]
\left[\sum_\gamma L_\gamma(i_1,i_2)G_3^{[C]}(\gamma,i_3,1)\right]\\
&=\sum_{\beta,\gamma,i_3}
\underbrace{\sum_{i_1,i_2}L_\beta(i_1,i_2)L_\gamma(i_1,i_2)}_{\delta_{\beta\gamma}}
G_3^{[C]}(\beta,i_3,1)G_3^{[C]}(\gamma,i_3,1)\\
&=\sum_{\beta,i_3}G_3^{[C]}(\beta,i_3,1)^2
=\|G_3^{[C]}\|_F^2.
\end{aligned}
$$

また $G_3^{[C]\langle L\rangle}=\Sigma V^TR$ なので、$RR^T=I_{r_2}$ と $V^TV=I_\rho$ を順に使えば

$$
\begin{aligned}
\|G_3^{[C]}\|_F^2
&=\operatorname{tr}\!\left[(\Sigma V^TR)(\Sigma V^TR)^T\right]\\
&=\operatorname{tr}\!\left[\Sigma V^TRR^TV\Sigma\right]\\
&=\operatorname{tr}\!\left[\Sigma V^TI_{r_2}V\Sigma\right]\\
&=\operatorname{tr}\!\left[\Sigma I_\rho\Sigma\right]\\
&=\operatorname{tr}(\Sigma^2)
=\sum_\beta\sigma_\beta^2.
\end{aligned}
$$

従って $\|X\|_F=\|G_2^{[C]}\|_F=\|G_3^{[C]}\|_F$ である。

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

なら $r_{12}$ が二つの右ボンド方向を混ぜる。これをそのままボンド上のSchmidt係数とは読めない。SVDは左右の直交基底の間の係数を $\Sigma_{\beta\gamma}=\sigma_\beta\delta_{\beta\gamma}$ へ対角化する。QRもrank-1行列の和として

$$
A=Q_{\mathrm{QR}}R_{\mathrm{QR}}
=\sum_\beta (Q_{\mathrm{QR}})_{:,\beta}(R_{\mathrm{QR}})_{\beta,:}
$$

とも展開できるが、$R_{\mathrm{QR}}$ の異なる行は一般に直交しない。そのため、この和の各項はSchmidtの直交rank-1成分とは限らない。特異値は

$$
A^TA\,v_\beta=\sigma_\beta^2v_\beta,
\qquad
Av_\beta=\sigma_\beta u_\beta
$$

でも確かめられる。$\sigma_\beta$ は $A$ が右特異方向 $v_\beta$ をどれだけ伸ばすかを表す。QRは入力列 $A=[a_1,\ldots,a_{r_2}]$ をその順序で直交化する。例えば独立な最初の2列にGram–Schmidtを適用すると

$$
q_1=\frac{a_1}{\|a_1\|},\qquad
b_2=a_2-(q_1^Ta_2)q_1,\qquad
q_2=\frac{b_2}{\|b_2\|}
$$

となる。$q_2$ を作る前に $q_1$ 成分を引くため、列順を変えると一般に $Q_{\mathrm{QR}}$ と $R_{\mathrm{QR}}$ も変わる。QRの直交化だけでは、$A^TA$ の固有方向と伸縮率は露出しない。単に中心を動かす目的なら、通常はSVDより計算の軽いQRを使える。

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

一般次元でも、二部系の係数行列 $M\in\mathbb R^{d_A\times d_B}$ を用いて

$$
\begin{aligned}
|X\rangle&=\sum_{a=1}^{d_A}\sum_{b=1}^{d_B}
M_{a,b}|a\rangle_A\otimes|b\rangle_B,\\
M&=U_M\Sigma_MV_M^T,\\
M_{a,b}&=\sum_{\beta=1}^{\operatorname{rank}(M)}
(U_M)_{a,\beta}(\Sigma_M)_{\beta\beta}(V_M^T)_{\beta,b}.
\end{aligned}
$$

$\rho_M=\operatorname{rank}(M)$ と置けば、compact SVDのshapeは $U_M\in\mathbb R^{d_A\times\rho_M}$、$\Sigma_M\in\mathbb R^{\rho_M\times\rho_M}$、$V_M^T\in\mathbb R^{\rho_M\times d_B}$ である。

$|L_\beta\rangle_A:=\sum_a(U_M)_{a,\beta}|a\rangle_A$、$|R_\beta\rangle_B:=\sum_b(V_M^T)_{\beta,b}|b\rangle_B$ と置けば、$U_M$ の列と $V_M^T$ の行がそれぞれ正規直交し、上のSchmidt形が得られる。逆に二つの積状態の和でも、片側の状態同士が直交しなければSchmidt形とは呼べない。

今回のTT/MPSでは $C$ に対応するのは**全テンソルの** $X^{\langle2\rangle}$ である。中心行列 $A$ の $V$ だけが物理的な右Schmidt状態になるわけではない。右状態の係数は $V^TR$、左状態の係数は $(H\otimes I_{n_2})U$ である。左右の等長性により $A$ と $X^{\langle2\rangle}$ の非零特異値は一致する。

## 9. テンソル積の内積を4要素から確かめる

左の空間は $\mathcal H_L=\mathbb R^{n_1}\otimes\mathbb R^{n_2}$、右は $\mathcal H_R=\mathbb R^{n_3}$、全体は $\mathcal H_L\otimes\mathcal H_R$ である。単純テンソルの内積は

$$
\langle x\otimes y\mid x'\otimes y'\rangle
:=\langle x\mid x'\rangle_{\mathcal H_L}
\langle y\mid y'\rangle_{\mathcal H_R}
$$

と定義し、線形性で一般の和へ拡張する。第5節では積基底のdeltaからこの内積を分解した。2成分ずつを全要素で書くと、実数ベクトル $l=(a,b)^T$、$r=(c,d)^T$ に対して

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

## 10. Tucker分解との対応と違い

両方にSVDと「中心」が現れるが、今回の操作では既存の鎖 $G_1-G_2-G_3$ の第2・第3コア間のボンドだけを扱う。Tucker分解の典型的な3階表現は、各物理modeに因子を置き、一つの密なcore $\mathcal S$ に結ぶ形である。

$$
X=\mathcal S\times_1 U^{(1)}\times_2 U^{(2)}\times_3 U^{(3)},
\qquad
U^{(j)}\in\mathbb R^{n_j\times s_j}.
$$

$\times_j$ は第 $j$ modeに沿った行列との積を表す。以下の添字式が各mode積を全要素に展開したものである。

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{a_1=1}^{s_1}\sum_{a_2=1}^{s_2}\sum_{a_3=1}^{s_3}
\mathcal S(a_1,a_2,a_3)
U^{(1)}(i_1,a_1)U^{(2)}(i_2,a_2)U^{(3)}(i_3,a_3),\\
\mathcal S&\in\mathbb R^{s_1\times s_2\times s_3}.
\end{aligned}
$$

一方、TT/MPSは第1節の $\sum_{\alpha_1,\alpha_2}G_1G_2G_3$ という隣接ボンドの縮約である。Tuckerが一つのcoreから各modeへ因子を伸ばす構造なのに対し、TT/MPSの内部自由度は二つのボンド $\alpha_1,\alpha_2$ に沿って局所コアへ分散する。最小のexact表現なら、TT-rank $r_1,r_2$ は連続切断 $i_1\mid(i_2,i_3)$ と $(i_1,i_2)\mid i_3$ のunfolding rankに対応する。Tuckerのmultilinear rank $s_j$ は各mode $i_j\mid\text{他のmode}$ のunfolding rankに対応する。切断の取り方が異なる。

今回の $\Sigma\in\mathbb R^{\rho\times\rho}$ は一本のボンド上の**対角行列**であり、一般には密な3階Tucker core $\mathcal S$ ではない。TT/MPSでは鎖上の直交中心を隣のサイトへ移せるが、Tuckerの一つの密なcoreには通常そのようなサイト間の中心移動を考えない。Tucker/HOSVDのSVD、全テンソルからコアを順に作るTT-SVD、既存TTの局所的なSVD中心移動は、それぞれ分解対象と結果の構造が異なる。Tuckerの詳しい導出は [[22_Tucker分解とHOSVD]] に置く。

## 11. Notebook 08との対応

中心コアのleft unfoldingと全テンソルのcut unfoldingの違いは第1・4節、$V^TR_{\mathrm{old}}$の意味と右直交性は第3節、左ブロックの$L^TL=I_\rho$の行要素展開は第4節に示した。

Pythonの`reshape`で左右の添字をどの順にまとめるかは [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]]、`Vh @ R_old`、`tensordot`後のshape、`L_tensor`から`L_block`への行列化は [[00_基礎理論/05_PyTorch実装/28_TT_MPS実装で使うPyTorch_Python操作メモ]] を参照する。Notebook 08の保存済みshape・数値結果は [[40_TensorTrain_MPS/05_SVD中心移動とSchmidt形のPyTorch確認]] にまとめた。

## 参考資料

- [Schollwöck：The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477)：MPS正準形とSchmidt分解。
- [LMU：MPS I, Matrix Product States Canonical Forms](https://www2.physik.uni-muenchen.de/lehre/vorlesungen/sose_22/tensor_networks_22/skript/04-MPS-I-MatrixProductStatesCanonicalForms.pdf)：左右直交化、ボンド正準形とSVDによる中心移動。
