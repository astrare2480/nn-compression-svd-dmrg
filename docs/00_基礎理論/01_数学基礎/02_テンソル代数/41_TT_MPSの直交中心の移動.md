---
title: TT・MPSの直交中心の移動
aliases:
  - orthogonality centerの左右移動
  - TTの中心移動
tags:
  - TT
  - MPS
  - canonical
  - QR
---

# TT・MPSの直交中心の移動

## この章の範囲

[[39_TT_MPSの混合正準形への導入]] で得た第2サイト中心の3階TTから、QRで中心を隣へ一つ動かす。左右の行列化、三角因子の吸収、全テンソルの不変性を添字と全要素例で確認する。端点と一般の $d$ 階でも、中心以外に要求する直交性を明確にする。

この操作は打ち切りを含まない。rankを選ぶTT-SVDやSchmidt分解、局所最適化は別の段階で扱う。ここで導出を示すことは、学習Notebookの演習を完了したことを意味しない。ブロック状態としての物理的解釈は [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]]、PyTorchでの行列化と数値確認は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] に分ける。

## 1. 第2サイト中心の出発点

実数の3階TTについて、$r_0=r_3=1$ とし、

$$
X(i_1,i_2,i_3)
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[C]}(\alpha_1,i_2,\alpha_2)
G_3^{[R]}(\alpha_2,i_3,1)
$$

と書く。shape は順に $(1,n_1,r_1)$、$(r_1,n_2,r_2)$、$(r_2,n_3,1)$ である。第1コアは列直交、第3コアは行直交だが、中心 $G_2^{[C]}$ にはいずれの直交条件も課さない。したがって、中心が「唯一の非直交部分」とは、中心が必ず非直交であるという意味ではない。

$$
\sum_{i_1}G_1^{[L]}(1,i_1,\alpha_1)G_1^{[L]}(1,i_1,\beta_1)=\delta_{\alpha_1\beta_1},
\qquad
\sum_{i_3}G_3^{[R]}(\alpha_2,i_3,1)G_3^{[R]}(\beta_2,i_3,1)=\delta_{\alpha_2\beta_2}.
$$

この条件が中心ノルムへどうつながるかは [[40_TT_MPSの中心ノルムと内積の導出]] で示した。ここでは同じ $X$ を保ったまま、中心の位置を変える。

## 2. 第2サイトから右へ移す

$C=G_2^{[C]}$ を、行が $(\alpha_1,i_2)$、列が $\alpha_2$ の行列にする。0始まりの実装上の行位置は $\alpha_1n_2+i_2$ である。

$$
C_L((\alpha_1,i_2),\alpha_2)=C(\alpha_1,i_2,\alpha_2),
\qquad C_L\in\mathbb R^{(r_1n_2)\times r_2}.
$$

reduced QRを $C_L=Q_L T_L$ と書く。$q_2=\min(r_1n_2,r_2)$ とすれば、一般に $Q_L$ は $(r_1n_2)\times q_2$、$T_L$ は $q_2\times r_2$ であり、$Q_L^TQ_L=I_{q_2}$ である。$r_2\le r_1n_2$ の場合には $q_2=r_2$ となる。

$$
G_2^{[L]}(\alpha_1,i_2,\beta_2)=Q_L((\alpha_1,i_2),\beta_2),
\qquad
G_3^{[C]}(\beta_2,i_3,1)
=\sum_{\alpha_2=1}^{r_2}T_L(\beta_2,\alpha_2)G_3^{[R]}(\alpha_2,i_3,1).
$$

したがって、新しい第2コアは $(r_1,n_2,q_2)$、第3中心コアは $(q_2,n_3,1)$ となる。$q_2=r_2$ ならshapeは元と同じである。$q_2<r_2$ でも $C_L=Q_LT_L$ は厳密な因子分解なので、三角因子を吸収する限り近似誤差は生じない。ただし、この場合はボンド次元が変わるため、同じサイズの可逆行列を挿入するGauge変換とは区別する。

QRの行列積を要素へ戻すと、

$$
C(\alpha_1,i_2,\alpha_2)
=\sum_{\beta_2=1}^{q_2}
G_2^{[L]}(\alpha_1,i_2,\beta_2)T_L(\beta_2,\alpha_2).
$$

これを元のTTへ代入し、有限和の順序を変える。

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1,\alpha_2,\beta_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta_2)
T_L(\beta_2,\alpha_2)
G_3^{[R]}(\alpha_2,i_3,1)\\
&=\sum_{\alpha_1,\beta_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta_2)
\left[\sum_{\alpha_2}T_L(\beta_2,\alpha_2)G_3^{[R]}(\alpha_2,i_3,1)\right]\\
&=\sum_{\alpha_1,\beta_2}
G_1^{[L]}(1,i_1,\alpha_1)
G_2^{[L]}(\alpha_1,i_2,\beta_2)
G_3^{[C]}(\beta_2,i_3,1).
\end{aligned}
$$

また、$Q_L^TQ_L=I$ は成分で

$$
\sum_{\alpha_1,i_2}
G_2^{[L]}(\alpha_1,i_2,\beta_2)
G_2^{[L]}(\alpha_1,i_2,\gamma_2)
=\delta_{\beta_2\gamma_2}
$$

である。第2コアが左直交となり、中心が第3サイトへ移った。第3コアは中心になったので、吸収後に右直交である必要はない。

## 3. 第2サイトから左へ移す

逆向きでは $C$ を、行が $\alpha_1$、列が $(i_2,\alpha_2)$ の行列にする。0始まりの列位置は $i_2r_2+\alpha_2$ である。

$$
C_R(\alpha_1,(i_2,\alpha_2))=C(\alpha_1,i_2,\alpha_2),
\qquad C_R\in\mathbb R^{r_1\times(n_2r_2)}.
$$

求めるのは行直交な第2コアである。通常のQRは列を直交化するので、転置に適用する。

$$
C_R^T=Q_RT_R,
\qquad C_R=T_R^TQ_R^T,
\qquad Q_R^TQ_R=I_{q_1},
\qquad q_1=\min(n_2r_2,r_1).
$$

$Q_R$ は $(n_2r_2)\times q_1$、$T_R$ は $q_1\times r_1$ である。$r_1\le n_2r_2$ なら $q_1=r_1$ となる。

ここには役割の異なる二回の転置がある。最初の $C_R^T$ は、$C_R$ の各**行**をQRが直交化できる**列**として渡すためである。QR後の $Q_R^T$ は、得られた列をコアの右展開の添字順へ戻すためである。直交化したベクトルの成分が失われるわけではない。複合添字 $p=(i_2,\alpha_2)$ と置けば、戻した行の成分は $Q_R^T(\beta_1,p)=Q_R(p,\beta_1)$ であり、

$$
\begin{aligned}
\sum_p Q_R^T(\beta_1,p)Q_R^T(\gamma_1,p)
&=\sum_p Q_R(p,\beta_1)Q_R(p,\gamma_1)\\
&=(Q_R^TQ_R)_{\beta_1,\gamma_1}
=\delta_{\beta_1,\gamma_1}.
\end{aligned}
$$

従って「列直交な $Q_R$ を転置したから直交性が壊れる」のではなく、同じ成分を**行直交な $Q_R^T$** として読んでいる。この添字の確認は、右直交性を行列の見た目だけで判断しないために必要である。

$$
G_2^{[R]}(\beta_1,i_2,\alpha_2)=Q_R^T(\beta_1,(i_2,\alpha_2)),
\qquad
G_1^{[C]}(1,i_1,\beta_1)
=\sum_{\alpha_1=1}^{r_1}G_1^{[L]}(1,i_1,\alpha_1)T_R^T(\alpha_1,\beta_1).
$$

第1中心コアのshapeは $(1,n_1,q_1)$、第2右直交コアは $(q_1,n_2,r_2)$ となる。右直交性と再構成はそれぞれ

$$
\sum_{i_2,\alpha_2}
G_2^{[R]}(\beta_1,i_2,\alpha_2)
G_2^{[R]}(\gamma_1,i_2,\alpha_2)
=\delta_{\beta_1\gamma_1},
$$

$$
C(\alpha_1,i_2,\alpha_2)
=\sum_{\beta_1=1}^{q_1}T_R^T(\alpha_1,\beta_1)G_2^{[R]}(\beta_1,i_2,\alpha_2)
$$

で確認できる。後者を元のTTへ代入すると、$\alpha_1$ の和が $G_1^{[C]}$ の定義に一致する。

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1,\beta_1,\alpha_2}
G_1^{[L]}(1,i_1,\alpha_1)T_R^T(\alpha_1,\beta_1)
G_2^{[R]}(\beta_1,i_2,\alpha_2)G_3^{[R]}(\alpha_2,i_3,1)\\
&=\sum_{\beta_1,\alpha_2}
G_1^{[C]}(1,i_1,\beta_1)
G_2^{[R]}(\beta_1,i_2,\alpha_2)G_3^{[R]}(\alpha_2,i_3,1).
\end{aligned}
$$

第1コアは中心になったので、吸収後の左直交性は要求しない。

## 4. 同じ小さい例を左右の移動まで計算する

$n_1=n_2=n_3=r_1=r_2=2$ とし、境界を省略して第1・第3コアをそれぞれ $I_2$ とする。中心コアの全要素を、$i_2$ ごとの行列で

$$
C(:,1,:)=\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
C(:,2,:)=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

と定める。$G_1^{[L]}(1,i_1,\alpha_1)=\delta_{i_1\alpha_1}$、$G_3^{[R]}(\alpha_2,i_3,1)=\delta_{\alpha_2i_3}$ なので、$X(i_1,i_2,i_3)=C(i_1,i_2,i_3)$ である。

右移動に使う左展開では、行を $(1,1),(1,2),(2,1),(2,2)$ の順に並べる。

$$
C_L=\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}
=\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}}_{Q_L}
\underbrace{\begin{pmatrix}\sqrt2&0\\0&\sqrt2\end{pmatrix}}_{T_L}.
$$

従って新しい第2コアと第3中心コアの全要素は

$$
G_2^{[L]}(:,1,:)=\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
G_2^{[L]}(:,2,:)=\frac1{\sqrt2}\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad
G_3^{[C]}(:,:,1)=
\begin{pmatrix}\sqrt2&0\\0&\sqrt2\end{pmatrix}
$$

となる。直交性も積を省略せず書けば、

$$
Q_L^TQ_L
=\frac12
\begin{pmatrix}1&0&0&1\\0&1&1&0\end{pmatrix}
\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}
=\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$$

例えば $X(2,2,1)$ は移動前に $C(2,2,1)=1$、移動後には $(1/\sqrt2)\sqrt2=1$ である。他の非零要素も同じ形で保たれ、零要素は混合しない。

左移動に使う右展開では、列を $(1,1),(1,2),(2,1),(2,2)$ の順に並べる。

$$
C_R=\begin{pmatrix}1&0&0&1\\0&1&1&0\end{pmatrix},
\qquad
C_R^T=\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}}_{Q_R}
\underbrace{\begin{pmatrix}\sqrt2&0\\0&\sqrt2\end{pmatrix}}_{T_R}.
$$

よって第2右直交コアと第1中心コアの全要素は

$$
G_2^{[R]}(:,1,:)=\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
G_2^{[R]}(:,2,:)=\frac1{\sqrt2}\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad
G_1^{[C]}(1,:,:)=\begin{pmatrix}\sqrt2&0\\0&\sqrt2\end{pmatrix}.
$$

右展開の行Gramも

$$
G_2^{[R]\langle R\rangle}(G_2^{[R]\langle R\rangle})^T
=\frac12
\begin{pmatrix}1&0&0&1\\0&1&1&0\end{pmatrix}
\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}
=I_2.
$$

こちらも $X(2,2,1)=\sqrt2(1/\sqrt2)=1$ であり、全要素は

$$
X(:,1,:)=\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
X(:,2,:)=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

のままである。QRの符号は一意でないため、この手計算の $Q,T$ とライブラリの出力要素が同符号であることではなく、積、直交性、再構成を確認する。

### 非対角の三角因子も要素で追う

前の例では $T_L=\sqrt2 I_2$ なので、吸収は各成分のスケール変更だけに見える。非対角成分がある場合も同じ式が働くことを、$n_1=r_1=1,n_2=r_2=n_3=2$ の別の小さい局所例で確かめる。$G_1^{[L]}=1$、$G_3^{[R]}(:,:,1)=I_2$ とし、

$$
C(1,1,:)=(1,1),\qquad C(1,2,:)=(0,1)
$$

と置く。左展開と、そのreduced QRの一つは

$$
C_L=\begin{pmatrix}1&1\\0&1\end{pmatrix}
=\underbrace{\begin{pmatrix}1&0\\0&1\end{pmatrix}}_{Q_L}
\underbrace{\begin{pmatrix}1&1\\0&1\end{pmatrix}}_{T_L}.
$$

ここでは $Q_L^TQ_L=I_2$ であり、$G_2^{[L]}(1,1,:)=(1,0)$、$G_2^{[L]}(1,2,:)=(0,1)$ となる。三角因子を第3コアへ吸収する積を全要素で書くと、

$$
\begin{aligned}
G_3^{[C]}(:,:,1)
&=T_LG_3^{[R]}(:,:,1)\\
&=\begin{pmatrix}1&1\\0&1\end{pmatrix}
\begin{pmatrix}1&0\\0&1\end{pmatrix}
=\begin{pmatrix}1&1\\0&1\end{pmatrix}.
\end{aligned}
$$

従って $G_3^{[C]}(1,1,1)=1$、$G_3^{[C]}(1,2,1)=1$、$G_3^{[C]}(2,1,1)=0$、$G_3^{[C]}(2,2,1)=1$ である。新しい左コアとの積は

$$
\begin{pmatrix}1&0\\0&1\end{pmatrix}
\begin{pmatrix}1&1\\0&1\end{pmatrix}
=\begin{pmatrix}1&1\\0&1\end{pmatrix}
=C_LG_3^{[R]}(:,:,1)
$$

となり、$X(1,1,1)=1$、$X(1,1,2)=1$、$X(1,2,1)=0$、$X(1,2,2)=1$ の四つの物理成分をすべて保存する。$T_L(1,2)=1$ は、第3コアの第2行から第1行へ寄与する項であり、単なる各行の倍率ではない。

## 5. 端点と一般の $d$ 階

$d$ 階でサイト $c$ を中心にすると、$k<c$ のコアだけに左直交、$k>c$ のコアだけに右直交を要求する。このTT/MPS全体の配置が混合正準形（mixed-canonical form）であり、$G_c$ が直交中心（orthogonality center）である。中心 $G_c$ 自体には条件を課さない。$c=1$ なら左側は空、$c=d$ なら右側は空であり、空の側に架空の直交条件を追加しない。用語の定義は [[39_TT_MPSの混合正準形への導入]] と対応する。

中心の左側の物理添字を $I_L=(i_1,\ldots,i_{c-1})$、右側を $I_R=(i_{c+1},\ldots,i_d)$ とまとめる。左ブロック $\mathcal L_{c-1}$ と右ブロック $\mathcal R_c$ の、まだ行列化しないshapeは

$$
\mathcal L_{c-1}\in\mathbb R^{n_1\times\cdots\times n_{c-1}\times r_{c-1}},
\qquad
\mathcal R_c\in\mathbb R^{r_c\times n_{c+1}\times\cdots\times n_d}.
$$

$I_L,I_R$ をそれぞれ複合添字にすれば、左は $\bigl(\prod_{k=1}^{c-1}n_k\bigr)\times r_{c-1}$ 行列、右は $r_c\times\bigl(\prod_{k=c+1}^{d}n_k\bigr)$ 行列になる。この二つと中心コアの積を、消えるボンド添字まで書くと

$$
X(I_L,i_c,I_R)
=\sum_{a=1}^{r_{c-1}}\sum_{b=1}^{r_c}
\mathcal L_{c-1}(I_L,a)G_c^{[C]}(a,i_c,b)\mathcal R_c(b,I_R).
$$

ここで $a,b$ は和で消え、$I_L,i_c,I_R$ が全テンソルの物理添字として残る。$c=1$ なら $I_L$ は空で $r_0=1$、$\mathcal L_0(1,1)=1$ と読み、$c=d$ なら $I_R$ は空で $r_d=1$、$\mathcal R_d(1,1)=1$ と読む。空積は1なので、同じ式が端点にも使える。例えば3階では、第1サイト中心が $G_1^{[C]}G_2^{[R]}G_3^{[R]}$、第3サイト中心が $G_1^{[L]}G_2^{[L]}G_3^{[C]}$ である。

右端から作る右ブロックを $\mathcal R_k(\alpha_k,i_{k+1},\ldots,i_d)$ とする。$\mathcal R_{d-1}(\alpha_{d-1},i_d)=G_d^{[R]}(\alpha_{d-1},i_d,1)$ は右直交性から行Gramが単位行列である。これを帰納法の出発点とする。

$$
\mathcal R_k(\alpha_k,i_{k+1},I)
=\sum_{\alpha_{k+1}}
G_{k+1}^{[R]}(\alpha_k,i_{k+1},\alpha_{k+1})
\mathcal R_{k+1}(\alpha_{k+1},I),
\qquad I=(i_{k+2},\ldots,i_d).
$$

$\mathcal R_{k+1}$ の行Gramが単位行列と仮定して積を展開すると、

$$
\begin{aligned}
\sum_{i_{k+1},I}\mathcal R_k(\alpha_k,i_{k+1},I)\mathcal R_k(\beta_k,i_{k+1},I)
&=\sum_{i_{k+1},\mu,\nu}
G_{k+1}^{[R]}(\alpha_k,i_{k+1},\mu)
G_{k+1}^{[R]}(\beta_k,i_{k+1},\nu)
\underbrace{\sum_I\mathcal R_{k+1}(\mu,I)\mathcal R_{k+1}(\nu,I)}_{\delta_{\mu\nu}}\\
&=\sum_{i_{k+1},\mu}
G_{k+1}^{[R]}(\alpha_k,i_{k+1},\mu)
G_{k+1}^{[R]}(\beta_k,i_{k+1},\mu)
=\delta_{\alpha_k\beta_k}.
\end{aligned}
$$

従って第1サイト中心のときも、右側全体は正規直交基底になる。左側も途中式を確認する。左端からの左ブロックを $\mathcal L_k(i_1,\ldots,i_k,\alpha_k)$ と書く。$\mathcal L_1(i_1,\alpha_1)=G_1^{[L]}(1,i_1,\alpha_1)$ が帰納の出発点である。

$$
\mathcal L_k(J,i_k,\alpha_k)
=\sum_{\alpha_{k-1}}
\mathcal L_{k-1}(J,\alpha_{k-1})
G_k^{[L]}(\alpha_{k-1},i_k,\alpha_k),
\qquad J=(i_1,\ldots,i_{k-1}).
$$

$\mathcal L_{k-1}$ の列Gramが単位行列なら、

$$
\begin{aligned}
\sum_{J,i_k}\mathcal L_k(J,i_k,\alpha_k)\mathcal L_k(J,i_k,\beta_k)
&=\sum_{i_k,\mu,\nu}
G_k^{[L]}(\mu,i_k,\alpha_k)G_k^{[L]}(\nu,i_k,\beta_k)
\underbrace{\sum_J\mathcal L_{k-1}(J,\mu)\mathcal L_{k-1}(J,\nu)}_{\delta_{\mu\nu}}\\
&=\sum_{i_k,\mu}G_k^{[L]}(\mu,i_k,\alpha_k)G_k^{[L]}(\mu,i_k,\beta_k)
=\delta_{\alpha_k\beta_k}.
\end{aligned}
$$

従って、両側が存在する $1<c<d$ では $\sum_{I_L}\mathcal L_{c-1}(I_L,a)\mathcal L_{c-1}(I_L,a')=\delta_{aa'}$、$\sum_{I_R}\mathcal R_c(b,I_R)\mathcal R_c(b',I_R)=\delta_{bb'}$ である。全テンソルを二回掛ける際にはボンド添字を独立な $a,b$ と $a',b'$ にしてから和の順を変える。

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{I_L,i_c,I_R}
\left[\sum_{a,b}\mathcal L_{c-1}(I_L,a)G_c^{[C]}(a,i_c,b)\mathcal R_c(b,I_R)\right]
\left[\sum_{a',b'}\mathcal L_{c-1}(I_L,a')G_c^{[C]}(a',i_c,b')\mathcal R_c(b',I_R)\right]\\
&=\sum_{i_c,a,b,a',b'}G_c^{[C]}(a,i_c,b)G_c^{[C]}(a',i_c,b')
\underbrace{\sum_{I_L}\mathcal L_{c-1}(I_L,a)\mathcal L_{c-1}(I_L,a')}_{\delta_{aa'}}
\underbrace{\sum_{I_R}\mathcal R_c(b,I_R)\mathcal R_c(b',I_R)}_{\delta_{bb'}}\\
&=\sum_{a,i_c,b}G_c^{[C]}(a,i_c,b)^2=\|G_c^{[C]}\|_F^2.
\end{aligned}
$$

この等式は中心コアの各成分を単純に足したからではなく、二つの独立したボンド和の交差項を左右のGramで消した結果である。第2サイト中心の3階で各和をさらに細かく追う場合は [[40_TT_MPSの中心ノルムと内積の導出]] の導出を参照する。

第 $d$ サイト中心なら左側全体が正規直交基底となる。例えば $X(I,i_d)=\sum_\alpha\mathcal L_{d-1}(I,\alpha)G_d^{[C]}(\alpha,i_d,1)$ なので、

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{I,i_d}\sum_{\alpha,\beta}
\mathcal L_{d-1}(I,\alpha)\mathcal L_{d-1}(I,\beta)
G_d^{[C]}(\alpha,i_d,1)G_d^{[C]}(\beta,i_d,1)\\
&=\sum_{i_d,\alpha,\beta}\delta_{\alpha\beta}
G_d^{[C]}(\alpha,i_d,1)G_d^{[C]}(\beta,i_d,1)
=\|G_d^{[C]}\|_F^2.
\end{aligned}
$$

第1サイト中心では左ブロックは空である。右ブロックの行Gramを明示して消すと、

$$
\begin{aligned}
X(i_1,I_R)&=\sum_bG_1^{[C]}(1,i_1,b)\mathcal R_1(b,I_R),\\
\|X\|_F^2
&=\sum_{i_1,I_R,b,b'}G_1^{[C]}(1,i_1,b)G_1^{[C]}(1,i_1,b')
\mathcal R_1(b,I_R)\mathcal R_1(b',I_R)\\
&=\sum_{i_1,b,b'}G_1^{[C]}(1,i_1,b)G_1^{[C]}(1,i_1,b')
\underbrace{\sum_{I_R}\mathcal R_1(b,I_R)\mathcal R_1(b',I_R)}_{\delta_{bb'}}\\
&=\sum_{i_1,b}G_1^{[C]}(1,i_1,b)^2=\|G_1^{[C]}\|_F^2.
\end{aligned}
$$

従って端点でも内部でも、中心以外に要求する直交性だけで全体ノルムは中心へ局所化する。

## 6. 区別しておくこと

- QRと隣への因子吸収は、打ち切らなければ全テンソルを保存する。中心移動そのものは圧縮でもNNの精度改善でもない。
- $Q^TQ=I$ は左直交、$QQ^T=I$ ではない。右直交では $Q^T$ の行Gramを確認する。
- 正準形は中心の左右に条件を課す表現であり、同じ $X$ の表現はQR符号やGaugeの選び方で一意ではない。
- 複素MPSでは転置を随伴に置き換える。ここでのPyTorch例と既存のTT実装契約は実数Tensorを対象とする。
- `reshape`、QR、`tensordot` の軸対応と検証コードは [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] を参照する。
