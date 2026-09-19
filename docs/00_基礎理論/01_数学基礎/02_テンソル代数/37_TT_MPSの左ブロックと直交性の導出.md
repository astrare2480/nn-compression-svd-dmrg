---
title: TT・MPSの左ブロックと直交性の導出
aliases:
  - MPSの左ブロック基底
  - 左部分収縮の直交性
tags:
  - TT
  - MPS
  - orthogonality
  - derivation
---

# TT・MPSの左ブロックと直交性の導出

## サマリー

[[36_TT_MPSのGauge自由度と左QR直交化]] で第1・第2コアを左直交化したら、2コアを縮約した左ブロックの列も正規直交になる。本章では「各コアの直交性」から「ブロック状態の直交性」へ進む和の途中式を省略しない。

左部分収縮は $\mathcal L_2$ と書く。[[34_基底変換とTT-rank不変性]] の $L_2=U\otimes I_{n_2}$ とは別の行列である。

## 1. 定義・複合添字・列の意味

左直交なコアを $A_1(i_1,\alpha_1)=G_1^{[L]}(1,i_1,\alpha_1)$ と
$G_2^{[L]}(\alpha_1,i_2,\alpha_2)$ とする。第2cutでの左部分収縮は

$$
\mathcal L_2(i_1,i_2,\alpha_2)
=\sum_{\alpha_1=1}^{r_1}A_1(i_1,\alpha_1)G_2^{[L]}(\alpha_1,i_2,\alpha_2).
$$

3階配列としては $(n_1,n_2,r_2)$、行 $(i_1,i_2)$・列 $\alpha_2$ の行列としては

$$
\mathcal L_2\in\mathbb R^{(n_1n_2)\times r_2},\qquad
p=(i_1-1)n_2+i_2
$$

である。$\alpha_2$ を固定した1列は、左側の物理空間 $\mathcal H_1\otimes\mathcal H_2$ に属する一つの状態の係数を持つ。

$$
|L_{\alpha_2}\rangle
=\sum_{i_1=1}^{n_1}\sum_{i_2=1}^{n_2}
\mathcal L_2(i_1,i_2,\alpha_2)|i_1\rangle\otimes|i_2\rangle.
$$

ここで $r_2$ 本の列が直交するのであって、$G_1$ と $G_2$ という別のコア同士の内積を取っているわけではない。

### 状態を列へ並べた行列を係数で展開する

状態 $|L_{\alpha_2}\rangle$ を列に並べた行列を考える。
行列の要素自体はketではなく、そのketの物理基底に関する係数である。

$$
\ell_{\alpha_2}
=\begin{pmatrix}
\mathcal L_2(1,1,\alpha_2)\\
\mathcal L_2(1,2,\alpha_2)\\
\vdots\\
\mathcal L_2(n_1,n_2,\alpha_2)
\end{pmatrix},\qquad
\mathcal L_2=(\ell_1\ \ell_2\ \cdots\ \ell_{r_2}).
$$

同じ模式図をscalar成分へ戻すと

$$
\mathcal L_2=
\begin{pmatrix}
\mathcal L_2(1,1,1)&\cdots&\mathcal L_2(1,1,r_2)\\
\mathcal L_2(1,2,1)&\cdots&\mathcal L_2(1,2,r_2)\\
\vdots&\ddots&\vdots\\
\mathcal L_2(n_1,n_2,1)&\cdots&\mathcal L_2(n_1,n_2,r_2)
\end{pmatrix}.
$$

第 $(\beta_2,\alpha_2)$ 成分は $\ell_{\beta_2}^T\ell_{\alpha_2}$ であり、
これが実数係数での $\langle L_{\beta_2}|L_{\alpha_2}\rangle$ になる。
異なる列は異なるボンド状態を表すが、どの列も同じ物理基底順を使う。
第1サイトだけの $A_1$ でも、
$A_1=(A_1(:,1)\ \cdots\ A_1(:,r_1))$ という同じ列配置を意味する。
第1サイトの状態と2サイトブロックの状態では物理空間の次元が違うため、
模式図の $|L_\alpha\rangle$ という同じ記号だけから同じ状態だと判断しない。

## 2. ブロックのGram行列をコアまで展開する

ここで内積を取る対象は、「列 $\alpha_2$ が表す左状態」と「列 $\beta_2$ が表す左状態」である。
$\alpha_2,\beta_2$ は比較したい二つの状態ラベルなので、Gram行列の成分として最後まで残す。
一方、$i_1,i_2$ はその状態の係数を記述する物理基底の成分番号であり、内積では全成分について和を取る。
「添字がたくさんあるからすべてを縮約する」のではなく、比較する列のラベルを残して、ベクトルの成分だけを和で消す。

さらに、二つのブロックの内部ボンドには、初めから別々の和の添字 $\alpha_1,\beta_1$ を付ける。
各ブロックの定義には独立した和があるため、積を作った直後から $\alpha_1=\beta_1$ と置いてはいけない。
第1コアの直交性を使った後に初めて $\delta_{\alpha_1,\beta_1}$ が現れ、そのデルタが二つの和を一つへまとめる。
この順序が、以下の3行目から5行目への変形を正当化する。

前提となる二つの左直交条件は

$$
\sum_{i_1=1}^{n_1}A_1(i_1,\alpha_1)A_1(i_1,\beta_1)
=\delta_{\alpha_1,\beta_1},
$$

$$
\sum_{\alpha_1=1}^{r_1}\sum_{i_2=1}^{n_2}
G_2^{[L]}(\alpha_1,i_2,\alpha_2)G_2^{[L]}(\alpha_1,i_2,\beta_2)
=\delta_{\alpha_2,\beta_2}.
$$

Gram行列の $(\alpha_2,\beta_2)$ 成分へ、二つの $\mathcal L_2$ の定義を代入する。

$$
\begin{aligned}
(\mathcal L_2^T\mathcal L_2)_{\alpha_2,\beta_2}
&=\sum_{i_1,i_2}\mathcal L_2(i_1,i_2,\alpha_2)\mathcal L_2(i_1,i_2,\beta_2)\\
&=\sum_{i_1,i_2}
\left[\sum_{\alpha_1}A_1(i_1,\alpha_1)G_2^{[L]}(\alpha_1,i_2,\alpha_2)\right]
\left[\sum_{\beta_1}A_1(i_1,\beta_1)G_2^{[L]}(\beta_1,i_2,\beta_2)\right]\\
&=\sum_{i_2,\alpha_1,\beta_1}
G_2^{[L]}(\alpha_1,i_2,\alpha_2)G_2^{[L]}(\beta_1,i_2,\beta_2)
\left[\sum_{i_1}A_1(i_1,\alpha_1)A_1(i_1,\beta_1)\right]\\
&=\sum_{i_2,\alpha_1,\beta_1}
G_2^{[L]}(\alpha_1,i_2,\alpha_2)G_2^{[L]}(\beta_1,i_2,\beta_2)
\delta_{\alpha_1,\beta_1}\\
&=\sum_{i_2,\alpha_1}
G_2^{[L]}(\alpha_1,i_2,\alpha_2)G_2^{[L]}(\alpha_1,i_2,\beta_2)\\
&=\delta_{\alpha_2,\beta_2}.
\end{aligned}
$$

最初に $i_1$ の和で第1コアのGramを消し、次に $(\alpha_1,i_2)$ の和で第2コアのGramを消している。従って

$$
\boxed{\mathcal L_2^T\mathcal L_2=I_{r_2}}.
$$

第1コアの直交性だけでこの結論を出すことはできない。第2コアの左直交条件も必要である。

## 3. bra・ketから同じ内積を確認する

物理基底は $\langle j_1,j_2|i_1,i_2\rangle=\delta_{j_1,i_1}\delta_{j_2,i_2}$ とする。実数係数の場合、

$$
\begin{aligned}
\langle L_{\beta_2}|L_{\alpha_2}\rangle
&=\sum_{j_1,j_2,i_1,i_2}
\mathcal L_2(j_1,j_2,\beta_2)\mathcal L_2(i_1,i_2,\alpha_2)
\langle j_1,j_2|i_1,i_2\rangle\\
&=\sum_{j_1,j_2,i_1,i_2}
\mathcal L_2(j_1,j_2,\beta_2)\mathcal L_2(i_1,i_2,\alpha_2)
\delta_{j_1,i_1}\delta_{j_2,i_2}\\
&=\sum_{i_1,i_2}\mathcal L_2(i_1,i_2,\beta_2)\mathcal L_2(i_1,i_2,\alpha_2)\\
&=(\mathcal L_2^T\mathcal L_2)_{\beta_2,\alpha_2}
=\delta_{\beta_2,\alpha_2}.
\end{aligned}
$$

DMRGでは、これは左ブロックの有効状態 $|L_{\alpha_2}\rangle$ が正規直交な基底として使えることを意味する。ただし、左直交基底が自動的に縮約密度行列の固有基底になるわけではない。Schmidt基底と係数の対角化には、切断に対応するSVDが別に必要である。

## 4. 小さい2コアの全要素から $\mathcal L_2$ を作る

$n_1=n_2=r_1=r_2=2$ とし、本章独自の例を用いる。

$$
A_1=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad
H_1=G_2^{[L]}(:,1,:)=\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix},\qquad
H_2=G_2^{[L]}(:,2,:)=\frac1{\sqrt2}\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

第2コアの左unfoldingは、行 $(\alpha_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ の順で

$$
A_2=\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix},\qquad
A_1^TA_1=I_2,\qquad
A_2^TA_2=\frac12\begin{pmatrix}1+0+0+1&0\\0&0+1+1+0\end{pmatrix}=I_2.
$$

ブロックの行を $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ とすると、各行は

$$
\begin{aligned}
\mathcal L_2((1,1),:)&=A_1(1,:)H_1=\tfrac12\begin{pmatrix}1&1\end{pmatrix},\\
\mathcal L_2((1,2),:)&=A_1(1,:)H_2=\tfrac12\begin{pmatrix}1&1\end{pmatrix},\\
\mathcal L_2((2,1),:)&=A_1(2,:)H_1=\tfrac12\begin{pmatrix}1&-1\end{pmatrix},\\
\mathcal L_2((2,2),:)&=A_1(2,:)H_2=\tfrac12\begin{pmatrix}-1&1\end{pmatrix}.
\end{aligned}
$$

従って、全8要素は

$$
\boxed{\mathcal L_2=\frac12\begin{pmatrix}1&1\\1&1\\1&-1\\-1&1\end{pmatrix}}.
$$

3階配列で表示するなら、$\mathcal L_2(1,:,:)=\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix}$、
$\mathcal L_2(2,:,:)=\tfrac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ である。`reshape(4, 2)` はこの順で2枚を縦に並べ、値は変えない。

Gram行列の全4成分を展開すると

$$
\begin{aligned}
\mathcal L_2^T\mathcal L_2
&=\frac14\begin{pmatrix}
1^2+1^2+1^2+(-1)^2&1\cdot1+1\cdot1+1\cdot(-1)+(-1)\cdot1\\
1\cdot1+1\cdot1+(-1)\cdot1+1\cdot(-1)&1^2+1^2+(-1)^2+1^2
\end{pmatrix}\\
&=\frac14\begin{pmatrix}4&0\\0&4\end{pmatrix}=I_2.
\end{aligned}
$$

列に対応する左ブロック状態も全要素で書ける。

$$
|L_1\rangle=\tfrac12(|1,1\rangle+|1,2\rangle+|2,1\rangle-|2,2\rangle),
$$

$$
|L_2\rangle=\tfrac12(|1,1\rangle+|1,2\rangle-|2,1\rangle+|2,2\rangle).
$$

4次元の物理空間に2本の正規直交状態を埋め込んでいる。

## 5. $U\otimes I$ と2コア左ブロックを区別する

第2コアの左unfoldingを $A_2$ とすると、一般に

$$
\boxed{\mathcal L_2=(A_1\otimes I_{n_2})A_2}.
$$

成分の途中式は

$$
\begin{aligned}
[(A_1\otimes I_{n_2})A_2]_{(i_1,i_2),\alpha_2}
&=\sum_{\alpha_1,j_2}A_1(i_1,\alpha_1)\delta_{i_2,j_2}
G_2^{[L]}(\alpha_1,j_2,\alpha_2)\\
&=\sum_{\alpha_1}A_1(i_1,\alpha_1)G_2^{[L]}(\alpha_1,i_2,\alpha_2)
=\mathcal L_2(i_1,i_2,\alpha_2).
\end{aligned}
$$

前節の数値を代入すると、全行列で

$$
\mathcal L_2
=\underbrace{\frac1{\sqrt2}\begin{pmatrix}
1&0&1&0\\0&1&0&1\\1&0&-1&0\\0&1&0&-1
\end{pmatrix}}_{A_1\otimes I_2}
\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\\0&1\\1&0\end{pmatrix}}_{A_2}
=\frac12\begin{pmatrix}1&1\\1&1\\1&-1\\-1&1\end{pmatrix}.
$$

[[34_基底変換とTT-rank不変性]] の $L_2=U\otimes I_{n_2}$ は、行 $(i_1,i_2)$・列 $(\alpha_1,j_2)$ の $(n_1n_2)\times(r_1n_2)$ 行列である。今回の $\mathcal L_2$ はさらに第2コアを掛けた $(n_1n_2)\times r_2$ 行列である。同じ「左側の写像」でも段階が違う。

## 6. ブロックのGramと物理空間の射影

同じ数値例について、積の順を反対にすると

$$
P_L=\mathcal L_2\mathcal L_2^T
=\frac14\begin{pmatrix}
2&2&0&0\\2&2&0&0\\0&0&2&-2\\0&0&-2&2
\end{pmatrix}
=\frac12\begin{pmatrix}
1&1&0&0\\1&1&0&0\\0&0&1&-1\\0&0&-1&1
\end{pmatrix}\ne I_4.
$$

例えば $(1,3)$ 成分は $(1\cdot1+1\cdot(-1))/4=0$、$(3,4)$ 成分は $(1\cdot(-1)+(-1)\cdot1)/4=-1/2$ である。

$$
P_L^2=\mathcal L_2(\mathcal L_2^T\mathcal L_2)\mathcal L_2^T
=\mathcal L_2I_2\mathcal L_2^T=P_L.
$$

$\mathcal L_2^T\mathcal L_2$ はボンドでラベルした状態のGram、$P_L$ は物理空間で保持する2次元部分空間への射影である。[[35_TT_MPSの等長写像と射影]] の一般論を2コアブロックへ適用したものになる。

## 7. 一般の左ブロックへの帰納

左ブロックを $\mathcal L_{k-1}$、次のコアを $G_k^{[L]}$ として

$$
\mathcal L_k(i_1,\ldots,i_k,\alpha_k)
=\sum_{a=1}^{r_{k-1}}\mathcal L_{k-1}(i_1,\ldots,i_{k-1},a)G_k^{[L]}(a,i_k,\alpha_k)
$$

と置く。帰納法の仮定 $\mathcal L_{k-1}^T\mathcal L_{k-1}=I$ を使うと

$$
\begin{aligned}
(\mathcal L_k^T\mathcal L_k)_{\alpha_k,\beta_k}
&=\sum_{i_k,a,b}G_k^{[L]}(a,i_k,\alpha_k)G_k^{[L]}(b,i_k,\beta_k)
\left[\sum_{i_1,\ldots,i_{k-1}}\mathcal L_{k-1}(i_1,\ldots,i_{k-1},a)\mathcal L_{k-1}(i_1,\ldots,i_{k-1},b)\right]\\
&=\sum_{i_k,a,b}G_k^{[L]}(a,i_k,\alpha_k)G_k^{[L]}(b,i_k,\beta_k)\delta_{a,b}\\
&=\sum_{i_k,a}G_k^{[L]}(a,i_k,\alpha_k)G_k^{[L]}(a,i_k,\beta_k)
=\delta_{\alpha_k,\beta_k}.
\end{aligned}
$$

左端 $\mathcal L_0=(1)$ から繰り返せば、左直交化した範囲の全ブロックで列直交性が成立する。複素係数ではbra側の係数を共役にし、Gramは $\mathcal L_k^\dagger\mathcal L_k$ と読む。

次は [[38_TT_MPSの右QR直交化と右ブロック]] で、状態ラベルを行に置いた場合を比較する。

## 参考資料

- [Schollwöck：DMRG in the age of matrix product states](https://arxiv.org/abs/1008.3477)：canonical MPSと正規直交なブロック状態。
