---
title: TT-matrixのbond dimensionと表現能力
tags:
  - TT-matrix
  - TT-rank
  - bond-dimension
  - unfolding
  - low-rank-approximation
---

# TT-matrixのbond dimensionと表現能力

## 1. このノートで明らかにすること

TT-matrixの内部rankを小さくすると、なぜ保存量だけでなく表現能力も制限されるのかを、coreの要素式から順に説明する。

中心となる関係は、TT-matrixを第 $k$ bondで切ったunfoldingについての

$$
A=LR,
$$

$$
A
=
\sum_{a=1}^{r_k}L_{:,a}R_{a,:},
$$

$$
\operatorname{rank}(A)
\le
r_k
$$

である。

最後の不等式は、次の二通りに読める。

$$
\boxed{
\begin{array}{l}
\text{作る側：bond dimensionが }r_k\text{ なら、作れるunfoldingのrankは最大 }r_k,\\
\text{再現する側：rank }s\text{ のunfoldingを完全再現するには }r_k\ge s\text{ が必要。}
\end{array}
}
$$

元のdense重みのrankと、TT-matrixを特定のbondで切ったunfoldingのrankは同じ量ではない。この違いも添字から確認する。

## 2. bond dimensionとは何か

TT-matrixの第 $k$ coreを

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k}
$$

とする。次のcoreは

$$
G^{(k+1)}
\in
\mathbb R^{r_k\times m_{k+1}\times n_{k+1}\times r_{k+1}}
$$

である。前のcoreの最後の軸と、次のcoreの最初の軸は同じ大きさ $r_k$ を持つ。

$$
G^{(k)}
\ \xleftrightarrow{\qquad r_k\qquad}\
G^{(k+1)}.
$$

この共有軸がbondであり、その軸の要素数 $r_k$ がbond dimensionである。ここでいうdimensionはcoreの軸数ではない。coreは4階テンソルだが、bond dimensionはそのうちの共有軸のサイズである。

TTではこの内部軸サイズをTT-rank、MPSではbond dimensionと呼ぶ。境界rankは

$$
r_0=r_d=1
$$

であり、内部bondは

$$
r_1,\ldots,r_{d-1}
$$

である。

## 3. 2-coreの要素式では、bond dimensionが和の項数になる

まずcoreが2個の場合だけを見る。

$$
G^{(1)}
\in
\mathbb R^{1\times m_1\times n_1\times r_1},
\qquad
G^{(2)}
\in
\mathbb R^{r_1\times m_2\times n_2\times1}.
$$

再構成したweightの要素は

$$
W_{(i_1,i_2),(j_1,j_2)}
=
\sum_{a=1}^{r_1}
G^{(1)}_{1,i_1,j_1,a}
G^{(2)}_{a,i_2,j_2,1}
$$

である。$i_1,i_2$ は行側、$j_1,j_2$ は列側の分割添字、$a$ はbond添字である。

### 3.1 $r_1=1$

$$
W_{(i_1,i_2),(j_1,j_2)}
=
G^{(1)}_{1,i_1,j_1,1}
G^{(2)}_{1,i_2,j_2,1}.
$$

各weight要素は、左側の添字だけで決まる1個の値と、右側の添字だけで決まる1個の値の積になる。

### 3.2 $r_1=2$

$$
\begin{aligned}
W_{(i_1,i_2),(j_1,j_2)}
={}&
G^{(1)}_{1,i_1,j_1,1}
G^{(2)}_{1,i_2,j_2,1}\\
&+
G^{(1)}_{1,i_1,j_1,2}
G^{(2)}_{2,i_2,j_2,1}.
\end{aligned}
$$

左のパターンと右のパターンの積を2個作り、その和を取れる。

### 3.3 $r_1=3$ の数値例

ある1要素を計算するとき、左右のcoreから取り出した値が

$$
\begin{aligned}
\text{左側}&=(2,3,4),\\
\text{右側}&=(5,-1,2)
\end{aligned}
$$

なら、

$$
\begin{aligned}
W_{(i_1,i_2),(j_1,j_2)}
&=2\cdot5+3\cdot(-1)+4\cdot2\\
&=10-3+8\\
&=15.
\end{aligned}
$$

したがって、$r_1$ は正確には

$$
\boxed{
r_1
=
\text{共有添字 }a\text{ について足す項の数}
}
$$

である。ただし、項がゼロであったり、別の項と同じ方向を表したりすることがある。そのため、「$r_1$ 項を使える」と「独立な成分が必ず $r_1$ 個ある」は同じではない。

## 4. 要素式をunfoldingの行列積 $A=LR$ に直す

左側の添字ペア $(i_1,j_1)$ を1つの行番号 $u$、右側の添字ペア $(i_2,j_2)$ を1つの列番号 $v$ にまとめる。

1-basedの複合添字を使えば、

$$
u=(i_1-1)n_1+j_1,
\qquad
v=(i_2-1)n_2+j_2.
$$

次の行列を定義する。

$$
A_{uv}
:=
W_{(i_1,i_2),(j_1,j_2)},
$$

$$
L_{ua}
:=
G^{(1)}_{1,i_1,j_1,a},
$$

$$
R_{av}
:=
G^{(2)}_{a,i_2,j_2,1}.
$$

shapeは

$$
A\in\mathbb R^{(m_1n_1)\times(m_2n_2)},
$$

$$
L\in\mathbb R^{(m_1n_1)\times r_1},
\qquad
R\in\mathbb R^{r_1\times(m_2n_2)}
$$

である。元の要素式へ代入すると、

$$
A_{uv}
=
\sum_{a=1}^{r_1}L_{ua}R_{av}.
$$

右辺は行列積の第 $(u,v)$ 成分なので、

$$
\boxed{A=LR}
$$

となる。

ここで $A$ と元のdense weight $W$ は、同じ要素を異なる複合添字で並べた行列である。

$$
\begin{array}{c|c|c}
&\text{行側}&\text{列側}\\ \hline
W&(i_1,i_2)&(j_1,j_2)\\
A&(i_1,j_1)&(i_2,j_2)
\end{array}
$$

従って、一般に $\operatorname{rank}(W)$ と $\operatorname{rank}(A)$ は同じとは限らない。

## 5. $A$ の各列は、$L$ の列だけから作られる

例として

$$
L=
\begin{pmatrix}
1&0\\
0&1\\
1&1
\end{pmatrix},
\qquad
R=
\begin{pmatrix}
2&3&4\\
5&6&7
\end{pmatrix}
$$

を考える。共有軸のサイズは2なので、$r_k=2$ である。

$$
\begin{aligned}
A
&=LR\\
&=
\begin{pmatrix}
2&3&4\\
5&6&7\\
7&9&11
\end{pmatrix}.
\end{aligned}
$$

$A$ の第 $j$ 列は

$$
A_{:,j}
=
R_{1j}L_{:,1}
+
R_{2j}L_{:,2}
$$

である。各列を全要素で確認すると、

$$
\begin{aligned}
A_{:,1}
&=
2\begin{pmatrix}1\\0\\1\end{pmatrix}
+
5\begin{pmatrix}0\\1\\1\end{pmatrix}
=
\begin{pmatrix}2\\5\\7\end{pmatrix},\\[6pt]
A_{:,2}
&=
3\begin{pmatrix}1\\0\\1\end{pmatrix}
+
6\begin{pmatrix}0\\1\\1\end{pmatrix}
=
\begin{pmatrix}3\\6\\9\end{pmatrix},\\[6pt]
A_{:,3}
&=
4\begin{pmatrix}1\\0\\1\end{pmatrix}
+
7\begin{pmatrix}0\\1\\1\end{pmatrix}
=
\begin{pmatrix}4\\7\\11\end{pmatrix}.
\end{aligned}
$$

$A$ は3列を持つが、すべての列を作る材料は $L_{:,1}$ と $L_{:,2}$ の2本だけである。したがって独立な列は最大2本である。

一般の $r_k$ では、

$$
A_{:,j}
=
\sum_{a=1}^{r_k}R_{aj}L_{:,a}
$$

なので、

$$
\operatorname{col}(A)
\subseteq
\operatorname{span}\{L_{:,1},\ldots,L_{:,r_k}\}.
$$

従って、

$$
\boxed{
\operatorname{rank}(A)
\le
r_k
}
$$

である。

## 6. 行列積を外積の和として読む

$A=LR$ は、$L$ の列と $R$ の対応する行の外積の和として

$$
\boxed{
A
=
\sum_{a=1}^{r_k}L_{:,a}R_{a,:}
}
$$

と書ける。

例えば、

$$
L=
\begin{pmatrix}
1&0\\
2&1
\end{pmatrix},
\qquad
R=
\begin{pmatrix}
3&4\\
5&6
\end{pmatrix}
$$

なら、

$$
\begin{aligned}
A
&=
\begin{pmatrix}1\\2\end{pmatrix}
\begin{pmatrix}3&4\end{pmatrix}
+
\begin{pmatrix}0\\1\end{pmatrix}
\begin{pmatrix}5&6\end{pmatrix}\\[4pt]
&=
\begin{pmatrix}
3&4\\
6&8
\end{pmatrix}
+
\begin{pmatrix}
0&0\\
5&6
\end{pmatrix}\\[4pt]
&=
\begin{pmatrix}
3&4\\
11&14
\end{pmatrix}.
\end{aligned}
$$

各外積 $L_{:,a}R_{a,:}$ の全列は $L_{:,a}$ のスカラー倍なので、そのrankは最大1である。

## 7. rankの劣加法からも $\operatorname{rank}(A)\le r_k$ を示せる

同じshapeの行列 $B,C$ に対して、

$$
\operatorname{col}(B+C)
\subseteq
\operatorname{col}(B)+\operatorname{col}(C).
$$

従って、

$$
\begin{aligned}
\operatorname{rank}(B+C)
&=
\dim\operatorname{col}(B+C)\\
&\le
\dim\left(\operatorname{col}(B)+\operatorname{col}(C)\right)\\
&\le
\dim\operatorname{col}(B)
+
\dim\operatorname{col}(C)\\
&=
\operatorname{rank}(B)
+
\operatorname{rank}(C).
\end{aligned}
$$

これを繰り返して、

$$
\operatorname{rank}\left(\sum_{a=1}^{r_k}M_a\right)
\le
\sum_{a=1}^{r_k}\operatorname{rank}(M_a)
$$

を得る。$M_a=L_{:,a}R_{a,:}$ とすれば、

$$
\begin{aligned}
\operatorname{rank}(A)
&=
\operatorname{rank}\left(
\sum_{a=1}^{r_k}L_{:,a}R_{a,:}
\right)\\
&\le
\sum_{a=1}^{r_k}
\operatorname{rank}\left(L_{:,a}R_{a,:}\right)\\
&\le
\sum_{a=1}^{r_k}1\\
&=r_k.
\end{aligned}
$$

外積に重複や打ち消しがあれば、実際のrankは $r_k$ より小さくなる。従って等号ではなく上限である。

## 8. 完全再現と近似では、不等式の対象を分ける

元のunfoldingを $A$ とする。

### 8.1 完全再現

$$
A=LR
$$

を要求するなら、

$$
\operatorname{rank}(A)
\le
r_k
$$

が必要である。言い換えると、

$$
\boxed{
r_k
\ge
\operatorname{rank}(A)
}
$$

である。

### 8.2 近似

元の $A$ がrank 3でも、$r_k=2$ を指定することはできる。ただし再構成は別の行列

$$
\widetilde A=LR
$$

であり、

$$
\boxed{
\operatorname{rank}(\widetilde A)
\le
r_k
}
$$

である。元の $A$ のrankが指定によって変わるのではない。

## 9. TT-SVDでは、残すSVD成分数がbond dimensionになる

行列 $A$ のSVDを

$$
A=U\Sigma V^{\mathsf T}
$$

とする。上位 $r_k$ 成分を残すと、

$$
\widetilde A
=
U_{:,1:r_k}
\Sigma_{1:r_k,1:r_k}
(V^{\mathsf T})_{1:r_k,:}.
$$

左因子と右因子を

$$
L=U_{:,1:r_k},
$$

$$
R=
\Sigma_{1:r_k,1:r_k}
(V^{\mathsf T})_{1:r_k,:}
$$

と置けば、

$$
L\in\mathbb R^{p\times r_k},
\qquad
R\in\mathbb R^{r_k\times q}.
$$

共有軸のサイズが $r_k$ になり、これがTT-SVD後のbond dimensionになる。

## 10. 小さいrankが近似誤差を生む例

$$
A=
\begin{pmatrix}
3&0&0\\
0&2&0\\
0&0&1
\end{pmatrix}
$$

はrank 3であり、特異値は $(3,2,1)$ である。bond dimension 2へ打ち切ると、

$$
\widetilde A=
\begin{pmatrix}
3&0&0\\
0&2&0\\
0&0&0
\end{pmatrix}.
$$

これは

$$
\widetilde A
=
\underbrace{
\begin{pmatrix}
1&0\\
0&1\\
0&0
\end{pmatrix}
}_{L\in\mathbb R^{3\times2}}
\underbrace{
\begin{pmatrix}
3&0&0\\
0&2&0
\end{pmatrix}
}_{R\in\mathbb R^{2\times3}}
$$

と書ける。差は

$$
A-\widetilde A
=
\begin{pmatrix}
0&0&0\\
0&0&0\\
0&0&1
\end{pmatrix}
$$

なので、

$$
\|A-\widetilde A\|_F
=
\sqrt{1^2}
=1.
$$

各rankでの最良SVD近似誤差は、

$$
\begin{array}{c|c}
\text{残す成分数}&\text{Frobenius誤差}\\ \hline
1&\sqrt{2^2+1^2}=\sqrt5\\
2&\sqrt{1^2}=1\\
3&0
\end{array}
$$

である。ただし、捨てる特異値がもともと0ならrank上限を下げても誤差は増えない。

## 11. 次に何を比較するか

bond dimensionを上げると、unfoldingに許されるrankの上限が広がる。しかし、保存するcoreの要素数も増える。

$$
\boxed{
\text{bond dimension}
\longrightarrow
\begin{cases}
\text{unfoldingの表現能力},\\
\text{TT coreの保存要素数}
\end{cases}
}
$$

固定tensorizationでこのtrade-offを比較する方法は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/70_TT-matrixのrank_sweepと圧縮率_再構成誤差]]で扱う。PyTorchによる3-coreの確認と保存済み結果は、[[08_TT_MPS基礎実装検証/20_TT-matrixのrank_sweepと圧縮誤差のPyTorch確認]]を参照する。
