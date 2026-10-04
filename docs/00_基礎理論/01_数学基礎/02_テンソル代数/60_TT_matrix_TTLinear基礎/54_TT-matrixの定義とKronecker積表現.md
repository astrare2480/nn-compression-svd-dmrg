---
title: TT-matrixの定義とKronecker積表現
tags:
  - TT-matrix
  - MPO
  - Kronecker
  - Linear
---

# TT-matrixの定義とKronecker積表現

## 1. TT tensorではなく線形写像を分解する

通常のTT tensorは、各siteに一つのphysical indexを持つ。一方、dense線形層の重み

$$
W\in\mathbb R^{m\times n}
$$

は、出力側の行indexと入力側の列indexを持つ。そこで

$$
m=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

と因数分解し、flat indexをmulti-indexへ対応させる。

$$
i\longleftrightarrow(i_1,\ldots,i_d),
\qquad
j\longleftrightarrow(j_1,\ldots,j_d).
$$

0始まりindexを使い、最後のindexが最も速く変化する規約では、

$$
i=
\sum_{k=1}^{d}
i_k\prod_{\ell=k+1}^{d}m_\ell,
\qquad
j=
\sum_{k=1}^{d}
j_k\prod_{\ell=k+1}^{d}n_\ell.
$$

TT-matrixの第 $k$ コアは、同じsiteの出力index $i_k$ と入力index $j_k$ を一つずつ持つ。

数値線形代数でいうTT-matrixと、物理でいうMatrix Product Operator（MPO）は同じchain型の演算子表現を指す。本章ではdense線形層との対応を追うため、TT-matrixという名前を使う。

$$
\boxed{
G^{(k)}\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k}
}
$$

軸順は一貫して

$$
(\alpha_{k-1},i_k,j_k,\alpha_k)
=
(\text{left bond},\text{output mode},\text{input mode},\text{right bond})
$$

とし、開放境界条件

$$
r_0=r_d=1
$$

を課す。したがって、通常の3階TTコアへ単にsize 1の軸を付けただけではなく、線形写像の出力脚と入力脚を明示した4階コアである。

## 2. 一要素はbond行列の積である

重みの一要素は

$$
\boxed{
W_{i_1,\ldots,i_d;\,j_1,\ldots,j_d}
=
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
\prod_{k=1}^{d}
G^{(k)}_{\alpha_{k-1},i_k,j_k,\alpha_k}
}
$$

である。ここで $\alpha_0=\alpha_d=0$ とする。

各 $(i_k,j_k)$ を固定すると、

$$
G^{(k)}[:,i_k,j_k,:]
\in\mathbb R^{r_{k-1}\times r_k}
$$

というbond空間上の行列が残る。そのため、上式は

$$
G^{(1)}[0,i_1,j_1,:]
G^{(2)}[:,i_2,j_2,:]
\cdots
G^{(d)}[:,i_d,j_d,0]
$$

という行列積でもあり、両端rankが1なので最終結果はscalarになる。

## 3. 2 siteの式を完全に展開する

$d=2$ では

$$
m=m_1m_2,
\qquad
n=n_1n_2
$$

であり、flat indexは

$$
i=i_1m_2+i_2,
\qquad
j=j_1n_2+j_2
$$

となる。コアshapeを

$$
G^{(1)}\in\mathbb R^{1\times m_1\times n_1\times r},
\qquad
G^{(2)}\in\mathbb R^{r\times m_2\times n_2\times1}
$$

とすると、

$$
\boxed{
W[i_1m_2+i_2,\,j_1n_2+j_2]
=
\sum_{\alpha=0}^{r-1}
G^{(1)}[0,i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0]
}
$$

である。

## 4. rank 1は一つのKronecker積である

$r=1$ のとき

$$
A[i_1,j_1]=G^{(1)}[0,i_1,j_1,0],
$$

$$
B[i_2,j_2]=G^{(2)}[0,i_2,j_2,0]
$$

と置けば、

$$
W[i_1m_2+i_2,\,j_1n_2+j_2]
=
A[i_1,j_1]B[i_2,j_2].
$$

これはKronecker積の要素定義そのものなので、

$$
\boxed{W=A\otimes B}
$$

である。

具体的に

$$
A=
\begin{pmatrix}
a_{00}&a_{01}\\
a_{10}&a_{11}
\end{pmatrix},
\qquad
B=
\begin{pmatrix}
b_{00}&b_{01}\\
b_{10}&b_{11}
\end{pmatrix}
$$

なら、まずblock行列として

$$
A\otimes B
=
\begin{pmatrix}
a_{00}B&a_{01}B\\
a_{10}B&a_{11}B
\end{pmatrix}
$$

であり、全要素を展開すると

$$
A\otimes B
=
\begin{pmatrix}
a_{00}b_{00}&a_{00}b_{01}&a_{01}b_{00}&a_{01}b_{01}\\
a_{00}b_{10}&a_{00}b_{11}&a_{01}b_{10}&a_{01}b_{11}\\
a_{10}b_{00}&a_{10}b_{01}&a_{11}b_{00}&a_{11}b_{01}\\
a_{10}b_{10}&a_{10}b_{11}&a_{11}b_{10}&a_{11}b_{11}
\end{pmatrix}.
$$

行indexは $(i_1,i_2)$、列indexは $(j_1,j_2)$ の辞書順である。例えば第4行第2列は

$$
(i_1,i_2)=(1,1),
\qquad
(j_1,j_2)=(0,1)
$$

に対応し、その値は

$$
A[1,0]B[1,1]=a_{10}b_{11}
$$

である。

長方形の因子でも同じ対応になる。例えば

$$
A=
\begin{pmatrix}
a_{00}&a_{01}\\
a_{10}&a_{11}
\end{pmatrix}
\in\mathbb R^{2\times2},
$$

$$
B=
\begin{pmatrix}
b_{00}&b_{01}&b_{02}\\
b_{10}&b_{11}&b_{12}
\end{pmatrix}
\in\mathbb R^{2\times3}
$$

なら、

$$
A\otimes B
=
\begin{pmatrix}
a_{00}b_{00}&a_{00}b_{01}&a_{00}b_{02}&a_{01}b_{00}&a_{01}b_{01}&a_{01}b_{02}\\
a_{00}b_{10}&a_{00}b_{11}&a_{00}b_{12}&a_{01}b_{10}&a_{01}b_{11}&a_{01}b_{12}\\
a_{10}b_{00}&a_{10}b_{01}&a_{10}b_{02}&a_{11}b_{00}&a_{11}b_{01}&a_{11}b_{02}\\
a_{10}b_{10}&a_{10}b_{11}&a_{10}b_{12}&a_{11}b_{10}&a_{11}b_{11}&a_{11}b_{12}
\end{pmatrix}.
$$

ここで $m_2=2$、$n_2=3$ である。0始まりindex $(i,j)=(3,4)$ は

$$
3=1\cdot2+1,
\qquad
4=1\cdot3+1
$$

より

$$
(i_1,i_2)=(1,1),
\qquad
(j_1,j_2)=(1,1)
$$

に対応し、

$$
(A\otimes B)[3,4]
=A[1,1]B[1,1]
=a_{11}b_{11}
$$

となる。0始まりの $[3,4]$ は、人間が1から数える表現では4行目・5列目である。以降はPyTorchと同じ0始まりindexで統一する。

## 5. rank $r$ はKronecker積の和である

$r>1$ のとき、bond indexごとに

$$
A^{(\alpha)}[i_1,j_1]
=
G^{(1)}[0,i_1,j_1,\alpha],
$$

$$
B^{(\alpha)}[i_2,j_2]
=
G^{(2)}[\alpha,i_2,j_2,0]
$$

と定義する。すると

$$
\begin{aligned}
W[i_1m_2+i_2,\,j_1n_2+j_2]
&=
\sum_{\alpha=0}^{r-1}
A^{(\alpha)}[i_1,j_1]
B^{(\alpha)}[i_2,j_2]\\
&=
\left[
\sum_{\alpha=0}^{r-1}
A^{(\alpha)}\otimes B^{(\alpha)}
\right]
[i_1m_2+i_2,\,j_1n_2+j_2].
\end{aligned}
$$

任意の行・列要素が一致するので、行列として

$$
\boxed{
W=
\sum_{\alpha=0}^{r-1}
A^{(\alpha)}\otimes B^{(\alpha)}
}
$$

である。

$r=2$ なら

$$
W=A^{(0)}\otimes B^{(0)}+A^{(1)}\otimes B^{(1)}.
$$

$A^{(0)}=(a^{(0)}_{uv})$、$A^{(1)}=(a^{(1)}_{uv})$ と書けば、blockごとに

$$
W=
\begin{pmatrix}
a^{(0)}_{00}B^{(0)}+a^{(1)}_{00}B^{(1)}&
a^{(0)}_{01}B^{(0)}+a^{(1)}_{01}B^{(1)}\\
a^{(0)}_{10}B^{(0)}+a^{(1)}_{10}B^{(1)}&
a^{(0)}_{11}B^{(0)}+a^{(1)}_{11}B^{(1)}
\end{pmatrix}.
$$

したがってbond rankは、2 siteの場合には「何個のKronecker積を足し合わせるか」を表す。一般の $d$ では単純な2因子Kronecker和ではないが、bond rankが隣接site間の相関表現力を制御する点は同じである。

## 6. 保存要素数

dense行列の要素数は

$$
P_{\mathrm{dense}}=mn
$$

である。TT-matrixでは

$$
\boxed{
P_{\mathrm{TT\text{-}matrix}}
=
\sum_{k=1}^{d}
r_{k-1}m_kn_kr_k
}
$$

である。2 site、rank $(1,r,1)$ なら

$$
P_{\mathrm{TT\text{-}matrix}}
=m_1n_1r+rm_2n_2.
$$

rankを上げれば表現力は増えるが、保存量と縮約量も増える。さらに、保存要素数が少ないことだけでは実行時間の短縮を保証しない。小さい層では縮約やkernel起動のオーバーヘッドがdense GEMMを上回る場合がある。

## 7. この表現が解決することと、まだ解決しないこと

この定義で固定できるのは、

- コアの4軸の意味
- flat indexとmulti-indexの対応
- rank 1とKronecker積の対応
- bond rankと表現力・保存量の関係

である。一方、dense重みをどの軸順でtensorizeするか、どのrankで近似するか、学習精度や速度がどう変わるかは別の問題である。

次は [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/55_dense重みのTT-matrix tensorizationとTT-SVD初期化]] で、同じsiteの $(i_k,j_k)$ を正しく隣接させる変換を追う。
