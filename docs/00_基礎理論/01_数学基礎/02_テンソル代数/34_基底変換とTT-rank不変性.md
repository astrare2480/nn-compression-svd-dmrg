---
title: 基底変換とTT-rank不変性
aliases:
  - U tensor I
  - TT-rank invariance
  - Kronecker積
  - L2基底変換
tags:
  - TT
  - rank
  - KroneckerProduct
  - orthogonality
  - basis
---

# 基底変換とTT-rank不変性

## サマリー

第1 SVD後のremainder $B$ には元のサイト1添字 $i_1$ が存在しない。それでも、$B$ を次にSVDしたrankが元テンソルの第2 TT-rankと一致する。

核心は

$$
\boxed{
X^{\langle2\rangle}
=
L_2B_{\mathrm{cut2}},
\qquad
L_2=U\otimes I_{n_2}
}
$$

と書け、$U$ の列が直交するため

$$
L_2^TL_2=I
$$

となることである。$L_2$ は列フルランクで左逆を持つので、$B$ のrankを潰さない。

$$
\boxed{
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(B_{\mathrm{cut2}}\right)
}
$$

---

## 1. 出発点：第1 SVD

3階テンソル

$$
X_{i_1i_2i_3}
$$

を第1cut

$$
i_1\mid(i_2,i_3)
$$

でSVDする。

exact SVD後、

$$
X_{i_1i_2i_3}
=
\sum_{\alpha_1=1}^{r_1}
U_{i_1,\alpha_1}
B_{\alpha_1,i_2,i_3}
$$

と書ける。

ここで

$$
U\in\mathbb R^{n_1\times r_1},
$$

$$
B\in\mathbb R^{r_1\times n_2\times n_3}.
$$

$B$ に $i_1$ はない。代わりに、第1cutの左空間を表すbasis index $\alpha_1$ がある。

---

## 2. 第2cutを直接作る

元テンソルを

$$
(i_1,i_2)\mid i_3
$$

で行列化する。

$$
X^{\langle2\rangle}_{(i_1,i_2),i_3}
:=
X_{i_1i_2i_3}.
$$

shapeは

$$
X^{\langle2\rangle}
\in
\mathbb R^{(n_1n_2)\times n_3}.
$$

一方、$B$ も

$$
(\alpha_1,i_2)\mid i_3
$$

で行列化する。

$$
B_{\mathrm{cut2},(\alpha_1,i_2),i_3}
:=
B_{\alpha_1,i_2,i_3}.
$$

shapeは

$$
B_{\mathrm{cut2}}
\in
\mathbb R^{(r_1n_2)\times n_3}.
$$

---

## 3. $L_2$ の成分定義

$B$ の左側basis $(\alpha_1,i_2)$ を元の $(i_1,i_2)$ へ戻す行列を

$$
L_2
\in
\mathbb R^{(n_1n_2)\times(r_1n_2)}
$$

とする。

成分を

$$
\boxed{
(L_2)_{(i_1,i_2),(\alpha_1,j_2)}
:=
U_{i_1,\alpha_1}\delta_{i_2,j_2}
}
$$

と定義する。

Kronecker delta

$$
\delta_{i_2,j_2}
=
\begin{cases}
1 & i_2=j_2,\\
0 & i_2\ne j_2
\end{cases}
$$

によって、サイト2のindexは変えず、サイト1のbasisだけ

$$
\alpha_1\longrightarrow i_1
$$

へ $U$ で展開する。

---

## 4. $X^{\langle2\rangle}=L_2B_{\mathrm{cut2}}$ の完全な導出

行列積の成分は

$$
(L_2B_{\mathrm{cut2}})_{(i_1,i_2),i_3}
=
\sum_{\alpha_1=1}^{r_1}
\sum_{j_2=1}^{n_2}
(L_2)_{(i_1,i_2),(\alpha_1,j_2)}
(B_{\mathrm{cut2}})_{(\alpha_1,j_2),i_3}.
$$

$L_2$ の定義を代入する。

$$
=
\sum_{\alpha_1}
\sum_{j_2}
U_{i_1,\alpha_1}
\delta_{i_2,j_2}
B_{\alpha_1,j_2,i_3}.
$$

$\delta_{i_2,j_2}$ により $j_2=i_2$ の項だけが残る。

$$
=
\sum_{\alpha_1}
U_{i_1,\alpha_1}
B_{\alpha_1,i_2,i_3}.
$$

第1 SVD後の式から

$$
\sum_{\alpha_1}
U_{i_1,\alpha_1}
B_{\alpha_1,i_2,i_3}
=
X_{i_1i_2i_3}.
$$

第2 unfoldingの定義より

$$
X_{i_1i_2i_3}
=
X^{\langle2\rangle}_{(i_1,i_2),i_3}.
$$

したがって

$$
\boxed{
X^{\langle2\rangle}
=
L_2B_{\mathrm{cut2}}
}
$$

が得られる。

---

## 5. なぜ $L_2=U\otimes I_{n_2}$ なのか

Kronecker積の成分公式は

$$
(A\otimes B)_{(i,p),(j,q)}
=
A_{i,j}B_{p,q}.
$$

ここで

$$
A=U,
\qquad
B=I_{n_2}
$$

とすると、

$$
(U\otimes I_{n_2})_{(i_1,i_2),(\alpha_1,j_2)}
=
U_{i_1,\alpha_1}
(I_{n_2})_{i_2,j_2}.
$$

単位行列の成分は

$$
(I_{n_2})_{i_2,j_2}
=
\delta_{i_2,j_2}
$$

なので、

$$
(U\otimes I_{n_2})_{(i_1,i_2),(\alpha_1,j_2)}
=
U_{i_1,\alpha_1}\delta_{i_2,j_2}.
$$

これは前節の $L_2$ の定義そのものである。

よって

$$
\boxed{
L_2=U\otimes I_{n_2}
}
$$

となる。

### $2\times2$ の行列を使った全要素計算

一つの例として

$$
n_1=r_1=n_2=n_3=2
$$

とし、

$$
U
=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix},
\qquad
I_{n_2}
=
I_2
=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
$$

を使う。Kronecker積では、$U$ の各要素を対応する $I_2$ のblockへ掛けるので、

$$
\begin{aligned}
L_2
&=
U\otimes I_2\\
&=
\begin{pmatrix}
aI_2&bI_2\\
cI_2&dI_2
\end{pmatrix}\\
&=
\begin{pmatrix}
a&0&b&0\\
0&a&0&b\\
c&0&d&0\\
0&c&0&d
\end{pmatrix}.
\end{aligned}
$$

行・列の複合添字は、それぞれ

$$
\begin{aligned}
(i_1,i_2)
&=
(1,1),(1,2),(2,1),(2,2),\\
(\alpha_1,i_2)
&=
(1,1),(1,2),(2,1),(2,2)
\end{aligned}
$$

の順とする。remainderの第2cutを

$$
B_{\mathrm{cut2}}
=
\begin{pmatrix}
p&q\\
r&s\\
t&u\\
v&w
\end{pmatrix}
$$

と置く。各行は上から $(\alpha_1,i_2)=(1,1),(1,2),(2,1),(2,2)$、各列は $i_3=1,2$ に対応する。

このとき、

$$
\begin{aligned}
X_{\mathrm{cut2}}
&=
L_2B_{\mathrm{cut2}}\\
&=
\begin{pmatrix}
a&0&b&0\\
0&a&0&b\\
c&0&d&0\\
0&c&0&d
\end{pmatrix}
\begin{pmatrix}
p&q\\
r&s\\
t&u\\
v&w
\end{pmatrix}\\
&=
\begin{pmatrix}
a p+b t&a q+b u\\
a r+b v&a s+b w\\
c p+d t&c q+d u\\
c r+d v&c s+d w
\end{pmatrix}.
\end{aligned}
$$

積の各行と $X_{\mathrm{cut2}}$ の複合添字の対応は

$$
\begin{array}{c|c}
\text{行位置}&(i_1,i_2)\\
\hline
1&(1,1)\\
2&(1,2)\\
3&(2,1)\\
4&(2,2)
\end{array}
$$

である。例えば第2行では $i_2=2$ の成分 $r,s,v,w$ だけが選ばれ、第4行でも同じ $i_2=2$ の成分へ $c,d$ が作用する。これは一般式

$$
X_{i_1i_2i_3}
=
\sum_{\alpha_1}
U_{i_1,\alpha_1}
B_{\alpha_1,i_2,i_3}
$$

を全要素へ展開したものになっている。$U$ がSVDから得た列直交行列なら、この具体例でも $L_2^TL_2=I_4$ となり、後述するrank保存へつながる。

---

## 6. shapeの完全確認

$$
U\in\mathbb R^{n_1\times r_1}
$$

$$
I_{n_2}\in\mathbb R^{n_2\times n_2}
$$

だからKronecker積は

$$
U\otimes I_{n_2}
\in
\mathbb R^{(n_1n_2)\times(r_1n_2)}.
$$

さらに

$$
B_{\mathrm{cut2}}
\in
\mathbb R^{(r_1n_2)\times n_3}.
$$

したがって

$$
\underbrace{L_2}_{(n_1n_2)\times(r_1n_2)}
\underbrace{B_{\mathrm{cut2}}}_{(r_1n_2)\times n_3}
\in
\mathbb R^{(n_1n_2)\times n_3},
$$

これは

$$
X^{\langle2\rangle}
\in
\mathbb R^{(n_1n_2)\times n_3}
$$

と一致する。

---

## 7. $L_2^TL_2=I$ の途中式

SVDの左特異ベクトルは列直交なので

$$
U^TU=I_{r_1}.
$$

Kronecker積の積の性質

$$
(A\otimes B)(C\otimes D)
=
(AC)\otimes(BD)
$$

を使う。

$$
\begin{aligned}
L_2^TL_2
&=
(U\otimes I_{n_2})^T
(U\otimes I_{n_2})\\
&=
(U^T\otimes I_{n_2}^T)
(U\otimes I_{n_2})\\
&=
(U^TU)\otimes(I_{n_2}^TI_{n_2})\\
&=
I_{r_1}\otimes I_{n_2}\\
&=
I_{r_1n_2}.
\end{aligned}
$$

したがって

$$
\boxed{
L_2^TL_2=I_{r_1n_2}
}
$$

である。

$I_{r_1n_2}$ は当然

$$
(r_1n_2)\times(r_1n_2)
$$

の正方単位行列である。

---

## 8. $L_2L_2^T$ は一般に単位行列ではない

$U$ が細い行列

$$
U\in\mathbb R^{n_1\times r_1},
\qquad
r_1<n_1
$$

なら、

$$
U^TU=I_{r_1}
$$

だが、

$$
UU^T\ne I_{n_1}
$$

である。

$UU^T$ は $U$ の列空間への直交射影である。同様に

$$
L_2L_2^T
$$

も元空間全体のidentityではなく、$L_2$ の列空間への射影になる。

ここで必要なのは、$L_2$ が左逆 $L_2^T$ を持つこと、すなわち

$$
L_2^TL_2=I
$$

である。

---

## 9. rankが同じことの証明

まず

$$
X^{\langle2\rangle}
=
L_2B_{\mathrm{cut2}}
$$

なので、一般のrank不等式から

$$
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
=
\operatorname{rank}(L_2B_{\mathrm{cut2}})
\le
\operatorname{rank}(B_{\mathrm{cut2}}).
$$

一方、左から $L_2^T$ を掛けると

$$
\begin{aligned}
L_2^TX^{\langle2\rangle}
&=
L_2^TL_2B_{\mathrm{cut2}}\\
&=
I B_{\mathrm{cut2}}\\
&=
B_{\mathrm{cut2}}.
\end{aligned}
$$

したがって

$$
\operatorname{rank}(B_{\mathrm{cut2}})
=
\operatorname{rank}
\left(L_2^TX^{\langle2\rangle}\right)
\le
\operatorname{rank}
\left(X^{\langle2\rangle}\right).
$$

両方を合わせると、

$$
\boxed{
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(B_{\mathrm{cut2}}\right)
}
$$

である。

別の言い方では、$L_2$ は列フルランクで左逆を持つため、$B$ の独立な列方向を潰さない。

---

## 10. 何を証明したかったのか

TT-SVDは第1 SVD後、元の $X$ を直接使わず remainder $B$ だけを次へ渡す。

もし

$$
\operatorname{rank}(B_{\mathrm{cut2}})
$$

が元テンソルの第2cut rankと無関係なら、逐次SVDで得る内部rankは「途中表現の都合で生じた値」に過ぎなくなる。

しかし前節により

$$
\operatorname{rank}(B_{\mathrm{cut2}})
=
\operatorname{rank}(X^{\langle2\rangle})
$$

が分かる。そのため、第2 SVDで得るbond dimensionは元テンソルの本来の第2 TT-rankを表す。

---

## 11. $U\otimes I$ と $I\otimes U$、置換行列

Kronecker積の因子順を変えると、一般には同じ行列にはならない。

$$
U\otimes I_{n_2}
\ne
I_{n_2}\otimes U.
$$

ただし、行列化で複合添字を並べる順序も同時に変えれば、同じ添字操作を異なる座標順で表せる。

例えばblock順で

$$
X_{\mathrm{block}}
=
(I_{n_2}\otimes U)
B_{\mathrm{block}}
$$

と書いたものを、別の行順へ変換する置換行列 $P_{\mathrm{in}},P_{\mathrm{out}}$ を使えば

$$
X_{\mathrm{ordered}}
=
P_{\mathrm{out}}^T
(I_{n_2}\otimes U)
P_{\mathrm{in}}
B_{\mathrm{ordered}}.
$$

資料の最終整理では、PyTorchの `reshape(n_1*n_2, n_3)` と整合する行順を使うと、直接

$$
\boxed{
U\otimes I_{n_2}
}
$$

で書けるとした。

数学的なrank議論では、置換行列自体も可逆なのでrankを変えない。

PyTorchの実際の要素順・`torch.kron`との対応は [[29_TT_cutとPyTorchのreshape_Kronecker順序]] で扱う。

---

## 12. truncationした場合

第1 SVDを

$$
\widehat r_1<r_1
$$

へ打ち切ると、表現対象は元の $X$ ではなく

$$
\widehat X
$$

になる。

打ち切った $\widehat U$ と $\widehat B$ に対しては

$$
\widehat X^{\langle2\rangle}
=
(\widehat U\otimes I_{n_2})
\widehat B_{\mathrm{cut2}}
$$

が成立する。

$\widehat U$ の列も直交しているので

$$
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(\widehat B_{\mathrm{cut2}}\right).
$$

しかしこれは元の $X$ とのrank一致を意味しない。

$$
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
$$

はtruncation後の近似テンソルのrankであり、

$$
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
$$

より小さくなり得る。

---

## 13. この章で固定する理解

- $B$ の $\alpha_1$ はサイト1情報が消えた印ではなく、サイト1側をSVD基底で表したbond index。
- $L_2=U\otimes I_{n_2}$ は、サイト1だけを $\alpha_1$ basisから $i_1$ basisへ戻し、サイト2はそのままにする。
- $U^TU=I$ から $L_2^TL_2=I$。
- 列直交な左変換はrankを潰さない。
- これにより、途中remainderのSVDで得るrankが元テンソルのcut rankと対応する。
- truncation時は、元の $X$ ではなく近似 $\widehat X$ と $\widehat B$ の間で同じrank不変性が成立する。
