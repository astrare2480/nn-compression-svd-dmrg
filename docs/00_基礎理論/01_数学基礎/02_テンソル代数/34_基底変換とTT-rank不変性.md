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

### unfolding・SVD・remainderを省略せずつなぐ

ここでは実数テンソルを扱い、添字は1始まりとする。第1cutの列位置を

$$
\mu(i_2,i_3)=(i_2-1)n_3+i_3
$$

と置くと、

$$
X^{\langle1\rangle}_{i_1,\mu(i_2,i_3)}
=X_{i_1i_2i_3},
\qquad
X^{\langle1\rangle}\in\mathbb R^{n_1\times(n_2n_3)}
$$

である。非ゼロテンソルについて

$$
r_1=\operatorname{rank}(X^{\langle1\rangle}),
\qquad
X^{\langle1\rangle}=U\Sigma V^T
$$

とする。この節のexact SVDは、非ゼロ特異値をすべて残すrank-sized SVDであり、

$$
U\in\mathbb R^{n_1\times r_1},
\qquad
\Sigma\in\mathbb R^{r_1\times r_1},
\qquad
V\in\mathbb R^{(n_2n_3)\times r_1},
\qquad
U^TU=V^TV=I_{r_1}
$$

を満たす。対角行列を成分まで書けば、

$$
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_{r_1}),
\qquad
\Sigma_{\alpha_1,\beta_1}
=\sigma_{\alpha_1}\delta_{\alpha_1,\beta_1},
\qquad
\sigma_{\alpha_1}>0
$$

である。行列積を最初から成分和にすると、

$$
\begin{aligned}
X_{i_1i_2i_3}
&=X^{\langle1\rangle}_{i_1,\mu(i_2,i_3)}\\
&=\sum_{\alpha_1=1}^{r_1}
\sum_{\beta_1=1}^{r_1}
U_{i_1,\alpha_1}
\Sigma_{\alpha_1,\beta_1}
(V^T)_{\beta_1,\mu(i_2,i_3)}\\
&=\sum_{\alpha_1=1}^{r_1}
\sum_{\beta_1=1}^{r_1}
U_{i_1,\alpha_1}
\sigma_{\alpha_1}\delta_{\alpha_1,\beta_1}
V_{\mu(i_2,i_3),\beta_1}\\
&=\sum_{\alpha_1=1}^{r_1}
U_{i_1,\alpha_1}\sigma_{\alpha_1}
V_{\mu(i_2,i_3),\alpha_1}.
\end{aligned}
$$

第3行から第4行では、Kroneckerのデルタによって

$$
\sum_{\beta_1=1}^{r_1}
\delta_{\alpha_1,\beta_1}V_{\mu,\beta_1}
=V_{\mu,\alpha_1}
$$

となり、二つあったbond添字の和が一つになる。

次に、行列と3階配列を区別してremainderを定義する。

$$
B^{\mathrm{mat}}:=\Sigma V^T
\in\mathbb R^{r_1\times(n_2n_3)},
\qquad
B:=\operatorname{reshape}(B^{\mathrm{mat}},r_1,n_2,n_3).
$$

各要素の対応は

$$
\begin{aligned}
B_{\alpha_1,i_2,i_3}
&=B^{\mathrm{mat}}_{\alpha_1,\mu(i_2,i_3)}\\
&=\sum_{\beta_1=1}^{r_1}
\Sigma_{\alpha_1,\beta_1}(V^T)_{\beta_1,\mu(i_2,i_3)}\\
&=\sum_{\beta_1=1}^{r_1}
\sigma_{\alpha_1}\delta_{\alpha_1,\beta_1}
V_{\mu(i_2,i_3),\beta_1}\\
&=\sigma_{\alpha_1}V_{\mu(i_2,i_3),\alpha_1}
\end{aligned}
$$

である。このreshapeでは値も要素数も変えず、列ラベルを物理添字の組へ戻しているだけである。

実装のreduced SVDは通常、ゼロ特異値も含む

$$
q=\min(n_1,n_2n_3)
$$

本を返す。上の式はそのうち非ゼロの列を残したものなので、「reducedなら自動的に列数がexact rankになる」とは読まない。

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

### 元の座標からremainderの座標へ戻す途中式

列直交性により、逆向きには

$$
\begin{aligned}
U^TX^{\langle1\rangle}
&=U^T(U\Sigma V^T)\\
&=(U^TU)\Sigma V^T\\
&=I_{r_1}\Sigma V^T\\
&=B^{\mathrm{mat}}
\end{aligned}
$$

となる。成分で同じことを確かめると、

$$
\begin{aligned}
\sum_{i_1=1}^{n_1}U_{i_1,\alpha_1}X_{i_1i_2i_3}
&=\sum_{i_1=1}^{n_1}U_{i_1,\alpha_1}
\sum_{\beta_1=1}^{r_1}U_{i_1,\beta_1}B_{\beta_1,i_2,i_3}\\
&=\sum_{\beta_1=1}^{r_1}
\left(\sum_{i_1=1}^{n_1}U_{i_1,\alpha_1}U_{i_1,\beta_1}\right)
B_{\beta_1,i_2,i_3}\\
&=\sum_{\beta_1=1}^{r_1}\delta_{\alpha_1,\beta_1}
B_{\beta_1,i_2,i_3}\\
&=B_{\alpha_1,i_2,i_3}.
\end{aligned}
$$

したがって、物理添字がremainderからなくなることは情報の消失ではない。元のテンソルは、第1サイトの基底成分を持つ $U$ と、その基底での係数を持つ $B$ の組でexactに表される。

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

その前の因子の取り出しとデルタの和を分けて書くと

$$
\begin{aligned}
\sum_{\alpha_1=1}^{r_1}\sum_{j_2=1}^{n_2}
U_{i_1,\alpha_1}\delta_{i_2,j_2}B_{\alpha_1,j_2,i_3}
&=\sum_{\alpha_1=1}^{r_1}U_{i_1,\alpha_1}
\left(\sum_{j_2=1}^{n_2}\delta_{i_2,j_2}B_{\alpha_1,j_2,i_3}\right),\\
\sum_{j_2=1}^{n_2}\delta_{i_2,j_2}B_{\alpha_1,j_2,i_3}
&=0+\cdots+B_{\alpha_1,i_2,i_3}+\cdots+0\\
&=B_{\alpha_1,i_2,i_3}.
\end{aligned}
$$

$U_{i_1,\alpha_1}$ は $j_2$ に依存しないので、内側の和の外へ出せる。

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

その前に、元資料 `TT_MPS基礎理論.md` の9202–9221行にある一般のblock表示も残す。
$U$ の各成分を単位行列へ掛けることが、Kronecker積の具体的な構成である。

$$
U=\begin{pmatrix}
U_{1,1}&U_{1,2}&\cdots&U_{1,r_1}\\
U_{2,1}&U_{2,2}&\cdots&U_{2,r_1}\\
\vdots&\vdots&\ddots&\vdots\\
U_{n_1,1}&U_{n_1,2}&\cdots&U_{n_1,r_1}
\end{pmatrix},\qquad
L_2=\begin{pmatrix}
U_{1,1}I_{n_2}&U_{1,2}I_{n_2}&\cdots&U_{1,r_1}I_{n_2}\\
U_{2,1}I_{n_2}&U_{2,2}I_{n_2}&\cdots&U_{2,r_1}I_{n_2}\\
\vdots&\vdots&\ddots&\vdots\\
U_{n_1,1}I_{n_2}&U_{n_1,2}I_{n_2}&\cdots&U_{n_1,r_1}I_{n_2}
\end{pmatrix}.
$$

外側のblockの行は $i_1$、列は $\alpha_1$ に対応する。
各block内の行 $i_2$・列 $j_2$ に対応する成分は
$U_{i_1,\alpha_1}\delta_{i_2,j_2}$ なので、異なるサイト2の成分を混ぜない。
元資料でいう「サイト1のbasisを戻し、サイト2はそのまま通す」を、
このblock表示と成分式で同じ操作として確認できる。

元資料10640–10690行の $u_{11},u_{12},u_{21},u_{22}$ の表示は、
以下の同じ例では $a=u_{11},b=u_{12},c=u_{21},d=u_{22}$ に対応する。
四つの $I_2$ をそのまま代入する途中表示も、次の式で省略せず示す。

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
a\begin{pmatrix}1&0\\0&1\end{pmatrix}&
b\begin{pmatrix}1&0\\0&1\end{pmatrix}\\
c\begin{pmatrix}1&0\\0&1\end{pmatrix}&
d\begin{pmatrix}1&0\\0&1\end{pmatrix}
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

### 元資料のrank 1例：$B$ を係数付きで縦に広げる

元資料 `TT_MPS基礎理論.md` の5823〜5890行、6117〜6153行、8203〜8355行付近は、$r_1=1$ の場合を別に全要素表示している。
このとき $B_{1,i_2,i_3}$ の第1添字は常に1だが、その軸を省略して書けることと、サイト1の係数が失われたことは別である。
サイト1の成分は $U=u$ にあり、

$$
U=\begin{pmatrix}u_1\\u_2\end{pmatrix},\qquad
B_{\mathrm{cut2}}=\begin{pmatrix}a&b\\c&d\end{pmatrix}
$$

なら、行 $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ の順で

$$
\begin{aligned}
L_2=U\otimes I_2
&=\begin{pmatrix}u_1&0\\0&u_1\\u_2&0\\0&u_2\end{pmatrix},\\
X_{\mathrm{cut2}}=L_2B_{\mathrm{cut2}}
&=\begin{pmatrix}
u_1a&u_1b\\u_1c&u_1d\\u_2a&u_2b\\u_2c&u_2d
\end{pmatrix}
=\begin{pmatrix}u_1B_{\mathrm{cut2}}\\u_2B_{\mathrm{cut2}}\end{pmatrix}.
\end{aligned}
$$

上の2行はサイト1の基底成分 $u_1$、下の2行は $u_2$ が掛かった同じ $B$ である。

また5823–5874行で最初に使った一般の $B$ の表示は、
第1ボンドが1であることを省略して

$$
B_{\mathrm{cut2}}=
\begin{pmatrix}
b_{11}&b_{12}&\cdots&b_{1n_3}\\
b_{21}&b_{22}&\cdots&b_{2n_3}\\
\vdots&\vdots&\ddots&\vdots\\
b_{n_2,1}&b_{n_2,2}&\cdots&b_{n_2,n_3}
\end{pmatrix},\qquad b_{jk}=B_{1,j,k}
$$

としたものだった。$n_2=n_3=2$ の場合が、上の $a,b,c,d$ の4要素の例である。
以下ではこの4要素の添字対応へ戻る。

元資料6472–6491行の添字付きの表示も、この例では

$$
a=B_{1,1,1},\quad b=B_{1,1,2},\quad
c=B_{1,2,1},\quad d=B_{1,2,2},
$$

$$
B_{\mathrm{cut2}}=
\begin{pmatrix}B_{1,1,1}&B_{1,1,2}\\B_{1,2,1}&B_{1,2,2}\end{pmatrix},
\qquad
X_{\mathrm{cut2}}=
\begin{pmatrix}
u_1B_{1,1,1}&u_1B_{1,1,2}\\
u_1B_{1,2,1}&u_1B_{1,2,2}\\
u_2B_{1,1,1}&u_2B_{1,1,2}\\
u_2B_{1,2,1}&u_2B_{1,2,2}
\end{pmatrix}
$$

として対応する。値の違う第二の具体例ではなく、同じ4個のremainder成分の記号を戻した表示である。
行を増やしても、$u\ne0$ なら $B$ の独立な列方向を合体させて消すことはない。
実際、正規化をまだ仮定しなくても

$$
L_2^{\mathsf T}L_2
=\begin{pmatrix}u_1^2+u_2^2&0\\0&u_1^2+u_2^2\end{pmatrix},
\qquad
B_{\mathrm{cut2}}
=\frac{L_2^{\mathsf T}X_{\mathrm{cut2}}}{u_1^2+u_2^2}.
$$

この左逆から $\operatorname{rank}(X_{\mathrm{cut2}})=\operatorname{rank}(B_{\mathrm{cut2}})$ が従う。

元資料8327–8349行の転置と積の行列も途中で省略しない。

$$
\begin{aligned}
L_2^T L_2
&=\begin{pmatrix}u_1&0&u_2&0\\0&u_1&0&u_2\end{pmatrix}
\begin{pmatrix}u_1&0\\0&u_1\\u_2&0\\0&u_2\end{pmatrix}\\
&=\begin{pmatrix}
u_1^2+0^2+u_2^2+0^2&u_1\cdot0+0\cdot u_1+u_2\cdot0+0\cdot u_2\\
0\cdot u_1+u_1\cdot0+0\cdot u_2+u_2\cdot0&0^2+u_1^2+0^2+u_2^2
\end{pmatrix}\\
&=(u_1^2+u_2^2)I_2.
\end{aligned}
$$

$u_1^2+u_2^2=1$ の場合に限り、元資料の最後の $I_2$ になる。
元資料の $u=(1,2)^{\mathsf T}$ なら

$$
X_{\mathrm{cut2}}=\begin{pmatrix}B_{\mathrm{cut2}}\\2B_{\mathrm{cut2}}\end{pmatrix},
\qquad
L_2^{\mathsf T}L_2=5I_2.
$$

従って、この例はrank保存を示すが、そのままでは等長写像ではない。
SVDの列として使うなら $u=(1,2)^{\mathsf T}/\sqrt5$ と正規化し、$L_2^{\mathsf T}L_2=I_2$ にする。
第1cutのrankが1でも、第2cutのrankまで1とは限らない。上の $B$ がrank 2なら $r_1=1,r_2=2$ であり、サイト2と3の相関はremainderに残る。

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

### Kronecker積の性質を使わず、成分和から確認する

列ラベルを異なる組

$$
(\alpha_1,j_2),\qquad(\beta_1,k_2)
$$

として、転置・行列積・二つのデルタを順に展開する。

$$
\begin{aligned}
(L_2^TL_2)_{(\alpha_1,j_2),(\beta_1,k_2)}
&=\sum_{i_1=1}^{n_1}\sum_{i_2=1}^{n_2}
(L_2^T)_{(\alpha_1,j_2),(i_1,i_2)}
(L_2)_{(i_1,i_2),(\beta_1,k_2)}\\
&=\sum_{i_1=1}^{n_1}\sum_{i_2=1}^{n_2}
\left(U_{i_1,\alpha_1}\delta_{i_2,j_2}\right)
\left(U_{i_1,\beta_1}\delta_{i_2,k_2}\right)\\
&=\left(\sum_{i_1=1}^{n_1}U_{i_1,\alpha_1}U_{i_1,\beta_1}\right)
\left(\sum_{i_2=1}^{n_2}\delta_{i_2,j_2}\delta_{i_2,k_2}\right)\\
&=(U^TU)_{\alpha_1,\beta_1}\delta_{j_2,k_2}\\
&=\delta_{\alpha_1,\beta_1}\delta_{j_2,k_2}\\
&=(I_{r_1n_2})_{(\alpha_1,j_2),(\beta_1,k_2)}.
\end{aligned}
$$

第2の和は、$j_2=k_2$ なら $i_2=j_2$ の1項だけが1であり、異なれば全項が0になる。これが「第2サイトをそのまま通す」ことの成分表示である。

また、本節で使った積の性質自体も、一般の行列に対して

$$
\begin{aligned}
[(A\otimes B)(C\otimes D)]_{(i,j),(k,\ell)}
&=\sum_{a,b}A_{i,a}B_{j,b}C_{a,k}D_{b,\ell}\\
&=\left(\sum_aA_{i,a}C_{a,k}\right)
\left(\sum_bB_{j,b}D_{b,\ell}\right)\\
&=(AC)_{i,k}(BD)_{j,\ell}\\
&=[(AC)\otimes(BD)]_{(i,j),(k,\ell)}
\end{aligned}
$$

と導ける。各行列のshapeは、二つの積 $AC$ と $BD$ が定義できるものとする。

---


### 元資料の $I_{r_1n_2}$ のサイズ確認

元資料 `TT_MPS基礎理論.md` の9063〜9149行付近で質問された $I_{r_1n_2}$ は、shape $(r_1,n_2)$ の配列ではない。
下付き $r_1n_2$ は単位行列の一辺のサイズである。
例えば $r_1=2,n_2=3$ なら

$$
I_{r_1n_2}=I_6=
\begin{pmatrix}
1&0&0&0&0&0\\
0&1&0&0&0&0\\
0&0&1&0&0&0\\
0&0&0&1&0&0\\
0&0&0&0&1&0\\
0&0&0&0&0&1
\end{pmatrix}.
$$

$n_1=4$ なら $L_2$ は $12\times6$、転置は $6\times12$ なので、

$$
\operatorname{shape}(L_2^{\mathsf T}L_2)
=(6,6),\qquad
\underbrace{L_2^{\mathsf T}}_{6\times12}
\underbrace{L_2}_{12\times6}\in\mathbb R^{6\times6}
$$

というサイズ確認になる。ただし $L_2^{\mathsf T}L_2$ が実際に $I_6$ となるには、shapeだけでなく $U^{\mathsf T}U=I_2$ の条件が必要である。
また上付き $\langle2\rangle$ は第2cutの行列化というラベルであり、行列の二乗やTensorの階数2を意味しない。
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

shapeを見ると、元の第2cutの行数 $n_1n_2$ が途中行列では $r_1n_2$ に減る。
しかしexactな第1 SVDでは、元Tensorのサイト1方向の全列が $U$ の列空間内にある。
使われていない方向の座標を省いているだけなので、必要な方向同士を潰す変換ではない。
$L_2=U\otimes I_{n_2}$ はその保持座標を元サイト1空間へ展開し、第2サイトの番号をそのまま保つ。

この保持座標側では

$$
L_2z=0
\quad\Longrightarrow\quad
z=L_2^TL_2z=L_2^T0=0
$$

となる。非零の座標を零へ潰さないので、途中行列の独立な列の組合せは元行列へ戻しても独立である。
これが、行数が違ってもrankが保存される直観と、その条件を結ぶ説明である。
元空間の全方向に $L_2L_2^T=I$ が成り立つと仮定する必要はない。

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

この第1 SVDだけの打ち切りを、元テンソルへの射影としても書く。
保持列 $\widehat U$ に対して $\widehat B^{\mathrm{mat}}=\widehat U^TX^{\langle1\rangle}$ なので

$$
\begin{aligned}
\widehat X_{i_1,i_2,i_3}
&=\sum_{\alpha=1}^{\widehat r_1}\widehat U_{i_1,\alpha}
\left(\sum_{j_1=1}^{n_1}\widehat U_{j_1,\alpha}X_{j_1,i_2,i_3}\right)\\
&=\sum_{j_1=1}^{n_1}
(\widehat U\widehat U^T)_{i_1,j_1}X_{j_1,i_2,i_3}.
\end{aligned}
$$

第2cutの行順をそろえると

$$
\widehat X^{\langle2\rangle}
=(\widehat U\widehat U^T\otimes I_{n_2})X^{\langle2\rangle},
\qquad
\operatorname{rank}(\widehat X^{\langle2\rangle})
\le\operatorname{rank}(X^{\langle2\rangle}).
$$

一方、$\widehat L_2:=\widehat U\otimes I_{n_2}$ は列直交なので

$$
\begin{aligned}
\widehat L_2^T\widehat X^{\langle2\rangle}
&=\widehat L_2^T\widehat L_2\widehat B_{\mathrm{cut2}}
=\widehat B_{\mathrm{cut2}},\\
\operatorname{rank}(\widehat X^{\langle2\rangle})
&\le\operatorname{rank}(\widehat B_{\mathrm{cut2}})
\le\operatorname{rank}(\widehat X^{\langle2\rangle}).
\end{aligned}
$$

従って「近似テンソルとそのremainderは同rank」と
「元テンソルよりrankが下がり得る」は矛盾しない。

---

## 13. この章で固定する理解

- $B$ の $\alpha_1$ はサイト1情報が消えた印ではなく、サイト1側をSVD基底で表したbond index。
- $L_2=U\otimes I_{n_2}$ は、サイト1だけを $\alpha_1$ basisから $i_1$ basisへ戻し、サイト2はそのままにする。
- $U^TU=I$ から $L_2^TL_2=I$。
- 列直交な左変換はrankを潰さない。
- これにより、途中remainderのSVDで得るrankが元テンソルのcut rankと対応する。
- truncation時は、元の $X$ ではなく近似 $\widehat X$ と $\widehat B$ の間で同じrank不変性が成立する。
