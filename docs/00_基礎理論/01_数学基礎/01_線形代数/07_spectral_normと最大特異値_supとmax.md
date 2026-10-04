---
title: spectral normと最大特異値 supとmax
tags:
  - linear-algebra
  - spectral-norm
  - singular-value
  - supremum
  - operator-norm
---

# spectral normと最大特異値：supとmax

## 1. spectral normの定義

行列

$$
A\in\mathbb{R}^{m\times n}
$$

のspectral norm、すなわちvector 2-normから誘導されるoperator normは、

$$
\boxed{
\|A\|_2
=
\sup_{x\ne0}
\frac{\|Ax\|_2}{\|x\|_2}
}
$$

で定義される。これは、入力ベクトルを $A$ が最大で何倍まで伸ばし得るかを表す。

比は入力の大きさに依存しない。任意のスカラー $\alpha\ne0$ に対して、

$$
\frac{\|A(\alpha x)\|_2}{\|\alpha x\|_2}
=
\frac{|\alpha|\|Ax\|_2}{|\alpha|\|x\|_2}
=
\frac{\|Ax\|_2}{\|x\|_2}
$$

だからである。従って、単位ベクトルだけを調べればよい。

$$
\|A\|_2
=
\sup_{\|x\|_2=1}
\|Ax\|_2.
$$

## 2. SVDから最大特異値を導く

$A$ のfull SVDを

$$
A=U\Sigma V^{\mathsf T}
$$

とする。shapeは

$$
U\in\mathbb{R}^{m\times m},
\qquad
\Sigma\in\mathbb{R}^{m\times n},
\qquad
V\in\mathbb{R}^{n\times n}
$$

である。$U,V$ は直交行列なので、

$$
U^{\mathsf T}U=I_m,
\qquad
V^{\mathsf T}V=I_n.
$$

直交行列はEuclidean normを保存する。例えば、任意の $y\in\mathbb{R}^{m}$ に対して、

$$
\begin{aligned}
\|Uy\|_2^2
&=(Uy)^{\mathsf T}(Uy)\\
&=y^{\mathsf T}U^{\mathsf T}Uy\\
&=y^{\mathsf T}y\\
&=\|y\|_2^2.
\end{aligned}
$$

単位ベクトル $x\in\mathbb{R}^{n}$ に対して

$$
z=V^{\mathsf T}x
$$

と置く。$V$ も直交行列なので、

$$
\|z\|_2=\|x\|_2=1.
$$

従って、

$$
\begin{aligned}
\|Ax\|_2
&=\|U\Sigma V^{\mathsf T}x\|_2\\
&=\|U\Sigma z\|_2\\
&=\|\Sigma z\|_2.
\end{aligned}
$$

ここで

$$
r=\min(m,n),
\qquad
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r\ge0
$$

とする。長方形の $\Sigma$ に対しても、非零になり得る対角成分は先頭 $r$ 個である。そのため、

$$
\begin{aligned}
\|\Sigma z\|_2^2
&=
\sum_{i=1}^{r}
\sigma_i^2z_i^2\\
&\le
\sigma_1^2
\sum_{i=1}^{r}z_i^2\\
&\le
\sigma_1^2
\sum_{i=1}^{n}z_i^2\\
&=
\sigma_1^2\|z\|_2^2\\
&=
\sigma_1^2.
\end{aligned}
$$

従って、全ての単位ベクトルに対して

$$
\|Ax\|_2\le\sigma_1
$$

である。

一方、最大特異値に対応する第1右特異ベクトル $v_1$ を選ぶと、

$$
Av_1=\sigma_1u_1,
$$

$$
\|Av_1\|_2
=
\sigma_1\|u_1\|_2
=
\sigma_1.
$$

上界を実際に達成する方向が存在するので、

$$
\boxed{
\|A\|_2
=
\sigma_{\max}(A)
=
\sigma_1(A)
}
$$

となる。これは有限次元の実行列で常に成立する。条件が必要なのは「特定の入力 $x$ で等号になるか」であり、normそのものが最大特異値と等しいかではない。

## 3. 長方形行列で総和範囲を確認する

例えば、

$$
A\in\mathbb{R}^{3\times5}
$$

なら、

$$
\Sigma\in\mathbb{R}^{3\times5},
\qquad
r=\min(3,5)=3.
$$

$z\in\mathbb{R}^{5}$ に対して、

$$
\Sigma z
=
\begin{pmatrix}
\sigma_1z_1\\
\sigma_2z_2\\
\sigma_3z_3
\end{pmatrix}
\in\mathbb{R}^{3}.
$$

従って、

$$
\begin{aligned}
\|\Sigma z\|_2^2
&=
\sum_{i=1}^{3}\sigma_i^2z_i^2\\
&\le
\sigma_1^2\sum_{i=1}^{3}z_i^2\\
&\le
\sigma_1^2\sum_{i=1}^{5}z_i^2.
\end{aligned}
$$

$z_4,z_5$ は入力ベクトルのnormには含まれるが、$\Sigma z$ では0になる。長方形行列で導出するときは、特異値の総和範囲 $1\le i\le r$ と、入力座標の総和範囲 $1\le i\le n$ を混同しない。

## 4. `sup`と`max`の違い

集合 $S\subset\mathbb{R}$ の上界とは、全ての $s\in S$ に対して

$$
s\le M
$$

を満たす数 $M$ である。上界全体を

$$
\mathcal{U}
=
\{M\in\mathbb{R}\mid s\le M\text{ for all }s\in S\}
$$

と書くと、上限 $\sup S$ は最小上界である。

$$
\sup S
=
\min\mathcal{U}.
$$

より厳密には、$a=\sup S$ は次を満たす。

1. 全ての $s\in S$ に対して $s\le a$。
2. 任意の $\varepsilon>0$ に対して、$a-\varepsilon<s$ を満たす $s\in S$ が存在する。

一方、最大値 $\max S$ は $S$ 自身の要素でなければならない。

$$
S=(0,1)
$$

では

$$
\sup S=1
$$

だが、$1\notin S$ なので最大値は存在しない。

$$
S=[0,1]
$$

なら

$$
\sup S=\max S=1.
$$

従って、一般には

$$
\sup S
$$

と

$$
\max S
$$

は同じとは限らない。

## 5. なぜ有限次元行列では`sup = max`なのか

spectral normでは、scale不変性により単位球面

$$
\mathbb{S}^{n-1}
=
\{x\in\mathbb{R}^{n}\mid\|x\|_2=1\}
$$

上の関数

$$
g(x)=\|Ax\|_2
$$

を考えればよい。

有限次元Euclidean空間では、単位球面は閉かつ有界なのでcompactである。また、

$$
Ax
=
\begin{pmatrix}
\sum_{j=1}^{n}A_{1j}x_j\\
\vdots\\
\sum_{j=1}^{n}A_{mj}x_j
\end{pmatrix}
$$

は $x$ の連続関数であり、Euclidean normも連続なので $g$ は連続である。極値定理により、連続関数 $g$ はcompact集合 $\mathbb{S}^{n-1}$ 上で最大値を取る。

従って、有限次元行列では

$$
\sup_{\|x\|_2=1}\|Ax\|_2
=
\max_{\|x\|_2=1}\|Ax\|_2.
$$

SVDは、その最大値が $\sigma_1$ であり、達成方向が $v_1$ であることまで具体的に示す。

operator normの定義を一般の関数空間にも使える形にするため、定義には通常 $\sup$ を用いる。無限次元では、上限を達成するベクトルが存在しない場合がある。

## 6. 全要素を表示した小さい例

$$
A
=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix},
\qquad
x
=
\begin{pmatrix}
x_1\\
x_2
\end{pmatrix}
$$

とする。このとき、

$$
Ax
=
\begin{pmatrix}
3x_1\\
x_2
\end{pmatrix},
$$

$$
\|x\|_2^2=x_1^2+x_2^2,
\qquad
\|Ax\|_2^2=9x_1^2+x_2^2.
$$

従って、

$$
\left(
\frac{\|Ax\|_2}{\|x\|_2}
\right)^2
=
\frac{9x_1^2+x_2^2}{x_1^2+x_2^2}
\le9.
$$

$x=e_1=(1,0)^{\mathsf T}$ なら、

$$
\frac{\|Ae_1\|_2}{\|e_1\|_2}
=3.
$$

$x=e_2=(0,1)^{\mathsf T}$ なら、

$$
\frac{\|Ae_2\|_2}{\|e_2\|_2}
=1.
$$

従って、

$$
\|A\|_2=3.
$$

同じ行列のFrobenius normは

$$
\|A\|_F
=
\sqrt{3^2+1^2}
=
\sqrt{10}
$$

である。spectral normは最大方向の増幅率を、Frobenius normは全特異方向の二乗和を測る。

## 7. 代表的な行列normとの違い

行列 $A=(A_{ij})$ に対して、代表的なnormは次の意味を持つ。

- spectral norm：$\|A\|_2=\sigma_1$。vector 2-normに関する最大増幅率。
- Frobenius norm：$\|A\|_F=(\sum_{i,j}|A_{ij}|^2)^{1/2}=(\sum_k\sigma_k^2)^{1/2}$。全要素・全特異方向の総量。
- nuclear norm：$\|A\|_*=\sum_k\sigma_k$。特異値の総和。
- induced 1-norm：$\|A\|_1=\max_j\sum_i|A_{ij}|$。列絶対値和の最大。
- induced infinity-norm：$\|A\|_\infty=\max_i\sum_j|A_{ij}|$。行絶対値和の最大。

誤差ベクトルをEuclidean normで測り、行列作用後の最悪増幅を一様に抑えたいときはspectral normが自然に現れる。

## 8. TTLinearの誤差伝播への接続

第2 Linearの重みを $W_2$、hidden activation誤差を $\delta h$ とすると、

$$
\|W_2\delta h\|_2
\le
\|W_2\|_2\|\delta h\|_2.
$$

従って、$\|W_2\|_2$ はhidden errorがlogitsへ伝わるときの最悪増幅率になる。batch全体の導出は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]を参照する。

