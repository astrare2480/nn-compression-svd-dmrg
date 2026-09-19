---
title: TT・MPSの等長写像と射影
aliases:
  - TTのisometry
  - U転置UとUU転置
  - MPSの等長写像
  - TTの射影
tags:
  - TT
  - MPS
  - isometry
  - projection
  - orthogonality
---

# TT・MPSの等長写像と射影

## サマリー

TT-SVDで得る左特異ベクトル行列

$$
U\in\mathbb R^{m\times r},
\qquad
m\ge r
$$

は、列を正規直交化して使うため

$$
\boxed{U^TU=I_r}
$$

を満たす。

これは $r$ 次元の入力を $m$ 次元空間へ送る**等長写像**であり、入力側の長さ・内積を保存する。一方、

$$
\boxed{P=UU^T}
$$

は一般には単位行列ではなく、$U$ の列空間への直交射影である。

TT-SVDではこの二つの意味を区別することが重要である。

---

## 1. $U^TU=I$ は列直交性

$U$ の列を

$$
u_1,\ldots,u_r
$$

と書く。

SVDの左特異ベクトルは直交規格化されるので、

$$
u_a^Tu_b
=
\delta_{ab}.
$$

これを行列でまとめると、

$$
\boxed{U^TU=I_r}
$$

である。

PyTorchでは

```python
UtU = U.T @ U
```

に対応する。

shapeは

```text
U   : (m, r)
U.T : (r, m)

U.T @ U
→ (r, r)
```

である。

---

## 2. なぜ等長写像なのか

任意の

$$
x\in\mathbb R^r
$$

に対して、$U$ を掛けた後のノルムを計算する。

$$
\begin{aligned}
\|Ux\|_2^2
&=
(Ux)^T(Ux)\\
&=
x^TU^TUx\\
&=
x^TI_rx\\
&=
x^Tx\\
&=
\|x\|_2^2.
\end{aligned}
$$

したがって、

$$
\boxed{\|Ux\|_2=\|x\|_2}
$$

である。

同様に $x,y\in\mathbb R^r$ に対して、

$$
\begin{aligned}
\langle Ux,Uy\rangle
&=(Ux)^T(Uy)\\
&=x^TU^TUy\\
&=x^Ty\\
&=\langle x,y\rangle.
\end{aligned}
$$

よって $U$ は、

- 長さ
- 内積
- 角度
- 入力ベクトル間の区別

を保ったまま、低次元空間をより大きい空間へ埋め込む。

この意味で

$$
\boxed{U:\mathbb R^r\to\mathbb R^m}
$$

は等長写像（isometry）である。

---

## 3. $UU^T$ は一般には単位行列ではない

同じ $U$ について、

$$
UU^T
$$

を考える。

shapeは

```text
U @ U.T
→ (m, m)
```

である。

$r<m$ の長方形 $U$ では一般に

$$
UU^T\ne I_m.
$$

代わりに

$$
\boxed{P:=UU^T}
$$

は $U$ の列空間への直交射影である。

---

## 4. なぜ $UU^T$ が射影なのか

まず、

$$
P=UU^T
$$

は対称である。

$$
P^T
=
(UU^T)^T
=
UU^T
=
P.
$$

さらに、

$$
\begin{aligned}
P^2
&=(UU^T)(UU^T)\\
&=U(U^TU)U^T\\
&=UI_rU^T\\
&=UU^T\\
&=P.
\end{aligned}
$$

したがって、

$$
\boxed{P^2=P}
$$

である。

任意の

$$
y\in\mathbb R^m
$$

に対して、

$$
Py
=
UU^Ty
$$

は、まず

$$
U^Ty
$$

で $y$ の $U$ 列方向の成分を取り出し、その係数を

$$
U(U^Ty)
$$

で元の $m$ 次元空間へ戻す。

したがって、$y$ のうち $U$ の列空間に含まれる部分だけが残る。

射影先の次元は、対角和からも確かめられる。トレースの巡回性と $U^TU=I_r$ を使うと

$$
\operatorname{tr}(P)
=\operatorname{tr}(UU^T)
=\operatorname{tr}(U^TU)
=\operatorname{tr}(I_r)=r.
$$

$P^T=P$、$P^2=P$、$\operatorname{tr}(P)=r$ は、それぞれ対称性、冪等性、射影先の次元を確認する別の検査である。数値計算で表示を4桁に丸めた行列では、積の等式は厳密な0や1ではなく丸め誤差の範囲で読む。対角成分が個別に0か1とは限らないが、その和は正規直交列の本数 $r$ に近い。トレースだけが $r$ に近い任意の行列を射影と判定してはいけない。

### 列が正規化されていないときの反例

$Q^TQ=I$ なら $QQ^T$ は直交射影である。しかし任意の $Q$ について成り立つわけではなく、正規化されていない1列だけでも違いが現れる。$q=(2,0)^T$ と $x=(3,5)^T$ を使い、全要素を掛けると

$$
q^Tq=\begin{pmatrix}2&0\end{pmatrix}
\begin{pmatrix}2\\0\end{pmatrix}=4,
\qquad
qq^T=\begin{pmatrix}2\\0\end{pmatrix}
\begin{pmatrix}2&0\end{pmatrix}
=\begin{pmatrix}4&0\\0&0\end{pmatrix}.
$$

$$
(qq^T)x
=\begin{pmatrix}4&0\\0&0\end{pmatrix}
\begin{pmatrix}3\\5\end{pmatrix}
=\begin{pmatrix}12\\0\end{pmatrix},
\qquad
(qq^T)^2=\begin{pmatrix}16&0\\0&0\end{pmatrix}\ne qq^T.
$$

$x$ を $q$ の張る $x$ 軸へ正しく射影した結果は $(3,0)^T$ である。非零の1列 $q$ なら、長さの二乗で割って

$$
P_q=\frac{qq^T}{q^Tq}
=\frac14\begin{pmatrix}4&0\\0&0\end{pmatrix}
=\begin{pmatrix}1&0\\0&0\end{pmatrix},
\qquad
P_qx=\begin{pmatrix}3\\0\end{pmatrix}
$$

となる。QRで得た $Q$ は列が正規直交するため、この分母を別に掛けずに $QQ^T$ を左ブロックの直交射影として使える。一般のフル列ランク行列 $A$ では $A(A^TA)^{-1}A^T$ が対応する式であり、$A^TA=I$ のときに $AA^T$ へ簡約される。

---

## 5. 小さい具体例

$$
U=
\begin{bmatrix}
1&0\\
0&1\\
0&0
\end{bmatrix}
\in\mathbb R^{3\times2}
$$

を考える。

このとき、

$$
U^TU
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
=
I_2.
$$

一方、

$$
UU^T
=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&0
\end{bmatrix}.
$$

この $P=UU^T$ の対角和は $1+1+0=2$ で、列が2本という先ほどの $\operatorname{tr}(P)=r$ と一致する。$I_3$ の対角和は3なので、二つの行列は「射影先の次元」からも区別できる。

これは $I_3$ ではない。

$$
y=
\begin{bmatrix}
a\\b\\c
\end{bmatrix}
$$

へ作用させると、

$$
UU^Ty
=
\begin{bmatrix}
a\\b\\0
\end{bmatrix}.
$$

つまり $z$ 方向を落とし、$U$ の列が張る $xy$ 平面へ射影する。

### 行列積・入力側の埋め込みまで展開する

この $3\times2$ の $U$ について、転置を含む積と入力への作用を全要素で確認する。

$$
\begin{aligned}
U^T U
&=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}
\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}\\
&=\begin{pmatrix}
1\cdot1+0\cdot0+0\cdot0&1\cdot0+0\cdot1+0\cdot0\\
0\cdot1+1\cdot0+0\cdot0&0\cdot0+1\cdot1+0\cdot0
\end{pmatrix}
=I_2,\\
UU^T
&=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}
\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}
=\begin{pmatrix}1&0&0\\0&1&0\\0&0&0\end{pmatrix}.
\end{aligned}
$$

入力 $x=(x_1,x_2)^T$ への作用も全要素で

$$
\begin{aligned}
Ux
&=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}
\begin{pmatrix}x_1\\x_2\end{pmatrix}
=\begin{pmatrix}x_1\\x_2\\0\end{pmatrix},\\
\|Ux\|_2^2&=x_1^2+x_2^2+0^2=x_1^2+x_2^2=\|x\|_2^2,\\
U^T(Ux)&=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}
\begin{pmatrix}x_1\\x_2\\0\end{pmatrix}
=\begin{pmatrix}x_1\\x_2\end{pmatrix},\\
U^Ty&=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}
\begin{pmatrix}a\\b\\c\end{pmatrix}
=\begin{pmatrix}a\\b\end{pmatrix},\\
U(U^Ty)&=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}
=\begin{pmatrix}a\\b\\0\end{pmatrix}.
\end{aligned}
$$

入力側の往復 $U^TU$ は全ての $x\in\mathbb R^2$ を戻す。
出力側の往復 $UU^T$ は、$c\ne0$ の $y\in\mathbb R^3$ を元どおりに戻さない。
長方形の等長写像と全空間で可逆なユニタリ変換の違いが、二つの往復の差として具体的に見える。

---

## 6. TT-SVDでの意味

TT-SVDの第 $k$ 段階では、SVDする行列は

$$
M_k
\in
\mathbb R^{(r_{k-1}n_k)\times N_R}
$$

である。

残す左特異ベクトルを

$$
U_k
\in
\mathbb R^{(r_{k-1}n_k)\times r_k}
$$

とする。

すると、

$$
U_k^TU_k
=
I_{r_k}.
$$

この $U_k$ を

$$
G^{(k)}
=
\operatorname{reshape}
\left(U_k,r_{k-1},n_k,r_k\right)
$$

としてTT coreにする。

成分で書けば、

$$
\boxed{
\sum_{\alpha_{k-1},i_k}
G^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
G^{(k)}_{\alpha_{k-1},i_k,\beta_k}
=
\delta_{\alpha_k,\beta_k}
}
$$

である。

これは $U_k^TU_k=I$ をcore添字で書き直したものに過ぎない。

ここで意味するのは、**別々のTT core同士が直交することではない**。現在coreを左側

$$
(\alpha_{k-1},i_k)
$$

と右bond

$$
\alpha_k
$$

に行列化したとき、その列が正規直交しているという意味である。

---

## 7. exactとtruncationで $UU^T$ をどう読むか

### exact TT-SVD

数値rankを全て残した場合、$U$ の列は現在のSVD行列で必要な左部分空間を全て張る。

したがって、現在のremainderが実際に持つ列空間に対しては情報を失わない。

### truncated TT-SVD

$$
\widetilde r_k<r_k
$$

へ打ち切ると、保持した $U$ の列空間は元より小さい。

$$
UU^T
$$

は、選んだ上位特異方向が張る低rank部分空間への射影になる。

つまり、

```text
U^T U = I
→ 残した座標同士の長さ・内積は保存

U U^T = P
→ 元空間から「残した部分空間」だけを取り出す
```

と読める。

---

## 8. rank保存との関係

$U^TU=I$ なら $U$ は左逆 $U^T$ を持つ。

$$
B=U^T(UB).
$$

したがって、

$$
\operatorname{rank}(UB)
\le
\operatorname{rank}(B)
$$

と、

$$
\operatorname{rank}(B)
=
\operatorname{rank}(U^TUB)
\le
\operatorname{rank}(UB)
$$

を合わせて、

$$
\boxed{
\operatorname{rank}(UB)
=
\operatorname{rank}(B)
}
$$

となる。

TTの第2cutで使う

$$
L_2
=
U\otimes I_{n_2}
$$

についても同じ構造で、

$$
L_2^TL_2=I
$$

なのでrankを保存する。完全な導出は [[34_基底変換とTT-rank不変性]] を参照する。

---

## 9. 量子・MPSでは転置を随伴に置き換える

複素量子状態では、実数の転置 $T$ の代わりに共役転置

$$
\dagger
$$

を使う。

長方形の等長写像では、

$$
\boxed{U^\dagger U=I}
$$

である。

一方、

$$
UU^\dagger
=
\Pi
$$

は出力空間のうち $U$ で到達できる部分空間への射影である。

正方行列でさらに出力空間全体も覆う場合には、

$$
U^\dagger U
=
UU^\dagger
=
I
$$

となり、これはユニタリ変換である。

したがって、

| 式 | 数学的意味 | TT/MPSでの読み方 |
| --- | --- | --- |
| $U^TU=I$ | 列直交・等長写像 | 残したbond座標のノルム・内積を保存 |
| $UU^T=P$ | 列空間への直交射影 | 元空間から保持した左部分空間を取り出す |
| $U^\dagger U=I$ | 複素空間での等長写像 | MPS側の等長条件 |
| $U^\dagger U=UU^\dagger=I$ | 正方ユニタリ | 全空間で可逆なノルム保存変換 |

---

## 10. この章で固定する理解

- 長方形の列直交行列では $U^TU=I$ だが、一般に $UU^T\ne I$。
- $U^TU=I$ は入力側の長さと内積を保存する等長写像を意味する。
- $UU^T$ は $U$ の列空間への直交射影。
- TT-SVDの $U$ は、現在の左blockから保持する直交basisを作る。
- truncated TT-SVDでは $UU^T$ を「保持した低rank部分空間への射影」と読める。
- 量子/MPSでは転置を共役転置へ置き換え、$U^\dagger U=I$ と読む。

[[16_TT_MPSの物理解釈_縮約密度行列と局所写像]] と合わせると、TT-SVDの直交性を数値線形代数と物理の両方から読める。
