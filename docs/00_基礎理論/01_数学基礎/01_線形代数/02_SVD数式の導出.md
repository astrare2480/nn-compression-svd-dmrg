---
title: SVD数式の導出
aliases:
  - SVD途中式
  - 特異値二乗の理由
  - SVD数式補足
tags:
  - SVD
  - 線形代数
  - 低ランク近似
  - 数式導出
---

# SVD数式の導出

## 位置づけ

このノートは、SVD基礎ノートで使う式を**途中式を省略せず**導出する補助ノートである。

主に参照する先：

- [[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]

> [!note]
> 最終式だけでなく、成立条件と等式変形も確認する。Eckart–Young–Mirskyの証明スケッチと入力分布を含む期待出力誤差では、行列近似の保証とtask性能を区別する。

---

## 0. このノートではReduced SVDを基準にする

長方形行列

$$
W\in\mathbb R^{m\times n}
$$

に対して

$$
k=\min(m,n)
$$

と置く。

このノートの導出では、shapeを曖昧にしないためReduced SVD

$$
\boxed{
W=U\Sigma V^{\mathsf T}
}
$$

を基準にする。

$$
U\in\mathbb R^{m\times k},
\qquad
\Sigma\in\mathbb R^{k\times k},
\qquad
V\in\mathbb R^{n\times k}
$$

で、

$$
U^{\mathsf T}U=I_k,
\qquad
V^{\mathsf T}V=I_k
$$

を満たす。

$U,V$ の第$i$列をそれぞれ

$$
u_i=Ue_i,
\qquad
v_i=Ve_i
$$

とする。ここで $e_i\in\mathbb R^k$ は第$i$標準基底である。

完全SVDでも同じ特異ベクトル関係が成り立つが、Reduced SVDを使うと $\Sigma e_i=\sigma_i e_i$ などの途中式のshapeがそのまま一致する。

---

## 1. `Wv_i = σ_i u_i` をSVDから出す

$$
W=U\Sigma V^{\mathsf T}
$$

と

$$
v_i=Ve_i
$$

から、

$$
\begin{aligned}
Wv_i
&=U\Sigma V^{\mathsf T}v_i\\
&=U\Sigma V^{\mathsf T}Ve_i\\
&=U\Sigma e_i
\qquad(V^{\mathsf T}V=I_k)\\
&=U(\sigma_i e_i)\\
&=\sigma_i Ue_i\\
&=\sigma_i u_i.
\end{aligned}
$$

よって、

$$
\boxed{Wv_i=\sigma_i u_i}
$$

となる。

同様に、

$$
W^{\mathsf T}
=(U\Sigma V^{\mathsf T})^{\mathsf T}
=V\Sigma^{\mathsf T}U^{\mathsf T}
$$

であり、Reduced SVDでは $\Sigma^{\mathsf T}=\Sigma$ なので、

$$
\begin{aligned}
W^{\mathsf T}u_i
&=V\Sigma^{\mathsf T}U^{\mathsf T}u_i\\
&=V\Sigma^{\mathsf T}U^{\mathsf T}Ue_i\\
&=V\Sigma e_i\\
&=V(\sigma_i e_i)\\
&=\sigma_iVe_i\\
&=\sigma_i v_i.
\end{aligned}
$$

したがって、

$$
\boxed{W^{\mathsf T}u_i=\sigma_i v_i}
$$

である。

完全SVDで $\Sigma\in\mathbb R^{m\times n}$ と書く場合には、転置式は必ず

$$
W^{\mathsf T}=V\Sigma^{\mathsf T}U^{\mathsf T}
$$

となる点に注意する。

---

## 2. `W^T W` と `WW^T` の固有値が `σ_i^2` になる途中式

上で得た

$$
Wv_i=\sigma_i u_i
$$

へ左から $W^{\mathsf T}$ を掛ける。

$$
W^{\mathsf T}Wv_i
=
\sigma_i W^{\mathsf T}u_i
$$

さらに

$$
W^{\mathsf T}u_i=\sigma_i v_i
$$

を代入すると、

$$
\begin{aligned}
W^{\mathsf T}Wv_i
&=\sigma_i(\sigma_i v_i)\\
&=\sigma_i^2v_i.
\end{aligned}
$$

よって、

$$
\boxed{W^{\mathsf T}Wv_i=\sigma_i^2v_i}
$$

である。

つまり、

```text
W^T Wの固有ベクトル = v_i
W^T Wの固有値       = σ_i^2
```

となる。

同様に、

$$
\begin{aligned}
WW^{\mathsf T}u_i
&=W(\sigma_i v_i)\\
&=\sigma_i Wv_i\\
&=\sigma_i(\sigma_i u_i)\\
&=\sigma_i^2u_i,
\end{aligned}
$$

だから、

$$
\boxed{WW^{\mathsf T}u_i=\sigma_i^2u_i}
$$

となる。

したがって $v_i$ は入力側、$u_i$ は出力側の特異方向であり、

$$
W:\mathbb R^n\to\mathbb R^m,
\qquad
Wv_i=\sigma_i u_i
$$

というshapeと作用から、この対応に統一する。

---

## 3. SVDをrank-1行列の和へ展開する

対角行列は標準基底を使って

$$
\Sigma
=
\sum_{i=1}^{k}
\sigma_i e_i e_i^{\mathsf T}
$$

と書ける。

したがって、

$$
\begin{aligned}
W
&=U\Sigma V^{\mathsf T}\\
&=U
\left(
\sum_{i=1}^{k}
\sigma_i e_i e_i^{\mathsf T}
\right)
V^{\mathsf T}\\
&=
\sum_{i=1}^{k}
\sigma_i
(Ue_i)
(e_i^{\mathsf T}V^{\mathsf T})\\
&=
\sum_{i=1}^{k}
\sigma_i
u_i v_i^{\mathsf T}.
\end{aligned}
$$

よって、

$$
\boxed{
W=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
}
$$

である。

各 $u_iv_i^{\mathsf T}$ はrank 1なので、上位$r$項だけ残せば

$$
\boxed{
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
}
$$

となる。

行列形式では、

$$
W_r
=U_r\Sigma_rV_r^{\mathsf T}
$$

である。

---

## 4. なぜ特異値を「二乗」してenergyを測るのか

特異値は

$$
\sigma_i\ge0
$$

なので、符号を消すために二乗しているわけではない。

理由は、SVD低ランク近似で使うFrobeniusノルムの**二乗**が特異値の二乗和になるからである。

$$
\|W\|_F^2
=
\operatorname{tr}(W^{\mathsf T}W)
$$

へSVDを代入する。

$$
\begin{aligned}
W^{\mathsf T}W
&=(U\Sigma V^{\mathsf T})^{\mathsf T}
(U\Sigma V^{\mathsf T})\\
&=V\Sigma^{\mathsf T}U^{\mathsf T}U\Sigma V^{\mathsf T}\\
&=V\Sigma^{\mathsf T}\Sigma V^{\mathsf T}.
\end{aligned}
$$

traceの巡回性と $V^{\mathsf T}V=I_k$ を使うと、

$$
\begin{aligned}
\|W\|_F^2
&=\operatorname{tr}
(V\Sigma^{\mathsf T}\Sigma V^{\mathsf T})\\
&=\operatorname{tr}
(\Sigma^{\mathsf T}\Sigma V^{\mathsf T}V)\\
&=\operatorname{tr}
(\Sigma^{\mathsf T}\Sigma)\\
&=\sum_{i=1}^{k}\sigma_i^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|W\|_F^2
=
\sum_{i=1}^{k}\sigma_i^2
}
$$

である。

Frobenius energyと呼ぶ量は、この二乗ノルムに対応する。

---

## 5. truncated SVDのFrobenius誤差を途中から出す

rank$r$近似は

$$
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
$$

なので、

$$
\begin{aligned}
W-W_r
&=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
-
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}\\
&=
\sum_{i=r+1}^{k}
\sigma_i u_i v_i^{\mathsf T}.
\end{aligned}
$$

rank-1成分同士のFrobenius内積は、

$$
\begin{aligned}
\left\langle
u_i v_i^{\mathsf T},
 u_j v_j^{\mathsf T}
\right\rangle_F
&=
\operatorname{tr}
\left[
(u_i v_i^{\mathsf T})^{\mathsf T}
(u_j v_j^{\mathsf T})
\right]\\
&=
\operatorname{tr}
\left[
v_i u_i^{\mathsf T}u_jv_j^{\mathsf T}
\right]\\
&=(u_i^{\mathsf T}u_j)
(v_i^{\mathsf T}v_j)\\
&=\delta_{ij}.
\end{aligned}
$$

したがって、異なる$i$の成分はFrobenius内積で直交し、

$$
\begin{aligned}
\|W-W_r\|_F^2
&=
\left\|
\sum_{i=r+1}^{k}
\sigma_i u_i v_i^{\mathsf T}
\right\|_F^2\\
&=
\sum_{i=r+1}^{k}
\sigma_i^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|W-W_r\|_F
=
\sqrt{
\sum_{i=r+1}^{k}\sigma_i^2
}
}
$$

となる。

$r=k$ なら和は空和なので

$$
\|W-W_k\|_F=0
$$

である。

---

## 6. retained energyと相対Frobenius誤差の関係

保持energyを

$$
E(r)
=
\frac{
\sum_{i=1}^{r}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}
$$

とする。

相対Frobenius誤差の二乗は、

$$
\begin{aligned}
\varepsilon_F(r)^2
&=
\frac{
\|W-W_r\|_F^2
}{
\|W\|_F^2
}\\
&=
\frac{
\sum_{i=r+1}^{k}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}.
\end{aligned}
$$

全体は

$$
\sum_{i=1}^{k}\sigma_i^2
=
\sum_{i=1}^{r}\sigma_i^2
+
\sum_{i=r+1}^{k}\sigma_i^2
$$

だから、

$$
\sum_{i=r+1}^{k}\sigma_i^2
=
\sum_{i=1}^{k}\sigma_i^2
-
\sum_{i=1}^{r}\sigma_i^2.
$$

これを代入すると、

$$
\begin{aligned}
\varepsilon_F(r)^2
&=
1-
\frac{
\sum_{i=1}^{r}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}\\
&=1-E(r).
\end{aligned}
$$

したがって、

$$
\boxed{
\varepsilon_F(r)=\sqrt{1-E(r)}
}
$$

となる。

---

## 7. Eckart–Young–Mirskyの最適性を式で追う

切り詰めSVD $W_r$ は、rankが$r$以下の行列の中でFrobenius normおよびspectral normに対する最良近似である。

$$
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F
$$

$$
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
$$

ここではFrobenius側の理由を、直交変換不変性まで追う。

### この小節では完全SVDへ拡張する

任意の比較行列 $B$ のnormとrankまで保存するため、**この小節だけ**は

$$
U\in\mathbb R^{m\times m},
\qquad
V\in\mathbb R^{n\times n},
\qquad
\Sigma\in\mathbb R^{m\times n},
\qquad
U^TU=UU^T=I_m,
\qquad
V^TV=VV^T=I_n
$$

という完全SVDの記号を使う。冒頭のreduced SVDの列直交行列とはshapeが異なる。reducedの長方形 $U,V$ のまま、任意の $B$ に対して以下のnorm・rankの等号を使うことはできない。

例えば、

$$
U_{\mathrm{red}}=\begin{pmatrix}1\\0\end{pmatrix},
\qquad
V_{\mathrm{red}}=(1),
\qquad
B=\begin{pmatrix}0\\1\end{pmatrix}
$$

なら、

$$
U_{\mathrm{red}}^TBV_{\mathrm{red}}=(0),
\qquad
\|B\|_F=1,
\qquad
\operatorname{rank}(B)=1
$$

であり、射影でnormもrankも落ちる。完全SVDへ拡張してから座標変換する必要がある。

任意のrank$r$以下の $B$ に対し、

$$
C
=U^{\mathsf T}BV
$$

と置く。直交変換はFrobenius normを保つので、

$$
\begin{aligned}
\|W-B\|_F
&=\|U^{\mathsf T}(W-B)V\|_F\\
&=\|U^{\mathsf T}WV-U^{\mathsf T}BV\|_F\\
&=\|\Sigma-C\|_F.
\end{aligned}
$$

また、直交行列を掛けてもrankは変わらないため、

$$
\operatorname{rank}(C)
=\operatorname{rank}(B)
\le r.
$$

したがって問題は、対角行列 $\Sigma$ をrank$r$以下の $C$ で近似する問題へ移る。

以下の $\operatorname{diag}$ は、ここでは長方形の $\Sigma$ と同じ $m\times n$ のshapeを保つ略記である。すなわち $\Sigma_r$ の $(i,i)$ 成分を $i\le r$ では $\sigma_i$、$i>r$ では0とし、対角以外の全成分も0にする。

上位$r$個の対角成分だけを残した

$$
\Sigma_r
=
\operatorname{diag}
(\sigma_1,\ldots,\sigma_r,0,\ldots)
$$

を使えば、

$$
\|\Sigma-\Sigma_r\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2.
$$

Eckart–Young–Mirskyの定理は、rank$r$以下のどの $C$ もこの値より小さい誤差にはできないことを保証する。元の座標へ戻すと、

$$
U\Sigma_rV^{\mathsf T}
=W_r
$$

であり、

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2
}
$$

となる。

spectral normでは $r<k$ のとき

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
=
\sigma_{r+1}
}
$$

であり、$r=k$ なら誤差は0である。

### 「どのrank-$r$行列もこれより良くならない」を導く

前の議論は直交変換で問題を移したところまでだった。最適性そのものの下界を、定理名に委ねず計算する。

任意の $\operatorname{rank}(B)=p\le r$ に対し、$B$ の列空間への直交射影を $P$ とする。

$$
P^T=P,\qquad P^2=P,\qquad PB=B,\qquad\operatorname{tr}(P)=p.
$$

残差を列空間とその直交補空間へ分けると、

$$
\begin{aligned}
W-B&=P(W-B)+(I-P)(W-B),\\
\langle P(W-B),(I-P)(W-B)\rangle_F
&=\operatorname{tr}((W-B)^TP(I-P)(W-B))=0,\\
\|W-B\|_F^2
&=\|P(W-B)\|_F^2+\|(I-P)W\|_F^2\\
&\ge\|(I-P)W\|_F^2\\
&=\operatorname{tr}(W^T(I-P)W)\\
&=\|W\|_F^2-\operatorname{tr}(PWW^T).
\end{aligned}
$$

完全な左特異基底を使い、$i>k$ では $\sigma_i=0$ と置く。重み

$$
w_i:=u_i^TPu_i=\|Pu_i\|_2^2
$$

は

$$
0\le w_i\le1,\qquad
\sum_{i=1}^{m}w_i=\operatorname{tr}(U^TPU)=\operatorname{tr}(P)=p\le r
$$

を満たす。$1\le r<k$ について、特異値の降順性から

$$
\begin{aligned}
\operatorname{tr}(PWW^T)
&=\sum_{i=1}^{m}\sigma_i^2w_i\\
&\le\sum_{i=1}^{r}\sigma_i^2w_i
+\sigma_r^2\sum_{i=r+1}^{m}w_i\\
&\le\sum_{i=1}^{r}\sigma_i^2w_i
+\sigma_r^2\left(r-\sum_{i=1}^{r}w_i\right)\\
&=r\sigma_r^2+\sum_{i=1}^{r}(\sigma_i^2-\sigma_r^2)w_i\\
&\le r\sigma_r^2+\sum_{i=1}^{r}(\sigma_i^2-\sigma_r^2)\\
&=\sum_{i=1}^{r}\sigma_i^2.
\end{aligned}
$$

したがって、

$$
\|W-B\|_F^2
\ge\sum_{i=1}^{k}\sigma_i^2-\sum_{i=1}^{r}\sigma_i^2
=\sum_{i=r+1}^{k}\sigma_i^2.
$$

$B=W_r$ が既出の誤差公式によってこの下界を達成するので、Frobenius側の最適性が証明された。$r=0$ では $B=0$ のみ、$r=k$ では $B=W$ で誤差0になる。

spectral側も、$S=\operatorname{span}\{v_1,\ldots,v_{r+1}\}$ への $B$ の制限のrankが $r$ 以下なので、単位ベクトル $x\in S$ で $Bx=0$ となるものが存在する。これを

$$
x=\sum_{i=1}^{r+1}c_iv_i,\qquad\sum_{i=1}^{r+1}c_i^2=1
$$

と書けば、

$$
\begin{aligned}
\|W-B\|_2^2
&\ge\|(W-B)x\|_2^2\\
&=\|Wx\|_2^2\\
&=\sum_{i=1}^{r+1}\sigma_i^2c_i^2\\
&\ge\sigma_{r+1}^2\sum_{i=1}^{r+1}c_i^2\\
&=\sigma_{r+1}^2.
\end{aligned}
$$

一方、

$$
\|(W-W_r)x\|_2^2
=\sum_{i=r+1}^{k}\sigma_i^2(v_i^Tx)^2
\le\sigma_{r+1}^2\|x\|_2^2
$$

であり、$x=v_{r+1}$ で等号になる。よって $W_r$ がspectral側の下界も達成する。

> [!important]
> 「最良」は、切り詰めSVDより**小さい誤差**を持つrank$r$以下の行列がない、という意味である。境界で特異値が重複する場合、同じ最小誤差を持つ別の最適解が存在し得るので、常に一意とは限らない。

---

## 8. weight errorから1入力のoutput errorへ

差分を

$$
\Delta W=W-W_r
$$

とする。

入力$x$に対する出力差は

$$
\Delta y
=(W-W_r)x
=\Delta W x.
$$

行列のspectral normの定義から、

$$
\|\Delta W x\|_2
\le
\|\Delta W\|_2\|x\|_2.
$$

$r<k$ のtruncated SVDでは

$$
\|W-W_r\|_2
=
\sigma_{r+1}
$$

なので、

$$
\boxed{
\|(W-W_r)x\|_2
\le
\sigma_{r+1}\|x\|_2
}
$$

となる。

$r=k$ なら $W_r=W$ なので出力差は0である。

これは上限であって、実データで必ずこの大きさの誤差が出るという意味ではない。

---

## 9. 入力分布を含めた期待output error

weight Frobenius errorと実データ上のoutput errorを分けて考える。

$$
\Delta W=W-W_r
$$

と置くと、

$$
\begin{aligned}
\|\Delta W x\|_2^2
&=(\Delta W x)^{\mathsf T}(\Delta W x)\\
&=x^{\mathsf T}\Delta W^{\mathsf T}\Delta W x.
\end{aligned}
$$

scalarはtraceで書けるので、

$$
\begin{aligned}
x^{\mathsf T}\Delta W^{\mathsf T}\Delta W x
&=
\operatorname{tr}
(x^{\mathsf T}\Delta W^{\mathsf T}\Delta W x)\\
&=
\operatorname{tr}
(\Delta W x x^{\mathsf T}\Delta W^{\mathsf T}).
\end{aligned}
$$

期待値を取る。

$$
\begin{aligned}
\mathbb E
[\|\Delta W x\|_2^2]
&=
\mathbb E
\left[
\operatorname{tr}
(\Delta W x x^{\mathsf T}\Delta W^{\mathsf T})
\right]\\
&=
\operatorname{tr}
\left[
\Delta W
\mathbb E[xx^{\mathsf T}]
\Delta W^{\mathsf T}
\right].
\end{aligned}
$$

$$
C_x=\mathbb E[xx^{\mathsf T}]
$$

と置けば、

$$
\boxed{
\mathbb E
[\|(W-W_r)x\|_2^2]
=
\operatorname{tr}
[(W-W_r)C_x(W-W_r)^{\mathsf T}]
}
$$

となる。

ここで $C_x$ は一般には**非中心化二次モーメント**である。平均

$$
\mu=\mathbb E[x]
$$

を使う共分散は

$$
\operatorname{Cov}(x)
=
\mathbb E[(x-\mu)(x-\mu)^{\mathsf T}]
$$

なので、$\mu=0$ のときに両者が一致する。

この式から、通常SVDがweightだけを見て最良でも、実データ分布 $C_x$ に対するoutput errorで常に最良とは限らないことが分かる。

---

### 入力分布の重み付きFrobenius誤差へ変形する

出力保存目的との接続を、traceから続ける。入力の二次モーメントを $C_x=\mathbb E[xx^{\mathsf T}]$ とし、

$$
C_x^{1/2}=(C_x^{1/2})^{\mathsf T},
\qquad
C_x^{1/2}C_x^{1/2}=C_x
$$

を満たす半正定値平方根を使うと、

$$
\begin{aligned}
\mathbb E[\|\Delta W x\|_2^2]
&=\operatorname{tr}(\Delta W C_x\Delta W^{\mathsf T})\\
&=\operatorname{tr}[
(\Delta W C_x^{1/2})(\Delta W C_x^{1/2})^{\mathsf T}]\\
&=\|\Delta W C_x^{1/2}\|_F^2.
\end{aligned}
$$

平均 $\mu=\mathbb E[x]$ が0でないときは、

$$
\begin{aligned}
C_x
&=\mathbb E[((x-\mu)+\mu)((x-\mu)+\mu)^{\mathsf T}]\\
&=\operatorname{Cov}(x)+\mu\mu^{\mathsf T},\\
\mathbb E[\|\Delta W x\|_2^2]
&=\operatorname{tr}(\Delta W\operatorname{Cov}(x)\Delta W^{\mathsf T})
+\|\Delta W\mu\|_2^2.
\end{aligned}
$$

したがって、二次モーメントを無条件に中心化済みの共分散へ置き換えると、平均方向の誤差が抜ける。

本節独自の小さい実行列で、例えば

$$
\Delta W=\begin{pmatrix}1&2\end{pmatrix},
\qquad
C_x=\begin{pmatrix}4&1\\1&3\end{pmatrix}
$$

なら、

$$
\Delta WC_x=\begin{pmatrix}1\cdot4+2\cdot1&1\cdot1+2\cdot3\end{pmatrix}
=\begin{pmatrix}6&7\end{pmatrix},
$$

$$
\mathbb E[\|\Delta Wx\|_2^2]
=\begin{pmatrix}6&7\end{pmatrix}\begin{pmatrix}1\\2\end{pmatrix}
=6+14=20.
$$

平方根の代わりに $C_x=LL^{\mathsf T}$ を満たす因子を使っても同じnormになる。この例では、

$$
L=\begin{pmatrix}2&0\\1/2&\sqrt{11}/2\end{pmatrix},
\qquad
LL^{\mathsf T}=\begin{pmatrix}4&1\\1&3\end{pmatrix},
\qquad
\Delta WL=\begin{pmatrix}3&\sqrt{11}\end{pmatrix},
$$

$$
\|\Delta WL\|_F^2=3^2+(\sqrt{11})^2=20.
$$

一方、重み付けのない $\|\Delta W\|_F^2=1^2+2^2=5$ である。これが、weight Frobenius誤差の最小化と入力分布を考慮した出力誤差の最小化を区別する具体例になる。

---

## 10. Linear 2層化のparameter条件を最初から導く

元Linearのweight数は

$$
P_W=D_{\mathrm{in}}D_{\mathrm{out}}.
$$

rank$r$の2因子は

$$
A\in\mathbb R^{r\times D_{\mathrm{in}}},
\qquad
B\in\mathbb R^{D_{\mathrm{out}}\times r}.
$$

したがって、

$$
\begin{aligned}
P_{\mathrm{lowrank},W}
&=rD_{\mathrm{in}}+D_{\mathrm{out}}r\\
&=r(D_{\mathrm{in}}+D_{\mathrm{out}}).
\end{aligned}
$$

biasありでも、元と圧縮後の最終biasはどちらも $D_{\mathrm{out}}$ 個なので、

$$
\begin{aligned}
&r(D_{\mathrm{in}}+D_{\mathrm{out}})+D_{\mathrm{out}}
<
D_{\mathrm{in}}D_{\mathrm{out}}+D_{\mathrm{out}}\\
\Longleftrightarrow\;&
r(D_{\mathrm{in}}+D_{\mathrm{out}})
<
D_{\mathrm{in}}D_{\mathrm{out}}\\
\Longleftrightarrow\;&
\boxed{
r<
\frac{D_{\mathrm{in}}D_{\mathrm{out}}}
{D_{\mathrm{in}}+D_{\mathrm{out}}}
}.
\end{aligned}
$$

rankは整数なので、圧縮が成立する最大整数rankは

$$
\boxed{
r_{\max,\mathrm{compress}}
=
\left\lceil
\frac{D_{\mathrm{in}}D_{\mathrm{out}}}
{D_{\mathrm{in}}+D_{\mathrm{out}}}
\right\rceil-1
}
$$

である。

---

## 11. 対象層削減率からモデル全体削減率 `qs` を導く

モデル総parameterを $P$、対象層を $P_t$ とする。

対象層がモデルに占める割合を

$$
q=\frac{P_t}{P}
$$

対象層内部の削減率を

$$
s=1-\frac{P_{t,\mathrm{low}}}{P_t}
$$

とする。

後者から、

$$
P_{t,\mathrm{low}}=(1-s)P_t
$$

である。

圧縮後モデル全体は、

$$
\begin{aligned}
P_{\mathrm{comp}}
&=P-P_t+P_{t,\mathrm{low}}\\
&=P-P_t+(1-s)P_t\\
&=P-sP_t\\
&=P-s(qP)\\
&=P(1-qs).
\end{aligned}
$$

モデル全体削減率は、

$$
\begin{aligned}
R_{\mathrm{model}}
&=1-\frac{P_{\mathrm{comp}}}{P}\\
&=1-(1-qs)\\
&=\boxed{qs}.
\end{aligned}
$$

---

## 12. Pareto dominanceを数式で書く

validation lossとparameter数を小さくする2目的でPareto frontierを考える。

候補$i$の目的ベクトルを

$$
f_i=(p_i,\ell_i)
$$

とする。$p_i$ はparameter数、$\ell_i$ はvalidation lossで、ともに小さいほどよい。

候補$a$が候補$b$を支配するとは、

$$
p_a\le p_b,
\qquad
\ell_a\le\ell_b
$$

かつ少なくとも一方で厳密に

$$
p_a<p_b
\quad\text{or}\quad
\ell_a<\ell_b
$$

が成り立つこと。

Pareto frontierは、他候補に支配されない候補の集合である。

---

## 13. Pareto kneeのcross product式を途中から出す

Pareto frontierのparameterとvalidation lossをMin-Max正規化する。

$$
x_i'
=
\frac{x_i-x_{\min}}{x_{\max}-x_{\min}}
$$

$$
y_i'
=
\frac{y_i-y_{\min}}{y_{\max}-y_{\min}}
$$

両端を

$$
p_1=(x_1',y_1'),
\qquad
p_2=(x_M',y_M')
$$

とし、直線方向ベクトルを

$$
\ell=p_2-p_1
=(\Delta x,\Delta y)
$$

とする。

長さは

$$
\|\ell\|_2
=
\sqrt{(\Delta x)^2+(\Delta y)^2}.
$$

候補点を

$$
p_i=(x_i',y_i')
$$

とすると、$p_1$から候補点へのベクトルは

$$
q_i=p_i-p_1.
$$

2次元cross productの大きさ

$$
|\ell\times q_i|
$$

は、底辺 $\|\ell\|_2$、高さ $d_i$ の平行四辺形の面積なので、

$$
|\ell\times q_i|
=
\|\ell\|_2d_i.
$$

したがって、点と直線の距離は

$$
\boxed{
d_i
=
\frac{|\ell\times(p_i-p_1)|}
{\|\ell\|_2}
}.
$$

成分で書けば、

$$
\ell=(a,b),
\qquad
p_i-p_1=(c,d)
$$

なので、

$$
\ell\times(p_i-p_1)=ad-bc
$$

となり、

$$
\boxed{
d_i
=
\frac{|ad-bc|}{\sqrt{a^2+b^2}}
}.
$$

最後に、

$$
\boxed{
k^*=\arg\max_i d_i}
$$

をglobal knee候補とする。

これは唯一の最適解ではなく、Pareto frontierの両端を結ぶ基準線から最も大きく曲がった点を取るヒューリスティックである。

---

## 14. まとめ

```text
Reduced SVDのshape
→ W=UΣV^T
→ Wv_i=σ_i u_i
→ W^T u_i=σ_i v_i
→ W^T W v_i=σ_i^2 v_i

rank-1展開
→ W=Σ_i σ_i u_i v_i^T
→ W-W_r=Σ_{i>r} σ_i u_i v_i^T
→ ||W-W_r||_F^2=Σ_{i>r}σ_i^2

全energy=Σσ_i^2
→ retained energy E(r)
→ relative error^2=1-E(r)

Eckart–Young–Mirsky
→ 直交変換でΣの近似問題へ移す
→ 上位r成分を残す
→ tail singular valuesが最小誤差

weight error
→ Δy=ΔWx
→ spectral norm上限
→ E[xx^T]を含むdata-dependent output error

元parameter
→ 2因子parameter
→ bias相殺
→ break-even rank

Pareto
→ dominance
→ Min-Max
→ 端点ベクトル
→ cross product / line length
→ knee
```
