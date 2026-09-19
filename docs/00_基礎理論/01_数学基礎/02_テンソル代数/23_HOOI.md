---
title: HOOI
aliases:
  - Higher-Order Orthogonal Iteration
  - Tucker ALS
  - HOSVDの反復精密化
tags:
  - HOOI
  - Tucker
  - HOSVD
  - Tensor
  - 交互最適化
---

# HOOI

## サマリー

HOOI（Higher-Order Orthogonal Iteration）は、固定したmultilinear rankのTucker近似で、factorを1 modeずつ反復更新して再構成Frobenius誤差を小さくする方法である。

標準的な直交factorのTucker近似では、目的を

$$
\boxed{
\begin{aligned}
\min_{
\mathcal{G},
U^{(0)},\ldots,U^{(N-1)}
}
&\quad
\left\|
\mathcal{X}
-
\mathcal{G}
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
\right\|_F^2\\
\text{s.t.}
&\quad
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
\qquad(n=0,\ldots,N-1)
\end{aligned}
}
$$

と書ける。

partial HOOIでは、全modeではなく圧縮対象modeの集合を

$$
\mathcal M
\subseteq
\{0,1,\ldots,N-1\}
$$

として、$n\in\mathcal M$ のfactorだけを持つ。現在のsrcでは `ranks.keys()` がこの $\mathcal M$ に対応する。

この問題はfactorを同時に求めると非凸なので、HOOIは1 modeずつ交互に更新する。HOSVDを高速な初期解として使い、**同じrankのまま再構成誤差を精密化する方法**として理解する。

---

## 1. HOSVDとの違い

| 観点 | HOSVD | HOOI |
| --- | --- | --- |
| factorの求め方 | 各modeを元Tensorから独立にSVD | 他modeを固定して1 modeずつ更新 |
| 反復 | なし | あり |
| rank | 指定rank | 同じ指定rank |
| 主な用途 | 高速な近似・初期解・rank探索 | 同rankでの再構成精密化 |
| コスト | 小さい | sweep分だけ大きい |

HOSVDは

$$
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\left(
\operatorname{left\ singular\ vectors}(X_{(n)})
\right)
$$

を各modeで**元の同じTensorから独立に**行う。

HOOIでは、更新対象以外のfactorを使ってTensorをrank空間へ射影してから、その対象modeをunfoldしてSVDする。

---

## 2. factorが固定されたときのcore

ここで「factor固定」は、各modeで保持する基底は既に決まっていて、その基底の組合せへどの係数を付けるかだけを決める、という意味である。
coreを元Tensorの一部の値から選ぶのではなく、元Tensorと基底の積との内積で係数を測る。
正規直交factorなら、各基底の組合せもTensorのFrobenius内積に対して正規直交するので、係数同士の混合を解く必要がない。

3階の場合、基底の組合せを

$$
E^{(\alpha_0,\alpha_1,\alpha_2)}_{i_0,i_1,i_2}
:=U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
$$

と書けば、coreの一要素は

$$
G_{\alpha_0,\alpha_1,\alpha_2}
=\langle E^{(\alpha_0,\alpha_1,\alpha_2)},X\rangle_F
=\sum_{i_0,i_1,i_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
X_{i_0,i_1,i_2}.
$$

これはベクトルの座標 $c_\alpha=u_\alpha^{\mathsf T}x$ を三つのmodeへ広げた式である。
coreをこの最適な係数で作り直した上で、次にfactor自体を更新するのがHOOIである。
coreを固定したまま全factorを一度ずつSVDするというアルゴリズムとは区別する。

factorの列が直交して

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
$$

を満たすとする。

全mode Tuckerなら、現在factorに対するcoreは

$$
\boxed{
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
}
$$

である。

partial Tuckerなら、

$$
\boxed{
\mathcal G
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
U^{(m)\mathsf T}
}
$$

と、圧縮対象modeだけを射影する。

再構成は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\underset{m\in\mathcal M}{\times_m}
U^{(m)}
}
$$

である。

modeごとに

$$
P_m
=
U^{(m)}U^{(m)\mathsf T}
$$

と置けば、$P_m$ はfactor列空間への直交射影であり、再構成は

$$
\hat{\mathcal X}
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
P_m
$$

とも書ける。

---

## 3. なぜ「誤差最小化」と「core norm最大化」が同じか

HOOIの更新式を理解するうえで重要な途中式である。

まず、

$$
\begin{aligned}
\|\mathcal X-\hat{\mathcal X}\|_F^2
={}&
\|\mathcal X\|_F^2
-2\langle\mathcal X,\hat{\mathcal X}\rangle_F
+\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

$\hat{\mathcal X}$ は直交射影なので、残差

$$
\mathcal R
=
\mathcal X-\hat{\mathcal X}
$$

は射影部分 $\hat{\mathcal X}$ と直交する。

$$
\langle\mathcal R,\hat{\mathcal X}\rangle_F=0
$$

したがって、

$$
\begin{aligned}
\langle\mathcal X,\hat{\mathcal X}\rangle_F
&=
\langle\mathcal R+\hat{\mathcal X},\hat{\mathcal X}\rangle_F\\
&=
\langle\mathcal R,\hat{\mathcal X}\rangle_F
+
\|\hat{\mathcal X}\|_F^2\\
&=
\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\hat{\mathcal X}\|_F^2
}
$$

となる。

さらに、列直交factorをmode productしてもFrobenius normは変わらない。1 modeだけで確認すると、

$$
\mathcal Y
=
\mathcal G\times_nU^{(n)}
$$

ならunfoldingで

$$
Y_{(n)}
=
U^{(n)}G_{(n)}.
$$

したがって、

$$
\begin{aligned}
\|\mathcal Y\|_F^2
&=\|Y_{(n)}\|_F^2\\
&=\operatorname{tr}
\left[
(U^{(n)}G_{(n)})^{\mathsf T}
(U^{(n)}G_{(n)})
\right]\\
&=\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}
U^{(n)\mathsf T}U^{(n)}
G_{(n)}
\right]\\
&=\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}G_{(n)}
\right]\\
&=\|\mathcal G\|_F^2.
\end{aligned}
$$

これを各modeへ順に適用すると、

$$
\|\hat{\mathcal X}\|_F^2
=
\|\mathcal G\|_F^2.
$$

したがって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\mathcal G\|_F^2
}
$$

である。

$\|\mathcal X\|_F^2$ は固定なので、

$$
\boxed{
\min\|\mathcal X-\hat{\mathcal X}\|_F^2
\quad\Longleftrightarrow\quad
\max\|\mathcal G\|_F^2
}
$$

となる。

これが「HOOIは射影後coreができるだけ多くのFrobenius energyを持つようにfactorを更新する」と言える理由である。

---

## 4. 1 factorだけ更新すると何を最大化するか

更新対象以外のfactorを固定するのは、そのmodeだけの行列SVDで解ける局所問題にするためである。
他modeで先に座標を取り出したTensor $Z$ は、現在の他factorで保持できる情報だけを含む。
その $Z$ を対象modeでunfoldし、保持できる二乗normが大きい左特異方向を選ぶ。
元Tensor $X$ を毎回独立にunfoldするHOSVDとは、SVDへ入力する行列が違う。

固定された $Z$ の対象modeの列を $z_c$ と書くと、候補factor $U$ による保持量は

$$
\|U^{\mathsf T}Z_{(n)}\|_F^2
=\sum_c\|U^{\mathsf T}z_c\|_2^2
=\sum_c\sum_{\alpha=1}^{R_n}(u_\alpha^{\mathsf T}z_c)^2.
$$

つまり、全列について「選んだ基底に沿う成分の二乗」を足した量である。
一つの列だけに最適な方向ではなく、全列をまとめて保持する部分空間を選ぶ。
第5節のtrace最大化はこの量を行列記法で書き直したものである。

更新対象をmode $n\in\mathcal M$ とする。

他のactive modeを現在のfactorで射影したTensorを

$$
\boxed{
\mathcal Z^{(n)}
=
\mathcal X
\underset{m\in\mathcal M\setminus\{n\}}{\times_m}
U^{(m)\mathsf T}
}
$$

とする。

**更新対象自身のfactor $U^{(n)}$ はこのprojectionに使わない。**

全modeを圧縮する場合、$\mathcal Z^{(n)}$ のshapeは

$$
R_0\times\cdots\times R_{n-1}
\times I_n
\times R_{n+1}\times\cdots\times R_{N-1}
$$

である。

partial HOOIでは、$m\notin\mathcal M$ の次元は元の $I_m$ のまま残る。

mode $n$でunfoldした行列を

$$
Z
=
Z_{(n)}^{(n)}
$$

と略記する。

全mode圧縮なら、

$$
Z
\in
\mathbb R^{I_n\times\prod_{m\ne n}R_m}
$$

である。

候補factor

$$
U
=U^{(n)}
\in
\mathbb R^{I_n\times R_n},
\qquad
U^{\mathsf T}U=I_{R_n}
$$

を最後にmode $n$へ射影すると、coreのmode-$n$ unfoldingは

$$
G_{(n)}
=
U^{\mathsf T}Z
$$

となる。

前節から、他factorを固定したときは

$$
\boxed{
\max_{U^{\mathsf T}U=I}
\|U^{\mathsf T}Z\|_F^2
}
$$

を解けばよい。

---

## 5. なぜ更新factorが「上位左特異ベクトル」になるか

前節の目的を途中から展開する。

$$
\begin{aligned}
\|U^{\mathsf T}Z\|_F^2
&=
\operatorname{tr}
\left[
(U^{\mathsf T}Z)(U^{\mathsf T}Z)^{\mathsf T}
\right]\\
&=
\operatorname{tr}
\left[
U^{\mathsf T}ZZ^{\mathsf T}U
\right].
\end{aligned}
$$

したがって、1 factor更新は

$$
\boxed{
\max_{U^{\mathsf T}U=I_{R_n}}
\operatorname{tr}
\left(U^{\mathsf T}ZZ^{\mathsf T}U\right)
}
$$

という直交部分空間の最大化問題になる。

### 固定factorに対するcoreが最適になる理由

前節で「直交射影」と述べた部分を、未知のcoreに対する最小化から確認する。現在factorによる展開演算子を

$$
\mathcal A(C):=C\underset{m\in\mathcal M}{\times_m}U^{(m)}
$$

と置く。Frobenius内積に対する随伴はfactor転置による射影であり、

$$
\mathcal A^*(X)=X\underset{m\in\mathcal M}{\times_m}U^{(m)\mathsf T},
\qquad
\mathcal A^*\mathcal A=I
$$

である。$G=\mathcal A^*X$ とし、任意の候補coreを $C=G+D$ と書くと、

$$
\begin{aligned}
\mathcal A^*(X-\mathcal A(G))
&=\mathcal A^*X-\mathcal A^*\mathcal A(G)=G-G=0,\\
\langle X-\mathcal A(G),\mathcal A(D)\rangle_F
&=\langle\mathcal A^*(X-\mathcal A(G)),D\rangle_F=0,\\
\|X-\mathcal A(C)\|_F^2
&=\|X-\mathcal A(G)-\mathcal A(D)\|_F^2\\
&=\|X-\mathcal A(G)\|_F^2+\|\mathcal A(D)\|_F^2\\
&=\|X-\mathcal A(G)\|_F^2+\|D\|_F^2.
\end{aligned}
$$

したがって $D=0$ で最小になり、factor固定時には第2節のcoreが最適である。第3節で使った残差と再構成の直交性も、この随伴の計算から従う。

### Ky Fanの重みを展開して上界を達成する

定理名だけで済ませず、上位固有ベクトルを取る理由を計算する。以下では $d=I_n$、$R=R_n$ とし、$ZZ^{\mathsf T}$ の完全な正規直交固有基底を $Q=(q_1,\ldots,q_d)$ とする。固有値は

$$
ZZ^{\mathsf T}=Q\Lambda Q^{\mathsf T},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d),
\qquad
\lambda_i=\sigma_i^2,
\qquad
\lambda_1\ge\cdots\ge\lambda_d\ge0
$$

であり、reduced SVDで返らない方向があれば固有値0として補う。候補factorの座標を $H=Q^{\mathsf T}U$ と置くと、

$$
H^{\mathsf T}H=U^{\mathsf T}QQ^{\mathsf T}U=I_R,
\qquad
w_i:=\sum_{a=1}^{R}H_{i,a}^2
=q_i^{\mathsf T}UU^{\mathsf T}q_i
$$

である。射影の性質とtraceから、

$$
0\le w_i\le1,
\qquad
\sum_{i=1}^{d}w_i
=\sum_{a=1}^{R}\sum_{i=1}^{d}H_{i,a}^2
=R
$$

となり、目的関数は

$$
\begin{aligned}
\operatorname{tr}(U^{\mathsf T}ZZ^{\mathsf T}U)
&=\operatorname{tr}(H^{\mathsf T}\Lambda H)\\
&=\sum_{a=1}^{R}\sum_{i=1}^{d}\lambda_iH_{i,a}^2\\
&=\sum_{i=1}^{d}\lambda_iw_i.
\end{aligned}
$$

$1\le R<d$ なら、降順性から

$$
\begin{aligned}
\sum_{i=1}^{d}\lambda_iw_i
&\le\sum_{i=1}^{R}\lambda_iw_i+\lambda_R\sum_{i=R+1}^{d}w_i\\
&=R\lambda_R+\sum_{i=1}^{R}(\lambda_i-\lambda_R)w_i\\
&\le R\lambda_R+\sum_{i=1}^{R}(\lambda_i-\lambda_R)\\
&=\sum_{i=1}^{R}\lambda_i.
\end{aligned}
$$

$U=(q_1,\ldots,q_R)$ なら $w_1=\cdots=w_R=1$、残りが0となって上界を達成する。$R=d$ では全空間を取るので同じ結論である。これにより、既存のSVDによる更新式を、固有基底での成分和から証明できる。境界固有値が重複すると最適部分空間は一意とは限らない。

ここでSVDを

$$
Z
=
Q\Sigma V^{\mathsf T}
$$

とする。

すると、

$$
\begin{aligned}
ZZ^{\mathsf T}
&=
Q\Sigma V^{\mathsf T}
V\Sigma^{\mathsf T}Q^{\mathsf T}\\
&=
Q\Sigma\Sigma^{\mathsf T}Q^{\mathsf T}.
\end{aligned}
$$

$ZZ^{\mathsf T}$ の固有ベクトルは $Q$ の列、固有値は特異値の二乗

$$
\sigma_1^2\ge\sigma_2^2\ge\cdots\ge0
$$

である。

Rayleigh–Ritz / Ky Fanの最大化原理より、$R_n$ 次元の直交部分空間で

$$
\operatorname{tr}
(U^{\mathsf T}ZZ^{\mathsf T}U)
$$

を最大にするには、最大の $R_n$ 個の固有値に対応する固有ベクトルを列に取ればよい。

したがって、

$$
\boxed{
U^{(n)}
\leftarrow
Q_{[:,1:R_n]}
}
$$

という数学的な列番号表記になる。

Pythonの0始まりsliceで書けば、

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-004]]

である。

つまり、

$$
\boxed{
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\left(
\operatorname{left\ singular\ vectors}
\left(Z_{(n)}^{(n)}\right)
\right)
}
$$

となる。

```text
更新対象以外で射影
→ 対象modeでunfold
→ SVD
→ 上位左特異ベクトル
```

というHOOIの更新は、単なる経験則ではなく、**他factorを固定した局所問題を最適に解く更新**になっている。

ただしfactor全体を同時に見たTucker問題は非凸なので、各局所更新が最適でもグローバル最適解は保証されない。

---

## 6. Tucker-2での更新

Conv weight

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

でmode 0 / 1だけを圧縮する。

active mode集合は

$$
\mathcal M=\{0,1\}
$$

である。

### $U_{\mathrm{out}}$ の更新

現在の $U_{\mathrm{in}}$ を固定してmode 1を射影する。

$$
\boxed{
Z_{\mathrm{out}}
=
W
\times_1
U_{\mathrm{in}}^{\mathsf{T}}
}
$$

shapeは

$$
(C_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
\rightarrow
(C_{\mathrm{out}},R_{\mathrm{in}},K_h,K_w)
$$

である。

mode 0 unfoldingは

$$
(Z_{\mathrm{out}})_{(0)}
\in
\mathbb R^{
C_{\mathrm{out}}
\times
(R_{\mathrm{in}}K_hK_w)
}
$$

になる。その上位左特異ベクトルを使って

$$
\boxed{
U_{\mathrm{out}}
\leftarrow
\operatorname{top\text{-}}R_{\mathrm{out}}
\left(
\operatorname{left\ singular\ vectors}
\left[
(W\times_1U_{\mathrm{in}}^{\mathsf{T}})_{(0)}
\right]
\right)
}
$$

とする。

### $U_{\mathrm{in}}$ の更新

次に、**同じsweep内ですでに更新された新しい** $U_{\mathrm{out}}$ を使う。

$$
\boxed{
Z_{\mathrm{in}}
=
W
\times_0
U_{\mathrm{out}}^{\mathsf{T}}
}
$$

shapeは

$$
(C_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
\rightarrow
(R_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
$$

で、mode 1 unfoldingは

$$
(Z_{\mathrm{in}})_{(1)}
\in
\mathbb R^{
C_{\mathrm{in}}
\times
(R_{\mathrm{out}}K_hK_w)
}
$$

になる。したがって、

$$
\boxed{
U_{\mathrm{in}}
\leftarrow
\operatorname{top\text{-}}R_{\mathrm{in}}
\left(
\operatorname{left\ singular\ vectors}
\left[
(W\times_0U_{\mathrm{out}}^{\mathsf{T}})_{(1)}
\right]
\right)
}
$$

とする。

---

## 7. Tucker-2の1 sweepを $t\rightarrow t+1$ で書く

初期値をHOSVDで

$$
U_{\mathrm{out}}^{(0)},
\qquad
U_{\mathrm{in}}^{(0)}
$$

とする。

iteration $t$ で、まず

$$
Z_{\mathrm{out}}^{(t)}
=
W\times_1
(U_{\mathrm{in}}^{(t)})^{\mathsf T}
$$

を作り、

$$
U_{\mathrm{out}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{out}}
\operatorname{leftSVD}
\left[(Z_{\mathrm{out}}^{(t)})_{(0)}\right]
$$

と更新する。

次に**更新済み**factorを使って、

$$
Z_{\mathrm{in}}^{(t)}
=
W\times_0
(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
$$

を作り、

$$
U_{\mathrm{in}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{in}}
\operatorname{leftSVD}
\left[(Z_{\mathrm{in}}^{(t)})_{(1)}\right]
$$

とする。

この2更新で1 sweepである。

3階Tensorなら、今回の実装はGauss-Seidel型に

| 更新対象 | projectionに使うfactor | 更新後 |
| ---: | --- | --- |
| 0 | $U_1,U_2$ | 新しい $U_0$ |
| 1 | 新しい $U_0$, $U_2$ | 新しい $U_1$ |
| 2 | 新しい $U_0$, 新しい $U_1$ | 新しい $U_2$ |

と進む。

同じsweep内で最新factorを使う点が、自作 `hooi_sweep()` の重要な仕様である。

---

## 8. core・再構成・errorを $t+1$ まで追う

### 一つのsweepで誤差が増えないことの途中式

固定rankと列直交factorの下で、coreをその都度最適な射影で計算するとする。
保持coreの二乗normを

$$
F(U_{\mathrm{out}},U_{\mathrm{in}})
:=\|W\times_0U_{\mathrm{out}}^{\mathsf T}
\times_1U_{\mathrm{in}}^{\mathsf T}\|_F^2
$$

と置く。第5節の局所最大化により、出力factor更新では

$$
F(U_{\mathrm{out}}^{(t+1)},U_{\mathrm{in}}^{(t)})
\ge F(U_{\mathrm{out}}^{(t)},U_{\mathrm{in}}^{(t)}).
$$

更新済みの出力factorを固定した入力factor更新では

$$
F(U_{\mathrm{out}}^{(t+1)},U_{\mathrm{in}}^{(t+1)})
\ge F(U_{\mathrm{out}}^{(t+1)},U_{\mathrm{in}}^{(t)}).
$$

第3節の誤差式へ代入すると、$W\ne0$ に対して

$$
\begin{aligned}
e_{t+1}^2
&=1-\frac{F(U_{\mathrm{out}}^{(t+1)},U_{\mathrm{in}}^{(t+1)})}{\|W\|_F^2}\\
&\le1-\frac{F(U_{\mathrm{out}}^{(t+1)},U_{\mathrm{in}}^{(t)})}{\|W\|_F^2}\\
&\le1-\frac{F(U_{\mathrm{out}}^{(t)},U_{\mathrm{in}}^{(t)})}{\|W\|_F^2}\\
&=e_t^2.
\end{aligned}
$$

理想的な厳密SVD更新では $e_t\ge0$ の単調非増加列なので誤差値は収束する。
これはfactorの一意な収束や大域最適性の証明ではない。
丸め・近似SVD・異なる停止条件を含む実装では、微小な増加も別途確認する。

factorを1 sweep更新した後、

$$
\boxed{
G^{(t+1)}
=
W
\times_0
(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
\times_1
(U_{\mathrm{in}}^{(t+1)})^{\mathsf T}
}
$$

を計算する。

再構成は

$$
\boxed{
\hat W^{(t+1)}
=
G^{(t+1)}
\times_0U_{\mathrm{out}}^{(t+1)}
\times_1U_{\mathrm{in}}^{(t+1)}
}
$$

である。

relative Frobenius errorは

$$
\boxed{
e_{t+1}
=
\frac{
\lVert W-\hat W^{(t+1)}\rVert_F
}{
\lVert W\rVert_F
}
}
$$

となる。

### 小さいTensorでHOSVD初期値から1 sweepを最後まで追う

更新の効果が要素で見えるように、shape $(2,2,2)$、圧縮modeを0と1、指定rankを $(1,1)$ とする。Tensorを

$$
X_{:,:,0}
=
\begin{pmatrix}
0&10\\
9&0
\end{pmatrix},
\qquad
X_{:,:,1}
=
\begin{pmatrix}
8&0\\
0&0
\end{pmatrix}
$$

と置く。

HOSVD初期化ではmodeごとに元の $X$ を独立にunfoldする。mode 0の2行の二乗normは

$$
10^2+8^2=164,
\qquad
9^2=81
$$

であり、mode 1の2行、すなわち元Tensorの第2添字に関する二乗normは

$$
9^2+8^2=145,
\qquad
10^2=100
$$

である。各unfoldingの行は互いに直交するため、rank 1のHOSVD初期値は

この例のunfoldingも省略せず書くと、残る添字は後ろの添字が速く変わる順に並べて

$$
X_{(0)}=\begin{pmatrix}0&8&10&0\\9&0&0&0\end{pmatrix},
\qquad
X_{(1)}=\begin{pmatrix}0&8&9&0\\10&0&0&0\end{pmatrix}.
$$

各行の内積から

$$
\begin{aligned}
X_{(0)}X_{(0)}^{\mathsf T}
&=\begin{pmatrix}
0^2+8^2+10^2+0^2&0\cdot9+8\cdot0+10\cdot0+0\cdot0\\
9\cdot0+0\cdot8+0\cdot10+0\cdot0&9^2+0^2+0^2+0^2
\end{pmatrix}
=\begin{pmatrix}164&0\\0&81\end{pmatrix},\\
X_{(1)}X_{(1)}^{\mathsf T}
&=\begin{pmatrix}
0^2+8^2+9^2+0^2&0\cdot10+8\cdot0+9\cdot0+0\cdot0\\
10\cdot0+0\cdot8+0\cdot9+0\cdot0&10^2+0^2+0^2+0^2
\end{pmatrix}
=\begin{pmatrix}145&0\\0&100\end{pmatrix}.
\end{aligned}
$$

対角Gram行列の最大固有値はそれぞれ164と145で、どちらも固有ベクトルは $(1,0)^{\mathsf T}$ である。従って、下の初期factorになる。

$$
U_0^{(0)}
=
\begin{pmatrix}
1\\
0
\end{pmatrix},
\qquad
U_1^{(0)}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

このときcoreは、両modeで第0成分を選ぶので

$$
G^{(0)}_{0,0,:}
=
\begin{pmatrix}
X_{0,0,0}&X_{0,0,1}
\end{pmatrix}
=
\begin{pmatrix}
0&8
\end{pmatrix}.
$$

再構成に残るのは $\hat X^{(0)}_{0,0,1}=8$ だけである。したがって

$$
\begin{aligned}
\lVert X-\hat X^{(0)}\rVert_F
&=
\sqrt{10^2+9^2}\\
&=
\sqrt{181}\\
&\approx13.4536,
\end{aligned}
$$

また $\lVert X\rVert_F=\sqrt{10^2+9^2+8^2}=\sqrt{245}$ より、relative errorは

$$
e_0
=
\frac{\sqrt{181}}{\sqrt{245}}
\approx0.8595
$$

である。

次にHOOIの1 sweepをGauss-Seidel順で実行する。まずmode 0を更新するため、mode 1を $U_1^{(0)}=(1,0)^{\mathsf T}$ へ射影する。この射影後Tensorをmode 0でunfoldすると

$$
Z_{0,(0)}
=
\begin{pmatrix}
0&8\\
9&0
\end{pmatrix}.
$$

2行の二乗normは $8^2=64$ と $9^2=81$ なので、上位左特異ベクトルは

更新時もGram行列を計算すると

$$
\begin{aligned}
Z_{0,(0)}Z_{0,(0)}^{\mathsf T}
&=\begin{pmatrix}
0^2+8^2&0\cdot9+8\cdot0\\
9\cdot0+0\cdot8&9^2+0^2
\end{pmatrix}
=\begin{pmatrix}64&0\\0&81\end{pmatrix},\\
\det(Z_{0,(0)}Z_{0,(0)}^{\mathsf T}-\lambda I_2)
&=(64-\lambda)(81-\lambda).
\end{aligned}
$$

最大固有値81の方向が第2行に移るため、次のfactorになる。

$$
U_0^{(1)}
=
\begin{pmatrix}
0\\
1
\end{pmatrix}.
$$

続いてmode 1を更新するときは、このsweep内ですでに更新した $U_0^{(1)}$ を使う。mode 0を $(0,1)^{\mathsf T}$ へ射影した後のmode 1 unfoldingは

$$
Z_{1,(1)}
=
\begin{pmatrix}
9&0\\
0&0
\end{pmatrix},
$$

したがって

この第2更新のGram行列も

$$
Z_{1,(1)}Z_{1,(1)}^{\mathsf T}
=\begin{pmatrix}9^2+0^2&9\cdot0+0\cdot0\\0\cdot9+0\cdot0&0^2+0^2\end{pmatrix}
=\begin{pmatrix}81&0\\0&0\end{pmatrix}
$$

となり、最大固有値81の固有ベクトルは $(1,0)^{\mathsf T}$ である。

$$
U_1^{(1)}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

1 sweep後のcoreは

$$
G^{(1)}_{0,0,:}
=
\begin{pmatrix}
X_{1,0,0}&X_{1,0,1}
\end{pmatrix}
=
\begin{pmatrix}
9&0
\end{pmatrix}.
$$

再構成に残るのは $\hat X^{(1)}_{1,0,0}=9$ だけなので、

$$
\begin{aligned}
\lVert X-\hat X^{(1)}\rVert_F
&=
\sqrt{10^2+8^2}\\
&=
\sqrt{164}\\
&\approx12.8062,\\
e_1
&=
\frac{\sqrt{164}}{\sqrt{245}}
\approx0.8182.
\end{aligned}
$$

この1 sweepでは $e_1<e_0$ となる。HOSVDは各modeを元Tensorから独立に初期化する一方、HOOIは他modeの最新factorへ射影してから更新する、という違いが具体的な要素選択として現れている。

---

## 9. 収束判定

単純な絶対差なら、

$$
|e_t-e_{t-1}|<\varepsilon
$$

で止められる。

相対改善率を明示する別案なら、

$$
\frac{|e_t-e_{t-1}|}
{\max(|e_{t-1}|,\epsilon)}
<\mathrm{tol}
$$

と書ける。

現在のsrc実装では、絶対許容と相対許容を合わせて

$$
\boxed{
|e_t-e_{t-1}|
\leq
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}|e_{t-1}|
}
$$

を使う。

この形なら、誤差の絶対スケールが変わっても相対的な変化を考慮できる。

また最大反復回数 `max_iter` を置いて無限反復を防ぐ。

現在の既定値は、

```text
max_iter = 30
abs_tol = 1e-8
rel_tol = 1e-5
```

である。

---

## 10. HOOIが同rankでも意味を持つ理由

HOOIはrankを変更しない。

したがってHOSVDとHOOIで、

- core shape
- factor shape
- 圧縮後の層構造
- parameters
- MACs

は同じ。

違うのはfactor/coreの**値**であり、直接改善する対象は

$$
\lVert W-\hat W\rVert_F
$$

である。

実験では、同rankでHOOIがweight近似誤差を改善しても、圧縮直後のvalidation accuracyが同じ順序で改善するとは限らない。ここでは一般的な解釈だけを扱い、CIFAR-10 `conv2` の具体値は [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]] を正本とする。

---

## 11. なぜweight誤差とaccuracyが一致しないか

HOOIの直接目的は

$$
\min\lVert W-\hat W\rVert_F^2
$$

であり、分類モデルの目的

$$
\min\mathcal{L}_{\mathrm{CE}}
$$

ではない。

weight空間では全要素の誤差を同じ二乗尺度で評価するが、実データがよく使う入力方向や分類境界に重要な方向は一様ではない。

したがって、

```text
weight reconstruction quality
≠
layer output quality
≠
logit quality
≠
classification accuracy
```

と考える必要がある。

task-awareな発展としては、例えば次の目的がある。

| 目的 | 代表的な誤差 |
| --- | --- |
| 元weightを再現 | Weight Frobenius |
| 中間表現を再現 | Feature-map MSE |
| baseline出力を再現 | Logit MSE / KD |
| 正解ラベルへ最適化 | Cross Entropy |

ただし、標準HOOIのSVDによる閉形式更新はFrobenius二乗誤差の構造を利用している。Cross Entropy等へ目的を変更する場合は、単にHOOIの誤差関数を差し替えるのではなく、HOSVD/HOOIを初期化としてTucker-2層を作り、通常のbackpropagationでfine-tuningするのが今回の実験に対応する。

fine-tuningで補助損失を使うなら、例えば

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda_{\mathrm{feature}}\mathcal{L}_{\mathrm{feature}}
+
\lambda_{\mathrm{logit}}\mathcal{L}_{\mathrm{logit}}
$$

のような発展が考えられる。ただし今回の実験では標準Cross Entropyによるfine-tuningを主に確認しており、この補助損失自体を実験したわけではない。

---

## 12. HOSVDとHOOIの使い分け

```text
HOSVD
→ 高速な標準分解
→ rank探索
→ HOOIの初期解

HOOI
→ 選んだrank・重要layerの精密化
→ 同rankでweight再構成誤差を改善
```

大量のrank候補を探索する段階ではHOSVDが合理的。HOOIは、候補を絞った後のオフライン精密化として使いやすい。

---

## 13. srcとの対応

```text
src/nn_compression/compression/hooi.py
├─ has_converged
├─ hooi_sweep
├─ core_from_factors
└─ hooi
```

`ranks` のkeyが「圧縮・更新対象mode」を表すため、

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-005]]

なら通常の3-mode HOOI、

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-006]]

なら4階Conv weightに対するpartial HOOI / Tucker-2として同じ実装を再利用できる。

より長い途中導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]] にも残す。

PyTorchでの汎用HOOIとConv2d Tucker-2の実装は [[06_Tucker基礎実装検証/26_Tucker_HOOIのPyTorch実装]] を参照する。

検証結果：

- [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
- [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
