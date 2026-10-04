---
title: 第2Linearによるlogits誤差伝播と合成上界
tags:
  - TTLinear
  - logits
  - error-propagation
  - Frobenius-norm
  - spectral-norm
---

# 第2Linearによるlogits誤差伝播と合成上界

## 1. このノートの目的

第1 Linearの重み近似誤差が、ReLU後のhidden activationを経由し、第2 Linearのlogitsへどう伝わるかを導く。

既習事項は次である。

$$
\Delta Z_1
=
X\Delta W_1^{\mathsf T},
$$

$$
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2,
$$

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F.
$$

このノートでは、さらに

$$
\Delta L
=
\Delta H W_2^{\mathsf T}
$$

を導き、

$$
\boxed{
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
}
$$

までつなぐ。

## 2. 変数とshape

batch sizeを $B$、入力次元を $d$、hidden次元を $h$、class数を $c$ とする。

$$
X\in\mathbb{R}^{B\times d},
$$

$$
W_1,\widetilde W_1,\Delta W_1
\in
\mathbb{R}^{h\times d},
$$

$$
Z_1,\widetilde Z_1,\Delta Z_1,
H,\widetilde H,\Delta H
\in
\mathbb{R}^{B\times h},
$$

$$
W_2\in\mathbb{R}^{c\times h},
\qquad
b_2\in\mathbb{R}^{c},
$$

$$
L,\widetilde L,\Delta L
\in
\mathbb{R}^{B\times c}.
$$

各軸の意味は次である。

- $X$ の第1軸：batch sample。
- $X$ の第2軸：入力特徴。
- $H$ の第1軸：batch sample。
- $H$ の第2軸：hidden unit。
- $L$ の第1軸：batch sample。
- $L$ の第2軸：outputまたはclass。
- $W_2$ の第1軸：outputまたはclass。
- $W_2$ の第2軸：hidden unit。

誤差の符号規約は、常に

$$
\Delta W_1
=
\widetilde W_1-W_1,
\qquad
\Delta H
=
\widetilde H-H,
\qquad
\Delta L
=
\widetilde L-L
$$

とする。

## 3. 第2 Linearの厳密な恒等式

dense側と近似側で、第2 Linearの同じ $W_2,b_2$ を用いる。

$$
L
=
HW_2^{\mathsf T}+b_2,
$$

$$
\widetilde L
=
\widetilde H W_2^{\mathsf T}+b_2.
$$

PyTorchのbatch表現でbiasを各sampleへbroadcastすることを明示するなら、$\boldsymbol{1}_B\in\mathbb{R}^{B}$ を使って

$$
L
=
HW_2^{\mathsf T}
+
\boldsymbol{1}_B b_2^{\mathsf T},
$$

$$
\widetilde L
=
\widetilde H W_2^{\mathsf T}
+
\boldsymbol{1}_B b_2^{\mathsf T}
$$

と書ける。差を取ると、

$$
\begin{aligned}
\Delta L
&=
\widetilde L-L\\
&=
\left(
\widetilde H W_2^{\mathsf T}
+
\boldsymbol{1}_B b_2^{\mathsf T}
\right)
-
\left(
H W_2^{\mathsf T}
+
\boldsymbol{1}_B b_2^{\mathsf T}
\right)\\
&=
\widetilde H W_2^{\mathsf T}
-
H W_2^{\mathsf T}\\
&=
\left(
\widetilde H-H
\right)
W_2^{\mathsf T}\\
&=
\Delta H W_2^{\mathsf T}.
\end{aligned}
$$

従って、

$$
\boxed{
\Delta L
=
\Delta H W_2^{\mathsf T}
}
$$

である。これは上界ではなく、同じ第2 Linearを用いる限り成り立つ厳密な代数恒等式である。同じbiasは両方のlogitsを同じだけ平行移動するため、差では打ち消し合う。

shapeも

$$
(B\times h)(h\times c)
=
B\times c
$$

となり、$\Delta L$ のshapeと一致する。

## 4. 成分ごとの導出

sample indexを $n=1,\ldots,B$、hidden unitを $j=1,\ldots,h$、classを $k=1,\ldots,c$ とする。

$$
L_{n,k}
=
\sum_{j=1}^{h}
H_{n,j}(W_2)_{k,j}
+
(b_2)_k,
$$

$$
\widetilde L_{n,k}
=
\sum_{j=1}^{h}
\widetilde H_{n,j}(W_2)_{k,j}
+
(b_2)_k.
$$

差を取ると、

$$
\begin{aligned}
(\Delta L)_{n,k}
&=
\widetilde L_{n,k}-L_{n,k}\\
&=
\sum_{j=1}^{h}
\left(
\widetilde H_{n,j}-H_{n,j}
\right)
(W_2)_{k,j}\\
&=
\sum_{j=1}^{h}
(\Delta H)_{n,j}(W_2)_{k,j}.
\end{aligned}
$$

一方、行列積の $(n,k)$ 要素は、

$$
\begin{aligned}
\left(
\Delta H W_2^{\mathsf T}
\right)_{n,k}
&=
\sum_{j=1}^{h}
(\Delta H)_{n,j}
(W_2^{\mathsf T})_{j,k}\\
&=
\sum_{j=1}^{h}
(\Delta H)_{n,j}(W_2)_{k,j}.
\end{aligned}
$$

従って、全ての $n,k$ について

$$
(\Delta L)_{n,k}
=
\left(
\Delta H W_2^{\mathsf T}
\right)_{n,k}
$$

が成り立つ。

## 5. 全要素を表示した小さい例

$$
\Delta H
=
\begin{pmatrix}
1&-2\\
\frac12&1
\end{pmatrix},
\qquad
W_2
=
\begin{pmatrix}
2&1\\
-1&3
\end{pmatrix}
$$

とする。このとき、

$$
W_2^{\mathsf T}
=
\begin{pmatrix}
2&-1\\
1&3
\end{pmatrix}.
$$

行列積を全要素で計算すると、

$$
\begin{aligned}
\Delta L
&=
\Delta H W_2^{\mathsf T}\\
&=
\begin{pmatrix}
1&-2\\
\frac12&1
\end{pmatrix}
\begin{pmatrix}
2&-1\\
1&3
\end{pmatrix}\\
&=
\begin{pmatrix}
1\cdot2+(-2)\cdot1
&
1\cdot(-1)+(-2)\cdot3\\
\frac12\cdot2+1\cdot1
&
\frac12\cdot(-1)+1\cdot3
\end{pmatrix}\\
&=
\begin{pmatrix}
0&-7\\
2&\frac52
\end{pmatrix}.
\end{aligned}
$$

第1行は第1 sampleのhidden errorが2つのclass logitsへ伝わった結果、第2行は第2 sampleの結果である。

## 6. 第2 LinearのFrobenius誤差上界

$\Delta H$ の第 $n$ 行を行ベクトル $\delta h_n^{\mathsf T}$、$\Delta L$ の第 $n$ 行を $\delta l_n^{\mathsf T}$ と書く。

$$
\Delta H
=
\begin{pmatrix}
\delta h_1^{\mathsf T}\\
\vdots\\
\delta h_B^{\mathsf T}
\end{pmatrix},
\qquad
\Delta L
=
\begin{pmatrix}
\delta l_1^{\mathsf T}\\
\vdots\\
\delta l_B^{\mathsf T}
\end{pmatrix}.
$$

恒等式 $\Delta L=\Delta H W_2^{\mathsf T}$ の第 $n$ 行から、

$$
\delta l_n^{\mathsf T}
=
\delta h_n^{\mathsf T}W_2^{\mathsf T}
$$

である。転置すると、

$$
\delta l_n
=
W_2\delta h_n.
$$

spectral normの定義から、

$$
\|\delta l_n\|_2
=
\|W_2\delta h_n\|_2
\le
\|W_2\|_2
\|\delta h_n\|_2.
$$

各sampleについて二乗し、総和を取る。

$$
\begin{aligned}
\|\Delta L\|_F^2
&=
\sum_{n=1}^{B}
\|\delta l_n\|_2^2\\
&\le
\sum_{n=1}^{B}
\|W_2\|_2^2
\|\delta h_n\|_2^2\\
&=
\|W_2\|_2^2
\sum_{n=1}^{B}
\|\delta h_n\|_2^2\\
&=
\|W_2\|_2^2
\|\Delta H\|_F^2.
\end{aligned}
$$

両辺は非負なので平方根を取ると、

$$
\boxed{
\|\Delta L\|_F
\le
\|\Delta H\|_F
\|W_2\|_2
}
$$

を得る。これは一般の行列不等式

$$
\|AB\|_F
\le
\|A\|_F\|B\|_2
$$

へ $A=\Delta H$、$B=W_2^{\mathsf T}$ を代入した結果でもある。転置しても特異値は変わらないため、

$$
\|W_2^{\mathsf T}\|_2
=
\|W_2\|_2.
$$

spectral normが最大特異値になる途中式、長方形行列での総和範囲、$\sup$ と $\max$ の違いは、[[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]を参照する。

## 7. 第1 Linearからlogitsまでの合成上界

第1 Linearでは、

$$
\Delta Z_1
=
X\Delta W_1^{\mathsf T}
$$

であり、

$$
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2.
$$

ReLUは1-Lipschitzなので、

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F.
$$

第2 Linearでは、

$$
\|\Delta L\|_F
\le
\|\Delta H\|_F\|W_2\|_2.
$$

まず、$\|W_2\|_2\ge0$ なので、

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F
$$

の両辺へ $\|W_2\|_2$ を掛けても不等号の向きは変わらない。

$$
\|\Delta H\|_F\|W_2\|_2
\le
\|\Delta Z_1\|_F\|W_2\|_2.
$$

さらに、

$$
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2
$$

を代入する。従って、

$$
\begin{aligned}
\|\Delta L\|_F
&\le
\|\Delta H\|_F\|W_2\|_2\\
&\le
\|\Delta Z_1\|_F\|W_2\|_2\\
&\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2.
\end{aligned}
$$

よって、

$$
\boxed{
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
}
$$

である。

## 8. 3つの係数の意味

合成上界

$$
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

には、異なる役割を持つ3つの係数が現れる。

### 8.1 $\|X\|_F$

現在のbatchに含まれる入力の総量を表す。入力のscaleが大きいほど、同じ重み誤差でもpre-activation誤差は大きくなり得る。

### 8.2 $\|\Delta W_1\|_2$

第1 Linearの近似誤差が、入力ベクトルを最悪の場合に何倍へ増幅するかを表す。TT-rankを変えると、この量も変化する。

### 8.3 $\|W_2\|_2$

hidden activation誤差に対する第2 Linearの感度を表す。

- $\|W_2\|_2<1$ なら、上界上はhidden errorを縮小する。
- $\|W_2\|_2=1$ なら、上界上は同じ大きさまでである。
- $\|W_2\|_2>1$ なら、方向によってはhidden errorを増幅し得る。

ただし、特定の $\Delta H$ が実際に $\|W_2\|_2$ 倍されるとは限らない。等号へ近づくには、各sampleのhidden errorが最大右特異ベクトル方向へ強く整列する必要がある。

## 9. 上界がlooseになりやすい理由

合成上界はworst-case boundである。等号へ近づくには、少なくとも次が同時に強く成立する必要がある。

1. 入力batch $X$ が $\Delta W_1$ の最大増幅方向と整合する。
2. ReLUがpre-activation誤差を減衰させない。
3. hidden errorが $W_2$ の最大右特異ベクトル方向と整合する。

実データと学習済みモデルでこれらが全て揃うとは限らない。そのため、実測値

$$
\|\Delta L\|_F
$$

は上界よりかなり小さくなる場合がある。

不等式の緩さを段階別に見るには、分母が非零のとき、

$$
\rho_1
=
\frac{
\|\Delta Z_1\|_F
}{
\|X\|_F\|\Delta W_1\|_2
},
$$

$$
\rho_{\mathrm{ReLU}}
=
\frac{
\|\Delta H\|_F
}{
\|\Delta Z_1\|_F
},
$$

$$
\rho_2
=
\frac{
\|\Delta L\|_F
}{
\|\Delta H\|_F\|W_2\|_2
}
$$

を記録する。すると、

$$
\frac{
\|\Delta L\|_F
}{
\|X\|_F\|\Delta W_1\|_2\|W_2\|_2
}
=
\rho_1
\rho_{\mathrm{ReLU}}
\rho_2.
$$

全体の乖離を、どの段階で生じたかへ分解できる。

## 10. 圧縮設計での使い方

### 10.1 TT-rank候補の比較

rank候補 $r$ ごとに近似重み $\widetilde W_1(r)$ を作り、

$$
\Delta W_1(r)
=
\widetilde W_1(r)-W_1
$$

を実測する。各rankのworst-case指標は、

$$
U(r)
=
\|X\|_F
\|\Delta W_1(r)\|_2
\|W_2\|_2
$$

で比較できる。

ただし、人工的に

$$
\Delta W_1(\alpha)=\alpha E
$$

と置いてscaleだけを変える実験は、誤差伝播の測定系を検証するtoy実験である。TT-rankを変えて得た圧縮結果そのものではない。実TT圧縮では、先に $\widetilde W_1(r)$ を作り、その後に $\Delta W_1(r)$ を測る。

### 10.2 圧縮対象層の比較

同程度の重み近似誤差でも、後段に大きなspectral normを持つ層が続けば、outputまでの上界は大きくなる。深いネットワークでは、1-Lipschitz activationを挟む限り、後段の重み行列のspectral normが連鎖して現れる。

従って、圧縮する層はparameter数だけでなく、

$$
\text{その層の近似誤差}
\times
\text{後段の感度}
$$

も含めて評価する。

### 10.3 実測値との併記

上界だけでなく、実際のbatchでの

$$
\|\Delta L\|_F
$$

も記録する。これにより、

$$
\text{数学上のworst case}
\quad\text{と}\quad
\text{データ分布上の実挙動}
$$

を分けて評価できる。

## 11. この上界から言えないこと

$\|\Delta L\|_F$ が小さいことは、logits全体が元モデルから大きく逸脱しにくいことを示す。一方、次を単独では保証しない。

- 全sampleでpredictionが一致すること。
- test accuracyが維持されること。
- rank候補がtask性能の意味で最適であること。
- fine-tuning後の性能。

argmaxが変わらない十分条件へ進むには、sampleごとのlogit誤差とclassification marginを比較する必要がある。導出、十分条件と必要条件の区別、誤差方向の直接判定は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]で扱う。

PyTorchによる恒等式・norm API・assertの組み方は、[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]を参照する。小さい2層MLPによる実験設定、人工摂動scale sweepの保存済み結果と考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]へ分ける。
