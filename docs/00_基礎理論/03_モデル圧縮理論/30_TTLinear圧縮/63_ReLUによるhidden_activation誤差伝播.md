---
title: ReLUによるhidden activation誤差伝播
tags:
  - TTLinear
  - ReLU
  - hidden-activation
  - Frobenius-norm
  - error-propagation
---

# ReLUによるhidden activation誤差伝播

このノートでは、TT-rank打ち切りなどでLinear重みが変化したとき、Linear直後の誤差がReLU後のhidden activationへどう伝わるかを、一般式と全要素を表示した具体行列の両方で確認する。

ReLUが1-Lipschitzであること自体は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]を前提とする。Linear出力誤差の2種類のnorm上界は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]を参照する。

## 1. dense重みと近似重み

batch sizeを $B$、入力次元を $n$、hidden次元を $h$ とする。

$$
X\in\mathbb{R}^{B\times n},
\qquad
W,\widetilde W\in\mathbb{R}^{h\times n},
\qquad
b\in\mathbb{R}^{h}.
$$

dense重み $W$ と近似重み $\widetilde W$ によるpre-activationを

$$
Z=XW^{\mathsf T}+b,
\qquad
\widetilde Z=X\widetilde W^{\mathsf T}+b
$$

とする。PyTorchのbatch表現では、bias $b$ は各行へbroadcastされる。したがって、

$$
Z,\widetilde Z\in\mathbb{R}^{B\times h}
$$

である。

重み誤差とpre-activation誤差を

$$
\Delta W=\widetilde W-W,
\qquad
\Delta Z=\widetilde Z-Z
$$

と定義する。同じbiasを使っているため、bias項は差を取ると消える。

$$
\begin{aligned}
\Delta Z
&=\widetilde Z-Z\\
&=\left(X\widetilde W^{\mathsf T}+b\right)
-\left(XW^{\mathsf T}+b\right)\\
&=X\widetilde W^{\mathsf T}-XW^{\mathsf T}\\
&=X\left(\widetilde W-W\right)^{\mathsf T}\\
&=X\Delta W^{\mathsf T}.
\end{aligned}
$$

## 2. ReLU後のhidden activation

ReLUを要素ごとに適用して、

$$
H=\operatorname{ReLU}(Z),
\qquad
\widetilde H=\operatorname{ReLU}(\widetilde Z)
$$

とする。hidden activation誤差は

$$
\Delta H=\widetilde H-H
$$

である。shapeはすべて

$$
Z,\widetilde Z,\Delta Z,H,\widetilde H,\Delta H
\in
\mathbb{R}^{B\times h}
$$

となる。

ここで第1軸 $i=0,\ldots,B-1$ はsample、すなわちbatch内の行を表す。第2軸 $j=0,\ldots,h-1$ はhidden unitを表す。

## 3. 要素ごとの誤差不等式

スカラーReLUの1-Lipschitz性を、各要素 $(i,j)$ へ適用する。

$$
\begin{aligned}
|\Delta H_{ij}|
&=
|\widetilde H_{ij}-H_{ij}|\\
&=
|\operatorname{ReLU}(\widetilde Z_{ij})
-\operatorname{ReLU}(Z_{ij})|\\
&\le
|\widetilde Z_{ij}-Z_{ij}|\\
&=
|\Delta Z_{ij}|.
\end{aligned}
$$

したがって、すべてのsampleとhidden unitについて

$$
\boxed{
|\Delta H_{ij}|
\le
|\Delta Z_{ij}|
}
$$

が成り立つ。

これは要素ごとの主張であり、この段階ではまだFrobeniusノルムの不等式ではない。

## 4. 要素不等式からFrobeniusノルムへ

絶対値は非負なので、要素ごとの不等式を二乗しても大小関係は変わらない。

$$
|\Delta H_{ij}|^2
\le
|\Delta Z_{ij}|^2.
$$

全sampleと全hidden unitについて総和を取る。

$$
\sum_{i=0}^{B-1}
\sum_{j=0}^{h-1}
|\Delta H_{ij}|^2
\le
\sum_{i=0}^{B-1}
\sum_{j=0}^{h-1}
|\Delta Z_{ij}|^2.
$$

Frobeniusノルムの定義

$$
\|A\|_F^2
=
\sum_i\sum_j|A_{ij}|^2
$$

を使えば、

$$
\|\Delta H\|_F^2
\le
\|\Delta Z\|_F^2
$$

となる。両辺は非負であり、平方根は非負実数上で単調増加なので、

$$
\sqrt{\|\Delta H\|_F^2}
\le
\sqrt{\|\Delta Z\|_F^2}.
$$

よって、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z\|_F
}
$$

を得る。

## 5. Linear出力誤差上界との結合

$\Delta Z=X\Delta W^{\mathsf T}$ に対して、行列積のnorm不等式から

$$
\begin{aligned}
\|\Delta Z\|_F
&=
\|X\Delta W^{\mathsf T}\|_F\\
&\le
\|X\|_F
\|\Delta W^{\mathsf T}\|_2\\
&=
\|X\|_F
\|\Delta W\|_2
\end{aligned}
$$

が成り立つ。したがって、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z\|_F
\le
\|X\|_F\|\Delta W\|_2
}
$$

である。

別の組合せでは、

$$
\begin{aligned}
\|\Delta Z\|_F
&=
\|X\Delta W^{\mathsf T}\|_F\\
&\le
\|X\|_2
\|\Delta W^{\mathsf T}\|_F\\
&=
\|X\|_2
\|\Delta W\|_F
\end{aligned}
$$

なので、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z\|_F
\le
\|X\|_2\|\Delta W\|_F
}
$$

も得られる。

ここで $\|\Delta W\|_2$ は行列のspectral norm、すなわち最大特異値である。添字2が付いていても、全要素を並べたvector 2-normではない。

## 6. 全要素を表示した $2\times4$ 具体例

batch size $B=2$、hidden次元 $h=4$ とし、

$$
Z=
\begin{pmatrix}
2&-2&3&-4\\
1&-5&-1&4
\end{pmatrix},
\qquad
\widetilde Z=
\begin{pmatrix}
5&-7&-2&2\\
3&-1&2&1
\end{pmatrix}
$$

とする。差は

$$
\begin{aligned}
\Delta Z
&=\widetilde Z-Z\\
&=
\begin{pmatrix}
5-2&-7-(-2)&-2-3&2-(-4)\\
3-1&-1-(-5)&2-(-1)&1-4
\end{pmatrix}\\
&=
\begin{pmatrix}
3&-5&-5&6\\
2&4&3&-3
\end{pmatrix}.
\end{aligned}
$$

ReLUを要素ごとに作用させると、

$$
H=\operatorname{ReLU}(Z)
=
\begin{pmatrix}
2&0&3&0\\
1&0&0&4
\end{pmatrix},
$$

$$
\widetilde H=\operatorname{ReLU}(\widetilde Z)
=
\begin{pmatrix}
5&0&0&2\\
3&0&2&1
\end{pmatrix}.
$$

したがって、

$$
\begin{aligned}
\Delta H
&=\widetilde H-H\\
&=
\begin{pmatrix}
5-2&0-0&0-3&2-0\\
3-1&0-0&2-0&1-4
\end{pmatrix}\\
&=
\begin{pmatrix}
3&0&-3&2\\
2&0&2&-3
\end{pmatrix}.
\end{aligned}
$$

絶対値を全要素について取ると、

$$
|\Delta Z|
=
\begin{pmatrix}
3&5&5&6\\
2&4&3&3
\end{pmatrix},
\qquad
|\Delta H|
=
\begin{pmatrix}
3&0&3&2\\
2&0&2&3
\end{pmatrix}.
$$

全8要素の比較は、

$$
\begin{array}{c|c|c|c}
(i,j)&|\Delta H_{ij}|&|\Delta Z_{ij}|&\text{判定}\\
\hline
(0,0)&3&3&3\le3\\
(0,1)&0&5&0\le5\\
(0,2)&3&5&3\le5\\
(0,3)&2&6&2\le6\\
(1,0)&2&2&2\le2\\
(1,1)&0&4&0\le4\\
(1,2)&2&3&2\le3\\
(1,3)&3&3&3\le3
\end{array}
$$

となる。

## 7. 4種類の符号ケースを行列内で確認する

同じ具体例の中に、スカラー証明の4場合がすべて含まれている。

### 7.1 両方が非負：$(i,j)=(0,0)$

$$
Z_{00}=2,
\qquad
\widetilde Z_{00}=5.
$$

ReLU前後で差は変わらない。

$$
|\Delta H_{00}|=3=|\Delta Z_{00}|.
$$

### 7.2 両方が負：$(i,j)=(0,1)$

$$
Z_{01}=-2,
\qquad
\widetilde Z_{01}=-7.
$$

両方ともReLUで0になる。

$$
|\Delta H_{01}|=0<5=|\Delta Z_{01}|.
$$

### 7.3 dense側が非負、近似側が負：$(i,j)=(0,2)$

$$
Z_{02}=3,
\qquad
\widetilde Z_{02}=-2.
$$

$$
|\Delta H_{02}|=|0-3|=3<5=|-2-3|=|\Delta Z_{02}|.
$$

### 7.4 dense側が負、近似側が非負：$(i,j)=(0,3)$

$$
Z_{03}=-4,
\qquad
\widetilde Z_{03}=2.
$$

$$
|\Delta H_{03}|=|2-0|=2<6=|2-(-4)|=|\Delta Z_{03}|.
$$

## 8. 具体例のFrobeniusノルム

pre-activation誤差について、

$$
\begin{aligned}
\|\Delta Z\|_F^2
&=3^2+(-5)^2+(-5)^2+6^2\\
&\quad+2^2+4^2+3^2+(-3)^2\\
&=9+25+25+36+4+16+9+9\\
&=133.
\end{aligned}
$$

したがって、

$$
\|\Delta Z\|_F
=
\sqrt{133}
\approx
11.5326.
$$

hidden activation誤差について、

$$
\begin{aligned}
\|\Delta H\|_F^2
&=3^2+0^2+(-3)^2+2^2\\
&\quad+2^2+0^2+2^2+(-3)^2\\
&=9+0+9+4+4+0+4+9\\
&=39.
\end{aligned}
$$

したがって、

$$
\|\Delta H\|_F
=
\sqrt{39}
\approx
6.2450.
$$

この例では、

$$
\boxed{
\|\Delta H\|_F
=
\sqrt{39}
<
\sqrt{133}
=
\|\Delta Z\|_F
}
$$

となり、ReLUによって全体誤差が厳密に縮小した。ただし、これはこの具体例の結果である。すべての要素がReLUの正領域にあれば、全体でも等号になり得る。

## 9. exact置換との違い

打ち切りなしのTTLinear置換で

$$
\widetilde W=W
$$

なら、

$$
\Delta W=0,
\qquad
\Delta Z=0,
\qquad
\Delta H=0
$$

である。この場合、ReLUの1-Lipschitz上界を使わなくても、入力が同じなので出力も同じだと直接言える。

一方、rank打ち切りによって $\widetilde W\ne W$ となる場合に、

$$
\|\Delta H\|_F
\le
\|\Delta Z\|_F
$$

が誤差伝播の保証として必要になる。

## 10. この結果が保証しないもの

ここまでで保証したのは、同じbiasと同じ入力batchに対して、ReLUがLinear直後のFrobenius誤差を増幅しないことである。

このノート単体では、次をまだ保証していない。

- 後続Linearを通したlogits誤差。一般的な恒等式とnorm上界は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]で扱う
- classification marginに基づくprediction不変性
- Fashion-MNIST test accuracyの維持
- rankごとのaccuracy、parameter数、MACs、latency
- TT coreをfine-tuningした後の性能
- backward、gradient、optimizer、`state_dict`を含む学習可能なTTLinear module

PyTorchのReLUまでのtoy確認は、[[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]、第2 Linearまでのtoy確認は、[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]へ進む。学習済みFashion-MNIST MLPでの評価は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]で別段階として扱う。
