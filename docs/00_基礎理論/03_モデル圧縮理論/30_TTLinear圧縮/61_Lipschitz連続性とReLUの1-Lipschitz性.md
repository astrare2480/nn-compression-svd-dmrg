---
title: Lipschitz連続性とReLUの1-Lipschitz性
tags:
  - Lipschitz
  - ReLU
  - norm
  - error-propagation
---

# Lipschitz連続性とReLUの1-Lipschitz性

## 1. Lipschitz連続性の定義

写像

$$
f:\mathbb{R}^n\to\mathbb{R}^m
$$

がLipschitz連続であるとは、ある有限の定数 $K\ge0$ が存在して、任意の $x,y$ に対して

$$
\boxed{
\|f(x)-f(y)\|
\le
K\|x-y\|
}
$$

が成り立つことである。量化記号を明示すると、

$$
\exists K<\infty,\quad
\forall x,y,\quad
\|f(x)-f(y)\|
\le
K\|x-y\|.
$$

重要なのは、すべての点の組に共通する一つの $K$ が必要なことである。

$$
\forall x,y,\quad
\exists K_{x,y}
$$

では定義にならない。各点対ごとに定数を選べるなら、ほとんど何でも覆えてしまう。

## 2. boundedとLipschitzは異なる

関数値が発散しないことだけではLipschitz連続性は保証されない。見るべき量は差分商

$$
\frac{\|f(x)-f(y)\|}{\|x-y\|}
$$

が一様に有界かどうかである。

例えば

$$
f(x)=\sqrt{x},
\qquad
x\in[0,1]
$$

は $0\le f(x)\le1$ なのでboundedであり、閉区間上で一様連続でもある。しかし $y=0$、$x>0$ とすると、

$$
\frac{|f(x)-f(0)|}{|x-0|}
=
\frac{\sqrt{x}}{x}
=
\frac{1}{\sqrt{x}}.
$$

$x\to0^+$ でこれは発散するため、$[0,1]$ 上でLipschitz連続ではない。

## 3. ReLUの1-Lipschitz性

スカラーReLUを

$$
\rho(a)=\max(0,a)
$$

とする。区分的に書けば、

$$
\rho(a)
=
\begin{cases}
a,&a\ge0,\\
0,&a<0
\end{cases}
$$

このとき示したいのは、任意の実数 $a,b$ に対して

$$
\boxed{
|\rho(a)-\rho(b)|
\le
|a-b|
}
$$

である。

### 3.1 $a\ge0,\ b\ge0$

このとき

$$
\rho(a)=a,
\qquad
\rho(b)=b
$$

なので、

$$
|\rho(a)-\rho(b)|
=
|a-b|.
$$

この場合は不等号が等号になる。例えば $a=5$、$b=2$ なら、ReLU前後の差はどちらも3である。正の領域ではReLUの傾きが1なので、入力間の誤差は縮まらず、そのまま残る。

### 3.2 $a<0,\ b<0$

このとき

$$
\rho(a)=\rho(b)=0
$$

だから、

$$
|\rho(a)-\rho(b)|
=
0
\le
|a-b|.
$$

$a\ne b$ なら誤差は厳密に小さくなる。例えば $a=-2$、$b=-5$ では、ReLU前の差は

$$
|-2-(-5)|=3
$$

である一方、ReLU後は両方とも0になるので差は0である。

### 3.3 $a\ge0,\ b<0$

このとき

$$
\rho(a)=a,
\qquad
\rho(b)=0.
$$

従って、

$$
|\rho(a)-\rho(b)|
=
a.
$$

$b<0$ より $-b>0$ なので、

$$
|a-b|
=
a-b
=
a+(-b)
>
a.
$$

したがって、

$$
|\rho(a)-\rho(b)|
=
a
<
|a-b|
$$

となる。例えば $a=2$、$b=-3$ では、ReLU前の差は5、ReLU後の差は2である。

### 3.4 $a<0,\ b\ge0$

前の場合と対称に、

$$
|\rho(a)-\rho(b)|
=
b
<
b-a
=
|a-b|.
$$

ここで $a<0$ なので $-a>0$ であり、

$$
b-a=b+(-a)>b
$$

を使った。例えば $a=-4$、$b=1$ では、ReLU前の差は5、ReLU後の差は1である。

4場合を合わせると、ReLUは1-Lipschitzである。

4場合の違いをまとめると、次のようになる。

| $a$ の符号 | $b$ の符号 | ReLU後の誤差 | ReLU前との関係 |
| --- | --- | --- | --- |
| 非負 | 非負 | $\lvert a-b\rvert$ | 等号、誤差を保存 |
| 負 | 負 | $0$ | 異なる2点なら厳密に縮小 |
| 非負 | 負 | $a$ | 厳密に縮小 |
| 負 | 非負 | $b$ | 厳密に縮小 |

したがって「1-Lipschitz」は「必ず誤差を小さくする」という意味ではない。正確には、ReLUは入力間の距離を増幅しない、すなわちnon-expansiveである。

## 4. 定数1は最小である

$a=1$、$b=0$ とすれば、

$$
|\rho(1)-\rho(0)|
=
1
=
|1-0|.
$$

従って $K<1$ では不等式を満たせない。ReLUの最小Lipschitz定数は

$$
\boxed{K_{\min}=1}
$$

である。

## 5. 微分による見方

微分可能な区間で

$$
|\rho'(x)|\le1
$$

であることからも、平均値の定理を通じて1-Lipschitz性を理解できる。ReLUでは

$$
\rho'(x)=
\begin{cases}
0,&x<0,\\
1,&x>0
\end{cases}
$$

であり、$x=0$ では微分不可能である。そのため、全実数上の直接証明としては4場合分けの方が完全である。

## 6. ベクトル・テンソルへの拡張

ReLUを各要素へ作用させる。各成分で

$$
|\rho(A_i)-\rho(B_i)|
\le
|A_i-B_i|
$$

だから、両辺を二乗して総和を取ると、

$$
\sum_i
|\rho(A_i)-\rho(B_i)|^2
\le
\sum_i
|A_i-B_i|^2.
$$

平方根を取れば、

$$
\boxed{
\|\operatorname{ReLU}(A)-\operatorname{ReLU}(B)\|_F
\le
\|A-B\|_F
}
$$

を得る。行列でも高階テンソルでも、全要素を並べたEuclidean normがFrobenius normなので同じ証明でよい。

## 7. exact等価性と近似誤差伝播を区別する

入力が完全に同じなら、

$$
\widetilde Z=Z
$$

から

$$
\operatorname{ReLU}(\widetilde Z)
=
\operatorname{ReLU}(Z)
$$

が直接従う。この場合、Lipschitz性は不要である。

rank打ち切りにより

$$
\widetilde Z\ne Z
$$

となる場合に初めて、

$$
\|\operatorname{ReLU}(\widetilde Z)-\operatorname{ReLU}(Z)\|_F
\le
\|\widetilde Z-Z\|_F
$$

という誤差上界が役立つ。

この不等式はReLUが誤差を必ず小さくするという意味ではない。正の領域では傾きが1なので、誤差がそのまま残って等号になることもある。保証しているのは、Frobenius normで測った誤差を増幅しないことである。

## 8. 現在の学習境界

このノートで完了しているのはLipschitz連続性の定義、ReLUのスカラー4場合証明、テンソルへの一般的な拡張である。

Linearの重み誤差からhidden activation誤差までを、batch軸とhidden軸、全要素を表示した具体行列、Frobeniusノルムの途中式に接続する部分は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]で扱う。PyTorchでの要素比較と保存済み数値結果は、[[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]を参照する。

後続Linearまで含めると、さらに

$$
\widetilde Z_1
=
X\widetilde W_1^{\mathsf T}+b_1,
\qquad
\widetilde H
=
\operatorname{ReLU}(\widetilde Z_1)
$$

から出発してlogits誤差を評価する必要がある。このノートとReLUのtoy検証だけをもって、学習済みMLP全体のlogits誤差、prediction安定性、accuracyまで検証済みとは扱わない。

Linear層単体の誤差は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]を参照する。
