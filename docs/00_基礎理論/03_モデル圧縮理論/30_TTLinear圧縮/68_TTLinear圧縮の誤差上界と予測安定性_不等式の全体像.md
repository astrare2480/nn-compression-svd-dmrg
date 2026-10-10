---
title: TTLinear圧縮の誤差上界と予測安定性 不等式の全体像
tags:
  - TTLinear
  - error-bound
  - Lipschitz
  - classification-margin
  - prediction-stability
  - sufficient-condition
---

# TTLinear圧縮の誤差上界と予測安定性：不等式の全体像

## 1. このノートの目的

TT/MPSやTTLinearによってNeural Networkの重みを圧縮すると、元の重み $W_1$ は近似重み $\widetilde W_1$ に置き換わる。そのとき、本当に知りたいのは重みの差そのものではなく、次の関係である。

$$
\text{weight誤差}
\longrightarrow
\text{hidden誤差}
\longrightarrow
\text{logits誤差}
\longrightarrow
\text{predictionが変わるか}
$$

これまで学んだ複数の不等式は、この矢印を1段ずつつなぐためにある。

このノートでは、個別の証明を繰り返すことよりも、次を一つの流れとして整理する。

- 各不等式が何を上から押さえるのか。
- なぜ等式ではなく不等式が必要なのか。
- 複数の不等式をどの順序で接続するのか。
- TT-rank選択、layer感度評価、prediction stability判定にどう使うのか。
- 上界が成立しないとき、何が言えて何が言えないのか。

個々の完全な証明、全要素行列、PyTorch実装、保存済み実験結果は、本文中の関連ノートへ分離している。

## 2. 先に全体の結論を見る

2層MLPの第1 Linearだけを近似したとき、これまでの不等式は最終的に次の1本のchainにまとまる。

$$
\boxed{
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
}
$$

ここで、

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

と置くと、

$$
\|\delta l_n\|_\infty
\le
U
$$

である。さらにsample $n$ のclassification marginを $\gamma_n$ とすると、

$$
\boxed{
U
<
\frac{\gamma_n}{2}
}
$$

は、そのsampleの予測classが変わらないための十分条件になる。

batch内のminimum marginを、

$$
\gamma_{\min}
=
\min_{1\le n\le B}\gamma_n
$$

とすれば、

$$
\boxed{
U
<
\frac{\gamma_{\min}}{2}
}
$$

により、batch内の全sampleの予測が同時に不変であることを保証できる。

すなわち、これまでの不等式は、

$$
\boxed{
\text{TT-rankで決まる重み誤差}
\longrightarrow
\text{prediction不変の十分条件}
}
$$

を作るための道具である。

## 3. 問題設定

batch sizeを $B$、入力次元を $d$、hidden次元を $h$、class数を $c$ とする。

Dense modelを、

$$
Z_1
=
XW_1^{\mathsf T}
+
\mathbf 1b_1^{\mathsf T},
$$

$$
H
=
\operatorname{ReLU}(Z_1),
$$

$$
L
=
HW_2^{\mathsf T}
+
\mathbf 1b_2^{\mathsf T}
$$

とする。shapeは、

$$
X\in\mathbb R^{B\times d},
\qquad
W_1\in\mathbb R^{h\times d},
\qquad
W_2\in\mathbb R^{c\times h},
$$

$$
Z_1,H\in\mathbb R^{B\times h},
\qquad
L\in\mathbb R^{B\times c}
$$

である。

第1 Linearの重みをTT/MPSで近似し、再構成したdense重みを $\widetilde W_1$ とする。誤差の符号は、

$$
\Delta W_1
=
\widetilde W_1-W_1
$$

で統一する。近似modelの各量を、

$$
\widetilde Z_1,
\qquad
\widetilde H,
\qquad
\widetilde L
$$

とし、

$$
\Delta Z_1
=
\widetilde Z_1-Z_1,
$$

$$
\Delta H
=
\widetilde H-H,
$$

$$
\Delta L
=
\widetilde L-L
$$

と置く。

## 4. まず等式と不等式を区別する

誤差伝播に現れる式は、全てが上界ではない。

第1 Linearでは、biasが共通ならば、

$$
\boxed{
\Delta Z_1
=
X\Delta W_1^{\mathsf T}
}
$$

が厳密に成立する。

第2 Linearの重みとbiasも共通ならば、

$$
\boxed{
\Delta L
=
\Delta H W_2^{\mathsf T}
}
$$

も厳密な等式である。

一方、次は不等式である。

$$
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2,
$$

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F,
$$

$$
\|\Delta L\|_F
\le
\|\Delta H\|_F\|W_2\|_2.
$$

等式は実際の誤差の向きを保持する。不等式はその向きを一部捨て、大きさだけを安全側に評価する。

この区別は重要である。

- 等式が成立しない場合は、符号、転置、bias、shapeのいずれかが間違っている可能性が高い。
- 不等式に大きな余裕がある場合は、計算ミスとは限らない。最大増幅方向と実際の誤差方向が揃っていないため、上界がlooseになることがある。

## 5. 基本道具：行列積のFrobenius上界

互いに掛けられる行列 $A,B$ に対して、

$$
\boxed{
\|AB\|_F
\le
\|A\|_F\|B\|_2
}
$$

が成立する。$\|B\|_2$ は行列のspectral norm、すなわち最大特異値である。

$A$ の第 $i$ 行を $a_i^{\mathsf T}$ とすると、$AB$ の第 $i$ 行は $a_i^{\mathsf T}B$ である。spectral normの定義から、

$$
\begin{aligned}
\|a_i^{\mathsf T}B\|_2
&=
\|B^{\mathsf T}a_i\|_2\\
&\le
\|B^{\mathsf T}\|_2\|a_i\|_2\\
&=
\|B\|_2\|a_i\|_2.
\end{aligned}
$$

両辺を2乗し、全行で和を取ると、

$$
\begin{aligned}
\|AB\|_F^2
&=
\sum_i\|a_i^{\mathsf T}B\|_2^2\\
&\le
\sum_i\|B\|_2^2\|a_i\|_2^2\\
&=
\|B\|_2^2
\sum_i\|a_i\|_2^2\\
&=
\|B\|_2^2\|A\|_F^2.
\end{aligned}
$$

両辺は非負なので平方根を取れば、

$$
\|AB\|_F
\le
\|A\|_F\|B\|_2
$$

を得る。

同様に、

$$
\boxed{
\|AB\|_F
\le
\|A\|_2\|B\|_F
}
$$

も成立する。したがって、同じ行列積に対して複数の上界を計算できるならば、小さい方を使うことができる。

完全な成分展開と数値例は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]を参照する。spectral normが最大特異値になる理由は、[[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]を参照する。

## 6. 第1 Linear：重み誤差からpre-activation誤差へ

厳密な等式は、

$$
\Delta Z_1
=
X\Delta W_1^{\mathsf T}
$$

である。前節の行列積の上界を使うと、

$$
\begin{aligned}
\|\Delta Z_1\|_F
&=
\|X\Delta W_1^{\mathsf T}\|_F\\
&\le
\|X\|_F
\|\Delta W_1^{\mathsf T}\|_2\\
&=
\|X\|_F
\|\Delta W_1\|_2.
\end{aligned}
$$

従って、

$$
\boxed{
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2
}
$$

である。

これは、重み誤差が入力batchによってどの程度pre-activation誤差へ変換され得るかを示す。

- $\|X\|_F$ が大きいbatchでは、同じ $\Delta W_1$ でも出力誤差が大きくなり得る。
- $\|\Delta W_1\|_2$ は、重み近似誤差が最悪の入力方向を何倍に増幅するかを表す。
- TT-rankを変えると $\widetilde W_1$ が変わるため、$\|\Delta W_1\|_2$ も変化する。

別の上界として、

$$
\|\Delta Z_1\|_F
\le
\|X\|_2\|\Delta W_1\|_F
$$

も使える。一方のみを無条件に採用するのではなく、実際の値と計算コストを見て使い分ける。

## 7. ReLU：pre-activation誤差を増幅しない

scalar ReLUを、

$$
\rho(t)
=
\max(0,t)
$$

とする。ReLUは1-Lipschitzなので、任意の $a,b\in\mathbb R$ に対して、

$$
|\rho(a)-\rho(b)|
\le
|a-b|
$$

が成立する。これは次の4種類の符号ケースで確認できる。

| $a$ | $b$ | ReLU後の誤差 |
| --- | --- | --- |
| $a\ge0$ | $b\ge0$ | $|a-b|$のまま |
| $a<0$ | $b<0$ | $0$ |
| $a\ge0$ | $b<0$ | $a\le a-b=|a-b|$ |
| $a<0$ | $b\ge0$ | $b\le b-a=|a-b|$ |

行列の各要素に適用すると、

$$
|\Delta H_{n,j}|
\le
|\Delta Z_{1,n,j}|.
$$

両辺を2乗して全sample・全hidden unitで和を取ると、

$$
\sum_{n=1}^{B}
\sum_{j=1}^{h}
|\Delta H_{n,j}|^2
\le
\sum_{n=1}^{B}
\sum_{j=1}^{h}
|\Delta Z_{1,n,j}|^2.
$$

従って、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F
}
$$

である。

この不等式が示すのは、ReLUが誤差を必ず小さくすることではない。両方のpre-activationが正ならば誤差はそのまま残り、等号になる。保証するのは、ReLUがFrobenius誤差を増幅しないことである。

4種類の符号ケースの完全な導出は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]を参照する。全要素を表示した $2\times4$ 行列例は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]を参照する。

## 8. 第2 Linear：hidden誤差からlogits誤差へ

第2 Linearの重み $W_2$ は共通なので、

$$
\Delta L
=
\Delta H W_2^{\mathsf T}
$$

が厳密に成立する。行列積の上界を使うと、

$$
\begin{aligned}
\|\Delta L\|_F
&=
\|\Delta H W_2^{\mathsf T}\|_F\\
&\le
\|\Delta H\|_F
\|W_2^{\mathsf T}\|_2\\
&=
\|\Delta H\|_F
\|W_2\|_2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|\Delta L\|_F
\le
\|\Delta H\|_F\|W_2\|_2
}
$$

である。

$\|W_2\|_2$ は、hidden誤差をlogits誤差へ送るときの最大増幅率である。

- $\|W_2\|_2<1$ ならば、第2 Linearはどの方向のhidden誤差もnormでは縮小する。
- $\|W_2\|_2>1$ ならば、方向によっては増幅し得る。
- 実際に $\|W_2\|_2$ 倍になるには、hidden誤差が $W_2$ の最大右特異ベクトル方向へ揃う必要がある。

成分ごとの導出と全要素を表示した小行列例は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]を参照する。

## 9. 3段階を合成する

ここまでの3つの不等式を順番に並べる。

$$
\|\Delta L\|_F
\le
\|\Delta H\|_F\|W_2\|_2,
$$

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F,
$$

$$
\|\Delta Z_1\|_F
\le
\|X\|_F\|\Delta W_1\|_2.
$$

第1式の $\|\Delta H\|_F$ を第2式で、さらに $\|\Delta Z_1\|_F$ を第3式で上から押さえると、

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

従って、

$$
\boxed{
\|\Delta L\|_F
\le
U
}
$$

である。ただし、

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

とした。

この上界は、「実際のlogits誤差が $U$ に近い」と予測する式ではない。最悪方向に基づく上限であり、

$$
\text{実測誤差}
\le
\text{理論上界}
$$

を保証するために使う。

## 10. Batch全体の誤差を1 sampleへ接続する

$\Delta L\in\mathbb R^{B\times c}$ の第 $n$ 行をsample $n$ のlogits誤差とし、

$$
\delta l_n
\in
\mathbb R^c
$$

と書く。数学的に $\delta l_n$ を列ベクトルとするならば、$\Delta L$ の第 $n$ 行は $\delta l_n^{\mathsf T}$ である。

まず、vectorの定義から、

$$
\begin{aligned}
\|\delta l_n\|_\infty
&=
\max_k|\delta l_{n,k}|\\
&\le
\left(
\sum_{k=1}^{c}
|\delta l_{n,k}|^2
\right)^{1/2}\\
&=
\|\delta l_n\|_2.
\end{aligned}
$$

従って、

$$
\boxed{
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
}
$$

である。

次に、Frobenius normの二乗を第 $n$ 行とそれ以外に分ける。

$$
\begin{aligned}
\|\Delta L\|_F^2
&=
\sum_{i=1}^{B}
\sum_{k=1}^{c}
|\Delta L_{i,k}|^2\\
&=
\sum_{k=1}^{c}
|\Delta L_{n,k}|^2
+
\sum_{\substack{i=1\\i\ne n}}^{B}
\sum_{k=1}^{c}
|\Delta L_{i,k}|^2\\
&=
\|\delta l_n\|_2^2
+
\sum_{\substack{i=1\\i\ne n}}^{B}
\sum_{k=1}^{c}
|\Delta L_{i,k}|^2\\
&\ge
\|\delta l_n\|_2^2.
\end{aligned}
$$

両辺は非負なので平方根を取り、

$$
\boxed{
\|\delta l_n\|_2
\le
\|\Delta L\|_F
}
$$

を得る。

この2つを合わせると、

$$
\boxed{
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
U
}
$$

となる。

ここで初めて、batch-levelの行列誤差と、1 sampleのclass-wise最大誤差がつながる。完全な導出、normの使い分け、具体行列は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]を参照する。

## 11. Classification marginへ接続する

sample $n$ のdense logitsを $l_n$ とする。dense modelの予測classを、

$$
y_n^*
=
\operatorname*{argmax}_{0\le k<c}(l_n)_k
$$

とする。classification marginは、

$$
\gamma_n
=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k
$$

である。$\gamma_n>0$ ならば、dense logitsのtop classは一意である。

compressed logitsを、

$$
\widetilde l_n
=
l_n+\delta l_n
$$

とする。任意のcompetitor $k\ne y_n^*$ に対して、圧縮後のgapは、

$$
\begin{aligned}
(\widetilde l_n)_{y_n^*}
-(\widetilde l_n)_k
&=
(l_n)_{y_n^*}
-(l_n)_k
+
\delta l_{n,y_n^*}
-
\delta l_{n,k}\\
&\ge
\gamma_n
-
|\delta l_{n,y_n^*}|
-
|\delta l_{n,k}|\\
&\ge
\gamma_n
-
2\|\delta l_n\|_\infty.
\end{aligned}
$$

従って、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
$$

ならば、

$$
(\widetilde l_n)_{y_n^*}
-(\widetilde l_n)_k
>
0
$$

が全competitorで成立する。したがって、

$$
\boxed{
\operatorname*{argmax}_k
(\widetilde l_n)_k
=
\operatorname*{argmax}_k
(l_n)_k
}
$$

が保証される。

ここで等号を含む $\le$ ではなく、strictな $<$ を使う。等号ではtop classとcompetitorがtieになる可能性を除外できないからである。

さらに、

$$
\|\delta l_n\|_\infty
\le
U
$$

なので、より強い条件である、

$$
\boxed{
U
<
\frac{\gamma_n}{2}
}
$$

でもprediction不変を保証できる。

十分条件と必要条件の違い、誤差方向を使った直接判定、common shiftの反例は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]を参照する。

## 12. Minimum marginでbatch内全sampleを保証する

batch内の全sampleのmarginを、

$$
\gamma
=
\begin{pmatrix}
\gamma_1\\
\gamma_2\\
\vdots\\
\gamma_B
\end{pmatrix}
$$

と並べ、minimum marginを、

$$
\gamma_{\min}
=
\min_{1\le n\le B}\gamma_n
$$

とする。minimumの定義から、任意の $n$ に対して、

$$
\gamma_{\min}
\le
\gamma_n
$$

なので、

$$
\frac{\gamma_{\min}}{2}
\le
\frac{\gamma_n}{2}.
$$

もし、

$$
U
<
\frac{\gamma_{\min}}{2}
$$

ならば、全sampleに対して、

$$
\begin{aligned}
\|\delta l_n\|_\infty
&\le
U\\
&<
\frac{\gamma_{\min}}{2}\\
&\le
\frac{\gamma_n}{2}.
\end{aligned}
$$

従って、

$$
\boxed{
U
<
\frac{\gamma_{\min}}{2}
\quad\Longrightarrow\quad
\widetilde y
=
y^*
}
$$

である。ここで $y^*$ と $\widetilde y$ は、dense modelとcompressed modelのbatch prediction vectorである。

minimum marginを持つsampleは、marginだけを使う共通保証のbottleneckである。ただし、そのsampleが実際に最初にprediction changeを起こすとは限らない。実際の変化には誤差の向きも関係する。

完全な導出と全sampleの数値確認は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]を参照する。

## 13. 一つの数値例で全chainを読む

保存済みtoy Notebookでは、第3 sampleに対して、

$$
\|\delta l_3\|_\infty
=
0.01800000,
$$

$$
\|\delta l_3\|_2
=
0.02670206,
$$

$$
\|\Delta L\|_F
=
0.03997499,
$$

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
=
0.05221958
$$

を得ている。dense logitsは、

$$
l_3
=
\begin{pmatrix}
6.0\\
4.5\\
-6.0
\end{pmatrix}
$$

なので、classification marginは、

$$
\begin{aligned}
\gamma_3
&=
6.0-4.5\\
&=
1.5,
\end{aligned}
$$

$$
\frac{\gamma_3}{2}
=
0.75
$$

である。したがって、

$$
\boxed{
0.01800000
\le
0.02670206
\le
0.03997499
\le
0.05221958
<
0.75000000
}
$$

が成立する。

この1行は、次を意味する。

1. 第3 sampleの各class logitの実測誤差は、最大で $0.018$ である。
2. その誤差は、sampleのvector 2-normとbatch全体のFrobenius normで順に上から押さえられる。
3. batchのlogits誤差は、weight誤差と行列のspectral normから得られる $U$ 以下である。
4. $U$ がmarginの半分より小さいため、第3 sampleのargmaxは変わらない。

batch内のmargin vectorは、

$$
\gamma
=
\begin{pmatrix}
4.5\\
2.5\\
1.5
\end{pmatrix},
$$

従って、

$$
\gamma_{\min}
=
1.5,
\qquad
\frac{\gamma_{\min}}{2}
=
0.75.
$$

$$
U
=
0.05221958
<
0.75
$$

なので、batch全体でもcertificateが成立する。実際に保存済み結果では、

$$
y^*
=
\widetilde y
=
\begin{pmatrix}
0\\
1\\
0
\end{pmatrix}
$$

である。

これらの数値は現行Notebookの保存済み出力に基づく。このまとめを作る際に、新しいkernelからFresh Run Allした結果ではない。設定、全要素の計算、結果の考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]と[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]を参照する。

## 14. なぜ不等式で評価するのか

### 14.1 実際の誤差を計算する前に、安全側の上限を作れる

厳密な $\Delta L$ を得るには、dense modelとcompressed modelの両方をforwardし、差を計算する必要がある。

一方、

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

は、入力の大きさ、重み近似誤差、後段layerの最大増幅率から計算できる。そのため、複数のrank候補を安全側にスクリーニングする指標にできる。

### 14.2 未知の誤差方向を全て含む

spectral normは、行列が入力ベクトルを最大で何倍に伸ばすかを表す。実際の $\Delta W_1$ や $\Delta H$ の向きが分からなくても、最悪方向を含めて上から押さえられる。

これは保守的だが、「この値を超えない」という保証に使える。

### 14.3 異なる種類の量を接続できる

重み誤差 $\Delta W_1$ はparameter空間の量である。予測はclass順位の離散的な量である。両者はそのままでは比較できない。

中間に、

$$
\Delta W_1
\to
\Delta Z_1
\to
\Delta H
\to
\Delta L
\to
\delta l_n
\to
\gamma_n
$$

という連続量の列を作り、それぞれを不等式でつなぐことによって、重み近似からargmax stabilityまでを論理的に接続できる。

## 15. 何に使うのか

### 15.1 TT-rank候補のスクリーニング

rank候補 $r$ ごとに $\widetilde W_1(r)$ を作り、

$$
\Delta W_1(r)
=
\widetilde W_1(r)-W_1
$$

を計算する。その上で、

$$
U(r)
=
\|X\|_F
\|\Delta W_1(r)\|_2
\|W_2\|_2
$$

を比較する。

$$
U(r)
<
\frac{\gamma_{\min}}{2}
$$

を満たすrankは、そのbatchについてdense/compressed predictionの一致を保証できる。複数のrankが条件を満たすならば、parameter数やMACsなども考慮してより小さいrankを候補にできる。

ただし、certificateがFalseのrankをすぐに不適格と判定してはいけない。Falseが意味するのは、この上界では保証できないということだけである。実測logits誤差、prediction agreement、accuracyも別途計測する。

### 15.2 どのlayerを圧縮するかの比較

同じ大きさの重み誤差でも、後段の $\|W_2\|_2$ が大きいlayerほどlogitsへの最大影響が大きくなり得る。

したがって、

$$
\|\Delta W_1\|_2
\|W_2\|_2
$$

のような量は、第1 Linearの圧縮誤差と後段の増幅率を同時に考えるlayer感度の指標になる。

ただし、これもtask lossやaccuracyを直接測る量ではない。同じ圧縮率でどのlayerがより影響を受けやすいかを予備的に比較するために使う。

### 15.3 Bottleneck sampleの発見

$$
\gamma_{\min}
=
\min_n\gamma_n
$$

を調べると、marginに基づく共通保証を最も厳しくするsampleを発見できる。

これsampleを調べることで、例えば次を分析できる。

- dense modelの時点でtop-1とtop-2が接近しているか。
- 圧縮誤差がtop classに不利な方向へ作用しているか。
- 十分条件が不成立でも、実際のpredictionは保存されているか。

### 15.4 上界のtightnessの評価

実験では、

$$
\rho_{\mathrm{bound}}
=
\frac{\|\Delta L\|_F}{U}
$$

を記録できる。$0<U$ ならば、

$$
0
\le
\rho_{\mathrm{bound}}
\le
1
$$

である。

- $\rho_{\mathrm{bound}}$ が1に近いとき、その条件では上界が比較的tightである。
- $\rho_{\mathrm{bound}}$ が小さいとき、複数段階のworst-case評価による余裕が大きい。
- $U=0$ のときは実測誤差も0になるが、比率は $0/0$ で未定義なので`NaN`として0と扱わない。

上界と実測値を並べることで、式が成立するかだけでなく、圧縮設計に実用できる程度のtightnessがあるかを評価できる。

### 15.5 Prediction agreementとaccuracyの切り分け

certificateが保証するのは、

$$
\widetilde y_n
=
y_n^*
$$

というdense/compressed prediction agreementである。dense prediction $y_n^*$ がground-truth labelと一致することは、別に確認する必要がある。

将来ground-truth label $t_n$ を導入したとき、

$$
y_n^*
=
t_n
$$

であり、かつcertificateが成立するsampleでは、

$$
\widetilde y_n
=
y_n^*
=
t_n
$$

となる。従って、「dense modelが正解し、さらにprediction不変を保証できたsample」は、compressed modelでも正解と保証できる。

ただし、このノートではground-truthを用いたdataset-levelのaccuracy lower boundまでは導出しない。これは次段階のテーマである。

## 16. なぜ上界がlooseになるのか

合成上界は、次の段階で方向情報を捨てる。

### 16.1 重み誤差からLinear出力誤差へ

$$
\|X\Delta W_1^{\mathsf T}\|_F
\le
\|X\|_F\|\Delta W_1\|_2
$$

では、入力と $\Delta W_1$ の特異ベクトルとの向きを捨てる。等号に近づくには、batch内の入力が $\Delta W_1$ の最大増幅方向と強く整列する必要がある。

### 16.2 ReLU

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F
$$

では、ReLUが負の成分を0にすることによる実際の減衰を上界に反映しない。全成分が正の領域に留まるならば等号も起こり得るが、負の成分が多いと実際の $\|\Delta H\|_F$ はより小さくなる。

### 16.3 Hidden誤差からlogits誤差へ

$$
\|\Delta H W_2^{\mathsf T}\|_F
\le
\|\Delta H\|_F\|W_2\|_2
$$

では、$\Delta H$ の向きが $W_2$ の最大特異方向と揃っているかを捨てる。

### 16.4 Sample誤差からbatch誤差へ

$$
\|\delta l_n\|_2
\le
\|\Delta L\|_F
$$

では、注目sample以外の誤差もFrobenius normに加える。他sampleの誤差が大きいと、注目sampleの判定に使う上界も大きくなる。

### 16.5 Vector 2-normからinfinity normへ

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
$$

では、classごとの誤差の分布と向きを捨てる。全classに同じ値を加えるcommon shiftはargmaxを変えないが、normは大きくなり得る。

このように、各不等式は個別には正しくても、連続して使うと余裕が積み重なる。そのため、理論上界と実測誤差は必ず並べて記録する。

## 17. より細かい評価へ改善する方法

### 17.1 Sample-wiseな入力normを使う

batch Frobenius norm $\|X\|_F$ を使う代わりに、sample $n$ の入力 $x_n$ に限定すると、同じ導出から、

$$
\|\delta l_n\|_\infty
\le
\|x_n\|_2
\|\Delta W_1\|_2
\|W_2\|_2
$$

を得られる。従って、

$$
U_n
=
\|x_n\|_2
\|\Delta W_1\|_2
\|W_2\|_2
$$

と置き、

$$
U_n
<
\frac{\gamma_n}{2}
$$

をsampleごとに判定できる。通常、

$$
\|x_n\|_2
\le
\|X\|_F
$$

なので、batch全体の $U$ を各sampleに使うよりもtightになり得る。

### 17.2 実際のlogits誤差方向を使う

$\ell_\infty$ normのみを使う十分条件は、top classを下げ、competitorを上げる最悪方向を考える。実際の $\delta l_n$ が得られているならば、各competitor $k$ に対して、

$$
(l_n)_{y_n^*}
-(l_n)_k
+
\delta l_{n,y_n^*}
-
\delta l_{n,k}
>
0
$$

を直接確認できる。これはnormだけを使う条件よりも、誤差方向の情報を保つ。

### 17.3 Batch certificateだけでなくcertified sample率を記録する

minimum marginを使うbatch certificateは、1 sampleでもmarginが小さいとFalseになる。その場合も、sampleごとの条件、

$$
U_n
<
\frac{\gamma_n}{2}
$$

を判定し、

$$
\text{certified fraction}
=
\frac{十分条件が成立したsample数}{全sample数}
$$

を記録すると、「全sampleを保証できるか」だけでなく、どの程度のsampleを保証できるかを比較できる。

## 18. より深いnetworkへの見方

Linear層のspectral normはそのlayerの最大増幅率、ReLUのLipschitz定数は1である。そのため、より深いnetworkでも1つのlayerに生じた誤差は、後段layerのLipschitz定数の積を使って上から押さえられる。

圧縮したlayerより後ろにLinear層が $q$ 個あり、その重みを $W_{j+1},\ldots,W_{j+q}$ とすると、概念的には、

$$
\text{output誤差}
\le
\text{local誤差}
\prod_{s=1}^{q}
\|W_{j+s}\|_2
$$

と評価する。ReLUは係数1なので積を増やさない。

ただし、複数layerを同時に圧縮する場合は、それぞれの局所誤差が異なる経路から出力へ伝播する。単に1つの $\Delta W$ を置く場合とは異なり、triangle inequalityを使った複数誤差の合成が必要になる。

## 19. これらの不等式で言えること

次を言える。

- 重み近似誤差から、pre-activation、hidden activation、logitsの誤差上界を順番に作れる。
- ReLUはFrobenius normで測った誤差を増幅しない。
- 後段Linearが誤差を最大でどの程度増幅し得るかを、spectral normで評価できる。
- Batch-levelのlogits誤差を1 sampleのclass-wise最大誤差へ接続できる。
- Marginと比較することで、sampleごとのprediction不変を十分条件として保証できる。
- Minimum marginを使うことで、1つのbatchの全sampleを同時に保証できる。
- Rank候補や圧縮layerの感度を、安全側の指標で比較できる。

## 20. これらの不等式だけでは言えないこと

次は自動的には保証されない。

- CertificateがFalseのsampleで、実際にpredictionが変わること。
- Dense predictionがground-truth labelに対して正しいこと。
- Dataset全体のaccuracyが維持されること。
- 異なるminibatchやdataset全体へ、1つのbatchの $\gamma_{\min}$ をそのまま使えること。
- 理論上界と実測誤差が近いこと。
- 人工的に作った $\Delta W_1$ によるtoy実験が、特定のTT-rankで実際に得られた圧縮誤差を再現していること。
- Parameter数、MACs、latency、training後のfine-tuning性能まで最適であること。

とくに、十分条件の論理を逆向きに使ってはいけない。

$$
\text{certificate=True}
\quad\Longrightarrow\quad
\text{prediction不変}
$$

は正しいが、

$$
\text{certificate=False}
\quad\Longrightarrow\quad
\text{prediction変化}
$$

は正しくない。

## 21. 実験で並べて記録する量

TT-rank候補を比較するときは、少なくとも次を分けて記録する。

### 21.1 圧縮量

- TT-rank。
- Dense parameter数とTT parameter数。
- Compression ratio。
- 必要に応じてMACsとlatency。

### 21.2 重み誤差

$$
\|\Delta W_1\|_2,
\qquad
\|\Delta W_1\|_F.
$$

### 21.3 中間誤差とlogits誤差

$$
\|\Delta Z_1\|_F,
\qquad
\|\Delta H\|_F,
\qquad
\|\Delta L\|_F.
$$

### 21.4 理論上界とtightness

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2,
$$

$$
\frac{\|\Delta L\|_F}{U}.
$$

### 21.5 Prediction stability

- Sampleごとの $\gamma_n$。
- $\gamma_{\min}$ とbottleneck sample。
- Sampleごのcertificate。
- Certified sample率。
- Dense/compressed prediction agreement率。

### 21.6 Task性能

- Dense accuracy。
- Compressed accuracy。
- Accuracy drop。
- Fine-tuning前後のaccuracy。

上界だけ、accuracyだけのどちらか一方を記録するのではない。圧縮量、誤差、保証、実測task性能を分けて並べることで、何を理論が保証し、何を実験で確認したかを区別できる。

## 22. まとめ

最も重要な流れは、

$$
\boxed{
\begin{aligned}
\|\delta l_n\|_\infty
&\le
\|\delta l_n\|_2\\
&\le
\|\Delta L\|_F\\
&\le
\|\Delta H\|_F\|W_2\|_2\\
&\le
\|\Delta Z_1\|_F\|W_2\|_2\\
&\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2\\
&=
U.
\end{aligned}
}
$$

である。

これをmarginと比較し、

$$
U
<
\frac{\gamma_n}{2}
$$

ならsample $n$ のprediction不変、

$$
U
<
\frac{\gamma_{\min}}{2}
$$

ならbatch内全sampleのprediction不変を保証できる。

これらの不等式の目的は、実測誤差をぴったり予測することではない。TT-rank打ち切りで生じた重み誤差がNeural Networkの出力と予測に与える最悪影響を、計算可能な形で上から押さえることにある。

その上で、理論上界と実測値、prediction agreementとaccuracyを分けて記録することで、圧縮率だけでなく「どこまで圧縮しても予測を保てるか」を実験と理論の両方から評価できる。

## 関連ノート

- [[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
- [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]
- [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]
