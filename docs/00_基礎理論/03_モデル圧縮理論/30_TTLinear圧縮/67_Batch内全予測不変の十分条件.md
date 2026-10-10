---
title: Batch内全予測不変の十分条件
tags:
  - TTLinear
  - logits
  - classification-margin
  - minimum-margin
  - batch-wide-certificate
---

# Batch内全予測不変の十分条件

## 1. このノートの目的

前段では、batch内の任意のsample $n$ について、

$$
\|\delta l_n\|_\infty
\le
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

と、sample-wiseな予測不変の十分条件、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
$$

を接続した。

このノートでは、batch内で最も小さいclassification margin、

$$
\gamma_{\min}
=
\min_{1\le n\le B}\gamma_n
$$

を導入し、単一の共通上界からbatch内の全sampleを同時に保証する。

中心となる結論は、

$$
\boxed{
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_{\min}}{2}
}
$$

なら、

$$
\boxed{
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k
\qquad
\text{for all }n=1,\ldots,B
}
$$

となることである。

ここで保証するのは、1つのbatch内におけるdense modelとcompressed modelのprediction agreementである。ground-truth label、classification accuracy、dataset全体、minibatchをまたぐglobal minimum margin、TT-rank sweep、fine-tuningには進まない。

## 2. 記号とshape

batch sizeを $B$、class数を $c\ge2$ とする。dense logitsとcompressed logitsを、

$$
L,
\widetilde L
\in
\mathbb{R}^{B\times c}
$$

とする。行がsample軸、列がclass軸である。

$$
L
=
\begin{pmatrix}
l_1^T\\
l_2^T\\
\vdots\\
l_B^T
\end{pmatrix},
\qquad
\widetilde L
=
\begin{pmatrix}
\widetilde l_1^T\\
\widetilde l_2^T\\
\vdots\\
\widetilde l_B^T
\end{pmatrix}.
$$

各sampleのlogitsは、

$$
l_n,
\widetilde l_n
\in
\mathbb{R}^{c}
$$

である。

## 3. Dense prediction vector $y^*$

### 3.1 意味と定義

sample $n$ のdense predicted classを、

$$
y_n^*
=
\operatorname*{argmax}_{0\le k<c}(l_n)_k
$$

と定義する。batch内の予測をsample順に並べたvectorは、

$$
y^*
=
\begin{pmatrix}
y_1^*\\
y_2^*\\
\vdots\\
y_B^*
\end{pmatrix}
\in
\{0,\ldots,c-1\}^{B}
$$

である。

### 3.2 Shapeと軸

$y^*$のshapeは$(B,)$であり、唯一の軸はsample軸である。各要素はlogit値ではなくclass indexである。

### 3.3 計算元と用途

$y^*$はdense logits $L$の各行でargmaxを取って計算する。compressed prediction vectorと比較する基準として使う。

## 4. Sample margin $\gamma_n$

### 4.1 意味と定義

dense top classが一意であるとする。sample $n$ のclassification marginは、

$$
\begin{aligned}
\gamma_n
&=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k\\
&=
\text{top-1 logit}
-
\text{top-2 logit}.
\end{aligned}
$$

$\gamma_n$はsample $n$ のtop classがstrongest competitorに対して持つ余裕である。

### 4.2 Shapeと軸

$\gamma_n\in\mathbb{R}$はscalarであり、軸を持たない。

### 4.3 計算元と用途

$\gamma_n$はdense logits $l_n$の最大値と2番目に大きい値から計算する。sample-wise prediction stabilityのthreshold、

$$
\frac{\gamma_n}{2}
$$

を作るために使う。

## 5. Margin vector $\gamma$

### 5.1 意味と定義

全sampleのmarginを並べて、

$$
\gamma
=
\begin{pmatrix}
\gamma_1\\
\gamma_2\\
\vdots\\
\gamma_B
\end{pmatrix}
\in
\mathbb{R}^{B}
$$

とする。

### 5.2 Shapeと軸

$\gamma$のshapeは$(B,)$であり、唯一の軸はsample軸である。第 $n$ 成分がsample $n$ のmarginに対応する。

### 5.3 計算元と用途

$\gamma$は各行のtop-1 logitとtop-2 logitの差から計算する。batch内で最も厳しいsampleを探すために使う。

## 6. Minimum margin $\gamma_{\min}$

### 6.1 意味と定義

batch内のminimum marginを、

$$
\boxed{
\gamma_{\min}
=
\min_{1\le n\le B}\gamma_n
}
$$

と定義する。

$\gamma_{\min}$は、batch内で最も小さいclassification marginである。marginに基づく保証において最も余裕が少ないsample、すなわちweakest sampleまたはbottleneck sampleのmarginである。

### 6.2 Shapeと軸

$\gamma_{\min}\in\mathbb{R}$はscalarであり、軸を持たない。

### 6.3 計算元と用途

$\gamma_{\min}$はmargin vector $\gamma$の最小値である。全sampleに共通する最も厳しいthreshold、

$$
\frac{\gamma_{\min}}{2}
$$

を作るために使う。

### 6.4 Bottleneck sample index

minimum marginを持つsample indexを、

$$
n_{\mathrm{bottleneck}}
=
\operatorname*{argmin}_{1\le n\le B}\gamma_n
$$

とする。これはscalar indexである。

ただし、bottleneck sampleは「marginに基づく共通保証を最も厳しくするsample」である。実際の圧縮で必ず最初にpredictionが変わるsampleという意味ではない。実際の変化には誤差の大きさと方向も関係する。

## 7. Minimum marginを全要素で確認する

例えば、

$$
L
=
\begin{pmatrix}
5 & 2 & 1\\
1 & 3 & 2.6\\
0 & 1 & 4
\end{pmatrix}
\in
\mathbb{R}^{3\times3}
$$

とする。class indexを$0,1,2$とする。

第1 sampleでは、

$$
l_1
=
\begin{pmatrix}
5\\
2\\
1
\end{pmatrix},
\qquad
y_1^*=0,
\qquad
\gamma_1=5-2=3.
$$

第2 sampleでは、

$$
l_2
=
\begin{pmatrix}
1\\
3\\
2.6
\end{pmatrix},
\qquad
y_2^*=1,
\qquad
\gamma_2=3-2.6=0.4.
$$

第3 sampleでは、

$$
l_3
=
\begin{pmatrix}
0\\
1\\
4
\end{pmatrix},
\qquad
y_3^*=2,
\qquad
\gamma_3=4-1=3.
$$

従って、

$$
y^*
=
\begin{pmatrix}
0\\
1\\
2
\end{pmatrix},
\qquad
\gamma
=
\begin{pmatrix}
3\\
0.4\\
3
\end{pmatrix}.
$$

minimum marginは、

$$
\gamma_{\min}
=
\min(3,0.4,3)
=
0.4
$$

である。bottleneckは第2 sampleである。

第2 sampleは1位と2位の差が最も小さい。ただし、誤差がtop classに有利な方向へ働けば、他sampleより先にpredictionが変わるとは限らない。

## 8. なぜminimum marginを使うのか

minimumの定義から、任意のsample $n$について、

$$
\gamma_{\min}
\le
\gamma_n.
$$

$2>0$なので、両辺を2で割っても不等号の向きは変わらない。

$$
\boxed{
\frac{\gamma_{\min}}{2}
\le
\frac{\gamma_n}{2}
\qquad
\text{for all }n=1,\ldots,B
}.
$$

従って、$\gamma_{\min}/2$は、全sampleのsample-wise thresholdのうち最も小さい値である。

具体例では、

$$
\frac{\gamma}{2}
=
\begin{pmatrix}
1.5\\
0.2\\
1.5
\end{pmatrix},
\qquad
\frac{\gamma_{\min}}{2}
=
0.2.
$$

単一のscalar upper boundで全sampleを同時に保証するには、最も小さい$0.2$よりも誤差上界を小さくする必要がある。

## 9. 共通上界 $U$

### 9.1 意味と定義

長い積を、

$$
\boxed{
U
\coloneqq
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
}
$$

と置く。$U$は新しい誤差ではなく、既に導いた上界の右辺に付けた略称である。

### 9.2 Shapeと軸

$U\in\mathbb{R}_{\ge0}$はscalarであり、sample軸もclass軸も持たない。

### 9.3 計算元と用途

$U$はbatch入力 $X$、第1 Linearの重み近似誤差 $\Delta W_1$、第2 Linearの重み $W_2$から計算する。

前段の結果は、

$$
\|\Delta L\|_F
\le
U
$$

であり、さらに任意のsampleについて、

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
U
$$

である。

$U=0.1$とは、実際のbatch logits errorが必ず$0.1$という意味ではない。

$$
\|\Delta L\|_F
\le
0.1
$$

という意味であり、実際には$0.02$など、さらに小さい可能性がある。

## 10. Sample-wise条件からbatch-wide条件へ

次のbatch-wide conditionを仮定する。

$$
\boxed{
U
<
\frac{\gamma_{\min}}{2}
}.
$$

batch内から任意のsample $n$を一つ選ぶ。前段のerror boundより、

$$
\|\delta l_n\|_\infty
\le
U.
$$

仮定より、

$$
U
<
\frac{\gamma_{\min}}{2}.
$$

minimum marginの定義より、

$$
\frac{\gamma_{\min}}{2}
\le
\frac{\gamma_n}{2}.
$$

三つを順につなげると、

$$
\boxed{
\|\delta l_n\|_\infty
\le
U
<
\frac{\gamma_{\min}}{2}
\le
\frac{\gamma_n}{2}
}.
$$

従って、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}.
$$

これはsample $n$のargmaxが変わらない十分条件である。

重要なのは、$n$を特定せず、batch内から任意に選んだことである。同じ議論を全sampleへ適用できるので、

$$
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k
\qquad
\text{for all }n=1,\ldots,B
$$

が同時に成立する。

## 11. 数値で全sampleを確認する

先ほどの、

$$
\gamma
=
\begin{pmatrix}
3\\
0.4\\
3
\end{pmatrix},
\qquad
\frac{\gamma_{\min}}{2}=0.2
$$

に対して、共通上界を、

$$
U=0.1
$$

とする。このとき、

$$
0.1
<
0.2
=
\frac{\gamma_{\min}}{2}.
$$

各sampleについて、

$$
\begin{aligned}
n=1:&\quad
U=0.1<0.2\le1.5=\frac{\gamma_1}{2},\\
n=2:&\quad
U=0.1<0.2\le0.2=\frac{\gamma_2}{2},\\
n=3:&\quad
U=0.1<0.2\le1.5=\frac{\gamma_3}{2}.
\end{aligned}
$$

$n=2$の等号は問題にならない。必要なのは、

$$
U
<
\frac{\gamma_2}{2}
$$

であり、$0.1<0.2$はstrictに成立している。

## 12. Compressed prediction vector $\widetilde y$

### 12.1 意味と定義

sample $n$のcompressed predicted classを、

$$
\widetilde y_n
=
\operatorname*{argmax}_{0\le k<c}(\widetilde l_n)_k
$$

とする。batch prediction vectorは、

$$
\widetilde y
=
\begin{pmatrix}
\widetilde y_1\\
\widetilde y_2\\
\vdots\\
\widetilde y_B
\end{pmatrix}
\in
\{0,\ldots,c-1\}^{B}
$$

である。

### 12.2 Shapeと軸

$\widetilde y$のshapeは$(B,)$であり、唯一の軸はsample軸である。

### 12.3 計算元と用途

$\widetilde y$はcompressed logits $\widetilde L$の各行でargmaxを取って計算する。dense prediction vector $y^*$との全要素比較に使う。

## 13. Prediction vectorとしての最終結論

全sampleで、

$$
\widetilde y_n
=
y_n^*
$$

が成立するので、vectorとして、

$$
\boxed{
\widetilde y
=
y^*
}
$$

と書ける。

$U$を元の積へ戻すと、最終的な十分条件は、

$$
\boxed{
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_{\min}}{2}
}
$$

$$
\boxed{
\Longrightarrow
\quad
\widetilde y
=
y^*
}
$$

である。

## 14. Prediction agreementと正解を区別する

この結論が保証するのは、

$$
\text{compressed prediction}
=
\text{dense prediction}
$$

である。ground-truth labelを$t_n$としていないので、

$$
y_n^*=t_n
$$

や、

$$
\widetilde y_n=t_n
$$

はこの条件からは分からない。

dense modelが誤分類しているsampleでは、certificateはcompressed modelが同じ誤分類を維持することを保証し得る。従って、prediction agreementとclassification accuracyは別の概念である。

## 15. 十分条件であって必要条件ではない

証明した論理は、

$$
U
<
\frac{\gamma_{\min}}{2}
\quad
\Longrightarrow
\quad
\widetilde y=y^*
$$

である。

逆に、

$$
U
\ge
\frac{\gamma_{\min}}{2}
$$

であっても、batch内でprediction changeが必ず起きるとは言えない。言えるのは、この共通上界とminimum marginだけでは全sample同時の保証を出せないということだけである。

保証が成立しなくてもprediction agreementが保たれ得る理由には、次がある。

- $U$が実際の$\|\Delta L\|_F$よりかなり大きい。
- 各sampleのactual error $\|\delta l_n\|_\infty$はさらに小さい。
- logits errorの方向がtop classとcompetitorの順位を壊す方向ではない。
- bottleneck sampleでも、top classに有利な方向へlogitsが動く。
- class全体へのcommon shiftのように、normは大きくてもclass間gapを変えない。

## 16. Tieを含むbatch

あるsampleでdense top logitsが同率なら、

$$
\gamma_n=0.
$$

そのsampleを含むbatchでは、

$$
\gamma_{\min}=0.
$$

$U\ge0$なので、

$$
U
<
\frac{\gamma_{\min}}{2}
=0
$$

は成立しない。これは定理の誤りではなく、tieを持つsampleには正のstability radiusを与えられないことを表す。

## 17. Batch-wideという語の範囲

ここでのbatch-wideは、現在の1つのbatch、

$$
\{1,2,\ldots,B\}
$$

内で全sampleを同時に扱うという意味である。

異なるminibatch、training set全体、test set全体へ同じ$\gamma_{\min}$を流用できるとは限らない。batchが変われば、入力$X$、logits、margin vector、minimum margin、上界$U$も変わる。

## 18. このノートから言えること・言えないこと

言えることは次である。

- minimum marginは、1つのbatch内で最も厳しいsample-wise marginである。
- $U<\gamma_{\min}/2$なら、全sampleで$U<\gamma_n/2$が同時に成立する。
- 既習のsample-wise stabilityを全sampleへ適用し、$\widetilde y=y^*$を保証できる。
- certificateがFalseでも、実測predictionが不一致とは限らない。

このノートだけでは次は言えない。

- dense predictionまたはcompressed predictionがground-truthに対して正しいこと。
- dense accuracy、compressed accuracy、accuracy lower bound。
- dataset全体または異なるminibatchをまたぐ保証。
- 実TT-rank打ち切りにより得た$\Delta W_1$であること。
- rankとaccuracyの関係。
- fine-tuning、training、backward後の性能。

PyTorchでの`argmax`、`topk`、minimumの値とindex、scalar certificateの実装は、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]を参照する。固定toy MLPの保存済み数値、全sampleのthreshold、結果と制約は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]へ分ける。
