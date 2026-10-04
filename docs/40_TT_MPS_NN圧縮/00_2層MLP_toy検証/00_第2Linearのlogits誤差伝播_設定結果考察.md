---
title: 第2Linearのlogits誤差伝播 toy実験の設定 結果 考察
tags:
  - TTLinear
  - MLP
  - toy-experiment
  - logits
  - error-propagation
  - results
---

# 第2Linearのlogits誤差伝播：toy実験の設定・結果・考察

## 1. 実験の目的

第1 Linearの重み誤差が、

$$
\Delta W_1
\rightarrow
\Delta Z_1
\rightarrow
\Delta H
\rightarrow
\Delta L
$$

と伝播する過程を、小さい2層MLPで数値確認する。

確認対象は、次の厳密な恒等式とnorm上界である。

$$
\Delta Z_1
=
X\Delta W_1^{\mathsf T},
$$

$$
\Delta L
=
\Delta H W_2^{\mathsf T},
$$

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F,
$$

$$
\|\Delta L\|_F
\le
\|\Delta H\|_F\|W_2\|_2,
$$

$$
\boxed{
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
}.
$$

理論導出は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]を参照する。

## 2. 証拠レベル

対象Notebookは、[03_second_linear_logits_error_propagation.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/03_second_linear_logits_error_propagation.ipynb) である。

このノートに記録する数値は、Notebookファイルに残っている保存済み出力である。教材化時に新しいkernelからRestart Kernel、Run Allを行った結果ではない。

特にperturbation-scale cellは、現行ソースと保存済み表が一致していない。

- 現行ソース：scale 0の比率を`NaN`にする。
- 保存済み表：scale 0の比率が旧値`0.0`のまま。

従って、現行ソースのFresh Run All成功とは扱わない。

## 3. 実験設定

### 3.1 実行条件

現行Notebookでは次を用いる。

~~~text
torch.manual_seed(0)
dtype = torch.float64
device = CPU
batch size B = 4
input dimension d = 6
hidden dimension h = 5
output/class dimension c = 3
~~~

`nn.Linear`は使わず、行列積を直接記述する。

$$
Z_1
=
XW_1^{\mathsf T}+b_1,
\qquad
H
=
\operatorname{ReLU}(Z_1),
$$

$$
L
=
HW_2^{\mathsf T}+b_2.
$$

### 3.2 tensorのshape

| 変数 | 数式 | shape | 軸の意味 |
| --- | --- | ---: | --- |
| `x` | $X$ | `(4, 6)` | batch、input feature |
| `w1` | $W_1$ | `(5, 6)` | hidden unit、input feature |
| `b1` | $b_1$ | `(5,)` | hidden unit |
| `w2` | $W_2$ | `(3, 5)` | output/class、hidden unit |
| `b2` | $b_2$ | `(3,)` | output/class |
| `z1`, `z1_tilde` | $Z_1,\widetilde Z_1$ | `(4, 5)` | batch、hidden unit |
| `h_dense`, `h_tilde` | $H,\widetilde H$ | `(4, 5)` | batch、hidden unit |
| `logits`, `logits_tilde` | $L,\widetilde L$ | `(4, 3)` | batch、output/class |

### 3.3 重み誤差の作り方

単一設定では、小さいrandomな重み誤差を作り、

$$
\widetilde W_1
=
W_1+\Delta W_1
$$

とする。符号規約は

$$
\Delta W_1
=
\widetilde W_1-W_1
$$

である。

scale sweepでは、固定したrandom方向 $E$ を1回だけ生成し、

$$
\Delta W_1(\alpha)
=
\alpha E
$$

とする。使用するscaleは、

$$
\alpha
\in
\{0,0.01,0.03,0.05,0.1,0.2\}
$$

である。

$X,W_1,b_1,W_2,b_2,E$ は固定し、$\alpha$ だけを変える。これにより、誤差方向とモデルの違いを混入させず、重み誤差の大きさだけを比較する。

現行Notebookの $E$ は`torch.randn`で作った値をそのまま使い、norm 1へ正規化していない。従って、$\alpha$ は摂動方向へ掛ける係数であり、$\|\Delta W_1\|_F$または$\|\Delta W_1\|_2$そのものではない。

## 4. 単一摂動での結果

### 4.1 第2 Linearの恒等式

実forwardの差

$$
\Delta L
=
\widetilde L-L
$$

と、理論式から別計算した

$$
\Delta H W_2^{\mathsf T}
$$

の最大絶対差は、保存済み出力で

$$
\max_{n,k}
\left|
(\Delta L)_{n,k}
-
(\Delta H W_2^{\mathsf T})_{n,k}
\right|
=
3.608224830032\times10^{-16}
$$

だった。float64の丸め誤差水準で、

$$
\Delta L
=
\Delta H W_2^{\mathsf T}
$$

が成立している。

### 4.2 spectral norm

保存済み結果は、

~~~text
||W2||_2           = 4.100457731042
max singular value = 4.100457731042
~~~

である。従って、数値的にも

$$
\|W_2\|_2
=
\sigma_{\max}(W_2)
$$

を確認した。

### 4.3 第2 Linearの上界

保存済み結果は、

~~~text
||Delta_H||_F          = 0.3306492197740
||W2||_2               = 4.100457731042
||Delta_L||_F          = 0.4669371802503
||Delta_H||_F ||W2||_2 = 1.355813149485
~~~

である。従って、

$$
0.4669371802503
\le
1.355813149485
$$

となり、第2 Linearの上界を満たす。

実測値と第2 Linear上界の比は、

$$
\frac{
0.4669371802503
}{
1.355813149485
}
\approx
0.344396
$$

である。

### 4.4 第1 Linearからの合成上界

保存済み結果は次である。

~~~text
||Delta_Z1||_F                          = 0.4662638322511
||X||_F ||Delta_W1||_2                  = 0.7309086424606
||Delta_H||_F                           = 0.3306492197740
||Delta_L||_F                           = 0.4669371802503
||X||_F ||Delta_W1||_2 ||W2||_2         = 2.997059993663
~~~

従って、

$$
\|\Delta L\|_F
=
0.4669371802503
\le
2.997059993663
$$

である。実測値と合成上界の比は、

$$
\frac{
0.4669371802503
}{
2.997059993663
}
\approx
0.155798
$$

となる。

## 5. perturbation-scale sweepの結果

保存済み表は次である。

| scale | $\|\Delta W_1\|_2$ | $\|\Delta Z_1\|_F$ | $\|\Delta H\|_F$ | $\|\Delta L\|_F$ | full upper bound | saved ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.01 | 0.047625 | 0.115559 | 0.059680 | 0.155727 | 0.994283 | 0.156622 |
| 0.03 | 0.142876 | 0.346676 | 0.179041 | 0.467181 | 2.982848 | 0.156622 |
| 0.05 | 0.238127 | 0.577793 | 0.298401 | 0.778634 | 4.971414 | 0.156622 |
| 0.10 | 0.476254 | 1.155586 | 0.596802 | 1.557269 | 9.942828 | 0.156622 |
| 0.20 | 0.952507 | 2.311172 | 1.193608 | 3.117233 | 19.885655 | 0.156758 |

保存済みグラフでは、横軸をperturbation scale、縦軸を $\|\Delta L\|_F$ と合成上界としている。両者はほぼ直線的に増加し、上界は全scaleで実測値より大きい。

## 6. 結果の考察

### 6.1 恒等式の確認が意味すること

最大絶対差が約 $3.61\times10^{-16}$ だったことは、第2 Linearのforward差と

$$
\Delta H W_2^{\mathsf T}
$$

がfloat64の丸め誤差水準で一致したことを示す。これにより、少なくともこのtoy例では、誤差の符号、転置、biasの打消し、batch/class軸の扱いが整合している。

### 6.2 scaleに対して上界が比例する理由

固定方向 $E$ に対して

$$
\Delta W_1(\alpha)=\alpha E
$$

なので、normの斉次性から

$$
\|\Delta W_1(\alpha)\|_2
=
|\alpha|\|E\|_2.
$$

従って、

$$
\|X\|_F
\|\Delta W_1(\alpha)\|_2
\|W_2\|_2
$$

も $|\alpha|$ に正比例する。表で上界がscaleとともに直線的に増えるのは、理論どおりである。

### 6.3 実測logits誤差もほぼ比例する理由

第1 Linearの差は、

$$
\Delta Z_1(\alpha)
=
X\Delta W_1(\alpha)^{\mathsf T}
=
\alpha XE^{\mathsf T}
$$

なので、$\alpha$ に厳密に比例する。

ReLUの正負maskが変わらない範囲では、ReLUは固定された線形maskとして働く。その範囲では $\Delta H$ と $\Delta L$ も $\alpha$ に比例する。そのため、scale 0.01から0.10まで、実測値と上界の比が約`0.156622`でほぼ一定になったと解釈できる。

### 6.4 scale 0.20の小さいずれ

scale 0.10の実測値を単純に2倍すると、

$$
2\times1.557269
=
3.114538
$$

である。一方、scale 0.20の保存値は

$$
3.117233
$$

であり、差は約

$$
0.002695
$$

である。

摂動が大きくなって一部のpre-activationが0を横切り、ReLUの正負maskが変化した可能性がある。ただし、保存済みのnormだけでは原因を確定できない。確認にはscaleごとのmask変化数が必要である。

### 6.5 上界が実測値より大きい理由

合成上界へ近づくには、次が同時に成立する必要がある。

1. $X$ が $\Delta W_1$ の最大増幅方向と整合する。
2. ReLUが誤差を減衰させない。
3. $\Delta H$ が $W_2$ の最大右特異ベクトル方向と整合する。

このtoy例では、scale 0.10で段階ごとの比は概算で

$$
\frac{
\|\Delta Z_1\|_F
}{
\|X\|_F\|\Delta W_1\|_2
}
\approx
0.477,
$$

$$
\frac{
\|\Delta H\|_F
}{
\|\Delta Z_1\|_F
}
\approx
0.516,
$$

$$
\frac{
\|\Delta L\|_F
}{
\|\Delta H\|_F\|W_2\|_2
}
\approx
0.636
$$

だった。積は

$$
0.477\times0.516\times0.636
\approx
0.1566
$$

であり、全体の実測値／上界比と一致する。上界の緩さは一つの不等式だけでなく、3段階の余裕が積み重なった結果である。

scale 0.10での実際の第2 Linear増幅率は、

$$
\frac{
\|\Delta L\|_F
}{
\|\Delta H\|_F
}
=
\frac{1.557269}{0.596802}
\approx
2.61
$$

である。これは最大増幅率

$$
\|W_2\|_2
\approx
4.10
$$

より小さい。実際のhidden errorが最大右特異ベクトル方向へ完全には整列していないことと整合する。

### 6.6 上界が約6.4倍でも失敗ではない

scale 0.01から0.10では、実測値／上界比が約`0.1566`なので、上界は実測値の約

$$
\frac{1}{0.1566}
\approx
6.4
$$

倍である。

この上界は特定batchに対する平均的な予測ではなく、各段階で最大増幅方向を使うworst-case boundである。上界が実測値より大きいこと自体は失敗ではない。一方、rank選択に利用するときは、上界だけでなく実測logits誤差も併記し、どの程度looseかを確認する必要がある。

### 6.7 scale 0の比率

$\alpha=0$ では、

$$
\Delta W_1=0,
\qquad
\Delta L=0,
\qquad
\text{upper bound}=0.
$$

比率は

$$
\frac{0}{0}
$$

なので未定義である。現行コードが`NaN`を記録するのが数学的に正しく、保存済み表の`0.000000`は修正前の便宜値である。

## 7. この実験から言えること

- 第2 Linearの誤差恒等式を実forward差で確認できた。
- spectral normと最大特異値が一致した。
- 第2 Linear上界と全体上界が保存済み全scaleで成立した。
- 固定した誤差方向では、誤差scaleとlogits誤差・上界がほぼ比例した。
- worst-case上界と実測誤差の乖離を段階別に分解できた。

## 8. この実験から言えないこと

今回の $\Delta W_1$ は人工的に作った誤差である。従って、次はまだ示していない。

- TT-SVDまたはTT-roundingで実際に得た近似重み。
- TT-rankと $\|\Delta W_1\|$ の関係。
- TT-rank、parameter数、MACs、latencyのtrade-off。
- Fashion-MNIST test accuracy。
- predictionの維持。
- classification marginによるargmax不変条件。
- TT coreのfine-tuning。

実際のTT圧縮では、rank $r$ ごとに先に

$$
\widetilde W_1(r)
$$

を作り、その後に

$$
\Delta W_1(r)
=
\widetilde W_1(r)-W_1
$$

を測る。今回のscale sweepとは因果の向きが逆である。

## 9. 次の実験への接続

次は、実際のTT-rank候補ごとに次を記録する。

- TT-rank。
- TT parameter数と圧縮率。
- $\|\Delta W_1\|_F$ と $\|\Delta W_1\|_2$。
- $\|\Delta H\|_F$。
- $\|\Delta L\|_F$。
- 合成上界と実測値／上界比。
- sampleごとのclassification margin。
- prediction一致率とtest accuracy。

学習済みFashion-MNIST MLPへの適用は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]で扱う。

