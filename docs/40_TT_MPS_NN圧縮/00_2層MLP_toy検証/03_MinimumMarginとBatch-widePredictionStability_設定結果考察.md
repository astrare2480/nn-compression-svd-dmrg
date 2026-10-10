---
title: Minimum MarginとBatch-wide Prediction Stability toy実験の設定 結果 考察
tags:
  - TTLinear
  - logits
  - minimum-margin
  - batch-wide-certificate
  - toy-experiment
  - results
---

# Minimum MarginとBatch-wide Prediction Stability：toy実験の設定・結果・考察

## 1. 実験の目的

固定した2層MLPと第1 Linearの重み摂動を使い、batch内の全sampleについて、

$$
\|\delta l_n\|_\infty
\le
U
<
\frac{\gamma_{\min}}{2}
\le
\frac{\gamma_n}{2}
$$

が同時に成立することを確認する。

そのうえで、

$$
\widetilde y
=
y^*
$$

を実測値でも確認する。

一般理論は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]を参照する。PyTorch APIと実装上の訂正は、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]へ分ける。

## 2. 証拠レベル

対象Notebookは、[06_minimum_margin_batchwide_stability.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/06_minimum_margin_batchwide_stability.ipynb)である。

このノートには、現行Notebookのsource、保存済み出力、実行済みassert、および固定値から直接計算できる値を記録する。

ただし、code cellの`execution_count`は17から46まで非連続である。教材化時に新しいkernelからRestart Kernel、Run Allを実行した結果ではない。従って、次を区別する。

- 現行sourceと保存済み出力は確認済み。
- 保存された実行順でassertにerror outputがないことは確認済み。
- 現行sourceを先頭から同一kernelでFresh Run Allしたことは未確認。

## 3. 実験設定

実行条件は次である。

~~~text
torch.manual_seed(0)
dtype = torch.float64
device = CPU
B = 3
d = 2
h = 2
c = 3
rtol = 1e-10
atol = 1e-12
eps = 1e-12
~~~

入力、重み、固定摂動は前段のtoy実験と同じである。

$$
X
=
\begin{pmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{pmatrix},
\qquad
\Delta W_1
=
\begin{pmatrix}
0.010 & -0.005\\
0.004 & 0.008
\end{pmatrix},
$$

$$
W_2
=
\begin{pmatrix}
2 & 0\\
0 & 1.5\\
-1 & -1
\end{pmatrix}.
$$

$\Delta W_1$は実際のTT-SVDまたはTT-roundingから得た値ではなく、誤差伝播を確認するための手入力proxyである。第1 Linearからlogitsまでの全要素計算は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]に記録している。

## 4. Dense logitsとprediction vector

Dense logitsは、

$$
L
=
\begin{pmatrix}
6.0 & 1.5 & -4.0\\
2.0 & 4.5 & -4.0\\
6.0 & 4.5 & -6.0
\end{pmatrix}.
$$

Notebookの保存済み表示は、

~~~text
[[ 6.   1.5 -4. ]
 [ 2.   4.5 -4. ]
 [ 6.   4.5 -6. ]]
n=0: [6.0, 1.5, -4.0]
n=1: [2.0, 4.5, -4.0]
n=2: [6.0, 4.5, -6.0]
~~~

である。

各行でargmaxを取ると、

$$
y^*
=
\begin{pmatrix}
0\\
1\\
0
\end{pmatrix}.
$$

## 5. Top-2 valuesとclass indices

各sampleの上位2 logit値は、

$$
\operatorname{top2values}(L)
=
\begin{pmatrix}
6.0 & 1.5\\
4.5 & 2.0\\
6.0 & 4.5
\end{pmatrix}.
$$

対応するclass indicesは、

$$
\operatorname{top2indices}(L)
=
\begin{pmatrix}
0 & 1\\
1 & 0\\
0 & 1
\end{pmatrix}.
$$

1列目はtop-1、2列目はtop-2に対応する。値とclass indexを混同しない。

## 6. Margin vectorを全sampleで計算する

第1 sampleは、

$$
\gamma_1
=
6.0-1.5
=
4.5.
$$

第2 sampleは、

$$
\gamma_2
=
4.5-2.0
=
2.5.
$$

第3 sampleは、

$$
\gamma_3
=
6.0-4.5
=
1.5.
$$

従って、

$$
\gamma
=
\begin{pmatrix}
4.5\\
2.5\\
1.5
\end{pmatrix}.
$$

全要素が正なので、このbatchでは各sampleのdense top classは一意である。

## 7. Minimum marginとbottleneck sample

$$
\begin{aligned}
\gamma_{\min}
&=
\min(4.5,2.5,1.5)\\
&=
1.5.
\end{aligned}
$$

従って、

$$
\frac{\gamma_{\min}}{2}
=
0.75.
$$

0始まりのbottleneck sample indexは、

$$
n_{\mathrm{bottleneck}}
=
2
$$

であり、第3 sampleに対応する。

保存済み出力は次である。

~~~text
||Delta L||_F       = 0.03997499
U                    = 0.05221958
gamma                = [4.5, 2.5, 1.5]
gamma_min             = 1.50000000
gamma_min / 2         = 0.75000000
bottleneck sample idx = 2
~~~

## 8. 上流の共通上界

前段と同じ上界は、

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
\approx
0.05221958
$$

である。

実測batch logits errorは、

$$
\|\Delta L\|_F
\approx
0.03997499
$$

なので、

$$
0.03997499
\le
0.05221958
$$

が成立する。

## 9. Sample-wise thresholdを全要素で比較する

各sampleのthresholdは、

$$
\frac{\gamma}{2}
=
\begin{pmatrix}
2.25\\
1.25\\
0.75
\end{pmatrix}.
$$

minimum thresholdは、

$$
\frac{\gamma_{\min}}{2}
=
0.75.
$$

従って、

$$
0.75
\le
\begin{pmatrix}
2.25\\
1.25\\
0.75
\end{pmatrix}
$$

が要素ごとに成立する。

共通上界をつなぐと、第1 sampleでは、

$$
0.05221958
<
0.75
\le
2.25,
$$

第2 sampleでは、

$$
0.05221958
<
0.75
\le
1.25,
$$

第3 sampleでは、

$$
0.05221958
<
0.75
\le
0.75
$$

である。従って、

$$
U
<
\frac{\gamma_n}{2}
\qquad
\text{for all }n=1,2,3
$$

が成立する。

## 10. Actual sample errorも確認する

Batch logits errorは、

$$
\Delta L
=
\begin{pmatrix}
0.020 & 0.006 & -0.014\\
-0.010 & 0.012 & -0.003\\
0.010 & 0.018 & -0.017
\end{pmatrix}.
$$

各sampleのinfinity normは、

$$
\|\delta l_1\|_\infty
=
0.020,
$$

$$
\|\delta l_2\|_\infty
=
0.012,
$$

$$
\|\delta l_3\|_\infty
=
0.018.
$$

従って、各sampleで、

$$
\|\delta l_n\|_\infty
\le
U
=
0.05221958
$$

も成立している。

全chainは、

$$
\boxed{
\|\delta l_n\|_\infty
\le
0.05221958
<
0.75
\le
\frac{\gamma_n}{2}
\qquad
\text{for all }n=1,2,3
}.
$$

## 11. Compressed logitsとprediction vector

Compressed logitsは、

$$
\widetilde L
=
\begin{pmatrix}
6.020 & 1.506 & -4.014\\
1.990 & 4.512 & -4.003\\
6.010 & 4.518 & -6.017
\end{pmatrix}.
$$

各行でargmaxを取ると、

$$
\widetilde y
=
\begin{pmatrix}
0\\
1\\
0
\end{pmatrix}.
$$

従って、

$$
\boxed{
\widetilde y
=
y^*
=
\begin{pmatrix}
0\\
1\\
0
\end{pmatrix}
}
$$

である。

現行Notebookでは、`torch.equal(y_tilde, y_star)`を含む検証cellにerror outputは保存されていない。

## 12. Certificate実装の訂正

初期実装で確認された誤った式は、

~~~python
batch_certificate = torch.all(
    full_upper_bound / 2 <= gamma_min / 2
)
~~~

であった。これは、

$$
\frac{U}{2}
\le
\frac{\gamma_{\min}}{2}
\quad
\Longleftrightarrow
\quad
U
\le
\gamma_{\min}
$$

を判定しており、目的の条件とは異なる。

正しい式は、

~~~python
batch_certificate = full_upper_bound < gamma_min / 2
~~~

である。

現行Notebook sourceでは、

~~~python
batch_certificate = torch.all(
    full_upper_bound < gamma_min / 2
)
~~~

へ訂正済みである。括弧内はscalar boolなので`torch.all`は不要だが、結果は正しい。

今回の値では、

$$
0.05221958
<
0.75
$$

なので、

$$
\text{batch certificate}
=
\mathrm{True}
$$

である。

初期の誤った条件も今回の数値ではTrueになるため、assertが通ることだけでは式の誤りを発見できなかった。

例えば、

$$
\gamma_{\min}=1.5,
\qquad
U=1.0
$$

なら、誤った条件は、

$$
1.0
\le
1.5
$$

でTrueだが、正しい条件は、

$$
1.0
<
0.75
$$

でFalseになる。

## 13. 結果の考察

### 13.1 Bottleneck sampleがbatch certificateを決める

第1・第2 sampleのthresholdはそれぞれ$2.25$、$1.25$である。しかし全sampleへ共通の$U$を使う場合、第3 sampleの最小threshold$0.75$に合わせる必要がある。

これは第3 sampleが必ず最初にprediction changeを起こすという意味ではない。marginだけを使う共通保証において、最も厳しいという意味である。

### 13.2 Certificateと実測一致は別の量である

この実験では、

- `batch_certificate=True`
- `all_predictions_match=True`

の両方が成立した。前者はnorm上界から得た保証、後者は具体的なlogitsに対する実測である。

### 13.3 上界はactual errorより大きい

$$
\|\Delta L\|_F
=
0.03997499
<
U
=
0.05221958.
$$

$U$は実際の誤差ではなく、安全側の共通上界である。

### 13.4 Marginに大きな余裕がある

$$
U
=
0.05221958
\ll
0.75
=
\frac{\gamma_{\min}}{2}.
$$

このtoy settingはbatch certificateが明確に成立するように、$\Delta W_1$を小さく設定している。単一toy例で実データにおけるcertificate成立率を示したわけではない。

## 14. この実験から言えること

- 3 sampleのdense prediction vectorは$(0,1,0)^T$である。
- Margin vectorは$(4.5,2.5,1.5)^T$である。
- Minimum marginは$1.5$、bottleneck sample indexは2である。
- 共通上界$U=0.05221958$は$\gamma_{\min}/2=0.75$より小さい。
- 全sampleで$U<\gamma_n/2$が同時に成立する。
- Compressed prediction vectorも$(0,1,0)^T$であり、dense prediction vectorと一致する。
- 現行sourceではbatch certificateの比較式が訂正済みである。

## 15. この実験から言えないこと

- Predictionがground-truth labelに対して正しいこと。
- Dense accuracyまたはcompressed accuracy。
- Dataset全体、異なるminibatchをまたぐprediction agreement。
- 特定TT-rankでの実圧縮結果。
- Rankとaccuracy、parameter数、MACsの関係。
- Fine-tuning、training、backward後の性能。

また、Fresh Run Allを行っていないため、現行source全体の再実行完了を証明した結果ではない。

## 16. 次段階との境界

この段階で、1つのbatch内のprediction agreementに関する理論は閉じる。

次に進む場合はground-truth labelを導入し、次を区別する。

- Dense modelのaccuracy。
- Compressed modelのaccuracy。
- Dense/compressed prediction agreement。
- Batch certificateからaccuracyについて何が言えて、何が言えないか。

現行Notebook末尾にはTT-rank / bond dimension sweepへ進む記載があるが、この教材系列で確定した直後の学習テーマとは一致しない。今回の実験結果には影響しないためNotebookは変更せず、次段階の境界だけをここで明確にする。
