---
title: Batch Logits ErrorからPrediction Stability Certificate toy実験の設定 結果 考察
tags:
  - TTLinear
  - logits
  - Frobenius-norm
  - classification-margin
  - certificate
  - toy-experiment
  - results
---

# Batch Logits ErrorからPrediction Stability Certificate：toy実験の設定・結果・考察

## 1. 実験の目的

第1 Linearの重みに小さい固定摂動を加えた2層MLPを使い、batch-levelのlogits errorからsample-wiseなprediction stabilityへ至る次のchainを確認する。

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}.
$$

さらに、dense modelと近似modelのargmaxが実際に一致することを、certificateとは独立に確認する。

一般理論は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]を参照する。PyTorch API、shape、assertの意味は、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]へ分ける。

## 2. 証拠レベル

対象Notebookは、[05_logits_bound_to_margin_certificate.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb)である。

このノートに記録する主要な数値は、Notebookファイルに保存されている出力と現行sourceから確認した。行列積とnorm値は、同じ固定値から独立にも再計算した。

ただし、現行Notebookのcode cellには非連続な`execution_count`と`null`が混在する。教材化時に新しいkernelからRestart Kernel、Run Allを実行した結果ではない。従って、次を区別する。

- 保存済み出力として確認できる数値。
- 現行sourceから決まる行列と式。
- Fresh Run Allによる再現性の確認。

今回確認したのは最初の二つであり、Fresh Run Allは未確認である。

## 3. 実験設定

### 3.1 実行条件

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
sample_index = 2
~~~

`sample_index = 2`は0始まりなので、第3 sampleを指す。

### 3.2 入力batch

$$
X
=
\begin{pmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{pmatrix}
\in
\mathbb{R}^{3\times 2}.
$$

Frobenius normは、

$$
\begin{aligned}
\|X\|_F
&=
\sqrt{1^2+0^2+0^2+1^2+1^2+1^2}\\
&=
\sqrt{4}\\
&=
2.
\end{aligned}
$$

### 3.3 Dense第1 Linear

$$
W_1
=
\begin{pmatrix}
2 & 0\\
0 & 2
\end{pmatrix},
\qquad
b_1
=
\begin{pmatrix}
1\\
1
\end{pmatrix}.
$$

### 3.4 固定した重み摂動

$$
\Delta W_1
=
\begin{pmatrix}
0.010 & -0.005\\
0.004 & 0.008
\end{pmatrix}.
$$

従って、近似重みは、

$$
\begin{aligned}
\widetilde W_1
&=
W_1+\Delta W_1\\
&=
\begin{pmatrix}
2 & 0\\
0 & 2
\end{pmatrix}
+
\begin{pmatrix}
0.010 & -0.005\\
0.004 & 0.008
\end{pmatrix}\\
&=
\begin{pmatrix}
2.010 & -0.005\\
0.004 & 2.008
\end{pmatrix}.
\end{aligned}
$$

この $\Delta W_1$ は誤差伝播を確認するための手入力値であり、実際のTT-SVDまたはTT-roundingから得たものではない。

### 3.5 共通の第2 Linear

$$
W_2
=
\begin{pmatrix}
2 & 0\\
0 & 1.5\\
-1 & -1
\end{pmatrix},
\qquad
b_2
=
\begin{pmatrix}
0\\
0\\
0
\end{pmatrix}.
$$

## 4. Dense forwardを全要素で確認する

### 4.1 第1 Linear

$$
Z_1
=
XW_1^T+\boldsymbol{1}_3b_1^T.
$$

$W_1$ は対角行列なので、

$$
XW_1^T
=
\begin{pmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{pmatrix}
\begin{pmatrix}
2 & 0\\
0 & 2
\end{pmatrix}
=
\begin{pmatrix}
2 & 0\\
0 & 2\\
2 & 2
\end{pmatrix}.
$$

biasを各行へ加えると、

$$
\begin{aligned}
Z_1
&=
\begin{pmatrix}
2 & 0\\
0 & 2\\
2 & 2
\end{pmatrix}
+
\begin{pmatrix}
1 & 1\\
1 & 1\\
1 & 1
\end{pmatrix}\\
&=
\begin{pmatrix}
3 & 1\\
1 & 3\\
3 & 3
\end{pmatrix}.
\end{aligned}
$$

全要素が正なので、ReLU後も変わらない。

$$
H
=
\operatorname{ReLU}(Z_1)
=
\begin{pmatrix}
3 & 1\\
1 & 3\\
3 & 3
\end{pmatrix}.
$$

### 4.2 第2 Linear

$$
L
=
HW_2^T.
$$

$$
W_2^T
=
\begin{pmatrix}
2 & 0 & -1\\
0 & 1.5 & -1
\end{pmatrix}.
$$

第1行は、

$$
\begin{pmatrix}
3 & 1
\end{pmatrix}
W_2^T
=
\begin{pmatrix}
3\cdot2+1\cdot0 &
3\cdot0+1\cdot1.5 &
3\cdot(-1)+1\cdot(-1)
\end{pmatrix}
=
\begin{pmatrix}
6 & 1.5 & -4
\end{pmatrix}.
$$

同様に全行を計算すると、

$$
L
=
\begin{pmatrix}
6 & 1.5 & -4\\
2 & 4.5 & -4\\
6 & 4.5 & -6
\end{pmatrix}.
$$

## 5. 近似modelのforwardを全要素で確認する

### 5.1 第1 Linear

$$
\widetilde Z_1
=
X\widetilde W_1^T+\boldsymbol{1}_3b_1^T.
$$

$$
\widetilde W_1^T
=
\begin{pmatrix}
2.010 & 0.004\\
-0.005 & 2.008
\end{pmatrix}.
$$

まず、

$$
X\widetilde W_1^T
=
\begin{pmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{pmatrix}
\begin{pmatrix}
2.010 & 0.004\\
-0.005 & 2.008
\end{pmatrix}
=
\begin{pmatrix}
2.010 & 0.004\\
-0.005 & 2.008\\
2.005 & 2.012
\end{pmatrix}.
$$

biasを加えると、

$$
\widetilde Z_1
=
\begin{pmatrix}
3.010 & 1.004\\
0.995 & 3.008\\
3.005 & 3.012
\end{pmatrix}.
$$

これも全要素が正なので、

$$
\widetilde H
=
\operatorname{ReLU}(\widetilde Z_1)
=
\begin{pmatrix}
3.010 & 1.004\\
0.995 & 3.008\\
3.005 & 3.012
\end{pmatrix}.
$$

### 5.2 第2 Linear

$$
\widetilde L
=
\widetilde H W_2^T
=
\begin{pmatrix}
6.020 & 1.506 & -4.014\\
1.990 & 4.512 & -4.003\\
6.010 & 4.518 & -6.017
\end{pmatrix}.
$$

例えば第3行は、

$$
\begin{aligned}
\widetilde L_{3,:}
&=
\begin{pmatrix}
3.005 & 3.012
\end{pmatrix}
W_2^T\\
&=
\begin{pmatrix}
3.005\cdot2+3.012\cdot0 &
3.005\cdot0+3.012\cdot1.5 &
3.005\cdot(-1)+3.012\cdot(-1)
\end{pmatrix}\\
&=
\begin{pmatrix}
6.010 & 4.518 & -6.017
\end{pmatrix}.
\end{aligned}
$$

## 6. Batch logits error

$$
\Delta L
=
\widetilde L-L.
$$

全要素を引くと、

$$
\begin{aligned}
\Delta L
&=
\begin{pmatrix}
6.020 & 1.506 & -4.014\\
1.990 & 4.512 & -4.003\\
6.010 & 4.518 & -6.017
\end{pmatrix}
-
\begin{pmatrix}
6 & 1.5 & -4\\
2 & 4.5 & -4\\
6 & 4.5 & -6
\end{pmatrix}\\
&=
\begin{pmatrix}
0.020 & 0.006 & -0.014\\
-0.010 & 0.012 & -0.003\\
0.010 & 0.018 & -0.017
\end{pmatrix}.
\end{aligned}
$$

選んだ第3 sampleのerrorは、

$$
\delta l_3
=
\begin{pmatrix}
0.010\\
0.018\\
-0.017
\end{pmatrix}.
$$

行列 $\Delta L$ の第3行は $\delta l_3^T$ である。

Notebookに保存されたPyTorchの表示は、

~~~text
[0.009999999999999787, 0.017999999999999794, -0.01699999999999946]
~~~

である。これは2進浮動小数点による表示上の丸めであり、上の $0.010$、$0.018$、$-0.017$ と同じ計算結果を表す。

## 7. Norm chainの数値確認

### 7.1 Sample infinity norm

$$
\begin{aligned}
\|\delta l_3\|_\infty
&=
\max(0.010,0.018,0.017)\\
&=
0.01800000.
\end{aligned}
$$

### 7.2 Sample vector 2-norm

$$
\begin{aligned}
\|\delta l_3\|_2
&=
\sqrt{0.010^2+0.018^2+(-0.017)^2}\\
&=
\sqrt{0.000100+0.000324+0.000289}\\
&=
\sqrt{0.000713}\\
&\approx
0.02670206.
\end{aligned}
$$

### 7.3 Batch Frobenius norm

$$
\begin{aligned}
\|\Delta L\|_F^2
&=
0.020^2+0.006^2+(-0.014)^2\\
&\quad+
(-0.010)^2+0.012^2+(-0.003)^2\\
&\quad+
0.010^2+0.018^2+(-0.017)^2\\
&=
0.000632+0.000253+0.000713\\
&=
0.001598.
\end{aligned}
$$

従って、

$$
\|\Delta L\|_F
=
\sqrt{0.001598}
\approx
0.03997499.
$$

### 7.4 重みから得る上界

独立再計算したspectral normは、

$$
\|\Delta W_1\|_2
\approx
0.01118034,
$$

$$
\|W_2\|_2
\approx
2.33533043.
$$

従って、

$$
\begin{aligned}
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
&\approx
2.0
\times
0.01118034
\times
2.33533043\\
&\approx
0.05221958.
\end{aligned}
$$

保存済みNotebook出力のchainは、

~~~text
0.01800000 <= 0.02670206 <= 0.03997499 <= 0.05221958
~~~

である。数式では、

$$
0.01800000
\le
0.02670206
\le
0.03997499
\le
0.05221958.
$$

## 8. 第3 sampleのclassification margin

dense logitsは、

$$
l_3
=
\begin{pmatrix}
6.0\\
4.5\\
-6.0
\end{pmatrix}.
$$

top classはclass 0、strongest competitorはclass 1である。

$$
y_3^*
=
0,
\qquad
k_3^{\mathrm{comp}}
=
1.
$$

classification marginは、

$$
\begin{aligned}
\gamma_3
&=
(l_3)_0-(l_3)_1\\
&=
6.0-4.5\\
&=
1.5.
\end{aligned}
$$

従って、thresholdは、

$$
\frac{\gamma_3}{2}
=
\frac{1.5}{2}
=
0.75.
$$

上流の理論上界も、このthresholdより小さい。

$$
0.05221958
<
0.75000000.
$$

全体を一列につなげると、

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
}.
$$

## 9. Certificateと実測prediction

近似modelの第3 sample logitsは、

$$
\widetilde l_3
=
\begin{pmatrix}
6.010\\
4.518\\
-6.017
\end{pmatrix}.
$$

保存済みNotebook出力は、

~~~text
certificate           = True
dense prediction      = 0
compressed prediction = 0
~~~

である。

上流のnorm boundが、

$$
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_3}{2}
$$

を満たしたので、理論上sample 3のpredictionは安定である。実際のargmaxも、

$$
\operatorname*{argmax}_k(l_3)_k
=
0,
$$

$$
\operatorname*{argmax}_k(\widetilde l_3)_k
=
0
$$

で一致した。

## 10. 結果の考察

### 10.1 Batch-level解析とsample-wise判定が接続した

前段で得た $\|\Delta L\|_F$ と重み誤差上界はbatch全体の量であり、そのままでは特定sampleのclass順位を表さない。今回、

$$
\|\delta l_3\|_\infty
\le
\|\delta l_3\|_2
\le
\|\Delta L\|_F
$$

を数値で確認し、batch-levelの誤差解析をsample-wise marginへ接続できた。

### 10.2 各上界で値が大きくなる

実測した最大class誤差は、

$$
0.01800000
$$

である。一方、最も上流の理論上界は、

$$
0.05221958
$$

である。上界は実測値の約 $2.90$ 倍である。

これは誤りではない。vector 2-norm、他sampleも含むFrobenius norm、各行列の最大増幅率を順に使うことで、方向に関する情報を捨てて安全側へ広げた結果である。

### 10.3 このtoy設定ではmarginに大きな余裕がある

$$
0.05221958
\ll
0.75
$$

なので、この例のcertificateは十分な余裕を持って成立する。certificateの成立を教材として確認しやすい設定になっている。

ただし、marginに大きな余裕がある単一toy例だけでは、実データでcertificateがどの程度成立するかは分からない。学習済みmodelではsampleごとにmarginが異なり、small-margin sampleほど保証が難しくなる。

### 10.4 Certificateとprediction一致は別の測定値である

今回、

- `certificate=True`
- dense/compressed predictionがともにclass 0

を別々に確認した。certificateがTrueならprediction一致は理論から従うが、実験記録では保証条件と実測結果を分けて残す。

### 10.5 ReLUの非線形性が表面化しない設定である

dense・近似の第1 Linear出力がすべて正なので、今回のReLUは恒等写像として働く。

$$
H=Z_1,
\qquad
\widetilde H=\widetilde Z_1.
$$

従って、この例はReLU境界をまたぐケースの難しさを検証していない。確認しているのは、既に導いた1-Lipschitz上界を含むchainとmargin certificateである。

## 11. この実験から言えること

- 固定した $3\times2$ 入力と2層MLPで、dense・近似logitsを全要素まで計算できた。
- 第3 sampleのvector infinity norm、vector 2-norm、batch Frobenius norm、重みから得る理論上界の順序を確認できた。
- 最も上流の理論上界がclassification marginの半分より小さく、sample-wise certificateが成立した。
- dense modelと近似modelのargmaxもclass 0で一致した。
- certificateと実測prediction一致を別の量として記録できた。

## 12. この実験から言えないこと

今回の $\Delta W_1$ は手入力したproxyであり、次はまだ示していない。

- 特定のTT-rankで $W_1$ を分解・打ち切りした結果。
- TT-SVDまたはTT-roundingで得た実際の $\Delta W_1$。
- batch内全sampleに対する同時certificate。
- minimum marginを使ったbatch全体またはdataset全体の保証。
- Fashion-MNIST test setのprediction一致率とaccuracy。
- rank、parameter数、MACs、誤差、accuracyのsweep。
- fine-tuning後の性能。
- ReLUの符号反転を含むケースでのtightness。

また、Notebookに保存された出力は確認したが、教材化時のFresh Run All結果ではない。

## 13. 次の段階との境界

このtoy実験で確立したのは、1 sampleについて、

$$
\text{weight error bound}
\longrightarrow
\text{batch logits error bound}
\longrightarrow
\text{sample logits error bound}
\longrightarrow
\text{prediction stability certificate}
$$

と接続する方法である。

直後の理論段階では、minimum marginを導入し、同じ共通上界からbatch内の全sampleを同時に保証する。[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]を参照する。

その後に実データへ進む場合は、学習済みFashion-MNIST MLPの第1 Linearを実際にrank打ち切りし、sampleごとの、

$$
\gamma_n,
\qquad
\|\delta l_n\|_\infty,
\qquad
\operatorname*{argmax}(\widetilde l_n)
=
\operatorname*{argmax}(l_n)
$$

を記録する。accuracy、dataset全体の保証、rank sweepは別段階として扱い、この教材の結果へ混ぜない。
