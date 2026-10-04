---
title: Classification MarginとArgmax安定性 toy実験の設定 結果 考察
tags:
  - TTLinear
  - logits
  - classification-margin
  - argmax
  - toy-experiment
  - results
---

# Classification MarginとArgmax安定性：toy実験の設定・結果・考察

## 1. 実験の目的

1 sampleのdense logitsと手で与えたlogits perturbationを使い、次を数値確認する。

$$
\gamma_n
=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k,
$$

$$
\varepsilon_n
=
\|\delta l_n\|_\infty,
$$

$$
\varepsilon_n
<
\frac{\gamma_n}{2}
\quad\Longrightarrow\quad
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k.
$$

また、top classを下げ、strongest competitorを上げるworst-case方向ではgapが $2\varepsilon_n$ 縮むこと、common shiftでは十分条件を満たさなくてもargmaxが変わらないことを確認する。

一般理論は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]を参照する。PyTorch APIと各行の意味は、[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]へ分ける。

## 2. 証拠レベル

対象Notebookは、[04_classification_margin_argmax_stability.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/04_classification_margin_argmax_stability.ipynb)である。

このノートに記録する数値は、Notebookファイルに保存されている出力と、保存された現行source・`execution_count`・assertから確認した値である。教材化時に新しいkernelからRestart Kernel、Run Allを実行した結果ではない。

現行Notebookのcode cellには`execution_count` 25から35が連続して保存され、error outputは残っていない。ただし、execution countはFresh Run Allの証明ではないため、「現行ファイルを新しいkernelで再実行済み」とは判定しない。

## 3. 実験設定

### 3.1 実行条件

~~~text
torch.manual_seed(0)
dtype = torch.float64
device = CPU
class count c = 5
sample count = 1
rtol = 1e-10
atol = 1e-12
~~~

入力は学習済みmodelから得たlogitsではなく、手で追えるように固定した1 sample分のtoy logitsである。

$$
l_n
=
\begin{pmatrix}
2.4\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix}.
$$

shapeは、

$$
l_n\in\mathbb{R}^{5}
$$

であり、PyTorchでは`(5,)`である。

### 3.2 このNotebookでの「compressed logits」

NotebookではTT-SVD、TT-matrix、TTLinear forwardを実行せず、logits errorを直接与えて、

$$
\widetilde l_n
=
l_n+\delta l_n
$$

とする。従って、これはclassification marginの測定系を検証するtoy実験であり、特定のTT-rankで圧縮したmodelの結果ではない。

## 4. Dense predictionとclassification margin

保存済み出力は次である。

~~~text
predicted class  = 0
competitor class = 1
top logit        = 2.400000
competitor       = 1.600000
margin gamma_n   = 0.800000
~~~

数式では、

$$
y_n^*
=
0,
\qquad
k_n^{\mathrm{comp}}
=
1,
$$

$$
\begin{aligned}
\gamma_n
&=
(l_n)_0-(l_n)_1\\
&=
2.4-1.6\\
&=
0.8.
\end{aligned}
$$

従って、十分条件のthresholdは、

$$
\frac{\gamma_n}{2}
=
\frac{0.8}{2}
=
0.4
$$

である。

## 5. Safe case

Notebookで与えたerrorは、

$$
\delta l_n^{\mathrm{safe}}
=
\begin{pmatrix}
-0.20\\
0.15\\
-0.10\\
0.05\\
0.10
\end{pmatrix}.
$$

compressed logitsは、

$$
\begin{aligned}
\widetilde l_n^{\mathrm{safe}}
&=
\begin{pmatrix}
2.4\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix}
+
\begin{pmatrix}
-0.20\\
0.15\\
-0.10\\
0.05\\
0.10
\end{pmatrix}\\
&=
\begin{pmatrix}
2.2\\
1.75\\
0.10\\
-0.45\\
1.30
\end{pmatrix}.
\end{aligned}
$$

誤差normは、

$$
\begin{aligned}
\varepsilon_n^{\mathrm{safe}}
&=
\max(0.20,0.15,0.10,0.05,0.10)\\
&=
0.20.
\end{aligned}
$$

従って、

$$
0.20
<
0.40
=
\frac{\gamma_n}{2}
$$

である。現行Notebookでは、次のassertが保存された実行順の中でerrorなく通っている。

- `l_tilde_safe.shape == (c,)`
- `epsilon_safe.ndim == 0`
- `y_tilde_safe.ndim == 0`
- `epsilon_safe < gamma_n / 2`
- `y_tilde_safe == y_star`
- 全competitorについて`compressed_gap >= gamma_n - 2 * epsilon_safe`
- 全competitorについて`compressed_gap > 0`

safe caseは、十分条件が成立するとpredictionが保存されることと整合する。

## 6. Worst-case方向によるgap縮小

Notebookでは、

$$
\varepsilon_{\mathrm{worst}}
=
0.30
$$

とし、top class 0を $0.30$ 下げ、strongest competitor class 1を $0.30$ 上げる。

$$
\delta l_{\mathrm{worst}}
=
\begin{pmatrix}
-0.30\\
0.30\\
0\\
0\\
0
\end{pmatrix}.
$$

$$
\widetilde l_{\mathrm{worst}}
=
\begin{pmatrix}
2.10\\
1.90\\
0.20\\
-0.50\\
1.20
\end{pmatrix}.
$$

保存済み出力は次である。

~~~text
epsilon_worst = 0.300000
gap before    = 0.800000
2 * epsilon   = 0.600000
gap after     = 0.200000
~~~

従って、

$$
\begin{aligned}
\text{gap after}
&=
2.10-1.90\\
&=
0.20\\
&=
0.80-2\cdot0.30\\
&=
\text{gap before}-2\varepsilon_{\mathrm{worst}}.
\end{aligned}
$$

理論に現れる $2\varepsilon$ が、topを下げ、competitorを上げる二つの変化の和であることを数値的に確認した。

## 7. Common shiftの結果

Notebookでは、全classへ同じ $+1$ を加える。

$$
\delta l_n^{\mathrm{shift}}
=
\begin{pmatrix}
1\\
1\\
1\\
1\\
1
\end{pmatrix}.
$$

従って、

$$
\widetilde l_n^{\mathrm{shift}}
=
\begin{pmatrix}
3.4\\
2.6\\
1.2\\
0.5\\
2.2
\end{pmatrix}.
$$

$$
\varepsilon_n^{\mathrm{shift}}
=
1.0
\ge
0.4
=
\frac{\gamma_n}{2}.
$$

十分条件は成立しない。一方、top classはshift前後ともclass 0である。現行Notebookでは、

~~~python
assert epsilon_large_same_shift >= gamma_n / 2
assert int(y_tilde_large_same_shift) == int(y_star)
~~~

が保存された実行順の中でerrorなく通っている。

各classへ同じ値を足すと、任意のclass $i,j$ について、

$$
\left[(l_n)_i+1\right]
-
\left[(l_n)_j+1\right]
=
(l_n)_i-(l_n)_j
$$

となる。error normは大きくても全てのclass間gapは不変なので、predictionも不変である。

## 8. 結果の考察

### 8.1 Safe caseの意味

safe caseでpredictionが一致したことは、理論上の十分条件と整合する。ただし、単一の数値例だけで一般定理を証明したわけではない。一般性は理論ノートの全competitorに対する導出から得られ、Notebookはその具体例を検算する役割を持つ。

### 8.2 Worst-case方向の意味

$\ell_\infty$ normはerrorの最大絶対値だけを保持し、方向を捨てる。方向を知らずに全てのerrorを安全側で覆うには、top classが $-\varepsilon$、competitorが $+\varepsilon$ へ動く場合を考える必要がある。そのためmargin下界に $2\varepsilon$ が現れる。

### 8.3 Common shiftが示すこと

common shiftは、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
$$

が必要条件ではないことを明確に示す。条件がFalseであることは、predictionがFalseになることではない。

### 8.4 保証と実測を分ける

Notebookでは、

- `epsilon < gamma / 2`という保証条件。
- `argmax(l_tilde) == argmax(l)`という実測結果。

を別々にassertする。圧縮評価でも、この二つを同じ指標として扱わない。

## 9. この実験から言えること

- 1 sample、5 classesのtoy logitsでclassification marginを計算できた。
- safe errorが十分条件を満たし、argmaxも保存された。
- worst-case方向ではtop-vs-competitor gapが厳密に $2\varepsilon$ 縮んだ。
- common shiftでは十分条件を満たさなくてもargmaxが保存された。
- 十分条件の成立と実際のprediction一致は別の判定であることを確認できた。

## 10. この実験から言えないこと

今回のerrorは手で与えたtoy vectorであり、次はまだ示していない。

- TT-SVDまたはTT-roundingから得たlogits error。
- 特定のTT-rankとclassification marginの関係。
- 学習済みFashion-MNIST MLPのsampleごとのmargin。
- test set全体のprediction一致率またはaccuracy。
- batch-level $\|\Delta L\|_F$ とsample-level marginの実測接続。
- fine-tuning後のmarginとaccuracy。

また、favorable errorとadversarial errorの対比例、および全competitorに対する直接条件の関数は、教材として理論・実装ノートへ収録したが、現行Notebookの保存済み結果ではない。

## 11. 次の実験への接続

次の段階では、実際のTT-rank候補ごとに得たlogits errorについて、sample $n$ ごとに、

$$
\gamma_n,
\qquad
\|\delta l_n\|_\infty,
\qquad
\operatorname*{argmax}(\widetilde l_n)
=
\operatorname*{argmax}(l_n)
$$

を記録する。

さらに、

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
$$

を使えば、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]のbatch-level logits errorからsample-level条件へ接続できる。ただし、globalなFrobenius boundは保守的になりやすいため、実際のsampleごとのerrorとprediction一致も併記する。
