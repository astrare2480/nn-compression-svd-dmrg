---
title: Classification MarginとArgmax安定性
tags:
  - TTLinear
  - logits
  - classification-margin
  - argmax
  - robustness
---

# Classification MarginとArgmax安定性

## 1. このノートの目的

TT/MPS圧縮によってlogitsが変化したとき、dense modelとcompressed modelの予測classが一致するための十分条件を、1 sampleについて導く。

中心となる結論は、

$$
\boxed{
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
}
\quad\Longrightarrow\quad
\boxed{
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k
}
$$

である。

ただし、この条件は十分条件であって必要条件ではない。そこで、次も区別する。

- normだけから得るworst-caseの保証。
- 実際の誤差ベクトルの向きまで使う直接判定。
- 十分条件を満たさなくてもpredictionが変わらない例。
- 十分条件を満たさず、実際にpredictionが変わる例。

このノートではsample $n$ を固定する。batch全体のminimum margin、accuracy保証、rank sweep、fine-tuningは扱わない。

## 2. 記号とshape

class数を $c$ とする。dense modelとcompressed modelの1 sample分のlogitsを、

$$
l_n
=
\begin{pmatrix}
(l_n)_0\\
(l_n)_1\\
\vdots\\
(l_n)_{c-1}
\end{pmatrix}
\in
\mathbb{R}^{c},
$$

$$
\widetilde l_n
=
\begin{pmatrix}
(\widetilde l_n)_0\\
(\widetilde l_n)_1\\
\vdots\\
(\widetilde l_n)_{c-1}
\end{pmatrix}
\in
\mathbb{R}^{c}
$$

とする。logits errorは、

$$
\delta l_n
=
\widetilde l_n-l_n
\in
\mathbb{R}^{c}
$$

である。従って各class $k$ について、

$$
(\widetilde l_n)_k
=
(l_n)_k+(\delta l_n)_k
$$

が成り立つ。

logitsはsoftmax前の実数scoreであり、確率ではない。負の値を取ってもよく、各成分が $[0,1]$ に収まる必要も、総和が1になる必要もない。

## 3. `max`と`argmax`を区別する

$$
l_n
=
\begin{pmatrix}
1.0\\
4.0\\
3.0
\end{pmatrix}
$$

とする。このとき、

$$
\max_k(l_n)_k
=
4.0
$$

は最大のlogit値である。一方、

$$
\operatorname*{argmax}_k(l_n)_k
=
1
$$

は最大値を持つclass indexである。

dense modelの予測classを、

$$
y_n^*
=
\operatorname*{argmax}_k(l_n)_k
$$

と書く。$y_n^*$ はlogit値ではなく、整数のclass indexである。

## 4. Classification margin

dense top classを除いた中で最大のlogitを持つclassを、strongest competitorと呼ぶ。

$$
k_n^{\mathrm{comp}}
=
\operatorname*{argmax}_{k\ne y_n^*}(l_n)_k.
$$

classification marginは、top classとstrongest competitorのlogit差である。

$$
\begin{aligned}
\gamma_n
&=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k\\
&=
(l_n)_{y_n^*}
-
(l_n)_{k_n^{\mathrm{comp}}}.
\end{aligned}
$$

同じ量は、全competitorに対するgapの最小値としても書ける。

$$
\boxed{
\gamma_n
=
\min_{k\ne y_n^*}
\left[
(l_n)_{y_n^*}-(l_n)_k
\right]
}.
$$

実際、定数 $(l_n)_{y_n^*}$ から最も大きいcompetitor logitを引くことは、top classとの差が最も小さいcompetitorを選ぶことと同じである。

先ほどの例では、

$$
y_n^*=1,
\qquad
k_n^{\mathrm{comp}}=2,
$$

$$
\gamma_n
=
4.0-3.0
=
1.0.
$$

### 4.1 一意なtop classとtie

以下ではdense logitsのtop classが一意であると仮定する。

$$
\gamma_n>0.
$$

もしtop logitsが同率なら、

$$
\gamma_n=0
$$

である。任意に小さい摂動で順位が変わり得るため、正の安定半径は得られない。また、同率最大値に対する`argmax`の結果はtieの扱いに依存するので、予測不変を実装依存なしに保証するときはstrictな大小関係を用いる。

### 4.2 Marginは正解・不正解を表さない

$\gamma_n$ はdense model自身のtop classと2位classの差である。正解labelとの比較ではない。

従って、marginが大きくてもdense modelが誤分類している場合はある。margin条件が保証するのは、compressed modelがdense modelと同じclassを選ぶことであり、ground-truth labelに対する正解を保証することではない。

## 5. Logits errorの$\ell_\infty$ norm

logits errorの最大成分誤差を、

$$
\varepsilon_n
=
\|\delta l_n\|_\infty
=
\max_k|(\delta l_n)_k|
$$

とする。より一般に、既知の上界 $\varepsilon_n$ を使うなら、

$$
\|\delta l_n\|_\infty
\le
\varepsilon_n
$$

と置く。

この不等式から、全てのclass $k$ について、

$$
|(\delta l_n)_k|
\le
\varepsilon_n
$$

が同時に従う。絶対値不等式を上下から書けば、

$$
-\varepsilon_n
\le
(\delta l_n)_k
\le
\varepsilon_n.
$$

逆に、全成分がこの範囲にあれば、その最大値も $\varepsilon_n$ 以下なので、

$$
\|\delta l_n\|_\infty
\le
\varepsilon_n
$$

である。従って、このnorm boundと全成分のboundは同値である。

例えば、

$$
\delta l_n
=
\begin{pmatrix}
-0.12\\
0.05\\
0.20
\end{pmatrix}
$$

なら、

$$
\|\delta l_n\|_\infty
=
\max(0.12,0.05,0.20)
=
0.20.
$$

## 6. Argmax安定性の十分条件

任意のcompetitor $k\ne y_n^*$ を一つ固定する。compressed logitsにおけるtop classとclass $k$ の差は、

$$
(\widetilde l_n)_{y_n^*}
-
(\widetilde l_n)_k
$$

である。$\widetilde l_n=l_n+\delta l_n$ を成分ごとに代入すると、

$$
\begin{aligned}
(\widetilde l_n)_{y_n^*}
-
(\widetilde l_n)_k
&=
\left[
(l_n)_{y_n^*}
+
(\delta l_n)_{y_n^*}
\right]
-
\left[
(l_n)_k
+
(\delta l_n)_k
\right]\\
&=
\left[
(l_n)_{y_n^*}-(l_n)_k
\right]
+
\left[
(\delta l_n)_{y_n^*}-(\delta l_n)_k
\right].
\end{aligned}
$$

ここで、

$$
\|\delta l_n\|_\infty
\le
\varepsilon_n
$$

なら、top classの誤差には、

$$
(\delta l_n)_{y_n^*}
\ge
-\varepsilon_n
$$

が成り立つ。competitorの誤差には、

$$
(\delta l_n)_k
\le
\varepsilon_n
$$

が成り立つ。従って、誤差によるgap変化は、

$$
\begin{aligned}
(\delta l_n)_{y_n^*}
-
(\delta l_n)_k
&\ge
(-\varepsilon_n)-(+\varepsilon_n)\\
&=
-2\varepsilon_n.
\end{aligned}
$$

また、$\gamma_n$ は全competitorに対するdense gapの最小値なので、

$$
(l_n)_{y_n^*}-(l_n)_k
\ge
\gamma_n.
$$

二つを元の恒等式へ代入すると、

$$
\begin{aligned}
(\widetilde l_n)_{y_n^*}
-
(\widetilde l_n)_k
&=
\left[
(l_n)_{y_n^*}-(l_n)_k
\right]
+
\left[
(\delta l_n)_{y_n^*}-(\delta l_n)_k
\right]\\
&\ge
\gamma_n-2\varepsilon_n.
\end{aligned}
$$

従って、

$$
\varepsilon_n
<
\frac{\gamma_n}{2}
$$

なら、

$$
\gamma_n-2\varepsilon_n
>0.
$$

よって全ての $k\ne y_n^*$ に対して、

$$
(\widetilde l_n)_{y_n^*}
>
(\widetilde l_n)_k
$$

が成り立ち、compressed modelでも $y_n^*$ が一意なtop classになる。

$$
\boxed{
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
}
\quad\Longrightarrow\quad
\boxed{
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k
=
y_n^*
}.
$$

### 6.1 なぜ$2\varepsilon_n$なのか

worst caseでは、top classが $\varepsilon_n$ 下がり、同じcompetitorが $\varepsilon_n$ 上がる。

$$
(l_n)_{y_n^*}
\longrightarrow
(l_n)_{y_n^*}-\varepsilon_n,
$$

$$
(l_n)_k
\longrightarrow
(l_n)_k+\varepsilon_n.
$$

したがってgapは、

$$
\begin{aligned}
&\left[(l_n)_{y_n^*}-\varepsilon_n\right]
-
\left[(l_n)_k+\varepsilon_n\right]\\
&=
(l_n)_{y_n^*}-(l_n)_k-2\varepsilon_n
\end{aligned}
$$

まで縮み得る。

### 6.2 なぜstrict inequalityなのか

もし、

$$
\varepsilon_n
=
\frac{\gamma_n}{2}
$$

なら、worst caseでは、

$$
\gamma_n-2\varepsilon_n
=
0
$$

となり、top classとcompetitorがtieになり得る。元のclassが一意なtopであり続けることを保証するには、

$$
\varepsilon_n
<
\frac{\gamma_n}{2}
$$

が必要である。

## 7. 十分条件と必要条件を区別する

次の二つの命題を置く。

$$
A:
\quad
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2},
$$

$$
B:
\quad
\operatorname*{argmax}_k(\widetilde l_n)_k
=
y_n^*.
$$

証明したのは、

$$
A
\Longrightarrow
B
$$

である。従って、$A$ は $B$ の十分条件である。

一方、

$$
\neg A
\Longrightarrow
\neg B
$$

は証明していない。つまり、

$$
\|\delta l_n\|_\infty
\ge
\frac{\gamma_n}{2}
$$

から直ちに「predictionが変わる」とは言えない。この場合に言えるのは、$\ell_\infty$ normだけを使ったworst-case保証が成立しない、ということだけである。

## 8. 誤差の向きまで使う直接条件

top class $y_n^*$ がcompressed logitsでも全competitorにstrictに勝つ条件は、

$$
(\widetilde l_n)_{y_n^*}
>
(\widetilde l_n)_k
\qquad
\text{for all }k\ne y_n^*
$$

である。$\widetilde l_n=l_n+\delta l_n$ を代入すると、

$$
(l_n)_{y_n^*}
+
(\delta l_n)_{y_n^*}
>
(l_n)_k
+
(\delta l_n)_k.
$$

誤差の項を左辺、dense gapを右辺へ整理すると、

$$
\boxed{
(\delta l_n)_k
-
(\delta l_n)_{y_n^*}
<
(l_n)_{y_n^*}
-
(l_n)_k
}
\qquad
\text{for all }k\ne y_n^*.
$$

左辺は、圧縮誤差によってcompetitor $k$ がtop classに対して得た相対的な優位である。右辺は、dense modelでtop classがそのcompetitorに対して持っていた余裕である。

$$
\boxed{
\text{誤差によるcompetitorの相対的な追い上げ}
<
\text{dense時点のlogit gap}
}.
$$

この不等式が全competitorについて成り立つことと、$y_n^*$ がcompressed logitsでもstrictなtopであることは同値である。

逆に、あるcompetitor $k$ について、

$$
(\delta l_n)_k
-
(\delta l_n)_{y_n^*}
>
(l_n)_{y_n^*}
-
(l_n)_k
$$

なら、そのcompetitorはtop classを追い越す。等号の場合はtieであり、実際の`argmax` indexはtieの扱いに依存する。

## 9. 数値例

### 9.1 十分条件を満たすsafe case

$$
l_n
=
\begin{pmatrix}
1.0\\
4.0\\
3.0
\end{pmatrix},
\qquad
\delta l_n
=
\begin{pmatrix}
0.1\\
-0.2\\
0.3
\end{pmatrix}.
$$

dense topはclass 1、strongest competitorはclass 2なので、

$$
\gamma_n
=
4.0-3.0
=
1.0.
$$

誤差の無限大normは、

$$
\varepsilon_n
=
\max(0.1,0.2,0.3)
=
0.3.
$$

従って、

$$
0.3
<
\frac{1.0}{2}
=
0.5.
$$

compressed logitsを全要素で足すと、

$$
\begin{aligned}
\widetilde l_n
&=
l_n+\delta l_n\\
&=
\begin{pmatrix}
1.0\\
4.0\\
3.0
\end{pmatrix}
+
\begin{pmatrix}
0.1\\
-0.2\\
0.3
\end{pmatrix}\\
&=
\begin{pmatrix}
1.1\\
3.8\\
3.3
\end{pmatrix}.
\end{aligned}
$$

class 1とclass 2のgapは、

$$
3.8-3.3
=
0.5
>0
$$

であり、predictionはclass 1のままである。

### 9.2 同じnormでも誤差方向で結果が変わる

同じdense logits

$$
l_n
=
\begin{pmatrix}
1.0\\
4.0\\
3.0
\end{pmatrix}
$$

に対し、まずtop classに有利な誤差を考える。

$$
\delta l_n^{\mathrm{fav}}
=
\begin{pmatrix}
0.0\\
0.8\\
-0.8
\end{pmatrix},
\qquad
\|\delta l_n^{\mathrm{fav}}\|_\infty
=
0.8.
$$

$$
\widetilde l_n^{\mathrm{fav}}
=
\begin{pmatrix}
1.0\\
4.8\\
2.2
\end{pmatrix}.
$$

$$
0.8
>
\frac{\gamma_n}{2}
=
0.5
$$

なので十分条件は満たさない。しかし、

$$
4.8-2.2
=
2.6
>0
$$

であり、predictionはclass 1のままである。

次に、同じ大きさでtop classに不利な誤差を考える。

ここでの`adversarial`は、入力に対するadversarial attackを意味しない。単に、logit順位にとって最も不利な向き、すなわちtop classを下げ、competitorを上げる誤差方向を指す。

$$
\delta l_n^{\mathrm{adv}}
=
\begin{pmatrix}
0.0\\
-0.8\\
0.8
\end{pmatrix},
\qquad
\|\delta l_n^{\mathrm{adv}}\|_\infty
=
0.8.
$$

$$
\widetilde l_n^{\mathrm{adv}}
=
\begin{pmatrix}
1.0\\
3.2\\
3.8
\end{pmatrix}.
$$

class 1とclass 2のgapは、

$$
3.2-3.8
=
-0.6
<0
$$

となり、predictionはclass 2へ変わる。

二つの誤差は同じ $\ell_\infty$ normを持つが、相対的な向きが異なる。normだけを使う十分条件が保守的になる理由である。

### 9.3 全classへのcommon shift

任意の $a\in\mathbb{R}$ に対して、

$$
\delta l_n
=
a\boldsymbol{1}_c
$$

とする。このとき任意のclass $i,j$ について、

$$
\begin{aligned}
(\widetilde l_n)_i-(\widetilde l_n)_j
&=
\left[(l_n)_i+a\right]
-
\left[(l_n)_j+a\right]\\
&=
(l_n)_i-(l_n)_j.
\end{aligned}
$$

従って全てのclass間gapが保存され、argmaxも保存される。一方、

$$
\|\delta l_n\|_\infty
=
|a|
$$

なので、$|a|$ はいくらでも大きくできる。logit誤差の絶対的な大きさだけでなく、class間の相対差が重要であることを示す例である。

## 10. $\varepsilon_n$の意味に関する注意

### 10.1 実測値と上界

実際のerror vectorを計算できるなら、

$$
\varepsilon_n
=
\|\delta l_n\|_\infty
$$

と置ける。理論的な上界しか分からない場合は、

$$
\|\delta l_n\|_\infty
\le
\varepsilon_n
$$

と置く。

いずれの場合も、単に $\varepsilon_n$ が存在するだけでは安定性は保証されない。marginと比較して、

$$
\varepsilon_n
<
\frac{\gamma_n}{2}
$$

まで確認する必要がある。

### 10.2 Logitそのものより大きい誤差も定義できる

logitsは非正規化scoreなので、error normが元のlogit成分の絶対値より大きくても数学的な矛盾はない。重要なのは元のlogitの絶対値ではなく、top classと各competitorのgapに対し、相対誤差がどの向きへどれだけ作用するかである。

## 11. Batch-level上界との接続

sample $n$ のlogits errorは、batch error matrix $\Delta L\in\mathbb{R}^{B\times c}$ の第 $n$ 行である。従って、

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F.
$$

[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]で導いた、

$$
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

と組み合わせれば、

$$
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}
$$

はsample $n$ のargmaxを保存する、さらに上流からの十分条件になる。

ただし、batch全体のFrobenius normを1 sampleへ使い、さらに各段階でspectral norm上界を使うため、非常に保守的になり得る。各normの定義、二つの不等式の途中式、具体行列、上流の重み誤差上界までの完全なchainは、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]で扱う。batch全体の保証やminimum marginの設計へはまだ進まない。

## 12. この条件から言えること・言えないこと

言えることは、次である。

- unique topを持つdense logitsについて、$\|\delta l_n\|_\infty<\gamma_n/2$ ならcompressed logitsのunique topも同じclassである。
- 条件を満たさない場合でも、実際の誤差方向を使えばargmax不変性を直接判定できる。
- common shiftのように、normが大きくてもclass間gapを変えない誤差がある。

この条件だけでは、次は言えない。

- dense predictionがground-truth labelに対して正しいこと。
- dataset全体のaccuracyが維持されること。
- 全sampleでpredictionが一致すること。
- TT-rankが最適であること。
- fine-tuning後の性能。

PyTorchでの計算手順と実装上の注意は、[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]を参照する。toy Notebookの設定、保存済み結果、考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]へ分ける。batch-level logits errorからこの条件を実際につなぐ次段階は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]を参照する。
