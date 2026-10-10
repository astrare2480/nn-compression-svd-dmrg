---
title: Batch Logits ErrorからSample-wise Prediction Stabilityへ
tags:
  - TTLinear
  - logits
  - Frobenius-norm
  - classification-margin
  - certificate
---

# Batch Logits ErrorからSample-wise Prediction Stabilityへ

## 1. このノートの目的

前段では、2層MLPの第1 Linearを近似したとき、batch全体のlogits errorに対して、

$$
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

を得た。また、1 sampleのlogits errorについて、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
$$

ならdense modelとcompressed modelの予測classが一致することを示した。

このノートでは、両者の間にある次の橋を、normの定義まで戻って導く。

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

従って、

$$
\boxed{
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}
}
$$

は、sample $n$ の予測classが変わらないための十分条件になる。

ここではsample $n$ を固定する。batch内全sampleの同時保証、minimum margin、accuracy、実データ、rank sweep、fine-tuningには進まない。

## 2. 2層MLPと誤差の記号

batch sizeを $B$、入力次元を $d$、hidden次元を $h$、class数を $c$ とする。入力を行に並べたbatchを、

$$
X
\in
\mathbb{R}^{B\times d}
$$

とする。dense modelの第1 Linearとその近似を、

$$
W_1,
\widetilde W_1
\in
\mathbb{R}^{h\times d},
\qquad
\Delta W_1
=
\widetilde W_1-W_1
$$

とする。第2 Linearの重みは、

$$
W_2
\in
\mathbb{R}^{c\times h}
$$

である。

第1 Linearの出力、ReLU後のhidden activation、最終logitsを、

$$
Z_1
=
XW_1^T+\boldsymbol{1}_B b_1^T,
\qquad
H
=
\operatorname{ReLU}(Z_1),
$$

$$
L
=
HW_2^T+\boldsymbol{1}_B b_2^T
\in
\mathbb{R}^{B\times c}
$$

とする。近似modelについても同様に、

$$
\widetilde Z_1
=
X\widetilde W_1^T+\boldsymbol{1}_B b_1^T,
\qquad
\widetilde H
=
\operatorname{ReLU}(\widetilde Z_1),
$$

$$
\widetilde L
=
\widetilde H W_2^T+\boldsymbol{1}_B b_2^T
\in
\mathbb{R}^{B\times c}
$$

とする。batch全体のlogits errorは、

$$
\Delta L
=
\widetilde L-L
\in
\mathbb{R}^{B\times c}
$$

である。

## 3. Batch行列と1 sampleの関係

第 $n$ sampleのdense logitsとcompressed logitsを、列ベクトルとして、

$$
l_n,
\widetilde l_n
\in
\mathbb{R}^{c}
$$

とする。その誤差は、

$$
\delta l_n
=
\widetilde l_n-l_n
\in
\mathbb{R}^{c}
$$

である。

一方、$L$ と $\widetilde L$ はsampleを行に積むので、厳密な行列表記は、

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
\end{pmatrix}
$$

である。従って、

$$
\Delta L
=
\begin{pmatrix}
\delta l_1^T\\
\delta l_2^T\\
\vdots\\
\delta l_B^T
\end{pmatrix}.
$$

$\delta l_n$ を列ベクトルと定義したため、行列の第 $n$ 行には $\delta l_n^T$ が入る。単に「$\delta l_n$ は $\Delta L$ の第 $n$ 行」と言う場合は、同じ $c$ 個の成分を指している。

PyTorchのshape `(c,)` は1階Tensorであり、行ベクトルと列ベクトルを区別しない。数学上の転置は、`delta_l_n = delta_L[n]`という要素抽出に追加の計算を要求するものではない。

## 4. この導出に現れる4種類のnorm

### 4.1 Vectorの$\ell_\infty$ norm

$$
v
=
\begin{pmatrix}
v_1\\
v_2\\
\vdots\\
v_c
\end{pmatrix}
\in
\mathbb{R}^{c}
$$

に対して、

$$
\|v\|_\infty
=
\max_{1\le k\le c}|v_k|
$$

である。logits errorでは、最も大きく動いたclass成分を表す。

### 4.2 Vectorの$\ell_2$ norm

$$
\|v\|_2
=
\left(
\sum_{k=1}^{c}|v_k|^2
\right)^{1/2}
$$

である。全class成分の誤差を二乗和で集約する。

### 4.3 MatrixのFrobenius norm

$$
A
=
(a_{ij})
\in
\mathbb{R}^{m\times n}
$$

に対して、

$$
\|A\|_F
=
\left(
\sum_{i=1}^{m}
\sum_{j=1}^{n}
|a_{ij}|^2
\right)^{1/2}
$$

である。全要素を1本のvectorへ並べた、

$$
\operatorname{vec}(A)
\in
\mathbb{R}^{mn}
$$

を使えば、

$$
\boxed{
\|A\|_F
=
\|\operatorname{vec}(A)\|_2
}
$$

と書ける。Frobenius normは行列の全要素を集計するnormである。

### 4.4 Matrixのspectral norm

同じ添字2でも、行列に対する、

$$
\|A\|_2
=
\max_{x\ne 0}
\frac{\|Ax\|_2}{\|x\|_2}
=
\sigma_{\max}(A)
$$

はspectral normである。これは全要素の二乗和ではなく、線形写像 $x\mapsto Ax$ がvectorの長さを最大で何倍にするかを表す。

従って、次を混同しない。

$$
\underbrace{\|v\|_2}_{\text{vectorのEuclidean norm}},
\qquad
\underbrace{\|A\|_F}_{\text{matrix全要素の二乗和}},
\qquad
\underbrace{\|A\|_2}_{\text{matrixの最大増幅率}}.
$$

このノートの $\|\delta l_n\|_2$ はvector 2-norm、$\|\Delta L\|_F$ と $\|X\|_F$ はFrobenius norm、$\|\Delta W_1\|_2$ と $\|W_2\|_2$ はmatrix spectral normである。

## 5. 第1段階：$\|\delta l_n\|_\infty\le\|\delta l_n\|_2$

一般のvector $v\in\mathbb{R}^{c}$ について証明する。絶対値が最大となる添字を $j$ とする。

$$
|v_j|
=
\max_k|v_k|
=
\|v\|_\infty.
$$

全ての二乗は非負なので、

$$
|v_j|^2
\le
\sum_{k=1}^{c}|v_k|^2.
$$

両辺は非負であり、平方根は非負実数上で単調増加だから、

$$
|v_j|
\le
\left(
\sum_{k=1}^{c}|v_k|^2
\right)^{1/2}.
$$

左辺と右辺をnormで書き直せば、

$$
\boxed{
\|v\|_\infty
\le
\|v\|_2
}.
$$

$v=\delta l_n$ と置くことで、

$$
\boxed{
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
}
$$

を得る。

例えば、

$$
\delta l_n
=
\begin{pmatrix}
0.10\\
0.18\\
-0.17
\end{pmatrix}
$$

なら、

$$
\|\delta l_n\|_\infty
=
\max(0.10,0.18,0.17)
=
0.18,
$$

$$
\begin{aligned}
\|\delta l_n\|_2
&=
\sqrt{0.10^2+0.18^2+(-0.17)^2}\\
&=
\sqrt{0.0713}\\
&\approx
0.2670.
\end{aligned}
$$

確かに $0.18\le0.2670$ である。

## 6. 第2段階：$\|\delta l_n\|_2\le\|\Delta L\|_F$

batch logits errorを全要素で書く。

$$
\Delta L
=
\begin{pmatrix}
\delta l_{11} & \delta l_{12} & \cdots & \delta l_{1c}\\
\delta l_{21} & \delta l_{22} & \cdots & \delta l_{2c}\\
\vdots & \vdots & \ddots & \vdots\\
\delta l_{B1} & \delta l_{B2} & \cdots & \delta l_{Bc}
\end{pmatrix}.
$$

第 $n$ 行のvector 2-normの二乗は、

$$
\|\delta l_n\|_2^2
=
\sum_{k=1}^{c}|\delta l_{nk}|^2.
$$

一方、行列全体のFrobenius normの二乗は、

$$
\begin{aligned}
\|\Delta L\|_F^2
&=
\sum_{i=1}^{B}
\sum_{k=1}^{c}
|\delta l_{ik}|^2\\
&=
\underbrace{
\sum_{k=1}^{c}|\delta l_{nk}|^2
}_{\|\delta l_n\|_2^2}
+
\underbrace{
\sum_{\substack{i=1\\i\ne n}}^{B}
\sum_{k=1}^{c}|\delta l_{ik}|^2
}_{\text{他sampleの非負な寄与}}.
\end{aligned}
$$

従って、

$$
\|\delta l_n\|_2^2
\le
\|\Delta L\|_F^2.
$$

両辺は非負なので平方根を取ると、

$$
\boxed{
\|\delta l_n\|_2
\le
\|\Delta L\|_F
}.
$$

### 6.1 行列全体を使った数値例

$$
\Delta L
=
\begin{pmatrix}
0.2 & -0.7 & 0.1\\
0.3 & 0.4 & -0.2
\end{pmatrix}
$$

とする。第1 sampleの誤差は、

$$
\delta l_1
=
\begin{pmatrix}
0.2\\
-0.7\\
0.1
\end{pmatrix}.
$$

そのvector 2-normは、

$$
\begin{aligned}
\|\delta l_1\|_2
&=
\sqrt{0.2^2+(-0.7)^2+0.1^2}\\
&=
\sqrt{0.04+0.49+0.01}\\
&=
\sqrt{0.54}.
\end{aligned}
$$

行列全体では、

$$
\begin{aligned}
\|\Delta L\|_F
&=
\sqrt{
0.2^2+(-0.7)^2+0.1^2
+0.3^2+0.4^2+(-0.2)^2
}\\
&=
\sqrt{0.54+0.09+0.16+0.04}\\
&=
\sqrt{0.83}.
\end{aligned}
$$

従って、

$$
\sqrt{0.54}
\le
\sqrt{0.83}.
$$

Frobenius normは第1行だけでなく、第2行の誤差も非負量として加えるため、小さくならない。

## 7. Batch-level上界まで接続する

第1 Linearの重みだけを近似し、biasと第2 Linearを共通とする。すると、

$$
\begin{aligned}
\Delta Z_1
&=
\widetilde Z_1-Z_1\\
&=
X\widetilde W_1^T-XW_1^T\\
&=
X(\widetilde W_1-W_1)^T\\
&=
X\Delta W_1^T.
\end{aligned}
$$

ReLUの1-Lipschitz性から、

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F.
$$

また、

$$
\begin{aligned}
\Delta L
&=
\widetilde L-L\\
&=
\widetilde H W_2^T-HW_2^T\\
&=
(\widetilde H-H)W_2^T\\
&=
\Delta H W_2^T.
\end{aligned}
$$

Frobenius normとspectral normの積の不等式、

$$
\|AB\|_F
\le
\|A\|_F\|B\|_2
$$

を順に使うと、

$$
\begin{aligned}
\|\Delta L\|_F
&=
\|\Delta H W_2^T\|_F\\
&\le
\|\Delta H\|_F\|W_2^T\|_2\\
&=
\|\Delta H\|_F\|W_2\|_2\\
&\le
\|\Delta Z_1\|_F\|W_2\|_2\\
&=
\|X\Delta W_1^T\|_F\|W_2\|_2\\
&\le
\|X\|_F
\|\Delta W_1^T\|_2
\|W_2\|_2\\
&=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2.
\end{aligned}
$$

ここへ前節までの結果をつなげる。

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
}.
$$

このchainには異なる役割のnormが並ぶ。

- 最左辺は、sample $n$ の各class logitが最大でどれだけ動いたか。
- 次は、同じsampleの全class誤差を二乗和で集めた量。
- 次は、batch内の全sample・全class誤差を集めた量。
- 最右辺は、入力、重み近似誤差、第2 Linearの増幅率だけで計算できる上界。

## 8. なぜこの橋が必要なのか

classification marginによる予測安定性が直接必要とするのは、

$$
\|\delta l_n\|_\infty
$$

である。しかし、matrix演算の誤差解析から自然に得られた量は、batch全体の、

$$
\|\Delta L\|_F
$$

である。両者は対象もnormも異なる。

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
$$

は、batch-levelの解析結果をsample-wiseなclassification判定へ運ぶための橋である。この橋がなければ、batch全体の誤差が小さいという結果だけから、特定sampleのargmaxについて何を言えるかが明示されない。

## 9. Classification marginとの接続

dense logitsのtop classを、

$$
y_n^*
=
\operatorname*{argmax}_k(l_n)_k
$$

とする。top classが一意であるとき、classification marginを、

$$
\gamma_n
=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k
>0
$$

と定義する。

[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]より、

$$
\|\delta l_n\|_\infty
<
\frac{\gamma_n}{2}
$$

なら、

$$
\operatorname*{argmax}_k(\widetilde l_n)_k
=
\operatorname*{argmax}_k(l_n)_k.
$$

ところが、

$$
\|\delta l_n\|_\infty
\le
\|\Delta L\|_F
$$

なので、より強い条件、

$$
\boxed{
\|\Delta L\|_F
<
\frac{\gamma_n}{2}
}
$$

を確認しても予測安定性を保証できる。さらに、

$$
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

だから、

$$
\boxed{
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}
}
$$

も十分条件になる。

例えば、

$$
\|\delta l_n\|_\infty
=0.10,
\qquad
\|\Delta L\|_F
=0.40,
\qquad
\frac{\gamma_n}{2}
=0.50
$$

なら、

$$
0.10
\le
0.40
<
0.50
$$

であり、batch Frobenius normを使う強い条件からsample $n$ のprediction安定性を保証できる。

## 10. 十分条件が保守的になる理由

導出の各段階では、predictionに有利な情報を捨てている。

### 10.1 $\ell_\infty$からmargin条件へ

$\|\delta l_n\|_\infty$ は各classの誤差方向を捨てる。全classへ同じ値を加えるcommon shiftはargmaxを変えないが、normは大きくなり得る。

### 10.2 1 sampleからbatch全体へ

$\|\delta l_n\|_2\le\|\Delta L\|_F$ では、sample $n$ と無関係な他sampleの誤差も加える。あるsampleの誤差が非常に小さくても、他sampleが大きければbatch Frobenius normは大きくなる。

### 10.3 実測誤差から演算子norm上界へ

$$
\|\Delta L\|_F
\le
\|X\|_F\|\Delta W_1\|_2\|W_2\|_2
$$

では、実際の $X$、$\Delta W_1$、$W_2$ の方向関係を捨て、各段階の最大増幅率で覆う。

従って、certificateがFalseでも、predictionが変わったとは言えない。正しい論理は、

$$
\text{certificate=True}
\Longrightarrow
\text{prediction is stable}
$$

であり、一般には、

$$
\text{certificate=False}
\centernot\Longrightarrow
\text{prediction changes}
$$

である。

## 11. 実測値と理論上界を区別する

次の4量は、すべて同じものではない。

$$
e_{\infty,n}
=
\|\delta l_n\|_\infty,
$$

$$
e_{2,n}
=
\|\delta l_n\|_2,
$$

$$
e_{F,\mathrm{batch}}
=
\|\Delta L\|_F,
$$

$$
u_{\mathrm{theory}}
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2.
$$

同じ実験で、

$$
e_{\infty,n}
\le
e_{2,n}
\le
e_{F,\mathrm{batch}}
\le
u_{\mathrm{theory}}
$$

を個別に記録する。最左辺はsampleの実測最大誤差、最右辺はmodel構成から得た保証用上界である。

## 12. このノートから言えること・言えないこと

言えることは次である。

- batch logits errorのFrobenius normは、任意の1 sampleのlogits errorのvector 2-normを上から押さえる。
- vector 2-normは、同じvectorのinfinity normを上から押さえる。
- batch-levelの重み誤差上界を、1 sampleのmargin-based certificateへ接続できる。
- certificateはprediction安定性の十分条件であり、必要条件ではない。

このノートだけでは次は言えない。

- batch内の全sampleが同時に安定すること。
- dataset全体のaccuracyが維持されること。
- certificateがFalseのsampleでpredictionが変わること。
- $\Delta W_1$ が特定のTT-rank打ち切りから得られたこと。
- どのTT-rankが最適か。
- fine-tuning後の性能。

PyTorchでのshape、norm API、boolean判定は、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]を参照する。固定toy行列の設定、保存済み結果、考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]へ分ける。次にminimum marginを導入し、同じ上界からbatch内の全sampleを同時に保証する場合は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]へ進む。
