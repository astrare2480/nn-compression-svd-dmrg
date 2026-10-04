---
title: TT-rank打ち切りによるLinear出力誤差とnorm上界
tags:
  - TT-matrix
  - TT-rank
  - Linear
  - Frobenius-norm
  - spectral-norm
  - error-bound
---

# TT-rank打ち切りによるLinear出力誤差とnorm上界

## 1. 問題設定

batch-firstのLinear層を

$$
X\in\mathbb{R}^{B\times n},
\qquad
W\in\mathbb{R}^{h\times n},
\qquad
b\in\mathbb{R}^{h}
$$

として、

$$
Y=XW^{\mathsf T}+b
\in\mathbb{R}^{B\times h}
$$

と書く。

TT-rankを打ち切って得た近似重みを

$$
\widetilde W\in\mathbb{R}^{h\times n}
$$

とする。biasは圧縮せず、同じ $b$ を用いる。

$$
\widetilde Y
=
X\widetilde W^{\mathsf T}+b.
$$

重み誤差と出力誤差を

$$
\Delta W=\widetilde W-W
\in\mathbb{R}^{h\times n},
$$

$$
\Delta Y=\widetilde Y-Y
\in\mathbb{R}^{B\times h}
$$

と定義する。

## 2. 出力誤差の行列導出

定義から、

$$
\begin{aligned}
\Delta Y
&=
\widetilde Y-Y\\
&=
\left(X\widetilde W^{\mathsf T}+b\right)
-
\left(XW^{\mathsf T}+b\right)\\
&=
X\widetilde W^{\mathsf T}
+b
-XW^{\mathsf T}
-b\\
&=
X\left(
\widetilde W^{\mathsf T}
-W^{\mathsf T}
\right)\\
&=
X\left(
\widetilde W-W
\right)^{\mathsf T}\\
&=
X\Delta W^{\mathsf T}.
\end{aligned}
$$

従って、

$$
\boxed{
\Delta Y
=
\widetilde Y-Y
=
X\Delta W^{\mathsf T}
}
$$

である。これは誤差上界ではなく、同じbiasを使う限り厳密な代数恒等式である。

shapeも

$$
(B\times n)(n\times h)
=
B\times h
$$

となり、$\Delta Y$ と一致する。

## 3. 全要素での導出

batch indexを $b$、出力特徴を $i$、入力特徴を $j$ とする。

$$
Y_{b,i}
=
\sum_{j=0}^{n-1}
X_{b,j}W_{i,j}
+b_i,
$$

$$
\widetilde Y_{b,i}
=
\sum_{j=0}^{n-1}
X_{b,j}\widetilde W_{i,j}
+b_i.
$$

差を取ると、

$$
\begin{aligned}
(\Delta Y)_{b,i}
&=
\widetilde Y_{b,i}-Y_{b,i}\\
&=
\left(
\sum_{j=0}^{n-1}
X_{b,j}\widetilde W_{i,j}
+b_i
\right)
-
\left(
\sum_{j=0}^{n-1}
X_{b,j}W_{i,j}
+b_i
\right)\\
&=
\sum_{j=0}^{n-1}
X_{b,j}
\left(
\widetilde W_{i,j}-W_{i,j}
\right)\\
&=
\sum_{j=0}^{n-1}
X_{b,j}(\Delta W)_{i,j}.
\end{aligned}
$$

一方、

$$
\begin{aligned}
\left(X\Delta W^{\mathsf T}\right)_{b,i}
&=
\sum_{j=0}^{n-1}
X_{b,j}
(\Delta W^{\mathsf T})_{j,i}\\
&=
\sum_{j=0}^{n-1}
X_{b,j}
(\Delta W)_{i,j}.
\end{aligned}
$$

したがって全ての $b,i$ で

$$
(\Delta Y)_{b,i}
=
\left(X\Delta W^{\mathsf T}\right)_{b,i}.
$$

## 4. Frobenius normとspectral norm

行列 $A\in\mathbb{R}^{m\times n}$ のFrobenius normは

$$
\boxed{
\|A\|_F
=
\left(
\sum_{i=1}^{m}
\sum_{j=1}^{n}
A_{ij}^2
\right)^{1/2}
}
$$

である。SVDを

$$
A=U\Sigma V^{\mathsf T},
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)
$$

とすれば、

$$
\|A\|_F
=
\left(
\sum_{k=1}^{r}\sigma_k^2
\right)^{1/2}.
$$

spectral normはvector 2-normから誘導されるoperator normで、

$$
\boxed{
\|A\|_2
=
\sup_{x\ne0}
\frac{\|Ax\|_2}{\|x\|_2}
=
\sigma_1
}
$$

である。

有限次元行列では単位球がcompactで写像が連続なので、この上限は最大値として達成される。ただし定義を一般の関数空間にも使える形にするため、通常は $\sup$ と書く。

長方形SVDから最大特異値を得る途中式、特異値と入力座標の総和範囲、$\sup$ と $\max$ の違いは、[[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]にまとめる。

## 5. spectral normの定義から出るベクトル不等式

$v\ne0$ に対して、

$$
\frac{\|Av\|_2}{\|v\|_2}
\le
\sup_{z\ne0}
\frac{\|Az\|_2}{\|z\|_2}
=
\|A\|_2.
$$

両辺に $\|v\|_2$ を掛ければ、

$$
\boxed{
\|Av\|_2
\le
\|A\|_2\|v\|_2
}
$$

を得る。これは「どの入力方向も最大増幅率を超えて伸びない」という定義そのものである。

例えば、任意の

$$
A=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix},
\qquad
v=
\begin{pmatrix}
v_1\\
v_2
\end{pmatrix}
$$

を考える。このとき、

$$
Av=
\begin{pmatrix}
3v_1\\
v_2
\end{pmatrix}.
$$

入力と出力のnormの二乗は、

$$
\|v\|_2^2
=
v_1^2+v_2^2,
$$

$$
\|Av\|_2^2
=
9v_1^2+v_2^2.
$$

従って増幅率の二乗は、

$$
\left(
\frac{\|Av\|_2}{\|v\|_2}
\right)^2
=
\frac{9v_1^2+v_2^2}{v_1^2+v_2^2}.
$$

分母を固定すると、分子は $v_1$ 方向の係数9と $v_2$ 方向の係数1からなる。そのため、この比が最大になるのは $v_2=0$ の方向である。例えば $v=(1,0)^{\mathsf T}$ なら、

$$
\frac{\|Av\|_2}{\|v\|_2}
=
\frac{3}{1}
=
3.
$$

よって、

$$
\|A\|_2
=
\sup_{v\ne0}
\frac{\|Av\|_2}{\|v\|_2}
=
3.
$$

この3が、全ての入力方向に共通して使える最小の増幅上限である。

別の方向 $v=(1,2)^{\mathsf T}$ でも確認すると、

$$
Av=(3,2)^{\mathsf T},
$$

$$
\|Av\|_2=\sqrt{13},
\qquad
\|A\|_2\|v\|_2=3\sqrt{5},
$$

なので確かに

$$
\sqrt{13}\le3\sqrt5.
$$

## 6. 第1の出力誤差上界

$X$ の第 $b$ 行を $x_b^{\mathsf T}$ とする。$\Delta Y$ の第 $b$ 行を列ベクトルとして見ると、

$$
(\Delta y_b)^{\mathsf T}
=
x_b^{\mathsf T}\Delta W^{\mathsf T},
$$

従って

$$
\Delta y_b
=
\Delta W x_b.
$$

spectral normの定義から、

$$
\|\Delta y_b\|_2
\le
\|\Delta W\|_2\|x_b\|_2.
$$

各batch sampleについて二乗し、総和を取る。

$$
\begin{aligned}
\|\Delta Y\|_F^2
&=
\sum_{b=1}^{B}
\|\Delta y_b\|_2^2\\
&\le
\sum_{b=1}^{B}
\|\Delta W\|_2^2
\|x_b\|_2^2\\
&=
\|\Delta W\|_2^2
\sum_{b=1}^{B}
\|x_b\|_2^2\\
&=
\|\Delta W\|_2^2
\|X\|_F^2.
\end{aligned}
$$

平方根を取ると、

$$
\boxed{
\|\Delta Y\|_F
\le
\|X\|_F\|\Delta W\|_2
}
$$

を得る。

## 7. 第2の出力誤差上界

一般に

$$
\|AB\|_F
\le
\|A\|_2\|B\|_F
$$

が成り立つ。$A=X$、$B=\Delta W^{\mathsf T}$ とすれば、

$$
\begin{aligned}
\|\Delta Y\|_F
&=
\|X\Delta W^{\mathsf T}\|_F\\
&\le
\|X\|_2
\|\Delta W^{\mathsf T}\|_F\\
&=
\|X\|_2
\|\Delta W\|_F.
\end{aligned}
$$

従って、

$$
\boxed{
\|\Delta Y\|_F
\le
\|X\|_2\|\Delta W\|_F
}
$$

である。

2本を同時に使えば、

$$
\boxed{
\|\Delta Y\|_F
\le
\min\left(
\|X\|_F\|\Delta W\|_2,
\|X\|_2\|\Delta W\|_F
\right)
}
$$

となる。

さらに

$$
\|A\|_2\le\|A\|_F
$$

より、粗いが簡単な

$$
\|\Delta Y\|_F
\le
\|X\|_F\|\Delta W\|_F
$$

も得られる。

## 8. なぜ一方だけspectral normなのか

Frobenius normは複数の行または列の誤差を二乗和で集計する。一方、行列を一つのベクトルへ作用させる局所段階では、最悪増幅率であるspectral normが自然に現れる。

第1の上界は、

$$
\text{各sampleの入力norm}
\times
\text{重み誤差の最大gain}
$$

をbatch全体で二乗和にしたものと読める。

第2の上界は左右を入れ替え、

$$
\text{入力行列の最大gain}
\times
\text{重み誤差の総量}
$$

と読める。

## 9. 二つのnormの大小関係

特異値を

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0
$$

とすると、

$$
\|A\|_2=\sigma_1,
\qquad
\|A\|_F
=
\sqrt{\sum_{k=1}^{r}\sigma_k^2}.
$$

従って、

$$
\boxed{
\|A\|_2
\le
\|A\|_F
\le
\sqrt{\operatorname{rank}(A)}\|A\|_2
}
$$

である。

### 9.1 rank 1行列

例えば、

$$
A=
\begin{pmatrix}
3&4\\
0&0
\end{pmatrix}
$$

を考える。

$$
AA^{\mathsf T}
=
\begin{pmatrix}
3&4\\
0&0
\end{pmatrix}
\begin{pmatrix}
3&0\\
4&0
\end{pmatrix}
=
\begin{pmatrix}
25&0\\
0&0
\end{pmatrix}.
$$

従って特異値は

$$
\sigma_1=5,
\qquad
\sigma_2=0
$$

である。よって、

$$
\|A\|_2
=
\sigma_1
=
5,
$$

$$
\|A\|_F
=
\sqrt{3^2+4^2}
=
5.
$$

rank 1なら非零特異値が一つだけなので、一般に

$$
\|A\|_2=\|A\|_F.
$$

### 9.2 二方向へ異なる強さで作用する対角行列

$$
A=
\begin{pmatrix}
3&0\\
0&4
\end{pmatrix}
$$

では、特異値は大きい順に

$$
\sigma_1=4,
\qquad
\sigma_2=3
$$

である。従って、

$$
\|A\|_2
=
4,
$$

$$
\|A\|_F
=
\sqrt{4^2+3^2}
=
5.
$$

spectral normは「最も伸びる方向での増幅率4」を表し、Frobenius normは「二つの特異方向をまとめた総量5」を表す。

### 9.3 単位行列

$d$ 次単位行列

$$
I_d=
\begin{pmatrix}
1&&0\\
&\ddots&\\
0&&1
\end{pmatrix}
$$

の特異値は全て1である。従って、

$$
\|I_d\|_2=1,
\qquad
\|I_d\|_F=\sqrt d.
$$

単位行列はどのベクトルも伸ばさないため最大増幅率は1である。一方、Frobenius normでは $d$ 個の対角成分を全て二乗和に含めるため $\sqrt d$ になる。この例が、最大方向と全方向の総量の違いを最も明確に示す。

参考として、節5で最大増幅率を調べた

$$
A=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix}
$$

なら、

$$
\|A\|_2=3,
\qquad
\|A\|_F=\sqrt{10}.
$$

spectral normは最大方向だけを、Frobenius normは全方向の寄与を測る。

## 10. TT/SVD打ち切りでの意味

rank-$r$ truncated SVDを $A_r$ とすると、

$$
\|A-A_r\|_F
=
\left(
\sum_{k=r+1}^{\operatorname{rank}(A)}
\sigma_k^2
\right)^{1/2},
$$

$$
\|A-A_r\|_2
=
\sigma_{r+1}.
$$

Frobenius normは捨てた全特異値の総量、spectral normは捨てた中で最大の一成分を測る。

Schmidt分解

$$
|\psi\rangle
=
\sum_{\alpha=1}^{\chi}
\lambda_\alpha
|\alpha_L\rangle
\otimes
|\alpha_R\rangle
$$

をrank $r$ で切ると、

$$
|\delta\psi\rangle
=
\sum_{\alpha=r+1}^{\chi}
\lambda_\alpha
|\alpha_L\rangle
\otimes
|\alpha_R\rangle.
$$

左右Schmidt状態の直交性から、

$$
\begin{aligned}
\|\delta\psi\|_2^2
&=
\langle\delta\psi|\delta\psi\rangle\\
&=
\sum_{\alpha=r+1}^{\chi}
\lambda_\alpha^2.
\end{aligned}
$$

従って、DMRGでいうdiscarded weight

$$
\epsilon_{\mathrm{disc}}
=
\sum_{\alpha>r}\lambda_\alpha^2
$$

は、捨てた状態成分のnorm二乗に対応する。

## 11. 機械学習での意味

### 11.1 Frobenius norm

$$
\|\Delta W\|_F^2
=
\sum_{i=1}^{h}
\sum_{j=1}^{n}
(\Delta W_{ij})^2
$$

は全重み要素の総二乗誤差である。要素ごとのRMSEは

$$
\operatorname{RMSE}_{\mathrm{weight}}
=
\frac{\|\Delta W\|_F}{\sqrt{hn}}.
$$

入力が白色化され、

$$
x\sim\mathcal N(0,I_n)
$$

なら、

$$
\begin{aligned}
\mathbb E\|\Delta W x\|_2^2
&=
\mathbb E
\left[
x^{\mathsf T}
\Delta W^{\mathsf T}
\Delta W x
\right]\\
&=
\operatorname{tr}
\left(
\Delta W^{\mathsf T}
\Delta W
\mathbb E[xx^{\mathsf T}]
\right)\\
&=
\operatorname{tr}
\left(
\Delta W^{\mathsf T}\Delta W
\right)\\
&=
\|\Delta W\|_F^2.
\end{aligned}
$$

一般の二次モーメント

$$
\Sigma_x=\mathbb E[xx^{\mathsf T}]
$$

に対しては、

$$
\boxed{
\mathbb E\|\Delta W x\|_2^2
=
\operatorname{tr}
\left(
\Delta W\Sigma_x\Delta W^{\mathsf T}
\right)
}
$$

となる。実データがほとんど使わない方向の重み誤差は、Frobenius normが大きくても出力へ強く現れない場合がある。

### 11.2 spectral norm

$$
\|\Delta W x\|_2
\le
\|\Delta W\|_2\|x\|_2
$$

は、入力がどの方向を向いても成り立つ最悪ケース保証である。$\|\Delta W\|_2$ はロバスト性や誤差伝播を考えるときの最大gainになる。

## 12. 実際に記録する量

重み近似では、

$$
E_W=\|W-\widetilde W\|_F,
\qquad
E_{W,\mathrm{rel}}
=
\frac{\|W-\widetilde W\|_F}{\|W\|_F}
$$

を記録する。

層出力では、

$$
E_Y=\|Y-\widetilde Y\|_F,
\qquad
E_{Y,\mathrm{rel}}
=
\frac{\|Y-\widetilde Y\|_F}{\|Y\|_F}
$$

を記録する。

これらは同じ量ではない。入力分布を通すことで、重み誤差が出力空間のどこへ現れるかが変わる。

実データ上では

$$
\frac{
\|X_{\mathrm{data}}\widetilde W^{\mathsf T}
-X_{\mathrm{data}}W^{\mathsf T}\|_F
}{
\|X_{\mathrm{data}}W^{\mathsf T}\|_F
}
$$

も有用である。ただし層出力誤差が小さくても、classification marginが小さいsampleの予測は変わり得る。accuracyまでの議論は別段階である。

## 13. 現在の学習境界

このノートが扱うのはLinear層単体である。ReLUが誤差を増幅しないことは、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]で説明する。

このノート単体では、後続層まで連結したlogits誤差上界、classification margin、accuracy保証、rank sweep、fine-tuningを完了扱いにしない。第2 Linearまで連結した理論上界は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]へ進む。

PyTorchによる小さい2-core例と保存済み数値結果は、[[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]を参照する。
