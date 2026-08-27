---
title: HOOI
aliases:
  - Higher-Order Orthogonal Iteration
  - Tucker ALS
  - HOSVDの反復精密化
tags:
  - HOOI
  - Tucker
  - HOSVD
  - Tensor
  - 交互最適化
---

# HOOI

## サマリー

HOOI（Higher-Order Orthogonal Iteration）は、固定したmultilinear rankのTucker近似で、factorを1 modeずつ反復更新して再構成Frobenius誤差を小さくする方法である。

標準的な直交factorのTucker近似では、目的を

$$
\boxed{
\begin{aligned}
\min_{
\mathcal{G},
U^{(0)},\ldots,U^{(N-1)}
}
&\quad
\left\|
\mathcal{X}
-
\mathcal{G}
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
\right\|_F^2\\
\text{s.t.}
&\quad
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
\qquad(n=0,\ldots,N-1)
\end{aligned}
}
$$

と書ける。

partial HOOIでは、全modeではなく圧縮対象modeの集合を

$$
\mathcal M
\subseteq
\{0,1,\ldots,N-1\}
$$

として、$n\in\mathcal M$ のfactorだけを持つ。現在のsrcでは `ranks.keys()` がこの $\mathcal M$ に対応する。

この問題はfactorを同時に求めると非凸なので、HOOIは1 modeずつ交互に更新する。HOSVDを高速な初期解として使い、**同じrankのまま再構成誤差を精密化する方法**として理解する。

---

## 1. HOSVDとの違い

| 観点 | HOSVD | HOOI |
| --- | --- | --- |
| factorの求め方 | 各modeを元Tensorから独立にSVD | 他modeを固定して1 modeずつ更新 |
| 反復 | なし | あり |
| rank | 指定rank | 同じ指定rank |
| 主な用途 | 高速な近似・初期解・rank探索 | 同rankでの再構成精密化 |
| コスト | 小さい | sweep分だけ大きい |

HOSVDは

$$
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\left(
\operatorname{left\ singular\ vectors}(X_{(n)})
\right)
$$

を各modeで**元の同じTensorから独立に**行う。

HOOIでは、更新対象以外のfactorを使ってTensorをrank空間へ射影してから、その対象modeをunfoldしてSVDする。

---

## 2. factorが固定されたときのcore

factorの列が直交して

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
$$

を満たすとする。

全mode Tuckerなら、現在factorに対するcoreは

$$
\boxed{
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
}
$$

である。

partial Tuckerなら、

$$
\boxed{
\mathcal G
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
U^{(m)\mathsf T}
}
$$

と、圧縮対象modeだけを射影する。

再構成は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\underset{m\in\mathcal M}{\times_m}
U^{(m)}
}
$$

である。

modeごとに

$$
P_m
=
U^{(m)}U^{(m)\mathsf T}
$$

と置けば、$P_m$ はfactor列空間への直交射影であり、再構成は

$$
\hat{\mathcal X}
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
P_m
$$

とも書ける。

---

## 3. なぜ「誤差最小化」と「core norm最大化」が同じか

HOOIの更新式を理解するうえで重要な途中式である。

まず、

$$
\begin{aligned}
\|\mathcal X-\hat{\mathcal X}\|_F^2
={}&
\|\mathcal X\|_F^2
-2\langle\mathcal X,\hat{\mathcal X}\rangle_F
+\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

$\hat{\mathcal X}$ は直交射影なので、残差

$$
\mathcal R
=
\mathcal X-\hat{\mathcal X}
$$

は射影部分 $\hat{\mathcal X}$ と直交する。

$$
\langle\mathcal R,\hat{\mathcal X}\rangle_F=0
$$

したがって、

$$
\begin{aligned}
\langle\mathcal X,\hat{\mathcal X}\rangle_F
&=
\langle\mathcal R+\hat{\mathcal X},\hat{\mathcal X}\rangle_F\\
&=
\langle\mathcal R,\hat{\mathcal X}\rangle_F
+
\|\hat{\mathcal X}\|_F^2\\
&=
\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\hat{\mathcal X}\|_F^2
}
$$

となる。

さらに、列直交factorをmode productしてもFrobenius normは変わらない。1 modeだけで確認すると、

$$
\mathcal Y
=
\mathcal G\times_nU^{(n)}
$$

ならunfoldingで

$$
Y_{(n)}
=
U^{(n)}G_{(n)}.
$$

したがって、

$$
\begin{aligned}
\|\mathcal Y\|_F^2
&=\|Y_{(n)}\|_F^2\\
&=\operatorname{tr}
\left[
(U^{(n)}G_{(n)})^{\mathsf T}
(U^{(n)}G_{(n)})
\right]\\
&=\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}
U^{(n)\mathsf T}U^{(n)}
G_{(n)}
\right]\\
&=\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}G_{(n)}
\right]\\
&=\|\mathcal G\|_F^2.
\end{aligned}
$$

これを各modeへ順に適用すると、

$$
\|\hat{\mathcal X}\|_F^2
=
\|\mathcal G\|_F^2.
$$

したがって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\mathcal G\|_F^2
}
$$

である。

$\|\mathcal X\|_F^2$ は固定なので、

$$
\boxed{
\min\|\mathcal X-\hat{\mathcal X}\|_F^2
\quad\Longleftrightarrow\quad
\max\|\mathcal G\|_F^2
}
$$

となる。

これが「HOOIは射影後coreができるだけ多くのFrobenius energyを持つようにfactorを更新する」と言える理由である。

---

## 4. 1 factorだけ更新すると何を最大化するか

更新対象をmode $n\in\mathcal M$ とする。

他のactive modeを現在のfactorで射影したTensorを

$$
\boxed{
\mathcal Z^{(n)}
=
\mathcal X
\underset{m\in\mathcal M\setminus\{n\}}{\times_m}
U^{(m)\mathsf T}
}
$$

とする。

**更新対象自身のfactor $U^{(n)}$ はこのprojectionに使わない。**

全modeを圧縮する場合、$\mathcal Z^{(n)}$ のshapeは

$$
R_0\times\cdots\times R_{n-1}
\times I_n
\times R_{n+1}\times\cdots\times R_{N-1}
$$

である。

partial HOOIでは、$m\notin\mathcal M$ の次元は元の $I_m$ のまま残る。

mode $n$でunfoldした行列を

$$
Z
=
Z_{(n)}^{(n)}
$$

と略記する。

全mode圧縮なら、

$$
Z
\in
\mathbb R^{I_n\times\prod_{m\ne n}R_m}
$$

である。

候補factor

$$
U
=U^{(n)}
\in
\mathbb R^{I_n\times R_n},
\qquad
U^{\mathsf T}U=I_{R_n}
$$

を最後にmode $n$へ射影すると、coreのmode-$n$ unfoldingは

$$
G_{(n)}
=
U^{\mathsf T}Z
$$

となる。

前節から、他factorを固定したときは

$$
\boxed{
\max_{U^{\mathsf T}U=I}
\|U^{\mathsf T}Z\|_F^2
}
$$

を解けばよい。

---

## 5. なぜ更新factorが「上位左特異ベクトル」になるか

前節の目的を途中から展開する。

$$
\begin{aligned}
\|U^{\mathsf T}Z\|_F^2
&=
\operatorname{tr}
\left[
(U^{\mathsf T}Z)(U^{\mathsf T}Z)^{\mathsf T}
\right]\\
&=
\operatorname{tr}
\left[
U^{\mathsf T}ZZ^{\mathsf T}U
\right].
\end{aligned}
$$

したがって、1 factor更新は

$$
\boxed{
\max_{U^{\mathsf T}U=I_{R_n}}
\operatorname{tr}
\left(U^{\mathsf T}ZZ^{\mathsf T}U\right)
}
$$

という直交部分空間の最大化問題になる。

ここでSVDを

$$
Z
=
Q\Sigma V^{\mathsf T}
$$

とする。

すると、

$$
\begin{aligned}
ZZ^{\mathsf T}
&=
Q\Sigma V^{\mathsf T}
V\Sigma^{\mathsf T}Q^{\mathsf T}\\
&=
Q\Sigma\Sigma^{\mathsf T}Q^{\mathsf T}.
\end{aligned}
$$

$ZZ^{\mathsf T}$ の固有ベクトルは $Q$ の列、固有値は特異値の二乗

$$
\sigma_1^2\ge\sigma_2^2\ge\cdots\ge0
$$

である。

Rayleigh–Ritz / Ky Fanの最大化原理より、$R_n$ 次元の直交部分空間で

$$
\operatorname{tr}
(U^{\mathsf T}ZZ^{\mathsf T}U)
$$

を最大にするには、最大の $R_n$ 個の固有値に対応する固有ベクトルを列に取ればよい。

したがって、

$$
\boxed{
U^{(n)}
\leftarrow
Q_{[:,1:R_n]}
}
$$

という数学的な列番号表記になる。

Pythonの0始まりsliceで書けば、

```python
U_new = Q[:, :R_n]
```

である。

つまり、

$$
\boxed{
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\left(
\operatorname{left\ singular\ vectors}
\left(Z_{(n)}^{(n)}\right)
\right)
}
$$

となる。

```text
更新対象以外で射影
→ 対象modeでunfold
→ SVD
→ 上位左特異ベクトル
```

というHOOIの更新は、単なる経験則ではなく、**他factorを固定した局所問題を最適に解く更新**になっている。

ただしfactor全体を同時に見たTucker問題は非凸なので、各局所更新が最適でもグローバル最適解は保証されない。

---

## 6. Tucker-2での更新

Conv weight

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

でmode 0 / 1だけを圧縮する。

active mode集合は

$$
\mathcal M=\{0,1\}
$$

である。

### $U_{\mathrm{out}}$ の更新

現在の $U_{\mathrm{in}}$ を固定してmode 1を射影する。

$$
\boxed{
Z_{\mathrm{out}}
=
W
\times_1
U_{\mathrm{in}}^{\mathsf{T}}
}
$$

shapeは

$$
(C_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
\rightarrow
(C_{\mathrm{out}},R_{\mathrm{in}},K_h,K_w)
$$

である。

mode 0 unfoldingは

$$
(Z_{\mathrm{out}})_{(0)}
\in
\mathbb R^{
C_{\mathrm{out}}
\times
(R_{\mathrm{in}}K_hK_w)
}
$$

になる。その上位左特異ベクトルを使って

$$
\boxed{
U_{\mathrm{out}}
\leftarrow
\operatorname{top\text{-}}R_{\mathrm{out}}
\left(
\operatorname{left\ singular\ vectors}
\left[
(W\times_1U_{\mathrm{in}}^{\mathsf{T}})_{(0)}
\right]
\right)
}
$$

とする。

### $U_{\mathrm{in}}$ の更新

次に、**同じsweep内ですでに更新された新しい** $U_{\mathrm{out}}$ を使う。

$$
\boxed{
Z_{\mathrm{in}}
=
W
\times_0
U_{\mathrm{out}}^{\mathsf{T}}
}
$$

shapeは

$$
(C_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
\rightarrow
(R_{\mathrm{out}},C_{\mathrm{in}},K_h,K_w)
$$

で、mode 1 unfoldingは

$$
(Z_{\mathrm{in}})_{(1)}
\in
\mathbb R^{
C_{\mathrm{in}}
\times
(R_{\mathrm{out}}K_hK_w)
}
$$

になる。したがって、

$$
\boxed{
U_{\mathrm{in}}
\leftarrow
\operatorname{top\text{-}}R_{\mathrm{in}}
\left(
\operatorname{left\ singular\ vectors}
\left[
(W\times_0U_{\mathrm{out}}^{\mathsf{T}})_{(1)}
\right]
\right)
}
$$

とする。

---

## 7. Tucker-2の1 sweepを $t\rightarrow t+1$ で書く

初期値をHOSVDで

$$
U_{\mathrm{out}}^{(0)},
\qquad
U_{\mathrm{in}}^{(0)}
$$

とする。

iteration $t$ で、まず

$$
Z_{\mathrm{out}}^{(t)}
=
W\times_1
(U_{\mathrm{in}}^{(t)})^{\mathsf T}
$$

を作り、

$$
U_{\mathrm{out}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{out}}
\operatorname{leftSVD}
\left[(Z_{\mathrm{out}}^{(t)})_{(0)}\right]
$$

と更新する。

次に**更新済み**factorを使って、

$$
Z_{\mathrm{in}}^{(t)}
=
W\times_0
(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
$$

を作り、

$$
U_{\mathrm{in}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{in}}
\operatorname{leftSVD}
\left[(Z_{\mathrm{in}}^{(t)})_{(1)}\right]
$$

とする。

この2更新で1 sweepである。

3階Tensorなら、今回の実装はGauss-Seidel型に

| 更新対象 | projectionに使うfactor | 更新後 |
| ---: | --- | --- |
| 0 | $U_1,U_2$ | 新しい $U_0$ |
| 1 | 新しい $U_0$, $U_2$ | 新しい $U_1$ |
| 2 | 新しい $U_0$, 新しい $U_1$ | 新しい $U_2$ |

と進む。

同じsweep内で最新factorを使う点が、自作 `hooi_sweep()` の重要な仕様である。

---

## 8. core・再構成・errorを $t+1$ まで追う

factorを1 sweep更新した後、

$$
\boxed{
G^{(t+1)}
=
W
\times_0
(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
\times_1
(U_{\mathrm{in}}^{(t+1)})^{\mathsf T}
}
$$

を計算する。

再構成は

$$
\boxed{
\hat W^{(t+1)}
=
G^{(t+1)}
\times_0U_{\mathrm{out}}^{(t+1)}
\times_1U_{\mathrm{in}}^{(t+1)}
}
$$

である。

relative Frobenius errorは

$$
\boxed{
e_{t+1}
=
\frac{
\lVert W-\hat W^{(t+1)}\rVert_F
}{
\lVert W\rVert_F
}
}
$$

となる。

---

## 9. 収束判定

単純な絶対差なら、

$$
|e_t-e_{t-1}|<\varepsilon
$$

で止められる。

相対改善率を明示する別案なら、

$$
\frac{|e_t-e_{t-1}|}
{\max(|e_{t-1}|,\epsilon)}
<\mathrm{tol}
$$

と書ける。

現在のsrc実装では、絶対許容と相対許容を合わせて

$$
\boxed{
|e_t-e_{t-1}|
\leq
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}|e_{t-1}|
}
$$

を使う。

この形なら、誤差の絶対スケールが変わっても相対的な変化を考慮できる。

また最大反復回数 `max_iter` を置いて無限反復を防ぐ。

現在の既定値は、

```text
max_iter = 30
abs_tol = 1e-8
rel_tol = 1e-5
```

である。

---

## 10. HOOIが同rankでも意味を持つ理由

HOOIはrankを変更しない。

したがってHOSVDとHOOIで、

- core shape
- factor shape
- 圧縮後の層構造
- parameters
- MACs

は同じ。

違うのはfactor/coreの**値**であり、直接改善する対象は

$$
\lVert W-\hat W\rVert_F
$$

である。

今回のCIFAR-10 `conv2` では、balanced rank $(32,16)$ で

```text
HOSVD error = 0.449042
HOOI error  = 0.442504
```

となり、同rankでweight近似は改善した。

ただし圧縮直後validation accuracyは

```text
HOSVD = 0.6280
HOOI  = 0.6202
```

で、weight誤差改善がaccuracy改善には直結しなかった。

---

## 11. なぜweight誤差とaccuracyが一致しないか

HOOIの直接目的は

$$
\min\lVert W-\hat W\rVert_F^2
$$

であり、分類モデルの目的

$$
\min\mathcal{L}_{\mathrm{CE}}
$$

ではない。

weight空間では全要素の誤差を同じ二乗尺度で評価するが、実データがよく使う入力方向や分類境界に重要な方向は一様ではない。

したがって、

```text
weight reconstruction quality
≠
layer output quality
≠
logit quality
≠
classification accuracy
```

と考える必要がある。

添付で検討したtask-awareな発展としては、例えば次の目的がある。

| 目的 | 代表的な誤差 |
| --- | --- |
| 元weightを再現 | Weight Frobenius |
| 中間表現を再現 | Feature-map MSE |
| baseline出力を再現 | Logit MSE / KD |
| 正解ラベルへ最適化 | Cross Entropy |

ただし、標準HOOIのSVDによる閉形式更新はFrobenius二乗誤差の構造を利用している。Cross Entropy等へ目的を変更する場合は、単にHOOIの誤差関数を差し替えるのではなく、HOSVD/HOOIを初期化としてTucker-2層を作り、通常のbackpropagationでfine-tuningするのが今回の実験に対応する。

fine-tuningで補助損失を使うなら、例えば

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda_{\mathrm{feature}}\mathcal{L}_{\mathrm{feature}}
+
\lambda_{\mathrm{logit}}\mathcal{L}_{\mathrm{logit}}
$$

のような発展が考えられる。ただし今回の実験では標準Cross Entropyによるfine-tuningを主に確認しており、この補助損失自体を実験したわけではない。

---

## 12. HOSVDとHOOIの使い分け

```text
HOSVD
→ 高速な標準分解
→ rank探索
→ HOOIの初期解

HOOI
→ 選んだrank・重要layerの精密化
→ 同rankでweight再構成誤差を改善
```

大量のrank候補を探索する段階ではHOSVDが合理的。HOOIは、候補を絞った後のオフライン精密化として使いやすい。

---

## 13. srcとの対応

```text
src/nn_compression/compression/hooi.py
├─ has_converged
├─ hooi_sweep
├─ core_from_factors
└─ hooi
```

`ranks` のkeyが「圧縮・更新対象mode」を表すため、

```python
ranks = {0: r0, 1: r1, 2: r2}
```

なら通常の3-mode HOOI、

```python
ranks = {0: rank_out, 1: rank_in}
```

なら4階Conv weightに対するpartial HOOI / Tucker-2として同じ実装を再利用できる。

より長い途中導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]] にも残す。

検証結果：

- [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
- [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
