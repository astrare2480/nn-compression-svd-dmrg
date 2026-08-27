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

今回のsrcでもfactorはSVDの左特異ベクトルから作るため、列直交なfactorを使う。この直交制約を明示すると、現在のfactorに対する最適なcoreは

$$
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
$$

となる。

したがって、factor側から見ると、射影後coreのFrobeniusノルムを大きくする問題としても捉えられる。

$$
\boxed{
\max_{
U^{(0)},\ldots,U^{(N-1)}
}
\left\|
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
\right\|_F^2
}
$$

ただし各factorには

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
$$

を課す。

この問題はfactorを同時に求めると非凸なので、HOOIは1 modeずつ交互に更新する。したがって「必ずグローバル最小値を返す」とは考えず、HOSVDを高速な初期解として使い、同じrankで再構成誤差を改善する精密化法として理解する。

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

## 2. 1 factorの更新式

更新対象をmode $n$とする。

他modeを現在のfactorで射影したTensorを

$$
\boxed{
\mathcal{Z}^{(n)}
=
\mathcal{X}
\underset{m\neq n}{\times_m}
U^{(m)\mathsf{T}}
}
$$

と略記する。これは、対象mode $n$を除くすべての圧縮対象modeについてmode productを順に行う、という意味である。

ここで**更新したいmode自身のfactorはprojectionに使わない**。

$\mathcal Z^{(n)}$ のshapeは、全modeを圧縮する場合、

$$
R_0\times\cdots\times R_{n-1}
\times I_n
\times R_{n+1}\times\cdots\times R_{N-1}
$$

となる。

これをmode $n$でunfoldすると、

$$
Z_{(n)}^{(n)}
\in
\mathbb R^{
I_n\times\prod_{m\neq n}R_m
}
$$

である。

そのSVDの上位 $R_n$ 本の左特異ベクトルを新しいfactorにする。

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

つまり、

```text
更新対象以外で縮約
→ 対象modeでunfold
→ SVD
→ 左特異ベクトル上位rank本
```

である。

partial HOOIでは、`ranks` に含まれるmodeだけをこの積と更新の対象にする。圧縮しないmodeは元の次元のまま残る。

---

## 3. Tucker-2での更新

Conv weight

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

でmode 0 / 1だけを圧縮する。

### $U_{\mathrm{out}}$ の更新

現在の $U_{\mathrm{in}}$ を固定して、mode 1を射影する。

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

## 4. 1 sweep

1 sweepとは、圧縮対象の全modeを一度ずつ更新すること。

Tucker-2では、

```text
1. U_outを更新
2. 更新済みU_outを使ってU_inを更新
```

で1 sweepになる。

3階Tensorなら、今回の実装はGauss-Seidel型に

| 更新対象 | projectionに使うfactor | 更新後 |
| ---: | --- | --- |
| 0 | $U_1,U_2$ | 新しい $U_0$ |
| 1 | 新しい $U_0$, $U_2$ | 新しい $U_1$ |
| 2 | 新しい $U_0$, 新しい $U_1$ | 新しい $U_2$ |

と進む。

同じsweep内で最新factorを使う点が、自作 `hooi_sweep()` の重要な仕様である。

---

## 5. coreの更新

factorを1 sweep更新した後、現在のfactorでcoreを計算する。

$$
\boxed{
\mathcal{G}
=
\mathcal{X}
\times_0 U^{(0)\mathsf{T}}
\times_1 U^{(1)\mathsf{T}}
\cdots
}
$$

Tucker-2なら、

$$
\boxed{
G
=
W
\times_0 U_{\mathrm{out}}^{\mathsf{T}}
\times_1 U_{\mathrm{in}}^{\mathsf{T}}
}
$$

再構成して誤差を測る。

$$
\hat W
=
G
\times_0 U_{\mathrm{out}}
\times_1 U_{\mathrm{in}}
$$

$$
e_t
=
\frac{\lVert W-\hat W_t\rVert_F}{\lVert W\rVert_F}
$$

---

## 6. 収束判定

単純な絶対差なら、

$$
|e_t-e_{t-1}|<\varepsilon
$$

で止められる。

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

## 7. HOOIが同rankでも意味を持つ理由

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

## 8. なぜweight誤差とaccuracyが一致しないか

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

## 9. HOSVDとHOOIの使い分け

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

## 10. srcとの対応

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

検証結果：

- [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
- [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
