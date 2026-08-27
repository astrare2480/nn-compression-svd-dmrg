---
title: Tucker分解とHOSVD
aliases:
  - Tucker decomposition
  - HOSVD
  - partial Tucker
  - Tucker rank
tags:
  - Tucker
  - HOSVD
  - Tensor
  - 低ランク近似
  - NN圧縮
---

# Tucker分解とHOSVD

## サマリー

Tucker分解は、Tensorを**小さいcore Tensorと、各modeのfactor行列**で近似する。

$N$階Tensor

$$
\mathcal{X}
\in
\mathbb{R}^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

に対し、Tucker近似は

$$
\boxed{
\mathcal{X}
\approx
\mathcal{G}
\times_0 U^{(0)}
\times_1 U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

と書く。

factorは

$$
U^{(n)}
\in
\mathbb{R}^{I_n\times R_n}
$$

coreは

$$
\mathcal{G}
\in
\mathbb{R}^{R_0\times R_1\times\cdots\times R_{N-1}}
$$

である。

$(R_0,R_1,\ldots,R_{N-1})$ がmultilinear rank（Tucker rank）を表す。

HOSVDは、このfactorを**各modeの元Tensor unfoldingに対するSVDから独立に求める**標準的な初期分解法である。

---

## 1. Tucker分解の意味

行列SVDでは、1つのrank $r$ を使って

$$
W_r=U_r\Sigma_rV_r^{\mathsf{T}}
$$

と近似した。

Tensorではmodeごとに異なるrankを持てる。

$$
(R_0,R_1,\ldots,R_{N-1})
$$

したがって、

```text
mode 0は強く圧縮
mode 1は弱く圧縮
mode 2は圧縮しない
```

のような設計が可能になる。

Tucker-2でConv weightのchannel軸だけを圧縮するのは、この性質を使っている。

---

## 2. factorとcoreの役割

factor $U^{(n)}$ は、mode $n$ のrank空間と元空間の対応を持つ。

$$
U^{(n)}:
\mathbb{R}^{R_n}
\rightarrow
\mathbb{R}^{I_n}
$$

転置は逆方向の射影として使う。

$$
U^{(n)\mathsf{T}}:
\mathbb{R}^{I_n}
\rightarrow
\mathbb{R}^{R_n}
$$

factorの列が直交規格化されているとき、現在のfactorに対応するcoreは

$$
\boxed{
\mathcal{G}
=
\mathcal{X}
\times_0 U^{(0)\mathsf{T}}
\times_1 U^{(1)\mathsf{T}}
\cdots
\times_{N-1}U^{(N-1)\mathsf{T}}
}
$$

で計算できる。

再構成は逆向きに

$$
\boxed{
\hat{\mathcal{X}}
=
\mathcal{G}
\times_0 U^{(0)}
\times_1 U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

とする。

---

## 3. HOSVD

mode $n$ unfoldingを

$$
X_{(n)}
\in
\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

とする。

その行列SVDを

$$
X_{(n)}
=
U_n\Sigma_nV_n^{\mathsf{T}}
$$

とし、上位 $R_n$ 本の左特異ベクトルをfactorに採用する。

$$
U^{(n)}
=
\begin{bmatrix}
u^{(n)}_1 & u^{(n)}_2 & \cdots & u^{(n)}_{R_n}
\end{bmatrix}
$$

概念的には、

$$
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\left(
\operatorname{left\ singular\ vectors}(X_{(n)})
\right)
$$

である。

全対象modeでこれを**元の同じTensorに対して独立に**行い、最後にfactorの転置を掛けてcoreを作る。

```text
元Tensor X
├─ mode 0 unfold → SVD → U0
├─ mode 1 unfold → SVD → U1
├─ ...
└─ mode n unfold → SVD → Un

X ×0 U0.T ×1 U1.T ... → core G
```

### 重要

HOSVDのfactor計算は、先に求めたfactorを使って次のfactorを求めるわけではない。

```text
HOSVD:
X → U0
X → U1
X → U2
```

と各modeを独立に見る。

これがHOOIとの最大の違いになる。

---

## 4. Partial HOSVD / Tucker-2

全modeを圧縮する必要はない。

Conv2d weight

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

で、mode 0 / 1だけを対象にすれば、

$$
U_{\mathrm{out}}
\in
\mathbb{R}^{C_{\mathrm{out}}\times R_{\mathrm{out}}}
$$

$$
U_{\mathrm{in}}
\in
\mathbb{R}^{C_{\mathrm{in}}\times R_{\mathrm{in}}}
$$

$$
G
\in
\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times K_h\times K_w}
$$

を得る。

近似式は

$$
\boxed{
W
\approx
G
\times_0 U_{\mathrm{out}}
\times_1 U_{\mathrm{in}}
}
$$

要素表示では、

$$
\boxed{
W_{o,i,a,b}
\approx
\sum_{\alpha=1}^{R_{\mathrm{out}}}
\sum_{\beta=1}^{R_{\mathrm{in}}}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}
}
$$

となる。

coreは

$$
\boxed{
G
=
W
\times_0 U_{\mathrm{out}}^{\mathsf{T}}
\times_1 U_{\mathrm{in}}^{\mathsf{T}}
}
$$

である。

---

## 5. 再構成誤差

今回の実験では、分解品質を相対Frobenius誤差で統一する。

$$
\boxed{
e_{\mathrm{rel}}
=
\frac{
\lVert \mathcal{X}-\hat{\mathcal{X}}\rVert_F
}{
\lVert \mathcal{X}\rVert_F
}
}
$$

Conv weightなら、

$$
e_W
=
\frac{
\lVert W-\hat W\rVert_F
}{
\lVert W\rVert_F
}
$$

である。

TensorのFrobeniusノルムは、全要素の二乗和の平方根。

$$
\lVert\mathcal{X}\rVert_F
=
\sqrt{
\sum_{i_0}\sum_{i_1}\cdots\sum_{i_{N-1}}
X_{i_0,i_1,\ldots,i_{N-1}}^2
}
$$

### 注意

小さいweight relative errorは「元weightに近い」ことを表すが、分類accuracyを直接最適化しているわけではない。

今回のHOOI実験では、この違いが実際に観測された。

---

## 6. Tuckerのパラメータ数

全modeをTucker分解すると、factorとcoreの要素数は概念的に

$$
N_{\mathrm{Tucker}}
=
\prod_{n=0}^{N-1}R_n
+
\sum_{n=0}^{N-1}I_nR_n
$$

となる。

ただしpartial Tuckerでは、圧縮しないmodeをcoreにそのまま残すため、そのshapeに合わせて数える。

Conv Tucker-2の具体式は
[[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
で扱う。

---

## 7. HOSVDは何を保証するか

行列のtruncated SVDでは、固定rankにおける最良近似をEckart–Young–Mirskyの定理で得られる。

一方、高階Tensorの固定multilinear rank近似は一般に行列ほど単純ではなく、各modeを独立にSVDするHOSVDが固定rankの**全体最適解そのものになるとは限らない**。

HOSVDは、

- 高速
- 一度のmode-wise SVDでfactorを得られる
- 実装しやすい
- HOOIの良い初期値になる

という位置づけで使う。

HOOIは、このHOSVD解を同じrankのまま反復的に精密化する。

---

## 8. srcとの対応

```text
src/nn_compression/compression/tucker.py
├─ hosvd
└─ reconstruct_tucker

src/nn_compression/metrics/tensor_approximation.py
└─ relative_frobenius_error
```

HOSVD実装では、各factorを**元のXから独立に**求めることが重要である。

次に読む：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
