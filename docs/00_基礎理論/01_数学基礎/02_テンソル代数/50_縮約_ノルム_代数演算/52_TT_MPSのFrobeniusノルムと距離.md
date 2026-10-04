---
title: TT_MPSのFrobeniusノルムと距離
tags:
  - TT
  - MPS
  - tensor-train
  - Frobenius
  - norm
  - distance
  - numerical-stability
---

# TT/MPSのFrobeniusノルムと距離

このノートでは、[[51_TT_MPSの内積とenvironment縮約]] で導いたTT内積から、Frobeniusノルムと2本のTT間の距離を導く。

重要なのは、差 $\mathcal A-\mathcal B$ を表す新しいTTを明示的に作らず、3回の内積から距離を得られることである。加減算TTの構成とrank増加は [[53_TT_MPSの加減算とrank増加]]、PyTorch実装と数値検証は [[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]] に分ける。

---

## 0. このノートの到達点

実数値TT $\mathcal A,\mathcal B$ に対し、

$$
\boxed{
\|\mathcal A\|_F
=
\sqrt{\langle\mathcal A,\mathcal A\rangle}
}
$$

および

$$
\boxed{
\|\mathcal A-\mathcal B\|_F^2
=
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
}
$$

を使う。

どの内積もTTコアを左からenvironment縮約して求められるため、dense tensorも差のTTも不要である。

---

## 1. Frobeniusノルムは自己内積の平方根

$d$ 階tensorのFrobeniusノルムは

$$
\|\mathcal A\|_F
=
\sqrt{
\sum_{i_1=1}^{n_1}
\cdots
\sum_{i_d=1}^{n_d}
\mathcal A_{i_1,\ldots,i_d}^2
}.
$$

一方、自己内積は

$$
\begin{aligned}
\langle\mathcal A,\mathcal A\rangle
&=
\sum_{i_1,ldots,i_d}
\mathcal A_{i_1,ldots,i_d}
\mathcal A_{i_1,ldots,i_d}\\
&=
\sum_{i_1,ldots,i_d}
\mathcal A_{i_1,ldots,i_d}^2.
\end{aligned}
$$

従って、

$$
\|\mathcal A\|_F^2
=
\langle\mathcal A,\mathcal A\rangle,
\qquad
\|\mathcal A\|_F
=
\sqrt{\langle\mathcal A,\mathcal A\rangle}.
$$

TTのコア列を `cores` とすれば、概念的には

```python
norm_sq = tt_inner(cores, cores)
```

である。同じ `cores` を2回渡すのは、内積の2引数へ同一のtensor $\mathcal A$ を指定して自己内積を作るためである。

### 1.1 小さいdense例

$$
A=
\begin{pmatrix}
1 & -2\\
3 & 4
\end{pmatrix}
$$

なら、

$$
\begin{aligned}
\langle A,A\rangle_F
&=
1^2+(-2)^2+3^2+4^2\\
&=
1+4+9+16\\
&=30.
\end{aligned}
$$

従って、

$$
\|A\|_F=\sqrt{30}.
$$

行列でも高階tensorでも、全要素を1列に並べたベクトルのEuclideanノルムと同じである。

$$
\boxed{
\|\mathcal A\|_F
=
\|\operatorname{vec}(\mathcal A)\|_2
}
$$

---

## 2. 2本のTT間の距離を展開する

差のFrobenius距離は

$$
\|\mathcal A-\mathcal B\|_F^2
=
\langle\mathcal A-\mathcal B,\mathcal A-\mathcal B\rangle.
$$

双線形性により、

$$
\begin{aligned}
\langle\mathcal A-\mathcal B,\mathcal A-\mathcal B\rangle
=
&\langle\mathcal A,\mathcal A\rangle
-\langle\mathcal A,\mathcal B\rangle\\
&-\langle\mathcal B,\mathcal A\rangle
+\langle\mathcal B,\mathcal B\rangle.
\end{aligned}
$$

実数のFrobenius内積は対称なので、

$$
\langle\mathcal B,\mathcal A\rangle
=
\langle\mathcal A,\mathcal B\rangle.
$$

したがって、

$$
\boxed{
\|\mathcal A-\mathcal B\|_F^2
=
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
}
\tag{1}
$$

である。距離は

$$
\boxed{
\|\mathcal A-\mathcal B\|_F
=
\sqrt{
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
}
}
$$

となる。

---

## 3. なぜ差のTTを作らなくてよいか

式 (1) で必要なのは

$$
\langle\mathcal A,\mathcal A\rangle,
\qquad
\langle\mathcal B,\mathcal B\rangle,
\qquad
\langle\mathcal A,\mathcal B\rangle
$$

の3量だけである。それぞれをleft environmentで縮約すればよい。

明示的な差TTを作ると、一般には内部bond dimensionが

$$
r_k^A+r_k^B
$$

まで増える。AとBがともに一様rank $r$ なら、構成直後のrankは最大 $2r$ である。中間コアの要素数はrankの積に比例するため、

$$
r^2
\longrightarrow
(2r)^2=4r^2
$$

と約4倍になり得る。

距離のためだけに差TTを構成し、その後roundingするより、3回の内積を直接計算した方が単純である。

---

## 4. TT-rounding誤差へ接続する

rounding前を $\mathcal X$、rounding後を $\widetilde{\mathcal X}$ とすれば、絶対誤差は

$$
e_{\mathrm{abs}}
=
\|\mathcal X-\widetilde{\mathcal X}\|_F
$$

である。相対誤差は $\|\mathcal X\|_F\ne0$ のとき

$$
e_{\mathrm{rel}}
=
\frac{
\|\mathcal X-\widetilde{\mathcal X}\|_F
}{
\|\mathcal X\|_F
}
$$

である。

保持された二乗ノルム比を確認するなら、

$$
\rho_{mathrm{norm}}
=
\frac{
\|\widetilde{\mathcal X}\|_F^2
}{
\|\mathcal X\|_F^2
}
$$

も内積だけで計算できる。

これらはdense復元が不可能な大規模tensorでも使える。ただし零テンソルでは相対誤差の分母が0になるため、別分岐が必要である。

---

## 5. 浮動小数点では距離二乗が微小に負になり得る

数学的には

$$
\|\mathcal A-\mathcal B\|_F^2\ge0
$$

である。しかし式 (1) は、大きさの近い値を引く。

$$
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
$$

$\mathcal A\simeq\mathcal B$ のときは、浮動小数点の丸めにより理論上0の値が

$$
-2\times10^{-16}
$$

のようになる場合がある。そのまま平方根を取ると `nan` になる。

PyTorchでは、平方根の直前に

```python
safe_distance_sq = distance_sq.clamp_min(0.0)
distance = torch.sqrt(safe_distance_sq)
```

とすれば、

$$
\operatorname{clamp\_min}(x,0)
=
\max(x,0)
$$

なので微小な負値を0へ戻せる。

### 5.1 `clamp_min`で大きな誤りを隠してはいけない

例えば

$$
\text{distance\_sq}=-10^{-3}
$$

や

$$
\text{distance\_sq}=-1
$$

まで負なら、単なる丸め誤差とは考えにくい。次を疑う。

- `einsum`の添字が誤っている
- AとBのphysical shapeが一致していない
- 複素MPSでbra側の共役がない
- dtypeやdeviceが意図せず混在している
- `inf` や `nan` が入力へ入っている

評価コードでは、スケールに応じた許容値 $\tau$ を決め、

$$
x<-\tau
$$

なら例外にし、

$$
-\tau\le x<0
$$

だけを0へ丸める方が安全である。

一方、学習lossの内部で `.item()` を使ってPython分岐すると、device同期やautograd上の扱いが変わる。学習用の微分可能処理と、検証用の厳密な異常検出は分けて設計する。

---

## 6. 複素MPSの場合

複素内積はsesquilinearであり、

$$
\langle\psi,\phi\rangle
=
\sum_{\boldsymbol i}
\overline{\psi_{\boldsymbol i}}
\phi_{\boldsymbol i}.
$$

自己内積は

$$
\langle\psi,\psi\rangle
=
\sum_{\boldsymbol i}|\psi_{\boldsymbol i}|^2
\in\mathbb R_{\ge0}
$$

である。距離二乗の一般形は

$$
\begin{aligned}
\|\psi-\phi\|_F^2
&=
\langle\psi,\psi\rangle
+
\langle\phi,\phi\rangle
-
\langle\psi,\phi\rangle
-
\langle\phi,\psi\rangle\\
&=
\langle\psi,\psi\rangle
+
\langle\phi,\phi\rangle
-
2\operatorname{Re}\langle\psi,\phi\rangle.
\end{aligned}
$$

従って実数版の `- 2 * inner_ab` をそのまま複素版へ流用するのではなく、共役と実部を正しく扱う必要がある。

---

## 7. dense検証で比較する3量

小規模問題では、TT-only計算をdense定義と照合する。

### 7.1 内積

$$
\langle\mathcal A,\mathcal B\rangle
=
\sum_{\boldsymbol i}
\mathcal A_{\boldsymbol i}
\mathcal B_{\boldsymbol i}.
$$

### 7.2 ノルム

$$
\|\mathcal A\|_F
=
\sqrt{
\sum_{\boldsymbol i}
\mathcal A_{\boldsymbol i}^2
}.
$$

### 7.3 距離

$$
\|\mathcal A-\mathcal B\|_F
=
\sqrt{
\sum_{\boldsymbol i}
(\mathcal A_{\boldsymbol i}-\mathcal B_{\boldsymbol i})^2
}.
$$

これら3量がfloat64の許容誤差内で一致すれば、少なくともその小規模例に対し、environment縮約、自己内積、距離恒等式が整合している。

ただしdense照合は検証用であり、本番の大規模TTでdense復元してよいという意味ではない。

---

## 8. 自己距離は重要な境界テスト

$\mathcal B=\mathcal A$ を代入すると、

$$
\begin{aligned}
\|\mathcal A-\mathcal A\|_F^2
&=
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal A,\mathcal A\rangle
-
2\langle\mathcal A,\mathcal A\rangle\\
&=0.
\end{aligned}
$$

したがって、

$$
\boxed{
\|\mathcal A-\mathcal A\|_F=0
}
$$

である。実装では完全な0または丸め誤差範囲の0になることを検査する。これは距離二乗の減算と微小負値処理を同時に確認する境界テストになる。

---

## 9. まとめ

内積縮約を1つ実装すれば、次が得られる。

$$
\begin{aligned}
\text{内積:}\quad
&\langle\mathcal A,\mathcal B\rangle,\\
\text{ノルム:}\quad
&\|\mathcal A\|_F
=
\sqrt{\langle\mathcal A,\mathcal A\rangle},\\
\text{距離:}\quad
&\|\mathcal A-\mathcal B\|_F
=
\sqrt{
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
}.
\end{aligned}
$$

距離公式の利点を理解するには、差のTTを実際に作るとrankがどう増えるかを確認する必要がある。次は [[53_TT_MPSの加減算とrank増加]] へ進む。
