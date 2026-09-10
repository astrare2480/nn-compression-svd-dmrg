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

Tucker分解は、Tensorを**core Tensorと各modeのfactor行列**で表す方法である。

$N$階Tensorを

$$
\mathcal X
\in
\mathbb R^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

とする。

全modeを圧縮するTucker近似は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0 U^{(0)}
\times_1 U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

である。

factorは

$$
U^{(n)}\in\mathbb R^{I_n\times R_n},
$$

coreは

$$
\mathcal G
\in
\mathbb R^{R_0\times R_1\times\cdots\times R_{N-1}}
$$

となる。

HOSVDは、各modeの元Tensor unfoldingへSVDを行い、上位左特異ベクトルからfactorを作る。

---

## 1. exact multilinear rankと指定Tucker rankを区別する

Tensor $\mathcal X$ 自身のmultilinear rankは、各mode unfoldingの行列rankを並べたものとして

$$
\boxed{
\operatorname{rank}_{\mathrm{ML}}(\mathcal X)
=
\left(
\operatorname{rank}(X_{(0)}),
\operatorname{rank}(X_{(1)}),
\ldots,
\operatorname{rank}(X_{(N-1)})
\right)
}
$$

と定義する。

一方、近似計算でこちらが選ぶ

$$
(R_0,R_1,\ldots,R_{N-1})
$$

は**target Tucker ranks**である。

Tucker近似

$$
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\cdots
\times_{N-1}U^{(N-1)}
$$

では、mode $n$ unfoldingのrankはfactor列数を超えないため

$$
\boxed{
\operatorname{rank}(\hat X_{(n)})
\le R_n
}
$$

となる。

したがって、学習ノートで $(R_0,\ldots,R_{N-1})$ を「Tucker rank」と呼ぶ場合は、厳密には**指定する上限rank / target multilinear rank**として読む。

---

## 2. factorとcoreのshape

factor

$$
U^{(n)}
\in
\mathbb R^{I_n\times R_n}
$$

はrank空間から元mode空間への写像である。

$$
U^{(n)}:
\mathbb R^{R_n}
\rightarrow
\mathbb R^{I_n}
$$

転置は元空間からrank空間への射影に使う。

$$
U^{(n)\mathsf T}:
\mathbb R^{I_n}
\rightarrow
\mathbb R^{R_n}
$$

列直交なら

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}.
$$

全mode Tuckerのcoreは

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

各mode productにより

$$
I_n\rightarrow R_n
$$

と変わるので、最終的に

$$
\mathcal G
\in
\mathbb R^{R_0\times\cdots\times R_{N-1}}
$$

となる。

再構成は逆方向に

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

で、各modeが

$$
R_n\rightarrow I_n
$$

へ戻る。

---

## 3. HOSVDのfactorを1 modeずつ作る

mode $n$ unfoldingは

$$
X_{(n)}
\in
\mathbb R^{
I_n\times\prod_{m\ne n}I_m
}.
$$

Reduced SVDを

$$
X_{(n)}
=
Q_n\Sigma_nV_n^{\mathsf T}
$$

とする。

左特異ベクトルを

$$
Q_n
=
\begin{bmatrix}
q^{(n)}_1&q^{(n)}_2&\cdots
\end{bmatrix}
$$

と書けば、HOSVD factorは

$$
\boxed{
U^{(n)}
=
\begin{bmatrix}
q^{(n)}_1&q^{(n)}_2&\cdots&q^{(n)}_{R_n}
\end{bmatrix}
}
$$

である。

Pythonの0始まりsliceなら

```python
U_n = Q_n[:, :R_n]
```

となる。

`[:, 1:R_n]` では第1列を落としてしまうので別物である。

### HOSVDで重要な点

factorを求めるときは、それぞれ**元の同じ $\mathcal X$** をunfoldする。

```text
X → unfold mode 0 → SVD → U^(0)
X → unfold mode 1 → SVD → U^(1)
X → unfold mode 2 → SVD → U^(2)
...
```

先に求めたfactorで $X$ を射影してから次factorを計算するのではない。

この「各modeを独立に初期化する」点がHOOIとの大きな違いである。

---

## 4. HOSVDのcoreと再構成

factorをすべて得た後、元Tensorをfactor転置で射影する。

$$
\begin{aligned}
\mathcal X
&\xrightarrow{\times_0U^{(0)\mathsf T}}
\mathcal X^{(1)}\\
&\xrightarrow{\times_1U^{(1)\mathsf T}}
\mathcal X^{(2)}\\
&\quad\vdots\\
&\xrightarrow{\times_{N-1}U^{(N-1)\mathsf T}}
\mathcal G.
\end{aligned}
$$

したがって

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

再構成は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

となる。

### 小さい3階Tensorでfactor・core・再構成を全要素確認する

各modeの次元が2である3階Tensorを、2枚のsliceで

$$
X_{:,:,0}
=
\begin{pmatrix}
3&0\\
0&0
\end{pmatrix},
\qquad
X_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix}
$$

とする。すなわち、非零要素は

$$
X_{0,0,0}=3,
\qquad
X_{1,1,1}=1
$$

だけである。各mode unfoldingで、列側の複合添字を辞書順に並べると

$$
X_{(0)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix},
\quad
X_{(1)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix},
\quad
X_{(2)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix}.
$$

この例では3つのunfoldingが同じ数値行列になるが、それぞれ異なるmodeを行indexにしている。どのmodeでも

$$
X_{(n)}X_{(n)}^{\mathsf T}
=
\begin{pmatrix}
9&0\\
0&1
\end{pmatrix}
$$

なので、特異値は $3,1$、左特異ベクトルは標準基底である。指定Tucker rankを $(1,1,1)$ とすると、HOSVDは各modeを**元のTensorから独立に**計算して

$$
U^{(0)}
=
U^{(1)}
=
U^{(2)}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}
$$

を得る。coreは1要素だけになり、

$$
\begin{aligned}
G_{0,0,0}
&=
\sum_{i_0=0}^{1}
\sum_{i_1=0}^{1}
\sum_{i_2=0}^{1}
U^{(0)}_{i_0,0}
U^{(1)}_{i_1,0}
U^{(2)}_{i_2,0}
X_{i_0,i_1,i_2}\\
&=
1\cdot1\cdot1\cdot X_{0,0,0}\\
&=3.
\end{aligned}
$$

したがって再構成は

$$
\hat X_{:,:,0}
=
\begin{pmatrix}
3&0\\
0&0
\end{pmatrix},
\qquad
\hat X_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix}.
$$

残差も全要素で書けば

$$
(X-\hat X)_{:,:,0}
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix},
\qquad
(X-\hat X)_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix},
$$

よって

$$
\lVert X-\hat X\rVert_F
=
\sqrt{0^2+1^2}
=1
$$

となる。この例では、一般式の「各mode unfoldingからfactorを作る」「全factorを得た後にcoreへ射影する」「coreから元のshapeへ戻す」という3段階を、全要素で追跡できる。

---

## 5. Partial HOSVD / Tucker-2

全modeを圧縮する必要はない。

圧縮対象mode集合を

$$
\mathcal M
\subseteq
\{0,1,\ldots,N-1\}
$$

とする。

partial Tuckerでは

$$
\boxed{
\mathcal G
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
U^{(m)\mathsf T}
}
$$

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\underset{m\in\mathcal M}{\times_m}
U^{(m)}
}
$$

となり、$m\notin\mathcal M$ のmodeは元shapeのままcoreに残る。

Conv2d weight

$$
W
\in
\mathbb R^{
C_{out}\times C_{in}\times K_h\times K_w
}
$$

でchannel mode 0 / 1だけを圧縮するTucker-2では

$$
\mathcal M=\{0,1\}.
$$

factorは

$$
U_{out}
\in
\mathbb R^{C_{out}\times R_{out}},
$$

$$
U_{in}
\in
\mathbb R^{C_{in}\times R_{in}}.
$$

coreは

$$
\boxed{
G
=
W
\times_0U_{out}^{\mathsf T}
\times_1U_{in}^{\mathsf T}
}
$$

で、shapeは

$$
\boxed{
G
\in
\mathbb R^{R_{out}\times R_{in}\times K_h\times K_w}
}.
$$

再構成は

$$
\boxed{
\hat W
=
G
\times_0U_{out}
\times_1U_{in}
}.
$$

---

## 6. Tucker-2の要素表示をmode productから出す

まずmode 0を展開すると

$$
H_{o,\beta,a,b}
=
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}.
$$

次にmode 1へ $U_{in}$ を掛けると

$$
\hat W_{o,i,a,b}
=
\sum_{\beta=1}^{R_{in}}
H_{o,\beta,a,b}
U^{in}_{i,\beta}.
$$

$H$ を代入して

$$
\begin{aligned}
\hat W_{o,i,a,b}
&=
\sum_{\beta=1}^{R_{in}}
\left(
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
\right)
U^{in}_{i,\beta}\\
&=
\boxed{
\sum_{\alpha=1}^{R_{out}}
\sum_{\beta=1}^{R_{in}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
}.
\end{aligned}
$$

---

## 7. 再構成誤差

Tensor Frobenius normは

$$
\boxed{
\|\mathcal X\|_F
=
\sqrt{
\sum_{i_0=1}^{I_0}
\sum_{i_1=1}^{I_1}
\cdots
\sum_{i_{N-1}=1}^{I_{N-1}}
X_{i_0,\ldots,i_{N-1}}^2
}
}
$$

である。

relative errorは

$$
\boxed{
e_{rel}
=
\frac{
\|\mathcal X-\hat{\mathcal X}\|_F
}{
\|\mathcal X\|_F
}
}
$$

で、Conv weightなら

$$
\boxed{
e_W
=
\frac{
\|W-\hat W\|_F
}{
\|W\|_F
}
}.
$$

これは元weightへの近さを測る指標であって、classification lossを直接測るものではない。

---

## 8. Tuckerのparameter数

全modeを圧縮する場合、coreの要素数は

$$
P_{core}
=
\prod_{n=0}^{N-1}R_n.
$$

factor $n$ の要素数は

$$
P_n
=I_nR_n.
$$

したがって合計は

$$
\begin{aligned}
P_{Tucker}
&=P_{core}+\sum_{n=0}^{N-1}P_n\\
&=
\boxed{
\prod_{n=0}^{N-1}R_n
+
\sum_{n=0}^{N-1}I_nR_n
}.
\end{aligned}
$$

partial Tuckerでは、圧縮しないmodeの次元がcoreにそのまま残るため、この全mode用の式を機械的に使わず、実際のcore shapeから数える。

Conv Tucker-2の式は [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]] で導出する。

---

## 9. HOSVDは行列SVDと同じ意味でのglobal optimumではない

行列のtruncated SVDでは、Eckart–Young–Mirskyにより固定rankの最良近似を得る。

HOSVDでは各modeを独立に最適化するため、高階Tensorの固定target multilinear rank

$$
(R_0,\ldots,R_{N-1})
$$

に対する全体最適解そのものとは限らない。

```text
行列 truncated SVD
→ 固定rankに対するglobal best approximation

HOSVD
→ mode-wise SVDを組み合わせた高速な初期近似

HOOI
→ HOSVD factorを初期値にして同rankで交互精密化
```

> [!note] 補足一般理論
> この節の「global optimumではない」という区別は標準的なTucker理論による補足であり、添付資料の実験値から導いた主張ではない。今回の実験では実際にHOOIがHOSVDよりFrobenius errorを下げたことを、別の検証ノートで確認している。

---

## 10. srcとの対応

```text
src/nn_compression/compression/tucker.py
├─ hosvd
└─ reconstruct_tucker

src/nn_compression/metrics/tensor_approximation.py
└─ relative_frobenius_error
```

実装上の重要点は、HOSVD factorを各modeについて**元のXから独立に**求めること。

詳細な途中式：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
