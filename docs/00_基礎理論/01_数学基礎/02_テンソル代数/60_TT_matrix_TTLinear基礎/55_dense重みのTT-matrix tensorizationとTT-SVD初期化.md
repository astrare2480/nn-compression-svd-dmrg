---
title: dense重みのTT-matrix tensorizationとTT-SVD初期化
tags:
  - TT-matrix
  - tensorization
  - TT-SVD
  - reshape
  - permute
---

# dense重みのTT-matrix tensorizationとTT-SVD初期化

## 1. 目的

学習済みdense重み

$$
W\in\mathbb R^{m\times n}
$$

をTT-matrixへ変換するとき、通常のTT-SVDへそのまま2階行列を渡すのではない。出力側と入力側を同数のmodeへ分け、同じsiteの組 $(i_k,j_k)$ が一つのTT physical indexになるように並べ替える。

全体の流れは

$$
W
\longrightarrow
\mathcal W_{\mathrm{grouped}}
\longrightarrow
\mathcal W_{\mathrm{interleaved}}
\longrightarrow
\mathcal A
\longrightarrow
\{C^{(k)}\}
\longrightarrow
\{G^{(k)}\}
$$

である。

## 2. grouped順とinterleaved順を区別する

まず

$$
m=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

とする。dense行列をreshapeした直後は

$$
\mathcal W_{\mathrm{grouped}}
\in
\mathbb R^{m_1\times\cdots\times m_d\times n_1\times\cdots\times n_d}
$$

であり、軸の意味は

$$
(i_1,\ldots,i_d,j_1,\ldots,j_d)
$$

である。TT-matrixでは、同じsiteの出力・入力を隣接させたいので、

$$
(i_1,j_1,i_2,j_2,\ldots,i_d,j_d)
$$

へpermuteする。このtensorを $\mathcal W_{\mathrm{interleaved}}$ と呼ぶ。

groupedからinterleavedへの一般のaxis permutationは

$$
\boxed{
(0,d,1,d+1,\ldots,d-1,2d-1)
}
$$

である。Pythonの0始まりindexで書くと、

```python
perm_grouped_to_interleaved = [
    axis
    for k in range(d)
    for axis in (k, d + k)
]
```

となる。`perm` に入れるのはmode sizeではなく、「元tensorのaxis番号」である。

## 3. $d=3$ でaxis番号を全て書く

grouped順は

```text
axis:       0   1   2   3   4   5
meaning:   i1  i2  i3  j1  j2  j3
shape:     m1  m2  m3  n1  n2  n3
```

である。欲しい順は

```text
meaning:   i1  j1  i2  j2  i3  j3
```

なので、元axis番号をその順に並べると

$$
\boxed{(0,3,1,4,2,5)}
$$

となる。

```python
W_grouped = W.reshape(m1, m2, m3, n1, n2, n3)
W_interleaved = W_grouped.permute(0, 3, 1, 4, 2, 5)
```

重要なのは、$(i_1,j_1)$、$(i_2,j_2)$、$(i_3,j_3)$ を組にすることである。$(j_1,i_2)$ をcombineしているのではない。

## 4. shapeが同じでも軸の意味は同じとは限らない

例えば

$$
W\in\mathbb R^{6\times12},
\qquad
6=2\cdot3,
\qquad
12=3\cdot4
$$

とする。

$$
(m_1,m_2)=(2,3),
\qquad
(n_1,n_2)=(3,4)
$$

なら、grouped tensorのshapeは

$$
(m_1,m_2,n_1,n_2)=(2,3,3,4)
$$

である。interleaved後も

$$
(m_1,n_1,m_2,n_2)=(2,3,3,4)
$$

となり、数値としてのshapeは偶然同じである。しかし軸の意味と値の配置は異なる。

$$
(i_1,i_2,j_1,j_2)
\neq
(i_1,j_1,i_2,j_2).
$$

したがって、shapeだけをprintしてもpermuteの正しさは検証できない。任意のmulti-indexを選び、対応する要素が同じ位置関係を保つことまで確認する必要がある。

## 5. site内の二つのindexをcombineする

interleaved後は同じsiteの $(i_k,j_k)$ が隣接しているので、

$$
a_k=i_kn_k+j_k,
\qquad
q_k=m_kn_k
$$

として一つのphysical indexへcombineできる。

$$
\mathcal A
\in
\mathbb R^{q_1\times\cdots\times q_d}
$$

を

$$
\mathcal A[a_1,\ldots,a_d]
=
\mathcal W_{\mathrm{interleaved}}
[i_1,j_1,\ldots,i_d,j_d]
$$

で定義する。軸がすでに $(i_k,j_k)$ の順に隣接しているため、この段階の`reshape`は値の並べ替えを行わず、隣接した二軸をまとめるだけである。

## 6. mixed-radixとしてのflat index

$d=3$ の出力側を例にする。

$$
(m_1,m_2,m_3)=(2,3,2)
$$

なら、multi-indexとflat indexの対応は

$$
\boxed{
i=i_1(m_2m_3)+i_2m_3+i_3
}
$$

である。全要素を並べると

```text
(i1,i2,i3) -> i
(0,0,0) ->  0   (0,0,1) ->  1
(0,1,0) ->  2   (0,1,1) ->  3
(0,2,0) ->  4   (0,2,1) ->  5
(1,0,0) ->  6   (1,0,1) ->  7
(1,1,0) ->  8   (1,1,1) ->  9
(1,2,0) -> 10   (1,2,1) -> 11
```

となる。最後のaxis $i_3$ は1刻みで変わる。$i_2$ が1増えると $i_3$ の全組合せ $m_3=2$ 個を飛ばし、$i_1$ が1増えると $(i_2,i_3)$ の全組合せ $m_2m_3=6$ 個を飛ばす。

例えば

$$
(i_1,i_2,i_3)=(1,0,1)
$$

なら

$$
\begin{aligned}
i
&=1(3\cdot2)+0\cdot2+1\\
&=7.
\end{aligned}
$$

入力mode

$$
(n_1,n_2,n_3)=(2,2,3)
$$

と

$$
(j_1,j_2,j_3)=(0,1,2)
$$

なら

$$
\begin{aligned}
j
&=0(2\cdot3)+1\cdot3+2\\
&=5.
\end{aligned}
$$

したがって

$$
W[7,5]
=
\mathcal W_{\mathrm{grouped}}[1,0,1,0,1,2].
$$

同じsiteのcombined indexは

$$
a_1=1\cdot2+0=2,
\qquad
a_2=0\cdot2+1=1,
\qquad
a_3=1\cdot3+2=5
$$

なので、

$$
\boxed{W[7,5]=\mathcal A[2,1,5]}
$$

となる。

## 7. 通常のTT-SVDへ接続する

$\mathcal A$ は通常の $d$ 階tensorなので、既存のTT-SVDをそのまま適用できる。

$$
\mathcal A[a_1,\ldots,a_d]
\approx
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
C^{(1)}[0,a_1,\alpha_1]
\cdots
C^{(d)}[\alpha_{d-1},a_d,0].
$$

通常のTTコアは

$$
C^{(k)}
\in
\mathbb R^{r_{k-1}\times q_k\times r_k}
$$

である。$q_k=m_kn_k$ なので、

$$
\boxed{
G^{(k)}
=
\operatorname{reshape}
\left(
C^{(k)},
(r_{k-1},m_k,n_k,r_k)
\right)
}
$$

とすればTT-matrixコアになる。このreshapeでも、combined indexを再び $(i_k,j_k)$ の二軸へ分けているだけで、値の順序は変えない。

## 8. 2 siteのshapeを最後まで追う

先ほどの

$$
(m_1,m_2)=(2,3),
\qquad
(n_1,n_2)=(3,4)
$$

では、

$$
W:(6,12)
$$

$$
\mathcal W_{\mathrm{grouped}}:(2,3,3,4)
$$

$$
\mathcal W_{\mathrm{interleaved}}:(2,3,3,4)
$$

$$
\mathcal A:(2\cdot3,3\cdot4)=(6,12)
$$

となる。TT-SVDの通常コアが

$$
C^{(1)}:(1,6,r),
\qquad
C^{(2)}:(r,12,1)
$$

なら、TT-matrixコアは

$$
G^{(1)}:(1,2,3,r),
\qquad
G^{(2)}:(r,3,4,1)
$$

となる。

## 9. exact初期化とtruncated初期化

各cutでfull rankを保つTT-SVDなら、理論上は元の $W$ をexactに再構成でき、float64では丸め誤差程度の差になる。

rankを切り詰めた場合、TT-matrixコアが表す重みを $W_{\mathrm{TT}}$ として

$$
\boxed{
\frac{\lVert W-W_{\mathrm{TT}}\rVert_F}
{\lVert W\rVert_F}
}
$$

を評価する。ここでは次を区別する。

- 元のdense重み $W$
- tensorization後の $\mathcal A$
- TT-SVDで得た3階コア $C^{(k)}$
- 4階へreshapeしたTT-matrixコア $G^{(k)}$
- そのコアから再構成した $W_{\mathrm{TT}}$

randomなTTコアのforward検証では、比較対象はそのコアが定義する $W_{\mathrm{TT}}$ であり、無関係な元dense重みではない。元dense重みとの比較が意味を持つのは、その重みからTT-SVD初期化した場合である。

## 10. PyTorch操作との対応

変換本体は、概念的には次の3行である。

```python
# (m, n) -> (m1, ..., md, n1, ..., nd)
grouped = W.reshape(*out_modes, *in_modes)

# grouped -> (m1, n1, ..., md, nd)
interleaved = grouped.permute(*perm_grouped_to_interleaved)

# (m1, n1, ..., md, nd) -> (q1, ..., qd)
A = interleaved.reshape(*q_modes)
```

ただし短いコードでも、軸の意味を誤ると別のtensorizationになる。実装上の入力検証、要素照合、保存済み結果は [[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]] で扱う。

逆変換は [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/56_TT-matrixのdense reconstruction]] へ進む。
