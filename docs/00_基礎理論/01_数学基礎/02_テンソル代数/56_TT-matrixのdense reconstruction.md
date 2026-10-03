---
title: TT-matrixのdense reconstruction
tags:
  - TT-matrix
  - dense-reconstruction
  - permute
  - validation
---

# TT-matrixのdense reconstruction

## 1. 役割

dense reconstructionは、TT-matrixコアが表す行列

$$
W_{\mathrm{TT}}\in\mathbb R^{m\times n}
$$

を明示的に作る操作である。主な用途は小規模な検証である。

- コアのbond縮約が正しいか
- interleaved順とgrouped順のpermuteが正しいか
- flat indexとmulti-indexが一致するか
- TT-SVD初期化後に元のdense重みを再現できるか
- denseを作らないTT-Linear forwardと一致するか

本番forwardで毎回 $W_{\mathrm{TT}}$ を作る用途ではない。dense行列を生成すれば、TT形式で保存する利点の一部を失う。

## 2. 左からコアを縮約する

コアは

$$
G^{(k)}\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k},
\qquad
r_0=r_d=1
$$

である。左端rankを除き、最初の中間tensorを

$$
T^{(1)}[i_1,j_1,\alpha_1]
=
G^{(1)}[0,i_1,j_1,\alpha_1]
$$

とする。

第 $k$ コアをつなぐと、

$$
\begin{aligned}
&T^{(k)}
[i_1,j_1,\ldots,i_k,j_k,\alpha_k]\\
&=
\sum_{\alpha_{k-1}}
T^{(k-1)}
[i_1,j_1,\ldots,i_{k-1},j_{k-1},\alpha_{k-1}]
G^{(k)}[\alpha_{k-1},i_k,j_k,\alpha_k].
\end{aligned}
$$

したがって中間tensorの軸順は常に

$$
(i_1,j_1,\ldots,i_k,j_k,\alpha_k)
$$

である。最後の右rank $r_d=1$ を落とすと、

$$
\mathcal W_{\mathrm{interleaved}}
[i_1,j_1,\ldots,i_d,j_d]
$$

が得られる。

## 3. 2 siteを途中式から確認する

$d=2$ なら、最初に

$$
T^{(1)}[i_1,j_1,\alpha]
=
G^{(1)}[0,i_1,j_1,\alpha]
$$

を作り、次に

$$
\begin{aligned}
\mathcal W_{\mathrm{interleaved}}
[i_1,j_1,i_2,j_2]
&=
\sum_{\alpha=0}^{r-1}
T^{(1)}[i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0]\\
&=
\sum_{\alpha=0}^{r-1}
G^{(1)}[0,i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0]
\end{aligned}
$$

を得る。この時点の軸は

$$
(i_1,j_1,i_2,j_2)
$$

であり、まだ通常の行列の行・列をまとめた順序ではない。

## 4. interleavedからgroupedへ戻す

欲しい軸順は

$$
(i_1,\ldots,i_d,j_1,\ldots,j_d)
$$

である。interleaved順

$$
(i_1,j_1,i_2,j_2,\ldots,i_d,j_d)
$$

では、出力indexは偶数axis、入力indexは奇数axisにある。したがって逆permuteは

$$
\boxed{
(0,2,4,\ldots,2d-2,\,1,3,5,\ldots,2d-1)
}
$$

である。

```python
perm_interleaved_to_grouped = (
    list(range(0, 2 * d, 2))
    + list(range(1, 2 * d, 2))
)
```

$d=3$ では

$$
(i_1,j_1,i_2,j_2,i_3,j_3)
\longrightarrow
(i_1,i_2,i_3,j_1,j_2,j_3)
$$

なので、

$$
\boxed{(0,2,4,1,3,5)}
$$

となる。tensorization側の

$$
(0,3,1,4,2,5)
$$

と混同しない。前者はinterleavedからgrouped、後者はgroupedからinterleavedであり、互いに逆の操作である。

## 5. grouped tensorをdense行列へreshapeする

permute後のtensorは

$$
\mathcal W_{\mathrm{grouped}}
\in
\mathbb R^{m_1\times\cdots\times m_d\times n_1\times\cdots\times n_d}
$$

である。これを

$$
\boxed{
W_{\mathrm{TT}}
=
\operatorname{reshape}
\left(
\mathcal W_{\mathrm{grouped}},
\left(\prod_{k=1}^{d}m_k,\prod_{k=1}^{d}n_k\right)
\right)
}
$$

とすれば、通常のdense行列へ戻る。

`permute`はaxisと値の配置を並べ替える。最後の`reshape`は、すでに正しい順に並んだ出力側axis群と入力側axis群をそれぞれ一つにまとめる。

## 6. $d=3$ のshape遷移

$$
(m_1,m_2,m_3)=(2,3,2),
\qquad
(n_1,n_2,n_3)=(2,2,3)
$$

とし、TT-rankを

$$
(r_0,r_1,r_2,r_3)=(1,2,3,1)
$$

とする。コアshapeは

$$
G^{(1)}:(1,2,2,2),
$$

$$
G^{(2)}:(2,3,2,3),
$$

$$
G^{(3)}:(3,2,3,1)
$$

である。左から縮約すると

$$
(2,2,2)
$$

$$
\longrightarrow
(2,2,3,2,3)
$$

$$
\longrightarrow
(2,2,3,2,2,3,1)
$$

となる。最後のrank 1を落とすと

$$
\mathcal W_{\mathrm{interleaved}}:(2,2,3,2,2,3)
$$

であり、軸の意味は

$$
(i_1,j_1,i_2,j_2,i_3,j_3)
$$

である。`permute(0,2,4,1,3,5)` 後は

$$
\mathcal W_{\mathrm{grouped}}:(2,3,2,2,2,3)
$$

となり、最後に

$$
W_{\mathrm{TT}}:(12,12)
$$

へreshapeする。

## 7. 一要素を独立な経路で検証する

同じ例で

$$
(i_1,i_2,i_3)=(1,0,1),
\qquad
(j_1,j_2,j_3)=(0,1,2)
$$

を選ぶ。flat indexは

$$
i=1(3\cdot2)+0\cdot2+1=7,
$$

$$
j=0(2\cdot3)+1\cdot3+2=5.
$$

各コアでphysical indexを固定すると

$$
A_1[\alpha_1]
=G^{(1)}[0,1,0,\alpha_1]
\in\mathbb R^{r_1},
$$

$$
A_2[\alpha_1,\alpha_2]
=G^{(2)}[\alpha_1,0,1,\alpha_2]
\in\mathbb R^{r_1\times r_2},
$$

$$
A_3[\alpha_2]
=G^{(3)}[\alpha_2,1,2,0]
\in\mathbb R^{r_2}
$$

が残る。したがって

$$
\boxed{
W_{\mathrm{TT}}[7,5]
=
\sum_{\alpha_1,\alpha_2}
A_1[\alpha_1]
A_2[\alpha_1,\alpha_2]
A_3[\alpha_2]
}
$$

である。これは行列積

$$
A_1A_2A_3
$$

でも、bond添字を明示したEinstein和でも計算できる。dense reconstruction側の一要素と、この独立なbond縮約が一致すれば、

- core contraction
- interleavedからgroupedへのpermute
- groupedからdenseへのreshape
- multi-indexからflat indexへの変換

をまとめて検証できる。

## 8. メモリ上の注意

例えば $4096\times4096$ のfloat64行列だけで

$$
4096^2\times8\ \mathrm{bytes}
=128\ \mathrm{MiB}
$$

を要する。途中tensorやgradientまで含めればさらに増える。dense reconstructionは小さいsanity check、テスト、誤差計測に限定し、本番TT-Linear forwardはコアを直接縮約する。

その直接forwardは [[00_基礎理論/01_数学基礎/02_テンソル代数/57_TT-Linear_forwardの縮約とshape]] で導出する。
