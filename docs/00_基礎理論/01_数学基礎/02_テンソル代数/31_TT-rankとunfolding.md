---
title: TT-rankとunfolding
aliases:
  - TTランク
  - sequential unfolding
  - cut unfolding
  - Schmidt rank
tags:
  - TT
  - MPS
  - rank
  - unfolding
  - SchmidtRank
---

# TT-rankとunfolding

## サマリー

TT-rankは、各TTコアの都合で付けた内部サイズではない。元テンソルを左から順に二分して行列化したときのrankで決まる。

$$
\boxed{
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right),
\qquad
k=1,\ldots,d-1
}
$$

ここで

$$
X^{\langle k\rangle}
\in
\mathbb R^{(n_1\cdots n_k)\times(n_{k+1}\cdots n_d)}
$$

はcut

$$
(1,\ldots,k)\mid(k+1,\ldots,d)
$$

に対応するTT unfoldingである。

物理では同じ $r_k$ を、その切断におけるbond dimension、厳密状態ならSchmidt rankとして読む。

---

## 1. 行列rankの復習

行列

$$
A\in\mathbb R^{m\times n}
$$

のrank

$$
\operatorname{rank}(A)
$$

は、互いに同値な次の量として読める。

- 独立な列方向の数
- 独立な行方向の数
- 非ゼロ特異値の本数
- $A=PQ$ と因子分解するときに必要な最小内部次元

SVD

$$
A=U\Sigma V^T
$$

の非ゼロ特異値が

$$
\sigma_1,\ldots,\sigma_r
$$

の $r$ 個なら、

$$
\operatorname{rank}(A)=r.
$$

---

## 2. テンソルでは「どこで切るか」が必要

行列には自然な左右分割が1つしかないが、高階テンソルでは複数の切り方がある。

3階テンソル

$$
X_{i_1i_2i_3}
$$

では、TTで使う切断は

$$
i_1\mid(i_2,i_3)
$$

と

$$
(i_1,i_2)\mid i_3
$$

である。

それぞれ

$$
X^{\langle1\rangle}_{i_1,(i_2,i_3)}
=
X_{i_1i_2i_3},
$$

$$
X^{\langle2\rangle}_{(i_1,i_2),i_3}
=
X_{i_1i_2i_3}
$$

と行列化する。

したがって、3階テンソルのTT-rankは

$$
\boxed{
r_1
=
\operatorname{rank}
\left(X^{\langle1\rangle}\right)
}
$$

$$
\boxed{
r_2
=
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
}
$$

である。

### 小さい3階テンソルの第1cutと第2cut

PyTorchと対応させるため、この例だけ添字を0始まりとする。$X\in\mathbb R^{2\times2\times2}$ の全要素を

$$
X_{0,:,:}
=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix},
\qquad
X_{1,:,:}
=
\begin{pmatrix}
5&6\\
7&8
\end{pmatrix}
$$

とする。

第1cut $i_1\mid(i_2,i_3)$ では、列の複合添字を

$$
(i_2,i_3)
=
(0,0),(0,1),(1,0),(1,1)
$$

の順に並べるので、

$$
X^{\langle1\rangle}
=
\begin{pmatrix}
1&2&3&4\\
5&6&7&8
\end{pmatrix}.
$$

第2cut $(i_1,i_2)\mid i_3$ では、行の複合添字を

$$
(i_1,i_2)
=
(0,0),(0,1),(1,0),(1,1)
$$

の順に並べるので、

$$
X^{\langle2\rangle}
=
\begin{pmatrix}
1&2\\
3&4\\
5&6\\
7&8
\end{pmatrix}.
$$

0始まりの実際の行位置は

$$
\operatorname{row}
=
i_1n_2+i_2
=
2i_1+i_2
$$

であり、全対応は

$$
\begin{array}{c|c}
(i_1,i_2)&\operatorname{row}\\
\hline
(0,0)&0\\
(0,1)&1\\
(1,0)&2\\
(1,1)&3
\end{array}
$$

となる。

一方、Tuckerのmode-2 unfolding $i_2\mid(i_1,i_3)$ は、行を $i_2$、列を

$$
(i_1,i_3)
=
(0,0),(0,1),(1,0),(1,1)
$$

とするため、

$$
X^{\mathrm{mode}\text{-}2}
=
\begin{pmatrix}
1&2&5&6\\
3&4&7&8
\end{pmatrix}.
$$

したがって、TTの第2cut $X^{\langle2\rangle}$ とTuckerのmode-2 unfoldingは、同じ8要素を使っていても行列のshapeと添字の左右分割が異なる。PyTorchでの `reshape` / `permute` との対応は [[29_TT_cutとPyTorchのreshape_Kronecker順序]] で確認する。

---

## 3. 一般のcut unfolding

$d$ 階テンソル

$$
X\in\mathbb R^{n_1\times\cdots\times n_d}
$$

に対して、第 $k$ cutで左側の複合添字を

$$
I_L=(i_1,\ldots,i_k)
$$

右側を

$$
I_R=(i_{k+1},\ldots,i_d)
$$

とまとめる。

すると

$$
X^{\langle k\rangle}_{I_L,I_R}
:=
X_{i_1,\ldots,i_d}
$$

で、shapeは

$$
\boxed{
X^{\langle k\rangle}
\in
\mathbb R^{
(n_1\cdots n_k)
\times
(n_{k+1}\cdots n_d)
}
}
$$

となる。

rankのshape上限は

$$
\boxed{
r_k
\le
\min
\left(
\prod_{j=1}^{k}n_j,
\prod_{j=k+1}^{d}n_j
\right)
}
$$

である。

---

## 4. 複合添字とは何か

$(i_1,i_2)$ は新しい物理添字ではない。複数の添字を「行列の1個の行番号」としてまとめて読むための記法である。

例えば

$$
X\in\mathbb R^{2\times3\times4}
$$

で第2cutを作ると、

$$
X^{\langle2\rangle}
\in
\mathbb R^{6\times4}
$$

になる。行は6通りのペア

$$
(i_1,i_2)
\in
\{1,2\}\times\{1,2,3\}
$$

を表す。

複合添字を1個の整数へflattenする具体的規則は実装のメモリ順に依存する。数学上重要なのは、**元の複数添字と行列の行・列の対応を一貫して固定すること**である。PyTorchの具体的なreshape順は [[29_TT_cutとPyTorchのreshape_Kronecker順序]] で扱う。

---

## 5. $r_k$ は左右を結ぶ独立チャネル数

第 $k$ cutの行列をrank factorizationとして

$$
X^{\langle k\rangle}
=
LR
$$

と書く。

成分では

$$
X_{i_1,\ldots,i_d}
=
\sum_{\alpha_k=1}^{r_k}
L_{(i_1,\ldots,i_k),\alpha_k}
R_{\alpha_k,(i_{k+1},\ldots,i_d)}.
$$

ここで $\alpha_k$ は左右を結ぶチャネル番号である。

- $r_k=1$ なら左右は1個の積へ完全に分離できる。
- $r_k>1$ なら複数の独立チャネルが必要である。

したがって

$$
\boxed{
r_k
=
\text{そのcutをまたぐ独立な情報チャネル数}
}
$$

と読むことができる。

---

## 6. 物理ではSchmidt rankになる

状態

$$
|\Psi\rangle
$$

をcut

$$
(1,\ldots,k)\mid(k+1,\ldots,d)
$$

で二分しSchmidt分解すると、

$$
|\Psi\rangle
=
\sum_{\alpha_k=1}^{r_k}
\lambda_{\alpha_k}
|L_{\alpha_k}\rangle
\otimes
|R_{\alpha_k}\rangle.
$$

この項数がSchmidt rankであり、係数行列のrankと同じである。

よって厳密表現では

$$
\boxed{
\text{TT-rank}
=
\text{bond dimension}
=
\text{Schmidt rank}
}
$$

と対応する。

ただし近似TTでは、意図的にbond dimensionを

$$
\widetilde r_k<r_k
$$

へ落とす。その場合、$\widetilde r_k$ は元テンソルの厳密TT-rankではなく、近似表現として選んだbond dimensionである。詳細は [[33_TT-SVDの打ち切りと誤差]] を参照する。

---

## 7. 第2 SVDのrankはなぜ元の第2cut rankなのか

TT-SVDでは第1 SVD後、元テンソル $X$ ではなく残りテンソル

$$
B_{\alpha_1,i_2,i_3}
$$

を次のSVDへ渡す。

そのため自然な疑問は、

> $B$ には元の $i_1$ 添字がないのに、$B$ の第2 SVDで得るrankが、元の $X$ を $(i_1,i_2)\mid i_3$ で切ったrankと本当に同じなのか

である。

第1 SVDがexactなら

$$
X^{\langle2\rangle}
=
(U\otimes I_{n_2})B_{\mathrm{cut2}}
$$

と書ける。$U$ は列直交なので

$$
(U\otimes I_{n_2})^T(U\otimes I_{n_2})=I.
$$

したがって左から掛ける変換はrankを潰さず、

$$
\boxed{
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(B_{\mathrm{cut2}}\right)
}
$$

となる。

この完全な途中式と、$U\otimes I$ が出る理由は [[34_基底変換とTT-rank不変性]] で導出する。

---

## 8. 数学的rankと数値rank

理論では

$$
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right)
$$

を「非ゼロ特異値の本数」と定義する。

一方、浮動小数点計算では本来0である特異値が

$$
10^{-15},\quad10^{-14}
$$

のような小さい値として現れる場合がある。そのため実装では閾値に基づく numerical rank を使う。

このプロジェクトの `tt_svd_exact` も数学的symbolic rankではなく、`torch.linalg.matrix_rank` のdefault toleranceに基づく数値rankを使う。したがって「exact」は**その数値rankを打ち切らず残す**という意味である。実装contractの詳細は [[27_TT_MPS基礎のPyTorch実装]] を参照する。

### 最大可能rankと実際のrankは同じとは限らない

行列のshapeが$m\times n$なら、

$$
\operatorname{rank}(A)
\le
\min(m,n)
$$

である。しかし右辺は最大可能rankであり、実際のrankが常にそこまで達するという意味ではない。例えば、

$$
X_1
:=
\begin{pmatrix}
1&2&3\\
2&4&6
\end{pmatrix}
$$

では、第2行が第1行の2倍である。

$$
\begin{pmatrix}
2&4&6
\end{pmatrix}
=
2
\begin{pmatrix}
1&2&3
\end{pmatrix}.
$$

独立な行は1本しかないため、

$$
\operatorname{rank}(X_1)=1.
$$

一方、shapeから決まる上限は

$$
\min(X_1.\mathrm{shape})
=
\min(2,3)
=
2
$$

なので、

$$
\boxed{
\operatorname{rank}(X_1)
=1
<
2
=
\min(X_1.\mathrm{shape})
}
$$

となる。TT-SVDでも、

```python
min(X1.shape)
```

はSVDで取り得る成分数の上限であり、exact TT-rankではない。実際に必要なrankは

```python
torch.linalg.matrix_rank(X1).item()
```

で評価する。

---

## 9. この章で固定する理解

- TT unfoldingは先頭 $k$ modeと残りを二分する行列化。
- TT-rankは各cut unfoldingのrank。
- $r_k$ は左右をつなぐ独立チャネル数。
- 物理ではbond dimension / Schmidt rankと対応する。
- 近似時の $\widetilde r_k$ は元テンソルの厳密rankではなく選択したbond dimension。
- TT-SVDの途中テンソルで得るrankが元のcut rankと一致する理由は、前段の列直交基底変換がrankを保存するから。

---

## 10. 参考資料の使い方

資料では、TTをいきなり一般$d$階の式だけで読むより、次の線形代数を先に固定すると理解しやすいと整理した。

1. reduced / thin SVD
2. 列空間・像・基底変換
3. 列直交行列によるrank保存
4. Kronecker積

特に

$$
U^TU=I
\quad\Longrightarrow\quad
\operatorname{rank}(UB)=\operatorname{rank}(B)
$$

という見方が、第2cut以降を理解する核心になる。

TTの定義・TT-rank・TT-SVDを確認する資料としては、Oseledetsの *Tensor-Train Decomposition* を原典・辞書として参照し、TuckerとTTを含むテンソルネットワーク全体の橋渡しにはCichockiらのレビュー、物理直観にはMPS/Schmidt分解を扱うテンソルネットワーク入門を併用する、という学習方針だった。

一般論へ急ぐより、まず3サイトで

$$
X^{\langle1\rangle},
\qquad
X^{\langle2\rangle}
$$

を横に並べ、途中remainderとの関係を手計算・小コードで確認するのがこのプロジェクトの学習方針である。
