---
title: TT-SVD
aliases:
  - Tensor Train SVD
  - sequential SVD
  - MPS逐次Schmidt分解
tags:
  - TT
  - MPS
  - SVD
  - TT-SVD
  - TensorTrain
---

# TT-SVD

## サマリー

TT-SVDは、高階テンソルを左から順に

```text
reshape → SVD → UをTTコアへ → ΣV^Tを次のremainderへ
```

と分解していくアルゴリズムである。

$d$ 階テンソルならSVDは $d-1$ 回行う。第 $k$ 段階では、前段から来た左bond dimension $r_{k-1}$ と現在のmode size $n_k$ をまとめて

$$
(r_{k-1}n_k)
\times
(n_{k+1}\cdots n_d)
$$

の行列を作る。

左特異ベクトルを

$$
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

へreshapeし、$\Sigma V^T$ を次段へ渡す。

---

## 1. 3サイトから始める

元テンソルを

$$
X\in\mathbb R^{n_1\times n_2\times n_3}
$$

とする。

### 第1cut

最初に

$$
i_1\mid(i_2,i_3)
$$

で行列化する。

$$
M^{(1)}
=
X^{\langle1\rangle}
\in
\mathbb R^{n_1\times(n_2n_3)}.
$$

要素数は変わらない。

$$
n_1n_2n_3
=
n_1(n_2n_3).
$$

成分では

$$
M^{(1)}_{i_1,(i_2,i_3)}
=
X_{i_1i_2i_3}.
$$

---

## 2. 第1 SVD

reduced SVDを

$$
M^{(1)}
=
U^{(1)}\Sigma^{(1)}V^{(1)T}
$$

とする。

厳密TT-SVDなら

$$
r_1
=
\operatorname{rank}(M^{(1)}).
$$

shapeは

$$
U^{(1)}
\in
\mathbb R^{n_1\times r_1},
$$

$$
\Sigma^{(1)}
\in
\mathbb R^{r_1\times r_1},
$$

$$
V^{(1)T}
\in
\mathbb R^{r_1\times(n_2n_3)}.
$$

積のshapeは

$$
(n_1\times r_1)
(r_1\times r_1)
(r_1\times n_2n_3)
=
(n_1\times n_2n_3)
$$

で元のunfoldingへ戻る。

---

## 3. 第1 TTコア

$U^{(1)}$ を

$$
G^{(1)}
=
\operatorname{reshape}
\left(U^{(1)},1,n_1,r_1\right)
$$

とする。

$$
\boxed{
G^{(1)}
\in
\mathbb R^{1\times n_1\times r_1}
}
$$

成分では

$$
G^{(1)}_{1,i_1,\alpha_1}
=
U^{(1)}_{i_1,\alpha_1}.
$$

これは値を変えず、左端bond $r_0=1$ を明示しただけである。

---

## 4. 右へ渡すremainder

第1 SVDでまだ使っていない部分を

$$
B^{(1)}
=
\Sigma^{(1)}V^{(1)T}
$$

とする。

shapeは

$$
B^{(1)}
\in
\mathbb R^{r_1\times(n_2n_3)}.
$$

これを

$$
\mathcal B^{(1)}
=
\operatorname{reshape}
\left(B^{(1)},r_1,n_2,n_3\right)
$$

として、

$$
\mathcal B^{(1)}
\in
\mathbb R^{r_1\times n_2\times n_3}
$$

へ戻す。

成分では

$$
\mathcal B^{(1)}_{\alpha_1,i_2,i_3}
=
\left[\Sigma^{(1)}V^{(1)T}\right]_{\alpha_1,(i_2,i_3)}.
$$

ここで $i_1$ が消えているように見えるが、サイト1側の情報は完全に捨てられたのではない。$U^{(1)}$ の列で張られる基底へ座標変換され、そのチャネル番号が $\alpha_1$ になっている。

---

## 5. 第2 SVDの準備

次は

$$
(\alpha_1,i_2)\mid i_3
$$

で行列化する。

左側の組 $(\alpha_1,i_2)$ は $r_1n_2$ 通りなので、

$$
M^{(2)}
=
\operatorname{reshape}
\left(\mathcal B^{(1)},r_1n_2,n_3\right)
\in
\mathbb R^{(r_1n_2)\times n_3}.
$$

ここで重要なのは、**第2 SVDの左側が単なる $i_2$ ではない**ことである。第1cutで生まれたbond index $\alpha_1$ を現在のphysical index $i_2$ とまとめて左側にする。

---

## 6. 第2 SVDと第2コア

$$
M^{(2)}
=
U^{(2)}\Sigma^{(2)}V^{(2)T}
$$

とする。

厳密rankを

$$
r_2
=
\operatorname{rank}(M^{(2)})
$$

とすると、

$$
U^{(2)}
\in
\mathbb R^{(r_1n_2)\times r_2},
$$

$$
\Sigma^{(2)}
\in
\mathbb R^{r_2\times r_2},
$$

$$
V^{(2)T}
\in
\mathbb R^{r_2\times n_3}.
$$

$U^{(2)}$ を

$$
G^{(2)}
=
\operatorname{reshape}
\left(U^{(2)},r_1,n_2,r_2\right)
$$

とすれば、

$$
\boxed{
G^{(2)}
\in
\mathbb R^{r_1\times n_2\times r_2}
}
$$

である。

成分では

$$
G^{(2)}_{\alpha_1,i_2,\alpha_2}
=
U^{(2)}_{(\alpha_1,i_2),\alpha_2}.
$$

---

## 7. 最後のコア

残り

$$
B^{(2)}
=
\Sigma^{(2)}V^{(2)T}
\in
\mathbb R^{r_2\times n_3}
$$

へ右端bond $r_3=1$ を加える。

$$
G^{(3)}
=
\operatorname{reshape}
\left(B^{(2)},r_2,n_3,1\right).
$$

したがって

$$
\boxed{
G^{(3)}
\in
\mathbb R^{r_2\times n_3\times1}
}
$$

である。

---

## 8. 3サイトTTの完成

最終的に

$$
G^{(1)}:1\times n_1\times r_1,
$$

$$
G^{(2)}:r_1\times n_2\times r_2,
$$

$$
G^{(3)}:r_2\times n_3\times1
$$

が得られ、

$$
\boxed{
X_{i_1i_2i_3}
=
\sum_{\alpha_1,\alpha_2}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
G^{(3)}_{\alpha_2,i_3,1}
}
$$

と復元できる。

shapeだけ追えば

$$
(1\times n_1\times r_1)
(r_1\times n_2\times r_2)
(r_2\times n_3\times1)
$$

の内部 $r_1,r_2$ を縮約し、

$$
1\times n_1\times n_2\times n_3\times1
$$

が残る。両端のsize 1を除けば元のshapeである。

---

## 9. 一般の$d$階TT-SVD

元テンソル

$$
X\in\mathbb R^{n_1\times\cdots\times n_d}
$$

に対し、初期状態を

$$
R^{(0)}=X,
\qquad
r_0=1
$$

とする。

第 $k$ 段階 $(k=1,\ldots,d-1)$ で、remainderを

$$
R^{(k-1)}
\in
\mathbb R^{r_{k-1}\times n_k\times\cdots\times n_d}
$$

とする。

これを

$$
M_k
=
\operatorname{reshape}
\left(
R^{(k-1)},
(r_{k-1}n_k),
(n_{k+1}\cdots n_d)
\right)
$$

へ行列化する。

SVD

$$
M_k
=
U_k\Sigma_kV_k^T
$$

を行い、厳密版なら

$$
r_k
=
\operatorname{rank}(M_k)
$$

を使う。

左特異ベクトルを

$$
\boxed{
G^{(k)}
=
\operatorname{reshape}
\left(U_k,r_{k-1},n_k,r_k\right)
}
$$

へ変換し、

$$
R^{(k)}
=
\operatorname{reshape}
\left(
\Sigma_kV_k^T,
r_k,n_{k+1},\ldots,n_d
\right)
$$

を次段へ渡す。

最後に

$$
G^{(d)}
=
\operatorname{reshape}
\left(R^{(d-1)},r_{d-1},n_d,1\right)
$$

とする。

---

## 10. 「元テンソルのcut unfolding」と「途中remainderの行列化」は別物

TT-rankを定義するときは、元テンソルを直接

$$
X^{\langle k\rangle}
\in
\mathbb R^{(n_1\cdots n_k)\times(n_{k+1}\cdots n_d)}
$$

へ行列化する。

一方、TT-SVDの第 $k$ SVDは

$$
M_k
\in
\mathbb R^{(r_{k-1}n_k)\times(n_{k+1}\cdots n_d)}
$$

を使う。

第1段階では同じ形だが、第2段階以降は異なる。TT-SVDがなぜこの小さいremainderだけで元テンソルのcut rankを再現できるかは、前段で取り出した $U$ が列直交でrankを保存するからである。詳細は [[34_基底変換とTT-rank不変性]] を参照する。

---

## 11. 物理では逐次Schmidt分解として読める

最初のcut

$$
1\mid2\cdots d
$$

でSVDすることは、そのbipartitionに対するSchmidt分解に対応する。

次に $\alpha_1$ とサイト2をまとめてSVDすることで、左側のSchmidt基底を引き継ぎながら

$$
12\mid3\cdots d
$$

に対応するbond構造を取り出す。

したがってTT-SVDは、応用数学では逐次低rank factorization、物理では左から右への逐次Schmidt分解として読める。

---

## 12. exact TT-SVDの意味

打ち切りをしなければ、各段階のnumerical/nonzero rankを全て保持するので、理論上は元テンソルを完全に表現する。

$$
\widehat X=X
$$

浮動小数点実装では

$$
\frac{\|X-\widehat X\|_F}{\|X\|_F}
$$

がmachine precision程度になることを確認する。

圧縮のためにrankを落とす場合は次章 [[33_TT-SVDの打ち切りと誤差]] へ進む。
