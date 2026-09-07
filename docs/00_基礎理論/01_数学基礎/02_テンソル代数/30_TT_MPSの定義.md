---
title: TT・MPSの定義
aliases:
  - Tensor Train
  - Matrix Product State
  - TTコア
  - MPSサイトテンソル
tags:
  - TT
  - MPS
  - TensorTrain
  - TensorNetwork
  - Schmidt分解
---

# TT・MPSの定義

## サマリー

Tensor Train（TT）は、高階テンソルを1個の巨大な配列として保持せず、各modeを担当する小さな3階テンソルを鎖状につないで表す形式である。物理では、開放境界条件の Matrix Product State（MPS）と同じ数学的構造を使う。

元テンソルを

$$
X\in\mathbb R^{n_1\times n_2\times\cdots\times n_d}
$$

とすると、第 $k$ TTコアは

$$
\boxed{
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
}
$$

で、両端は

$$
r_0=r_d=1
$$

とする。元テンソルの各成分は、内部bond indexを縮約して

$$
\boxed{
X_{i_1\cdots i_d}
=
\sum_{\alpha_1=1}^{r_1}\cdots
\sum_{\alpha_{d-1}=1}^{r_{d-1}}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,1}
}
$$

と復元する。

物理のMPS記法

$$
A^{[k]}_{\alpha_{k-1},s_k,\alpha_k}
$$

との対応は

$$
s_k\longleftrightarrow i_k,
\qquad
A^{[k]}\longleftrightarrow G^{(k)}
$$

である。

---

## 1. TTコアとは何か

TTコアは、元の大きなテンソルを復元するための「各mode担当の小テンソル」である。

3階テンソル

$$
X\in\mathbb R^{n_1\times n_2\times n_3}
$$

を考える。TTではこれを

$$
G^{(1)}\in\mathbb R^{1\times n_1\times r_1},
$$

$$
G^{(2)}\in\mathbb R^{r_1\times n_2\times r_2},
$$

$$
G^{(3)}\in\mathbb R^{r_2\times n_3\times1}
$$

という3個のコアへ分ける。

各コアは独立ではなく、内部添字

$$
\alpha_1,\alpha_2
$$

を共有する。

```text
G^(1) -- α1 -- G^(2) -- α2 -- G^(3)
```

この鎖構造が Tensor Train という名前の由来である。

3サイトの場合、元テンソルは

$$
\boxed{
X_{i_1i_2i_3}
=
\sum_{\alpha_1=1}^{r_1}
\sum_{\alpha_2=1}^{r_2}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
G^{(3)}_{\alpha_2,i_3,1}
}
$$

となる。

---

## 2. コアの3本の添字

一般の第 $k$ コア

$$
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

には3種類の添字がある。

| 軸 | 添字 | 意味 |
| --- | --- | --- |
| 左bond | $\alpha_{k-1}$ | 前のコアとの接続、左側に蓄積された情報 |
| physical / mode | $i_k$ | 元テンソルの第 $k$ mode |
| 右bond | $\alpha_k$ | 次のコアとの接続、右側へ渡す情報 |

物理のMPSでは $i_k$ を局所物理状態 $s_k$ と読む。左右のbond indexは仮想添字であり、局所自由度そのものではなく、左右の部分系を結ぶ相関チャネルを表す。

特定のphysical index $i_k$ を固定すると、中央コアは

$$
G^{(k)}_{:,i_k,:}
\in
\mathbb R^{r_{k-1}\times r_k}
$$

という行列として読める。したがってTTの1成分は

$$
\underbrace{G^{(1)}_{1,i_1,:}}_{1\times r_1}
\underbrace{G^{(2)}_{:,i_2,:}}_{r_1\times r_2}
\cdots
\underbrace{G^{(d)}_{:,i_d,1}}_{r_{d-1}\times1}
$$

という行列積で得られる1個のスカラーである。

---

## 3. なぜ端のbond dimensionは1か

第1コアの左側には前のコアがない。最後のコアの右側にも次のコアがない。そのため

$$
\boxed{r_0=r_d=1}
$$

と置く。

これは「新しい物理自由度を1個増やす」という意味ではなく、全コアを

$$
(r_{k-1},n_k,r_k)
$$

という同じ3軸形式で統一するためのサイズ1の境界軸である。

例えば第1 SVDで

$$
U^{(1)}\in\mathbb R^{n_1\times r_1}
$$

が得られたとする。この行列を第1TTコアとして

$$
G^{(1)}
=
\operatorname{reshape}
\left(U^{(1)},1,n_1,r_1\right)
$$

と書く。

成分ではただ

$$
\boxed{
G^{(1)}_{1,i_1,\alpha_1}
=
U^{(1)}_{i_1,\alpha_1}
}
$$

である。値も要素数も変化しない。

$$
n_1r_1
=
1\cdot n_1r_1.
$$

したがってこれは「4階テンソルにする」操作ではない。`reshape(U, 1, n_1, r_1)` の3個の数字は3本の軸のサイズを指定しており、結果は3階テンソルである。

---

## 4. 3サイトのTTをSVDから作る見取り図

3サイトの係数テンソルを

$$
\Psi_{s_1s_2s_3}
\in
\mathbb C^{n_1\times n_2\times n_3}
$$

とする。

最初に

$$
s_1\mid(s_2,s_3)
$$

で行列化し、

$$
\Psi^{\langle1\rangle}
\in
\mathbb C^{n_1\times(n_2n_3)}
$$

へ変形する。reduced SVDを

$$
\Psi^{\langle1\rangle}
=
U^{(1)}\Sigma^{(1)}V^{(1)\dagger}
$$

とし、

$$
U^{(1)}\in\mathbb C^{n_1\times r_1},
$$

$$
\Sigma^{(1)}\in\mathbb R^{r_1\times r_1},
$$

$$
V^{(1)\dagger}\in\mathbb C^{r_1\times(n_2n_3)}
$$

とする。

左因子を

$$
G^{(1)}\in\mathbb C^{1\times n_1\times r_1}
$$

へreshapeし、残りを

$$
B^{(1)}
=
\Sigma^{(1)}V^{(1)\dagger}
\in
\mathbb C^{r_1\times(n_2n_3)}
$$

とする。

これを

$$
\mathcal B^{(1)}
\in
\mathbb C^{r_1\times n_2\times n_3}
$$

へ戻し、次は

$$
(\alpha_1,s_2)\mid s_3
$$

でSVDする。

この「SVDで左側を1サイトずつコアとして取り出し、$\Sigma V^\dagger$ を右へ渡す」操作を繰り返すことでTT/MPSが得られる。詳細は [[32_TT-SVD]] を参照する。

---

## 5. 行列SVDからSchmidt分解へ

TTとMPSをつなぐ最初の重要点は、係数行列のSVDを状態ベクトルとして読み直すとSchmidt分解になることである。

3サイト状態を

$$
|\Psi\rangle
=
\sum_{s_1,s_2,s_3}
\psi_{s_1s_2s_3}
|s_1\rangle\otimes|s_2s_3\rangle
$$

とする。

右側の複合添字を

$$
\mu=(s_2,s_3)
$$

と置けば、

$$
|\Psi\rangle
=
\sum_{s_1,\mu}
\Psi^{\langle1\rangle}_{s_1,\mu}
|s_1\rangle\otimes|\mu\rangle.
$$

係数行列をSVDする。

$$
\Psi^{\langle1\rangle}_{s_1,\mu}
=
\sum_{\alpha_1=1}^{r_1}
U_{s_1,\alpha_1}
\lambda_{\alpha_1}
V^*_{\mu,\alpha_1}.
$$

これを状態展開へ代入する。

$$
\begin{aligned}
|\Psi\rangle
&=
\sum_{s_1,\mu}
\left(
\sum_{\alpha_1}
U_{s_1,\alpha_1}
\lambda_{\alpha_1}
V^*_{\mu,\alpha_1}
\right)
|s_1\rangle\otimes|\mu\rangle\\
&=
\sum_{\alpha_1}
\lambda_{\alpha_1}
\sum_{s_1,\mu}
U_{s_1,\alpha_1}
V^*_{\mu,\alpha_1}
|s_1\rangle\otimes|\mu\rangle\\
&=
\sum_{\alpha_1}
\lambda_{\alpha_1}
\left(
\sum_{s_1}U_{s_1,\alpha_1}|s_1\rangle
\right)
\otimes
\left(
\sum_{\mu}V^*_{\mu,\alpha_1}|\mu\rangle
\right).
\end{aligned}
$$

ここで

$$
|L_{\alpha_1}\rangle
:=
\sum_{s_1}U_{s_1,\alpha_1}|s_1\rangle,
$$

$$
|R_{\alpha_1}\rangle
:=
\sum_{\mu}V^*_{\mu,\alpha_1}|\mu\rangle
$$

と定義すると、

$$
\boxed{
|\Psi\rangle
=
\sum_{\alpha_1=1}^{r_1}
\lambda_{\alpha_1}
|L_{\alpha_1}\rangle
\otimes
|R_{\alpha_1}\rangle
}
$$

となる。これが切断 $1\mid23$ に対するSchmidt分解である。

SVDの列直交性から

$$
\langle L_{\beta_1}|L_{\alpha_1}\rangle
=
\delta_{\beta_1\alpha_1},
$$

$$
\langle R_{\beta_1}|R_{\alpha_1}\rangle
=
\delta_{\beta_1\alpha_1}.
$$

したがって対応は

$$
\boxed{
\text{SVDの左特異ベクトル}
\leftrightarrow
\text{左Schmidtベクトル}
}
$$

$$
\boxed{
\text{SVDの右特異ベクトル}
\leftrightarrow
\text{右Schmidtベクトル}
}
$$

$$
\boxed{
\text{特異値}
\leftrightarrow
\text{Schmidt係数}
}
$$

$$
\boxed{
\text{行列rank}
\leftrightarrow
\text{Schmidt rank}
}
$$

である。

---

## 6. TTとMPSで同じ対象をどう読むか

| 応用数学・数値線形代数 | 物理・MPS |
| --- | --- |
| Tensor Train | Matrix Product State |
| mode index $i_k$ | local physical index $s_k$ |
| TT core $G^{(k)}$ | site tensor $A^{[k]}$ |
| TT-rank $r_k$ | bond dimension |
| cut unfolding | bipartition |
| SVD | Schmidt decomposition |
| singular value | Schmidt coefficient |
| truncated SVD | 小さいSchmidt成分の切り捨て |

この対応を使うと、TTを「高階テンソルの低ランク表現」として読む視点と、MPSを「部分系間の相関をbondに保持する状態表現」として読む視点を同時に持てる。

---

## 7. TTパラメータ数

TTコア列が保持する総要素数は

$$
\boxed{
P_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}n_kr_k
}
$$

である。

元のdenseテンソルは

$$
P_{\mathrm{dense}}
=
\prod_{k=1}^{d}n_k
$$

個の値を持つ。TTが圧縮になるかどうかは、TT-rankが十分小さく

$$
P_{\mathrm{TT}}
<
P_{\mathrm{dense}}
$$

となるかで決まる。小さいテンソルでは、TT表現の方がパラメータ数が多くなる場合もある。

---

## 8. この章で固定する理解

- TTコアは各modeを担当する小さな3階テンソルである。
- 第 $k$ コアのshapeは $(r_{k-1},n_k,r_k)$。
- $n_k$ はphysical/mode index、$r_{k-1},r_k$ は左右bond dimension。
- 両端の $1$ は境界を統一形式で表すdummy axisであり、新しい自由度ではない。
- TTは内部bondを縮約して元テンソルを復元する。
- 開放境界MPSとTTは同じ鎖状テンソルネットワークである。
- 係数行列のSVDを状態として読み直すとSchmidt分解になる。

次は [[31_TT-rankとunfolding]] で、bond dimensionがなぜ元テンソルの各cut rankと結び付くかを整理する。
