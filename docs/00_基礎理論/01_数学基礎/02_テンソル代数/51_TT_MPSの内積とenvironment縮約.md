---
title: TT_MPSの内積とenvironment縮約
tags:
  - TT
  - MPS
  - tensor-train
  - inner-product
  - contraction
  - environment
  - einsum
---

# TT/MPSの内積とenvironment縮約

このノートでは、2本のTT/MPSをdense tensorへ復元せずに内積する方法を、成分表示からleft environmentの更新式まで導出する。

TT-rounding後の圧縮誤差、2本のTTの差、MPSのoverlapを評価するには、まず

$$
\langle \mathcal A,\mathcal B\rangle
$$

をTTコアのまま計算できなければならない。中心コアだけから内積を読む特殊な正準形は [[40_TT_MPSの中心ノルムと内積の導出]] で扱った。本ノートでは正準形を仮定せず、任意のshape整合したTTに使える逐次縮約を扱う。

Frobeniusノルムと距離は [[52_TT_MPSのFrobeniusノルムと距離]]、加減算を明示的に作る場合のrank増加は [[53_TT_MPSの加減算とrank増加]]、PyTorch実装と保存済み数値結果は [[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]] に分ける。

---

## 0. このノートの到達点

2本の実数値TTを

$$
\mathcal A
=
\llbracket A^{(1)},\ldots,A^{(d)}\rrbracket,
\qquad
\mathcal B
=
\llbracket B^{(1)},\ldots,B^{(d)}\rrbracket
$$

とする。各コアは

$$
A^{(k)}
\in
\mathbb R^{r_{k-1}^{A}\times n_k\times r_k^{A}},
\qquad
B^{(k)}
\in
\mathbb R^{r_{k-1}^{B}\times n_k\times r_k^{B}}
$$

で、境界rankは

$$
r_0^A=r_d^A=r_0^B=r_d^B=1
$$

とする。AとBのTT-rankは一致しなくてよいが、physical shape

$$
(n_1,\ldots,n_d)
$$

は一致しなければならない。

left environmentを

$$
E_k\in\mathbb R^{r_k^A\times r_k^B}
$$

とすると、内積は次の更新で計算できる。

$$
\boxed{
E_k(\alpha_k,\beta_k)
=
\sum_{\alpha_{k-1},\beta_{k-1},i_k}
E_{k-1}(\alpha_{k-1},\beta_{k-1})
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
}
$$

初期値は

$$
E_0=[1]\in\mathbb R^{1\times1}
$$

であり、最後は

$$
E_d
=
\begin{bmatrix}
\langle\mathcal A,\mathcal B\rangle
\end{bmatrix}
\in\mathbb R^{1\times1}
$$

となる。

---

## 1. TTの一要素を全添字で書く

physical indexを

$$
i_k\in\{1,\ldots,n_k\}
$$

とする。A側のbond indexを $\alpha_k$、B側を $\beta_k$ と区別する。

固定した $i_k$ に対し、コアのsliceは行列である。

$$
A^{(k)}[:,i_k,:]
\in
\mathbb R^{r_{k-1}^A\times r_k^A}
$$

したがって、TTの一要素は行列列の積である。

$$
\mathcal A_{i_1,\ldots,i_d}
=
A^{(1)}[:,i_1,:]
A^{(2)}[:,i_2,:]
\cdots
A^{(d)}[:,i_d,:]
$$

積のshapeは

$$
(1\times r_1^A)
(r_1^A\times r_2^A)
\cdots
(r_{d-1}^A\times1)
=
1\times1
$$

なので、結果はスカラーになる。

成分表示では

$$
\boxed{
\mathcal A_{i_1,\ldots,i_d}
=
\sum_{\alpha_1=1}^{r_1^A}
\cdots
\sum_{\alpha_{d-1}=1}^{r_{d-1}^A}
\prod_{k=1}^{d}
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
}
$$

である。ただし

$$
\alpha_0=\alpha_d=1
$$

と約束した。同様に、

$$
\boxed{
\mathcal B_{i_1,\ldots,i_d}
=
\sum_{\beta_1=1}^{r_1^B}
\cdots
\sum_{\beta_{d-1}=1}^{r_{d-1}^B}
\prod_{k=1}^{d}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
}
$$

である。

ここで固定されているphysical index $i_1,\ldots,i_d$ は「dense tensorのどの要素か」を指定する。和を取るのは内部bond indexだけである。

### 1.1 3階TTで行列積を一段ずつ展開する

$d=3$ のとき、

$$
\begin{aligned}
A^{(1)}&\in\mathbb R^{1\times n_1\times r_1^A},\\
A^{(2)}&\in\mathbb R^{r_1^A\times n_2\times r_2^A},\\
A^{(3)}&\in\mathbb R^{r_2^A\times n_3\times1}.
\end{aligned}
$$

最初の2行列を掛けると、$(1,\alpha_2)$ 要素は

$$
\sum_{\alpha_1=1}^{r_1^A}
A^{(1)}_{1,i_1,\alpha_1}
A^{(2)}_{\alpha_1,i_2,\alpha_2}
$$

である。さらに第3コアのsliceを掛けると、

$$
\begin{aligned}
\mathcal A_{i_1,i_2,i_3}
&=
\sum_{\alpha_2=1}^{r_2^A}
\left(
\sum_{\alpha_1=1}^{r_1^A}
A^{(1)}_{1,i_1,\alpha_1}
A^{(2)}_{\alpha_1,i_2,\alpha_2}
\right)
A^{(3)}_{\alpha_2,i_3,1}\\
&=
\sum_{\alpha_1=1}^{r_1^A}
\sum_{\alpha_2=1}^{r_2^A}
A^{(1)}_{1,i_1,\alpha_1}
A^{(2)}_{\alpha_1,i_2,\alpha_2}
A^{(3)}_{\alpha_2,i_3,1}.
\end{aligned}
$$

$\alpha_1$ はコア1と2、$\alpha_2$ はコア2と3を接続するため、行列積の定義に従って和を取る。

### 1.2 TT-SVD由来でなくても成立する

この成分表示はTT-SVDの出力に限定されない。隣接コアのbond次元が一致し、境界rankが1である任意のコア列が、一意のdense tensorを定義する。

---

## 2. dense tensorの内積へ代入する

実数tensorのFrobenius内積は

$$
\langle\mathcal A,\mathcal B\rangle
=
\sum_{i_1=1}^{n_1}
\cdots
\sum_{i_d=1}^{n_d}
\mathcal A_{i_1,\ldots,i_d}
\mathcal B_{i_1,\ldots,i_d}.
$$

ここでは同じ位置の要素を掛けて全physical indexについて和を取る。TT成分を代入すると、

$$
\begin{aligned}
\langle\mathcal A,\mathcal B\rangle
=
&\sum_{i_1,\ldots,i_d}
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
\sum_{\beta_1,\ldots,\beta_{d-1}}\\
&\left(
A^{(1)}_{1,i_1,\alpha_1}
\cdots
A^{(d)}_{\alpha_{d-1},i_d,1}
\right)\\
&\qquad\times
\left(
B^{(1)}_{1,i_1,\beta_1}
\cdots
B^{(d)}_{\beta_{d-1},i_d,1}
\right).
\end{aligned}
$$

この巨大な和をphysical indexの全組合せごとに評価すると、TTで持っていてもdense全要素を走査するのに近い。重要なのは、積がsiteごとの因子に分離していることである。結合法則と分配法則を使い、左から部分縮約する。

---

## 3. left environmentの定義

第 $k$ siteまでのphysical indexと内部bond indexをすべて縮約し、右へ出る2本のbondだけを残したものを $E_k$ とする。

$$
\begin{aligned}
E_k(\alpha_k,\beta_k)
=
&\sum_{i_1,\ldots,i_k}
\sum_{\alpha_1,\ldots,\alpha_{k-1}}
\sum_{\beta_1,\ldots,\beta_{k-1}}\\
&\left[
A^{(1)}_{1,i_1,\alpha_1}
\cdots
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
\right]\\
&\times
\left[
B^{(1)}_{1,i_1,\beta_1}
\cdots
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
\right].
\end{aligned}
$$

残る自由添字は $\alpha_k$ と $\beta_k$ だけなので、

$$
\boxed{
E_k\in\mathbb R^{r_k^A\times r_k^B}
}
$$

である。

environmentはdenseな左部分テンソルそのものではない。左側で既に処理した全siteの寄与を、未処理の右側へ接続するbond空間上の行列へ正確に集約したものである。近似はまだ一切入っていない。

### 3.1 初期environmentはなぜ $[1]$ か

まだ何も縮約していない $k=0$ では積は空積であり、その値は乗法単位元1である。また $r_0^A=r_0^B=1$ なので、

$$
\boxed{
E_0=
\begin{bmatrix}
1
\end{bmatrix}
\in\mathbb R^{1\times1}
}
$$

と置く。

ここで $(1,1)$ はshapeであり、中身が常に1という意味ではない。縮約を進めるとshapeは変わり、最後は再び $(1,1)$ になるが、その唯一の要素は一般には内積である。

---

## 4. 3階TTでenvironmentを全展開する

$d=3$ では、最初のenvironmentは

$$
E_1(\alpha_1,\beta_1)
=
\sum_{i_1}
A^{(1)}_{1,i_1,\alpha_1}
B^{(1)}_{1,i_1,\beta_1}.
$$

第2 siteまで進めると、

$$
\begin{aligned}
E_2(\alpha_2,\beta_2)
=
\sum_{\alpha_1,\beta_1,i_2}
E_1(\alpha_1,\beta_1)
A^{(2)}_{\alpha_1,i_2,\alpha_2}
B^{(2)}_{\beta_1,i_2,\beta_2}.
\end{aligned}
$$

$E_1$ を代入すれば、

$$
\begin{aligned}
E_2(\alpha_2,\beta_2)
=
\sum_{i_1,i_2}
\sum_{\alpha_1,\beta_1}
&A^{(1)}_{1,i_1,\alpha_1}
A^{(2)}_{\alpha_1,i_2,\alpha_2}\\
&\times
B^{(1)}_{1,i_1,\beta_1}
B^{(2)}_{\beta_1,i_2,\beta_2}.
\end{aligned}
$$

最後に、

$$
E_3(1,1)
=
\sum_{\alpha_2,\beta_2,i_3}
E_2(\alpha_2,\beta_2)
A^{(3)}_{\alpha_2,i_3,1}
B^{(3)}_{\beta_2,i_3,1}.
$$

$E_2$ を代入すると、AとBの全コアおよび全physical indexの和が復元される。

$$
E_3(1,1)
=
\langle\mathcal A,\mathcal B\rangle.
$$

「次のsiteへenvironmentを渡す」とは、左側の全要素を捨てることではなく、右側の計算に必要な情報をbond添字 $\alpha_k,\beta_k$ ごとに集約して渡すことである。

---

## 5. environment更新式を定義から導く

$E_k$ の定義で、最後のsite $k$ の因子だけを外へ出す。

$$
\begin{aligned}
E_k(\alpha_k,\beta_k)
=
\sum_{\alpha_{k-1},\beta_{k-1},i_k}
&\Bigg[
\sum_{i_1,\ldots,i_{k-1}}
\sum_{\alpha_1,\ldots,\alpha_{k-2}}
\sum_{\beta_1,\ldots,\beta_{k-2}}\\
&\quad
A^{(1)}_{1,i_1,\alpha_1}
\cdots
A^{(k-1)}_{\alpha_{k-2},i_{k-1},\alpha_{k-1}}\\
&\quad\times
B^{(1)}_{1,i_1,\beta_1}
\cdots
B^{(k-1)}_{\beta_{k-2},i_{k-1},\beta_{k-1}}
\Bigg]\\
&\times
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}.
\end{aligned}
$$

角括弧内は、まさに

$$
E_{k-1}(\alpha_{k-1},\beta_{k-1})
$$

の定義である。従って、

$$
\boxed{
E_k(\alpha_k,\beta_k)
=
\sum_{\alpha_{k-1},\beta_{k-1},i_k}
E_{k-1}(\alpha_{k-1},\beta_{k-1})
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
}
\tag{1}
$$

を得る。

更新で消える添字は

$$
\alpha_{k-1},\quad\beta_{k-1},\quad i_k
$$

であり、次へ残す添字は

$$
\alpha_k,\quad\beta_k
$$

である。

shapeを並べると、

$$
\begin{aligned}
E_{k-1}&:\ (r_{k-1}^A,r_{k-1}^B),\\
A^{(k)}&:\ (r_{k-1}^A,n_k,r_k^A),\\
B^{(k)}&:\ (r_{k-1}^B,n_k,r_k^B),\\
E_k&:\ (r_k^A,r_k^B).
\end{aligned}
$$

### 5.1 最初と最後で確認する

$k=1$ では $E_0(1,1)=1$ なので、式 (1) は

$$
E_1(\alpha_1,\beta_1)
=
\sum_{i_1}
A^{(1)}_{1,i_1,\alpha_1}
B^{(1)}_{1,i_1,\beta_1}
$$

となる。

$k=d$ では $\alpha_d=\beta_d=1$ しかなく、

$$
E_d(1,1)
=
\sum_{\alpha_{d-1},\beta_{d-1},i_d}
E_{d-1}(\alpha_{d-1},\beta_{d-1})
A^{(d)}_{\alpha_{d-1},i_d,1}
B^{(d)}_{\beta_{d-1},i_d,1}
$$

となり、これは全siteの内積である。

---

## 6. `einsum`の添字と完全に対応させる

式 (1) はPyTorchでは次の1行に対応する。

```python
env = torch.einsum("ab,aic,bid->cd", env, core_a, core_b)
```

対応は

$$
\begin{aligned}
a&=\alpha_{k-1},&
b&=\beta_{k-1},&
i&=i_k,\\
c&=\alpha_k,&
d&=\beta_k.
\end{aligned}
$$

入力に現れ、出力 `cd` に現れない `a,b,i` は和を取る添字である。出力に残る `c,d` が新しいenvironmentの2軸になる。

例えば

$$
E_{k-1}\in\mathbb R^{2\times3},
\quad
A^{(k)}\in\mathbb R^{2\times7\times4},
\quad
B^{(k)}\in\mathbb R^{3\times7\times5}
$$

なら、

$$
E_k\in\mathbb R^{4\times5}
$$

である。AとBのleft rankが異なっていても、`a` と `b` は別添字なので問題ない。一方、physical dimensionは同じ `i` を共有するため一致が必要である。

---

## 7. 最終environmentはなぜスカラーか

境界rankから

$$
r_d^A=r_d^B=1
$$

なので、

$$
E_d\in\mathbb R^{1\times1}.
$$

その唯一の要素は内積である。

$$
E_d
=
\begin{bmatrix}
\langle\mathcal A,\mathcal B\rangle
\end{bmatrix}
$$

$1\times1$ は値が1という意味ではない。例えば内積が $3.25$ なら、

$$
E_d=
\begin{bmatrix}
3.25
\end{bmatrix}.
$$

PyTorchで `(1, 1)` のtensorに `squeeze()` を適用すると、値を変えずにshape `()` の0次元scalar tensorになる。`.item()` と異なりtensorのままなので、autogradの計算グラフを保てる。具体的なshape確認は実装ノートに分ける。

---

## 8. 実数TTと複素MPSの違い

実数TTでは

$$
\langle\mathcal A,\mathcal B\rangle
=
\sum_{\boldsymbol i}
\mathcal A_{\boldsymbol i}
\mathcal B_{\boldsymbol i}
$$

でよい。

複素MPSではbra側を複素共役する。

$$
\boxed{
\langle\psi,\phi\rangle
=
\sum_{\boldsymbol i}
\overline{\psi_{\boldsymbol i}}
\phi_{\boldsymbol i}
}
$$

従って更新式もA側を共役して

$$
E_k(\alpha_k,\beta_k)
=
\sum_{\alpha_{k-1},\beta_{k-1},i_k}
E_{k-1}(\alpha_{k-1},\beta_{k-1})
\overline{A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
$$

とする。

例えば $z=i$ なら、共役なしでは

$$
z z=i^2=-1
$$

だが、内積に必要なのは

$$
\overline z z=(-i)i=1
$$

である。実装ではbra側コアへ `.conj()` を適用する。

---

## 9. 計算量と中間tensor

全physical dimensionを $n_k=n$、AとBのrankを同程度の $r$ とする。

dense tensorの要素数は

$$
n^d
$$

であり、保存も内積も指数的に増える。

TTの保存量は

$$
\sum_{k=1}^{d}r_{k-1}n_kr_k
=
O(dnr^2)
$$

である。environmentの保存量は

$$
O(r^2)
$$

である。

3入力を素朴に一度で数えると1 siteが $O(nr^4)$ に見える。しかし縮約順を分けると、

$$
T(i_k,\alpha_k,\beta_{k-1})
=
\sum_{\alpha_{k-1}}
E_{k-1}(\alpha_{k-1},\beta_{k-1})
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
$$

を先に作り、

$$
E_k(\alpha_k,\beta_k)
=
\sum_{i_k,\beta_{k-1}}
T(i_k,\alpha_k,\beta_{k-1})
B^{(k)}_{\beta_{k-1},i_k,\beta_k}
$$

とすれば、同程度のrankでは各段階が $O(nr^3)$ である。従って全体は典型的に

$$
\boxed{O(dnr^3)}
$$

となる。

例えば $n=2,d=50,r=16$ ではdense要素数は

$$
2^{50}\simeq1.13\times10^{15}
$$

で、float64なら約9 PBに達する。一方、均一rankとみなしたTTの概算保存量は

$$
50\times2\times16^2=25{,}600
$$

要素であり、内積の演算量の代表スケールは

$$
50\times2\times16^3=409{,}600
$$

である。

ただしTT-rankが大きく成長すれば利点は小さくなる。計算量削減は「TTなら無条件」ではなく、問題に対してrankが十分小さいことに依存する。

---

## 10. 何に接続するか

この逐次縮約から、次をdense復元なしで計算できる。

- $\langle\mathcal A,\mathcal B\rangle$
- $\|\mathcal A\|_F$
- $\|\mathcal A-\mathcal B\|_F$
- TT-rounding前後の誤差
- 圧縮前後の重み差
- 複素MPSのoverlap

また、left environmentを順に更新する考え方は、DMRGで左右の環境を縮約し、局所自由度へ有効問題を集約する考え方の入口でもある。ただし本ノートのenvironmentは内積の完全縮約であり、Hamiltonianを含むDMRGの有効環境そのものではない。

次は [[52_TT_MPSのFrobeniusノルムと距離]] で、内積からノルムと距離を導く。
