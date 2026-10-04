---
title: TT-matrixのmode分解とrank設計
tags:
  - TT-matrix
  - mode-design
  - TT-rank
  - parameter-count
---

# TT-matrixのmode分解とrank設計

## 1. 何を決める問題か

<code>nn.Linear(in_features, out_features)</code> の重みは

$$
W\in\mathbb{R}^{m\times n},
\qquad
m=\mathrm{out\_features},
\qquad
n=\mathrm{in\_features}
$$

である。TT-matrix化では、

$$
m=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

となるmode列

$$
\boldsymbol m=(m_1,\ldots,m_d),
\qquad
\boldsymbol n=(n_1,\ldots,n_d)
$$

を選ぶ。

$m_k$ は出力mode、$n_k$ は入力modeである。両者を逆にすると、重みの転置やforwardの添字対応まで変わる。

## 2. dense添字と複合添字

$d=2$なら、dense row index $i$ とcolumn index $j$ は

$$
i=i_1m_2+i_2,
\qquad
j=j_1n_2+j_2.
$$

従って、

$$
W[i,j]
=
\mathcal W[i_1,j_1,i_2,j_2]
$$

の具体的対応は

$$
\boxed{
\mathcal W[i_1,j_1,i_2,j_2]
=
W[i_1m_2+i_2,\ j_1n_2+j_2]
}
$$

である。

一般の $d$ ではrow-major添字を

$$
i
=
\sum_{k=1}^{d}
i_k\prod_{\ell=k+1}^{d}m_\ell,
\qquad
j
=
\sum_{k=1}^{d}
j_k\prod_{\ell=k+1}^{d}n_\ell
$$

と展開する。

## 3. siteのphysical size

各siteでは出力modeと入力modeを一組にし、

$$
q_k=m_kn_k
$$

というcombined physical sizeを作る。TT-SVDは

$$
(q_1,q_2,\ldots,q_d)
$$

のテンソルへ適用される。

mode設計では $m_k$ と $n_k$ を個別に均等化するだけでなく、実際のTT-SVDが見る

$$
(q_1,\ldots,q_d)
$$

の偏りも確認する必要がある。

## 4. パラメータ数

TT-matrix coreのshapeは

$$
G^{(k)}
\in
\mathbb{R}^{r_{k-1}\times m_k\times n_k\times r_k}.
$$

従ってTT重みのパラメータ数は

$$
\boxed{
N_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}m_kn_kr_k
}
$$

である。biasも含めるなら、

$$
N_{\mathrm{TTLinear}}
=
N_{\mathrm{TT}}+m.
$$

dense Linearは

$$
N_{\mathrm{dense}}
=
mn+m.
$$

圧縮率を

$$
C
=
\frac{N_{\mathrm{dense}}}{N_{\mathrm{TTLinear}}}
$$

と定義すれば、$C>1$ でパラメータ数が減る。

## 5. 素因数分解は出発点であって答えではない

$m$ と $n$ の素因数分解から候補を作れるが、同じ積でも並べ方は一意ではない。

例えば

$$
512=2^9,
\qquad
784=2^4\cdot7^2.
$$

3 siteなら、

$$
\boldsymbol m=(8,8,8)
$$

は自然な候補だが、入力側には

$$
(4,14,14),\quad
(7,8,14),\quad
(8,7,14)
$$

など多くの候補がある。積が同じでもmode順序が違えばcut unfoldingが変わるため、必要TT-rankと近似誤差も変わり得る。

## 6. 均等化の簡単な規則

現在の入門段階では、各次元を非減少な整数因子へ分け、次の辞書式scoreが最小の候補を選ぶ。

$$
\left(
\frac{\max_k s_k}{\min_k s_k},
\quad
\max_k s_k-\min_k s_k,
\quad
\max_k s_k
\right).
$$

ここで $(s_1,\ldots,s_d)$ は $m$ または $n$ の因子列である。最初にmax/min比、次にrange、最後に最大因子を小さくする。

Fashion-MNIST MLPの第1層

$$
W_1\in\mathbb{R}^{512\times784}
$$

を4 siteへ分けると、この規則では

$$
\boldsymbol m=(4,4,4,8),
\qquad
\boldsymbol n=(4,4,7,7)
$$

となる。積を確認すると、

$$
4\cdot4\cdot4\cdot8=512,
\qquad
4\cdot4\cdot7\cdot7=784.
$$

combined physical sizeは

$$
\boldsymbol q
=
(16,16,28,56).
$$

これは自動的に最良圧縮を与えるという定理ではなく、極端に偏ったsiteを避けるための初期設計である。

### 6.1 3 site候補をcombined physical sizeで比較する

3 siteで

$$
\boldsymbol m=(8,8,8),
\qquad
\boldsymbol n=(7,7,16)
$$

とすると、各siteのcombined physical sizeは

$$
\begin{aligned}
(q_1,q_2,q_3)
&=
(m_1n_1,m_2n_2,m_3n_3)\\
&=
(8\cdot7,8\cdot7,8\cdot16)\\
&=
(56,56,128).
\end{aligned}
$$

最大値と最小値の比は

$$
\frac{q_{\max}}{q_{\min}}
=
\frac{128}{56}
\approx
2.29.
$$

一方、同じ3 siteで

$$
\boldsymbol m=(4,8,16),
\qquad
\boldsymbol n=(4,7,28)
$$

とすると、

$$
\begin{aligned}
(q_1,q_2,q_3)
&=
(4\cdot4,8\cdot7,16\cdot28)\\
&=
(16,56,448),
\end{aligned}
$$

$$
\frac{q_{\max}}{q_{\min}}
=
\frac{448}{16}
=
28.
$$

後者の方が各コアのphysical sizeの偏りが大い。ただし、偏りが小さいことはTT-SVDの近似誤差が必ず小さいことを意味しない。必要rankは重みの相関構造とmode順序にも依存するため、最終的には特異値と再構成誤差を実測する。

## 7. mode順序は独立した設計変数

例えば同じ因子集合でも、

$$
\boldsymbol m=(2,8,32),
\qquad
\boldsymbol n=(49,8,2)
$$

なら

$$
\boldsymbol q=(98,64,64)
$$

である。一方、中央に大きいsiteを置けば

$$
\boldsymbol q=(64,98,64)
$$

となる。

固定rank

$$
(r_0,r_1,r_2,r_3)=(1,8,8,1)
$$

では、最初の例のTT重みパラメータ数は

$$
\begin{aligned}
N_{\mathrm{TT}}
&=
1\cdot98\cdot8
+
8\cdot64\cdot8
+
8\cdot64\cdot1\\
&=
784+4096+512\\
&=
5392.
\end{aligned}
$$

bias 512を含めると

$$
N_{\mathrm{TTLinear}}=5904.
$$

dense Linearの

$$
512\cdot784+512=401920
$$

と比べた名目圧縮率は

$$
\frac{401920}{5904}
\approx68.08
$$

である。ただし、これはrankを先に $(1,8,8,1)$ へ固定したパラメータ数の見積もりにすぎない。再構成誤差やaccuracyが許容範囲に入ることは保証しない。

## 8. shapeとrankは結合している

同じrank列を使っても、mode分解を変えると

$$
N_{\mathrm{TT}}
=
\sum_k r_{k-1}q_kr_k
$$

が変わる。またTT-SVDの各cut行列のshapeと特異値分布も変わるため、同じ最大rankでも近似誤差が変わる。

逆に同じmode分解でも、rankを大きくすれば一般に表現力とパラメータ数が増える。

従って、

$$
\boxed{
\text{mode shapeとTT-rankは独立に最適化できるとは限らない}
}
$$

と考える。

### 8.1 2-coreでrankとパラメータ数を最後まで追う

小さな例として、

$$
\boldsymbol m=(4,5),
\qquad
\boldsymbol n=(3,4)
$$

を考える。このとき、

$$
W\in\mathbb R^{20\times12}
$$

であり、2つのTT-matrix coreは

$$
G^{(1)}
\in
\mathbb R^{1\times4\times3\times r_1},
$$

$$
G^{(2)}
\in
\mathbb R^{r_1\times5\times4\times1}
$$

となる。TT重みのパラメータ数は、

$$
\begin{aligned}
N_{\mathrm{TT}}
&=
1\cdot4\cdot3\cdot r_1
+
r_1\cdot5\cdot4\cdot1\\
&=
12r_1+20r_1\\
&=
32r_1.
\end{aligned}
$$

従って、

- $r_1=1$ なら $N_{\mathrm{TT}}=32$
- $r_1=2$ なら $N_{\mathrm{TT}}=64$
- $r_1=5$ なら $N_{\mathrm{TT}}=160$
- $r_1=12$ なら $N_{\mathrm{TT}}=384$

である。dense重みのパラメータ数は

$$
20\cdot12=240
$$

なので、小さな行列を打ち切りなしで表すとTTの方が大きくなることがある。実際、この2-core TT-matrixは

$$
M
\in
\mathbb R^{(m_1n_1)\times(m_2n_2)}
=
\mathbb R^{12\times20}
$$

に対するrank-$r_1$ SVDと同じである。randomなdense重みで $M$ がfull rankなら、

$$
r_1
=
\operatorname{rank}(M)
=
\min(12,20)
=
12
$$

がexact復元に必要になり、$N_{\mathrm{TT}}=384>240$ となる。TT化すれば必ず圧縮になるのではなく、rank打ち切りが圧縮の本体である。

### 8.2 同じrankでもshapeが変わるとパラメータ数が変わる

Shape Aを

$$
\boldsymbol m=(8,8,8),
\qquad
\boldsymbol n=(7,7,16)
$$

とする。このとき

$$
(q_1,q_2,q_3)=(56,56,128).
$$

内部rankを $(r_1,r_2)=(r,r)$ とすると、

$$
\begin{aligned}
N_{\mathrm{TT}}^{(A)}
&=
1\cdot56\cdot r
+
r\cdot56\cdot r
+
r\cdot128\cdot1\\
&=
184r+56r^2.
\end{aligned}
$$

次にShape Bを

$$
\boldsymbol m=(4,8,16),
\qquad
\boldsymbol n=(4,7,28)
$$

とすると、

$$
(q_1,q_2,q_3)=(16,56,448)
$$

であり、同じrank $(r,r)$ でも

$$
\begin{aligned}
N_{\mathrm{TT}}^{(B)}
&=
1\cdot16\cdot r
+
r\cdot56\cdot r
+
r\cdot448\cdot1\\
&=
464r+56r^2.
\end{aligned}
$$

両者の差は

$$
\begin{aligned}
N_{\mathrm{TT}}^{(B)}
-
N_{\mathrm{TT}}^{(A)}
&=
(464r+56r^2)-(184r+56r^2)\\
&=
280r.
\end{aligned}
$$

さらに、Shape AとShape BではTT-SVDが適用するcut unfolding自体が異なる。従って同じrankを使っても、一般に

$$
\|W-\widetilde W_A\|_F
\ne
\|W-\widetilde W_B\|_F
$$

である。shapeとrankは別々に数値を指定できるが、圧縮率と誤差の評価では組として比較する必要がある。

## 9. 三つの探索方法

### 9.1 shape固定、rank探索

mode分解を一つ決め、rankだけを変える。例えば

$$
\boldsymbol m=(8,8,8),
\qquad
\boldsymbol n=(7,7,16)
$$

を固定し、

$$
(r_1,r_2)
\in
\{(2,2),(4,4),(8,8),(16,16)\}
$$

を比較する。各候補で、

$$
\frac{\|W-\widetilde W\|_F}{\|W\|_F},
\qquad
\frac{\|X\widetilde W^{\mathsf T}-XW^{\mathsf T}\|_F}{\|XW^{\mathsf T}\|_F},
$$

$$
N_{\mathrm{TT}},
\qquad
\text{accuracy}
$$

を記録する。rankと誤差の関係を学ぶ最初の実験に向く。

### 9.2 rank固定、shape探索

rank列を固定し、複数のmode分解についてパラメータ数と再構成誤差を比べる。mode順序の効果を調べられる。ただし、上のShape A/Bが示すように、同じrankでもパラメータ数は一致しない。そのため、「rankを合わせた」ことと「同じ圧縮率で比較した」ことは区別する。

### 9.3 parameter budget固定、shapeとrankを同時探索

$$
N_{\mathrm{TTLinear}}
\le
N_{\mathrm{budget}}
$$

を満たす候補の中で、weight error、output error、validation accuracyなどを比較する。実験としては最も直接的だが、探索範囲が大きくなる。

例えば、指定されたbudget内でrelative weight errorを最小化する問題は、

$$
\boxed{
\min_{\mathcal S,\boldsymbol r}
\frac{\|W-\widetilde W_{\mathcal S,\boldsymbol r}\|_F}{\|W\|_F}
\quad
\text{subject to}
\quad
N_{\mathrm{TTLinear}}(\mathcal S,\boldsymbol r)
\le
N_{\mathrm{budget}}
}
$$

と書ける。ここで

$$
\mathcal S
=
(m_1,\ldots,m_d;\,n_1,\ldots,n_d),
$$

$$
\boldsymbol r
=
(r_0,\ldots,r_d),
\qquad
r_0=r_d=1
$$

が探索変数である。

### 9.4 shape固定後の特異値にrank選択

shapeを固定すると、第 $k$ cutのunfoldingに対して

$$
\sigma_1^{(k)}
\ge
\sigma_2^{(k)}
\ge
\cdots
$$

が得られる。rank $r_k$ を選ぶとは、

$$
\sigma_1^{(k)},\ldots,\sigma_{r_k}^{(k)}
$$

を残し、

$$
\sigma_{r_k+1}^{(k)},
\sigma_{r_k+2}^{(k)},
\ldots
$$

を捨てることである。第 $k$ cutの局所誤差許容量を $\varepsilon_k$ とするなら、

$$
\sum_{j>r_k}
\left(\sigma_j^{(k)}\right)^2
\le
\varepsilon_k^2
$$

を満たす最小rankを選ぶ。すなわち、

$$
\boxed{
r_k
=
\min
\left\{
r:
\sum_{j>r}
\left(\sigma_j^{(k)}\right)^2
\le
\varepsilon_k^2
\right\}
}
$$

である。全体の許容誤差 $\varepsilon$ を $d-1$ 個のcutへ均等に二乗配分する基本例では、

$$
\varepsilon_k
=
\frac{\varepsilon}{\sqrt{d-1}}
$$

とする。ただし、shapeが変わるとunfolding自体が変わるため、同じ $\varepsilon$ でも特異値列、選ばれるrank列、パラメータ数は一般に変わる。

この尾部特異値と誤差予算の関係は、[[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/50_TT-roundingの誤差予算とrank選択]]で詳しく扱う。

現在の学習段階では、均等なmode分解を自動選択し、そのshapeを固定してrankの効果を学ぶ。joint optimizationは後続のモデル圧縮実験へ分ける。

## 10. exact TTが圧縮にならない例

4 siteの

$$
\boldsymbol m=(4,4,4,8),
\qquad
\boldsymbol n=(4,4,7,7)
$$

に対して打ち切りなしTT-SVDを行うと、実例では

$$
(r_0,r_1,r_2,r_3,r_4)
=
(1,16,256,56,1)
$$

となった。TT重みのパラメータ数は

$$
\begin{aligned}
N_{\mathrm{TT}}
&=
1\cdot4\cdot4\cdot16\\
&\quad+
16\cdot4\cdot4\cdot256\\
&\quad+
256\cdot4\cdot7\cdot56\\
&\quad+
56\cdot8\cdot7\cdot1\\
&=
470336.
\end{aligned}
$$

これはdense重みの

$$
512\cdot784=401408
$$

より大きい。biasを含めても

$$
470336+512=470848
>
401408+512=401920.
$$

この結果は失敗ではない。exact変換では情報を捨てないため、圧縮率よりforward等価性を検証している。実際の圧縮にはrank打ち切りが必要である。

PyTorchで使ったmode選択と保存済み結果は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]を参照する。
