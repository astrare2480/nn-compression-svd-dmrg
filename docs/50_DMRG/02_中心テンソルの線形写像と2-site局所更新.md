---
title: 中心テンソルの線形写像と2-site DMRG局所更新
aliases:
  - 中心テンソルと全状態の線形写像
  - DMRGの局所有効Hamiltonian
  - 2-site DMRG局所更新
tags:
  - DMRG
  - MPS
  - 有効Hamiltonian
  - Rayleigh商
  - SVD
  - bond dimension
---

# 中心テンソルの線形写像と2-site DMRG局所更新

## このノートの目的

DMRGの局所更新では、MPS全体を一度に最適化するのではなく、左右の環境を固定し、1個または隣接する2個の中心テンソルだけを変数とする。

このとき重要なのは、次の流れである。

```text
中心テンソル
    ↓ 固定した左右環境と縮約
全状態
    ↓ Hamiltonianを作用
全状態空間のベクトル
    ↓ 中心空間へ戻す
局所有効Hamiltonian
```

本ノートでは、この流れを

$$
x = Pc,
\qquad
H_{\mathrm{eff}} = P^{\mathsf T} H P
$$

という線形代数として整理し、3-site MPSの小さい数値例で、2-site局所最適化、SVDによる再分割、bond dimensionの打ち切りまでを追う。

扱う範囲は**1回の2-site局所更新と、それをsweepへ接続するための理論**である。MPOを用いた大規模実装、完全なDMRG sweep、ニューラルネットワークへの適用は扱わない。

関連する基礎理論は次を参照する。

- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/35_TT_MPSの等長写像と射影]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/40_TT_MPSの中心ノルムと内積の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/43_TT_MPSのSVD中心移動とSchmidt形]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/46_Eckart_Young_Mirsky定理とTT_MPS単一ボンド打ち切りの最適性]]

---

## 1. 固定した環境の中では、中心から全状態への写像は線形である

3-siteの実数MPSを考える。各コアのshapeを

$$
G^{(1)} \in \mathbb{R}^{1 \times n_1 \times r_1},
\qquad
G^{(2)} \in \mathbb{R}^{r_1 \times n_2 \times r_2},
\qquad
G^{(3)} \in \mathbb{R}^{r_2 \times n_3 \times 1}
$$

とする。第2コアを中心として、左右の固定部分を

$$
L_{i_1,\alpha}
=
G^{(1)}_{1,i_1,\alpha},
\qquad
R_{\beta,i_3}
=
G^{(3)}_{\beta,i_3,1}
$$

と書く。

第2コアと同じshapeを持つ任意の中心テンソル

$$
C \in \mathbb{R}^{r_1 \times n_2 \times r_2}
$$

から全状態を作る写像 $\mathcal F$ を

$$
\mathcal F(C)_{i_1,i_2,i_3}
=
\sum_{\alpha=1}^{r_1}
\sum_{\beta=1}^{r_2}
L_{i_1,\alpha}
C_{\alpha,i_2,\beta}
R_{\beta,i_3}
$$

と定義する。元のMPS全体は

$$
X = \mathcal F\!\left(G^{(2)}\right)
$$

である。

$L$ と $R$ を固定すると、$C$ は式の中に1回だけ現れるため、任意のスカラー $a,b$ と中心テンソル $C,D$ に対して

$$
\mathcal F(aC+bD)
=
a\mathcal F(C)+b\mathcal F(D)
$$

が成り立つ。

ここで線形なのは、**他のコアを固定したときの中心テンソルに関する写像**である。全コアを同時に変数としたMPS表現は、各コアの積を含むため、全変数について同時に線形ではない。

### 1.1 小さい成分例

$n_1=n_2=n_3=r_1=r_2=2$ とし、

$$
L
=
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix},
\qquad
R
=
\begin{bmatrix}
1 & 0 \\
3 & 1
\end{bmatrix}
$$

を固定する。中心の1成分だけを

$$
\Delta C_{1,1,2}=t
$$

と変え、他の成分を0とすると、全状態の変化は

$$
\Delta X_{:,1,:}
=
\begin{bmatrix}
3t & t \\
0 & 0
\end{bmatrix}
$$

となり、$i_2=2$ のsliceは0のままである。中心の局所的な変化が左右環境を通じて全状態へ伝わる一方、$t$ に対する依存は線形であることが分かる。

---

## 2. vectorizeすると $x=Pc$ になる

中心テンソル $C$ を1列のベクトル $c$ に、全状態 $X$ を1列のベクトル $x$ に並べる。$\mathcal F$ は有限次元の線形写像なので、ある行列 $P$ を用いて

$$
x = Pc
$$

と書ける。

中心空間と全状態空間の次元は

$$
d_{\mathrm{center}}
=
r_1 n_2 r_2,
\qquad
d_{\mathrm{full}}
=
n_1 n_2 n_3
$$

であり、

$$
P
\in
\mathbb{R}^{d_{\mathrm{full}} \times d_{\mathrm{center}}}
$$

となる。

$P$ は現在の中心テンソルの値ではなく、固定した左右環境とvectorizeの順序で決まる。

### 2.1 基底ベクトルから $P$ を作る

中心空間の標準基底を $e_k$ とする。各 $e_k$ を中心テンソルのshapeへ戻して全状態まで縮約し、その結果をvectorizeすると、$P$ の第 $k$ 列が得られる。

$$
P_{:,k}
=
\operatorname{vec}
\left(
\mathcal F
\left(
\operatorname{reshape}(e_k)
\right)
\right)
$$

したがって、小さい系では次の手順で $P$ を明示的に確認できる。

1. 中心空間の単位ベクトルを1本作る。
2. 中心テンソルのshapeへ戻す。
3. 左右環境と縮約する。
4. 全状態をvectorizeする。
5. 全ての基底について得た列を並べる。

大規模DMRGでは $P$ 自体を密行列として作らず、environment縮約によって $P$、$P^{\mathsf T}$、$P^{\mathsf T}HP$ の作用を計算する。明示的な $P$ は、写像の意味、shape、vectorize順序を確認するための教材用表現である。

---

## 3. 正準形では $P$ が等長写像になる

左右環境が

$$
L^{\mathsf T}L=I_{r_1},
\qquad
RR^{\mathsf T}=I_{r_2}
$$

を満たすとき、中心から全状態への写像はFrobeniusノルムを保存する。

$$
\left\|\mathcal F(C)\right\|_F
=
\left\|C\right\|_F
$$

vectorize後には

$$
\left\|Pc\right\|_2
=
\left\|c\right\|_2
$$

であり、全ての $c$ について成立するため

$$
P^{\mathsf T}P=I
$$

となる。この条件は「中心の座標が、全状態空間内の正規直交基底に対する係数になっている」ことを意味する。

### 3.1 正規化していない環境ではnorm行列が残る

例えば、左側だけが

$$
G^{(1)}_{1,:,1}
=
\begin{bmatrix}
3/5 & 4/5
\end{bmatrix}^{\mathsf T}
$$

で、2-site中心の次元が4なら、

$$
P
=
\begin{bmatrix}
\frac{3}{5}I_4 \\
\frac{4}{5}I_4
\end{bmatrix},
\qquad
P^{\mathsf T}P=I_4
$$

となる。

一方、左側を正規化せず

$$
G^{(1)}_{1,:,1}
=
\begin{bmatrix}
3 & 4
\end{bmatrix}^{\mathsf T}
$$

とすれば、

$$
P^{\mathsf T}P=25I_4,
\qquad
\left\|Pb\right\|_2^2
=
25\left\|b\right\|_2^2
$$

であり、等長写像ではない。

この違いは局所有効問題の形に直接現れる。

---

## 4. 全状態のエネルギーを中心空間へ引き戻す

実対称Hamiltonian

$$
H \in \mathbb{R}^{d_{\mathrm{full}} \times d_{\mathrm{full}}}
$$

に対するRayleigh商は

$$
E(x)
=
\frac{x^{\mathsf T}Hx}{x^{\mathsf T}x}
$$

である。$x=Pc$ を代入すると、

$$
E(c)
=
\frac{c^{\mathsf T}P^{\mathsf T}HPc}
{c^{\mathsf T}P^{\mathsf T}Pc}
$$

となる。ここで

$$
H_{\mathrm{eff}}
=
P^{\mathsf T}HP,
\qquad
N_{\mathrm{eff}}
=
P^{\mathsf T}P
$$

と定義すると、局所最適化は

$$
E(c)
=
\frac{c^{\mathsf T}H_{\mathrm{eff}}c}
{c^{\mathsf T}N_{\mathrm{eff}}c}
$$

の最小化になる。

$H_{\mathrm{eff}}$ は更新後のコアそのものではない。全Hamiltonianを、現在の固定環境が張る中心空間へ射影した演算子である。

```text
c
↓ P
x
↓ H
Hx
↓ P^T
H_eff c
```

### 4.1 正準形の場合

$P^{\mathsf T}P=I$ なら

$$
E(c)
=
\frac{c^{\mathsf T}H_{\mathrm{eff}}c}
{c^{\mathsf T}c}
$$

となり、停留条件は通常の固有値問題

$$
H_{\mathrm{eff}}c=Ec
$$

である。

### 4.2 正準形でない場合

$N_{\mathrm{eff}}\neq I$ なら、局所問題は一般化固有値問題

$$
H_{\mathrm{eff}}c
=
E N_{\mathrm{eff}}c
$$

となる。したがって、左右環境の正準化を確認せずに常に通常の固有値問題として解くのは正しくない。

正準形は単なる見た目の整理ではなく、norm行列を単位行列にして局所問題を安定で分かりやすい形にする役割を持つ。

---

## 5. なぜ最小固有値の固有ベクトルを選ぶのか

$H_{\mathrm{eff}}$ が実対称で、その正規直交固有ベクトルを $v_k$、固有値を $\lambda_k$ とする。

$$
H_{\mathrm{eff}}v_k
=
\lambda_k v_k
$$

正規化した中心ベクトルを

$$
c
=
\sum_k a_k v_k,
\qquad
\sum_k |a_k|^2=1
$$

と展開すると、

$$
c^{\mathsf T}H_{\mathrm{eff}}c
=
\sum_k |a_k|^2\lambda_k
$$

となる。これは固有値の加重平均なので、最小値は

$$
\lambda_{\min}
$$

であり、それを達成する中心は最小固有値に属する固有ベクトルである。

固有ベクトルの符号は一意ではない。

$$
v
\quad\text{と}\quad
-v
$$

は同じ固有状態を表す。また、最小固有値が縮退していれば、最小固有空間内のどの正規化ベクトルも解になる。実装を検証するときは特定成分の符号ではなく、

$$
\left\|H_{\mathrm{eff}}c-Ec\right\|_2,
\qquad
\left\|c\right\|_2,
\qquad
\frac{c^{\mathsf T}H_{\mathrm{eff}}c}{c^{\mathsf T}c}
$$

を確認する。

---

## 6. 1-site中心と2-site中心

1-site更新では、1個のコア $C$ を変数とする。左右bond dimensionは固定されるため、追加の工夫がなければ局所更新だけでrankを増やせない。

2-site更新では、隣接する2コアを一度結合し、

$$
B_{\alpha_{k-1},i_k,i_{k+1},\alpha_{k+1}}
=
\sum_{\alpha_k}
G^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
G^{(k+1)}_{\alpha_k,i_{k+1},\alpha_{k+1}}
$$

を中心テンソルとして最適化する。

結合後のshapeは

$$
B
\in
\mathbb{R}^{r_{k-1}\times n_k\times n_{k+1}\times r_{k+1}}
$$

である。最適化後に

$$
M
=
\operatorname{reshape}
\left(
B,
(r_{k-1}n_k)\times(n_{k+1}r_{k+1})
\right)
$$

としてSVDすれば、新しい中間bond dimensionを選び直せる。これが2-site更新の重要な利点である。

---

## 7. 3-site MPSによる2-site局所更新の完全な数値例

各siteの物理次元を2とし、最初のbond dimensionを全て1とする。

$$
G^{(1)}_{1,:,1}
=
\begin{bmatrix}
1 & 0
\end{bmatrix},
$$

$$
G^{(2)}_{1,:,1}
=
\begin{bmatrix}
1 & 2
\end{bmatrix},
\qquad
G^{(3)}_{1,:,1}
=
\begin{bmatrix}
3 & 4
\end{bmatrix}
$$

とする。

### 7.1 site 2とsite 3を結合する

2-site中心 $B$ は外積になり、matrixへreshapeすると

$$
M_{\mathrm{old}}
=
\begin{bmatrix}
3 & 4 \\
6 & 8
\end{bmatrix}
$$

である。

$(i_2,i_3)=(1,1),(1,2),(2,1),(2,2)$ の順にvectorizeすると、

$$
b_{\mathrm{old}}
=
\begin{bmatrix}
3 & 4 & 6 & 8
\end{bmatrix}^{\mathsf T}
$$

となる。

### 7.2 中心から全状態への写像

$G^{(1)}$ が第1物理indexの $i_1=1$ だけを選ぶため、全状態は

$$
x
=
\begin{bmatrix}
b \\
0_4
\end{bmatrix}
$$

となる。したがって、

$$
P
=
\begin{bmatrix}
I_4 \\
0_{4\times4}
\end{bmatrix}
\in
\mathbb{R}^{8\times4},
\qquad
P^{\mathsf T}P=I_4
$$

である。

### 7.3 Hamiltonianと局所有効Hamiltonian

全Hamiltonianを

$$
H
=
\operatorname{diag}(h,h)
\in
\mathbb{R}^{8\times8}
$$

とし、

$$
h
=
\begin{bmatrix}
-1 & 0 & 0 & -1 \\
0 & 1 & -1 & 0 \\
0 & -1 & 1 & 0 \\
-1 & 0 & 0 & -1
\end{bmatrix}
$$

とする。このとき

$$
H_{\mathrm{eff}}
=
P^{\mathsf T}HP
=
h
$$

である。

$h$ の固有値は

$$
-2,\ 0,\ 0,\ 2
$$

であり、最小固有値に対応する正規化固有ベクトルは、符号を除いて

$$
b_{\mathrm{opt}}
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 0 & 0 & 1
\end{bmatrix}^{\mathsf T}
$$

である。

これを2行2列へ戻すと、

$$
M_{\mathrm{opt}}
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

となる。

更新前の正規化エネルギーは

$$
E_{\mathrm{old}}
=
-\frac{117}{125}
=
-0.936
$$

であり、局所最適化後は

$$
E_{\mathrm{opt}}=-2
$$

である。

この例ではHamiltonianがsite 2とsite 3だけに非自明に作用し、固定したsite 1が必要な部分空間を既に含む。そのため、1回の局所更新だけでこの全問題の基底状態へ到達する。これは一般のDMRGが1回の更新で収束することを意味しない。

---

## 8. 最適化した2-site中心をSVDで2コアへ戻す

$M_{\mathrm{opt}}$ を

$$
M_{\mathrm{opt}}
=
U\Sigma V^{\mathsf T}
$$

とSVDする。この例では

$$
\sigma_1=\sigma_2=\frac{1}{\sqrt{2}}
$$

である。

### 8.1 右向きsweepでの配置

中心を右へ動かしたい場合は、

$$
G^{(2)}_{\mathrm{new}}
\leftarrow
U,
\qquad
G^{(3)}_{\mathrm{new}}
\leftarrow
\Sigma V^{\mathsf T}
$$

と配置する。左側の新しいコアが左正準になり、特異値は右側へ運ばれる。

### 8.2 左向きsweepでの配置

中心を左へ動かしたい場合は、

$$
G^{(2)}_{\mathrm{new}}
\leftarrow
U\Sigma,
\qquad
G^{(3)}_{\mathrm{new}}
\leftarrow
V^{\mathsf T}
$$

と配置する。右側の新しいコアが右正準になり、特異値は左側へ運ばれる。

打ち切りをしなければ、どちらも

$$
U\Sigma V^{\mathsf T}
=
M_{\mathrm{opt}}
$$

を厳密に再構成する。したがって、局所最適化は状態を変えるが、**打ち切りなしのSVD分割と特異値の置き場所の変更は表現だけを変え、状態とエネルギーを変えない**。

「最適化したから右正準になる」のではない。右側を $V^{\mathsf T}$ として配置したときに右正準になる。

---

## 9. bond dimensionを打ち切ると何が変わるか

この例の $M_{\mathrm{opt}}$ はrank 2である。中間bond dimensionを1へ制限するなら、rank-1近似を作る。

2個の特異値は等しいため、最良rank-1近似は一意ではない。参照用に第1方向を残すと、

$$
\widetilde M
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
$$

となる。

再正規化前のFrobenius誤差は、捨てた特異値の二乗和に等しい。

$$
\left\|M_{\mathrm{opt}}-\widetilde M\right\|_F^2
=
\sigma_2^2
=
\frac{1}{2}
$$

再正規化すると中心ベクトルは

$$
\widetilde b
=
\begin{bmatrix}
1 & 0 & 0 & 0
\end{bmatrix}^{\mathsf T}
$$

となり、この代表的なrank-1解のエネルギーは

$$
E_{\mathrm{trunc}}=-1
$$

である。したがって、この例のエネルギー変化は

$$
-0.936
\quad\longrightarrow\quad
-2
\quad\longrightarrow\quad
-2
\quad\longrightarrow\quad
-1
$$

である。各段階は順に、更新前、局所最適化後、打ち切りなしSVD分割後、rank-1打ち切り後を表す。

ここで目的関数の違いを区別する必要がある。

- 局所固有値問題は、現在の局所部分空間の中でエネルギーを最小化する。
- SVD打ち切りは、指定rankの中で状態のFrobenius誤差を最小化する。

SVD打ち切りはエネルギーを直接最小化していないため、打ち切り後にエネルギーが上がることがある。この例では $-2$ から $-1$ へ上がる。

また、特異値が縮退しているため、数値SVDが選ぶ特異ベクトルの向きは一意ではない。別の最良rank-1近似では異なる状態やエネルギーが得られる可能性がある。したがって、「自動的に常に $-1$ になる」とは一般化できない。

---

## 10. 1回の局所更新からsweepへ

結合 $(2,3)$ を更新するときは、site 1を固定して

$$
x=P_{23}b,
\qquad
H_{\mathrm{eff},23}
=
P_{23}^{\mathsf T}HP_{23}
$$

を作る。

結合 $(1,2)$ を更新するときは、site 3を固定して

$$
x=P_{12}a,
\qquad
H_{\mathrm{eff},12}
=
P_{12}^{\mathsf T}HP_{12}
$$

を作る。

中心の位置が変われば、固定される環境、写像 $P$、中心空間の次元、有効Hamiltonianも変わり得る。さらに、別の結合を更新すると環境そのものが変化するため、以前訪れた結合の $H_{\mathrm{eff}}$ も更新される。

このため、DMRGでは結合を一方向に1回だけ訪れて終わるのではなく、

```text
(1, 2) → (2, 3) → ... → (N-1, N)
                                  ↓
(1, 2) ← (2, 3) ← ... ← (N-1, N)
```

のように往復し、環境を更新しながら同じ結合を再訪する。

「sweep」という語を片道に使うか往復に使うかは文献や実装で異なる。本プロジェクトの演習では、片道のpassを1 sweepと数える場合は、その規約を明記する。

---

## 11. 何をもって収束と判断するか

sweep $s$ のエネルギーを $E_s$ とすると、基本的な収束指標は

$$
\Delta E_s
=
\left|E_s-E_{s-1}\right|
$$

である。しかし、$\Delta E_s$ が小さいことだけで、真の基底状態へ十分近いとは限らない。

例えば、bond dimensionの上限が小さすぎれば、表現可能な状態の範囲でエネルギーが停滞する。この場合、最適化は数値的に安定していても、表現容量が不足している。

そのため、少なくとも次を分けて確認する。

1. **最適化の安定性**：sweep間のエネルギー変化 $\Delta E_s$。
2. **打ち切りの影響**：捨てた特異値の二乗和、discarded weight、局所再構成誤差。
3. **表現容量の影響**：最大bond dimensionを増やしたときのエネルギー変化。

これらは別の問いに答える指標である。エネルギーが止まったこと、打ち切り誤差が小さいこと、bond dimensionが十分であることを同一視しない。

---

## 12. 各操作で状態とエネルギーは変わるか

処理を混同しないため、次のように整理する。

- **2コアの結合**：表現を1個の2-siteテンソルへまとめるだけなので、状態は変わらない。
- **局所固有値問題による最適化**：中心を更新するため、状態は一般に変わる。固定環境が定める局所部分空間の中では、厳密に最小固有対を解けばエネルギーは増加しない。
- **打ち切りなしSVD分割**：同じ2-siteテンソルを2コアへ戻すだけなので、状態とエネルギーは変わらない。
- **特異値を左右どちらへ置くかの変更**：打ち切りがなければgauge・正準中心の位置だけが変わり、状態は変わらない。
- **bond dimensionの打ち切り**：状態は一般に変わり、エネルギーも上がり得る。
- **次の結合への中心移動**：打ち切りを伴わない正確な因子移動だけなら、状態は変わらない。

特に、局所最適化とSVD打ち切りは目的が違う。

```text
局所最適化
→ エネルギーを下げる

SVD打ち切り
→ 指定rankで状態近似誤差を小さくする
```

---

## 13. 演習Notebookとの対応

[[notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb|00_two_site_dmrg_local_update_exercises.ipynb]] では、本ノートの数値例を次の順序で確認する。

1. 隣接する2コアを2-site中心へ結合する。
2. 中心空間の基底を1本ずつ写して $P$ を構成する。
3. $P^{\mathsf T}P=I$ と $x=Pb$ を確認する。
4. $H_{\mathrm{eff}}=P^{\mathsf T}HP$ を構成する。
5. 最小固有対を求め、局所エネルギーが下がることを確認する。
6. 最適化した中心をSVDで2コアへ戻す。
7. 特異値の置き場所と正準方向の対応を確認する。
8. rank-1打ち切りによる状態誤差とエネルギー変化を分けて確認する。

Notebookは小さい密行列を明示的に作る学習用実装である。大規模DMRGで同じ密行列 $P$ や $H_{\mathrm{eff}}$ を構成する実装を推奨するものではない。

---

## 14. このノートから言えること・まだ言えないこと

### 言えること

- 左右環境を固定すると、中心テンソルから全状態への写像は線形になる。
- 正準環境ではその写像が等長になり、局所norm行列が単位行列になる。
- 全Hamiltonianを中心空間へ引き戻すと、局所有効Hamiltonianが得られる。
- 2-site更新では局所最適化後にSVDを行い、bond dimensionを選び直せる。
- 打ち切りなしの分割と、打ち切りを伴う近似は、状態保存の有無が異なる。

### まだ言えないこと

- 一般の多site問題が1回の局所更新で収束すること。
- 小さいbond dimensionでも任意の基底状態を高精度に表現できること。
- 打ち切り後もエネルギーが必ず単調に下がること。
- この例だけから、大規模DMRGの計算量や収束速度を評価すること。
- 同じ局所更新をニューラルネットワークへそのまま適用できること。

ニューラルネットワークへ進むには、何をMPSまたはMPOとして表すか、何を目的関数とするか、どの変数を固定して局所更新するかを別途定義する必要がある。

---

## 15. 要点

DMRGの1回の局所更新は、中心テンソルを座標 $c$ とする部分空間上の変分問題として理解できる。

$$
x=Pc
$$

により全状態を表し、

$$
H_{\mathrm{eff}}=P^{\mathsf T}HP,
\qquad
N_{\mathrm{eff}}=P^{\mathsf T}P
$$

を作ると、局所問題は

$$
H_{\mathrm{eff}}c
=
E N_{\mathrm{eff}}c
$$

になる。正準形では $N_{\mathrm{eff}}=I$ なので、最小固有値問題へ簡約される。

2-site DMRGでは、最適化した2-site中心をSVDで再分割する。打ち切りなしのSVDは状態を保存するが、bond dimensionを制限する打ち切りは状態近似であり、エネルギー最小化とは別の操作である。この区別が、局所更新、正準中心移動、sweep、収束判定を理解する基礎になる。

## 参考文献

- U. Schollwöck, “The density-matrix renormalization group in the age of matrix product states,” *Annals of Physics* 326, 96–192 (2011). https://arxiv.org/abs/1008.3477
- TensorNetwork.org, “DMRG.” https://tensornetwork.org/mps/algorithms/dmrg/
- [[50_DMRG/01_SVDからDMRGへのつながり]]

