---
title: TT-MLPのDMRG-like 2-core局所更新
aliases:
  - TT-MLPの2-core局所勾配更新
  - 物理DMRGからNNの2-core更新への接続
tags:
  - DMRG-like
  - TTLinear
  - 2-core update
  - gradient descent
  - SVD truncation
  - smoothness
---

# TT-MLPのDMRG-like 2-core局所更新

## このノートの目的

物理DMRGの2-site更新で使った

```text
隣接2サイトを結合
→ 局所変数を更新
→ SVDで再分割
→ 直交中心を移動
→ 次のbondへ進む
```

という構造を、TT-matrixで表したニューラルネットワークの重みへ接続する。

ただし、物理DMRGとニューラルネットワークでは局所問題の解き方が異なる。

- 物理DMRGでは、固定したenvironmentのもとでRayleigh商を局所的に最小化する。
- 本ノートのNN更新では、分類損失に対して結合テンソルを勾配で1ステップ更新する。
- したがって、NN側を「DMRGそのもの」とは呼ばず、**DMRG-likeな2-core局所更新**と呼ぶ。

本ノートは、次の内容を途中式から整理する。

1. TT-matrixの隣接2 coreの結合と行列化。
2. 2-core更新でbond rankが増え得る理由。
3. 結合・切り捨てなし再分割が重みを変えないこと。
4. rank制約の入れ子と、最適化誤差・ansatz誤差の分解。
5. 物理DMRGの局所単調性と、NNの勾配1ステップの違い。
6. 勾配更新後のSVD打ち切りまで含めて損失低下を保証する十分条件。
7. 正準形が局所誤差と重み全体の誤差を一致させる理由。
8. 局所更新方式とデータ利用方式を混同しないための分類。

PyTorchでの具体的な軸順、autograd、SVD、書き戻し、および保存済み数値結果は、[[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]へ分ける。全学習データの勾配を使って左右sweepするFashion-MNIST実験設計は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]へ分ける。

---

## 0. 物理とNNの局所問題を線形写像から見る

### 0.1 物理側：$\psi=Pt$

2-site中心テンソルの全要素を並べたベクトルを $t$ とする。中心以外のMPS coreを固定すれば、全状態 $\psi$ は $t$ の線形写像として

$$
\psi=Pt
$$

と書ける。$P$ の各列は、局所basis vectorを固定した左右environmentへ埋め込んで得られる全系basis vectorである。

全状態のノルムは

$$
\begin{aligned}
\lVert\psi\rVert_2^2
&=
(Pt)^{\mathsf T}(Pt)
\\
&=
t^{\mathsf T}P^{\mathsf T}Pt
\\
&=
t^{\mathsf T}Nt
\end{aligned}
$$

であり、

$$
N=P^{\mathsf T}P
$$

を局所norm行列と呼ぶ。任意の $u$ について

$$
u^{\mathsf T}Nu
=
u^{\mathsf T}P^{\mathsf T}Pu
=
\lVert Pu\rVert_2^2
\ge
0
$$

なので、$N$ は正定値半正定値である。

pairの左側を左正準、右側を右正準にすると、$P$ の列basisが正規直交になり、

$$
P^{\mathsf T}P=I
$$

となる。これが、正準環境で局所norm行列が単位行列になる理由である。

Hamiltonianを $H$ とすると、Rayleigh商は

$$
\begin{aligned}
E(t)
&=
\frac{\psi^{\mathsf T}H\psi}{\psi^{\mathsf T}\psi}
\\
&=
\frac{(Pt)^{\mathsf T}H(Pt)}{(Pt)^{\mathsf T}(Pt)}
\\
&=
\frac{t^{\mathsf T}P^{\mathsf T}HPt}
{t^{\mathsf T}P^{\mathsf T}Pt}
\\
&=
\frac{t^{\mathsf T}H_{\mathrm{eff}}t}
{t^{\mathsf T}Nt},
\end{aligned}
$$

$$
H_{\mathrm{eff}}=P^{\mathsf T}HP
$$

である。正準環境では $N=I$ なので

$$
E(t)
=
\frac{t^{\mathsf T}H_{\mathrm{eff}}t}{t^{\mathsf T}t}
$$

となり、通常の最小固有値問題へ帰着する。非正準環境では

$$
H_{\mathrm{eff}}t
=
ENt
$$

という一般化固有値問題になる。

### 0.2 NN側：固定した他の部分のもとでのaffine写像

入力sample $x$ とpair以外のparameterを固定する。局所テンソル $T$ の全要素を並べたものを $t$ とする。$T$ の直後までが線形なら、そのpre-activationは

$$
z=P_xt+b
$$

と書ける。$P_x$ は入力 $x$ とpair外のcoreを縮約して得るsample依存の線形写像、$b$ はbiasである。

class $y$ に対するsoftmax cross-entropyは

$$
\ell(z,y)
=
\log
\left(
\sum_{c=1}^{C}e^{z_c}
\right)
-z_y
$$

である。softmax確率を

$$
p_c
=
\frac{e^{z_c}}{\sum_j e^{z_j}}
$$

とすると、勾配とヘッセ行列は

$$
\nabla_z\ell
=
p-e_y,
$$

$$
\nabla_z^2\ell
=
\operatorname{diag}(p)-pp^{\mathsf T}
$$

である。任意のベクトル $v$ に対して

$$
\begin{aligned}
v^{\mathsf T}
\nabla_z^2\ell
v
&=
\sum_c p_cv_c^2
-
\left(
\sum_cp_cv_c
\right)^2
\\
&=
\operatorname{Var}_{c\sim p}(v_c)
\\
&\ge
0
\end{aligned}
$$

なので、$\ell$ はlogits $z$ について凸である。

$z=P_xt+b$ がaffineなら、chain ruleから

$$
\nabla_t^2\ell(P_xt+b,y)
=
P_x^{\mathsf T}
\left(
\operatorname{diag}(p)-pp^{\mathsf T}
\right)
P_x
$$

である。任意の $u$ について

$$
u^{\mathsf T}
\nabla_t^2\ell
u
=
(P_xu)^{\mathsf T}
\left(
\operatorname{diag}(p)-pp^{\mathsf T}
\right)
(P_xu)
\ge0
$$

なので、局所変数 $t$ についても凸性が保存される。sample平均も凸関数の非負加重和なので凸である。

### 0.3 現在のTT-MLPでは、この凸性を一般保証に使えない

上の結論には、**logitsが $t$ のaffine写像である**という条件がある。現在のNotebook 04では更新対象が第1層 `fc1` のTT coreであり、その後にReLU、`fc2`、再度ReLU、`fc3` が続く。

ReLUのactivation patternが変わると、$t$ から最終logitsへの写像は1つのaffine写像ではなくなる。したがって、softmax cross-entropyがlogitsについて凸であることだけから、$L(t)$ の大域的凸性は言えない。

1変数・2 classの反例として

$$
z_1(t)=\operatorname{ReLU}(t),
\qquad
z_2(t)=0
$$

とし、正解classを1とする。損失は

$$
L(t)
=
\log
\left(
1+e^{-\operatorname{ReLU}(t)}
\right)
$$

である。

$$
L(-1)=\log2\approx0.6931,
$$

$$
L(1)=\log(1+e^{-1})\approx0.3133,
$$

$$
L(0)=\log2\approx0.6931
$$

となる。凸なら中点で

$$
L(0)
\le
\frac{L(-1)+L(1)}{2}
\approx
0.5032
$$

でなければならないが、実際には

$$
0.6931>0.5032
$$

なので凸ではない。

したがって、

- 最終分類器を局所変数として直接更新し、logitsまでaffineな場合。
- 第1層のTT coreをReLUを通して更新する現在のTT-MLP。

を同じ凸局所問題として扱ってはいけない。現在の設計が勾配1ステップとvalidation診断を使う理由の1つがここにある。

---

## 1. TT-matrixの隣接2 coreを結合する

TT-matrixの第 $k$ coreと第 $k+1$ coreを

$$
G^{(k)}
\in
\mathbb{R}^{r_{k-1} \times m_k \times n_k \times r_k},
$$

$$
G^{(k+1)}
\in
\mathbb{R}^{r_k \times m_{k+1} \times n_{k+1} \times r_{k+1}}
$$

とする。$m_k$ は出力mode、$n_k$ は入力mode、$r_k$ は共有bond dimensionである。

2 coreを共有bond index $eta$ で縮約した結合テンソルを

$$
T
\in
\mathbb{R}^{
r_{k-1}
\times m_k
\times n_k
\times m_{k+1}
\times n_{k+1}
\times r_{k+1}
}
$$

とすると、全要素は

$$
\begin{aligned}
&T_{
\alpha,
i_k,
j_k,
i_{k+1},
j_{k+1},
\gamma
}
\\
&\qquad=
\sum_{\beta=1}^{r_k}
G^{(k)}_{\alpha,i_k,j_k,\beta}
G^{(k+1)}_{\beta,i_{k+1},j_{k+1},\gamma}
\end{aligned}
$$

である。

左側の複合indexと右側の複合indexを

$$
p=(\alpha,i_k,j_k),
\qquad
q=(i_{k+1},j_{k+1},\gamma)
$$

とまとめると、$T$ は行列

$$
M_{p,q}
=
T_{
\alpha,
i_k,
j_k,
i_{k+1},
j_{k+1},
\gamma
}
$$

にreshapeできる。行列のshapeは

$$
M
\in
\mathbb{R}^{
(r_{k-1}m_kn_k)
\times
(m_{k+1}n_{k+1}r_{k+1})
}
$$

である。

更新前は、左coreを行列 $A$、右coreを行列 $B$ と見れば

$$
M=AB,
$$

$$
A
\in
\mathbb{R}^{(r_{k-1}m_kn_k)\times r_k},
\qquad
B
\in
\mathbb{R}^{r_k\times(m_{k+1}n_{k+1}r_{k+1})}
$$

なので、

$$
\operatorname{rank}(M)
\le
r_k
$$

である。

---

## 2. 2-core局所更新ではrankが増え得る

### 2.1 1-core更新との違い

2 coreを分けたまま更新する場合、結合行列は常に

$$
M=AB
$$

という積である。$A$ の列数と $B$ の行数が $r_k$ なら、更新後も

$$
\operatorname{rank}(M)
\le
r_k
$$

から出られない。

一方、2-core更新では、$M$ または $T$ 自体を独立変数として動かす。例えば

$$
T_{\mathrm{new}}
=
T-eta\nabla_TL(T)
$$

と更新する。勾配 $\nabla_TL$ は、元のrank $r_k$ の積構造を保つ方向に限定されない。そのため、行列化した

$$
M_{\mathrm{new}}
=
M-eta\nabla_ML(M)
$$

のrankは $r_k$ を超え得る。

### 2.2 小さい全要素例

更新前を

$$
M
=
\begin{pmatrix}
1 & 2 \\
2 & 4
\end{pmatrix}
$$

とする。第2行は第1行の2倍なので、

$$
\operatorname{rank}(M)=1
$$

である。

更新量が

$$
-\eta\nabla_ML
=
\begin{pmatrix}
0 & 0 \\
0 & 1
\end{pmatrix}
$$

なら、

$$
M_{\mathrm{new}}
=
\begin{pmatrix}
1 & 2 \\
2 & 5
\end{pmatrix}
$$

となる。行列式は

$$
\det(M_{\mathrm{new}})
=
1\cdot5-2\cdot2
=
1
\neq
0
$$

なので、

$$
\operatorname{rank}(M_{\mathrm{new}})=2
$$

である。

これはrankが必ず増えることを示す例ではない。**結合テンソルを直接更新すれば、元bondのrank制約の外へ出る自由度がある**ことを示す例である。

### 2.3 形状から決まる最大rank

更新後のrankの上限は、元のbond dimensionだけではなく、行列化後の左右の次元で決まる。

$$
\operatorname{rank}(M_{\mathrm{new}})
\le
\min
\left(
r_{k-1}m_kn_k,
m_{k+1}n_{k+1}r_{k+1}
\right)
$$

物理次元が左右とも $d$、外側bondが左右とも $chi$ なら、行列shapeは

$$
(\chi d)\times(d\chi)
$$

なので、形状上の上限は $d\chi$ である。これは行列shapeから導く上限であり、特定の更新で必ずそのrankになるという主張ではない。

---

## 3. 結合と切り捨てなし再分割は重みを変えない

### 3.1 結合は縮約順序の変更である

pairより左の縮約結果を $L$、右の縮約結果を $R$ と書く。重みの1要素は概念的に

$$
W
=
\sum_{\alpha,\beta,\gamma}
L_{\alpha}
G^{(k)}_{\alpha,\beta}
G^{(k+1)}_{\beta,\gamma}
R_{\gamma}
$$

と書ける。先に $eta$ を縮約して

$$
T_{\alpha,gamma}
=
\sum_{\beta}
G^{(k)}_{\alpha,\beta}
G^{(k+1)}_{\beta,gamma}
$$

とすれば、

$$
W
=
\sum_{\alpha,\gamma}
L_{\alpha}
T_{\alpha,gamma}
R_{\gamma}
$$

である。有限和の順序を変えただけなので、結合だけでは $W$ は変わらない。

小さい行列例として

$$
l=egin{pmatrix}1&2\end{pmatrix},
\quad
G_1=
\begin{pmatrix}
1&2\\
0&1
\end{pmatrix},
\quad
G_2=
\begin{pmatrix}
1&0\\
1&1
\end{pmatrix},
\quad
r=
\begin{pmatrix}
1\\
1
\end{pmatrix}
$$

を考える。順に掛ければ

$$
lG_1
=
\begin{pmatrix}1&4\end{pmatrix},
\qquad
G_2r
=
\begin{pmatrix}1\\2\end{pmatrix},
$$

$$
(lG_1)(G_2r)
=
1\cdot1+4\cdot2
=
9
$$

である。一方、先に結合すると

$$
G_1G_2
=
\begin{pmatrix}
3&2\\
1&1
\end{pmatrix},
$$

$$
l(G_1G_2)
=
\begin{pmatrix}5&4\end{pmatrix},
$$

$$
l(G_1G_2)r
=
5+4
=
9
$$

となり一致する。

### 3.2 切り捨てなしSVDは結合テンソルを再現する

行列化した $M$ の還元SVDを

$$
M
=
U\Sigma V^{\mathsf T}
$$

とする。全ての特異値を保持すれば、

$$
M
=
U(\Sigma V^{\mathsf T})
$$

である。$U$ と $\Sigma V^{\mathsf T}$ をそれぞれcore shapeへreshapeすれば、再結合後は元の $T$ に戻る。

例えば

$$
M
=
\begin{pmatrix}
2&1\\
1&2
\end{pmatrix}
$$

に対して、

$$
U
=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix},
$$

$$
\Sigma V^{\mathsf T}
=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
3&3\\
1&-1
\end{pmatrix}
$$

と取れる。積の各要素は

$$
\begin{aligned}
(1,1)&:\ \frac12(1\cdot3+1\cdot1)=2,\\
(1,2)&:\ \frac12(1\cdot3+1\cdot(-1))=1,\\
(2,1)&:\ \frac12(1\cdot3+(-1)\cdot1)=1,\\
(2,2)&:\ \frac12(1\cdot3+(-1)\cdot(-1))=2
\end{aligned}
$$

なので、元の $M$ を全要素で再現する。

### 3.3 exact split後のbond dimension

切り捨てなし再分割後のbond dimensionは、還元SVDが返す行列shape上の成分数である。数値的に0に近い特異値を自動で削る処理を入れない限り、数値rankと同じとは限らない。

一方、数学的な最小bond dimensionは

$$
\rho=\operatorname{rank}(M)
$$

である。元のbond dimensionより小さくなる場合もある。例えば

$$
G_k
=
\begin{pmatrix}
1&0\\
2&0
\end{pmatrix},
\qquad
G_{k+1}
=
\begin{pmatrix}
1&2\\
0&0
\end{pmatrix}
$$

はbond 2で表されているが、積は

$$
G_kG_{k+1}
=
\begin{pmatrix}
1&2\\
2&4
\end{pmatrix}
$$

でrank 1である。

---

## 4. rank制約の入れ子

bond dimension 1の

$$
G_k
=
\begin{pmatrix}
1\\
2
\end{pmatrix},
\qquad
G_{k+1}
=
\begin{pmatrix}
1&2
\end{pmatrix}
$$

の積は

$$
G_kG_{k+1}
=
\begin{pmatrix}
1&2\\
2&4
\end{pmatrix}
$$

である。ゼロ列とゼロ行を加えて

$$
G_k'
=
\begin{pmatrix}
1&0\\
2&0
\end{pmatrix},
\qquad
G_{k+1}'
=
\begin{pmatrix}
1&2\\
0&0
\end{pmatrix}
$$

とすればbond dimensionは2になるが、

$$
G_k'G_{k+1}'
=
\begin{pmatrix}
1&2\\
2&4
\end{pmatrix}
$$

のままである。

rank上限 $r$ のTTで表せる重み集合を $\mathcal{M}_{\mathrm{TT}}(r)$ とすれば、

$$
\mathcal{M}_{\mathrm{TT}}(r)
\subseteq
\mathcal{M}_{\mathrm{TT}}(r+1)
$$

である。損失の制約付き最小値を

$$
L_r^*
=
\min_{W\in\mathcal{M}_{\mathrm{TT}}(r)}L(W)
$$

と定義すると、候補集合が広がるため

$$
L_{r+1}^*
\le
L_r^*
$$

である。

これは、rankを増やせば実際の学習損失が必ず下がるという意味ではない。**rank制約の範囲内で到達可能な理想最小値は悪化しない**という意味である。実際のアルゴリズムがその最小値へ到達するかは別問題である。

---

## 5. 最適化誤差とansatz誤差

制約なしの最小値を

$$
L_{\mathrm{full}}^*
=
\min_WL(W)
$$

とする。実際に得られたrank $r$ 以下のモデルを $\widetilde W$ とすると、$L_r^*$ を足して引くことで

$$
\begin{aligned}
L(\widetilde W)-L_{\mathrm{full}}^*
&=
\left(L(\widetilde W)-L_r^*\right)
+
\left(L_r^*-L_{\mathrm{full}}^*\right)
\\
&=
\text{最適化誤差}
+
\text{ansatz誤差}
\end{aligned}
$$

と分解できる。

- 最適化誤差 $L(\widetilde W)-L_r^*$ は、rank $r$ の候補集合の中で最良点へ届いていない分である。
- ansatz誤差 $L_r^*-L_{\mathrm{full}}^*$ は、rank $r$ に候補を制限したために避けられない分である。

$\widetilde W\in\mathcal{M}_{\mathrm{TT}}(r)$ なら、

$$
L_r^*
\le
L(\widetilde W)
$$

なので、両方の差は非負である。$\widetilde W$ がrank制約の外にある場合、この非負性の説明をそのまま使ってはいけない。

### 5.1 数値例

$$
X
=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix},
\qquad
L(W)=\lVert X-W\rVert_F^2
$$

を考える。制約なしなら $W=X$ とできるので、

$$
L_{\mathrm{full}}^*=0
$$

である。rank 1制約ではEckart--Young--Mirsky定理から、第2特異値 $1$ を捨てるため

$$
L_1^*=1^2=1
$$

である。学習途中で

$$
L(\widetilde W)=1.01
$$

のrank 1モデルが得られたなら、

$$
\underbrace{1.01-0}_{\text{全体の差}}
=
\underbrace{1.01-1}_{\text{最適化誤差 }0.01}
+
\underbrace{1-0}_{\text{ansatz誤差 }1}
$$

である。最適化を改善して0.01を減らしても、rank 1のままではansatz誤差1は消えない。

---

## 6. 物理DMRGとNN局所更新の単調性は異なる

### 6.1 物理DMRG

固定したenvironmentのもとで、2-site中心をvectorizeしたものを $c$ とする。正準形なら局所エネルギーは

$$
E(c)
=
\frac{c^{\mathsf T}H_{\mathrm{eff}}c}{c^{\mathsf T}c}
$$

である。局所固有値問題を厳密に解いて得た $c_{\mathrm{new}}$ は、古い $c_{\mathrm{old}}$ も含む同じ候補空間での最小点なので、切り捨て前には

$$
E(c_{\mathrm{new}})
\le
E(c_{\mathrm{old}})
$$

が成り立つ。

ただし、SVDでbond dimensionを制限した後は局所最適解そのものからずれるため、切り捨て後のエネルギーがわずかに上がる場合がある。

### 6.2 NNの勾配1ステップ

結合テンソルの全要素を並べたベクトルを $t$、損失を $L(t)$、勾配を

$$
g=\nabla L(t)
$$

とする。勾配1ステップは

$$
t_{\mathrm{new}}
=
t-eta g
$$

である。

$L$ が更新点の近くで十分滑らかで、ヘッセ行列を $H$ とすると、Taylor展開は

$$
\begin{aligned}
L(t-\eta g)
&=
L(t)
+
\nabla L(t)^{\mathsf T}(-\eta g)
+
\frac12(-\eta g)^{\mathsf T}H(-\eta g)
+
\text{高次項}
\\
&=
L(t)
-
\eta\lVert g\rVert_2^2
+
\frac{\eta^2}{2}g^{\mathsf T}Hg
+
\text{高次項}
\end{aligned}
$$

となる。2次近似で $g^{\mathsf T}Hg>0$ なら、

$$
-\eta\lVert g\rVert_2^2
+
\frac{\eta^2}{2}g^{\mathsf T}Hg
<0
$$

となる十分小さいstepは

$$
0<\eta<
\frac{2\lVert g\rVert_2^2}{g^{\mathsf T}Hg}
$$

である。物理DMRGの局所最小化とは違い、任意の学習率で損失低下が保証されるわけではない。

### 6.3 1変数の例

$$
L(t)=(t-3)^2
$$

とし、$t=0$ から始める。勾配とヘッセ行列は

$$
g=L'(0)=-6,
\qquad
H=L''(0)=2
$$

である。2次式なので高次項は0であり、条件は

$$
\eta
<
\frac{2\cdot(-6)^2}{(-6)\cdot2\cdot(-6)}
=1
$$

となる。

- $\eta=0.1$ なら $t_{\mathrm{new}}=0.6$、損失は $9\to5.76$。
- $\eta=1$ なら $t_{\mathrm{new}}=6$、損失は $9\to9$。
- $\eta=1.1$ なら $t_{\mathrm{new}}=6.6$、損失は $9\to12.96$。

---

## 7. $\beta$-smoothnessと降下補題

### 7.1 定義

微分可能な関数 $L:\mathbb{R}^N\to\mathbb{R}$ の勾配が $\beta$-Lipschitzであるとは、全ての $x,y$ について

$$
\lVert\nabla L(x)-\nabla L(y)\rVert_2
\le
\beta\lVert x-y\rVert_2
$$

が成り立つことである。このとき $L$ を $\beta$-smoothと呼ぶ。

例えば

$$
L(t)=(t-3)^2
$$

では

$$
|L'(x)-L'(y)|
=
|2(x-3)-2(y-3)|
=
2|x-y|
$$

なので $\beta=2$ と取れる。

また、

$$
L(M)=\frac12\lVert M-X\rVert_F^2
$$

では

$$
\nabla L(M)=M-X
$$

だから、

$$
\lVert\nabla L(M)-\nabla L(M')\rVert_F
=
\lVert M-M'\rVert_F
$$

であり $\beta=1$ である。

ReLUを $\operatorname{ReLU}(t)=\max(0,t)$ とすると、

$$
f(t)
=
\frac12\operatorname{ReLU}(t)^2
$$

は

$$
f'(t)
=
\operatorname{ReLU}(t)
$$

を持つ。ReLUは1-Lipschitzなので、

$$
|f'(x)-f'(y)|
\le
|x-y|
$$

であり、この $f$ は $\beta=1$ のsmoothな例である。ReLUそのものが折れ点で微分不能であることと、ReLUを合成した全ての関数が非smoothであることは同じ主張ではない。

$L$ が2回微分可能なら、考えている領域で

$$
\lVert\nabla^2L(z)\rVert_2
\le
\beta
$$

が成り立つことは、勾配が $\beta$-Lipschitzであるための十分条件になる。特に正定値半正定値行列 $H$ を使う二次関数

$$
L(x)
=
\frac12x^{\mathsf T}Hx+b^{\mathsf T}x+c
$$

では

$$
\nabla^2L(x)=H
$$

なので、

$$
\beta=\lambda_{\max}(H)
$$

と取れる。

ReLUそのものは折れ点で微分不能なので、ReLUを含むNN損失が全空間で同じ $\beta$ によりsmoothであるとは一般に仮定できない。以下の結果をNNへ使うときは、この仮定を独立に確認するか、診断的な目安として扱う。

### 7.2 降下補題の導出

$$
\varphi(s)
=
L(x+s(y-x)),
\qquad
0\le s\le1
$$

とおく。chain ruleから

$$
\varphi'(s)
=
\nabla L(x+s(y-x))^{\mathsf T}(y-x)
$$

である。微積分の基本定理により

$$
L(y)-L(x)
=
\int_0^1
\nabla L(x+s(y-x))^{\mathsf T}(y-x)
\,ds
$$

となる。ここから

$$
\nabla L(x)^{\mathsf T}(y-x)
=
\int_0^1
\nabla L(x)^{\mathsf T}(y-x)
\,ds
$$

を引くと、

$$
\begin{aligned}
&L(y)-L(x)-\nabla L(x)^{\mathsf T}(y-x)
\\
&\quad=
\int_0^1
\left(
\nabla L(x+s(y-x))-\nabla L(x)
\right)^{\mathsf T}
(y-x)
\,ds
\end{aligned}
$$

である。Cauchy--Schwarz不等式とLipschitz性から、被積分関数の絶対値は

$$
\begin{aligned}
&\left|
\left(
\nabla L(x+s(y-x))-\nabla L(x)
\right)^{\mathsf T}
(y-x)
\right|
\\
&\quad\le
\lVert
\nabla L(x+s(y-x))-\nabla L(x)
\rVert_2
\lVert y-x\rVert_2
\\
&\quad\le
\beta s
\lVert y-x\rVert_2^2
\end{aligned}
$$

である。したがって、

$$
\begin{aligned}
&\left|
L(y)-L(x)-\nabla L(x)^{\mathsf T}(y-x)
\right|
\\
&\quad\le
\int_0^1
\beta s\lVert y-x\rVert_2^2
\,ds
\\
&\quad=
\frac{\beta}{2}
\lVert y-x\rVert_2^2
\end{aligned}
$$

を得る。

絶対値不等式の上側だけを書けば、

$$
L(y)
\le
L(x)
+
\nabla L(x)^{\mathsf T}(y-x)
+
\frac{\beta}{2}
\lVert y-x\rVert_2^2
$$

である。損失の上限を使って低下を示すため、以下ではこの上側を使う。

---

## 8. 勾配ステップによる損失低下

$x=t$、$y=t_{\mathrm{new}}=t-\eta g$ とする。このとき

$$
y-x=-\eta g,
$$

$$
\nabla L(t)^{\mathsf T}(y-x)
=
g^{\mathsf T}(-\eta g)
=
-\eta\lVert g\rVert_2^2,
$$

$$
\lVert y-x\rVert_2^2
=
\eta^2\lVert g\rVert_2^2
$$

である。降下補題へ代入すると、

$$
\begin{aligned}
L(t_{\mathrm{new}})
&\le
L(t)
-
\eta\lVert g\rVert_2^2
+
\frac{\beta\eta^2}{2}
\lVert g\rVert_2^2
\\
&=
L(t)
-
\eta
\left(
1-\frac{\beta\eta}{2}
\right)
\lVert g\rVert_2^2
\end{aligned}
$$

である。

したがって、$g\neq0$ かつ

$$
0<\eta<\frac{2}{\beta}
$$

なら損失は低下する。さらに

$$
0<\eta\le\frac{1}{\beta}
$$

なら、

$$
1-\frac{\beta\eta}{2}
\ge
\frac12
$$

なので、

$$
L(t_{\mathrm{new}})
\le
L(t)
-
\frac{\eta}{2}
\lVert g\rVert_2^2
$$

を得る。

---

## 9. SVD打ち切りによる損失増加の上界

勾配更新後のテンソルをrank $r$ へ打ち切ったものを

$$
t_{\mathrm{trunc}}
=
t_{\mathrm{new}}+e
$$

とする。また、

$$
g_{\mathrm{new}}
=
\nabla L(t_{\mathrm{new}}),
\qquad
\delta_r
=
\lVert e\rVert_2
$$

とおく。

降下補題の上側に

$$
x=t_{\mathrm{new}},
\qquad
y=t_{\mathrm{trunc}}
$$

を入れると、

$$
L(t_{\mathrm{trunc}})
\le
L(t_{\mathrm{new}})
+
g_{\mathrm{new}}^{\mathsf T}e
+
\frac{\beta}{2}\lVert e\rVert_2^2
$$

である。Cauchy--Schwarz不等式から

$$
g_{\mathrm{new}}^{\mathsf T}e
\le
\left|g_{\mathrm{new}}^{\mathsf T}e\right|
\le
\lVert g_{\mathrm{new}}\rVert_2
\lVert e\rVert_2
$$

なので、

$$
L(t_{\mathrm{trunc}})
\le
L(t_{\mathrm{new}})
+
\lVert g_{\mathrm{new}}\rVert_2\delta_r
+
\frac{\beta}{2}\delta_r^2
$$

を得る。

### 9.1 $\delta_r$ と捨てた特異値

$T_{\mathrm{new}}$ を行列化したものを

$$
M_{\mathrm{new}}
=
\sum_j
\sigma_j
u_jv_j^{\mathsf T}
$$

とする。rank $r$ 打ち切りは

$$
M_r
=
\sum_{j\le r}
\sigma_j
u_jv_j^{\mathsf T}
$$

なので、

$$
M_{\mathrm{new}}-M_r
=
\sum_{j>r}
\sigma_j
u_jv_j^{\mathsf T}
$$

である。Frobeniusノルムの二乗を展開すると、

$$
\begin{aligned}
&\left\lVert
\sum_{j>r}
\sigma_j
u_jv_j^{\mathsf T}
\right\rVert_F^2
\\
&\quad=
\sum_{i>r}
\sum_{j>r}
\sigma_i\sigma_j
\operatorname{tr}
\left(
v_i u_i^{\mathsf T}u_jv_j^{\mathsf T}
\right)
\\
&\quad=
\sum_{i>r}
\sum_{j>r}
\sigma_i\sigma_j
(u_i^{\mathsf T}u_j)
(v_j^{\mathsf T}v_i)
\end{aligned}
$$

である。特異ベクトルの正規直交性

$$
u_i^{\mathsf T}u_j=\delta_{ij},
\qquad
v_j^{\mathsf T}v_i=\delta_{ij}
$$

により $i=j$ だけが残るので、

$$
\delta_r^2
=
\lVert M_{\mathrm{new}}-M_r\rVert_F^2
=
\sum_{j>r}\sigma_j^2
$$

である。

### 9.2 Eckart--Young--Mirsky最適性の範囲

Eckart--Young--Mirsky定理により、

$$
M_r
=
U_r\Sigma_rV_r^{\mathsf T}
$$

は、rank $r$ 以下の全行列の中で $M_{\mathrm{new}}$ に最も近いFrobenius近似である。

$$
\min_{\operatorname{rank}(B)\le r}
\lVert M_{\mathrm{new}}-B\rVert_F
=
\lVert M_{\mathrm{new}}-M_r\rVert_F
=
\left(
\sum_{j>r}\sigma_j^2
\right)^{1/2}
$$

ここで最小化しているのは局所行列とのFrobenius距離である。したがって、この定理だけから

$$
L(M_r)
\le
L(B)
$$

が全てのrank $r$ 候補 $B$ について成り立つとは言えない。分類損失の最小化、複数bondを同時に制約したTT全体の大域最適性、validation accuracyの最適性は別の問題である。

---

## 10. 勾配更新と打ち切りを合わせた十分条件

$$
0<\eta\le\frac{1}{\beta}
$$

とする。勾配のLipschitz性から

$$
\begin{aligned}
\lVert g_{\mathrm{new}}-g\rVert_2
&=
\lVert
\nabla L(t_{\mathrm{new}})-\nabla L(t)
\rVert_2
\\
&\le
\beta
\lVert t_{\mathrm{new}}-t\rVert_2
\\
&=
\beta\eta\lVert g\rVert_2
\end{aligned}
$$

である。三角不等式から

$$
\begin{aligned}
\lVert g_{\mathrm{new}}\rVert_2
&\le
\lVert g\rVert_2
+
\lVert g_{\mathrm{new}}-g\rVert_2
\\
&\le
(1+\beta\eta)
\lVert g\rVert_2
\\
&\le
2\lVert g\rVert_2
\end{aligned}
$$

を得る。

勾配ステップと打ち切りの上界を合わせると、

$$
L(t_{\mathrm{trunc}})
\le
L(t)
-
\frac{\eta}{2}\lVert g\rVert_2^2
+
2\lVert g\rVert_2\delta_r
+
\frac{\beta}{2}\delta_r^2
$$

である。

ここで、

$$
\delta_r
\le
\frac{\eta\lVert g\rVert_2}{8}
$$

とする。すると、

$$
2\lVert g\rVert_2\delta_r
\le
2\lVert g\rVert_2
\frac{\eta\lVert g\rVert_2}{8}
=
\frac{\eta}{4}
\lVert g\rVert_2^2
$$

である。また、

$$
\begin{aligned}
\frac{\beta}{2}\delta_r^2
&\le
\frac{\beta}{2}
\frac{\eta^2\lVert g\rVert_2^2}{64}
\\
&=
\frac{\beta\eta}{128}
\eta\lVert g\rVert_2^2
\\
&\le
\frac{\eta}{128}
\lVert g\rVert_2^2
\end{aligned}
$$

である。したがって、

$$
\begin{aligned}
L(t_{\mathrm{trunc}})
&\le
L(t)
+
\left(
-\frac12
+\frac14
+\frac1{128}
\right)
\eta\lVert g\rVert_2^2
\\
&=
L(t)
-
\frac{31}{128}
\eta\lVert g\rVert_2^2
\\
&\le
L(t)
-
\frac{\eta}{5}
\lVert g\rVert_2^2
\end{aligned}
$$

を得る。すなわち、

$$
\boxed{
\nabla L\text{ が }\beta\text{-Lipschitz},
\quad
0<\eta\le\frac1\beta,
\quad
\sqrt{\sum_{j>r}\sigma_j^2}
\le
\frac{\eta\lVert g\rVert_2}{8}
}
$$

なら、勾配更新とrank打ち切りを合わせた1回の局所更新後も、更新前より損失が下がる。

### 10.1 この条件が言っていること

- 勾配更新で得る利得は $\eta\lVert g\rVert_2^2$ のscaleを持つ。
- SVD打ち切りで失う量は、捨てた特異値のtail $\delta_r$ で抑えられる。
- tailが勾配stepのscale $\eta\lVert g\rVert_2$ に比べて十分小さければ、打ち切り後にも利得が残る。

この条件は十分条件であり、必要条件ではない。条件を満たさなくても損失が下がる場合はある。

### 10.2 十分条件を満たさなくても下がる数値例

$$
L(M)
=
\frac12\lVert M-X\rVert_F^2,
\qquad
X
=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix}
$$

とし、初期値を

$$
M_0
=
\begin{pmatrix}
1&2\\
2&4
\end{pmatrix}
$$

とする。差は

$$
M_0-X
=
\begin{pmatrix}
-2&2\\
2&3
\end{pmatrix}
$$

なので、

$$
L(M_0)
=
\frac12
\left(
(-2)^2+2^2+2^2+3^2
\right)
=
\frac12\cdot21
=
10.5
$$

である。この二次損失の勾配は

$$
\nabla L(M_0)=M_0-X
$$

である。$\eta=0.5$ とすると、

$$
\begin{aligned}
M_{\mathrm{new}}
&=
M_0-0.5(M_0-X)
\\
&=
\frac12M_0+\frac12X
\\
&=
\begin{pmatrix}
2&1\\
1&2.5
\end{pmatrix}
\end{aligned}
$$

となる。差は

$$
M_{\mathrm{new}}-X
=
\begin{pmatrix}
-1&1\\
1&1.5
\end{pmatrix}
$$

なので、

$$
L(M_{\mathrm{new}})
=
\frac12
\left(
1+1+1+2.25
\right)
=
2.625
$$

まで下がる。

$M_{\mathrm{new}}$ をrank 1へ打ち切ると、この例では打ち切り誤差が約

$$
\delta_1
\approx
1.22
$$

となり、打ち切り後の損失は約

$$
4.62
$$

になる。打ち切り前の勾配は

$$
g_{\mathrm{new}}
=
M_{\mathrm{new}}-X
=
\begin{pmatrix}
-1&1\\
1&1.5
\end{pmatrix}
$$

なので、

$$
\lVert g_{\mathrm{new}}\rVert_F
=
\sqrt{1+1+1+2.25}
=
\sqrt{5.25}
\approx
2.29
$$

である。$\beta=1$ の打ち切り上界は

$$
\begin{aligned}
L(M_r)
&\le
L(M_{\mathrm{new}})
+
\lVert g_{\mathrm{new}}\rVert_F\delta_1
+
\frac12\delta_1^2
\\
&\approx
2.625
+
2.29\cdot1.22
+
\frac12\cdot1.22^2
\\
&\approx
6.16
\end{aligned}
$$

となり、実際の約4.62を上から抑える。

一方、勾配step scaleは

$$
s
=
\eta\lVert\nabla L(M_0)\rVert_F
=
0.5\sqrt{21}
\approx
2.29
$$

なので、十分条件の閾値は

$$
\frac{s}{8}
\approx
0.286
$$

である。$\delta_1\approx1.22$ はこの閾値を超えているため、十分条件は成立しない。それでも

$$
4.62<10.5
$$

なので、打ち切り後の損失は更新前より低い。

### 10.3 実用上の限界

NNでこの不等式を保証として使うには、少なくとも次が必要である。

- 対象の損失が考えている領域で $\beta$-smoothであること。
- $\beta$ の妥当な上界があること。
- 学習率が $\eta\le1/\beta$ を満たすこと。
- 同じ固定目的関数について更新前後を比較すること。

ReLUを含むネットワークでは全域smoothnessを一般に主張できず、$\beta$ も通常は未知である。そのため、実装では

$$
s=\eta\lVert g\rVert_F,
\qquad
\delta_r,
\qquad
\frac{\delta_r}{s}
$$

を**診断値**として記録する。$\delta_r\le s/8$ だけを確認して「理論保証が成立した」と表示してはいけない。

---

## 11. 正準形が局所誤差と重み誤差を一致させる

pair外のcoreを固定すると、結合テンソルから重み全体への写像は線形である。vectorizeした結合テンソルを $t$、重みを $w$ として

$$
w=At
$$

と書く。

混合正準形でpairの左側が左直交、右側が右直交なら、pair外のbasisは正規直交になる。このとき

$$
A^{\mathsf T}A=I
$$

であり、任意の差 $e$ について

$$
\lVert Ae\rVert_2^2
=
e^{\mathsf T}A^{\mathsf T}Ae
=
e^{\mathsf T}e
=
\lVert e\rVert_2^2
$$

となる。したがって、

$$
\lVert W(T)-W(\widetilde T)\rVert_F
=
\lVert T-widetilde T\rVert_F
$$

である。

一方、logitsへの写像は入力batchと後段の層に依存する。

$$
T
\longmapsto
W(T)
\longmapsto
f(X;T)
$$

なので、局所Frobenius誤差と重み誤差が一致しても、logits誤差や分類損失の変化が同じ値になるわけではない。

### 11.1 smoothness定数の移し替え

重み空間の損失を $\ell(w)$ とし、

$$
L(t)=\ell(At)
$$

とする。chain ruleから

$$
\nabla_tL(t)
=
A^{\mathsf T}\nabla\ell(At)
$$

である。$\nabla\ell$ が $\beta_W$-Lipschitzなら、

$$
\begin{aligned}
\lVert\nabla_tL(t)-\nabla_tL(t')\rVert_2
&=
\left\lVert
A^{\mathsf T}
\left(
\nabla\ell(At)-\nabla\ell(At')
\right)
\right\rVert_2
\\
&\le
\lVert A\rVert_2
\beta_W
\lVert A(t-t')\rVert_2
\\
&\le
\lVert A\rVert_2^2
\beta_W
\lVert t-t'\rVert_2
\end{aligned}
$$

なので、

$$
\beta_T
\le
\lVert A\rVert_2^2\beta_W
$$

である。$A^{\mathsf T}A=I$ なら $\lVert A\rVert_2=1$ なので、

$$
\beta_T\le\beta_W
$$

となる。

---

## 12. 方向付きSVDと直交中心の移動

更新後の行列を

$$
M_r
=
U_r\Sigma_rV_r^{\mathsf T}
$$

とする。

### 12.1 左から右へ進む場合

$$
G_k'
=
\operatorname{reshape}(U_r),
$$

$$
G_{k+1}'
=
\operatorname{reshape}(\Sigma_rV_r^{\mathsf T})
$$

とする。$U_r$ の列が正規直交なので、左coreは左直交になる。特異値を右coreへ含めるため、直交中心は右siteへ移る。

### 12.2 右から左へ進む場合

$$
G_k'
=
\operatorname{reshape}(U_r\Sigma_r),
$$

$$
G_{k+1}'
=
\operatorname{reshape}(V_r^{\mathsf T})
$$

とする。$V_r^{\mathsf T}$ の行が正規直交なので、右coreは右直交になる。特異値を左coreへ含めるため、直交中心は左siteへ移る。

NN重みでは、物理状態のようにSVD後のcoreを単位ノルムへ再正規化しない。再正規化すれば重みのscaleを変えてしまうからである。

---

## 13. forwardは局所更新のどこで必要か

NNの2-core局所更新では、forwardを役割別に分ける。

1. 結合前後の等価性確認用forward。
2. 結合テンソル $T$ から分類損失を作り、$\nabla_TL$ を得る学習用forward。
3. 勾配更新後の $T_{\mathrm{new}}$ の損失を測る記録用forward。
4. SVD・rank制限後のcore列で切り捨ての影響を測る記録用forward。
5. sweep終了後に通常モデルで学習・validation指標を測る評価用forward。

本体は2である。$T$ が計算graphへ入っていなければ、

$$
\nabla_TL
$$

は得られない。1、3、4は動作と損失変化を分解して理解するための診断であり、全てを本学習の毎stepへ入れる必要はない。

---

## 14. 2種類の「A/B」を混同しない

議論では、異なる2軸にA/Bという仮称が使われた。技術ノートでは次のように分ける。

### 14.1 更新変数の選び方

#### 局所テンソル更新方式

$$
T
\longrightarrow
T-\eta\nabla_TL
\longrightarrow
\text{SVD再分割}
$$

とする。denseな重み全体を作らず、結合テンソルを直接分類損失で更新する。Notebook 04と、計画中のFashion-MNIST Notebook 00はこの方式を使う。

#### 重み全体更新・再射影方式

$$
W^+
=
W-\eta\nabla_WL
$$

を先に作り、その固定目標 $W^+$ をTTへ近似する。

$$
\min_{G_1,\ldots,G_d}
\lVert W^+-\operatorname{TT}(G_1,\ldots,G_d)\rVert_F^2
$$

をsweepで解く設計である。素朴にはdenseな $W^+$ を必要とし、TT圧縮のメモリ目的と衝突する。TT多様体の接空間射影とretractionを使えばdense化を避ける別設計も可能だが、本プロジェクトの最初の実装範囲には含めない。

### 14.2 データの使い方

#### minibatchごとのsweep

1つのminibatchを固定し、そのbatchの損失で複数sweepしてから次のbatchへ進む。batchごとに目的関数が切り替わるため、現在batchへの適合、過去batchへの干渉、validationの変化を分けて測る必要がある。

#### 全データ勾配によるsweep

各pairで全学習データの平均損失の勾配を計算し、その勾配で1回更新してから次のpairへ進む。全pair・全sweepで同じ経験損失を目的にできるが、計算量は大きい。

Fashion-MNIST Notebook 00の最初の基準設計は、**局所テンソル更新方式と、全データ勾配によるsweepの組合せ**である。

---

## 15. sweep全体の停止を「数学的収束」と呼ばない

物理DMRGでは、固定environmentの局所固有値問題を解く構造がある。一方、TT-MLPでは次が加わる。

- ReLUを含む非凸損失。
- 勾配1ステップであり、局所問題を厳密に解かない。
- rank制限SVDは分類損失の最小化ではない。
- 通常のdense parameterも同時更新する場合、更新blockが互いに重なる。

したがって、指定sweep数の完了やvalidation early stoppingを、そのまま数学的収束の証明とは呼ばない。停止理由を

```text
early_stopping
max_sweeps
non_finite
明示的な数値計算失敗
```

のように分け、何が起きたかを記録する。

---

## 16. 文献との関係

隣接テンソルを結合し、局所損失の勾配で結合テンソルを更新し、SVD・打ち切り後に左右sweepする手順は、次の一次資料にも現れる。

- [Jahromi and Orús, Variational tensor neural networks for deep learning](https://www.nature.com/articles/s41598-024-69366-8)
- [Stoudenmire and Schwab, Supervised Learning with Quantum-Inspired Tensor Networks](https://arxiv.org/abs/1605.05775)

ただし、前者のTNN/MPO構成や後者の特徴写像MPS分類器は、本プロジェクトのReLUを含むTT-MLPと同一ではない。論文の存在は、局所結合・勾配更新・SVD再分割・sweepという一般構造の裏づけであり、このFashion-MNIST設計の性能や最適性を保証しない。

勾配step後にTT-rank制約へ戻すretractionという別系統の考え方は、例えば次のTT多様体最適化で扱われる。

- [Cai, Li, and Xia, Provable Tensor-Train Format Tensor Completion by Riemannian Optimization](https://jmlr.org/papers/v23/21-1138.html)

これもtensor completionの理論であり、今回のTT-MLP学習へ収束保証を移せるという意味ではない。

---

## 17. このノートから実装へ進むときの確認事項

1. 結合前coreと結合テンソルの軸順が一致している。
2. 結合した $T$ が独立したleafであり、損失から $T$ までgraphがつながっている。
3. 切り捨てなしSVDで $T_{\mathrm{new}}$ と出力を再現する。
4. rank制限SVDの誤差を、残差と捨てた特異値の二乗和の両方で確認する。
5. 局所誤差と重み誤差の一致を主張するときは、pair外の正準条件を確認する。
6. 勾配更新後の損失と、SVD打ち切り後の損失を分けて記録する。
7. 損失低下、勾配非零、accuracy向上を一般的なassertにしない。
8. 方向に応じて特異値の吸収先を変え、直交中心を次のpairへ運ぶ。
9. 数値異常の候補をモデルへ部分反映しない。
10. 物理DMRG、局所勾配sweep、重み全体更新後のTT再射影を別の方式として記述する。

## 関連

- [[50_DMRG/02_中心テンソルの線形写像と2-site局所更新]]
- [[50_DMRG/04_environmentを用いた2-site_DMRG_1往復]]
- [[50_DMRG/05_複数往復_停止判定_bond上限比較]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/35_TT_MPSの等長写像と射影]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/46_Eckart_Young_Mirsky定理とTT_MPS単一ボンド打ち切りの最適性]]
- [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
- [[notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb|04_tt_mlp_two_core_local_update.ipynb]]
