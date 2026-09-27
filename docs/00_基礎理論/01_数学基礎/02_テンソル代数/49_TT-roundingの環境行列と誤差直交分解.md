---
title: TT-roundingの環境行列と誤差直交分解
tags:
  - TT
  - MPS
  - TT-rounding
  - environment
  - canonical-form
  - gauge
  - Frobenius
  - orthogonality
  - Kronecker
---

# TT-roundingの環境行列と誤差直交分解

このノートでは、TT-roundingで局所SVDの尾部特異値がTT全体のFrobenius誤差へ接続される理由を導出する。

出発点は [[48_TT-roundingの定義と正準化sweep]] である。単一行列の最良低rank近似は [[46_Eckart_Young_Mirsky定理とTT_MPS単一ボンド打ち切りの最適性|Eckart–Young–Mirsky定理]]、mixed-canonical formとSchmidt係数は [[43_TT_MPSのSVD中心移動とSchmidt形]] を参照する。

このノートの中心は、局所残差 $E_k$ と全体誤差 $\Delta\mathcal X_k$ の間に左右環境が入ること、正準化がその環境を等長写像にすること、異なるSVD段階の実誤差が直交することである。

---

## 0. 結論

非正準なTTコアを局所的にSVDしても、一般には

$$
\|\Delta\mathcal X_k\|_F
\ne
\|E_k\|_F.
$$

局所誤差は左右環境を通って全体へ埋め込まれる。

$$
\Delta X_{\langle k-1\rangle}
=
L_{k-1}
E_k^T
\left(
I_{n_k}\otimes R_k
\right).
$$

一方、標準TT-roundingの各SVD段階では、

$$
L_{k-1}^TL_{k-1}=I,
\qquad
R_kR_k^T=I
$$

となる。そのため左右環境はFrobenius内積とノルムを保存し、実際に捨てた局所誤差を $e_k$ と書けば、exact arithmeticでは

$$
\boxed{
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\sum_{k=1}^{d-1}e_k^2
}
$$

となる。

---

## 1. 局所EYMとTT全体の誤差は別の主張

第 $k$ コアを右向きに行列化する。

$$
H_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1},n_kr_k)
\right).
$$

正準化せず、比較のためにその転置

$$
A_k
=
H_k^T
\in
\mathbb R^{(n_kr_k)\times r_{k-1}}
$$

を直接rank $\ell_k$ へ打ち切る仮想的な操作を考える。

$$
A_k
=
U_k\Sigma_kV_k^T,
$$

$$
A_k^{(\ell_k)}
=
U_{k,\mathrm{keep}}
\Sigma_{k,\mathrm{keep}}
V_{k,\mathrm{keep}}^T.
$$

局所残差を

$$
E_k
=
A_k-A_k^{(\ell_k)}
$$

とすると、EYM定理から

$$
\boxed{
\|E_k\|_F^2
=
\sum_{j>\ell_k}\sigma_{k,j}^2
}
$$

である。

これはあくまで局所行列 $A_k$ の誤差である。正準化していないTTでは、まだ

$$
\|\Delta\mathcal X_k\|_F
=
\|E_k\|_F
$$

とは言えない。

---

## 2. 左右環境の定義

実数TTを

$$
\begin{aligned}
\mathcal X(i_1,\ldots,i_d)
={}&
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}(1,i_1,\alpha_1)
\cdots
G^{(d)}(\alpha_{d-1},i_d,1)
\end{aligned}
$$

とする。

### 2.1 左環境

第1コアから第 $k-1$ コアまでを縮約し、$\alpha_{k-1}$ を列に残す。

$$
L_{k-1}
\in
\mathbb R^{(n_1\cdots n_{k-1})\times r_{k-1}}.
$$

成分は

$$
\begin{aligned}
&L_{k-1}
\bigl(
(i_1,\ldots,i_{k-1}),
\alpha_{k-1}
\bigr)\\
&=
\sum_{\alpha_1,\ldots,\alpha_{k-2}}
G^{(1)}(1,i_1,\alpha_1)
\cdots
G^{(k-1)}
(\alpha_{k-2},i_{k-1},\alpha_{k-1}).
\end{aligned}
$$

左側にコアが存在しない境界は

$$
L_0
=
\begin{pmatrix}
1
\end{pmatrix}
\in
\mathbb R^{1\times1}.
$$

### 2.2 右環境

第 $k+1$ コアから第 $d$ コアまでを縮約し、$\alpha_k$ を行に残す。

$$
R_k
\in
\mathbb R^{r_k\times(n_{k+1}\cdots n_d)}.
$$

成分は

$$
\begin{aligned}
&R_k
\bigl(
\alpha_k,
(i_{k+1},\ldots,i_d)
\bigr)\\
&=
\sum_{\alpha_{k+1},\ldots,\alpha_{d-1}}
G^{(k+1)}
(\alpha_k,i_{k+1},\alpha_{k+1})
\cdots
G^{(d)}
(\alpha_{d-1},i_d,1).
\end{aligned}
$$

右側にコアが存在しない境界は

$$
R_d
=
\begin{pmatrix}
1
\end{pmatrix}
\in
\mathbb R^{1\times1}.
$$

これは空の積を乗法単位元1とすること、および境界rank $r_d=1$ に対応する。

---

## 3. 局所残差から全体誤差を作る

元コアと近似コアの差を

$$
\Delta G^{(k)}
=
G^{(k)}-\widetilde G^{(k)}
$$

とする。右向き行列化との対応は

$$
E_k^T
=
\operatorname{reshape}
\left(
\Delta G^{(k)},
(r_{k-1},n_kr_k)
\right),
$$

$$
E_k^T
\bigl(
\alpha_{k-1},(i_k,\alpha_k)
\bigr)
=
\Delta G^{(k)}
(\alpha_{k-1},i_k,\alpha_k)
$$

である。

第 $k$ コアだけを $\Delta G^{(k)}$ に置き換えた全体差は

$$
\begin{aligned}
&\Delta\mathcal X_k(i_1,\ldots,i_d)\\
&=
\sum_{\alpha_{k-1}=1}^{r_{k-1}}
\sum_{\alpha_k=1}^{r_k}
L_{k-1}
\bigl((i_1,\ldots,i_{k-1}),\alpha_{k-1}\bigr)\\
&\quad\times
\Delta G^{(k)}
(\alpha_{k-1},i_k,\alpha_k)
R_k
\bigl(\alpha_k,(i_{k+1},\ldots,i_d)\bigr).
\end{aligned}
$$

これを、行を $(i_1,\ldots,i_{k-1})$、列を $(i_k,\ldots,i_d)$ とするunfoldingへ並べる。

$$
\Delta X_{\langle k-1\rangle}
\in
\mathbb R^{(n_1\cdots n_{k-1})\times(n_k\cdots n_d)}.
$$

---

## 4. なぜ $I_{n_k}\otimes R_k$ が現れるか

$E_k^T$ の列添字は複合添字

$$
(i_k,\alpha_k)
$$

である。$R_k$ はbond添字を

$$
\alpha_k
\longmapsto
(i_{k+1},\ldots,i_d)
$$

へ写すが、$i_k$ は含まない。必要な写像は

$$
(i_k,\alpha_k)
\longmapsto
(i_k,i_{k+1},\ldots,i_d)
$$

である。そのため、$i_k$ を恒等写像で保存し、各 $i_k$ ごとに同じ $R_k$ を作用させる。

$$
S_k
=
I_{n_k}\otimes R_k.
$$

成分は

$$
\boxed{
S_k
\bigl(
(i_k,\alpha_k),(j_k,q_R)
\bigr)
=
\delta_{i_k,j_k}
R_k(\alpha_k,q_R)
}
$$

である。ここで

$$
q_R
=
(i_{k+1},\ldots,i_d).
$$

### 4.1 $n_k=2$ のブロック表示

$$
I_2\otimes R_k
=
\begin{pmatrix}
R_k&0\\
0&R_k
\end{pmatrix}.
$$

$i_k=1$ と $i_k=2$ の成分は混ざらず、それぞれに同じ右環境が作用する。

### 4.2 全要素を表示する小さい例

$$
R_k
=
\begin{pmatrix}
a&b&c\\
d&e&f
\end{pmatrix}
$$

とし、$n_k=2$ とする。このとき、

$$
I_2\otimes R_k
=
\begin{pmatrix}
a&b&c&0&0&0\\
d&e&f&0&0&0\\
0&0&0&a&b&c\\
0&0&0&d&e&f
\end{pmatrix}.
$$

行順序を

$$
(i_k,\alpha_k)
=
(1,1),(1,2),(2,1),(2,2)
$$

とし、列順序を

$$
(j_k,q_R)
=
(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)
$$

とすれば、左上ブロックが $i_k=j_k=1$、右下ブロックが $i_k=j_k=2$ に対応する。非対角ブロックがゼロなので、異なる物理添字は混ざらない。

---

## 5. 全体誤差unfoldingの行列式

以上を使うと、

$$
\boxed{
\Delta X_{\langle k-1\rangle}
=
L_{k-1}
E_k^T
\left(
I_{n_k}\otimes R_k
\right)
}
\tag{1}
$$

となる。

shapeは

$$
L_{k-1}
:
(n_1\cdots n_{k-1})\times r_{k-1},
$$

$$
E_k^T
:
r_{k-1}\times(n_kr_k),
$$

$$
I_{n_k}\otimes R_k
:
(n_kr_k)\times(n_k\cdots n_d).
$$

右辺の成分を展開すると、

$$
\begin{aligned}
&\left[
L_{k-1}E_k^T
(I_{n_k}\otimes R_k)
\right]
\bigl(p,(j_k,q_R)\bigr)\\
&=
\sum_{\alpha_{k-1},i_k,\alpha_k}
L_{k-1}(p,\alpha_{k-1})
E_k^T
\bigl(\alpha_{k-1},(i_k,\alpha_k)\bigr)\\
&\quad\times
\delta_{i_k,j_k}
R_k(\alpha_k,q_R)\\
&=
\sum_{\alpha_{k-1},\alpha_k}
L_{k-1}(p,\alpha_{k-1})
\Delta G^{(k)}
(\alpha_{k-1},j_k,\alpha_k)
R_k(\alpha_k,q_R).
\end{aligned}
$$

これは第 $k$ コアだけを誤差コアへ置き換えたTT縮約そのものである。

式 (1) は、左辺をdenseに計算してから右辺へ変形する手順ではない。右辺の環境行列積を書き下すと、左辺で定義した誤差テンソルのunfoldingになる、という恒等式である。

---

## 6. 正準化なしの誤差上界

unfoldingは要素の並べ替えなのでFrobeniusノルムを保つ。

$$
\|\Delta\mathcal X_k\|_F
=
\|\Delta X_{\langle k-1\rangle}\|_F.
$$

式 (1) と

$$
\|ABC\|_F
\le
\|A\|_2\|B\|_F\|C\|_2
$$

を使うと、

$$
\begin{aligned}
\|\Delta\mathcal X_k\|_F
&\le
\|L_{k-1}\|_2
\|E_k^T\|_F
\|I_{n_k}\otimes R_k\|_2\\
&=
\|L_{k-1}\|_2
\|E_k\|_F
\|R_k\|_2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|\Delta\mathcal X_k\|_F
\le
\|L_{k-1}\|_2
\|R_k\|_2
\left(
\sum_{j>\ell_k}\sigma_{k,j}^2
\right)^{1/2}
}
$$

である。

例えば

$$
\|E_k\|_F=10^{-6},
\qquad
\|L_{k-1}\|_2=10^3,
\qquad
\|R_k\|_2=10^2
$$

なら、

$$
\|\Delta\mathcal X_k\|_F
\le
10^3\cdot10^{-6}\cdot10^2
=
10^{-1}.
$$

これは全体誤差が必ず $0.1$ になるという意味ではない。局所残差だけから全体誤差を $10^{-6}$ と保証できない、という意味である。

したがって、「正準化せずに局所SVDすると必ず悪い近似になる」という結論でもない。たまたま環境が局所残差を増幅しない場合や、近似後に全体誤差を直接評価する場合には、非正準形での処理が十分に機能することもある。問題は、非正準形の特異値尾部だけを見てTT全体の誤差だと断定できないことである。

---

## 7. Gram行列を使った厳密な式

左右環境のGram行列を

$$
M_L
=
L_{k-1}^TL_{k-1},
\qquad
M_R
=
R_kR_k^T
$$

とする。式 (1) から、

$$
\begin{aligned}
\|\Delta\mathcal X_k\|_F^2
&=
\operatorname{tr}
\left[
E_kM_LE_k^T
\left(
I_{n_k}\otimes M_R
\right)
\right].
\end{aligned}
$$

局所Frobeniusノルム

$$
\|E_k\|_F^2
=
\operatorname{tr}(E_kE_k^T)
$$

とは一般に異なる。非正準のまま処理するなら、環境込みのweighted norm、whitening、一般化固有値問題、または近似後の全体誤差再評価が必要になる。標準TT-roundingは、正準化によって $M_L,M_R$ を単位行列へ帰着させる。

### 7.1 小さい尺度例

局所誤差を

$$
\|E_k\|_F
=
10^{-3}
$$

とする。

正準環境なら

$$
\|\Delta\mathcal X_k\|_F
=
10^{-3}.
$$

一方、

$$
\|L_{k-1}\|_2=10,
\qquad
\|R_k\|_2=5
$$

なら上界は

$$
5\times10^{-2},
$$

$$
\|L_{k-1}\|_2=0.1,
\qquad
\|R_k\|_2=0.2
$$

なら上界は

$$
2\times10^{-5}
$$

である。後二つは上界であり、等号を主張する例ではない。

---

## 8. gauge自由度により局所特異値は変わる

第 $k-1$ と第 $k$ コアの間に任意の可逆行列

$$
M
\in
\mathbb R^{r_{k-1}\times r_{k-1}}
$$

とその逆行列を挿入する。

$$
\widehat G^{(k-1)}
=
G^{(k-1)}M,
$$

$$
\widehat G^{(k)}
=
M^{-1}G^{(k)}.
$$

bond添字を縮約すると、

$$
\sum_\beta
M(\alpha,\beta)
M^{-1}(\beta,\gamma)
=
\delta_{\alpha,\gamma}
$$

なので、全体テンソルは変わらない。

$$
\widehat{\mathcal X}
=
\mathcal X.
$$

しかし左環境と局所行列は

$$
\widehat L_{k-1}
=
L_{k-1}M,
$$

$$
\widehat H_k
=
M^{-1}H_k,
$$

$$
\widehat A_k
=
A_kM^{-T}
$$

と変化する。一般の非直交な $M$ に対して、

$$
\sigma_j(A_kM^{-T})
\ne
\sigma_j(A_k)
$$

である。

### 8.1 スカラーgauge

$$
M
=
cI,
\qquad
c\ne0
$$

なら、

$$
\widehat L_{k-1}
=
cL_{k-1},
\qquad
\widehat A_k
=
\frac1cA_k.
$$

局所特異値と局所残差は

$$
\widehat\sigma_j
=
\frac1{|c|}\sigma_j,
$$

$$
\|\widehat E_k\|_F
=
\frac1{|c|}\|E_k\|_F
$$

となる。しかし全体誤差では

$$
(cL_{k-1})
\left(
\frac1cE_k^T
\right)
=
L_{k-1}E_k^T
$$

と打ち消される。

例えば、局所特異値が

$$
(10,1,0.01)
$$

のとき、$M=100I$ を入れると

$$
(0.1,0.01,0.0001)
$$

へ変わる。同じテンソルを表しているにもかかわらず、rank 2の局所尾部は $0.01$ から $0.0001$ へ変わる。したがって、非正準な1コアだけの特異値は、テンソル固有の重要度や全体誤差ではない。

---

## 9. canonical formで不変になる特異値

bond $k$ でmixed-canonical formを作り、

$$
X_{\langle k\rangle}
=
L_kC_kR_k,
$$

$$
L_k^TL_k
=
I,
\qquad
R_kR_k^T
=
I
$$

とする。このとき中央行列 $C_k$ の特異値はunfolding $X_{\langle k\rangle}$ の特異値、すなわちSchmidt係数と一致する。

### 9.1 canonical条件を保つgauge

$$
\widehat L_k
=
L_kM
$$

もleft-canonicalであるためには、

$$
\begin{aligned}
\widehat L_k^T\widehat L_k
&=
M^TL_k^TL_kM\\
&=
M^TM\\
&=
I
\end{aligned}
$$

が必要である。したがって、実数では

$$
M
\in
O(r_k)
$$

である。非正準形で可能だった $M=cI$ は、canonical form内では

$$
c=\pm1
$$

に限られる。

### 9.2 中央特異値の不変性

直交gauge $Q,P$ により

$$
\widehat C_k
=
Q^TC_kP,
$$

$$
Q^TQ
=
I,
\qquad
P^TP
=
I
$$

とする。このとき、

$$
\begin{aligned}
\widehat C_k^T\widehat C_k
&=
P^TC_k^TQQ^TC_kP\\
&=
P^TC_k^TC_kP.
\end{aligned}
$$

$\widehat C_k^T\widehat C_k$ と $C_k^TC_k$ は直交相似なので固有値が等しく、

$$
\boxed{
\sigma_j(Q^TC_kP)
=
\sigma_j(C_k)
}
$$

である。

不変なのは、mixed-canonical formの中央行列の特異値である。任意の1コアを任意の向きに行列化した特異値がすべて不変、という主張ではない。

right-canonical化だけを終えた時点では左側はまだ一般に非正準である。その後の左から右へのSVDで左側も順にleft-canonicalとなり、処理中のcutがmixed-canonicalになる。

---

## 10. right-canonicalな右環境は等長

第 $t$ コアの物理添字スライスを

$$
G^{(t)}[i_t]
\in
\mathbb R^{r_{t-1}\times r_t}
$$

とする。right-canonical条件は

$$
\boxed{
\sum_{i_t=1}^{n_t}
G^{(t)}[i_t]
G^{(t)}[i_t]^T
=
I_{r_{t-1}}
}
\tag{2}
$$

である。

第2因子が転置のどの要素に対応するかまで書けば、

$$
\begin{aligned}
&\left[
G^{(t)}[i_t]
G^{(t)}[i_t]^T
\right]_{\alpha_{t-1},\beta_{t-1}}\\
&=
\sum_{\alpha_t=1}^{r_t}
\underbrace{
G^{(t)}
(\alpha_{t-1},i_t,\alpha_t)
}_{[G^{(t)}[i_t]]_{\alpha_{t-1},\alpha_t}}
\underbrace{
G^{(t)}
(\beta_{t-1},i_t,\alpha_t)
}_{[G^{(t)}[i_t]^T]_{\alpha_t,\beta_{t-1}}}.
\end{aligned}
$$

### 10.1 右環境の再帰

$$
\begin{aligned}
&R_{t-1}
\bigl(
\alpha_{t-1},(i_t,\ldots,i_d)
\bigr)\\
&=
\sum_{\alpha_t=1}^{r_t}
G^{(t)}
(\alpha_{t-1},i_t,\alpha_t)
R_t
\bigl(
\alpha_t,(i_{t+1},\ldots,i_d)
\bigr).
\end{aligned}
$$

### 10.2 逆向きの帰納法

命題を

$$
P(t):
\quad
R_tR_t^T
=
I_{r_t}
$$

とする。右端では

$$
R_d
=
\begin{pmatrix}1\end{pmatrix},
\qquad
r_d=1,
$$

$$
R_dR_d^T
=
\begin{pmatrix}1\end{pmatrix}
\begin{pmatrix}1\end{pmatrix}^T
=
\begin{pmatrix}1\end{pmatrix}
=
I_{r_d}.
$$

これが基底 $P(d)$ である。

$P(t)$ を仮定し、$R_{t-1}R_{t-1}^T$ の成分を展開する。

$$
\begin{aligned}
&[R_{t-1}R_{t-1}^T]
(\alpha_{t-1},\beta_{t-1})\\
&=
\sum_{i_t,\ldots,i_d}
\sum_{\alpha_t,\beta_t}
G^{(t)}
(\alpha_{t-1},i_t,\alpha_t)
G^{(t)}
(\beta_{t-1},i_t,\beta_t)\\
&\quad\times
R_t
(\alpha_t,i_{t+1},\ldots,i_d)
R_t
(\beta_t,i_{t+1},\ldots,i_d).
\end{aligned}
$$

右側物理添字について先に和を取ると、帰納法の仮定より

$$
\sum_{i_{t+1},\ldots,i_d}
R_t(\alpha_t,\cdots)
R_t(\beta_t,\cdots)
=
\delta_{\alpha_t,\beta_t}.
$$

したがって、

$$
\begin{aligned}
&[R_{t-1}R_{t-1}^T]
(\alpha_{t-1},\beta_{t-1})\\
&=
\sum_{i_t=1}^{n_t}
\sum_{\alpha_t=1}^{r_t}
G^{(t)}
(\alpha_{t-1},i_t,\alpha_t)
G^{(t)}
(\beta_{t-1},i_t,\alpha_t)\\
&=
\delta_{\alpha_{t-1},\beta_{t-1}}
\end{aligned}
$$

となる。最後の等号は式 (2) である。よって、

$$
P(t)
\Longrightarrow
P(t-1).
$$

これは $t+1$ へ進む通常表記ではなく、右端から左へindexを減らす逆向きの帰納法である。$s=d-t$ と置けば通常の $s\to s+1$ の形へ書き直せる。

以上から、$G^{(k+1)},\ldots,G^{(d)}$ がright-canonicalなら、

$$
\boxed{
R_kR_k^T
=
I_{r_k}
}
$$

である。

### 10.3 なぜ第1コアだけで全体ノルムを測れるか

右から左へのQR sweepが終わると、第2コアから第 $d$ コアまでがright-canonicalになる。第2コア以降を縮約した右ブロックを

$$
B_1
\in
\mathbb R^{r_1\times(n_2\cdots n_d)}
$$

とする。上の帰納法から、その行は正規直交である。

$$
B_1B_1^T
=
I_{r_1}.
$$

第1コアを

$$
A_1
=
\operatorname{reshape}
\left(
G^{(1)},
(n_1,r_1)
\right)
$$

とすると、第1cutのunfoldingは

$$
X_{\langle1\rangle}
=
A_1B_1.
$$

unfoldingは要素の並べ替えなのでFrobeniusノルムを保つ。したがって、

$$
\begin{aligned}
\|\mathcal X\|_F^2
&=
\|X_{\langle1\rangle}\|_F^2\\
&=
\operatorname{tr}
\left(
A_1B_1B_1^TA_1^T
\right)\\
&=
\operatorname{tr}
\left(
A_1I_{r_1}A_1^T
\right)\\
&=
\operatorname{tr}
\left(
A_1A_1^T
\right)\\
&=
\|A_1\|_F^2\\
&=
\|G^{(1)}\|_F^2.
\end{aligned}
$$

よって、

$$
\boxed{
\|\mathcal X\|_F
=
\|G^{(1)}\|_F
}
$$

である。denseテンソルを復元せず、第1コアの全要素二乗和だけから全体誤差予算の基準 $N$ を得られる。

---

## 11. 拡張右環境も等長

$$
S_k
=
I_{n_k}\otimes R_k
$$

とする。Kronecker積の積の性質から、

$$
\begin{aligned}
S_kS_k^T
&=
(I_{n_k}\otimes R_k)
(I_{n_k}\otimes R_k)^T\\
&=
(I_{n_k}I_{n_k})
\otimes
(R_kR_k^T)\\
&=
I_{n_k}\otimes I_{r_k}\\
&=
I_{n_kr_k}.
\end{aligned}
$$

$S_k$ は一般に横長であり、ここで必要なのは

$$
S_kS_k^T=I
$$

であって、$S_k^TS_k=I$ ではない。

任意の行ベクトル $x^T$ に対して、

$$
\begin{aligned}
\|x^TS_k\|_2^2
&=
x^TS_kS_k^Tx\\
&=
x^Tx\\
&=
\|x^T\|_2^2.
\end{aligned}
$$

これを $E_k^T$ の各行へ適用すれば、

$$
\boxed{
\left\|
E_k^T
(I_{n_k}\otimes R_k)
\right\|_F
=
\|E_k\|_F
}
$$

となる。

---

## 12. 左から右へのSVDが左環境を等長にする

標準TT-roundingのSVD段階では、

$$
V_k
=
U_k\Sigma_kW_k^T
$$

から保持した $U_{k,\mathrm{keep}}$ を新しい第 $k$ コアへreshapeする。

$$
U_{k,\mathrm{keep}}^T
U_{k,\mathrm{keep}}
=
I.
$$

したがって新コアはleft-canonicalである。

左境界は

$$
L_0
=
\begin{pmatrix}1\end{pmatrix},
\qquad
L_0^TL_0=I_1.
$$

第1段階で第1コアがleft-canonicalになれば、

$$
L_1^TL_1=I.
$$

第2段階で第2コアもleft-canonicalになれば、

$$
L_2^TL_2=I.
$$

同じ帰納を続けると、第 $k$ SVDの直前には

$$
\boxed{
L_{k-1}^TL_{k-1}
=
I_{r_{k-1}}
}
$$

が成り立つ。

よって、その時点の局所残差について左右両側が等長となり、

$$
\begin{aligned}
\|\Delta\mathcal X_k\|_F
&=
\left\|
L_{k-1}E_k^T
(I_{n_k}\otimes R_k)
\right\|_F\\
&=
\left\|
E_k^T
(I_{n_k}\otimes R_k)
\right\|_F\\
&=
\|E_k\|_F.
\end{aligned}
$$

---

## 13. 標準SVD向きでの同じ式

実際の左から右へのSVDでは、局所行列は

$$
V_k
\in
\mathbb R^{(r_{k-1}n_k)\times r_k}
$$

である。実局所残差を

$$
F_k
=
V_k-V_{k,\mathrm{keep}}
$$

と書けば、cut $k$ の全体誤差unfoldingは

$$
\boxed{
\Delta X_{\langle k\rangle}
=
\left(
L_{k-1}\otimes I_{n_k}
\right)
F_kR_k
}
\tag{3}
$$

と書ける。

$$
L_{k-1}\otimes I_{n_k}
:
(n_1\cdots n_k)\times(r_{k-1}n_k),
$$

$$
F_k
:
(r_{k-1}n_k)\times r_k,
$$

$$
R_k
:
r_k\times(n_{k+1}\cdots n_d).
$$

式 (1) と式 (3) は、どちら側の添字を局所行列へまとめるかが鏡映した表現である。標準sweepでは式 (3) の方が直接的である。

左右環境が等長なら、

$$
\boxed{
\|\Delta\mathcal X_k\|_F
=
\|F_k\|_F
=
e_k
}
$$

となる。

---

## 14. 保持部分と捨てる部分の直交性

局所SVDを

$$
V_k
=
\sum_{j=1}^{q_k}
\sigma_{k,j}u_{k,j}w_{k,j}^T
$$

とする。保持部分と捨てる部分を

$$
V_k^{\mathrm{keep}}
=
\sum_{j=1}^{\widetilde r_k}
\sigma_{k,j}u_{k,j}w_{k,j}^T,
$$

$$
V_k^{\mathrm{disc}}
=
\sum_{j=\widetilde r_k+1}^{q_k}
\sigma_{k,j}u_{k,j}w_{k,j}^T
$$

と分ける。Frobenius内積は

$$
\begin{aligned}
&\left\langle
V_k^{\mathrm{keep}},
V_k^{\mathrm{disc}}
\right\rangle_F\\
&=
\sum_{a\le\widetilde r_k}
\sum_{b>\widetilde r_k}
\sigma_{k,a}\sigma_{k,b}
\langle u_{k,a},u_{k,b}\rangle
\langle w_{k,a},w_{k,b}\rangle\\
&=0.
\end{aligned}
$$

一般に

$$
A^TA=I,
\qquad
BB^T=I
$$

なら、

$$
\begin{aligned}
\langle AYB,AZB\rangle_F
&=
\operatorname{tr}
\left(
B^TY^TA^TAZB
\right)\\
&=
\operatorname{tr}
\left(
Y^TZBB^T
\right)\\
&=
\operatorname{tr}(Y^TZ)\\
&=
\langle Y,Z\rangle_F.
\end{aligned}
$$

したがって、canonicalな左右環境は局所直交性も全体空間へ保存する。

---

## 15. 後続段階の誤差とも直交する理由

第 $k$ 段階の保持部分は

$$
\operatorname{span}
\left(
U_{k,\mathrm{keep}}
\right)
$$

にあり、捨てた部分は

$$
\operatorname{span}
\left(
U_{k,\mathrm{disc}}
\right)
$$

にある。

$$
U_{k,\mathrm{keep}}^T
U_{k,\mathrm{disc}}
=
0.
$$

次段へ送られるのは

$$
\Sigma_{k,\mathrm{keep}}
W_{k,\mathrm{keep}}^T
$$

だけである。後続のすべての操作は、すでに保持した左特異部分空間の内部で行われる。そのため、第 $k$ 段階で捨てた誤差と、第 $j>k$ 段階で新たに生じる誤差は直交する。

$$
\boxed{
\langle
\mathcal D_k,
\mathcal D_j
\rangle_F
=
0,
\qquad
k<j
}
$$

ここで

$$
\mathcal D_k
=
\mathcal X^{(k-1)}-\mathcal X^{(k)}
$$

は第 $k$ 段階の実際の段階差である。

---

## 16. 実局所誤差の厳密な二乗和

QR phaseはテンソルを変えない。その後のSVD phaseで、

$$
\mathcal X^{(0)}
=
\mathcal X,
\qquad
\mathcal X^{(d-1)}
=
\widetilde{\mathcal X}
$$

とする。望遠和により、

$$
\begin{aligned}
\mathcal X-\widetilde{\mathcal X}
&=
\sum_{k=1}^{d-1}
\left(
\mathcal X^{(k-1)}-\mathcal X^{(k)}
\right)\\
&=
\sum_{k=1}^{d-1}
\mathcal D_k.
\end{aligned}
$$

実局所誤差を

$$
e_k
=
\|\mathcal D_k\|_F
=
\left(
\sum_{j>\widetilde r_k}
\sigma_{k,j}^2
\right)^{1/2}
$$

とする。異なる段階差の直交性から、exact arithmeticでは

$$
\begin{aligned}
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
&=
\left\|
\sum_{k=1}^{d-1}\mathcal D_k
\right\|_F^2\\
&=
\sum_{k=1}^{d-1}
\|\mathcal D_k\|_F^2
+2\sum_{k<j}
\langle\mathcal D_k,\mathcal D_j\rangle_F\\
&=
\sum_{k=1}^{d-1}e_k^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|\mathcal X-\widetilde{\mathcal X}\|_F
=
\left(
\sum_{k=1}^{d-1}e_k^2
\right)^{1/2}
}
$$

である。

ここでの $e_k$ は、各段階の**更新済み局所行列**で実際に捨てた尾部である。最初のTTから各cutを独立に取り出して計算した値ではない。

浮動小数点実装では、直交性と等式は丸め誤差の範囲での近似一致として検証する。安全な保証を書くときは、事前予算 $\delta_k$ を使って

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\sum_{k=1}^{d-1}e_k^2
\le
\sum_{k=1}^{d-1}\delta_k^2
$$

と、実誤差の等号と予算による不等号を分ける。

---

## 17. 間違えやすい論点

### 17.1 EYMだけではTT全体誤差は出ない

EYMが直接与えるのは局所行列の最良rank近似である。TT全体へ戻すには環境行列を考える必要がある。

### 17.2 right-canonicalだけで全cutのSchmidt係数が見えるわけではない

right-canonical化後、右環境は等長になる。しかし処理中のcutでSchmidt係数として解釈するには、左側もleft-canonicalである必要がある。これは左から右へのSVD sweepで順に実現される。

### 17.3 非正準局所特異値はgauge依存

同じテンソルでも $M=cI$ によって局所特異値を任意に拡大・縮小できる。全体誤差は環境側の逆変化で不変である。

### 17.4 $I_{n_k}\otimes R_k$ は物理添字へ $R_k$ を掛けるのではない

$i_k$ はそのまま保持し、各 $i_k$ のブロックごとにbond添字 $\alpha_k$ を右側物理添字列へ接続する。

### 17.5 二乗和の等号と上界を混ぜない

理想化した標準sweepの実局所誤差 $e_k$ には等号がある。事前に割り当てる予算 $\delta_k$ へ置き換えると不等号になる。

---

## 18. 最終まとめ

### 非正準形

$$
\Delta X_{\langle k-1\rangle}
=
L_{k-1}E_k^T
(I_{n_k}\otimes R_k),
$$

$$
\|\Delta\mathcal X_k\|_F
\le
\|L_{k-1}\|_2
\|E_k\|_F
\|R_k\|_2.
$$

### 正準形

$$
L_{k-1}^TL_{k-1}=I,
\qquad
R_kR_k^T=I,
$$

$$
\|\Delta\mathcal X_k\|_F
=
\|E_k\|_F.
$$

### 複数段階

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\sum_{k=1}^{d-1}e_k^2.
$$

この等式を指定した全体許容誤差へ接続するため、次の [[50_TT-roundingの誤差予算とrank選択]] では $\varepsilon$、局所予算 $\delta_k$、実局所誤差 $e_k$、保持rank $\widetilde r_k$ を区別して定義する。
