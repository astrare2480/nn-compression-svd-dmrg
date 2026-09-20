---
title: TT-SVDの打ち切りと誤差
aliases:
  - truncated TT-SVD
  - TT圧縮
  - TT誤差評価
  - discarded singular values
tags:
  - TT
  - SVD
  - truncation
  - FrobeniusNorm
  - error
---

# TT-SVDの打ち切りと誤差

単一cutでSchmidt状態を捨てる物理的意味は [[44_Schmidt打ち切りとボンド状態の物理的意味]] を参照する。本章は局所SVD残差と複数cutのTT-SVD全体誤差を扱う。

## サマリー

厳密TT-SVDでは、各段階で必要な数値rankを残して元テンソルを表す。圧縮TT-SVDでは、厳密rank $r_k$ より小さい

$$
\widetilde r_k<r_k
$$

を選び、小さいが非ゼロの特異値を意図的に捨てる。

したがって、まず次を区別する。

$$
\boxed{
\text{厳密rank }r_k\text{ の先の特異値は元から0}
}
$$

$$
\boxed{
\text{圧縮rank }\widetilde r_k\text{ の先では、非ゼロ成分も人為的に捨てる}
}
$$

第 $k$ TT-SVD段階で実際に捨てた局所SVD残差の大きさを $\delta_k$ とすると、

$$
\delta_k^2
=
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2.
$$

TT-SVDの誤差について、まず覚える式を

$$
\boxed{
\|X-\widehat X\|_F
\le
\sqrt{
\sum_{k=1}^{d-1}\varepsilon_k^2
}
}
$$

とする。ここで $\varepsilon_k$ は各段階に許した局所誤差上限で、

$$
\delta_k\le\varepsilon_k
$$

を満たすようrankを選ぶ。

一方、標準TT-SVDを入れ子の直交射影として整理し、$\delta_k$ を「実際にその段階で捨てた局所残差」として元テンソル空間へ持ち上げる理想的な整理では、

$$
\|X-\widehat X\|_F^2
=
\sum_k\delta_k^2
$$

と書ける形がある。この等号と、実務上の誤差予算による不等号を混同しないことが重要である。

---

## 1. 厳密rank $r_k$ と圧縮rank $\widetilde r_k$

第 $k$ cutのunfoldingを

$$
X^{\langle k\rangle}
$$

とし、その特異値を大きい順に

$$
\sigma_1^{(k)}
\ge
\sigma_2^{(k)}
\ge
\cdots
\ge0
$$

とする。

厳密TT-rankは

$$
\boxed{
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right)
}
$$

である。

数学的な厳密rankなら、

$$
\sigma_1^{(k)},\ldots,\sigma_{r_k}^{(k)}>0,
$$

$$
\sigma_{r_k+1}^{(k)}
=
\sigma_{r_k+2}^{(k)}
=
\cdots
=
0.
$$

つまり「$r_k$ より後が0」というのは、**厳密rankの定義そのもの**である。

圧縮では別に

$$
\widetilde r_k\le r_k
$$

を選ぶ。

$\widetilde r_k<r_k$ のとき、捨てる

$$
\sigma_{\widetilde r_k+1}^{(k)},
\ldots,
\sigma_{r_k}^{(k)}
$$

は一般には0ではない。

```text
厳密rank r_k
→ 非ゼロ特異値はここまで
→ r_k より先は元から0

圧縮rank r̃_k
→ 上位 r̃_k 個だけ残す
→ r̃_k+1 ... r_k は小さいが非ゼロ
→ 近似側ではそれらを0として扱う
```

---

## 2. 「打ち切ると0にする」の意味

元の行列では

$$
\sigma_{\widetilde r_k+1}^{(k)}>0
$$

であっても、rank-$\widetilde r_k$ 近似では、それ以降の特異値を保持しない。

したがって近似側では

$$
\widehat\sigma_{\widetilde r_k+1}^{(k)}
=
\widehat\sigma_{\widetilde r_k+2}^{(k)}
=
\cdots
=
0.
$$

そのため

$$
\operatorname{rank}
\left(
\widehat X^{\langle k\rangle}
\right)
=
\widetilde r_k
$$

となる。

---

## 3. 1回のtruncated SVDの誤差

行列

$$
A=U\Sigma V^T
$$

のrank-$r$ 近似を

$$
A_r
=
U_r\Sigma_rV_r^T
$$

とする。

捨てた部分は

$$
A-A_r
=
\sum_{j=r+1}^{\rho}
\sigma_j u_jv_j^T
$$

である。

SVDの特異ベクトル対はFrobenius内積で直交するので、

$$
\begin{aligned}
\|A-A_r\|_F^2
&=
\left\|
\sum_{j=r+1}^{\rho}
\sigma_j u_jv_j^T
\right\|_F^2\\
&=
\sum_{j=r+1}^{\rho}
\sigma_j^2
\|u_jv_j^T\|_F^2\\
&=
\sum_{j=r+1}^{\rho}
\sigma_j^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|A-A_r\|_F^2
=
\sum_{j=r+1}^{\rho}\sigma_j^2
}
$$

である。

この途中式では、異なるrank-1成分の交差項を落としている。その根拠を成分和で書くと、

$$
\begin{aligned}
\langle u_jv_j^T,u_\ell v_\ell^T\rangle_F
&=\sum_{a,b}(u_j)_a(v_j)_b(u_\ell)_a(v_\ell)_b\\
&=\left(\sum_a(u_j)_a(u_\ell)_a\right)
\left(\sum_b(v_j)_b(v_\ell)_b\right)\\
&=\delta_{j,\ell}\delta_{j,\ell}
=\delta_{j,\ell},
\end{aligned}
$$

$$
\begin{aligned}
\|A-A_r\|_F^2
&=\sum_{j=r+1}^{\rho}\sum_{\ell=r+1}^{\rho}
\sigma_j\sigma_\ell
\langle u_jv_j^T,u_\ell v_\ell^T\rangle_F\\
&=\sum_{j=r+1}^{\rho}\sum_{\ell=r+1}^{\rho}
\sigma_j\sigma_\ell\delta_{j,\ell}\\
&=\sum_{j=r+1}^{\rho}\sigma_j^2.
\end{aligned}
$$

### 対角行列で打ち切り誤差を全要素確認

ある段階の特異値行列を

$$
\Sigma
=
\operatorname{diag}(5,2,0.5)
=
\begin{pmatrix}
5&0&0\\
0&2&0\\
0&0&0.5
\end{pmatrix}
$$

とする。非ゼロ特異値は3個なので、このcutの厳密rankは

$$
r_k=3
$$

である。圧縮rankを

$$
\widetilde r_k=2
$$

とすると、第3特異値を捨てた近似は

$$
\widetilde\Sigma
=
\operatorname{diag}(5,2,0)
=
\begin{pmatrix}
5&0&0\\
0&2&0\\
0&0&0
\end{pmatrix}
$$

となる。差とFrobenius normを全要素から計算すると、

$$
\Sigma-\widetilde\Sigma
=
\begin{pmatrix}
0&0&0\\
0&0&0\\
0&0&0.5
\end{pmatrix},
$$

$$
\begin{aligned}
\left\|
\Sigma-\widetilde\Sigma
\right\|_F
&=
\sqrt{
0^2+0^2+\cdots+0.5^2
}\\
&=
0.5.
\end{aligned}
$$

したがって、この段階で $r_k=3$ から $\widetilde r_k=2$ へ圧縮した局所誤差は $\delta_k=0.5$ である。TT-SVD全体では、この $\delta_k$ が各段階の局所誤差の一つになり、後述する

$$
\left\|
X-\widehat X
\right\|_F^2
=
\sum_k\delta_k^2
$$

という直交分解、または局所誤差予算 $\varepsilon_k$ を使う全体誤差上界へ接続する。

---

## 4. 具体例

あるcutで特異値が

$$
(10,3,0.4,0.02,0,0,\ldots)
$$

だったとする。

非ゼロ特異値は4個なので、

$$
r_k=4.
$$

### 厳密表現

$$
\widetilde r_k=r_k=4
$$

なら

$$
X^{\langle k\rangle}
=
\sum_{\alpha=1}^4
\sigma_\alpha
u_\alpha v_\alpha^T
$$

で誤差0である。

### rank 2へ圧縮

$$
\widetilde r_k=2
$$

なら、$10,3$ を残し、$0.4,0.02$ を捨てる。

$$
\widehat X^{\langle k\rangle}
=
10u_1v_1^T
+
3u_2v_2^T.
$$

局所誤差の二乗は

$$
\begin{aligned}
\delta_k^2
&=0.4^2+0.02^2\\
&=0.1604,
\end{aligned}
$$

したがって

$$
\delta_k
\approx0.4005.
$$

---

## 5. TT-SVDでは各段階で打ち切る

各段階で捨てるのは、その時点のremainderに対する特異方向である。
第1段階で捨てた成分を、第2段階で再び分解して捨てているわけではない。
そのため、第2段階以降の局所誤差は、既に保持した情報の中で新しく失う量を測る。
元Tensorの全cutを別々にSVDし、それぞれのtailをそのまま足す計算とは区別する。

第 $k$ 段階でSVDする行列を

$$
M_k
=
U_k\Sigma_kV_k^T
$$

とする。

上位 $\widetilde r_k$ 本だけ残すなら、

$$
M_k
\approx
M_{k,\widetilde r_k}
=
U_{k,\widetilde r_k}
\Sigma_{k,\widetilde r_k}
V_{k,\widetilde r_k}^T.
$$

実際に捨てた局所残差の大きさを

$$
\boxed{
\delta_k^2
:=
\sum_{\alpha=\widetilde r_k+1}^{\rho_k}
\left(\sigma_\alpha^{(k)}\right)^2
}
$$

とする。

ここで

$$
\rho_k
=
\operatorname{rank}(M_k).
$$

重要なのは、$k\ge2$ では $M_k$ が通常、元テンソル $X$ のcut unfoldingそのものではないことである。

前段階までに残ったremainderを

$$
(r_{k-1}n_k)
\times
(n_{k+1}\cdots n_d)
$$

へreshapeしたものが $M_k$ である。

---

## 6. rankを小さくすると何が小さくなるか

TT coreは

$$
G^{(k)}
\in
\mathbb R^{\widetilde r_{k-1}\times n_k\times\widetilde r_k}
$$

なので、総保存量は

$$
\boxed{
P_{\mathrm{TT}}
=
\sum_{k=1}^d
\widetilde r_{k-1}n_k\widetilde r_k
}
$$

である。

元のdenseテンソルは

$$
P_{\mathrm{dense}}
=
\prod_{k=1}^dn_k.
$$

したがって、

```text
rankを下げる
→ coreが小さくなる
→ 保存量・計算量が下がる
→ 捨てる特異方向が増える
→ 近似誤差が増える
```

というtrade-offになる。

parameter ratioを

$$
\frac{P_{\mathrm{TT}}}{P_{\mathrm{dense}}}
$$

と定義すれば、1未満で圧縮、1を超えるとTT表現の方が大きい。

---

## 7. 3サイトでの全体誤差の見取り図

局所誤差 $\delta_1,\delta_2$ はそれぞれスカラーだが、実際に失われるものは元Tensorと同じshapeへ展開できる二つの誤差Tensorである。
全体誤差を評価するには、その誤差Tensor同士の向きも重要になる。
一般には同じ向きに重なる可能性があるので三角不等式を使う。
標準TT-SVDでは、ある段階で捨てた方向と後段まで保持された方向が直交するため、後述の交差項が消える。
そこで二乗normの和から平方根を取る評価へ進めるのであって、単に「二つの数だから二乗和にする」わけではない。

3サイトなら打ち切り箇所は2個である。

```text
X
│
├─ 第1SVDで保持
│    └─ 第1局所残差 δ1
│
└─ 残ったremainder
     │
     ├─ 第2SVDで保持 → X_hat
     └─ 第2局所残差 δ2
```

単純な三角不等式だけなら

$$
\|X-\widehat X\|_F
\le
\delta_1+\delta_2
$$

程度しか言えない。

しかし標準TT-SVDでは、SVDによって得られる左interfaceの等長性を使い、局所SVD誤差をより強く全体誤差へ運ぶ。

---

## 8. Frobenius normとreshape

テンソルのFrobenius内積を

$$
\langle A,B\rangle_F
=
\sum_{i_1,\ldots,i_d}
A_{i_1,\ldots,i_d}
B_{i_1,\ldots,i_d}
$$

とする。

$$
\|A\|_F^2
=
\langle A,A\rangle_F.
$$

unfolding / reshapeは要素を別shapeで読み直すだけなので、

$$
\boxed{
\|A\|_F
=
\|A^{\langle k\rangle}\|_F
}
$$

である。

したがって局所SVDの行列誤差を、対応するテンソル誤差として同じFrobenius normで扱える。

---

## 9. 左特異空間への射影

第 $k$ 段階で残す左特異ベクトルを

$$
Q_k
:=
U_k(:,1:\widetilde r_k)
$$

とする。

列直交性より

$$
Q_k^TQ_k
=
I_{\widetilde r_k}.
$$

対応する射影は

$$
P_k
:=
Q_kQ_k^T.
$$

$$
P_k^T=P_k,
\qquad
P_k^2=P_k.
$$

局所残差は

$$
(I-P_k)M_k
$$

であり、

$$
\delta_k
=
\|(I-P_k)M_k\|_F.
$$

SVDの性質から

$$
\delta_k^2
=
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2.
$$

---

## 10. 左interfaceとは何か

ここで「左直交」を、**別々のcore同士が直交すること**と混同しない。

第 $k$ coreまでを左から縮約したinterfaceを

$$
L_k(i_1,\ldots,i_k;\alpha_k)
=
\sum_{\alpha_1,\ldots,\alpha_{k-1}}
G^{(1)}_{1,i_1,\alpha_1}
\cdots
G^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
$$

とする。

行列としては

$$
L_k
\in
\mathbb R^{(n_1\cdots n_k)\times r_k}.
$$

標準的な左から右へのTT-SVDでは、保持した左特異ベクトルから作るため、

$$
\boxed{
L_k^TL_k=I_{r_k}
}
$$

という列直交性を持つ。

したがって直交している対象は

$$
L_k(:,1),\ldots,L_k(:,r_k)
$$

という「左ブロック全体が作る基底」である。

$$
\boxed{
\text{左直交TT}
\ne
\text{異なるcore同士が互いに直交する、という意味}
}
$$

そもそも

$$
G^{(1)}\in\mathbb R^{1\times n_1\times r_1},
$$

$$
G^{(2)}\in\mathbb R^{r_1\times n_2\times r_2}
$$

のようにshape・添字空間が異なるため、通常は

$$
\langle G^{(1)},G^{(2)}\rangle
$$

自体を定義しない。

---

## 11. 左interfaceは局所誤差のnormを変えない

一般に

$$
Q^TQ=I
$$

なら、任意の $Z$ に対して

$$
\begin{aligned}
\|QZ\|_F^2
&=
\operatorname{tr}\left((QZ)^TQZ\right)\\
&=
\operatorname{tr}\left(Z^TQ^TQZ\right)\\
&=
\operatorname{tr}(Z^TZ)\\
&=
\|Z\|_F^2.
\end{aligned}
$$

したがって

$$
\boxed{
Q^TQ=I
\quad\Longrightarrow\quad
\|QZ\|_F=\|Z\|_F
}
$$

である。

TT-SVDでは、途中remainderの座標系で生じた局所残差を、前段の左interfaceを通して元テンソル空間へ戻しても、Frobenius normを増幅しない。この等長性が誤差評価の核になる。

---

## 12. 入れ子の直交射影としての整理

TT-SVDの誤差構造を理解するために、左から右への保持部分を入れ子の直交射影として整理する。

第 $k$ 段階までの左interfaceの列空間への射影を

$$
\Pi_k
:=
L_kL_k^T
$$

とする。

右側には何もしないので、テンソル全体上では概念的に

$$
\mathcal P_k
=
\Pi_k
\otimes
I_{n_{k+1}\cdots n_d}
$$

と読む。

標準的な左から右への構成では、後段の保持空間は前段の保持空間の内部にあるという見取り図を使う。

$$
\operatorname{Ran}(\mathcal P_{k+1})
\subseteq
\operatorname{Ran}(\mathcal P_k).
$$

そのため

$$
\mathcal P_{k+1}\mathcal P_k
=
\mathcal P_{k+1}
$$

という入れ子関係を使って誤差を整理する。

### 入れ子関係を左interfaceの構成式から導く

第 $k+1$ コアを左行列化した列直交行列を $Q_{k+1}$ とすると、

$$
L_{k+1}=(L_k\otimes I_{n_{k+1}})Q_{k+1},
\qquad
(L_k^TL_k)\otimes I_{n_{k+1}}=I
$$

である。第 $k$ 保持空間への射影を左側だけに作用させると、

$$
\begin{aligned}
(\Pi_k\otimes I_{n_{k+1}})L_{k+1}
&=(L_kL_k^T\otimes I_{n_{k+1}})
(L_k\otimes I_{n_{k+1}})Q_{k+1}\\
&=(L_kL_k^TL_k\otimes I_{n_{k+1}})Q_{k+1}\\
&=(L_k\otimes I_{n_{k+1}})Q_{k+1}\\
&=L_{k+1}.
\end{aligned}
$$

右側の未処理の物理軸へ恒等写像を追加すれば、

$$
\mathcal P_k\mathcal P_{k+1}=\mathcal P_{k+1}.
$$

転置を取り、各射影の対称性を使うと、

$$
\mathcal P_{k+1}\mathcal P_k
=(\mathcal P_k\mathcal P_{k+1})^T
=\mathcal P_{k+1}^T
=\mathcal P_{k+1}.
$$

したがって、両順序の積が同じ後段の射影になり、後段が前段の内部にあることを構成式から確認できる。

---

## 13. 段階ごとの誤差テンソル

第 $k$ 段階までの近似を

$$
X_k:=\mathcal P_kX,
$$

$$
X_0:=X
$$

とする。

最終近似を

$$
\widehat X=X_{d-1}
$$

とする。

第 $k$ 段階で新しく失う成分を

$$
\boxed{
E_k
:=
X_{k-1}-X_k
}
$$

と定義する。

入れ子関係を使うと、

$$
\begin{aligned}
E_k
&=
\mathcal P_{k-1}X-
\mathcal P_kX\\
&=
\mathcal P_{k-1}X-
\mathcal P_k\mathcal P_{k-1}X\\
&=
(I-\mathcal P_k)
\mathcal P_{k-1}X.
\end{aligned}
$$

つまり $E_k$ は、

$$
\boxed{
\text{第 }k-1\text{ 段階まで残っていた成分のうち、}
\text{第 }k\text{ 段階で捨てた直交補空間成分}
}
$$

である。

望遠鏡和により、

$$
\begin{aligned}
X-\widehat X
&=X_0-X_{d-1}\\
&=(X_0-X_1)+(X_1-X_2)+\cdots+(X_{d-2}-X_{d-1})\\
&=
\sum_{k=1}^{d-1}E_k.
\end{aligned}
$$

---

## 14. $E_k\perp E_\ell$ をどう読むか

例えば、

$$
E_1\perp E_2
$$

を述べるには、二つの残差が同じテンソル空間に埋め込まれていることと、その直交性を確認する必要がある。

重要なのは、

```text
第1段階で捨てた空間
        ⟂
第1段階で残った空間の内部で後段が扱う空間
```

という入れ子の直交射影構造である。

第2段階以降のremainderは、第1段階で残した側の**座標系**で表されている。後段の局所残差を元テンソル空間へ戻すときには、左interfaceによる等長埋め込みが入る。

したがって、最初に覚える安全な主張は

$$
\boxed{
\text{各段階の局所SVD残差のnormは、左直交interfaceを通しても変わらない}
}
$$

である。

「左直交core」からただちに

```text
core 1 ⟂ core 2
```

と読むのは誤りである。

入れ子の直交射影として誤差テンソル $E_k$ を元空間上で丁寧に定義した整理では、異なる段階の $E_k$ が直交する形を使ってPythagorasを適用できる。

### 異なる段階の内積を0まで計算する

ここではテンソル全要素を辞書順に並べたベクトルに射影を作用させるものとし、$\mathcal P_0=I$ とする。まず

$$
\begin{aligned}
\mathcal P_kE_k
&=\mathcal P_k(\mathcal P_{k-1}X-\mathcal P_kX)\\
&=\mathcal P_kX-\mathcal P_kX\\
&=0.
\end{aligned}
$$

一方、$k<\ell$ なら入れ子関係により

$$
\begin{aligned}
\mathcal P_kE_\ell
&=\mathcal P_k(\mathcal P_{\ell-1}X-\mathcal P_\ell X)\\
&=\mathcal P_{\ell-1}X-\mathcal P_\ell X\\
&=E_\ell.
\end{aligned}
$$

射影の対称性を使えば、

$$
\begin{aligned}
\langle E_k,E_\ell\rangle_F
&=\langle E_k,\mathcal P_kE_\ell\rangle_F\\
&=\langle\mathcal P_k^TE_k,E_\ell\rangle_F\\
&=\langle\mathcal P_kE_k,E_\ell\rangle_F\\
&=\langle0,E_\ell\rangle_F\\
&=0.
\end{aligned}
$$

これで、単なる直感ではなく、元の同一テンソル空間に戻した残差が直交することを示した。

さらに、第 $k$ 局所SVDの実際の残差を

$$
D_k:=M_k-Q_kQ_k^TM_k,
\qquad
F_k:=L_{k-1}\otimes I_{n_k},
\qquad
L_0=(1)
$$

と置く。この節では $Q_k$ は残す列だけを持ち、左interfaceの列数も保持bond次元である。元空間への埋め込みは

$$
E_k^{\langle k\rangle}=F_kD_k,
\qquad
F_k^TF_k=I
$$

なので、

$$
\begin{aligned}
\|E_k\|_F^2
&=\|E_k^{\langle k\rangle}\|_F^2\\
&=\|F_kD_k\|_F^2\\
&=\operatorname{tr}(D_k^TF_k^TF_kD_k)\\
&=\operatorname{tr}(D_k^TD_k)\\
&=\delta_k^2.
\end{aligned}
$$

したがって、望遠鏡和の二乗を交差項まで展開すると、

$$
\begin{aligned}
\|X-\widehat X\|_F^2
&=\left\|\sum_{k=1}^{d-1}E_k\right\|_F^2\\
&=\sum_{k=1}^{d-1}\|E_k\|_F^2
+2\sum_{k<\ell}\langle E_k,E_\ell\rangle_F\\
&=\sum_{k=1}^{d-1}\delta_k^2.
\end{aligned}
$$

この等式の前提は、各段階がexactなtruncated SVDであり、残差を左直交interfaceで元空間へ戻している標準的な左から右へのTT-SVDである。浮動小数点の直交性・SVD・縮約誤差まで厳密にこの等式へ含めるわけではない。

---

### 3サイトで、第1cutの同じ行列へ戻して内積を計算する

誤差の直交性を、3サイトの行列で確認する。
上の一般の射影証明を、remainderの具体的なshapeへ対応させるための補足である。
実係数として、保持した第1 SVDの左特異ベクトルと射影を

$$
Q:=U^{(1)}_{:,1:\widetilde r_1}
\in\mathbb R^{n_1\times\widetilde r_1},\qquad
Q^TQ=I_{\widetilde r_1},\qquad P_1:=QQ^T
$$

とする。第1cut行列 $C:=X^{\langle1\rangle}$ のshapeは $n_1\times(n_2n_3)$ である。
第1段階で残す部分と、その右側座標は

$$
\begin{aligned}
B^{(1)}&:=Q^TC
\in\mathbb R^{\widetilde r_1\times(n_2n_3)},\\
X_1^{\langle1\rangle}&=QB^{(1)}=QQ^TC=P_1C,\\
E_1^{\langle1\rangle}&=C-X_1^{\langle1\rangle}
=(I_{n_1}-P_1)C.
\end{aligned}
$$

第2 SVDを行うのは元の $C$ ではなく、

$$
\begin{aligned}
\mathcal B^{(1)}
&:=\operatorname{reshape}(B^{(1)},\widetilde r_1,n_2,n_3),\\
M^{(2)}
&:=\operatorname{reshape}(\mathcal B^{(1)},\widetilde r_1n_2,n_3).
\end{aligned}
$$

である。そのtruncated SVDで得た行列近似を $\widehat M^{(2)}$ とし、
第1cutに対応する座標の並びへ戻して

$$
\begin{aligned}
\widehat B^{(1)}
&:=\operatorname{reshape}(\widehat M^{(2)},\widetilde r_1,n_2n_3),\\
\Delta B^{(1)}&:=B^{(1)}-\widehat B^{(1)},\\
\widehat X^{\langle1\rangle}&=Q\widehat B^{(1)},\\
E_2^{\langle1\rangle}
&=X_1^{\langle1\rangle}-\widehat X^{\langle1\rangle}\\
&=QB^{(1)}-Q\widehat B^{(1)}\\
&=Q\Delta B^{(1)}.
\end{aligned}
$$

と定義する。$\Delta B^{(1)}$ と $E_1^{\langle1\rangle}$ はshapeが異なるため、
そのまま内積を取らない。$Q$ を掛けた $E_2^{\langle1\rangle}$ は
$E_1^{\langle1\rangle}$ と同じ $n_1\times(n_2n_3)$ 行列になっている。

射影の対称性 $P_1^T=P_1$ と、保持した列の正規直交性から

$$
\begin{aligned}
P_1Q&=QQ^TQ=Q(Q^TQ)=Q,\\
(I_{n_1}-P_1)Q&=Q-P_1Q=Q-Q=0.
\end{aligned}
$$

を得る。したがって、Frobenius内積は途中の積まで書けば

$$
\begin{aligned}
\langle E_1,E_2\rangle_F
&=\operatorname{tr}\left[
(E_1^{\langle1\rangle})^TE_2^{\langle1\rangle}\right]\\
&=\operatorname{tr}\left[
\bigl((I_{n_1}-P_1)C\bigr)^TQ\Delta B^{(1)}\right]\\
&=\operatorname{tr}\left[
C^T(I_{n_1}-P_1)^TQ\Delta B^{(1)}\right]\\
&=\operatorname{tr}\left[
C^T(I_{n_1}-P_1)Q\Delta B^{(1)}\right]\\
&=\operatorname{tr}\left[C^T0\Delta B^{(1)}\right]=0.
\end{aligned}
$$

となる。第2段階で捨てた成分も、元空間では $Q$ の列空間の内部にあり、
第1段階で捨てた直交補空間の成分を復活させない。
これが「異なるcore同士の直交」ではなく「元空間へ戻した誤差同士の直交」という意味である。

局所誤差 $\delta_2:=\|M^{(2)}-\widehat M^{(2)}\|_F$ はreshapeで変わらず、
$Q$ による埋め込みでも

$$
\begin{aligned}
\|E_2\|_F^2
&=\operatorname{tr}\left[(\Delta B^{(1)})^TQ^TQ\Delta B^{(1)}\right]\\
&=\operatorname{tr}\left[(\Delta B^{(1)})^T\Delta B^{(1)}\right]\\
&=\|\Delta B^{(1)}\|_F^2=\delta_2^2.
\end{aligned}
$$

となる。$\delta_1:=\|E_1\|_F$ と置けば、$X-\widehat X=E_1+E_2$ なので

$$
\begin{aligned}
\|X-\widehat X\|_F^2
&=\|E_1\|_F^2+2\langle E_1,E_2\rangle_F+\|E_2\|_F^2\\
&=\delta_1^2+0+\delta_2^2.
\end{aligned}
$$

を得る。この3サイトの計算も、上の標準的な左から右へのTT-SVD・exact arithmeticの前提で読む。

---

## 15. 実際の局所残差と誤差予算を分ける

記号を分けると混乱しにくい。

### 実際に捨てた局所誤差

$$
\boxed{
\delta_k
=
\text{第 }k\text{ 段階で実際に捨てた局所SVD残差のnorm}
}
$$

$$
\delta_k^2
=
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2.
$$

### 許容する上限

$$
\boxed{
\varepsilon_k
=
\text{第 }k\text{ 段階に許す局所誤差の予算}
}
$$

rankを

$$
\delta_k\le\varepsilon_k
$$

となるように選ぶ。

この区別をすると、

$$
\|X-\widehat X\|_F^2
=
\sum_k\delta_k^2
\le
\sum_k\varepsilon_k^2
$$

という関係を読みやすい。

したがって、

$$
\boxed{
\|X-\widehat X\|_F
\le
\sqrt{
\sum_{k=1}^{d-1}\varepsilon_k^2
}
}
$$

が実務で使いやすい形になる。

---

## 16. 全体誤差を $\varepsilon$ 以下にしたい場合

目標を

$$
\|X-\widehat X\|_F
\le
\varepsilon
$$

とする。

最も単純な予算配分は、$d-1$ 個のcutへ同じ二乗誤差予算を配ることである。

$$
\boxed{
\varepsilon_k
=
\frac{\varepsilon}{\sqrt{d-1}}
}
$$

とすれば、

$$
\begin{aligned}
\sum_{k=1}^{d-1}\varepsilon_k^2
&=
(d-1)
\left(
\frac{\varepsilon}{\sqrt{d-1}}
\right)^2\\
&=
\varepsilon^2.
\end{aligned}
$$

したがって

$$
\|X-\widehat X\|_F
\le
\varepsilon.
$$

第 $k$ 段階では、

$$
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2
\le
\frac{\varepsilon^2}{d-1}
$$

を満たすように $\widetilde r_k$ を選ぶ、という考え方になる。

---

## 17. 各局所SVDで最良でもTT全体で最良とは限らない

ここはTuckerとの比較でも重要である。

各段階のtruncated SVDは、その段階の行列 $M_k$ に対してはrank-$\widetilde r_k$ の最良近似を与える。

しかし、逐次的に左から一度だけ進むTT-SVD全体が、指定TT-rank

$$
(\widetilde r_1,\ldots,\widetilde r_{d-1})
$$

を持つ全てのTTテンソルの中で、必ず大域的な最小誤差解になるとは限らない。

一般には

$$
\widehat X_{\mathrm{TT\text{-}SVD}}
\ne
\underset{
\operatorname{TT\text{-}rank}(Y)
\le
(\widetilde r_1,\ldots,\widetilde r_{d-1})
}{\arg\min}
\|X-Y\|_F.
$$

最適TT近似誤差を

$$
E_{\mathrm{best}}
:=
\min_{
\operatorname{TT\text{-}rank}(Y)
\le
(\widetilde r_1,\ldots,\widetilde r_{d-1})
}
\|X-Y\|_F
$$

と置き、TT-SVDには

$$
\boxed{
\|X-\widehat X_{\mathrm{TT\text{-}SVD}}\|_F
\le
\sqrt{d-1}\,E_{\mathrm{best}}
}
$$

というquasi-optimalityの位置づけがある、と整理した。

### 準最適性の上界がどこから来るか

上界も結果だけでなく、局所問題への写像から追う。指定TT-rank上限を満たす任意のテンソル $Y$ について、

$$
\operatorname{rank}(Y^{\langle k\rangle})\le\widetilde r_k,
\qquad
M_k=F_k^TX^{\langle k\rangle},
\qquad
F_k=L_{k-1}\otimes I_{n_k}
$$

である。局所問題の比較候補を

$$
C_k:=F_k^TY^{\langle k\rangle}
$$

と置けば、

$$
\operatorname{rank}(C_k)
\le\operatorname{rank}(Y^{\langle k\rangle})
\le\widetilde r_k.
$$

局所truncated SVDの最適性と、$F_k^T$ がnormを増やさないことから、

$$
\begin{aligned}
\delta_k
&=\min_{\operatorname{rank}(C)\le\widetilde r_k}\|M_k-C\|_F\\
&\le\|M_k-C_k\|_F\\
&=\|F_k^T(X^{\langle k\rangle}-Y^{\langle k\rangle})\|_F\\
&\le\|X^{\langle k\rangle}-Y^{\langle k\rangle}\|_F\\
&=\|X-Y\|_F.
\end{aligned}
$$

最後の不等式は、$F_kF_k^T$ が直交射影なので

$$
\|F_k^TZ\|_F^2
=\operatorname{tr}(Z^TF_kF_k^TZ)
\le\operatorname{tr}(Z^TZ)
=\|Z\|_F^2
$$

から従う。全段階で足し合わせると、

$$
\|X-\widehat X_{\mathrm{TT\text{-}SVD}}\|_F^2
=\sum_{k=1}^{d-1}\delta_k^2
\le(d-1)\|X-Y\|_F^2.
$$

右辺を許される $Y$ の中で最小化し、平方根を取れば、既出の $\sqrt{d-1}$ の上界を得る。これは「全体最適である」という証明ではなく、「全体最適誤差の何倍以内か」という証明である。

つまり、

$$
\boxed{
\text{各局所SVDでは最適だが、逐次TT-SVD全体を大域的最適化と同一視しない}
}
$$

という注意である。

---

## 18. 「各cutで最良」と「後から全coreを最適化」は別

TT-SVDは左から右へ1回進む構成法である。

```text
現在のremainder
→ 最良rank-r_k SVD
→ 左coreを確定
→ remainderを次へ
```

一度確定した左coreを、後段階の情報を使って再最適化しない。

そのため、全coreを同時・交互に調整する変分最適化とは目的が異なる。

この差が、後に学ぶALS / one-site / two-site DMRGへつながる。ただしそれらは今回のdocs範囲外であり、ここでは「TT-SVDは逐次構成法で、大域的最適化そのものではない」とだけ固定する。

---

## 19. truncation後の第2cut rankで比較すべきもの

第1 SVDを

$$
\widehat r_1<r_1
$$

へ打ち切ると、表現対象は元の $X$ から近似

$$
\widehat X
$$

へ変わる。

打ち切った $\widehat U$ と $\widehat B$ に対して

$$
\widehat X^{\langle2\rangle}
=
(\widehat U\otimes I_{n_2})
\widehat B_{\mathrm{cut2}}
$$

が成立する。

$\widehat U$ の列も直交しているため、

$$
\boxed{
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(\widehat B_{\mathrm{cut2}}\right)
}
$$

である。

しかし一般には

$$
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
\ne
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
$$

となり得る。

したがってtruncation後のrank不変性実験では、

```text
X_hat cut rank
vs
B_hat cut rank
```

を比較し、別に

```text
original X cut rank
vs
X_hat cut rank
```

でtruncationによるrank変化を確認する。

---

## 20. numerical rankの注意

理論上0の特異値でも、浮動小数点計算では

$$
10^{-15},
\quad
10^{-14}
$$

のような小さい非ゼロ値として出ることがある。

したがって実装では「数値的に0と見なす閾値」が必要になる。

このプロジェクトの `tt_svd_exact` は

```python
torch.linalg.matrix_rank(mat)
```

のdefault toleranceに基づくnumerical rankを使う。

そのため、ここでのexactは

$$
\boxed{
\text{数学的symbolic rankではなく、採用したnumerical rankを打ち切らず保持する}
}
$$

という意味である。

---

## 21. Notebook 02で見たtrade-off

基礎Notebookでは

```python
max_ranks = [1, 2, 4, 8]
```

をsweepし、rank上限を増やしたとき

- bond rankが増える
- TT parameter数が増える
- parameter ratioが増える
- relative reconstruction errorが下がる

ことを確認した。

小さいランダムTensorではTT parameter数がdenseより大きくなることもあり、

$$
\text{parameter ratio}>1
$$

なら「TTにしたが圧縮にはなっていない」ことも重要な観察である。

---

## 22. この章で固定する理解

- 厳密rank $r_k$ の先の特異値は元から0。
- 圧縮rank $\widetilde r_k<r_k$ では、小さいが非ゼロの特異値を意図的に捨てる。
- 1回のtruncated SVDの局所誤差は、捨てた特異値の二乗和で測れる。
- TT-SVDの誤差評価で重要なのは「別々のcore同士が直交」ではなく、左interfaceが列直交で等長写像になること。
- $\delta_k$ を実際の局所残差、$\varepsilon_k$ を許容上限として分けると、等号と不等号を混同しにくい。
- 実務では

$$
\|X-\widehat X\|_F
\le
\sqrt{\sum_k\varepsilon_k^2}
$$

を安全な全体誤差上界として使う。
- 全体誤差 $\varepsilon$ を等分するなら、各cutに $\varepsilon/\sqrt{d-1}$ を配る考え方がある。
- 各局所SVDはその場では最良でも、TT-SVD全体を指定TT-rankでの大域的最良近似と同一視しない。
- truncation後のrank不変性は、元の $X$ ではなく近似 $\widehat X$ と $\widehat B$ の間で確認する。

次の [[34_基底変換とTT-rank不変性]] では、前段の列直交基底変換がなぜrankを保存するかを $U\otimes I$ から厳密に導く。
