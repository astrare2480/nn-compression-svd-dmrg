---
title: TT-roundingの定義と正準化sweep
tags:
  - TT
  - MPS
  - TT-rounding
  - canonical-form
  - QR
  - SVD
  - tensor-train
  - rank-truncation
---

# TT-roundingの定義と正準化sweep

このノートでは、すでにTT形式で与えられたテンソルの内部rankを下げる **TT-rounding** を扱う。

[[47_TT-SVDの準最適性と誤差上界|TT-SVD]] はdense tensorからTTを新規構成する。一方、TT-roundingは既存のTTコア列をdense tensorへ戻さず、

$$
\text{右から左へのQR正準化}
\quad\longrightarrow\quad
\text{左から右へのtruncated SVD}
$$

という2段階で再圧縮する。

局所SVDの誤差がTT全体の誤差になる理由は [[49_TT-roundingの環境行列と誤差直交分解]]、全体許容誤差から局所予算と保持rankを決める方法は [[50_TT-roundingの誤差予算とrank選択]]、PyTorch固有のreshape・`.mT`・検証方法は [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]] に分ける。

---

## 0. このノートの到達点

実数TTを

$$
\mathcal X
=
\llbracket
G^{(1)},\ldots,G^{(d)}
\rrbracket,
$$

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k},
\qquad
r_0=r_d=1
$$

とする。TT-roundingの目標は、物理shape

$$
(n_1,\ldots,n_d)
$$

を変えずに、内部TT-rank

$$
(r_1,\ldots,r_{d-1})
$$

を小さくしたコア列

$$
\widetilde{\mathcal X}
=
\llbracket
\widetilde G^{(1)},\ldots,\widetilde G^{(d)}
\rrbracket
$$

を作ることである。相対誤差を指定する場合の目標は

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\varepsilon\|\mathcal X\|_F
$$

である。

標準的な一回のTT-roundingは次で完了する。

1. 右端から左へreduced QRを行い、第2コアから第 $d$ コアまでをright-canonicalにする。
2. 左端から右へtruncated SVDを行い、各bond rankを選ぶ。
3. 各SVDで残す $U$ を左正準コアにし、$\Sigma W^T$ を右隣へ吸収する。
4. 最終コアまで到達したら終了する。追加のQR sweepは必須ではない。

---

## 1. TT-SVDとTT-roundingの違い

### 1.1 TT-SVD

TT-SVDの入力はdense tensor

$$
A
\in
\mathbb R^{n_1\times\cdots\times n_d}
$$

である。元のdense tensorまたは逐次remainderを行列化し、左から右へSVDしてTTコアを新しく作る。

### 1.2 TT-rounding

TT-roundingの入力はすでに存在するTTコア列である。

$$
G^{(1)},\ldots,G^{(d)}.
$$

各コアの局所行列化とgauge変換だけを使うため、原理上は

$$
n_1n_2\cdots n_d
$$

個のdense要素を生成する必要がない。

TT-roundingが必要になる典型例は、TT同士の加算、Hadamard積、線形作用素の適用などでrankが増えた後である。roundingによって以後の保存量と縮約計算量を抑える。

### 1.3 保証するものと保証しないもの

TT-roundingは、指定した誤差範囲内でrankを下げる逐次法である。各rankの全組合せを探索して、保存パラメータ数が大域的に最小となるTTを必ず返す方法ではない。

---

## 2. dense shapeとTT-rankは別物

3階テンソルのTT表示を

$$
\begin{aligned}
\mathcal X(i_1,i_2,i_3)
={}&
\sum_{\alpha_1=1}^{r_1}
\sum_{\alpha_2=1}^{r_2}
G^{(1)}(1,i_1,\alpha_1)
G^{(2)}(\alpha_1,i_2,\alpha_2)
G^{(3)}(\alpha_2,i_3,1)
\end{aligned}
$$

とする。

説明用に

$$
(n_1,n_2,n_3)
=
(2,3,2),
\qquad
(r_1,r_2)
=
(3,4)
$$

を使う。このとき、

$$
G^{(1)}:(1,2,3),
\qquad
G^{(2)}:(3,3,4),
\qquad
G^{(3)}:(4,2,1)
$$

である。rounding後にrankが

$$
(\widetilde r_1,\widetilde r_2)
=
(2,2)
$$

となれば、コアshapeは

$$
(1,2,2),
\qquad
(2,3,2),
\qquad
(2,2,1)
$$

になる。しかしdense tensorの物理shapeは前後とも

$$
(2,3,2)
$$

であり、dense要素数は

$$
2\cdot3\cdot2
=
12
$$

のままである。縮むのは内部添字 $\alpha_1,\alpha_2$ の範囲であり、物理添字 $i_1,i_2,i_3$ の範囲ではない。

---

## 3. 保存パラメータ数とMACs

TTコア列の保存要素数は

$$
N_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}n_kr_k
$$

である。

### 3.1 rounding前

$$
\begin{aligned}
N_{\mathrm{TT,old}}
&=
1\cdot2\cdot3
+3\cdot3\cdot4
+4\cdot2\cdot1\\
&=
6+36+8\\
&=
50.
\end{aligned}
$$

### 3.2 rounding後

$$
\begin{aligned}
N_{\mathrm{TT,new}}
&=
1\cdot2\cdot2
+2\cdot3\cdot2
+2\cdot2\cdot1\\
&=
4+12+4\\
&=
20.
\end{aligned}
$$

したがってTT表現の保存量は

$$
50
\longrightarrow
20
$$

となる。ただし、この説明用テンソルのdense要素数は12なので、rounding後も

$$
20>12
$$

である。これは小さな例でshapeを追うためのものであり、メモリ圧縮の有利さを示す例ではない。

一般に各物理次元を $n$、内部rankを概ね $r$ と見積もれば、

$$
N_{\mathrm{dense}}
=
n^d,
\qquad
N_{\mathrm{TT}}
\approx
dnr^2.
$$

dense表現は $d$ に対して指数的に増えるが、rankが制御されているTT表現は概ね線形に増える。

### 3.3 denseより保存量が小さくなる例

4階テンソルで

$$
(n_1,n_2,n_3,n_4)
=
(10,20,30,40),
$$

$$
(r_1,r_2,r_3)
=
(4,6,5)
$$

とする。dense要素数は

$$
N_{\mathrm{dense}}
=
10\cdot20\cdot30\cdot40
=
240000
$$

である。一方、TTコアの保存要素数は

$$
\begin{aligned}
N_{\mathrm{TT}}
&=
1\cdot10\cdot4
+4\cdot20\cdot6
+6\cdot30\cdot5
+5\cdot40\cdot1\\
&=
40+480+900+200\\
&=
1620.
\end{aligned}
$$

この例では、物理shapeを保ったまま保存量を大きく減らせる。ただし、これは与えたrankでの保存量比較であり、そのrankが指定誤差に対して大域的に最小であることまでは主張しない。

### 3.4 1要素を縮約するMACs

3階TTの1要素を計算する場合、

$$
v^{(1)}(\alpha_1)
=
G^{(1)}(1,i_1,\alpha_1),
$$

$$
v^{(2)}(\alpha_2)
=
\sum_{\alpha_1=1}^{r_1}
v^{(1)}(\alpha_1)
G^{(2)}(\alpha_1,i_2,\alpha_2),
$$

$$
\mathcal X(i_1,i_2,i_3)
=
\sum_{\alpha_2=1}^{r_2}
v^{(2)}(\alpha_2)
G^{(3)}(\alpha_2,i_3,1).
$$

主要な積和数を概算すると、rounding前は

$$
r_1r_2+r_2
=
3\cdot4+4
=
16,
$$

rounding後は

$$
\widetilde r_1\widetilde r_2+\widetilde r_2
=
2\cdot2+2
=
6
$$

となる。実行時間はGPU kernel起動、メモリアクセス、batch化などにも依存するため、必ず $16:6$ になるわけではない。しかし主要な行列積shapeがrankとともに小さくなる点は変わらない。

---

## 4. 二種類のmatricization

同じ第 $k$ コア

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

でも、sweep方向によって行列化が異なる。

### 4.1 右正準化用のhorizontal unfolding

左bondを行、物理添字と右bondを列にまとめる。

$$
H_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1},n_kr_k)
\right)
\in
\mathbb R^{r_{k-1}\times(n_kr_k)}.
$$

要素対応は

$$
H_k
\bigl(
\alpha_{k-1},(i_k,\alpha_k)
\bigr)
=
G^{(k)}
(\alpha_{k-1},i_k,\alpha_k)
$$

である。right-canonical条件は行直交性

$$
H_kH_k^T
=
I_{r_{k-1}}
$$

である。

### 4.2 左から右へのSVD用のvertical unfolding

左bondと物理添字を行、右bondを列にまとめる。

$$
V_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1}n_k,r_k)
\right)
\in
\mathbb R^{(r_{k-1}n_k)\times r_k}.
$$

要素対応は

$$
V_k
\bigl(
(\alpha_{k-1},i_k),\alpha_k
\bigr)
=
G^{(k)}
(\alpha_{k-1},i_k,\alpha_k)
$$

である。SVD後に保持する左特異ベクトルの列直交性がleft-canonical条件になる。

$$
U_{k,\mathrm{keep}}^T
U_{k,\mathrm{keep}}
=
I_{\widetilde r_k}.
$$

---

## 5. 右から左へのQR sweep

処理順序は

$$
k=d,d-1,\ldots,2
$$

である。

### 5.1 1ステップのreduced QR

right-canonical条件は $H_k$ の行直交性なので、QRを $H_k$ ではなく転置へ適用する。

$$
H_k^T
=
Q_kR_k.
$$

$$
H_k^T
\in
\mathbb R^{(n_kr_k)\times r_{k-1}}.
$$

reduced QRの共通次元を

$$
q_k
=
\min(n_kr_k,r_{k-1})
$$

とすれば、

$$
Q_k
\in
\mathbb R^{(n_kr_k)\times q_k},
\qquad
R_k
\in
\mathbb R^{q_k\times r_{k-1}}.
$$

列直交性は

$$
Q_k^TQ_k
=
I_{q_k}
$$

である。元の行列は

$$
H_k
=
R_k^TQ_k^T
$$

と復元される。

### 5.2 新しい右正準コア

$$
\widehat H_k
=
Q_k^T
\in
\mathbb R^{q_k\times(n_kr_k)}
$$

を3階コアへ戻す。

$$
\widehat G^{(k)}
=
\operatorname{reshape}
\left(
Q_k^T,
(q_k,n_k,r_k)
\right).
$$

すると、

$$
\widehat H_k\widehat H_k^T
=
Q_k^TQ_k
=
I_{q_k}
$$

なので、$\widehat G^{(k)}$ はright-canonicalである。

### 5.3 $R_k^T$ を左隣へ吸収する

$R_k^T$ のshapeは

$$
R_k^T
\in
\mathbb R^{r_{k-1}\times q_k}
$$

である。左隣のコアを

$$
G^{(k-1)}
\in
\mathbb R^{r_{k-2}\times n_{k-1}\times r_{k-1}}
$$

とすると、更新式は

$$
\begin{aligned}
\widehat G^{(k-1)}
(\alpha_{k-2},i_{k-1},\beta_{k-1})
={}&
\sum_{\alpha_{k-1}=1}^{r_{k-1}}
G^{(k-1)}
(\alpha_{k-2},i_{k-1},\alpha_{k-1})\\
&\times
R_k^T
(\alpha_{k-1},\beta_{k-1}).
\end{aligned}
$$

更新後のshapeは

$$
\widehat G^{(k-1)}
\in
\mathbb R^{r_{k-2}\times n_{k-1}\times q_k}.
$$

2コアの縮約を展開すると、

$$
\begin{aligned}
&\sum_{\alpha_{k-1}}
G^{(k-1)}
(\alpha_{k-2},i_{k-1},\alpha_{k-1})
G^{(k)}
(\alpha_{k-1},i_k,\alpha_k)\\
&=
\sum_{\beta_{k-1}}
\widehat G^{(k-1)}
(\alpha_{k-2},i_{k-1},\beta_{k-1})
\widehat G^{(k)}
(\beta_{k-1},i_k,\alpha_k).
\end{aligned}
$$

したがって、このQRステップはテンソルを変えないgauge変換である。

### 5.4 QRでbondが縮む場合

$$
r_{k-1}>n_kr_k
$$

なら、reduced QRのshapeだけから

$$
q_k=n_kr_k<r_{k-1}
$$

となり、左bondの表示次元が縮む。これは $H_k^T=Q_kR_k$ をすべて保持しているため、SVD truncationのような近似誤差を導入しない。元のbond空間に、右側へ独立に接続できない冗長な表示自由度が含まれていたことを意味する。

ただし通常の浮動小数点reduced QRは、近似的に小さい特異値を自動判定してさらに列を削るrank-revealing処理ではない。数値的なrank削減は後段のSVDで明示的に行う。

---

## 6. $d=3$ のQR shapeを最後まで追う

説明用コアをもう一度書く。

$$
G^{(1)}:(1,2,3),
\qquad
G^{(2)}:(3,3,4),
\qquad
G^{(3)}:(4,2,1).
$$

### 6.1 第3コア

$$
H_3
\in
\mathbb R^{4\times(2\cdot1)}
=
\mathbb R^{4\times2},
$$

$$
H_3^T
\in
\mathbb R^{2\times4}.
$$

reduced QRでは

$$
q_3
=
\min(2,4)
=
2,
$$

$$
Q_3
\in
\mathbb R^{2\times2},
\qquad
R_3
\in
\mathbb R^{2\times4}.
$$

したがって、

$$
\widehat G^{(3)}
:
(2,2,1),
$$

$$
\widehat G^{(2)}
:
(3,3,2).
$$

この $r_2:4\to2$ は近似ではなく、最終物理次元 $n_3=2$ と右端rank $r_3=1$ から許される独立な左bond次元が最大2であることによるexactな整理である。

### 6.2 更新後の第2コア

$$
H_2
\in
\mathbb R^{3\times(3\cdot2)}
=
\mathbb R^{3\times6},
$$

$$
H_2^T
\in
\mathbb R^{6\times3}.
$$

$$
q_2
=
\min(6,3)
=
3,
$$

$$
Q_2
\in
\mathbb R^{6\times3},
\qquad
R_2
\in
\mathbb R^{3\times3}.
$$

したがって、

$$
\widehat G^{(2)}
:
(3,3,2),
$$

$$
\widehat G^{(1)}
:
(1,2,3).
$$

QR sweep終了時には、第2・第3コアがright-canonicalで、第1コアがorthogonality centerである。

---

## 7. 左から右へのtruncated SVD sweep

処理順序は

$$
k=1,2,\ldots,d-1
$$

である。

### 7.1 局所SVD

vertical unfoldingへeconomy SVDを適用する。

$$
V_k
=
U_k\Sigma_kW_k^T.
$$

右特異ベクトルを $W_k$ と書くのは、局所行列 $V_k$ と記号が衝突しないためである。

保持rankを $\widetilde r_k$ とすれば、

$$
U_{k,\mathrm{keep}}
\in
\mathbb R^{(r_{k-1}n_k)\times\widetilde r_k},
$$

$$
\Sigma_{k,\mathrm{keep}}
\in
\mathbb R^{\widetilde r_k\times\widetilde r_k},
$$

$$
W_{k,\mathrm{keep}}^T
\in
\mathbb R^{\widetilde r_k\times r_k}.
$$

### 7.2 新しい第 $k$ コア

$$
\widetilde G^{(k)}
=
\operatorname{reshape}
\left(
U_{k,\mathrm{keep}},
(r_{k-1},n_k,\widetilde r_k)
\right).
$$

SVDの列直交性より、

$$
U_{k,\mathrm{keep}}^T
U_{k,\mathrm{keep}}
=
I_{\widetilde r_k}.
$$

よって新しい第 $k$ コアはleft-canonicalである。成分表示は

$$
\sum_{\alpha_{k-1}=1}^{r_{k-1}}
\sum_{i_k=1}^{n_k}
\widetilde G^{(k)}
(\alpha_{k-1},i_k,\beta_k)
\widetilde G^{(k)}
(\alpha_{k-1},i_k,\gamma_k)
=
\delta_{\beta_k,\gamma_k}.
$$

### 7.3 $\Sigma W^T$ を右隣へ吸収する

$$
B_k
=
\Sigma_{k,\mathrm{keep}}
W_{k,\mathrm{keep}}^T
\in
\mathbb R^{\widetilde r_k\times r_k}.
$$

右隣コアの更新は

$$
\begin{aligned}
\widetilde G^{(k+1)}
(\beta_k,i_{k+1},\alpha_{k+1})
={}&
\sum_{\alpha_k=1}^{r_k}
B_k(\beta_k,\alpha_k)\\
&\times
G^{(k+1)}
(\alpha_k,i_{k+1},\alpha_{k+1}).
\end{aligned}
$$

更新後のshapeは

$$
\widetilde G^{(k+1)}
\in
\mathbb R^{\widetilde r_k\times n_{k+1}\times r_{k+1}}.
$$

---

## 8. $d=3$ のSVD shapeを最後まで追う

QR sweep後のshapeは

$$
(1,2,3),
\qquad
(3,3,2),
\qquad
(2,2,1).
$$

### 8.1 第1コアをrank 2へ

$$
V_1
\in
\mathbb R^{(1\cdot2)\times3}
=
\mathbb R^{2\times3}.
$$

この行列が持てる特異値は

$$
q_1
=
\min(2,3)
=
2
$$

個だけである。したがって、保持rankを

$$
\widetilde r_1=2
$$

とする場合は、2本の特異値をすべて保持する。捨てる特異値は存在せず、局所誤差は空和として

$$
e_1^2
=
\sum_{j=\widetilde r_1+1}^{q_1}
\sigma_{1,j}^2
=
\sum_{j=3}^{2}
\sigma_{1,j}^2
=
0
$$

である。この $r_1:3\to2$ は、行列の行数が2であるために有効rankが最大2になることを反映したexactな次元整理であり、非零特異値を捨てる近似ではない。

このとき、

$$
U_{1,\mathrm{keep}}
:
(2,2),
$$

$$
B_1
=
\Sigma_{1,\mathrm{keep}}
W_{1,\mathrm{keep}}^T
:
(2,3).
$$

したがって、

$$
\widetilde G^{(1)}
:
(1,2,2),
$$

$$
\widetilde G^{(2)}
:
(2,3,2).
$$

### 8.2 第2コア

$$
V_2
\in
\mathbb R^{(2\cdot3)\times2}
=
\mathbb R^{6\times2}.
$$

この例では右bondはQR段階ですでに2なので、保持rank

$$
\widetilde r_2=2
$$

なら追加の打ち切りは起きない。

$$
\widetilde G^{(2)}
:
(2,3,2),
\qquad
\widetilde G^{(3)}
:
(2,2,1).
$$

最終shapeは

$$
(1,2,2),
\qquad
(2,3,2),
\qquad
(2,2,1)
$$

である。この例では、$r_2:4\to2$ はreduced QRによるexactな冗長次元整理、$r_1:3\to2$ はthin SVDで全特異値を保持したexactな次元整理である。どちらも表示rankは小さくなるが、近似誤差は導入しない。

第1cutで実際に近似するには、さらに

$$
\widetilde r_1=1
$$

として第2特異値を捨てる必要がある。この場合の局所誤差は

$$
e_1
=
\sigma_{1,2}
$$

であり、$\sigma_{1,2}>0$ なら非零の近似誤差が生じる。したがって、「bond次元が減ったこと」と「非零特異値を捨てたこと」を区別しなければならない。

---

## 9. なぜ正準化にQR、圧縮にSVDを使うか

### 9.1 QRもSVDも正準化できる

QRでもSVDでも直交因子を作れるため、right-canonical化そのものはどちらでも可能である。

SVDで

$$
H_k^T
=
U_k\Sigma_kW_k^T
$$

とすれば、$U_k^T$ を右正準コアにし、$W_k\Sigma_k$ に相当する係数を左隣へ送ることもできる。

### 9.2 標準法がQRを使う理由

正準化フェーズでは特異値が不要であり、情報も捨てない。QRはこの目的に直接対応し、一般にSVDより低コストである。

$$
\text{QR phase}
=
\text{gaugeを整えるexact変換},
$$

$$
\text{SVD phase}
=
\text{誤差予算に基づく近似}.
$$

役割を分けることで、どこで誤差が発生したかが明確になる。

### 9.3 最初のSVDでも切ればよいか

右から左への最初のSVDで特異値を切ること自体は可能である。しかし、その時点では反対側の環境が正準化されておらず、裸の局所尾部

$$
\sum_{j>r}\sigma_j^2
$$

をそのままTT全体のFrobenius誤差とは解釈できない。これは別の近似アルゴリズムであり、標準TT-roundingの誤差保証をそのまま流用できない。詳細は [[49_TT-roundingの環境行列と誤差直交分解]] で扱う。

---

## 10. 鏡映したsweepも可能

標準的な向きは

$$
\text{右から左へQR}
\quad\longrightarrow\quad
\text{左から右へSVD}
$$

である。

左右をすべて入れ替えた

$$
\text{左から右へQR}
\quad\longrightarrow\quad
\text{右から左へSVD}
$$

も同じ考え方で構成できる。この場合は、最初にleft-canonical formを作り、反対向きのSVDで右正準コアを順に残す。

重要なのは、正準化と打ち切りを逆向きに進め、各打ち切り位置で左右の環境が等長になるようにすることである。

---

## 11. 一回のrounding後に再びQRするか

標準TT-roundingの一回の呼び出しは

$$
\text{右から左へのQR}
\longrightarrow
\text{左から右へのtruncated SVD}
\longrightarrow
\text{終了}
$$

である。

SVD sweep中に残す $U_k$ が第1コアから第 $d-1$ コアまでをleft-canonicalにする。最後のコアがorthogonality centerとなるため、その出力を返すだけなら追加QRは不要である。

次の処理がright-canonical formを要求する場合、別の演算で正準性が崩れた場合、または同じTTへ再びroundingする場合には、その時点で必要な向きの正準化を行う。

---

## 12. TT-roundingとDMRGのsweepは目的が違う

TT-roundingは、与えられたテンソルを指定誤差内で再表現する圧縮である。局所目的関数を反復最適化するのではないため、通常はQR片道とSVD逆向きで終わる。

DMRGはHamiltonianに対するエネルギーなどを下げる変分最適化である。one-siteまたはtwo-siteの有効問題を解き、環境を更新しながら左右sweepを収束まで繰り返す。DMRGで使うSVDは、最適化後の二サイトテンソルの再分解、中心移動、必要に応じたbond truncationを担う。

したがって、両者はQR・SVD・sweepという外見が似ていても、

$$
\text{TT-rounding}
:
\text{固定されたテンソルの誤差制御付き圧縮},
$$

$$
\text{DMRG}
:
\text{状態を変えながら目的関数を下げる反復最適化}
$$

という違いがある。

---

## 13. 最終まとめ

### 入力と出力

$$
\mathcal X
=
\llbracket G^{(1)},\ldots,G^{(d)}\rrbracket
\quad\longrightarrow\quad
\widetilde{\mathcal X}
=
\llbracket\widetilde G^{(1)},\ldots,\widetilde G^{(d)}\rrbracket.
$$

物理shapeは変えず、内部rankを下げる。

### QR phase

$$
H_k^T
=
Q_kR_k,
\qquad
\widehat G^{(k)}
=
\operatorname{reshape}
(Q_k^T,(q_k,n_k,r_k)).
$$

$R_k^T$ は左隣へ吸収し、テンソルを変えずにright-canonical formを作る。

### SVD phase

$$
V_k
=
U_k\Sigma_kW_k^T,
$$

$$
\widetilde G^{(k)}
=
\operatorname{reshape}
(U_{k,\mathrm{keep}},(r_{k-1},n_k,\widetilde r_k)).
$$

$\Sigma_{k,\mathrm{keep}}W_{k,\mathrm{keep}}^T$ は右隣へ吸収する。

### 次に必要な理論

この2方向sweepによって局所SVDの誤差を全体誤差として扱える理由は、左右環境の等長性と誤差部分空間の直交性にある。これを [[49_TT-roundingの環境行列と誤差直交分解]] で導出する。
