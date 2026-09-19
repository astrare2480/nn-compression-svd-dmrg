---
title: Tucker分解とHOSVD
aliases:
  - Tucker decomposition
  - HOSVD
  - partial Tucker
  - Tucker rank
tags:
  - Tucker
  - HOSVD
  - Tensor
  - 低ランク近似
  - NN圧縮
---

# Tucker分解とHOSVD

## サマリー

Tucker分解は、Tensorを**core Tensorと各modeのfactor行列**で表す方法である。

$N$階Tensorを

$$
\mathcal X
\in
\mathbb R^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

とする。

全modeを圧縮するTucker近似は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0 U^{(0)}
\times_1 U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

である。

factorは

$$
U^{(n)}\in\mathbb R^{I_n\times R_n},
$$

coreは

$$
\mathcal G
\in
\mathbb R^{R_0\times R_1\times\cdots\times R_{N-1}}
$$

となる。

HOSVDは、各modeの元Tensor unfoldingへSVDを行い、上位左特異ベクトルからfactorを作る。

---

## 1. exact multilinear rankと指定Tucker rankを区別する

Tensor $\mathcal X$ 自身のmultilinear rankは、各mode unfoldingの行列rankを並べたものとして

$$
\boxed{
\operatorname{rank}_{\mathrm{ML}}(\mathcal X)
=
\left(
\operatorname{rank}(X_{(0)}),
\operatorname{rank}(X_{(1)}),
\ldots,
\operatorname{rank}(X_{(N-1)})
\right)
}
$$

と定義する。

一方、近似計算でこちらが選ぶ

$$
(R_0,R_1,\ldots,R_{N-1})
$$

は**target Tucker ranks**である。

Tucker近似

$$
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\cdots
\times_{N-1}U^{(N-1)}
$$

では、mode $n$ unfoldingのrankはfactor列数を超えないため

$$
\boxed{
\operatorname{rank}(\hat X_{(n)})
\le R_n
}
$$

となる。

したがって、学習ノートで $(R_0,\ldots,R_{N-1})$ を「Tucker rank」と呼ぶ場合は、厳密には**指定する上限rank / target multilinear rank**として読む。

---

## 2. factorとcoreのshape

### factorの列が基底、rank添字がその係数を表す

この節ではHOSVD・HOOIで使う、列が正規直交したfactorを考える。
「rank空間」は元Tensorに新しい物理的な軸を加えるという意味ではなく、
選んだ $R_n$ 本の基底に対する係数を並べる座標空間である。

一つのmodeに注目して $I=I_n,R=R_n$ と略記し、factorの列を

$$
U=\begin{pmatrix}u_1&\cdots&u_R\end{pmatrix},
\qquad u_\alpha\in\mathbb R^I
$$

と書く。長さ $R$ の係数 $c$ を元の空間へ展開する操作は

$$
\begin{aligned}
x_{\mathrm{keep}}=Uc
&=\begin{pmatrix}u_1&\cdots&u_R\end{pmatrix}
\begin{pmatrix}c_1\\\vdots\\c_R\end{pmatrix}\\
&=\sum_{\alpha=1}^{R}c_\alpha u_\alpha,
\qquad
(x_{\mathrm{keep}})_i=\sum_{\alpha=1}^{R}U_{i,\alpha}c_\alpha.
\end{aligned}
$$

である。$i$ は元の $I$ 個の成分の番号、$\alpha$ は保持基底の $R$ 個の番号である。
factorが $I\times R$ である理由は、「一つの基底を元空間の $I$ 成分で書き、その基底を $R$ 本並べる」ためである。
保持する基底を増減しても、元modeの意味や出力の $I$ 個の成分は変わらない。

factor

$$
U^{(n)}
\in
\mathbb R^{I_n\times R_n}
$$

はrank空間から元mode空間への写像である。

$$
U^{(n)}:
\mathbb R^{R_n}
\rightarrow
\mathbb R^{I_n}
$$

転置は元空間からrank空間への射影に使う。

$$
U^{(n)\mathsf T}:
\mathbb R^{I_n}
\rightarrow
\mathbb R^{R_n}
$$

列直交なら

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}.
$$

### 転置は座標を取り出し、元空間内の射影は $UU^{\mathsf T}$ が行う

列の正規直交性 $u_\alpha^{\mathsf T}u_\beta=\delta_{\alpha,\beta}$ を使うと、
既に保持部分空間内にある $Uc$ からは

$$
[U^{\mathsf T}(Uc)]_\alpha
=\sum_{\beta=1}^{R}(u_\alpha^{\mathsf T}u_\beta)c_\beta
=\sum_{\beta=1}^{R}\delta_{\alpha,\beta}c_\beta
=c_\alpha
$$

として座標を正確に取り出せる。しかし一般の $x\in\mathbb R^I$ には保持部分空間に直交する成分もある。
その場合は

$$
c_\alpha=u_\alpha^{\mathsf T}x=\sum_{i=1}^{I}U_{i,\alpha}x_i,
\qquad c=U^{\mathsf T}x,\qquad
x_{\mathrm{keep}}=Uc=UU^{\mathsf T}x
$$

となる。残差 $e=x-Uc$ は

$$
U^{\mathsf T}e
=U^{\mathsf T}x-U^{\mathsf T}Uc=c-c=0
$$

なので、保持した全ての基底に直交する。
従って $U^{\mathsf T}$ は圧縮後の**座標抽出**、$P=UU^{\mathsf T}$ は元空間内の**直交射影**と区別する。
$U^{\mathsf T}U=I_R$ から $UU^{\mathsf T}=I_I$ は導けない。
$R<I$ なら全ての元ベクトルを保持できず、展開が座標抽出の逆になるのは保持部分空間に限られる。

例えばこの節独自の $I=3,R=2$ の例として

$$
U=\begin{pmatrix}
1/\sqrt2&0\\
1/\sqrt2&0\\
0&1
\end{pmatrix},
\qquad
x=\begin{pmatrix}3\\1\\2\end{pmatrix}
$$

を取る。座標抽出と展開を全成分で計算すると

$$
\begin{aligned}
U^{\mathsf T}x
&=\begin{pmatrix}1/\sqrt2&1/\sqrt2&0\\0&0&1\end{pmatrix}
\begin{pmatrix}3\\1\\2\end{pmatrix}\\
&=\begin{pmatrix}3/\sqrt2+1/\sqrt2\\2\end{pmatrix}
=\begin{pmatrix}2\sqrt2\\2\end{pmatrix},\\
UU^{\mathsf T}x
&=\begin{pmatrix}(2\sqrt2)/\sqrt2\\(2\sqrt2)/\sqrt2\\2\end{pmatrix}
=\begin{pmatrix}2\\2\\2\end{pmatrix},\\
x-UU^{\mathsf T}x&=\begin{pmatrix}1\\-1\\0\end{pmatrix}.
\end{aligned}
$$

この残差と2本の保持基底との内積はどちらも0である。両方のGramを全要素表示すると

$$
U^{\mathsf T}U=\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
UU^{\mathsf T}=\begin{pmatrix}1/2&1/2&0\\1/2&1/2&0\\0&0&1\end{pmatrix}
\ne I_3.
$$

第1・第2成分の差は失われ、その平均だけが残る。shapeが3へ戻ることと、元の $(3,1,2)$ が戻ることは別である。
このベクトルの操作を、Tensorの各modeに沿った全てのベクトルへ適用するのが以下のcore計算である。
一般の非直交factorでは、転置だけで座標抽出や最適core計算ができるとは限らない。以下は列正規直交という前提の式である。

全mode Tuckerのcoreは

$$
\boxed{
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
}
$$

である。

各mode productにより

$$
I_n\rightarrow R_n
$$

と変わるので、最終的に

$$
\mathcal G
\in
\mathbb R^{R_0\times\cdots\times R_{N-1}}
$$

となる。

再構成は逆方向に

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

で、各modeが

$$
R_n\rightarrow I_n
$$

へ戻る。

### coreは「各modeの保持基底で測った係数」をまとめたTensor

3階Tensorなら、元添字は $i_0,i_1,i_2$、保持基底の添字は $\alpha_0,\alpha_1,\alpha_2$ とする。
まず $i_1,i_2$ を固定し、$i_0$ だけが動くベクトルから座標を取り出す。
次に $i_1$、最後に $i_2$ に沿うベクトルへ同じ操作をするので

$$
\begin{aligned}
H_{\alpha_0,i_1,i_2}
&=\sum_{i_0=1}^{I_0}U^{(0)}_{i_0,\alpha_0}X_{i_0,i_1,i_2},
&&\operatorname{shape}(H)=(R_0,I_1,I_2),\\
K_{\alpha_0,\alpha_1,i_2}
&=\sum_{i_1=1}^{I_1}U^{(1)}_{i_1,\alpha_1}H_{\alpha_0,i_1,i_2},
&&\operatorname{shape}(K)=(R_0,R_1,I_2),\\
G_{\alpha_0,\alpha_1,\alpha_2}
&=\sum_{i_2=1}^{I_2}U^{(2)}_{i_2,\alpha_2}K_{\alpha_0,\alpha_1,i_2},
&&\operatorname{shape}(G)=(R_0,R_1,R_2).
\end{aligned}
$$

足した添字 $i_n$ は結果から消え、そのmodeの基底番号 $\alpha_n$ が残る。
「軸を削除する」のではなく、元成分の番号を保持座標の番号へ置き換えている。
例えば $R_1\ge2$ なら、$G_{1,2,1}$ は、第0modeの第1基底、第1modeの第2基底、第2modeの第1基底を組み合わせた係数である。

その係数を元成分へ展開すると

$$
\hat X_{i_0,i_1,i_2}
=\sum_{\alpha_0=1}^{R_0}\sum_{\alpha_1=1}^{R_1}\sum_{\alpha_2=1}^{R_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
G_{\alpha_0,\alpha_1,\alpha_2}.
$$

coreの値が $G_{1,2,1}$ なら、その値を対応する三つの基底の積へ掛け、全基底の組合せを足して元shapeの成分を作る。
coreは元Tensorから小さい範囲を切り出した部分配列ではない。基底との内積で作られる新しい係数配列である。
第4節ではこの操作をテンソル全体の射影と具体的なsliceまで展開する。

---

## 3. HOSVDのfactorを1 modeずつ作る

mode $n$ unfoldingは

$$
X_{(n)}
\in
\mathbb R^{
I_n\times\prod_{m\ne n}I_m
}.
$$

Reduced SVDを

$$
X_{(n)}
=
Q_n\Sigma_nV_n^{\mathsf T}
$$

とする。

左特異ベクトルを

$$
Q_n
=
\begin{bmatrix}
q^{(n)}_1&q^{(n)}_2&\cdots
\end{bmatrix}
$$

と書けば、HOSVD factorは

$$
\boxed{
U^{(n)}
=
\begin{bmatrix}
q^{(n)}_1&q^{(n)}_2&\cdots&q^{(n)}_{R_n}
\end{bmatrix}
}
$$

である。

Pythonの0始まりsliceなら

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-003]]

となる。

`[:, 1:R_n]` では第1列を落としてしまうので別物である。

### HOSVDで重要な点

factorを求めるときは、それぞれ**元の同じ $\mathcal X$** をunfoldする。

```text
X → unfold mode 0 → SVD → U^(0)
X → unfold mode 1 → SVD → U^(1)
X → unfold mode 2 → SVD → U^(2)
...
```

先に求めたfactorで $X$ を射影してから次factorを計算するのではない。

この「各modeを独立に初期化する」点がHOOIとの大きな違いである。

---

## 4. HOSVDのcoreと再構成

factorをすべて得た後、元Tensorをfactor転置で射影する。

$$
\begin{aligned}
\mathcal X
&\xrightarrow{\times_0U^{(0)\mathsf T}}
\mathcal X^{(1)}\\
&\xrightarrow{\times_1U^{(1)\mathsf T}}
\mathcal X^{(2)}\\
&\quad\vdots\\
&\xrightarrow{\times_{N-1}U^{(N-1)\mathsf T}}
\mathcal G.
\end{aligned}
$$

したがって

$$
\boxed{
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots
\times_{N-1}U^{(N-1)\mathsf T}
}
$$

である。

再構成は

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
}
$$

となる。

### 一般の3階Tensorの射影・再構成・rank上限を成分で導く

具体例へ進む前に、一般の3階Tensorでも各射影の途中配列を明示する。要素添字は1始まりとする。

$$
\begin{aligned}
H^{(0)}_{\alpha_0,i_1,i_2}
&=\sum_{i_0=1}^{I_0}
(U^{(0)\mathsf T})_{\alpha_0,i_0}X_{i_0,i_1,i_2}
=\sum_{i_0=1}^{I_0}U^{(0)}_{i_0,\alpha_0}X_{i_0,i_1,i_2},\\
H^{(1)}_{\alpha_0,\alpha_1,i_2}
&=\sum_{i_1=1}^{I_1}U^{(1)}_{i_1,\alpha_1}
H^{(0)}_{\alpha_0,i_1,i_2},\\
G_{\alpha_0,\alpha_1,\alpha_2}
&=\sum_{i_2=1}^{I_2}U^{(2)}_{i_2,\alpha_2}
H^{(1)}_{\alpha_0,\alpha_1,i_2}\\
&=\sum_{i_0=1}^{I_0}\sum_{i_1=1}^{I_1}\sum_{i_2=1}^{I_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}
U^{(2)}_{i_2,\alpha_2}X_{i_0,i_1,i_2}.
\end{aligned}
$$

逆向きの展開では、

$$
\begin{aligned}
J^{(0)}_{i_0,\alpha_1,\alpha_2}
&=\sum_{\alpha_0=1}^{R_0}U^{(0)}_{i_0,\alpha_0}
G_{\alpha_0,\alpha_1,\alpha_2},\\
J^{(1)}_{i_0,i_1,\alpha_2}
&=\sum_{\alpha_1=1}^{R_1}U^{(1)}_{i_1,\alpha_1}
J^{(0)}_{i_0,\alpha_1,\alpha_2},\\
\hat X_{i_0,i_1,i_2}
&=\sum_{\alpha_2=1}^{R_2}U^{(2)}_{i_2,\alpha_2}
J^{(1)}_{i_0,i_1,\alpha_2}\\
&=\sum_{\alpha_0=1}^{R_0}\sum_{\alpha_1=1}^{R_1}\sum_{\alpha_2=1}^{R_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}
U^{(2)}_{i_2,\alpha_2}G_{\alpha_0,\alpha_1,\alpha_2}.
\end{aligned}
$$

さらにcoreの式を代入して、物理添字の和とrank添字の和をまとめ直すと、

$$
\begin{aligned}
\hat X_{i_0,i_1,i_2}
&=\sum_{j_0,j_1,j_2}
\left(\sum_{\alpha_0}U^{(0)}_{i_0,\alpha_0}U^{(0)}_{j_0,\alpha_0}\right)
\left(\sum_{\alpha_1}U^{(1)}_{i_1,\alpha_1}U^{(1)}_{j_1,\alpha_1}\right)
\left(\sum_{\alpha_2}U^{(2)}_{i_2,\alpha_2}U^{(2)}_{j_2,\alpha_2}\right)
X_{j_0,j_1,j_2}\\
&=\sum_{j_0,j_1,j_2}
(P_0)_{i_0,j_0}(P_1)_{i_1,j_1}(P_2)_{i_2,j_2}
X_{j_0,j_1,j_2},
\qquad
P_n=U^{(n)}U^{(n)\mathsf T}.
\end{aligned}
$$

したがって、再構成は各modeの保持部分空間への射影であり、shapeが元へ戻ることだけではexact再構成を意味しない。各 $P_nX_{(n)}=X_{(n)}$、すなわち必要な列空間をすべて保持していればexactになる。

また、列側を辞書順に並べたmode 0 unfoldingでは、

$$
\hat X_{(0)}
=U^{(0)}G_{(0)}(U^{(1)}\otimes U^{(2)})^{\mathsf T}
$$

である。右側の積の成分を確認すると、

$$
\begin{aligned}
[\hat X_{(0)}]_{i_0,(i_1,i_2)}
&=\sum_{\alpha_0,\alpha_1,\alpha_2}
U^{(0)}_{i_0,\alpha_0}
[G_{(0)}]_{\alpha_0,(\alpha_1,\alpha_2)}
[(U^{(1)}\otimes U^{(2)})^{\mathsf T}]_{(\alpha_1,\alpha_2),(i_1,i_2)}\\
&=\sum_{\alpha_0,\alpha_1,\alpha_2}
U^{(0)}_{i_0,\alpha_0}G_{\alpha_0,\alpha_1,\alpha_2}
U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
=\hat X_{i_0,i_1,i_2}.
\end{aligned}
$$

この因子化により、

$$
\operatorname{rank}(\hat X_{(0)})
\le\operatorname{rank}(U^{(0)})
\le R_0
$$

となる。他modeも対象modeを行へ出して同様に導けるので、第1節のrank上限は単なるshapeの推測ではない。

### 小さい3階Tensorでfactor・core・再構成を全要素確認する

各modeの次元が2である3階Tensorを、2枚のsliceで

$$
X_{:,:,0}
=
\begin{pmatrix}
3&0\\
0&0
\end{pmatrix},
\qquad
X_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix}
$$

とする。すなわち、非零要素は

$$
X_{0,0,0}=3,
\qquad
X_{1,1,1}=1
$$

だけである。各mode unfoldingで、列側の複合添字を辞書順に並べると

$$
X_{(0)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix},
\quad
X_{(1)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix},
\quad
X_{(2)}
=
\begin{pmatrix}
3&0&0&0\\
0&0&0&1
\end{pmatrix}.
$$

この例では3つのunfoldingが同じ数値行列になるが、それぞれ異なるmodeを行indexにしている。どのmodeでも

$$
X_{(n)}X_{(n)}^{\mathsf T}
=
\begin{pmatrix}
9&0\\
0&1
\end{pmatrix}
$$

なので、特異値は $3,1$、左特異ベクトルは標準基底である。指定Tucker rankを $(1,1,1)$ とすると、HOSVDは各modeを**元のTensorから独立に**計算して

$$
U^{(0)}
=
U^{(1)}
=
U^{(2)}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}
$$

を得る。coreは1要素だけになり、

$$
\begin{aligned}
G_{0,0,0}
&=
\sum_{i_0=0}^{1}
\sum_{i_1=0}^{1}
\sum_{i_2=0}^{1}
U^{(0)}_{i_0,0}
U^{(1)}_{i_1,0}
U^{(2)}_{i_2,0}
X_{i_0,i_1,i_2}\\
&=
1\cdot1\cdot1\cdot X_{0,0,0}\\
&=3.
\end{aligned}
$$

したがって再構成は

$$
\hat X_{:,:,0}
=
\begin{pmatrix}
3&0\\
0&0
\end{pmatrix},
\qquad
\hat X_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix}.
$$

残差も全要素で書けば

$$
(X-\hat X)_{:,:,0}
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix},
\qquad
(X-\hat X)_{:,:,1}
=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix},
$$

よって

$$
\lVert X-\hat X\rVert_F
=
\sqrt{0^2+1^2}
=1
$$

となる。この例では、一般式の「各mode unfoldingからfactorを作る」「全factorを得た後にcoreへ射影する」「coreから元のshapeへ戻す」という3段階を、全要素で追跡できる。

---

## 5. Partial HOSVD / Tucker-2

「partial」はfactorを作るmodeを選ぶという意味である。圧縮しないmodeを削除する意味ではない。
「Tucker-2」の2も、Tensorが2階になることではなく、ここではfactor化するmodeが2本であることを表す。
Conv weightならchannelの二つの軸をrank座標にし、kernel高さ・幅の番号はそのままcoreに残す。
例えば一つのkernel位置 $(a,b)$ を固定すると、$W_{:,:,a,b}$ はchannelの行列、$G_{:,:,a,b}$ はその圧縮後の座標行列である。
位置を変えるたびに異なるcore行列を持つが、channel基底のfactorは全kernel位置で共通に使う。

全modeを圧縮する必要はない。

圧縮対象mode集合を

$$
\mathcal M
\subseteq
\{0,1,\ldots,N-1\}
$$

とする。

partial Tuckerでは

$$
\boxed{
\mathcal G
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
U^{(m)\mathsf T}
}
$$

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\underset{m\in\mathcal M}{\times_m}
U^{(m)}
}
$$

となり、$m\notin\mathcal M$ のmodeは元shapeのままcoreに残る。

Conv2d weight

$$
W
\in
\mathbb R^{
C_{out}\times C_{in}\times K_h\times K_w
}
$$

でchannel mode 0 / 1だけを圧縮するTucker-2では

$$
\mathcal M=\{0,1\}.
$$

factorは

$$
U_{out}
\in
\mathbb R^{C_{out}\times R_{out}},
$$

$$
U_{in}
\in
\mathbb R^{C_{in}\times R_{in}}.
$$

coreは

$$
\boxed{
G
=
W
\times_0U_{out}^{\mathsf T}
\times_1U_{in}^{\mathsf T}
}
$$

で、shapeは

$$
\boxed{
G
\in
\mathbb R^{R_{out}\times R_{in}\times K_h\times K_w}
}.
$$

再構成は

$$
\boxed{
\hat W
=
G
\times_0U_{out}
\times_1U_{in}
}.
$$

---

## 6. Tucker-2の要素表示をmode productから出す

まずmode 0を展開すると

$$
H_{o,\beta,a,b}
=
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}.
$$

次にmode 1へ $U_{in}$ を掛けると

$$
\hat W_{o,i,a,b}
=
\sum_{\beta=1}^{R_{in}}
H_{o,\beta,a,b}
U^{in}_{i,\beta}.
$$

$H$ を代入して

$$
\begin{aligned}
\hat W_{o,i,a,b}
&=
\sum_{\beta=1}^{R_{in}}
\left(
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
\right)
U^{in}_{i,\beta}\\
&=
\boxed{
\sum_{\alpha=1}^{R_{out}}
\sum_{\beta=1}^{R_{in}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
}.
\end{aligned}
$$

---

## 7. 再構成誤差

Tensor Frobenius normは

$$
\boxed{
\|\mathcal X\|_F
=
\sqrt{
\sum_{i_0=1}^{I_0}
\sum_{i_1=1}^{I_1}
\cdots
\sum_{i_{N-1}=1}^{I_{N-1}}
X_{i_0,\ldots,i_{N-1}}^2
}
}
$$

である。

relative errorは

$$
\boxed{
e_{rel}
=
\frac{
\|\mathcal X-\hat{\mathcal X}\|_F
}{
\|\mathcal X\|_F
}
}
$$

で、Conv weightなら

$$
\boxed{
e_W
=
\frac{
\|W-\hat W\|_F
}{
\|W\|_F
}
}.
$$

これは元weightへの近さを測る指標であって、classification lossを直接測るものではない。

---

## 8. Tuckerのparameter数

全modeを圧縮する場合、coreの要素数は

$$
P_{core}
=
\prod_{n=0}^{N-1}R_n.
$$

factor $n$ の要素数は

$$
P_n
=I_nR_n.
$$

したがって合計は

$$
\begin{aligned}
P_{Tucker}
&=P_{core}+\sum_{n=0}^{N-1}P_n\\
&=
\boxed{
\prod_{n=0}^{N-1}R_n
+
\sum_{n=0}^{N-1}I_nR_n
}.
\end{aligned}
$$

partial Tuckerでは、圧縮しないmodeの次元がcoreにそのまま残るため、この全mode用の式を機械的に使わず、実際のcore shapeから数える。

Conv Tucker-2の式は [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]] で導出する。

---

## 9. HOSVDは行列SVDと同じ意味でのglobal optimumではない

行列のtruncated SVDでは、Eckart–Young–Mirskyにより固定rankの最良近似を得る。

HOSVDでは各modeを独立に最適化するため、高階Tensorの固定target multilinear rank

$$
(R_0,\ldots,R_{N-1})
$$

に対する全体最適解そのものとは限らない。

```text
行列 truncated SVD
→ 固定rankに対するglobal best approximation

HOSVD
→ mode-wise SVDを組み合わせた高速な初期近似

HOOI
→ HOSVD factorを初期値にして同rankで交互精密化
```

> [!note] 補足一般理論
> HOSVDが一般にglobal optimumを保証しないことは、Tucker近似の理論上の性質である。HOOIによるFrobenius errorの変化は、理論上の保証と実験での観察を区別して評価する。

---

## 10. srcとの対応

```text
src/nn_compression/compression/tucker.py
├─ hosvd
└─ reconstruct_tucker

src/nn_compression/metrics/tensor_approximation.py
└─ relative_frobenius_error
```

実装上の重要点は、HOSVD factorを各modeについて**元のXから独立に**求めること。

詳細な途中式：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
