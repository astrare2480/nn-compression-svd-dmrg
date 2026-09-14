---
title: SVDとは
aliases:
  - 特異値分解
  - Singular Value Decomposition
tags:
  - 線形代数
  - SVD
  - 低ランク近似
  - NN圧縮
---

# SVDとは

## サマリー

SVD（Singular Value Decomposition、特異値分解）は、**任意の実行列を、入力側の直交方向・方向ごとの拡大率・出力側の直交方向へ分解する方法**である。

行列

$$
W\in\mathbb R^{m\times n}
$$

が表す写像

$$
W:\mathbb R^n\to\mathbb R^m
$$

を

$$
\boxed{
W=U\Sigma V^{\mathsf T}
}
$$

と分解する。

右から順に

```text
V^T : 入力を右特異ベクトル基底で表す
Σ   : 各特異方向をσ_i倍する
U   : 左特異ベクトル方向へ写す
```

と読める。

小さい特異値に対応する成分を捨てると、固定rankの範囲で元行列をよく近似できる。この低rank近似がNN weight圧縮の基本になる。

---

## 1. SVDが扱う対象

$$
W\in\mathbb R^{m\times n},
\qquad
x\in\mathbb R^n
$$

に対し、

$$
y=Wx\in\mathbb R^m
$$

である。

SVDは $m=n$ の正方行列に限らず、$m\ne n$ の長方形行列にも適用できる。`nn.Linear` weightは一般に長方形なので、この性質が重要である。

---

## 2. 完全SVD

完全SVDでは

$$
W=U\Sigma V^{\mathsf T}
$$

で、

$$
U\in\mathbb R^{m\times m},
\qquad
\Sigma\in\mathbb R^{m\times n},
\qquad
V\in\mathbb R^{n\times n}
$$

である。

$U,V$ は直交行列なので、

$$
U^{\mathsf T}U=UU^{\mathsf T}=I_m
$$

$$
V^{\mathsf T}V=VV^{\mathsf T}=I_n
$$

を満たす。

$U$ の列を $u_i$、$V$ の列を $v_i$ と書く。

$$
U=\begin{bmatrix}u_1&u_2&\cdots&u_m\end{bmatrix}
$$

$$
V=\begin{bmatrix}v_1&v_2&\cdots&v_n\end{bmatrix}
$$

特異値は

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_k\ge0,
\qquad
k=\min(m,n)
$$

と並べる。

完全SVDのshapeは

$$
(m\times m)(m\times n)(n\times n)
=m\times n
$$

で元の $W$ と一致する。

---

## 3. Reduced SVD

数値計算・NN圧縮ではReduced SVDを使うことが多い。

$$
\boxed{
W=U_k\Sigma_kV_k^{\mathsf T}
}
$$

$$
U_k\in\mathbb R^{m\times k},
\qquad
\Sigma_k\in\mathbb R^{k\times k},
\qquad
V_k\in\mathbb R^{n\times k}
$$

なので、

$$
(m\times k)(k\times k)(k\times n)
=m\times n.
$$


元資料 `01_SVDとは.md` の157–177行で図示した特異値の対角部分を、
本章のreduced SVDの記号へ対応させて残す。

$$
\Sigma_k=\begin{pmatrix}
\sigma_1&0&\cdots&0\\
0&\sigma_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\sigma_k
\end{pmatrix}\in\mathbb R^{k\times k}.
$$

完全SVDの $\Sigma$ は一般に $m\times n$ の長方形なので、
この $k\times k$ の対角部分に、必要な零行・零列を補ったものになる。
元資料の図を、完全SVDの長方形全体が必ず正方対角行列であるという意味には読まない。
PyTorchの

```python
torch.linalg.svd(W, full_matrices=False)
```

はこの形に対応し、`S` は特異値の1次元Tensor、`Vh` は $V_k^{\mathsf T}$ を返す。

詳細なshapeを含む途中導出は [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]] を参照する。

---

### 長方形行列で完全SVDとreduced SVDを全要素比較する

shapeだけでなく、余分な列・零行がどこにあるかも確認する。本節の補足例を

$$
W=\begin{pmatrix}0&0\\6&0\\0&2\end{pmatrix}
\in\mathbb R^{3\times2}
$$

とする。完全SVDの一つは

$$
U=\begin{pmatrix}0&0&1\\1&0&0\\0&1&0\end{pmatrix},
\qquad
\Sigma=\begin{pmatrix}6&0\\0&2\\0&0\end{pmatrix},
\qquad
V=I_2=\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$$

全要素を掛けると、

$$
\begin{aligned}
U\Sigma V^{\mathsf T}
&=\begin{pmatrix}
0\cdot6+0\cdot0+1\cdot0&0\cdot0+0\cdot2+1\cdot0\\
1\cdot6+0\cdot0+0\cdot0&1\cdot0+0\cdot2+0\cdot0\\
0\cdot6+1\cdot0+0\cdot0&0\cdot0+1\cdot2+0\cdot0
\end{pmatrix}I_2\\
&=\begin{pmatrix}0&0\\6&0\\0&2\end{pmatrix}=W.
\end{aligned}
$$

reduced SVDでは $k=\min(3,2)=2$ として、

$$
U_k=\begin{pmatrix}0&0\\1&0\\0&1\end{pmatrix},
\qquad
\Sigma_k=\begin{pmatrix}6&0\\0&2\end{pmatrix},
\qquad
V_k=I_2,
\qquad
U_k\Sigma_kV_k^{\mathsf T}=W.
$$

完全SVDの第3列と $\Sigma$ の零行は積に寄与しないので、除いても同じ $W$ を再構成できる。ただし

$$
U_k^{\mathsf T}U_k=I_2,
\qquad
U_kU_k^{\mathsf T}=\begin{pmatrix}0&0&0\\0&1&0\\0&0&1\end{pmatrix}\ne I_3
$$

であり、長方形の列直交行列が両側で単位行列になるわけではない。

rank-1の和も、この同じ例なら

$$
\begin{aligned}
W
&=6\begin{pmatrix}0\\1\\0\end{pmatrix}\begin{pmatrix}1&0\end{pmatrix}
+2\begin{pmatrix}0\\0\\1\end{pmatrix}\begin{pmatrix}0&1\end{pmatrix}\\
&=\begin{pmatrix}0&0\\6&0\\0&0\end{pmatrix}
+\begin{pmatrix}0&0\\0&0\\0&2\end{pmatrix}.
\end{aligned}
$$

因子のshape・零成分・rank-1外積が、一般式と全要素で対応している。

---

## 4. 特異値と特異ベクトルの意味

ここでは第3節のreduced SVDを $W=U_k\Sigma_kV_k^{\mathsf T}$ と書き、$1\le i\le k$ とする。$e_i\in\mathbb R^k$ は第 $i$ 成分だけが1の標準基底であり、$u_i=U_ke_i$、$v_i=V_ke_i$ である。列直交性を一段ずつ代入すると、

$$
\begin{aligned}
Wv_i
&=U_k\Sigma_kV_k^{\mathsf T}(V_ke_i)\\
&=U_k\Sigma_k(V_k^{\mathsf T}V_k)e_i\\
&=U_k\Sigma_ke_i\\
&=U_k(\sigma_i e_i)\\
&=\sigma_i u_i.
\end{aligned}
$$

第$i$右特異ベクトルへ $W$ を作用させると、

$$
\boxed{
Wv_i=\sigma_i u_i
}
$$

となる。

したがって、

- $v_i$：入力空間 $\mathbb R^n$ の特異方向
- $u_i$：対応する出力空間 $\mathbb R^m$ の特異方向
- $\sigma_i$：その方向の拡大率

である。

$\sigma_i=0$ なら

$$
Wv_i=0
$$

となり、その入力方向はkernel（零空間）へ写る。

同じreduced SVDでは $\Sigma_k$ が正方対角行列なので、逆方向にも

$$
\begin{aligned}
W^{\mathsf T}u_i
&=(U_k\Sigma_kV_k^{\mathsf T})^{\mathsf T}(U_ke_i)\\
&=V_k\Sigma_k^{\mathsf T}(U_k^{\mathsf T}U_k)e_i\\
&=V_k\Sigma_ke_i\\
&=\sigma_iV_ke_i\\
&=\sigma_i v_i
\end{aligned}
$$

と導ける。ここで $\Sigma_k^{\mathsf T}=\Sigma_k$ とできるのはreducedの正方対角行列だからであり、完全SVDの長方形 $\Sigma$ とは区別する。

逆方向の関係は

$$
\boxed{
W^{\mathsf T}u_i=\sigma_i v_i
}
$$

である。

完全SVDでは転置を途中から書くと

$$
\boxed{
W^{\mathsf T}
=(U\Sigma V^{\mathsf T})^{\mathsf T}
=V\Sigma^{\mathsf T}U^{\mathsf T}
}
$$

であり、長方形 $\Sigma$ の場合に $\Sigma^{\mathsf T}$ を落としてはいけない。

---

## 5. 幾何学的イメージ

「右から作用する」は、行列を右から左の順に計算するという意味である。
reduced SVDの $W\in\mathbb R^{m\times n}$、$k=\min(m,n)$ では、各途中ベクトルは

$$
c:=V^{\mathsf T}x\in\mathbb R^k,\qquad
d:=\Sigma c\in\mathbb R^k,\qquad
y:=Ud\in\mathbb R^m
$$

となる。$c_i=v_i^{\mathsf T}x$ は入力と第 $i$ 特異方向との内積、
$d_i=\sigma_ic_i$ はその係数を拡大・縮小した値、$U$ は出力基底の線形結合を作る行列である。
「入力を回す→方向ごとに伸ばす→出力を回す」という見方を、係数を取り出す→係数を変える→基底を足す、という計算へ対応させている。

完全SVDの正方な $U,V$ は全空間の長さを保つが、reduced因子の転置は全空間の長さを保つとは限らない。
特に $n>k$ の場合、入力に $V$ の列空間外の成分があれば

$$
x=Vc+x_\perp,\qquad V^{\mathsf T}x_\perp=0,\qquad
\|x\|_2^2=\|c\|_2^2+\|x_\perp\|_2^2
$$

となる。ただし $Wx_\perp=U\Sigma V^{\mathsf T}x_\perp=0$ なので、その座標をreduced表現に持たなくても $Wx$ はexactに計算できる。
これと、非零特異方向まで減らすtruncated SVDによる近似は別である。
以下の直交行列のnorm保存は、正方な完全基底、または列正規直交行列を座標側から作用させる場合の式として読む。

$$
y=Wx=U\Sigma V^{\mathsf T}x
$$

は右から順に作用する。

```text
x
↓
V^T
右特異ベクトル基底での座標
↓
Σ
各方向をσ_i倍
↓
U
出力空間へ写す
↓
y
```

直交行列 $Q$ は長さを保つ。

$$
Q^{\mathsf T}Q=I
$$

より、

$$
\begin{aligned}
\|Qx\|_2^2
&=(Qx)^{\mathsf T}(Qx)\\
&=x^{\mathsf T}Q^{\mathsf T}Qx\\
&=x^{\mathsf T}x\\
&=\|x\|_2^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|Qx\|_2=\|x\|_2
}
$$

であり、SVDで長さを直接変えるのは $\Sigma$ である。

この最後の見方は、完全SVDの直交基底、または既に保持座標内にあるベクトルについての説明である。
reduced SVDの $V^{\mathsf T}$ が一般入力の長さまで保つという意味ではない。先に分けた $x_\perp$ は長さを持っていても $W$ の零空間に属し、出力には寄与しない。

---

### 元教材の「単位円が楕円へ移る」を式で追う

元教材 `01_SVDとは.md` の単位円の図は、入力方向・出力方向・拡大率を区別する説明である。ここでは $m=n=2$ の完全SVDを使い、図の内容に途中式を補う。右特異ベクトルは正規直交なので、

$$
x(\theta)
=
v_1\cos\theta+v_2\sin\theta
=
V
\begin{pmatrix}
\cos\theta\\
\sin\theta
\end{pmatrix}
$$

と書いた点の長さは

$$
\begin{aligned}
\lVert x(\theta)\rVert_2^2
&=
(v_1\cos\theta+v_2\sin\theta)^{\mathsf T}
(v_1\cos\theta+v_2\sin\theta)\\
&=
(v_1^{\mathsf T}v_1)\cos^2\theta
+
2(v_1^{\mathsf T}v_2)\cos\theta\sin\theta
+
(v_2^{\mathsf T}v_2)\sin^2\theta\\
&=
\cos^2\theta+0+\sin^2\theta\\
&=1
\end{aligned}
$$

となる。これを $W$ で写すと、

$$
\begin{aligned}
y(\theta)
&=Wx(\theta)\\
&=
U\Sigma V^{\mathsf T}V
\begin{pmatrix}
\cos\theta\\
\sin\theta
\end{pmatrix}\\
&=
U
\begin{pmatrix}
\sigma_1&0\\
0&\sigma_2
\end{pmatrix}
\begin{pmatrix}
\cos\theta\\
\sin\theta
\end{pmatrix}\\
&=
\sigma_1u_1\cos\theta
+
\sigma_2u_2\sin\theta.
\end{aligned}
$$

出力の左特異基底での座標を $\xi_i=u_i^{\mathsf T}y$ とすると、

$$
\xi_1=\sigma_1\cos\theta,
\qquad
\xi_2=\sigma_2\sin\theta,
\qquad
\frac{\xi_1^2}{\sigma_1^2}
+
\frac{\xi_2^2}{\sigma_2^2}
=
1
\quad(\sigma_1,\sigma_2>0)
$$

となり、楕円の式になる。軸の向きは $u_1,u_2$、対応する入力方向は $v_1,v_2$、**半軸長**は $\sigma_1,\sigma_2$ である。端から端までの軸長なら $2\sigma_i$ なので、元教材の「主軸の長さ」を半軸長として明確化する。$\sigma_2=0$ なら線分、両方0なら一点へ退化し、零の特異値で割った楕円式は使わない。

## 6. 固有値分解との関係

$$
Wv_i=\sigma_i u_i
$$

へ左から $W^{\mathsf T}$ を掛ける。

$$
W^{\mathsf T}Wv_i
=
\sigma_iW^{\mathsf T}u_i.
$$

さらに

$$
W^{\mathsf T}u_i=\sigma_i v_i
$$

を代入すると、

$$
\begin{aligned}
W^{\mathsf T}Wv_i
&=\sigma_i(\sigma_i v_i)\\
&=\sigma_i^2v_i.
\end{aligned}
$$

よって、

$$
\boxed{
W^{\mathsf T}Wv_i=\sigma_i^2v_i
}
$$

である。

同様に、

$$
\begin{aligned}
WW^{\mathsf T}u_i
&=W(W^{\mathsf T}u_i)\\
&=W(\sigma_i v_i)\\
&=\sigma_iWv_i\\
&=\sigma_i(\sigma_i u_i)\\
&=\sigma_i^2u_i.
\end{aligned}
$$

$$
\boxed{
WW^{\mathsf T}u_i=\sigma_i^2u_i
}
$$

となる。

したがって、

```text
W^T W の固有ベクトル = v_i
WW^T  の固有ベクトル = u_i
対応固有値             = σ_i^2
```

である。

特異値は

$$
\sigma_i=\sqrt{\lambda_i}
$$

と固有値の非負平方根として読める。

---

### 元教材の固有値分解との違いと、数値計算上の注意

元教材では、固有値分解とSVDを同じ分解として扱わないことも説明している。固有値分解 $A=P\Lambda P^{-1}$ は正方行列に対する表現であり、行列によってはこの形に対角化できない。固有値は負や複素数にもなり、固有ベクトルも一般には正規直交でない。一方、実行列のSVDは長方形行列にも存在し、入力側 $V$ と出力側 $U$ を分け、特異値は非負になる。

ここで $\sigma_i=\sqrt{\lambda_i}$ とした $\lambda_i$ は、$W$ 自身の固有値ではなく、半正定値行列 $W^{\mathsf T}W$ または $WW^{\mathsf T}$ の対応する固有値である。元教材の比較で重要なのは、この対象の違いである。

また、上の固有値との関係は理論を理解するための式であり、SVDを求める際に必ず $W^{\mathsf T}W$ を明示的に作るという実装手順ではない。元教材の数値誤差の注意を補うため、$m\ge n$ かつ $W$ が列full rankの場合を考える。$\sigma_n>0$ とすると、2-normでの条件数は

$$
\begin{aligned}
\kappa_2(W)
&=
\frac{\sigma_1}{\sigma_n},\\
\lambda_{\max}(W^{\mathsf T}W)
&=\sigma_1^2,\\
\lambda_{\min}(W^{\mathsf T}W)
&=\sigma_n^2,\\
\kappa_2(W^{\mathsf T}W)
&=
\frac{\lambda_{\max}(W^{\mathsf T}W)}
{\lambda_{\min}(W^{\mathsf T}W)}\\
&=
\frac{\sigma_1^2}{\sigma_n^2}\\
&=
\kappa_2(W)^2
\end{aligned}
$$

となる。元から特異値の大小差が大きいと、二乗によって条件がさらに悪化する。ここでの条件数の途中式は元教材の注意を本書で補ったものであり、実装ではライブラリのSVD関数を直接使う。rank欠損なら最小固有値が0なので、この有限な条件数の式をそのまま使わない。

## 7. rankと特異値

第2節の完全SVDを使う。正方直交行列 $U,V$ は可逆なので、左から $U$、右から $V^{\mathsf T}$ を掛けてもrankは変わらない。したがって、

$$
\begin{aligned}
\operatorname{rank}(W)
&=\operatorname{rank}(U\Sigma V^{\mathsf T})\\
&=\operatorname{rank}(\Sigma V^{\mathsf T})\\
&=\operatorname{rank}(\Sigma).
\end{aligned}
$$

長方形の $\Sigma$ の第 $i$ 列は、$i\le k$ なら $\sigma_i e_i^{(m)}$、$i>k$ なら零列である。非ゼロの列は異なる標準基底方向を向いていて独立なので、その本数がrankになる。すなわち、

$$
\operatorname{rank}(\Sigma)
=\dim\operatorname{span}\{e_i^{(m)}\mid 1\le i\le k,\ \sigma_i>0\}
=\#\{i\mid\sigma_i>0\}.
$$

この議論は完全SVDの可逆な基底変換を使っている。reducedの長方形因子を正方直交行列とみなしているわけではない。

行列rankは非零特異値の個数に等しい。

$$
\boxed{
\operatorname{rank}(W)
=
\#\{i\mid\sigma_i>0\}
}
$$

浮動小数点では数値的なthreshold $\varepsilon$ を用いて

$$
\operatorname{rank}_{\mathrm{num}}(W)
=
\#\{i\mid\sigma_i>\varepsilon\}
$$

と考える。

---

### 元教材の特異値 $(8,3,0,0)$ をそのまま数える

元教材のrank確認例は

$$
(\sigma_1,\sigma_2,\sigma_3,\sigma_4)
=
(8,3,0,0)
$$

である。正方な4次元の対角部分として全要素を書くと、

$$
\Sigma=
\begin{pmatrix}
8&0&0&0\\
0&3&0&0\\
0&0&0&0\\
0&0&0&0
\end{pmatrix}
$$

となる。第1列 $8e_1$ と第2列 $3e_2$ は独立、第3・4列は零列なので、

$$
\operatorname{rank}(W)
=
\operatorname{rank}(\Sigma)
=
\#\{1,2\}
=
2
$$

である。これは厳密に零の特異値を含む例である。浮動小数点で零付近の値を判定する数値rankとは区別し、thresholdは行列のscaleとdtypeに応じて選ぶ。

## 8. SVDをrank-1行列の和として見る

第3節のreduced SVDで、$\Sigma_k$ の成分は

$$
(\Sigma_k)_{p,q}=\sigma_p\delta_{p,q},
\qquad
(V_k^{\mathsf T})_{q,b}=(V_k)_{b,q}
$$

である。行列積を成分へ展開すると、

$$
\begin{aligned}
w_{a,b}
&=\sum_{p=1}^{k}\sum_{q=1}^{k}
(U_k)_{a,p}(\Sigma_k)_{p,q}(V_k^{\mathsf T})_{q,b}\\
&=\sum_{p=1}^{k}\sum_{q=1}^{k}
(U_k)_{a,p}\sigma_p\delta_{p,q}(V_k)_{b,q}\\
&=\sum_{p=1}^{k}\sigma_p(u_p)_a(v_p)_b\\
&=\left[\sum_{p=1}^{k}\sigma_pu_pv_p^{\mathsf T}\right]_{a,b}.
\end{aligned}
$$

すべての成分が一致するので、以下の行列等式を得る。$\sigma_p>0$ の項では、各列が同じ $u_p$ の倍数で少なくとも1列は非ゼロだからrank 1であり、$\sigma_p=0$ の項はrank 0である。

Reduced SVDから

$$
\boxed{
W
=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
}
$$

と展開できる。

任意入力 $x$ へ作用させると、

$$
\begin{aligned}
Wx
&=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}x\\
&=
\sum_{i=1}^{k}
\sigma_i
(v_i^{\mathsf T}x)
u_i.
\end{aligned}
$$

すなわち、入力を右特異方向へ射影し、その係数を $\sigma_i$ 倍して左特異方向へ出力する。

---

## 9. truncated SVD

ここで保持する成分数は $0\le r\le k$ とする。$r=0$ は零行列による近似、$r=k$ は全成分を保持する場合である。

上位 $r$ 成分だけ残すと、

$$
\boxed{
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
=
U_r\Sigma_rV_r^{\mathsf T}
}
$$

となる。

$$
U_r\in\mathbb R^{m\times r},
\quad
\Sigma_r\in\mathbb R^{r\times r},
\quad
V_r^{\mathsf T}\in\mathbb R^{r\times n}
$$

なので、

$$
(m\times r)(r\times r)(r\times n)
=m\times n.
$$

また

$$
\operatorname{rank}(W_r)\le r.
$$

この上限も列空間から確認できる。任意の $x\in\mathbb R^n$ に対し、

$$
W_rx
=\sum_{i=1}^{r}\sigma_i(v_i^{\mathsf T}x)u_i
\in\operatorname{span}\{u_1,\ldots,u_r\}.
$$

出力が高々 $r$ 次元の空間に入るのでrankは $r$ 以下である。さらに $W_rv_i=\sigma_i u_i$（$i\le r$）だから、保持した非ゼロ特異値の方向は実際に到達でき、

$$
\operatorname{rank}(W_r)
=\#\{i\mid 1\le i\le r,\ \sigma_i>0\}
\le r
$$

となる。保持する $r$ 個の特異値がすべて正ならrankはちょうど $r$、ゼロを含めて保持した場合はそれより小さい。

---

## 10. Eckart–Young–Mirskyの定理

rankが$r$以下の行列集合に対して、切り詰めSVD $W_r$ はFrobenius normの最小誤差を達成する。

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F
}
$$

spectral normについても、

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
}
$$

である。

ここで `∈ argmin` と書くのは、最適解が常に一意とは限らないためである。特に境界で特異値が重複する場合、同じ最小誤差を達成する別のrank-$r$部分空間が存在し得る。

したがって定理の意味は、

> rank$r$以下の別の行列が、切り詰めSVDより**小さい**行列近似誤差を持つことはない。

ということである。

証明スケッチと途中式は [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出#7. Eckart–Young–Mirskyの最適性を式で追う]] を参照する。

### この節で最適性の下界と達成値を追う

実行列 $W\in\mathbb R^{m\times n}$、$k=\min(m,n)$、
$1\le r<k$ とする。任意の候補 $B$ が $\operatorname{rank}(B)\le r$ を満たすとし、
その列空間への直交射影を $P$ と置く。$PB=B$ なので

$$
\begin{aligned}
W-B&=(I-P)W+(PW-B),\\
\langle(I-P)W,PW-B\rangle_F
&=\operatorname{tr}[W^{\mathsf T}(I-P)(PW-PB)]=0,\\
\|W-B\|_F^2
&=\|(I-P)W\|_F^2+\|PW-B\|_F^2\\
&\ge\|(I-P)W\|_F^2.
\end{aligned}
$$

完全な左特異基底を $u_1,\ldots,u_m$、
$WW^{\mathsf T}$ の固有値を $\lambda_i=\sigma_i^2$（$i\le k$）、$\lambda_i=0$（$i>k$）とする。
$w_i:=u_i^{\mathsf T}Pu_i=\|Pu_i\|_2^2$ は

$$
0\le w_i\le1,
\qquad
\sum_{i=1}^{m}w_i=\operatorname{tr}(P)=\operatorname{rank}(P)\le r
$$

を満たす。$\lambda_1\ge\cdots\ge\lambda_m\ge0$ を使うと

$$
\begin{aligned}
\sum_{i=1}^{m}\lambda_iw_i
&\le\sum_{i=1}^{r}\lambda_iw_i+\lambda_r\sum_{i=r+1}^{m}w_i\\
&\le r\lambda_r+\sum_{i=1}^{r}(\lambda_i-\lambda_r)w_i\\
&\le r\lambda_r+\sum_{i=1}^{r}(\lambda_i-\lambda_r)\\
&=\sum_{i=1}^{r}\lambda_i.
\end{aligned}
$$

従って全ての候補に対して

$$
\begin{aligned}
\|W-B\|_F^2
&\ge\operatorname{tr}[W^{\mathsf T}(I-P)W]\\
&=\sum_{i=1}^{k}\sigma_i^2-\sum_{i=1}^{m}\lambda_iw_i\\
&\ge\sum_{i=r+1}^{k}\sigma_i^2\\
&=\|W-W_r\|_F^2.
\end{aligned}
$$

最後の等号は、残差の直交する外積の二乗和から得られる。
これは候補全体に対する下界を $W_r$ が達成する証明である。

spectral normでは、$\operatorname{span}(v_1,\ldots,v_{r+1})$ は次元 $r+1$、
$B$ のrankは高々 $r$ なので、この部分空間内に $Bz=0$ を満たす単位ベクトル
$z=\sum_{i=1}^{r+1}c_iv_i$ が存在する。$\sum_i c_i^2=1$ とすると

$$
\begin{aligned}
\|W-B\|_2^2
&\ge\|(W-B)z\|_2^2=\|Wz\|_2^2\\
&=\left\|\sum_{i=1}^{r+1}\sigma_ic_iu_i\right\|_2^2\\
&=\sum_{i=1}^{r+1}\sigma_i^2c_i^2\\
&\ge\sigma_{r+1}^2\sum_{i=1}^{r+1}c_i^2\\
&=\sigma_{r+1}^2
=\|W-W_r\|_2^2.
\end{aligned}
$$

$W_r$ のspectral残差は捨てた特異値の最大値 $\sigma_{r+1}$ なので、こちらも下界を達成する。
$r=0$ では候補は零行列だけ、$r\ge k$ では $W$ 自体を候補にして誤差0を達成できる。
この証明は元資料の最良近似という結論を補うもので、task accuracyの最適性を主張しない。

---

## 11. Frobenius normと近似誤差

$$
\|A\|_F
=
\sqrt{
\sum_i\sum_j|a_{ij}|^2
}
$$

である。

### 全要素の二乗和をtraceへ書き換える

この節では実行列を扱う。$A\in\mathbb R^{m\times n}$ に対し、$A^{\mathsf T}A$ の第 $j$ 対角成分は

$$
(A^{\mathsf T}A)_{j,j}
=\sum_{i=1}^{m}(A^{\mathsf T})_{j,i}A_{i,j}
=\sum_{i=1}^{m}a_{i,j}^2
$$

である。対角成分をすべて足せば、

$$
\begin{aligned}
\operatorname{tr}(A^{\mathsf T}A)
&=\sum_{j=1}^{n}(A^{\mathsf T}A)_{j,j}\\
&=\sum_{j=1}^{n}\sum_{i=1}^{m}a_{i,j}^2\\
&=\sum_{i=1}^{m}\sum_{j=1}^{n}|a_{i,j}|^2\\
&=\|A\|_F^2.
\end{aligned}
$$

traceへの書き換えは別のnormを導入したのではなく、同じ全要素の二乗和を行列積の対角和で表しただけである。複素行列では転置を共役転置へ変え、$\|A\|_F^2=\operatorname{tr}(A^\dagger A)$ とする。

### 列直交性から特異値の二乗和まで計算する

第3節のreduced SVDを使い、$U_k^{\mathsf T}U_k=V_k^{\mathsf T}V_k=I_k$ を順に代入する。

$$
\begin{aligned}
W^{\mathsf T}W
&=(U_k\Sigma_kV_k^{\mathsf T})^{\mathsf T}
(U_k\Sigma_kV_k^{\mathsf T})\\
&=V_k\Sigma_k^{\mathsf T}(U_k^{\mathsf T}U_k)\Sigma_kV_k^{\mathsf T}\\
&=V_k\Sigma_k^{\mathsf T}\Sigma_kV_k^{\mathsf T}\\
&=V_k\operatorname{diag}(\sigma_1^2,\ldots,\sigma_k^2)V_k^{\mathsf T}.
\end{aligned}
$$

この行列の第 $b$ 対角成分を足して、

$$
\begin{aligned}
\|W\|_F^2
&=\operatorname{tr}(W^{\mathsf T}W)\\
&=\sum_{b=1}^{n}\sum_{p=1}^{k}(V_k)_{b,p}\sigma_p^2(V_k)_{b,p}\\
&=\sum_{p=1}^{k}\sigma_p^2\left(\sum_{b=1}^{n}(V_k)_{b,p}^2\right)\\
&=\sum_{p=1}^{k}\sigma_p^2(V_k^{\mathsf T}V_k)_{p,p}\\
&=\sum_{p=1}^{k}\sigma_p^2.
\end{aligned}
$$

使ったのは列直交性だけであり、長方形のreduced因子に対して $U_kU_k^{\mathsf T}=I_m$ や $V_kV_k^{\mathsf T}=I_n$ を仮定してはいない。

SVDでは

$$
\boxed{
\|W\|_F^2
=
\sum_{i=1}^{k}\sigma_i^2
}
$$

となる。

切り詰めSVDでは

$$
\boxed{
\|W-W_r\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2
}
$$

である。

### 残差を成分へ展開し、交差項が消えるところを示す

第8・9節の和を引くと、保持した成分が相殺されて、

$$
\begin{aligned}
R:=W-W_r
&=\sum_{p=1}^{k}\sigma_pu_pv_p^{\mathsf T}
-\sum_{p=1}^{r}\sigma_pu_pv_p^{\mathsf T}\\
&=\sum_{p=r+1}^{k}\sigma_pu_pv_p^{\mathsf T},\\
R_{a,b}
&=\sum_{p=r+1}^{k}\sigma_p(u_p)_a(v_p)_b.
\end{aligned}
$$

二乗和では、最初から異なる項の交差項を捨てずに展開する。

$$
\begin{aligned}
\|R\|_F^2
&=\sum_{a=1}^{m}\sum_{b=1}^{n}R_{a,b}^2\\
&=\sum_{a,b}
\left(\sum_{p=r+1}^{k}\sigma_p(u_p)_a(v_p)_b\right)
\left(\sum_{q=r+1}^{k}\sigma_q(u_q)_a(v_q)_b\right)\\
&=\sum_{p=r+1}^{k}\sum_{q=r+1}^{k}\sigma_p\sigma_q
\left(\sum_{a=1}^{m}(u_p)_a(u_q)_a\right)
\left(\sum_{b=1}^{n}(v_p)_b(v_q)_b\right)\\
&=\sum_{p=r+1}^{k}\sum_{q=r+1}^{k}\sigma_p\sigma_q
\delta_{p,q}\delta_{p,q}\\
&=\sum_{p=r+1}^{k}\sigma_p^2.
\end{aligned}
$$

列直交性によって、$p\ne q$ の交差項は0、$p=q$ の項は $\sigma_p^2$ になる。これは「捨てた特異値だから二乗和になる」という類推ではなく、元の行列要素の二乗和からの導出である。

したがって

$$
\boxed{
\|W-W_r\|_F
=
\sqrt{
\sum_{i=r+1}^{k}\sigma_i^2
}
}
$$

となる。

$r=k$ なら空和なので誤差は0。

### 第3節の行列をrank 1へ切り詰め、全要素で検算する

新しい別例へ切り替えず、第3節の $W$ とそのrank-1和を使う。

$$
W=\begin{pmatrix}0&0\\6&0\\0&2\end{pmatrix},
\qquad
W_1=6\begin{pmatrix}0\\1\\0\end{pmatrix}\begin{pmatrix}1&0\end{pmatrix}
=\begin{pmatrix}0&0\\6&0\\0&0\end{pmatrix}.
$$

引き算の全成分を残すと、

$$
\begin{aligned}
R=W-W_1
&=\begin{pmatrix}0-0&0-0\\6-6&0-0\\0-0&2-0\end{pmatrix}\\
&=\begin{pmatrix}0&0\\0&0\\0&2\end{pmatrix}.
\end{aligned}
$$

したがって、

$$
\begin{aligned}
\|W\|_F^2
&=0^2+0^2+6^2+0^2+0^2+2^2=40=6^2+2^2,\\
\|R\|_F^2
&=0^2+0^2+0^2+0^2+0^2+2^2=4=\sigma_2^2,\\
\|R\|_F&=\sqrt4=2.
\end{aligned}
$$

元行列・切り詰め行列・残差の6要素を直接計算しても、捨てた特異値の二乗和と一致する。

---

## 12. spectral normと近似誤差

行列spectral normは

$$
\|A\|_2
=
\max_{\|x\|_2=1}\|Ax\|_2.
$$

切り詰めSVDでは $r<k$ のとき

$$
\boxed{
\|W-W_r\|_2
=
\sigma_{r+1}
}
$$

である。

$r=k$ では

$$
\|W-W_k\|_2=0.
$$

### 単位入力の成分から最大値と達成方向を求める

$0\le r<k$ とする。この導出では第2節の完全な右特異基底 $v_1,\ldots,v_n$ を使う。$k<n$ の場合も、reduced SVDで省かれた零空間方向まで含めることで、任意の $x\in\mathbb R^n$ を

$$
x=\sum_{j=1}^{n}c_jv_j,
\qquad
c_j=v_j^{\mathsf T}x,
\qquad
\|x\|_2^2=\sum_{j=1}^{n}c_j^2
$$

と書ける。残差に代入し、$v_p^{\mathsf T}v_j=\delta_{p,j}$ で和を縮約すると、

$$
\begin{aligned}
Rx
&=\left(\sum_{p=r+1}^{k}\sigma_pu_pv_p^{\mathsf T}\right)
\left(\sum_{j=1}^{n}c_jv_j\right)\\
&=\sum_{p=r+1}^{k}\sum_{j=1}^{n}
\sigma_pc_ju_p(v_p^{\mathsf T}v_j)\\
&=\sum_{p=r+1}^{k}\sigma_pc_pu_p.
\end{aligned}
$$

次に、左特異ベクトルの列直交性で交差項を消す。

$$
\begin{aligned}
\|Rx\|_2^2
&=\sum_{p=r+1}^{k}\sum_{q=r+1}^{k}
\sigma_p\sigma_qc_pc_q(u_p^{\mathsf T}u_q)\\
&=\sum_{p=r+1}^{k}\sigma_p^2c_p^2\\
&\le\sigma_{r+1}^2\sum_{p=r+1}^{k}c_p^2\\
&\le\sigma_{r+1}^2\sum_{j=1}^{n}c_j^2\\
&=\sigma_{r+1}^2\|x\|_2^2.
\end{aligned}
$$

単位入力なら $\|Rx\|_2\le\sigma_{r+1}$。上界だけでなく、$x=v_{r+1}$ を選ぶと、

$$
\|v_{r+1}\|_2=1,
\qquad
Rv_{r+1}=\sigma_{r+1}u_{r+1},
\qquad
\|Rv_{r+1}\|_2=\sigma_{r+1}
$$

となって上界を達成する。したがって、

$$
\max_{\|x\|_2=1}\|Rx\|_2=\sigma_{r+1}.
$$

これがspectral normの誤差式の理由である。$\sigma_{r+1}=0$ なら残りもすべて0なので、最大値は0である。

任意入力について

$$
\|(W-W_r)x\|_2
\le
\|W-W_r\|_2\|x\|_2
$$

なので、$r<k$ なら

$$
\boxed{
\|(W-W_r)x\|_2
\le
\sigma_{r+1}\|x\|_2
}
$$

となる。

任意入力の不等式も、$x\ne0$ なら単位入力 $z=x/\|x\|_2$ を使って、

$$
\|Rx\|_2
=\|\|x\|_2Rz\|_2
=\|x\|_2\|Rz\|_2
\le\|x\|_2\|R\|_2
$$

と導ける。$x=0$ では両辺が0である。

第11節の残差へ $x=(s,t)^{\mathsf T}$ を入れると、

$$
Rx=\begin{pmatrix}0\\0\\2t\end{pmatrix},
\qquad
s^2+t^2=1
\quad\Longrightarrow\quad
\|Rx\|_2=2|t|\le2.
$$

$x=(0,1)^{\mathsf T}$ で2を達成するので $\|R\|_2=2=\sigma_2$。$x=(1,0)^{\mathsf T}$ では誤差0であり、同じ残差でも入力方向によって誤差が異なる。第11節のFrobenius誤差2と一致するのは、この例で捨てた非ゼロ成分が一つだからである。一般には、複数の成分の二乗和と最大成分は異なる。

---

## 13. 特異値energy

この節のenergyと相対誤差は $W\ne0$、すなわち $\sum_i\sigma_i^2>0$ の場合に定義する。零行列では分母が0になるので、そのまま比率を計算しない。

上位$r$成分がFrobenius norm二乗の何割を保持するかを

$$
\boxed{
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}
}
$$

で表す。

相対Frobenius誤差は

$$
\varepsilon_F(r)
=
\frac{\|W-W_r\|_F}{\|W\|_F}
$$

なので、

$$
\begin{aligned}
\varepsilon_F(r)^2
&=
\frac{\sum_{i=r+1}^{k}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}\\
&=
1-
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}\\
&=1-E(r).
\end{aligned}
$$

したがって、

$$
\boxed{
\varepsilon_F(r)=\sqrt{1-E(r)}
}
$$

である。

energy保持率とclassification accuracy保持率は別物である。

---

### 元MDの特異値例を、energyと残差まで追う

元資料の

$$
(\sigma_1,\sigma_2,\sigma_3,\sigma_4)=(10,4,1,0.5)
$$

をrank 2へ打ち切るとする。二乗和から

$$
\|W\|_F^2
=10^2+4^2+1^2+0.5^2
=100+16+1+0.25
=117.25,
$$

$$
E(2)=\frac{100+16}{117.25}
=\frac{116}{117.25}
\approx0.98934,
$$

$$
\|W-W_2\|_F^2=1^2+0.5^2=1.25,
\qquad
\|W-W_2\|_F=\sqrt{1.25}\approx1.11803
$$

となる。相対誤差へ進めば、

$$
\frac{\|W-W_2\|_F}{\|W\|_F}
=\sqrt{\frac{1.25}{117.25}}
=\sqrt{1-\frac{116}{117.25}}
\approx0.10325.
$$

energyを約98.93%保持しても、相対Frobenius誤差は約10.33%である。energyの損失率と、平方根を取ったnormの相対誤差は同じ数値ではない。

元教材 `03_SVDによる低ランク近似.md` の同じ特異値例では、spectral誤差も確認している。捨てた成分は $1$ と $0.5$ なので、

$$
\|W-W_2\|_2
=\max(\sigma_3,\sigma_4)
=\max(1,0.5)
=1
$$

となる。Frobenius誤差 $\sqrt{1.25}$ と最大成分だけを見るspectral誤差 $1$ は異なる。

---

## 14. NN weightの2因子化

Linear weightを

$$
W\in\mathbb R^{D_{\mathrm{out}}\times D_{\mathrm{in}}}
$$

とする。

$$
W_r
=U_r\Sigma_rV_r^{\mathsf T}
$$

に対して、例えば

$$
A
=
\Sigma_rV_r^{\mathsf T}
\in
\mathbb R^{r\times D_{\mathrm{in}}}
$$

$$
B
=
U_r
\in
\mathbb R^{D_{\mathrm{out}}\times r}
$$

と置けば、

$$
\boxed{W_r=BA}
$$

である。

元の

$$
y=Wx+b
$$

を

$$
h=Ax
$$

$$
y=Bh+b
$$

へ置き換えると、

$$
y=BAx+b=W_rx+b.
$$

2層の間に非線形関数を入れると $BAx$ ではなくなるため、単純なSVD置換では挟まない。

---

## 15. 実装上のSVD

NumPy：

```python
U, S, Vh = np.linalg.svd(W, full_matrices=False)
```

PyTorch：

```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
```

ここで

```text
U  : (m, k)
S  : (k,)
Vh : (k, n)
```

である。

rank$r$なら

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
W_r = (U_r * S_r) @ Vh_r
```

と書ける。

---

### 元教材の $5\times3$ の再構成例を手順ごとに確認する

元教材 `01_SVDとは.md` には、乱数で作った $5\times3$ 行列に対して、shape表示、対角行列の生成、再構成、許容誤差付き比較を順に行う例がある。呼び出し1行だけでなく、その確認手順も残す。

$$
m=5,\quad n=3,\quad k=\min(5,3)=3,
\qquad
U:(5,3),\ S:(3,),\ Vh:(3,3).
$$

`S` は1次元配列なので、行列積による復元では

$$
S=
\begin{pmatrix}\sigma_1\\\sigma_2\\\sigma_3\end{pmatrix},
\qquad
\operatorname{diag}(S)
=
\begin{pmatrix}
\sigma_1&0&0\\
0&\sigma_2&0\\
0&0&\sigma_3
\end{pmatrix},
\qquad
W_{\mathrm{restored}}
=
U\,\operatorname{diag}(S)\,Vh
$$

と対応させる。元教材のNumPy側のshape・seed・乱数生成法を保持した確認コードは次のとおりである。

```python
import numpy as np

# 元教材の5×3の行列。seedと生成法を固定して再構成を確認する。
rng = np.random.default_rng(seed=0)
W = rng.standard_normal((5, 3))
U, S, Vh = np.linalg.svd(W, full_matrices=False)

print("W :", W.shape)
print("U :", U.shape)
print("S :", S.shape)
print("Vh:", Vh.shape)
Sigma = np.diag(S)
W_restored = U @ Sigma @ Vh
assert np.allclose(W, W_restored, rtol=1e-10, atol=1e-10)
```

元教材のPyTorch側も同じshapeを確認する。両ライブラリでseedを0にしても、乱数生成器が異なるので行列の値まで同一になるという意味ではない。ここではそれぞれの入力が、それぞれのSVDから復元されるかを確認する。

```python
import torch

# 元教材のPyTorch例。全成分を残すreduced SVDは打ち切り近似ではない。
torch.manual_seed(0)
W = torch.randn(5, 3, dtype=torch.float64)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

print("W :", tuple(W.shape))
print("U :", tuple(U.shape))
print("S :", tuple(S.shape))
print("Vh:", tuple(Vh.shape))
W_restored = U @ torch.diag(S) @ Vh
assert torch.allclose(W, W_restored, rtol=1e-10, atol=1e-10)
```

元教材の `truncated_svd` という名前の関数は、復元済みの密行列 $W_r$ を返す例も含む。一方、[[07_PyTorch実装#3. 実装全体]] の同名関数は `(U_r, S_r, Vh_r)` の3因子を返す。名前が同じでも戻り値が異なるため、密行列を使いたいときは因子から `(U_r * S_r) @ Vh_r` と再構成する。具体的な列broadcastの要素対応は [[08_Linear層のSVD実装#対角行列積と列broadcastの要素を対応させる]] にまとめる。

## 16. よくある誤解

### SVDは正方行列だけ

誤り。長方形行列にも適用できる。

### $U$ が入力方向、$V$ が出力方向

このノートの約束

$$
W:\mathbb R^n\to\mathbb R^m
$$

では逆である。

$$
v_i\in\mathbb R^n
\xrightarrow{W}
\sigma_i u_i\in\mathbb R^m
$$

なので、$v_i$ が入力側、$u_i$ が出力側。

### 特異値の単純和を99%残す

Frobenius energyでは

$$
\sum_i\sigma_i^2
$$

を使う。

### low rankなら必ず推論が速い

parameters / MACsが減っても、kernel起動やハードウェア利用効率によりwall-clock latencyは改善しない場合がある。

---

## 17. このノートで押さえるポイント

- $W=U\Sigma V^{\mathsf T}$。
- $Wv_i=\sigma_i u_i$ なので $v_i$ は入力側、$u_i$ は出力側。
- 完全SVDの転置は $W^{\mathsf T}=V\Sigma^{\mathsf T}U^{\mathsf T}$。
- $W^{\mathsf T}Wv_i=\sigma_i^2v_i$、$WW^{\mathsf T}u_i=\sigma_i^2u_i$。
- truncated SVDは固定rankでFrobenius / spectral normの最良近似を与えるが、最適解が常に一意とは限らない。
- $\|W-W_r\|_F^2=\sum_{i>r}\sigma_i^2$。
- $r<k$ なら $\|W-W_r\|_2=\sigma_{r+1}$、full rankなら0。
- $\varepsilon_F(r)=\sqrt{1-E(r)}$。
- NNではweight近似の良さとtask accuracyを分けて評価する。

---

## 18. 次に読むノート

- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/02_nn.Linearとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[50_DMRG/01_SVDからDMRGへのつながり]]
