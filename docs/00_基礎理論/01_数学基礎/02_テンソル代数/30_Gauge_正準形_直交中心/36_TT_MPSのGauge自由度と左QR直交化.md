---
title: TT・MPSのGauge自由度と左QR直交化
aliases:
  - TT gauge freedom
  - MPSの左直交化
tags:
  - TT
  - MPS
  - QR
  - gauge
---

# TT・MPSのGauge自由度と左QR直交化

## サマリー

TT/MPSの内部ボンドの基底は一意ではない。隣接コアに可逆行列とその逆行列を挿入しても、表す全テンソルは変わらない。この自由度を利用し、QRの三角因子を右隣へ送ると左直交形を作れる。

[[30_TT_MPSの定義]] 〜 [[35_TT_MPSの等長写像と射影]] の続きとして読む。本章以降では $G_k\equiv G^{(k)}$ と書く。実数・開放境界を基本とし、コアの軸は常に $(\alpha_{k-1},i_k,\alpha_k)$ とする。

QRの三角因子は $T_k$ と書く。一般的なQR記法の $R_k$ に相当するが、後続の右ブロック $\mathcal R_k$ と混同しないために分ける。

## 1. Gauge変換の成分と全体不変性

可逆行列 $M\in\mathbb R^{r_1\times r_1}$ を第1ボンドへ入れる。

$$
\widetilde G_1(1,i_1,\beta_1)
=\sum_{\alpha_1=1}^{r_1}G_1(1,i_1,\alpha_1)M(\alpha_1,\beta_1),
$$

$$
\widetilde G_2(\beta_1,i_2,\alpha_2)
=\sum_{\gamma_1=1}^{r_1}M^{-1}(\beta_1,\gamma_1)G_2(\gamma_1,i_2,\alpha_2).
$$

元と新しいボンド添字は別の基底ラベルである。隣接2コアを縮約すると、途中式は

$$
\begin{aligned}
\sum_{\beta_1}\widetilde G_1(1,i_1,\beta_1)\widetilde G_2(\beta_1,i_2,\alpha_2)
&=\sum_{\beta_1,\alpha_1,\gamma_1}
G_1(1,i_1,\alpha_1)M(\alpha_1,\beta_1)
M^{-1}(\beta_1,\gamma_1)G_2(\gamma_1,i_2,\alpha_2)\\
&=\sum_{\alpha_1,\gamma_1}G_1(1,i_1,\alpha_1)
\underbrace{\sum_{\beta_1}M(\alpha_1,\beta_1)M^{-1}(\beta_1,\gamma_1)}_{\delta_{\alpha_1,\gamma_1}}
G_2(\gamma_1,i_2,\alpha_2)\\
&=\sum_{\alpha_1}G_1(1,i_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2).
\end{aligned}
$$

さらに $G_3$ を縮約すれば全要素について $\widetilde X=X$ となる。これは内部の仮想基底の変更であり、physical indexの基底変換やSVDによる打ち切りとは異なる。一般のGauge変換ではコアの直交性やコア単独のノルムは保存されないが、全体のテンソル・各cut rank・ノルムは不変である。

最も単純な例は $M=cI_{r_1}$、$c\ne0$ のスカラーGaugeである。このとき $M^{-1}=c^{-1}I_{r_1}$ だから、コアの全要素は $\widetilde G_1(1,i_1,\alpha_1)=cG_1(1,i_1,\alpha_1)$、$\widetilde G_2(\alpha_1,i_2,\alpha_2)=c^{-1}G_2(\alpha_1,i_2,\alpha_2)$ となる。各物理添字を固定した隣接収縮では

$$
\begin{aligned}
\sum_{\alpha_1}\widetilde G_1(1,i_1,\alpha_1)
\widetilde G_2(\alpha_1,i_2,\alpha_2)
&=\sum_{\alpha_1}cG_1(1,i_1,\alpha_1)
c^{-1}G_2(\alpha_1,i_2,\alpha_2)\\
&=\sum_{\alpha_1}G_1(1,i_1,\alpha_1)
G_2(\alpha_1,i_2,\alpha_2).
\end{aligned}
$$

例えば $c=100$ なら $\|\widetilde G_1\|_F=100\|G_1\|_F$、$\|\widetilde G_2\|_F=0.01\|G_2\|_F$ だが、$\widetilde X=X$ なので全体の $\|X\|_F$ は変わらない。コアの大きさを全体の大きさと同一視しないための反例である。

「内部ボンドの基底を替える」の意味も、第1コアの左展開 $A_1\in\mathbb R^{n_1\times r_1}$ で確かめられる。$A_1$ の各列を左側の状態ベクトルと読むと、$A_1M$ の各列は元の列の線形結合なので $\operatorname{col}(A_1M)\subseteq\operatorname{col}(A_1)$ である。一方、$M$ は可逆だから $A_1=(A_1M)M^{-1}$ となり、逆向きの包含も成立する。従って

$$
\operatorname{col}(A_1M)=\operatorname{col}(A_1).
$$

例えば $A_1=(u_1\ u_2)$、$M=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ なら $A_1M=(u_1\ u_1+u_2)$ で、二組は同じ列空間を張る。ただし新しい列同士が直交するとは限らない。列が一次独立なら二組は同じ空間の異なる基底であり、従属している場合も同じ生成集合の空間を保つ。右隣のコアに $M^{-1}$ を作用させるのは、このボンド上の列の取り替えに合わせて係数を逆変換し、全体の $X$ を保つためである。QRによる左直交化は、同じ空間内で直交する列を選ぶ特別な場合として後で扱う。

### 可逆行列を全要素で確認する

第1コアの境界軸を外した行列 $A_1$ と第2コアのphysical slice $H_j=G_2(:,j,:)$ を

$$
A_1=\begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad
M=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
M^{-1}=\begin{pmatrix}1&-1\\0&1\end{pmatrix},
$$

$$
H_1=\begin{pmatrix}1&0\\2&1\end{pmatrix},\qquad
H_2=\begin{pmatrix}0&1\\1&3\end{pmatrix}
$$

とする。積の全要素は

$$
MM^{-1}
=\begin{pmatrix}1\cdot1+1\cdot0&1\cdot(-1)+1\cdot1\\
0\cdot1+1\cdot0&0\cdot(-1)+1\cdot1\end{pmatrix}
=\begin{pmatrix}1&0\\0&1\end{pmatrix},
$$

$$
\widetilde A_1=A_1M=\begin{pmatrix}1&3\\3&7\end{pmatrix},\qquad
\widetilde H_1=M^{-1}H_1=\begin{pmatrix}-1&-1\\2&1\end{pmatrix},\qquad
\widetilde H_2=M^{-1}H_2=\begin{pmatrix}-1&-2\\1&3\end{pmatrix}
$$

となる。変換前後の局所縮約は

$$
A_1H_1=\begin{pmatrix}5&2\\11&4\end{pmatrix}
=\widetilde A_1\widetilde H_1,\qquad
A_1H_2=\begin{pmatrix}2&7\\4&15\end{pmatrix}
=\widetilde A_1\widetilde H_2.
$$

例えば第1sliceの $(1,1)$ 成分は、変換前が $1\cdot1+2\cdot2=5$、変換後が $1\cdot(-1)+3\cdot2=5$ である。他の成分も表示した行列全体で一致している。

### Gauge数値例の各要素の積まで展開する

上のGauge数値例は、変換行列の要素積まで展開できる。
同じ例の値を使って

$$
\begin{aligned}
\widetilde A_1=A_1M
&=\begin{pmatrix}
1\cdot1+2\cdot0&1\cdot1+2\cdot1\\
3\cdot1+4\cdot0&3\cdot1+4\cdot1
\end{pmatrix}
=\begin{pmatrix}1&3\\3&7\end{pmatrix},\\
\widetilde H_1=M^{-1}H_1
&=\begin{pmatrix}
1\cdot1+(-1)\cdot2&1\cdot0+(-1)\cdot1\\
0\cdot1+1\cdot2&0\cdot0+1\cdot1
\end{pmatrix}
=\begin{pmatrix}-1&-1\\2&1\end{pmatrix},\\
\widetilde H_2=M^{-1}H_2
&=\begin{pmatrix}
1\cdot0+(-1)\cdot1&1\cdot1+(-1)\cdot3\\
0\cdot0+1\cdot1&0\cdot1+1\cdot3
\end{pmatrix}
=\begin{pmatrix}-1&-2\\1&3\end{pmatrix}.
\end{aligned}
$$

変換後の積も

$$
\begin{aligned}
\widetilde A_1\widetilde H_1
&=\begin{pmatrix}
1\cdot(-1)+3\cdot2&1\cdot(-1)+3\cdot1\\
3\cdot(-1)+7\cdot2&3\cdot(-1)+7\cdot1
\end{pmatrix}
=\begin{pmatrix}5&2\\11&4\end{pmatrix}=A_1H_1,\\
\widetilde A_1\widetilde H_2
&=\begin{pmatrix}
1\cdot(-1)+3\cdot1&1\cdot(-2)+3\cdot3\\
3\cdot(-1)+7\cdot1&3\cdot(-2)+7\cdot3
\end{pmatrix}
=\begin{pmatrix}2&7\\4&15\end{pmatrix}=A_1H_2.
\end{aligned}
$$

変換前後で個々のcoreは変わるが、その積の全4要素はどちらのsliceでも変わらない。

全体の不変性とコアの直交性は別である。例えば、列直交な $Q=I_2$ に上と同じ非直交な可逆行列 $M$ をボンド側から掛けると、

$$
\begin{aligned}
(QM)^T(QM)
&=M^T(Q^TQ)M=M^TM\\
&=\begin{pmatrix}1&0\\1&1\end{pmatrix}
\begin{pmatrix}1&1\\0&1\end{pmatrix}
=\begin{pmatrix}1&1\\1&2\end{pmatrix}\ne I_2.
\end{aligned}
$$

右隣へ $M^{-1}$ を同時に掛ければ全テンソルは保存されるが、左コアの列直交性は失われる。Gauge自由度そのものは直交化ではなく、QRはその自由度を利用して**直交条件を満たす表示を選ぶ**操作である。

## 2. QRの添字は何で決まるか

一般の入力行列 $A\in\mathbb R^{m\times n}$ に対して、reduced QRは

$$
A=QT,\qquad q=\min(m,n),\qquad
Q\in\mathbb R^{m\times q},\qquad T\in\mathbb R^{q\times n},\qquad Q^TQ=I_q
$$

である。成分で書くと

$$
\boxed{A(a,b)=\sum_{\gamma=1}^{q}Q(a,\gamma)T(\gamma,b)}.
$$

入力の行添字 $a$ は $Q$ に残り、入力の列添字 $b$ は $T$ に残る。新しい $\gamma$ は和で消す基底添字であり、元の $b$ と同じラベルではない。$m\ge n$ のときに限って $q=n$ になる。

左直交化で元のボンドサイズを維持するには、$r_k\le r_{k-1}n_k$ が必要である。入力が縦長・列フルランクなら $T$ は正方可逆で、$A=QT$ を $Q=AT^{-1}$ と読める。この場合はGauge変換そのものになる。

横長では $q<r_k$ となり、QR後のボンドサイズは $q$ へ変わる。それでも因子を吸収すれば全体を厳密に保存するが、同サイズの可逆Gauge変換ではない。また、通常のreduced QRは特異値によるrank選択や圧縮誤差制御を行わない。rank欠損では $Q$ に補完方向が含まれ得るため、「常に $Q$ の列数が数値rankに等しい」とは言わない。PyTorchのrank欠損入力・微分に関する注意も参照する。

### QRの射影係数はなぜ内積で求まるか

第1列 $a_1\ne0$ を持つ2列の行列 $A=(a_1\ a_2)$ を考える。第1列を $t_{11}=\|a_1\|_2$ で割った $q_1=a_1/t_{11}$ は単位ベクトルである。第2列のうち $q_1$ と平行な**ベクトル**を $t_{12}q_1$、残りを $u_2=a_2-t_{12}q_1$ と置く。$t_{12}$ 自体は符号付きの長さを表す**スカラー**であり、$t_{12}q_1$ と同一視しない。

残りが第1列方向と直交する条件から係数を決めると、

$$
\begin{aligned}
0=q_1^Tu_2
&=q_1^T(a_2-t_{12}q_1)\\
&=q_1^Ta_2-t_{12}q_1^Tq_1
=q_1^Ta_2-t_{12},\\
t_{12}&=q_1^Ta_2=\sum_{i=1}^{m}q_1(i)a_2(i).
\end{aligned}
$$

$a_2\ne0$ のとき $q_1$ と $a_2$ のなす角を $\theta$ とすれば、$\|q_1\|_2=1$ より $t_{12}=\|a_2\|_2\cos\theta$ である。正なら $q_1$ と同じ向き、負なら逆向き、0なら平行成分はない。$a_2=0$ のときも内積式から $t_{12}=0$ と定まる。第2列は

$$
a_2=\underbrace{(q_1^Ta_2)q_1}_{\text{平行成分}}+
\underbrace{\bigl(a_2-(q_1^Ta_2)q_1\bigr)}_{\text{直交成分}}
$$

に分かれる。方向ベクトルが単位長でない $v\ne0$ の場合は、同じ直交条件 $v^T(a_2-cv)=0$ から

$$
c=\frac{v^Ta_2}{v^Tv},\qquad
\operatorname{proj}_{v}(a_2)=\frac{v^Ta_2}{v^Tv}v
$$

となる。分母を省けるのは $q_1^Tq_1=1$ のときだけである。後の第1QR数値例では $q_1=(1,1,0)^T/\sqrt2$、$a_2=(1,1,2)^T$ なので、$t_{12}=\sqrt2$、平行成分は $(1,1,0)^T$、直交成分は $(0,0,2)^T$ になる。

残差から第2列まで作り切る。$a_1$ と $a_2$ が一次独立なら $u_2\ne0$ なので、

$$
\begin{aligned}
t_{22}&=\|u_2\|_2=\sqrt{\sum_{i=1}^{m}u_2(i)^2},\\
q_2&=\frac{u_2}{t_{22}},\qquad
q_2(i)=\frac{a_2(i)-t_{12}q_1(i)}{t_{22}}.
\end{aligned}
$$

$q_1^Tu_2=0$ と $\|u_2\|_2=t_{22}$ から、二つの列の内積を省略せずに確かめる。

$$
\begin{aligned}
q_1^Tq_2
&=\frac{1}{t_{22}}\sum_{i=1}^{m}q_1(i)
\bigl(a_2(i)-t_{12}q_1(i)\bigr)
=\frac{t_{12}-t_{12}}{t_{22}}=0,\\
q_2^Tq_2
&=\frac{1}{t_{22}^2}\sum_{i=1}^{m}u_2(i)^2=1.
\end{aligned}
$$

第1列は $a_1=t_{11}q_1$、第2列は $u_2=a_2-t_{12}q_1=t_{22}q_2$ なので、列ごとに元の行列へ戻すと

$$
\begin{aligned}
A=(a_1\ a_2)
&=(t_{11}q_1\quad t_{12}q_1+t_{22}q_2)\\
&=(q_1\ q_2)
\begin{pmatrix}t_{11}&t_{12}\\0&t_{22}\end{pmatrix}
=QT,\\
Q^TQ
&=\begin{pmatrix}q_1^Tq_1&q_1^Tq_2\\q_2^Tq_1&q_2^Tq_2\end{pmatrix}
=I_2.
\end{aligned}
$$

$T$ の左下が0なのは、第1列 $a_1$ を作るとき $q_2$ 成分を使わないからである。成分ごとにも $A(i,1)=q_1(i)t_{11}$、$A(i,2)=q_1(i)t_{12}+q_2(i)t_{22}$ と確認できる。列が従属して $u_2=0$ なら $t_{22}=0$ であり、この式による $q_2$ の正規化はできない。rank欠損時のreduced QRでは、直交補空間から列を補う場合があるので、上の一次独立な2列の導出と区別する。

## 3. 第1コアの左QRと右隣への吸収

$$
A_1(i_1,\alpha_1)=G_1(1,i_1,\alpha_1),\qquad A_1\in\mathbb R^{n_1\times r_1}.
$$

まず $n_1\ge r_1$ の列フルランクの場合を考える。

$$
A_1(i_1,\alpha_1)
=\sum_{\beta_1=1}^{r_1}Q_1(i_1,\beta_1)T_1(\beta_1,\alpha_1).
$$

新コアと更新する隣接コアは

$$
G_1^{[L]}(1,i_1,\beta_1)=Q_1(i_1,\beta_1),
$$

$$
G_2^{\mathrm{tmp}}(\beta_1,i_2,\alpha_2)
=\sum_{\alpha_1}T_1(\beta_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2).
$$

和で消えるのは古い $\alpha_1$、新しく残るのは $\beta_1$ である。$Q_1^TQ_1=I$ から

$$
\sum_{i_1}G_1^{[L]}(1,i_1,\beta_1)G_1^{[L]}(1,i_1,\gamma_1)
=\delta_{\beta_1,\gamma_1}
$$

が成立する。全テンソルの途中式は

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1,\alpha_2,\beta_1}
Q_1(i_1,\beta_1)T_1(\beta_1,\alpha_1)
G_2(\alpha_1,i_2,\alpha_2)G_3(\alpha_2,i_3,1)\\
&=\sum_{\beta_1,\alpha_2}
G_1^{[L]}(1,i_1,\beta_1)G_2^{\mathrm{tmp}}(\beta_1,i_2,\alpha_2)G_3(\alpha_2,i_3,1).
\end{aligned}
$$

$T_1$ を捨てるのではなく、右隣へ吸収するから全体が不変になる。

## 4. 第2コアも左直交化する

更新済みの $G_2^{\mathrm{tmp}}$ を、行 $(\beta_1,i_2)$、列 $\alpha_2$ で並べる。

$$
A_2((\beta_1,i_2),\alpha_2)=G_2^{\mathrm{tmp}}(\beta_1,i_2,\alpha_2),\qquad
A_2\in\mathbb R^{(r_1n_2)\times r_2}.
$$

1始まりの行位置は $p=(\beta_1-1)n_2+i_2$ で、$i_2$ が速く動く。$r_1n_2\ge r_2$・列フルランクなら

$$
A_2((\beta_1,i_2),\alpha_2)
=\sum_{\beta_2}Q_2((\beta_1,i_2),\beta_2)T_2(\beta_2,\alpha_2),
$$

$$
G_2^{[L]}(\beta_1,i_2,\beta_2)=Q_2((\beta_1,i_2),\beta_2),\qquad
G_3^{\mathrm{new}}(\beta_2,i_3,1)
=\sum_{\alpha_2}T_2(\beta_2,\alpha_2)G_3(\alpha_2,i_3,1).
$$

ここで第2QRの**直前**の局所収縮に使うのは、元の $G_2$ ではなく、第1QRの $T_1$ を吸収済みの $G_2^{\mathrm{tmp}}$ である。第2QRの分解式を代入すると、

$$
\begin{aligned}
B_{\mathrm{before}}(\beta_1,i_2,i_3)
&:=\sum_{\alpha_2}G_2^{\mathrm{tmp}}(\beta_1,i_2,\alpha_2)
G_3(\alpha_2,i_3,1)\\
&=\sum_{\alpha_2,\beta_2}G_2^{[L]}(\beta_1,i_2,\beta_2)
T_2(\beta_2,\alpha_2)G_3(\alpha_2,i_3,1)\\
&=\sum_{\beta_2}G_2^{[L]}(\beta_1,i_2,\beta_2)
\underbrace{\sum_{\alpha_2}T_2(\beta_2,\alpha_2)
G_3(\alpha_2,i_3,1)}_{G_3^{\mathrm{new}}(\beta_2,i_3,1)}\\
&=:B_{\mathrm{after}}(\beta_1,i_2,i_3).
\end{aligned}
$$

この局所等式は $G_1^{[L]}$ を掛ける前に成立する。従って全テンソルの不変性より強い位置で第2QRの吸収を検査できる。第1QR前の $G_2$ と第2QR後の局所収縮を直接比較すると、異なる段階を比べることになる。

従って

$$
\sum_{\beta_1,i_2}G_2^{[L]}(\beta_1,i_2,\beta_2)G_2^{[L]}(\beta_1,i_2,\gamma_2)
=\delta_{\beta_2,\gamma_2},
$$

$$
X(i_1,i_2,i_3)
=\sum_{\beta_1,\beta_2}G_1^{[L]}(1,i_1,\beta_1)G_2^{[L]}(\beta_1,i_2,\beta_2)G_3^{\mathrm{new}}(\beta_2,i_3,1)
$$

となる。$T_1,T_2$ を左から右へ送るのが左QR sweepである。第1コアだけのQRでは第2コアまで左直交化したことにはならない。

## 5. 第1・第2QRを一貫した数値例で計算する

前節のGauge例とは別に、$n_1=3,n_2=n_3=r_1=r_2=2$ とする。境界軸を外した両端行列と中央の全sliceを

$$
A_1=\begin{pmatrix}1&1\\1&1\\0&2\end{pmatrix},\qquad
H_1=\begin{pmatrix}1&0\\0&1\end{pmatrix},\qquad
H_2=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
A_3=\begin{pmatrix}1&2\\3&4\end{pmatrix}
$$

と置く。つまり $G_1(1,:,:)=A_1$、$G_2(:,j,:)=H_j$、$G_3(:,:,1)=A_3$ で、全コアの値を指定している。

### 第1QR：列を順に正規直交化

$$
a_1=\begin{pmatrix}1\\1\\0\end{pmatrix},\quad
a_2=\begin{pmatrix}1\\1\\2\end{pmatrix},\quad
t_{11}=\|a_1\|_2=\sqrt2,\quad
q_1=\frac1{\sqrt2}\begin{pmatrix}1\\1\\0\end{pmatrix}.
$$

$$
t_{12}=q_1^Ta_2=\frac{1+1}{\sqrt2}=\sqrt2,\qquad
v_2=a_2-q_1t_{12}=\begin{pmatrix}0\\0\\2\end{pmatrix},\qquad
t_{22}=2,\quad q_2=\begin{pmatrix}0\\0\\1\end{pmatrix}.
$$

従って、対角を正に取った一つのQRは

$$
Q_1=\begin{pmatrix}1/\sqrt2&0\\1/\sqrt2&0\\0&1\end{pmatrix},\qquad
T_1=\begin{pmatrix}\sqrt2&\sqrt2\\0&2\end{pmatrix},\qquad
Q_1T_1=\begin{pmatrix}1&1\\1&1\\0&2\end{pmatrix}=A_1.
$$

吸収した中央の全sliceは

$$
H_1^{\mathrm{tmp}}=T_1H_1=\begin{pmatrix}\sqrt2&\sqrt2\\0&2\end{pmatrix},\qquad
H_2^{\mathrm{tmp}}=T_1H_2=\begin{pmatrix}\sqrt2&\sqrt2\\2&0\end{pmatrix}.
$$

### 第2QR：複合添字を行にする

行を $(\beta_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ とすると

$$
A_2=\begin{pmatrix}\sqrt2&\sqrt2\\\sqrt2&\sqrt2\\0&2\\2&0\end{pmatrix}.
$$

その第1・第2列を $b_1,b_2$ とする。

$$
\|b_1\|_2=\sqrt{2+2+0+4}=2\sqrt2,\qquad
u_1=\begin{pmatrix}1/2\\1/2\\0\\1/\sqrt2\end{pmatrix},\qquad
u_1^Tb_2=\frac{\sqrt2}{2}+\frac{\sqrt2}{2}=\sqrt2.
$$

$$
w_2=b_2-u_1\sqrt2=\begin{pmatrix}1/\sqrt2\\1/\sqrt2\\2\\-1\end{pmatrix},\qquad
\|w_2\|_2=\sqrt{1/2+1/2+4+1}=\sqrt6.
$$

よって

$$
Q_2=\begin{pmatrix}
1/2&1/\sqrt{12}\\
1/2&1/\sqrt{12}\\
0&2/\sqrt6\\
1/\sqrt2&-1/\sqrt6
\end{pmatrix},\qquad
T_2=\begin{pmatrix}2\sqrt2&\sqrt2\\0&\sqrt6\end{pmatrix}.
$$

例えば積の第4行は

$$
\begin{pmatrix}1/\sqrt2&-1/\sqrt6\end{pmatrix}T_2
=\begin{pmatrix}2&1-1\end{pmatrix}
=\begin{pmatrix}2&0\end{pmatrix}
$$

で、$A_2$ の $(\beta_1,i_2)=(2,2)$ 行と一致する。第2コアへ戻した全sliceは

$$
G_2^{[L]}(:,1,:)=\begin{pmatrix}1/2&1/\sqrt{12}\\0&2/\sqrt6\end{pmatrix},\qquad
G_2^{[L]}(:,2,:)=\begin{pmatrix}1/2&1/\sqrt{12}\\1/\sqrt2&-1/\sqrt6\end{pmatrix}.
$$

最後に三角因子を第3コアへ吸収すると

$$
A_3^{\mathrm{new}}=T_2A_3
=\begin{pmatrix}2\sqrt2\cdot1+\sqrt2\cdot3&2\sqrt2\cdot2+\sqrt2\cdot4\\
\sqrt6\cdot3&\sqrt6\cdot4\end{pmatrix}
=\begin{pmatrix}5\sqrt2&8\sqrt2\\3\sqrt6&4\sqrt6\end{pmatrix}.
$$

### 直交性と全テンソルを別々に確認

$$
Q_2^TQ_2
=\begin{pmatrix}
1/4+1/4+0+1/2&1/(2\sqrt{12})+1/(2\sqrt{12})-1/\sqrt{12}\\
1/(2\sqrt{12})+1/(2\sqrt{12})-1/\sqrt{12}&1/12+1/12+4/6+1/6
\end{pmatrix}
=\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$$

一方、元の2コアの縮約を行 $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)$ に並べると

$$
X^{\langle2\rangle}
=\begin{pmatrix}1&1\\1&1\\1&1\\1&1\\0&2\\2&0\end{pmatrix}
\begin{pmatrix}1&2\\3&4\end{pmatrix}
=\begin{pmatrix}4&6\\4&6\\4&6\\4&6\\6&8\\2&4\end{pmatrix}.
$$

新しい $Q_1,G_2^{[L]},A_3^{\mathrm{new}}$ からの再構成もこの6行2列になる。例えば $(i_1,i_2)=(3,2)$ では

$$
\begin{pmatrix}0&1\end{pmatrix}
G_2^{[L]}(:,2,:)A_3^{\mathrm{new}}
=\begin{pmatrix}1/\sqrt2&-1/\sqrt6\end{pmatrix}A_3^{\mathrm{new}}
=\begin{pmatrix}5-3&8-4\end{pmatrix}
=\begin{pmatrix}2&4\end{pmatrix}.
$$

直交条件と再構成不変性は別の検証項目である。$Q^TQ=I$ だけでは三角因子の吸収ミスを検出できない。

### QR後の左ブロックと全12要素を行列積で再構成する

一つの成分だけでなく、上で得た全コアを使って新しい左ブロックも全要素表示する。

$$
Q_1\otimes I_2
=\begin{pmatrix}
1/\sqrt2&0&0&0\\
0&1/\sqrt2&0&0\\
1/\sqrt2&0&0&0\\
0&1/\sqrt2&0&0\\
0&0&1&0\\
0&0&0&1
\end{pmatrix}.
$$

行を元の $(i_1,i_2)$ 順、途中の列を $(\beta_1,i_2)$ 順とすると、

上のKronecker行列の各行は非ゼロ係数を一つだけ持つ。したがって、非ゼロ項だけを残した各成分は次の積になり、他の項はすべて0である。

$$
\begin{aligned}
\mathcal L_2^{\mathrm{new}}
&=(Q_1\otimes I_2)Q_2\\
&=\begin{pmatrix}
\frac1{\sqrt2}\cdot\frac12&\frac1{\sqrt2}\cdot\frac1{\sqrt{12}}\\
\frac1{\sqrt2}\cdot\frac12&\frac1{\sqrt2}\cdot\frac1{\sqrt{12}}\\
\frac1{\sqrt2}\cdot\frac12&\frac1{\sqrt2}\cdot\frac1{\sqrt{12}}\\
\frac1{\sqrt2}\cdot\frac12&\frac1{\sqrt2}\cdot\frac1{\sqrt{12}}\\
1\cdot0&1\cdot\frac2{\sqrt6}\\
1\cdot\frac1{\sqrt2}&1\cdot\left(-\frac1{\sqrt6}\right)
\end{pmatrix}\\
&=\begin{pmatrix}
1/(2\sqrt2)&1/\sqrt{24}\\
1/(2\sqrt2)&1/\sqrt{24}\\
1/(2\sqrt2)&1/\sqrt{24}\\
1/(2\sqrt2)&1/\sqrt{24}\\
0&2/\sqrt6\\
1/\sqrt2&-1/\sqrt6
\end{pmatrix}.
\end{aligned}
$$

最後のコアとの積では、各行の二つの項を残したまま計算すると、

$$
\begin{aligned}
\mathcal L_2^{\mathrm{new}}A_3^{\mathrm{new}}
&=\begin{pmatrix}
5/2+3/2&4+2\\
5/2+3/2&4+2\\
5/2+3/2&4+2\\
5/2+3/2&4+2\\
0+6&0+8\\
5-3&8-4
\end{pmatrix}\\
&=\begin{pmatrix}4&6\\4&6\\4&6\\4&6\\6&8\\2&4\end{pmatrix}\\
&=X^{\langle2\rangle}.
\end{aligned}
$$

第1QRから第2QR、三角因子の吸収、全要素の再構成まで、同じ数値例のまま閉じた計算になっている。

## 6. この章の到達点

- 可逆Gauge変換で隣接縮約と全体が不変になる理由を、逆行列の成分和から説明できる。
- QRの古いボンド、新しいボンド、physical indexを分けて追える。
- 第1・第2コアを左直交化し、三角因子を最後のコアへ集める意味を説明できる。
- QRは全体を保存する基底整理で、SVD truncationとは区別する。

次は [[37_TT_MPSの左ブロックと直交性の導出]]。PyTorch操作との対応は [[28_TT_MPS実装で使うPyTorch_Python操作メモ]] のQR追補を参照する。

## 参考資料

- [PyTorch：torch.linalg.qr](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)：reducedのshape、符号の非一意性、rank欠損入力の注意。
- [Schollwöck：DMRG in the age of matrix product states](https://arxiv.org/abs/1008.3477)：Gauge自由度、canonical MPSとQRによる直交化。
