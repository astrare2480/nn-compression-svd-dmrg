---
title: SVDによる低ランク近似
aliases:
  - truncated SVD
  - 切り詰めSVD
  - 低ランク分解
tags:
  - SVD
  - 低ランク近似
  - NN圧縮
  - PyTorch
  - 行列近似
---

# SVDによる低ランク近似

## サマリー

SVDによる低ランク近似は、行列を

$$
W=U\Sigma V^{\mathsf T}
$$

と分解し、大きい特異値に対応する上位 $r$ 成分だけを残す方法である。

$$
\boxed{
W_r
=U_r\Sigma_rV_r^{\mathsf T}
=
\sum_{i=1}^{r}\sigma_i u_i v_i^{\mathsf T}
}
$$

ここで

$$
\operatorname{rank}(W_r)\le r.
$$

Linear weightなら、この $W_r$ を2因子

$$
W_r=BA
$$

として保持することで、元の入出力shapeを保ったままparameter数を減らせる。

詳細なSVD自体の導出は [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]] に置き、このノートでは**低rank近似として何が起きるか**を途中式込みで整理する。

---

## 1. rankと低rank行列

$$
W\in\mathbb R^{m\times n}
$$

に対して

$$
\operatorname{rank}(W)
\le
k,
\qquad
k=\min(m,n)
$$

である。

フルrankなら

$$
\operatorname{rank}(W)=k,
$$

低rankなら

$$
\operatorname{rank}(W)<k.
$$

SVDの非零特異値数とrankは一致する。

$$
\boxed{
\operatorname{rank}(W)
=
\#\{i\mid\sigma_i>0\}
}
$$

浮動小数点では厳密な0ではなくthresholdを使う場合がある。

---

## 2. rank-$r$ 行列を2因子で表す

### 「rankが $r$ 以下なら」の因子の作り方

実際のrankを $s:=\operatorname{rank}(W_r)\le r$ とする。
非零特異値だけを使ったSVDを

$$
W_r=\bar U_s\bar\Sigma_s\bar V_s^{\mathsf T}
$$

と書き、$r-s$ 個の0の行・列を足して

$$
A=
\begin{pmatrix}\bar V_s^{\mathsf T}\\0_{(r-s)\times D_{in}}\end{pmatrix},
\qquad
B=\begin{pmatrix}\bar U_s\bar\Sigma_s&0_{D_{out}\times(r-s)}\end{pmatrix}
$$

とすれば

$$
BA=\bar U_s\bar\Sigma_s\bar V_s^{\mathsf T}
+0_{D_{out}\times(r-s)}0_{(r-s)\times D_{in}}
=W_r.
$$

逆に $W_r=BA$ なら $\operatorname{col}(W_r)\subseteq\operatorname{col}(B)$ なので

$$
\operatorname{rank}(W_r)\le\operatorname{rank}(B)\le r.
$$

これで「rankが $r$ 以下」と「幅 $r$ の2因子で表せる」の両方向が確認できる。
零行列では非零特異値の部分を空とし、二つの因子を零行列にすればよい。

Linear weightを

$$
W_r
\in
\mathbb R^{D_{out}\times D_{in}}
$$

とする。

rankが $r$ 以下なら

$$
W_r=BA
$$

と書ける。

$$
A\in\mathbb R^{r\times D_{in}},
\qquad
B\in\mathbb R^{D_{out}\times r}
$$

なのでshapeは

$$
(D_{out}\times r)(r\times D_{in})
=
D_{out}\times D_{in}
$$

へ戻る。

入力$x$に対して

$$
h=Ax
$$

$$
y=Bh
$$

とすれば

$$
\begin{aligned}
y
&=B(Ax)\\
&=(BA)x\\
&=W_rx.
\end{aligned}
$$

---

## 3. SVDをrank-1成分へ展開する

Reduced SVDを

$$
W=U\Sigma V^{\mathsf T},
\qquad
k=\min(m,n)
$$

とする。

特異値は

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_k\ge0.
$$

対角行列を

$$
\Sigma
=
\sum_{i=1}^{k}
\sigma_i e_i e_i^{\mathsf T}
$$

と展開すると

$$
\begin{aligned}
W
&=U\Sigma V^{\mathsf T}\\
&=U
\left(
\sum_{i=1}^{k}
\sigma_i e_i e_i^{\mathsf T}
\right)
V^{\mathsf T}\\
&=
\sum_{i=1}^{k}
\sigma_i
(Ue_i)
(e_i^{\mathsf T}V^{\mathsf T})\\
&=
\boxed{
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
}.
\end{aligned}
$$

各項

$$
\sigma_i u_i v_i^{\mathsf T}
$$

はrank 1である。

正確には $\sigma_i>0$ の項がrank 1であり、$\sigma_i=0$ の項は零行列でrank 0である。外積の成分を明示すると、

$$
[\sigma_i u_i v_i^{\mathsf T}]_{a,b}
=\sigma_i(u_i)_a(v_i)_b,
\qquad
[\sigma_i u_i v_i^{\mathsf T}]_{:,b}
=\sigma_i(v_i)_b\,u_i.
$$

非ゼロ項ではすべての列が同じ $u_i$ の倍数で、少なくとも1列が非ゼロなので列空間は1次元となる。これがrank-1和という読み方の根拠である。

---

## 4. truncated SVD

上位$r$成分だけ残す。

$$
\boxed{
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
}
$$

行列形式では

$$
\boxed{
W_r
=U_r\Sigma_rV_r^{\mathsf T}
}
$$

である。

$$
U_r\in\mathbb R^{m\times r},
\quad
\Sigma_r\in\mathbb R^{r\times r},
\quad
V_r^{\mathsf T}\in\mathbb R^{r\times n}
$$

なので

$$
(m\times r)(r\times r)(r\times n)
=m\times n.
$$

元と同じshapeへ戻るが、内部自由度は$r$方向だけに制限される。

---

## 5. 入力に作用させたときの意味

一つの入力を通す計算では、$v_i$ は入力方向、$u_i$ は出力方向である。
両方が同じ空間の基底とは限らず、$W$ が長方形なら長さも違う。
例えば $W\in\mathbb R^{m\times n}$ では $v_i\in\mathbb R^n$、$u_i\in\mathbb R^m$ なので、
「同じ方向をそのまま $\sigma_i$ 倍する」のではなく、入力の特異方向を対応する出力の特異方向へ送る。

truncated SVDは元ベクトルの先頭 $r$ 成分を残す操作でもない。
保持するのは、座標軸ではなく右特異基底で測った $v_i^{\mathsf T}x$ のうち最初の $r$ 個である。
もし $v_i$ が元座標の複数成分を持つなら、保持係数はそれら全成分の線形結合になる。
この違いを意識すると、「rankを小さくしたのに入力・出力shapeは同じ」という意味を理解しやすい。

SVDの関係

$$
Wv_i=\sigma_i u_i
$$

から、任意入力$x$に対して

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

したがって

```text
v_i^T x
→ 入力が右特異方向v_iへ持つ係数

σ_i
→ その方向の拡大率

u_i
→ 対応する出力方向
```

と読める。

truncated SVDはこの和の後半を捨てる。

$$
W_rx
=
\sum_{i=1}^{r}
\sigma_i(v_i^{\mathsf T}x)u_i.
$$

---

## 6. Frobenius normと特異値二乗和

Frobenius normは

$$
\|W\|_F^2
=
\operatorname{tr}(W^{\mathsf T}W).
$$

SVDを代入すると

$$
\begin{aligned}
W^{\mathsf T}W
&=
V\Sigma^{\mathsf T}U^{\mathsf T}
U\Sigma V^{\mathsf T}\\
&=
V\Sigma^{\mathsf T}\Sigma V^{\mathsf T}.
\end{aligned}
$$

したがって

$$
\begin{aligned}
\|W\|_F^2
&=
\operatorname{tr}
(V\Sigma^{\mathsf T}\Sigma V^{\mathsf T})\\
&=
\operatorname{tr}
(\Sigma^{\mathsf T}\Sigma V^{\mathsf T}V)\\
&=
\operatorname{tr}(\Sigma^{\mathsf T}\Sigma)\\
&=
\boxed{
\sum_{i=1}^{k}\sigma_i^2
}.
\end{aligned}
$$

このため、過去資料で使う「特異値energy」は $\sigma_i$ の単純和ではなく二乗和で測る。

---

## 7. truncated SVDのFrobenius誤差

$$
W
=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
$$

$$
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
$$

なので

$$
\begin{aligned}
W-W_r
&=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
-
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}\\
&=
\sum_{i=r+1}^{k}
\sigma_i u_i v_i^{\mathsf T}.
\end{aligned}
$$

rank-1成分はFrobenius内積で直交する。

$$
\begin{aligned}
\left\langle
u_i v_i^{\mathsf T},
u_jv_j^{\mathsf T}
\right\rangle_F
&=
\operatorname{tr}
[(u_iv_i^{\mathsf T})^{\mathsf T}
(u_jv_j^{\mathsf T})]\\
&=(u_i^{\mathsf T}u_j)
(v_i^{\mathsf T}v_j)\\
&=\delta_{ij}.
\end{aligned}
$$

したがって

$$
\boxed{
\|W-W_r\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2
}
$$

であり

$$
\boxed{
\|W-W_r\|_F
=
\sqrt{
\sum_{i=r+1}^{k}\sigma_i^2
}
}.
$$

$r=k$なら空和なので誤差は0。

### $2\times2$ 行列でrank-1近似を全要素確認

一般式が各要素でどう働くかを見るため、対称行列

$$
W
:=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

を使う。この行列のSVDの一つは

$$
U
=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix},
\qquad
\Sigma
=
\begin{pmatrix}
4&0\\
0&2
\end{pmatrix},
\qquad
V^{\mathsf T}
=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
$$

である。実際に積を展開すると、

$$
\begin{aligned}
U\Sigma V^{\mathsf T}
&=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
\begin{pmatrix}
4&0\\
0&2
\end{pmatrix}
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}\\
&=
\frac{1}{2}
\begin{pmatrix}
4&2\\
4&-2
\end{pmatrix}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}\\
&=
\frac{1}{2}
\begin{pmatrix}
6&2\\
2&6
\end{pmatrix}\\
&=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
=W.
\end{aligned}
$$

rank 1へ打ち切ると、第1特異成分だけを残すので、

$$
\begin{aligned}
W_1
&=
4
\left(
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1\\
1
\end{pmatrix}
\right)
\left(
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1&1
\end{pmatrix}
\right)\\
&=
4\cdot\frac{1}{2}
\begin{pmatrix}
1&1\\
1&1
\end{pmatrix}\\
&=
\begin{pmatrix}
2&2\\
2&2
\end{pmatrix}.
\end{aligned}
$$

したがって残差は

$$
W-W_1
=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
-
\begin{pmatrix}
2&2\\
2&2
\end{pmatrix}
=
\begin{pmatrix}
1&-1\\
-1&1
\end{pmatrix},
$$

$$
\begin{aligned}
\lVert W-W_1\rVert_F
&=
\sqrt{
1^2+(-1)^2+(-1)^2+1^2
}\\
&=
\sqrt{4}\\
&=2\\
&=\sigma_2.
\end{aligned}
$$

つまり、全要素から計算したrank-1近似誤差は、捨てた特異値の二乗和

$$
\sqrt{\sigma_2^2}=2
$$

と一致する。

---

## 8. retained energyとrelative error

保持energyを

$$
\boxed{
E(r)
=
\frac{
\sum_{i=1}^{r}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}
}
$$

とする。

relative Frobenius errorは

$$
\varepsilon_F(r)
=
\frac{
\|W-W_r\|_F
}{
\|W\|_F
}.
$$

二乗して特異値式を代入すると

$$
\begin{aligned}
\varepsilon_F(r)^2
&=
\frac{
\sum_{i=r+1}^{k}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}\\
&=
\frac{
\sum_{i=1}^{k}\sigma_i^2
-
\sum_{i=1}^{r}\sigma_i^2
}{
\sum_{i=1}^{k}\sigma_i^2
}\\
&=1-E(r).
\end{aligned}
$$

したがって

$$
\boxed{
\varepsilon_F(r)
=
\sqrt{1-E(r)}
}.
$$

energy保持率はweight近似の指標であり、accuracy保持率ではない。

---

## 9. Eckart–Young–Mirsky

rankが$r$以下という制約の下で、truncated SVDはFrobenius normとspectral normの最小誤差を達成する。

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F
}
$$

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
}
$$

`=`ではなく`∈ argmin`と書くのは、境界特異値が重複する場合などに最適解が一意とは限らないためである。

Frobenius normの最小値は

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2
}.
$$

spectral normでは $r<k$ のとき

$$
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
=
\sigma_{r+1}
}
$$

である。

$r=k$なら

$$
W_k=W,
\qquad
\|W-W_k\|_2=0.
$$

> [!note] 補足導出
> 添付資料では主に「truncated SVDが固定rankで最良近似」という結論を使っている。最適解の非一意性や `∈ argmin` の表記は数学的に厳密化するための補足である。証明スケッチは [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]] に置く。

### 別の候補行列でも尾部誤差より小さくできない理由

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

## 10. spectral normから1入力のoutput errorを評価する

差分を

$$
\Delta W=W-W_r
$$

とする。

出力差は

$$
\Delta y
=\Delta Wx.
$$

spectral normの定義から

$x\ne0$ なら単位方向 $z:=x/\|x\|_2$ を使って

$$
\begin{aligned}
\|\Delta Wx\|_2
&=\|\Delta W(\|x\|_2z)\|_2\\
&=\|x\|_2\|\Delta Wz\|_2\\
&\le\|x\|_2\max_{\|u\|_2=1}\|\Delta Wu\|_2\\
&=\|x\|_2\|\Delta W\|_2.
\end{aligned}
$$

$x=0$ では両辺が0である。従って全ての入力について

$$
\|\Delta Wx\|_2
\le
\|\Delta W\|_2\|x\|_2.
$$

$r<k$なら

$$
\|\Delta W\|_2
=
\sigma_{r+1}
$$

なので

$$
\boxed{
\|(W-W_r)x\|_2
\le
\sigma_{r+1}\|x\|_2
}.
$$

$r=k$なら出力差は0。

---

## 11. SVD基底でoutput errorを直接見る

入力を右特異ベクトル方向へ分解する。

$$
x
=
\sum_{i=1}^{k}\alpha_i v_i+x_{\perp},
\qquad
\alpha_i=v_i^{\mathsf T}x.
$$

ここで $x_{\perp}$ はReduced SVDで明示した右特異ベクトル部分空間に直交する成分である。$m<n$ の場合はnullspace側成分を含み、$Wx_{\perp}=0$ となる。

したがって

$$
Wx
=
\sum_{i=1}^{k}
\sigma_i\alpha_i u_i,
$$

$$
W_rx
=
\sum_{i=1}^{r}
\sigma_i\alpha_i u_i.
$$

差は

$$
\boxed{
(W-W_r)x
=
\sum_{i=r+1}^{k}
\sigma_i\alpha_i u_i
}.
$$

つまりoutput errorは特異値だけでなく、実入力が捨てた右特異方向へどれだけ成分を持つかにも依存する。

---

## 12. バッチ出力誤差

PyTorch型のバッチ入力を

$$
X\in\mathbb R^{N\times D_{in}}
$$

とする。

元出力と圧縮後出力は

$$
Y=XW^{\mathsf T}+b,
$$

$$
Y_r=XW_r^{\mathsf T}+b.
$$

同じbiasを使うなら

$$
\begin{aligned}
Y-Y_r
&=XW^{\mathsf T}+b
-
(XW_r^{\mathsf T}+b)\\
&=X(W-W_r)^{\mathsf T}.
\end{aligned}
$$

したがって

$$
\boxed{
E_Y
=X(W-W_r)^{\mathsf T}
}.
$$

出力次元を $D_{out}$ とすると全要素MSEは

$$
\boxed{
\operatorname{MSE}
=
\frac{
\|Y-Y_r\|_F^2
}{
ND_{out}
}
}
$$

RMSEは

$$
\boxed{
\operatorname{RMSE}
=
\frac{
\|Y-Y_r\|_F
}{
\sqrt{ND_{out}}
}
}.
$$

---

## 13. 入力分布を考慮した期待output error

$$
\Delta W=W-W_r
$$

とする。

1入力の二乗誤差は

$$
\begin{aligned}
\|\Delta Wx\|_2^2
&=(\Delta Wx)^{\mathsf T}(\Delta Wx)\\
&=x^{\mathsf T}\Delta W^{\mathsf T}\Delta Wx.
\end{aligned}
$$

scalarをtraceで書き

$$
\begin{aligned}
x^{\mathsf T}\Delta W^{\mathsf T}\Delta Wx
&=
\operatorname{tr}
(x^{\mathsf T}\Delta W^{\mathsf T}\Delta Wx)\\
&=
\operatorname{tr}
(\Delta Wxx^{\mathsf T}\Delta W^{\mathsf T}).
\end{aligned}
$$

期待値を取る。

$$
\begin{aligned}
\mathbb E[\|\Delta Wx\|_2^2]
&=
\operatorname{tr}
\left[
\Delta W\,
\mathbb E[xx^{\mathsf T}]\,
\Delta W^{\mathsf T}
\right].
\end{aligned}
$$

$$
C_x=\mathbb E[xx^{\mathsf T}]
$$

と置けば

$$
\boxed{
\mathbb E[\|(W-W_r)x\|_2^2]
=
\operatorname{tr}
[(W-W_r)C_x(W-W_r)^{\mathsf T}]
}.
$$

$C_x$ は一般には共分散ではなく**非中心化二次モーメント**である。

$$
\operatorname{Cov}(x)
=
\mathbb E[(x-\mu)(x-\mu)^{\mathsf T}],
\qquad
\mu=\mathbb E[x].
$$

$\mu=0$なら両者が一致する。

> [!note] 補足導出
> 入力分布を含むtrace式は、過去資料にある「weight誤差と実データoutput誤差は別」という考察を数式として接続するための補足である。

---

## 14. 厳密低rankと近似的低rank

ある $r$ より後の特異値がすべて0なら

$$
\sigma_{r+1}
=
\sigma_{r+2}
=
\cdots
=0
$$

なので

$$
\|W-W_r\|_F=0
$$

かつ

$$
W=W_r.
$$

一方、特異値が0ではないが

$$
\sigma_1\ge\cdots\ge\sigma_r
\gg
\sigma_{r+1}\ge\cdots
$$

のように急減する場合、数学的にはfull rankでも近似的な低rank構造を持つと考えられる。

---

## 15. Linear圧縮のfactor

学習済みweightを

$$
W\in\mathbb R^{D_{out}\times D_{in}}
$$

とする。

SVD低rank近似

$$
W_r=U_r\Sigma_rV_r^{\mathsf T}
$$

に対し、例えば

$$
A
=
\Sigma_rV_r^{\mathsf T}
\in
\mathbb R^{r\times D_{in}},
$$

$$
B
=
U_r
\in
\mathbb R^{D_{out}\times r}
$$

と置くと

$$
\begin{aligned}
BA
&=U_r(\Sigma_rV_r^{\mathsf T})\\
&=U_r\Sigma_rV_r^{\mathsf T}\\
&=W_r.
\end{aligned}
$$

元Linear

$$
y=Wx+b
$$

を

$$
h=Ax,
$$

$$
y=Bh+b
$$

へ置換すれば

$$
y=BAx+b=W_rx+b.
$$

2因子の間にReLU等を入れると

$$
B\phi(Ax)
$$

となり $BAx$ ではなくなるため、単純なSVD置換では入れない。

---

## 16. parameter数と圧縮条件

元weight数は

$$
P_{orig,W}
=D_{in}D_{out}.
$$

2因子は

$$
P_A=rD_{in},
$$

$$
P_B=D_{out}r.
$$

したがって

$$
\begin{aligned}
P_{low,W}
&=rD_{in}+D_{out}r\\
&=r(D_{in}+D_{out}).
\end{aligned}
$$

biasを後段へ元と同じ $D_{out}$ 個だけ持たせるなら

$$
P_{orig,total}
=D_{in}D_{out}+D_{out}
$$

$$
P_{low,total}
=r(D_{in}+D_{out})+D_{out}.
$$

圧縮条件は

$$
\begin{aligned}
&r(D_{in}+D_{out})+D_{out}
<
D_{in}D_{out}+D_{out}\\
\Longleftrightarrow\;&
r(D_{in}+D_{out})
<
D_{in}D_{out}\\
\Longleftrightarrow\;&
\boxed{
r
<
\frac{D_{in}D_{out}}
{D_{in}+D_{out}}
}.
\end{aligned}
$$

最大整数圧縮rankは

$$
\boxed{
r_{max,compress}
=
\left\lceil
\frac{D_{in}D_{out}}
{D_{in}+D_{out}}
\right\rceil-1
}.
$$

---

## 17. 理論MACs

1サンプルの元Linear主MACsは

$$
\operatorname{MACs}_{orig}
=D_{in}D_{out}.
$$

2因子では

$$
\operatorname{MACs}_{A}
=rD_{in},
$$

$$
\operatorname{MACs}_{B}
=rD_{out}.
$$

したがって

$$
\boxed{
\operatorname{MACs}_{low}
=r(D_{in}+D_{out})
}.
$$

parameter数が減る条件と同じ不等式になるが、wall-clock latencyが同じ割合で減ることは保証しない。

---

## 18. rank選択

rankは1つの指標だけで決めない。

候補として

- retained energy $E(r)$
- relative Frobenius error $\varepsilon_F(r)$
- validation loss / accuracy
- parameters
- MACs
- fine-tuning後性能
- latency

を分けて見る。

energy thresholdなら

$$
r^*
=
\min\{r\mid E(r)\ge\tau\}
$$

と書ける。

ただし $E(r)$ はweight近似指標なので、task上の最適rankを保証しない。

---

### 元教材の「なぜ低rankを期待するか」と、保証しないこと

元教材 `03_SVDによる低ランク近似.md` は、学習済み重みが必ず低rankになるとは主張していない。低rank近似を試す理由として、隠れユニットが似た特徴を学ぶこと、入力が高次元空間の一部へ集中すること、モデルに余分な幅があること、複数出力が特徴を共有することを挙げている。また、最適化で作用が一部の方向へ集中したり、正則化で構造が単純になったり、学習データに現れない方向が十分使われなかったりする可能性もある。

これらは低rank性の証明ではなく、冗長性が生じ得る仕組みである。特異値スペクトルの急減として現れるかは実際の層ごとに確認する。初期重みと学習後のスペクトルを比較することは補助的な観察になるが、その比較だけで「低rank化したから汎化した」「小さい特異値はノイズ」と断定しない。

小さい特異値の方向にも、分類境界の微調整、特定クラスの識別、まれな入力パターン、後続層との組合せに必要な情報がある可能性がある。従ってスペクトルの折れ曲がりはrank候補であって、出力誤差・validation性能を確認しない最終決定ではない。層によって必要なrankが異なることと、小さい層では2層化の利点が小さいことも、元教材の注意として残す。

さらに、NNの重み行列を分解した特異値は、線形写像の方向ごとの拡大率である。重みは通常ハミルトニアンではなく、特異値をエネルギー準位と呼んだり、左右特異ベクトルを量子状態と断定したりしてはいけない。波動関数の係数行列をSVDした場合との数学的な対応は [[30_TT_MPSの定義#5. 行列SVDからSchmidt分解へ]] で扱うが、分解対象の違いを消して同一視するわけではない。ここではDMRGの局所最適化やsweepへは進まない。

元教材で並べた選び方は、固定rank、energy閾値、スペクトルの折れ曲がり、目標パラメータ数、validationで許容できる精度低下、fine-tuning後の性能である。圧縮直後のSVDは行列ノルムを基準にしているが、fine-tuningは入力分布・後続層・損失・分類境界に合わせ、低rankの因子をタスクへ再適応させる。因子の中間幅 $r$ を変えない限り、再学習しても合成重みのrank上限は $r$ のままである。各選び方と最終評価の分離は [[05_圧縮率とRank]]、[[10_SVD圧縮モデルの評価設計]]、因子の学習とrank制約は [[04_Linear層を2層へ置き換える#44. fine-tuning後もrank制約が保たれる理由]] に対応する。

## 19. 実装との対応

PyTorchのReduced SVDは

```python
U, S, Vh = torch.linalg.svd(
    weight,
    full_matrices=False,
)
```

rank$r$なら

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
```

である。

再構成は

```python
W_r = (U_r * S_r.unsqueeze(0)) @ Vh_r
```

で

$$
(U_r\Sigma_r)V_r^{\mathsf T}
$$

に対応する。

full rank

$$
r=k=\min(m,n)
$$

なら数値丸め誤差を除いて

$$
W_k=W.
$$

---

## 20. このノートで押さえるポイント

```text
SVD
→ rank-1和
→ 上位r成分だけ残す
→ W_r

W-W_r
→ 捨てたrank-1成分の和
→ Frobenius直交性
→ tail σ_i^2

retained energy
→ relative Frobenius error

spectral norm
→ 1入力の最悪方向error上限

実入力分布
→ E[xx^T]
→ weight errorとtask/output errorは別

W_r=BA
→ 2層Linear
→ parameter/MAC条件
```

関連：

- [[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]]
