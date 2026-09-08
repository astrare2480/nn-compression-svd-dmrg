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
