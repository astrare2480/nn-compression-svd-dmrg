---
title: Tucker・HOSVD・HOOI数式の導出
aliases:
  - Tucker途中式
  - HOSVD途中式
  - HOOI途中式
  - Tucker2途中式
tags:
  - Tucker
  - HOSVD
  - HOOI
  - Tensor
  - 数式導出
---

# Tucker・HOSVD・HOOI数式の導出

## 位置づけ

Tucker / HOSVD / HOOIで使う式を、shapeと途中式を飛ばさず一本にまとめる。

このノートではPyTorchのaxis番号に合わせ、modeを0始まりで数える。

$$
\mathcal X
\in
\mathbb R^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

Conv2d weightでは

$$
W
\in
\mathbb R^{C_{out}\times C_{in}\times K_h\times K_w}
$$

として

```text
mode 0 = C_out
mode 1 = C_in
mode 2 = K_h
mode 3 = K_w
```

とする。

---

## 1. mode-n unfolding

mode $n$ を行方向へ置き、残りのmodeを列方向へまとめる。

$$
\boxed{
X_{(n)}
\in
\mathbb R^{I_n\times\prod_{m\ne n}I_m}
}
$$

例えば

$$
W\in\mathbb R^{64\times32\times3\times3}
$$

なら

$$
W_{(0)}
\in
\mathbb R^{64\times(32\cdot3\cdot3)}
=
\mathbb R^{64\times288}
$$

$$
W_{(1)}
\in
\mathbb R^{32\times(64\cdot3\cdot3)}
=
\mathbb R^{32\times576}.
$$

unfoldは並べ替えであり、要素数を変えない。

$$
64\cdot32\cdot3\cdot3
=18432
=
64\cdot288.
$$

---

## 2. mode-n product

行列

$$
A\in\mathbb R^{J\times I_n}
$$

をmode $n$ へ掛ける操作を

$$
\mathcal Y
=
\mathcal X\times_nA
$$

と書く。

unfold表示では

$$
\boxed{
Y_{(n)}
=AX_{(n)}
}
$$

である。

shapeは

$$
(J\times I_n)
\left(I_n\times\prod_{m\ne n}I_m\right)
=
J\times\prod_{m\ne n}I_m.
$$

したがってfold後はmode $n$ だけが

$$
I_n\rightarrow J
$$

へ変わる。

### Conv mode 1の具体例

$$
U_{in}
\in
\mathbb R^{32\times R_{in}}
$$

なら

$$
U_{in}^{\mathsf T}
\in
\mathbb R^{R_{in}\times32}.
$$

よって

$$
\begin{aligned}
U_{in}^{\mathsf T}W_{(1)}
&:
(R_{in}\times32)(32\times576)\\
&\rightarrow R_{in}\times576.
\end{aligned}
$$

foldすると

$$
\boxed{
W\times_1U_{in}^{\mathsf T}
\in
\mathbb R^{64\times R_{in}\times3\times3}
}.
$$

---

## 3. factorの転置が圧縮方向になる理由

factorを

$$
U^{(n)}
\in
\mathbb R^{I_n\times R_n}
$$

とする。

元空間のvector

$$
x\in\mathbb R^{I_n}
$$

をrank空間へ射影すると

$$
\begin{aligned}
z
&=U^{(n)\mathsf T}x,\\
(R_n\times I_n)(I_n\times1)
&\rightarrow R_n\times1.
\end{aligned}
$$

逆にrank空間から元空間へ戻すと

$$
\begin{aligned}
x_{recon}
&=U^{(n)}z,\\
(I_n\times R_n)(R_n\times1)
&\rightarrow I_n\times1.
\end{aligned}
$$

したがって

```text
元空間 → rank空間 : U^T
rank空間 → 元空間 : U
```

である。

---

## 4. Tucker分解とcore

直交factorを用いるTucker近似を

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

とする。

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
$$

なら、factorが固定されたときのcoreは

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

で得られる。

partial Tuckerでは圧縮対象mode集合を

$$
\mathcal M
\subseteq
\{0,1,\ldots,N-1\}
$$

として

$$
\boxed{
\mathcal G
=
\mathcal X
\underset{m\in\mathcal M}{\times_m}
U^{(m)\mathsf T}
}
$$

とする。

---

## 5. Tucker-2 coreのshape

Conv weightでmode 0 / 1だけを圧縮する。

$$
\mathcal M=\{0,1\}.
$$

$$
U_{out}
\in
\mathbb R^{64\times R_{out}},
\qquad
U_{in}
\in
\mathbb R^{32\times R_{in}}.
$$

まずmode 0を射影すると

$$
(64,32,3,3)
\xrightarrow{\times_0U_{out}^{\mathsf T}}
(R_{out},32,3,3).
$$

次にmode 1を射影すると

$$
(R_{out},32,3,3)
\xrightarrow{\times_1U_{in}^{\mathsf T}}
(R_{out},R_{in},3,3).
$$

したがって

$$
\boxed{
G
=
W
\times_0U_{out}^{\mathsf T}
\times_1U_{in}^{\mathsf T}
}
$$

$$
\boxed{
G
\in
\mathbb R^{R_{out}\times R_{in}\times3\times3}
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
}
$$

で元shapeへ戻る。

---

## 6. HOSVDを1 stepずつ追う

HOSVDでは各factorを**元の同じTensorから独立に**求める。

### mode 0

$$
W_{(0)}
=U_0\Sigma_0V_0^{\mathsf T}
$$

とSVDする。

数学的には上位 $R_{out}$ 本を

$$
\boxed{
U_{out}
=
\begin{bmatrix}
u_{0,1}&u_{0,2}&\cdots&u_{0,R_{out}}
\end{bmatrix}
}
$$

と取る。

Pythonでは

```python
U_out = U0[:, :R_out]
```

である。

> [!important]
> `U0[:, 1:R_out]` ではない。Python sliceは0始まりなので、それでは第1左特異ベクトルを落とし、本数も $R_{out}-1$ になる。

### mode 1

mode 0でprojectしたTensorではなく、**再び元の $W$** をmode 1でunfoldする。

$$
W_{(1)}
=U_1\Sigma_1V_1^{\mathsf T}.
$$

上位 $R_{in}$ 本を

$$
\boxed{
U_{in}
=
\begin{bmatrix}
u_{1,1}&u_{1,2}&\cdots&u_{1,R_{in}}
\end{bmatrix}
}
$$

と取る。

Pythonでは

```python
U_in = U1[:, :R_in]
```

である。

最後に両factorで元 $W$ を射影してcoreを作る。

```text
元W → mode0 unfold → SVD → U_out
元W → mode1 unfold → SVD → U_in

元W
→ ×0 U_out.T
→ ×1 U_in.T
→ core
```

この「factor計算中は相互依存しない」点がHOOIとの違いである。

---

## 7. Tucker-2の要素表示

Tucker-2 weightは

$$
\boxed{
\hat W_{o,i,a,b}
=
\sum_{\alpha=1}^{R_{out}}
\sum_{\beta=1}^{R_{in}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
}
$$

である。

channelの流れは

$$
\boxed{
i\rightarrow\beta\rightarrow\alpha\rightarrow o
}
$$

となる。

---

## 8. Tucker-2を3層Convへ展開する

### input projection

$$
\boxed{
Z_{\beta,h,w}
=
\sum_{i=1}^{C_{in}}
U^{in}_{i,\beta}
X_{i,h,w}
}
$$

これはchannelだけを混ぜる1x1 Convである。

weight shapeは

$$
(R_{in},C_{in},1,1)
$$

で、行列としては $U_{in}^{\mathsf T}$ を使う。

### core convolution

簡略化してstride 1 / dilation 1の場合

$$
\boxed{
T_{\alpha,h,w}
=
\sum_{\beta=1}^{R_{in}}
\sum_a\sum_b
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
}
$$

である。

### output projection

$$
\boxed{
Y_{o,h,w}
=
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
T_{\alpha,h,w}
+b_o
}
$$

である。

この3式を代入すると

$$
\begin{aligned}
Y_{o,h,w}
&=
\sum_{i,a,b}
\left[
\sum_{\alpha,\beta}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
\right]
X_{i,h+a,w+b}
+b_o.
\end{aligned}
$$

角括弧内がeffective weight $\hat W_{o,i,a,b}$ である。

stride / padding / dilationを含む一般形は
[[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
で扱う。

---

## 9. Tucker-2 parameter数

元Conv weightのparameter数は

$$
\boxed{
N_{orig}
=C_{out}C_{in}K_hK_w
}.
$$

Tucker-2は

$$
\boxed{
N_{Tucker2}
=C_{in}R_{in}
+R_{out}R_{in}K_hK_w
+C_{out}R_{out}
}.
$$

今回の

$$
(C_{out},C_{in},K_h,K_w)
=(64,32,3,3)
$$

なら

$$
N_{orig}
=64\cdot32\cdot3\cdot3
=18432.
$$

balanced rank $(R_{out},R_{in})=(32,16)$ では

$$
\begin{aligned}
N_{Tucker2}
&=32\cdot16
+32\cdot16\cdot3\cdot3
+64\cdot32\\
&=512+4608+2048\\
&=7168.
\end{aligned}
$$

したがって対象Conv weightの削減率は

$$
\begin{aligned}
1-\frac{7168}{18432}
&=0.611111\ldots\\
&\approx61.11\%.
\end{aligned}
$$

rankを元channel数のまま

$$
(R_{out},R_{in})=(64,32)
$$

とすると

$$
32\cdot32
+64\cdot32\cdot9
+64\cdot64
=23552
$$

となり、元18432より増える。

つまりTucker形式にしただけでは圧縮にはならない。

---

## 10. HOOIの目的関数

標準的な直交Tucker近似では

$$
\boxed{
\min
\left\|
\mathcal X
-
\mathcal G
\times_0U^{(0)}
\cdots
\times_{N-1}U^{(N-1)}
\right\|_F^2
}
$$

subject to

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}.
$$

HOOIはrankを固定したままfactorを1 modeずつ更新する。

---

## 11. 誤差最小化とcore norm最大化

現在のfactorで作る直交射影を $\hat{\mathcal X}$ とする。

$$
\begin{aligned}
\|\mathcal X-\hat{\mathcal X}\|_F^2
={}&
\|\mathcal X\|_F^2
-2\langle\mathcal X,\hat{\mathcal X}\rangle_F
+\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

残差 $\mathcal X-\hat{\mathcal X}$ は射影部分と直交するため

$$
\langle\mathcal X,\hat{\mathcal X}\rangle_F
=
\|\hat{\mathcal X}\|_F^2.
$$

よって

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\hat{\mathcal X}\|_F^2
}.
$$

列直交factorはFrobenius normを保つので

$$
\|\hat{\mathcal X}\|_F
=
\|\mathcal G\|_F.
$$

したがって

$$
\boxed{
\min\|\mathcal X-\hat{\mathcal X}\|_F^2
\Longleftrightarrow
\max\|\mathcal G\|_F^2
}.
$$

---

## 12. HOOIの1 factor更新

更新対象をmode $n$ とする。

active mode集合を $\mathcal M$ とすれば、更新対象以外を現在factorで射影する。

$$
\boxed{
\mathcal Z^{(n)}
=
\mathcal X
\underset{m\in\mathcal M\setminus\{n\}}{\times_m}
U^{(m)\mathsf T}
}.
$$

mode $n$ unfoldingを

$$
Z=Z_{(n)}^{(n)}
$$

と書く。

候補factor $U$ による最終射影のcore unfoldingは

$$
G_{(n)}
=U^{\mathsf T}Z.
$$

したがって局所問題は

$$
\max_{U^{\mathsf T}U=I}
\|U^{\mathsf T}Z\|_F^2.
$$

trace表示すると

$$
\begin{aligned}
\|U^{\mathsf T}Z\|_F^2
&=
\operatorname{tr}
\left[
(U^{\mathsf T}Z)(U^{\mathsf T}Z)^{\mathsf T}
\right]\\
&=
\operatorname{tr}
(U^{\mathsf T}ZZ^{\mathsf T}U).
\end{aligned}
$$

SVDを

$$
Z=Q\Sigma V^{\mathsf T}
$$

とすると

$$
ZZ^{\mathsf T}
=Q\Sigma\Sigma^{\mathsf T}Q^{\mathsf T}.
$$

したがって、最大の $R_n$ 個の固有値に対応する $Q$ の列を取ればよい。

$$
\boxed{
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\operatorname{leftSVD}
(Z_{(n)}^{(n)})
}.
$$

Pythonなら

```python
U_new = Q[:, :R_n]
```

である。

---

## 13. Tucker-2 HOOIを $t\rightarrow t+1$ で追う

初期値はHOSVD。

$$
U_{out}^{(0)},
\qquad
U_{in}^{(0)}.
$$

### 13.1 $U_{out}$ 更新

$$
Z_{out}^{(t)}
=
W\times_1
(U_{in}^{(t)})^{\mathsf T}.
$$

$$
U_{out}^{(t+1)}
=
\operatorname{top\text{-}}R_{out}
\operatorname{leftSVD}
\left[(Z_{out}^{(t)})_{(0)}\right].
$$

### 13.2 $U_{in}$ 更新

同じsweep内で更新済みの $U_{out}^{(t+1)}$ を使う。

$$
Z_{in}^{(t)}
=
W\times_0
(U_{out}^{(t+1)})^{\mathsf T}.
$$

$$
U_{in}^{(t+1)}
=
\operatorname{top\text{-}}R_{in}
\operatorname{leftSVD}
\left[(Z_{in}^{(t)})_{(1)}\right].
$$

この最新factorを同一sweepで使う更新が、現在srcのGauss-Seidel型 `hooi_sweep()` に対応する。

---

## 14. core・再構成・error

$$
\boxed{
G^{(t+1)}
=
W
\times_0(U_{out}^{(t+1)})^{\mathsf T}
\times_1(U_{in}^{(t+1)})^{\mathsf T}
}
$$

$$
\boxed{
\hat W^{(t+1)}
=
G^{(t+1)}
\times_0U_{out}^{(t+1)}
\times_1U_{in}^{(t+1)}
}
$$

relative Frobenius errorは

$$
\boxed{
e_{t+1}
=
\frac{
\|W-\hat W^{(t+1)}\|_F
}{
\|W\|_F
}
}.
$$

---

## 15. 収束判定

単純な絶対差なら

$$
|e_t-e_{t-1}|<\mathrm{tol}
$$

である。

現在srcでは

$$
\boxed{
|e_t-e_{t-1}|
\le
\varepsilon_{abs}
+
\varepsilon_{rel}|e_{t-1}|
}
$$

を使う。

例えば

$$
\varepsilon_{abs}=10^{-8},
\qquad
\varepsilon_{rel}=10^{-5}
$$

かつ

$$
e_{t-1}\approx0.44
$$

なら右辺は

$$
10^{-8}+10^{-5}\cdot0.44
\approx4.41\times10^{-6}.
$$

---

## 16. 実験値を式で確認する

balanced rank $(32,16)$ では

$$
e_{HOSVD}=0.449042
$$

$$
e_{HOOI}=0.442504.
$$

絶対改善量は

$$
\Delta e
=0.449042-0.442504
=0.006538.
$$

HOSVD error基準の相対改善は

$$
\frac{0.006538}{0.449042}
\approx0.01456
\approx1.46\%.
$$

一方、圧縮直後validation accuracyは

$$
0.6280-0.6202
=0.0078
$$

だけHOSVDの方が高かった。

したがって、このrunでは

$$
\boxed{
\text{lower weight error}
\not\Rightarrow
\text{higher task accuracy}
}
$$

を確認した。

---

## 17. fine-tuning前後

同条件比較ではtest accuracyが

$$
0.7555-0.7508
=0.0047
$$

だけHOOI初期化側で高かった。

これは0.47 percentage pointだが、seed 0の1runだけなので統計的優位性とはしない。

weight errorはHOSVDで

$$
0.483660-0.449042
=0.034618
$$

増え、HOOIで

$$
0.481255-0.442504
=0.038751
$$

増えた。

それでもaccuracyは回復した。

これはfine-tuningが

$$
\min\|W-\hat W\|_F
$$

ではなくtask lossを最適化していることと整合する。

---

## 18. このノートの結論

```text
unfold
→ modeを行方向へ出す
→ shapeを確認

mode product
→ Y_(n)=A X_(n)
→ foldして対象modeだけ置換

HOSVD
→ 各modeを元Tensorから独立にSVD
→ top-R left singular vectors
→ Pythonでは [:, :R]
→ factor^Tでcore
→ factorで再構成

Tucker-2
→ C_in → R_in → R_out → C_out
→ 1x1 → kxk → 1x1

HOOI
→ HOSVD初期化
→ 他factorでproject
→ target modeをunfold
→ local objectiveをSVDで最適化
→ 同一sweepでは最新factorを使用
→ core / reconstruction / error
→ convergence
```

関連：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/21_テンソルとmode演算]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
- [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]
