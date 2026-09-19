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

Tucker / HOSVD / HOOIで使う式を、**定義 → shape → 途中式 → 最終式**の順で追えるようにまとめる。

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

> [!important]
> 直交射影からcore norm最大化への変形、HOOIの局所更新が上位左特異ベクトルになる理由を、途中式で確認する。各更新の最適性と、Tucker近似全体のglobal optimumは区別する。

---

## 1. mode-n unfolding

$N$階Tensor

$$
\mathcal X
\in
\mathbb R^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

に対し、mode $n$ を行方向へ置き、残りのmodeを列方向へまとめる。

$$
\boxed{
X_{(n)}
\in
\mathbb R^{I_n\times\prod_{m\ne n}I_m}
}
$$

要素数は変わらないので、

$$
I_n
\left(
\prod_{m\ne n}I_m
\right)
=
\prod_{m=0}^{N-1}I_m.
$$

例えば

$$
W\in\mathbb R^{64\times32\times3\times3}
$$

なら

$$
\begin{aligned}
W_{(0)}
&\in
\mathbb R^{64\times(32\cdot3\cdot3)}\\
&=
\mathbb R^{64\times288},
\end{aligned}
$$

$$
\begin{aligned}
W_{(1)}
&\in
\mathbb R^{32\times(64\cdot3\cdot3)}\\
&=
\mathbb R^{32\times576}.
\end{aligned}
$$

要素数を確認すると、

$$
64\cdot32\cdot3\cdot3
=18432,
$$

$$
64\cdot288
=18432,
$$

$$
32\cdot576
=18432.
$$

したがってunfoldは**並べ替え・行列化**であって、要素を捨てる処理ではない。

---

## 2. fold

`fold` はunfoldの逆操作である。

$$
\boxed{
\operatorname{fold}_n
\left(
X_{(n)}
\right)
=
\mathcal X
}
$$

したがって、正しい実装では

$$
\boxed{
\operatorname{fold}_n
\left(
\operatorname{unfold}_n(\mathcal X)
\right)
=
\mathcal X
}
$$

が成立する。

shapeで見ると、

$$
I_n\times\prod_{m\ne n}I_m
$$

という行列を、元の

$$
I_0\times I_1\times\cdots\times I_{N-1}
$$

へ戻す。

---

## 3. mode-n product

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

要素表示では、

$$
\boxed{
Y_{i_0,\ldots,i_{n-1},j,i_{n+1},\ldots,i_{N-1}}
=
\sum_{i_n=1}^{I_n}
A_{j,i_n}
X_{i_0,\ldots,i_n,\ldots,i_{N-1}}
}
$$

である。

これをmode $n$でunfoldすると、

$$
\boxed{
Y_{(n)}
=AX_{(n)}
}
$$

になる。

shapeは

$$
(J\times I_n)
\left(
I_n\times\prod_{m\ne n}I_m
\right)
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

mode 1 unfoldingは

$$
W_{(1)}
\in
\mathbb R^{32\times576}.
$$

したがって、

$$
\begin{aligned}
U_{in}^{\mathsf T}W_{(1)}
&:
(R_{in}\times32)(32\times576)\\
&\rightarrow
R_{in}\times576.
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

## 4. factorの転置が圧縮方向になる理由

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

をrank空間へ射影すると、

$$
\begin{aligned}
z
&=U^{(n)\mathsf T}x,\\
(R_n\times I_n)(I_n\times1)
&\rightarrow R_n\times1.
\end{aligned}
$$

逆にrank空間から元空間へ戻すと、

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

factorの列が直交規格化されていれば、

$$
U^{(n)\mathsf T}U^{(n)}=I_{R_n}.
$$

元空間へ戻してからもう一度射影すると、

$$
U^{(n)\mathsf T}
\left(
U^{(n)}z
\right)
=
\left(
U^{(n)\mathsf T}U^{(n)}
\right)z
=z.
$$

つまりfactor列空間の内部では、この圧縮・展開は整合する。

---

## 5. Tucker分解とcore

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

各factorは

$$
U^{(n)}
\in
\mathbb R^{I_n\times R_n},
\qquad
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
$$

を満たす。

全modeを圧縮する場合、core shapeは

$$
\boxed{
\mathcal G
\in
\mathbb R^{R_0\times R_1\times\cdots\times R_{N-1}}
}.
$$

factorが固定されたとき、元Tensorを各factor列空間へ射影してcoreを得る。

ここでは3階Tensorの各射影を途中配列まで書く。要素添字は1始まりとする。

$$
\begin{aligned}
H_{\alpha_0,i_1,i_2}
&=\sum_{i_0=1}^{I_0}U^{(0)}_{i_0,\alpha_0}X_{i_0,i_1,i_2},\\
J_{\alpha_0,\alpha_1,i_2}
&=\sum_{i_1=1}^{I_1}U^{(1)}_{i_1,\alpha_1}H_{\alpha_0,i_1,i_2},\\
G_{\alpha_0,\alpha_1,\alpha_2}
&=\sum_{i_2=1}^{I_2}U^{(2)}_{i_2,\alpha_2}J_{\alpha_0,\alpha_1,i_2}\\
&=\sum_{i_0,i_1,i_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}
U^{(2)}_{i_2,\alpha_2}X_{i_0,i_1,i_2}.
\end{aligned}
$$

各stageで物理添字を一つずつrank添字へ移している。同じ転置で全modeをまとめて置き換えるだけの記号操作ではない。

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

とする。

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

$$
\boxed{
\hat{\mathcal X}
=
\mathcal G
\underset{m\in\mathcal M}{\times_m}
U^{(m)}
}
$$

とする。

圧縮しないmodeはcoreに元の次元 $I_m$ のまま残る。

---

## 6. Tucker-2 coreのshape

Conv weightでmode 0 / 1だけを圧縮する。

$$
\mathcal M=\{0,1\}.
$$

$$
W
\in
\mathbb R^{C_{out}\times C_{in}\times K_h\times K_w},
$$

$$
U_{out}
\in
\mathbb R^{C_{out}\times R_{out}},
\qquad
U_{in}
\in
\mathbb R^{C_{in}\times R_{in}}.
$$

まずmode 0を射影する。

$$
\begin{aligned}
W^{(0)}
&=
W\times_0U_{out}^{\mathsf T},\\
(C_{out},C_{in},K_h,K_w)
&\rightarrow
(R_{out},C_{in},K_h,K_w).
\end{aligned}
$$

次にmode 1を射影する。

$$
\begin{aligned}
G
&=
W^{(0)}\times_1U_{in}^{\mathsf T}\\
&=
W\times_0U_{out}^{\mathsf T}
\times_1U_{in}^{\mathsf T},\\
(R_{out},C_{in},K_h,K_w)
&\rightarrow
(R_{out},R_{in},K_h,K_w).
\end{aligned}
$$

したがって

$$
\boxed{
G
\in
\mathbb R^{R_{out}\times R_{in}\times K_h\times K_w}
}.
$$

再構成は

$$
\begin{aligned}
\hat W
&=
G\times_0U_{out}\times_1U_{in},\\
(R_{out},R_{in},K_h,K_w)
&\xrightarrow{\times_0U_{out}}
(C_{out},R_{in},K_h,K_w)\\
&\xrightarrow{\times_1U_{in}}
(C_{out},C_{in},K_h,K_w).
\end{aligned}
$$

よって元weightと同じshapeへ戻る。

---

## 7. HOSVDを1 stepずつ追う

HOSVDでは各factorを**元の同じTensorから独立に**求める。

### 7.1 一般mode $n$

mode-$n$ unfoldingを

$$
X_{(n)}
\in
\mathbb R^{I_n\times\prod_{m\ne n}I_m}
$$

とし、Reduced SVDを

$$
\boxed{
X_{(n)}
=Q_n\Sigma_nV_n^{\mathsf T}
}
$$

とする。

左特異ベクトルを

$$
Q_n
=
\begin{bmatrix}
q^{(n)}_1 & q^{(n)}_2 & \cdots
\end{bmatrix}
$$

と書けば、rank $R_n$ のfactorは

$$
\boxed{
U^{(n)}
=
\begin{bmatrix}
q^{(n)}_1 & q^{(n)}_2 & \cdots & q^{(n)}_{R_n}
\end{bmatrix}
}
$$

である。

shapeは

$$
U^{(n)}
\in
\mathbb R^{I_n\times R_n}.
$$

Pythonの0始まりsliceでは

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-001]]

となる。

> [!important]
> 数学の「第1列〜第 $R_n$ 列」と、Pythonの `[:, 1:R_n]` を混同しない。Pythonでは `[:, :R_n]` が上位 $R_n$ 本である。

### 7.2 Conv mode 0

$$
W_{(0)}
=Q_0\Sigma_0V_0^{\mathsf T}.
$$

上位 $R_{out}$ 本を

$$
\boxed{
U_{out}
=
\begin{bmatrix}
q^{(0)}_1 & q^{(0)}_2 & \cdots & q^{(0)}_{R_{out}}
\end{bmatrix}
}
$$

とする。

### 7.3 Conv mode 1

mode 0でprojectしたTensorではなく、**再び元の $W$** をmode 1でunfoldする。

$$
W_{(1)}
=Q_1\Sigma_1V_1^{\mathsf T}.
$$

上位 $R_{in}$ 本を

$$
\boxed{
U_{in}
=
\begin{bmatrix}
q^{(1)}_1 & q^{(1)}_2 & \cdots & q^{(1)}_{R_{in}}
\end{bmatrix}
}
$$

とする。

最後に両factorで**元の $W$** を射影してcoreを作る。

$$
\boxed{
G
=
W\times_0U_{out}^{\mathsf T}
\times_1U_{in}^{\mathsf T}
}
$$

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

## 8. Tucker-2の要素表示

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

この式はmode productを要素で展開したものになる。

まずmode 1を戻すと、

$$
H_{\alpha,i,a,b}
=
\sum_{\beta=1}^{R_{in}}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}.
$$

次にmode 0を戻すと、

$$
\begin{aligned}
\hat W_{o,i,a,b}
&=
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
H_{\alpha,i,a,b}\\
&=
\sum_{\alpha=1}^{R_{out}}
U^{out}_{o,\alpha}
\left(
\sum_{\beta=1}^{R_{in}}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
\right)\\
&=
\sum_{\alpha=1}^{R_{out}}
\sum_{\beta=1}^{R_{in}}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}.
\end{aligned}
$$

channelの流れは

$$
\boxed{
i\rightarrow\beta\rightarrow\alpha\rightarrow o
}
$$

となる。

---

## 9. Tucker-2を3層Convへ展開する

### 9.1 input projection

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

### 9.2 core convolution

stride 1 / dilation 1で境界処理を省略した概念式なら、

$$
\boxed{
T_{\alpha,h,w}
=
\sum_{\beta=1}^{R_{in}}
\sum_{a=0}^{K_h-1}
\sum_{b=0}^{K_w-1}
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
}
$$

である。

### 9.3 output projection

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

### 9.4 3式を代入する

まずcore式をoutput projectionへ代入する。

$$
\begin{aligned}
Y_{o,h,w}
={}&
\sum_{\alpha}
U^{out}_{o,\alpha}
\left[
\sum_{\beta,a,b}
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
\right]
+b_o\\
={}&
\sum_{\alpha,\beta,a,b}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
+b_o.
\end{aligned}
$$

さらにinput projection

$$
Z_{\beta,h+a,w+b}
=
\sum_i
U^{in}_{i,\beta}
X_{i,h+a,w+b}
$$

を代入すると、

$$
\begin{aligned}
Y_{o,h,w}
={}&
\sum_{\alpha,\beta,a,b}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
\left[
\sum_iU^{in}_{i,\beta}X_{i,h+a,w+b}
\right]
+b_o\\
={}&
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

したがって角括弧内がeffective weightであり、

$$
\boxed{
\hat W_{o,i,a,b}
=
\sum_{\alpha,\beta}
U^{out}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{in}_{i,\beta}
}
$$

と一致する。

stride / padding / dilationを含む一般形は
[[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
で扱う。

---

## 10. Tucker-2 parameter数

元Conv weightのparameter数は

$$
\boxed{
N_{orig}
=C_{out}C_{in}K_hK_w
}.
$$

Tucker-2は3つのweightからなる。

入力1x1 Conv：

$$
N_{in}
=C_{in}R_{in}.
$$

core Conv：

$$
N_{core}
=R_{out}R_{in}K_hK_w.
$$

出力1x1 Conv：

$$
N_{out}
=C_{out}R_{out}.
$$

したがって、

$$
\begin{aligned}
N_{Tucker2}
&=N_{in}+N_{core}+N_{out}\\
&=
\boxed{
C_{in}R_{in}
+R_{out}R_{in}K_hK_w
+C_{out}R_{out}
}.
\end{aligned}
$$

biasは元Convと圧縮後の最終1x1 Convでともに $C_{out}$ 個なので、biasを保存する同一条件でweight圧縮成立条件を比較すると相殺できる。

圧縮条件は

$$
N_{Tucker2}<N_{orig}
$$

すなわち

$$
\boxed{
C_{in}R_{in}
+R_{out}R_{in}K_hK_w
+C_{out}R_{out}
<
C_{out}C_{in}K_hK_w
}.
$$

### 今回の具体値

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

balanced rank

$$
(R_{out},R_{in})=(32,16)
$$

では

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

とすると、

$$
\begin{aligned}
N_{Tucker2}
&=32\cdot32
+64\cdot32\cdot9
+64\cdot64\\
&=1024+18432+4096\\
&=23552,
\end{aligned}
$$

となり、

$$
23552-18432=5120
$$

だけ元Convより増える。

つまりTucker形式にしただけでは圧縮にはならない。

---

## 11. Tucker-2 MACs

元Convの出力空間を

$$
H_{out}\times W_{out}
$$

とすると、

$$
\boxed{
\operatorname{MACs}_{orig}
=
H_{out}W_{out}
C_{out}C_{in}K_hK_w
}.
$$

圧縮後は3層を別々に数える。

入力1x1 Convは入力空間で実行されるので、

$$
\operatorname{MACs}_{in}
=
H_{in}W_{in}C_{in}R_{in}.
$$

core Convは、

$$
\operatorname{MACs}_{core}
=
H_{out}W_{out}
R_{out}R_{in}K_hK_w.
$$

出力1x1 Convは、

$$
\operatorname{MACs}_{out}
=
H_{out}W_{out}
R_{out}C_{out}.
$$

したがって一般形は、

$$
\boxed{
\begin{aligned}
\operatorname{MACs}_{Tucker2}
={}&
H_{in}W_{in}C_{in}R_{in}\\
&+
H_{out}W_{out}R_{out}R_{in}K_hK_w\\
&+
H_{out}W_{out}R_{out}C_{out}.
\end{aligned}
}
$$

もし

$$
H_{in}W_{in}=H_{out}W_{out}=HW
$$

なら、

$$
\boxed{
\operatorname{MACs}_{Tucker2}
=
HW
\left(
C_{in}R_{in}
+R_{out}R_{in}K_hK_w
+C_{out}R_{out}
\right)
}.
$$

この簡略化は**空間サイズが実際に同じ場合だけ**使う。stride 1だけでは、paddingやkernel sizeによっては空間サイズが変わるため十分条件ではない。

---

## 12. HOOIの目的関数

標準的な直交Tucker近似では、

$$
\boxed{
\begin{aligned}
\min_{
\mathcal G,
U^{(0)},\ldots,U^{(N-1)}
}
&\quad
\left\|
\mathcal X
-
\mathcal G
\times_0U^{(0)}
\cdots
\times_{N-1}U^{(N-1)}
\right\|_F^2\\
\text{s.t.}
&\quad
U^{(n)\mathsf T}U^{(n)}=I_{R_n}
\end{aligned}
}
$$

である。

partial HOOIではactive mode集合を $\mathcal M$ とし、$n\in\mathcal M$ のfactorだけを最適化する。

HOOIはrankを固定したままfactorを1 modeずつ更新するため、各更新は局所的には最適でも、factor全体を同時に見た非凸問題のglobal optimumを保証しない。

---

## 13. 誤差最小化とcore norm最大化

### 残差と再構成が直交する理由も成分から確認する

以下の直交性は、coreが固定factorへの射影であることを前提とする。
3階の場合、factorの展開を $\mathcal A$、転置による圧縮を $\mathcal A^*$ と書くと

$$
[\mathcal A(C)]_{i_0,i_1,i_2}
=\sum_{\alpha_0,\alpha_1,\alpha_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
C_{\alpha_0,\alpha_1,\alpha_2}.
$$

内積の有限和を入れ替えて

$$
\begin{aligned}
\langle R,\mathcal A(C)\rangle_F
&=\sum_{i_0,i_1,i_2}R_{i_0,i_1,i_2}
\sum_{\alpha_0,\alpha_1,\alpha_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
C_{\alpha_0,\alpha_1,\alpha_2}\\
&=\sum_{\alpha_0,\alpha_1,\alpha_2}
\left(\sum_{i_0,i_1,i_2}
U^{(0)}_{i_0,\alpha_0}U^{(1)}_{i_1,\alpha_1}U^{(2)}_{i_2,\alpha_2}
R_{i_0,i_1,i_2}\right)C_{\alpha_0,\alpha_1,\alpha_2}\\
&=\langle\mathcal A^*(R),C\rangle_F.
\end{aligned}
$$

列直交性により各modeで $U^{(n)\mathsf T}U^{(n)}=I$ が消えるので

$$
\mathcal A^*\mathcal A(C)=C.
$$

$G:=\mathcal A^*(X)$、$\hat X:=\mathcal A(G)$ とすると

$$
\begin{aligned}
\mathcal A^*(X-\hat X)
&=\mathcal A^*X-\mathcal A^*\mathcal A(G)=G-G=0,\\
\langle X-\hat X,\hat X\rangle_F
&=\langle\mathcal A^*(X-\hat X),G\rangle_F
=\langle0,G\rangle_F=0.
\end{aligned}
$$

これが以下で使う交差項0の根拠である。
Partial Tuckerでも、非圧縮modeには恒等写像を使えば同じ計算になる。

ここからは**補足導出**。

現在の直交factorで再構成したTensorを

$$
\hat{\mathcal X}
$$

とし、残差を

$$
\mathcal R
=
\mathcal X-\hat{\mathcal X}
$$

とする。

直交射影なので、残差と射影部分は直交する。

$$
\boxed{
\langle\mathcal R,\hat{\mathcal X}\rangle_F=0
}
$$

元Tensorは

$$
\mathcal X
=
\mathcal R+\hat{\mathcal X}
$$

なので、

$$
\begin{aligned}
\|\mathcal X\|_F^2
&=
\|\mathcal R+\hat{\mathcal X}\|_F^2\\
&=
\|\mathcal R\|_F^2
+2\langle\mathcal R,\hat{\mathcal X}\rangle_F
+\|\hat{\mathcal X}\|_F^2\\
&=
\|\mathcal R\|_F^2
+\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\hat{\mathcal X}\|_F^2
}.
$$

次に、列直交factorをmode productしてもFrobenius normが変わらないことを確認する。

$$
\mathcal Y
=
\mathcal G\times_nU^{(n)}
$$

なら、

$$
Y_{(n)}
=
U^{(n)}G_{(n)}.
$$

したがって、

$$
\begin{aligned}
\|\mathcal Y\|_F^2
&=
\|Y_{(n)}\|_F^2\\
&=
\operatorname{tr}
\left[
(U^{(n)}G_{(n)})^{\mathsf T}
(U^{(n)}G_{(n)})
\right]\\
&=
\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}
U^{(n)\mathsf T}U^{(n)}
G_{(n)}
\right]\\
&=
\operatorname{tr}
\left[
G_{(n)}^{\mathsf T}G_{(n)}
\right]\\
&=
\|\mathcal G\|_F^2.
\end{aligned}
$$

これを各active modeへ順に適用すると、

$$
\boxed{
\|\hat{\mathcal X}\|_F
=
\|\mathcal G\|_F
}.
$$

よって、

$$
\boxed{
\|\mathcal X-\hat{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
-
\|\mathcal G\|_F^2
}.
$$

$\|\mathcal X\|_F^2$ は固定なので、

$$
\boxed{
\min\|\mathcal X-\hat{\mathcal X}\|_F^2
\Longleftrightarrow
\max\|\mathcal G\|_F^2
}.
$$

これが「HOOIは射影後coreへできるだけ多くのFrobenius energyを残すようにfactorを更新する」と読める理由である。

---

## 14. HOOIの1 factor更新

更新対象をmode

$$
n\in\mathcal M
$$

とする。

更新対象以外を現在factorで射影する。

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
Z
=Z_{(n)}^{(n)}
$$

と書く。

候補factorを

$$
U
\in
\mathbb R^{I_n\times R_n},
\qquad
U^{\mathsf T}U=I_{R_n}
$$

とする。

このfactorまで適用したcoreのmode-$n$ unfoldingは

$$
\boxed{
G_{(n)}
=U^{\mathsf T}Z
}.
$$

前節より、他factorを固定した局所問題は

$$
\boxed{
\max_{U^{\mathsf T}U=I_{R_n}}
\|U^{\mathsf T}Z\|_F^2
}
$$

である。

Frobenius normをtraceへ変形する。

$$
\begin{aligned}
\|U^{\mathsf T}Z\|_F^2
&=
\operatorname{tr}
\left[
(U^{\mathsf T}Z)
(U^{\mathsf T}Z)^{\mathsf T}
\right]\\
&=
\operatorname{tr}
\left[
U^{\mathsf T}ZZ^{\mathsf T}U
\right].
\end{aligned}
$$

したがって、

$$
\boxed{
\max_{U^{\mathsf T}U=I_{R_n}}
\operatorname{tr}
\left(
U^{\mathsf T}ZZ^{\mathsf T}U
\right)
}
$$

を解けばよい。

SVDを

$$
Z
=Q\Sigma V^{\mathsf T}
$$

とすると、

$$
\begin{aligned}
ZZ^{\mathsf T}
&=
Q\Sigma V^{\mathsf T}
V\Sigma^{\mathsf T}Q^{\mathsf T}\\
&=
Q\Sigma\Sigma^{\mathsf T}Q^{\mathsf T}.
\end{aligned}
$$

したがって $ZZ^{\mathsf T}$ の固有ベクトルは $Q$ の列で、固有値は

$$
\sigma_1^2\ge\sigma_2^2\ge\cdots\ge0
$$

である。

Rayleigh–Ritz / Ky Fanの最大化原理から、$R_n$ 次元直交部分空間でtraceを最大にするには、最大の $R_n$ 個の固有値に対応する固有ベクトルを選べばよい。

その最大化原理を、ここでも途中式まで展開する。$d=I_n$、$R=R_n$ とし、完全な固有基底を $\bar Q=(q_1,\ldots,q_d)$、固有値を $\lambda_i=\sigma_i^2$ とする。reduced SVDで返らない方向は零固有値として補う。

$$
H=\bar Q^{\mathsf T}U,\qquad
H^{\mathsf T}H=I_R,\qquad
w_i=\sum_{a=1}^{R}H_{i,a}^2=q_i^{\mathsf T}UU^{\mathsf T}q_i.
$$

射影の重みなので

$$
0\le w_i\le1,\qquad
\sum_{i=1}^{d}w_i=\operatorname{tr}(H^{\mathsf T}H)=R
$$

であり、

$$
\operatorname{tr}(U^{\mathsf T}ZZ^{\mathsf T}U)
=\operatorname{tr}(H^{\mathsf T}\operatorname{diag}(\lambda_i)H)
=\sum_{i=1}^{d}\lambda_iw_i.
$$

$1\le R<d$ について、降順性を使うと、

$$
\begin{aligned}
\sum_{i=1}^{d}\lambda_iw_i
&\le\sum_{i=1}^{R}\lambda_iw_i
+\lambda_R\sum_{i=R+1}^{d}w_i\\
&=R\lambda_R+\sum_{i=1}^{R}(\lambda_i-\lambda_R)w_i\\
&\le R\lambda_R+\sum_{i=1}^{R}(\lambda_i-\lambda_R)\\
&=\sum_{i=1}^{R}\lambda_i.
\end{aligned}
$$

上位 $R$ 本を取ると $w_i=1$（$i\le R$）、$w_i=0$（$i>R$）になり、この上界を達成する。$R=d$ は全空間で同じ結論になる。したがって、「上位左特異ベクトル」は、traceを固有基底の成分和に直して解いた結果である。

よって、

$$
\boxed{
U^{(n)}
\leftarrow
\begin{bmatrix}
q_1&q_2&\cdots&q_{R_n}
\end{bmatrix}
}
$$

すなわち、

$$
\boxed{
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\operatorname{leftSVD}
\left(
Z_{(n)}^{(n)}
\right)
}.
$$

Pythonなら

PyTorchの確認コード：[[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード#PyTorch確認-002]]

である。

ここで重要なのは、**更新対象自身の古いfactorはprojectionへ入れず、他factorだけを固定して対象modeの最適部分空間を再計算する**ことである。

---

## 15. Tucker-2 HOOIを $t\rightarrow t+1$ で追う

初期値はHOSVD。

$$
U_{out}^{(0)},
\qquad
U_{in}^{(0)}.
$$

### 15.1 $U_{out}$ 更新

iteration $t$ の入力factorを使って、

$$
\boxed{
Z_{out}^{(t)}
=
W\times_1
(U_{in}^{(t)})^{\mathsf T}
}
$$

を作る。

shapeは

$$
(C_{out},C_{in},K_h,K_w)
\rightarrow
(C_{out},R_{in},K_h,K_w).
$$

mode 0 unfoldingは

$$
(Z_{out}^{(t)})_{(0)}
\in
\mathbb R^{C_{out}\times(R_{in}K_hK_w)}.
$$

その上位左特異ベクトルを取り、

$$
\boxed{
U_{out}^{(t+1)}
=
\operatorname{top\text{-}}R_{out}
\operatorname{leftSVD}
\left[
(Z_{out}^{(t)})_{(0)}
\right]
}
$$

と更新する。

### 15.2 $U_{in}$ 更新

同じsweep内で**更新済み**の

$$
U_{out}^{(t+1)}
$$

を使う。

$$
\boxed{
Z_{in}^{(t)}
=
W\times_0
(U_{out}^{(t+1)})^{\mathsf T}
}
$$

shapeは

$$
(C_{out},C_{in},K_h,K_w)
\rightarrow
(R_{out},C_{in},K_h,K_w).
$$

mode 1 unfoldingは

$$
(Z_{in}^{(t)})_{(1)}
\in
\mathbb R^{C_{in}\times(R_{out}K_hK_w)}.
$$

したがって、

$$
\boxed{
U_{in}^{(t+1)}
=
\operatorname{top\text{-}}R_{in}
\operatorname{leftSVD}
\left[
(Z_{in}^{(t)})_{(1)}
\right]
}
$$

と更新する。

この最新factorを同一sweepで使う更新が、現在srcのGauss-Seidel型 `hooi_sweep()` に対応する。

---

## 16. core・再構成・error

1 sweep後のfactorからcoreを作る。

$$
\boxed{
G^{(t+1)}
=
W
\times_0(U_{out}^{(t+1)})^{\mathsf T}
\times_1(U_{in}^{(t+1)})^{\mathsf T}
}
$$

再構成は

$$
\boxed{
\hat W^{(t+1)}
=
G^{(t+1)}
\times_0U_{out}^{(t+1)}
\times_1U_{in}^{(t+1)}
}
$$

である。

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

したがって1 sweepは

```text
U_out^(t), U_in^(t)
↓
U_out^(t+1) を更新
↓
新 U_out^(t+1) を使って U_in^(t+1) を更新
↓
G^(t+1)
↓
W_hat^(t+1)
↓
e_(t+1)
```

という一続きの処理になる。

---

## 17. 収束判定

単純な絶対差なら

$$
|e_t-e_{t-1}|<\mathrm{tol}
$$

で止められる。

相対改善だけを見る案なら、

$$
\frac{|e_t-e_{t-1}|}
{\max(|e_{t-1}|,\epsilon)}
<\mathrm{tol}
$$

と書ける。

現在srcでは絶対許容と相対許容を合わせて

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

右辺は

```text
絶対的にこれ以下なら停止
+
現在のerrorスケールに比例する許容
```

を合わせたものになる。

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

なら、

$$
\begin{aligned}
\varepsilon_{abs}
+
\varepsilon_{rel}|e_{t-1}|
&=
10^{-8}+10^{-5}\cdot0.44\\
&=10^{-8}+4.4\times10^{-6}\\
&=4.41\times10^{-6}.
\end{aligned}
$$

---

## 18. 実験値で確認する論点

同じrankのHOSVDとHOOIを比較するときは、まず重み近似誤差の差

$$
\Delta e=e_{\mathrm{HOSVD}}-e_{\mathrm{HOOI}}
$$

と相対改善

$$
\frac{\Delta e}{e_{\mathrm{HOSVD}}}
$$

を分けて確認する。そのうえで、weight errorの大小とvalidation accuracyの大小を別の指標として比較する。

$$
\boxed{
\text{lower weight error}
\not\Rightarrow
\text{higher task accuracy}
}
$$

具体的なrunの値と途中計算は [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]] を正本とする。

---

## 19. fine-tuning前後

fine-tuning前後では、weight errorとtask accuracyをそれぞれ比較する。fine-tuning後にweight errorが増えてもtask accuracyが回復する場合がある。

これはfine-tuningが

$$
\min\|W-\hat W\|_F
$$

を解いているのではなく、分類task lossを勾配法で最適化していることと整合する。

具体的なrunの値、差分の途中計算、単一seedの解釈上の注意は [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]] を正本とする。

---

## 20. このノートの結論

```text
unfold
→ modeを行方向へ出す
→ shape / 要素数を確認

mode product
→ 要素表示
→ Y_(n)=A X_(n)
→ foldして対象modeだけ置換

Tucker
→ factor^Tでrank空間へ射影
→ core
→ factorで元空間へ再構成

HOSVD
→ 各modeを元Tensorから独立にunfold
→ SVD
→ top-R left singular vectors
→ Pythonでは [:, :R]
→ 元Tensorをfactor^Tで射影してcore

Tucker-2
→ C_in → R_in → R_out → C_out
→ 1x1 → kxk → 1x1
→ 3式を代入するとeffective weightへ戻る

HOOI
→ HOSVD初期化
→ 他factorでproject
→ target modeをunfold
→ core norm最大化の局所問題
→ trace最大化
→ top-R left singular vectors
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
