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

Tucker/HOOI編で使う式を、`CNN CIFAR-10→Tuckerを考える.md` の議論と現在の実験結果に合わせて、**shapeと途中式を飛ばさず**一本にまとめる。

このノートをcanonical derivationとし、21-26ではテーマ別に読む。

---

## 1. mode-n unfolding

$N$階Tensorを

$$
\mathcal X
\in
\mathbb R^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

とする。

mode $n$を行方向へ置き、残りを列方向へまとめると、

$$
X_{(n)}
\in
\mathbb R^{I_n\times\prod_{m\ne n}I_m}.
$$

Conv weight

$$
W\in\mathbb R^{64\times32\times3\times3}
$$

では、mode 0なら

$$
\begin{aligned}
W_{(0)}
&\in
\mathbb R^{64\times(32\cdot3\cdot3)}\\
&=
\mathbb R^{64\times288}.
\end{aligned}
$$

mode 1なら

$$
\begin{aligned}
W_{(1)}
&\in
\mathbb R^{32\times(64\cdot3\cdot3)}\\
&=
\mathbb R^{32\times576}.
\end{aligned}
$$

元と行列化後で要素数は、

$$
64\cdot32\cdot3\cdot3
=18432
$$

$$
64\cdot288
=18432
$$

で同じ。unfold自体は圧縮ではない。

---

## 2. mode-n productをunfold→行列積→foldで追う

行列

$$
A\in\mathbb R^{J\times I_n}
$$

をmode$n$へ掛ける。

$$
\mathcal Y=\mathcal X\times_nA
$$

unfoldした形では、

$$
\boxed{Y_{(n)}=AX_{(n)}}.
$$

shapeは

$$
(J\times I_n)
(I_n\times\prod_{m\ne n}I_m)
=
J\times\prod_{m\ne n}I_m.
$$

最後にfoldすると、

$$
I_n\rightarrow J
$$

だけが変わる。

### Conv mode 1の具体例

$$
U_{\mathrm{in}}
\in
\mathbb R^{32\times R_{\mathrm{in}}}
$$

だから、圧縮方向は

$$
U_{\mathrm{in}}^{\mathsf T}
\in
\mathbb R^{R_{\mathrm{in}}\times32}.
$$

mode 1 unfoldingは

$$
W_{(1)}
\in
\mathbb R^{32\times(64\cdot3\cdot3)}.
$$

左から掛けると、

$$
\begin{aligned}
U_{\mathrm{in}}^{\mathsf T}W_{(1)}
&:
(R_{\mathrm{in}}\times32)
(32\times576)\\
&\rightarrow
R_{\mathrm{in}}\times576.
\end{aligned}
$$

foldしてmode 1を元位置へ戻すと、

$$
\boxed{
W\times_1U_{\mathrm{in}}^{\mathsf T}
\in
\mathbb R^{64\times R_{\mathrm{in}}\times3\times3}
}.
$$

mode 0も同じで、

$$
U_{\mathrm{out}}^{\mathsf T}
\in
\mathbb R^{R_{\mathrm{out}}\times64}
$$

$$
W_{(0)}
\in
\mathbb R^{64\times288}
$$

なので、

$$
(R_{\mathrm{out}}\times64)(64\times288)
\rightarrow
R_{\mathrm{out}}\times288
$$

となり、foldすれば

$$
W\times_0U_{\mathrm{out}}^{\mathsf T}
\in
\mathbb R^{R_{\mathrm{out}}\times32\times3\times3}.
$$

---

## 3. factorを転置する理由

factorは

$$
U^{(n)}\in\mathbb R^{I_n\times R_n}
$$

である。

元空間のvectorを

$$
x\in\mathbb R^{I_n}
$$

とすると、rank空間へ落とすには

$$
\begin{aligned}
z
&=U^{(n)\mathsf T}x,\\
(R_n\times I_n)(I_n\times1)
&\rightarrow R_n\times1.
\end{aligned}
$$

逆にrank空間

$$
z\in\mathbb R^{R_n}
$$

から戻すときは

$$
\begin{aligned}
x_{\mathrm{recon}}
&=U^{(n)}z,\\
(I_n\times R_n)(R_n\times1)
&\rightarrow I_n\times1.
\end{aligned}
$$

したがって、

```text
元空間 → rank空間 : U^T
rank空間 → 元空間 : U
```

である。

---

## 4. Tucker分解の一般式からcoreを作る

Tucker近似を

$$
\hat{\mathcal X}
=
\mathcal G
\times_0U^{(0)}
\times_1U^{(1)}
\cdots
\times_{N-1}U^{(N-1)}
$$

とする。

factorの列が直交し、

$$
U^{(n)\mathsf T}U^{(n)}=I
$$

なら、元Tensorを各factorの列空間へ射影してcoreを作る。

$$
\mathcal G
=
\mathcal X
\times_0U^{(0)\mathsf T}
\times_1U^{(1)\mathsf T}
\cdots.
$$

Tucker-2なら、最初にmode 0を縮約する場合、

$$
(64,32,3,3)
\xrightarrow{\times_0U_{\mathrm{out}}^{\mathsf T}}
(R_{\mathrm{out}},32,3,3)
$$

さらにmode 1を縮約する。

$$
(R_{\mathrm{out}},32,3,3)
\xrightarrow{\times_1U_{\mathrm{in}}^{\mathsf T}}
(R_{\mathrm{out}},R_{\mathrm{in}},3,3).
$$

よって、

$$
\boxed{
G
=
W\times_0U_{\mathrm{out}}^{\mathsf T}
\times_1U_{\mathrm{in}}^{\mathsf T}
}
$$

$$
\boxed{
G\in
\mathbb R^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times3\times3}
}.
$$

再構成は逆向き。

$$
(R_{\mathrm{out}},R_{\mathrm{in}},3,3)
\xrightarrow{\times_0U_{\mathrm{out}}}
(64,R_{\mathrm{in}},3,3)
$$

$$
(64,R_{\mathrm{in}},3,3)
\xrightarrow{\times_1U_{\mathrm{in}}}
(64,32,3,3).
$$

---

## 5. HOSVDをConv weightで1 stepずつ追う

HOSVDでは各factorを**元の同じWから独立に**求める。

### mode 0

$$
W_{(0)}\in\mathbb R^{64\times288}
$$

へSVDする。

$$
W_{(0)}
=U_0\Sigma_0V_0^{\mathsf T}.
$$

上位 $R_{\mathrm{out}}$ 本を取り、

$$
U_{\mathrm{out}}
=U_0[:,1:R_{\mathrm{out}}]
\in
\mathbb R^{64\times R_{\mathrm{out}}}.
$$

### mode 1

mode 0でprojectしたTensorではなく、**再び元W**をmode 1でunfoldする。

$$
W_{(1)}\in\mathbb R^{32\times576}
$$

$$
W_{(1)}
=U_1\Sigma_1V_1^{\mathsf T}.
$$

上位 $R_{\mathrm{in}}$ 本を取り、

$$
U_{\mathrm{in}}
=U_1[:,1:R_{\mathrm{in}}]
\in
\mathbb R^{32\times R_{\mathrm{in}}}.
$$

その後に両factorを使ってcoreを作る。

```text
元W → mode0 unfold → SVD → U_out
元W → mode1 unfold → SVD → U_in

元W
→ ×0 U_out^T
→ ×1 U_in^T
→ core
```

この「factor計算中は相互依存しない」点がHOOIとの違い。

---

## 6. Tucker-2 weight式から3層Convを導く

Tucker-2のweight近似は

$$
\boxed{
W_{o,i,a,b}
\approx
\sum_{\alpha=1}^{R_{\mathrm{out}}}
\sum_{\beta=1}^{R_{\mathrm{in}}}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}
}.
$$

channel変換は

$$
i\rightarrow\beta\rightarrow\alpha\rightarrow o
$$

である。

### 第1段: input projection

$$
\boxed{
Z_{\beta,h,w}
=
\sum_{i=1}^{C_{\mathrm{in}}}
U^{\mathrm{in}}_{i,\beta}
X_{i,h,w}
}
$$

これは各位置でchannelだけを混ぜるので1x1 Conv。

### 第2段: core convolution

$$
\boxed{
T_{\alpha,h,w}
=
\sum_{\beta=1}^{R_{\mathrm{in}}}
\sum_a\sum_b
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
}
$$

### 第3段: output projection

$$
\boxed{
Y_{o,h,w}
=
\sum_{\alpha=1}^{R_{\mathrm{out}}}
U^{\mathrm{out}}_{o,\alpha}
T_{\alpha,h,w}
+b_o
}
$$

### 3式を代入して1式へ戻す

第3段へ第2段を代入する。

$$
\begin{aligned}
Y_{o,h,w}
&=
\sum_{\alpha}
U^{\mathrm{out}}_{o,\alpha}
\left[
\sum_{\beta,a,b}
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
\right]
+b_o.
\end{aligned}
$$

さらに $Z$ を代入する。

$$
\begin{aligned}
Y_{o,h,w}
&=
\sum_{\alpha,\beta,a,b}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
\left[
\sum_i
U^{\mathrm{in}}_{i,\beta}
X_{i,h+a,w+b}
\right]
+b_o\\
&=
\sum_{i,a,b}
\left[
\sum_{\alpha,\beta}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}
\right]
X_{i,h+a,w+b}
+b_o.
\end{aligned}
$$

角括弧内がTucker近似したeffective weightなので、

$$
\hat W_{o,i,a,b}
=
\sum_{\alpha,\beta}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}.
$$

したがって3層Convが元の1層ConvのTucker-2近似になる。

---

## 7. Tucker-2のparameter数を具体値まで計算する

元Conv weightは

$$
\begin{aligned}
N_{\mathrm{orig}}
&=C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w\\
&=64\times32\times3\times3\\
&=18432.
\end{aligned}
$$

Tucker-2は、

$$
N_{\mathrm{Tucker2}}
=C_{\mathrm{in}}R_{\mathrm{in}}
+R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w
+C_{\mathrm{out}}R_{\mathrm{out}}.
$$

今回のshapeを代入する。

$$
\boxed{
N_{\mathrm{Tucker2}}
=32R_{\mathrm{in}}
+9R_{\mathrm{out}}R_{\mathrm{in}}
+64R_{\mathrm{out}}
}.
$$

### rankを下げない `(64,32)`

$$
\begin{aligned}
N
&=32\times32
+9\times64\times32
+64\times64\\
&=1024+18432+4096\\
&=23552.
\end{aligned}
$$

増加量は

$$
23552-18432=5120.
$$

したがってTucker形式にしただけでは圧縮にならない。

### balanced `(32,16)`

$$
\begin{aligned}
N
&=32\times16
+9\times32\times16
+64\times32\\
&=512+4608+2048\\
&=7168.
\end{aligned}
$$

Conv2 weight削減率は

$$
\begin{aligned}
R
&=1-\frac{7168}{18432}\\
&=1-0.388888\ldots\\
&=0.611111\ldots\\
&\approx61.11\%.
\end{aligned}
$$

---

## 8. HOOIの目的関数

標準的な直交Tucker近似では、

$$
\min
\left\|
\mathcal X
-
\mathcal G
\times_0U^{(0)}
\cdots
\times_{N-1}U^{(N-1)}
\right\|_F^2
$$

subject to

$$
U^{(n)\mathsf T}U^{(n)}=I.
$$

HOOIはrankを変えずに、factorを1 modeずつ更新する。

### 数式上の補足導出

以下は上の直交射影式からの代数展開である。

固定したfactorで作る再構成を $\hat{\mathcal X}$ とすると、

$$
\begin{aligned}
\|\mathcal X-\hat{\mathcal X}\|_F^2
&=\|\mathcal X\|_F^2
-2\langle\mathcal X,\hat{\mathcal X}\rangle_F
+\|\hat{\mathcal X}\|_F^2.
\end{aligned}
$$

直交factorによる射影では、coreの展開が射影 $\hat{\mathcal X}$ なので、

$$
\langle\mathcal X,\hat{\mathcal X}\rangle_F
=\|\hat{\mathcal X}\|_F^2
=\|\mathcal G\|_F^2.
$$

したがって、

$$
\begin{aligned}
\|\mathcal X-\hat{\mathcal X}\|_F^2
&=\|\mathcal X\|_F^2
-\|\mathcal G\|_F^2.
\end{aligned}
$$

$\|\mathcal X\|_F^2$ は固定なので、再構成誤差を下げることは、現在rank空間へ投影されたcore normを大きくすることと対応する。

---

## 9. 一般HOOIの1 factor更新

更新対象をmode$n$とする。

更新対象以外を現在factorで射影する。

$$
\boxed{
\mathcal Z^{(n)}
=
\mathcal X
\underset{m\ne n}{\times_m}
U^{(m)\mathsf T}
}.
$$

全mode圧縮ならshapeは

$$
R_0\times\cdots\times R_{n-1}
\times I_n
\times R_{n+1}\times\cdots.
$$

mode$n$でunfoldすると、

$$
Z_{(n)}^{(n)}
\in
\mathbb R^{I_n\times\prod_{m\ne n}R_m}.
$$

これへSVDし、上位$R_n$左特異ベクトルを取る。

$$
\boxed{
U^{(n)}
\leftarrow
\operatorname{top\text{-}}R_n
\operatorname{leftSVD}
(Z_{(n)}^{(n)})
}.
$$

3階Tensorなら1 sweepは、

$$
U_0^{(t+1)}
\leftarrow
\operatorname{SVD}
\left[
(\mathcal X
\times_1(U_1^{(t)})^{\mathsf T}
\times_2(U_2^{(t)})^{\mathsf T})_{(0)}
\right]
$$

次に更新済み$U_0^{(t+1)}$を使って、

$$
U_1^{(t+1)}
\leftarrow
\operatorname{SVD}
\left[
(\mathcal X
\times_0(U_0^{(t+1)})^{\mathsf T}
\times_2(U_2^{(t)})^{\mathsf T})_{(1)}
\right]
$$

さらに更新済み$U_0,U_1$を使って$U_2$を更新する。

これが今回srcのGauss-Seidel型sweepに対応する。

---

## 10. Tucker-2 HOOIを `t → t+1` で完全に追う

初期値はHOSVD。

$$
U_{\mathrm{out}}^{(0)},
\quad
U_{\mathrm{in}}^{(0)}.
$$

### 10.1 `U_out`更新

現在の$U_{\mathrm{in}}^{(t)}$を固定する。

$$
\boxed{
Z_{\mathrm{out}}^{(t)}
=
W\times_1
(U_{\mathrm{in}}^{(t)})^{\mathsf T}
}.
$$

shapeは

$$
(64,32,3,3)
\rightarrow
(64,R_{\mathrm{in}},3,3).
$$

mode 0 unfoldingは

$$
(Z_{\mathrm{out}}^{(t)})_{(0)}
\in
\mathbb R^{64\times(R_{\mathrm{in}}\cdot3\cdot3)}.
$$

SVDして、

$$
\boxed{
U_{\mathrm{out}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{out}}
\operatorname{leftSVD}
[(Z_{\mathrm{out}}^{(t)})_{(0)}]
}.
$$

### 10.2 `U_in`更新

**同じsweepで今更新した** $U_{\mathrm{out}}^{(t+1)}$ を使う。

$$
\boxed{
Z_{\mathrm{in}}^{(t)}
=
W\times_0
(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
}.
$$

shapeは

$$
(64,32,3,3)
\rightarrow
(R_{\mathrm{out}},32,3,3).
$$

mode 1 unfoldingは

$$
(Z_{\mathrm{in}}^{(t)})_{(1)}
\in
\mathbb R^{32\times(R_{\mathrm{out}}\cdot3\cdot3)}.
$$

SVDして、

$$
\boxed{
U_{\mathrm{in}}^{(t+1)}
=
\operatorname{top\text{-}}R_{\mathrm{in}}
\operatorname{leftSVD}
[(Z_{\mathrm{in}}^{(t)})_{(1)}]
}.
$$

### 10.3 core

$$
\boxed{
G^{(t+1)}
=
W
\times_0(U_{\mathrm{out}}^{(t+1)})^{\mathsf T}
\times_1(U_{\mathrm{in}}^{(t+1)})^{\mathsf T}
}.
$$

### 10.4 reconstruction

$$
\boxed{
\hat W^{(t+1)}
=
G^{(t+1)}
\times_0U_{\mathrm{out}}^{(t+1)}
\times_1U_{\mathrm{in}}^{(t+1)}
}.
$$

### 10.5 error

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

その後、次のsweepへ進む。

---

## 11. 収束判定を3段階で整理する

学習過程で最初に使った単純形は、

$$
|e_t-e_{t-1}|<\mathrm{tol}.
$$

誤差のスケールを考える相対改善率なら、

$$
\boxed{
\frac{|e_{t-1}-e_t|}
{\max(|e_{t-1}|,\epsilon)}
<\mathrm{tol}
}.
$$

最終srcでは絶対許容と相対許容を合わせて、

$$
\boxed{
|e_t-e_{t-1}|
\le
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}|e_{t-1}|
}.
$$

例えば

```text
abs_tol = 1e-8
rel_tol = 1e-5
```

なら、前回誤差が0.44程度のとき右辺は

$$
10^{-8}+10^{-5}\times0.44
\approx4.41\times10^{-6}.
$$

絶対差だけでなく現在の誤差スケールも考慮できる。

---

## 12. HOOIの実測誤差を途中計算する

balanced rank `(32,16)` では、

$$
e_{\mathrm{HOSVD}}=0.449042
$$

$$
e_{\mathrm{HOOI}}=0.442504.
$$

絶対改善量は

$$
\begin{aligned}
\Delta e
&=0.449042-0.442504\\
&=0.006538.
\end{aligned}
$$

HOSVD errorを分母にした相対的な改善は

$$
\begin{aligned}
\frac{0.006538}{0.449042}
&\approx0.01456\\
&\approx1.46\%.
\end{aligned}
$$

一方、validation accuracyは

$$
0.6280-0.6202=0.0078
$$

だけHOSVDの方が高かった。

したがって、

$$
\text{lower weight error}
\not\Rightarrow
\text{higher task accuracy}
$$

を実測で確認した。

---

## 13. accuracy dropの符号を具体値で確認する

定義は

$$
\mathrm{accuracy\_drop}
=
\mathrm{acc}_{\mathrm{baseline}}
-
\mathrm{acc}_{\mathrm{compressed}}.
$$

`(64,32)` では、

$$
\begin{aligned}
\mathrm{drop}
&=0.7344-0.7350\\
&=-0.0006.
\end{aligned}
$$

負なのでcompressed側が0.0006だけ高く観測された。

validationが5000枚ならaccuracyの1sample刻みは

$$
\frac1{5000}=0.0002.
$$

したがって、

$$
\frac{0.0006}{0.0002}=3
$$

で、3sample分の差に相当する。これだけでTucker化による本質的な精度向上とは解釈しない。

---

## 14. balanced model parameter reduction

baseline model parameterは

$$
128842
$$

balanced Tucker-2は

$$
117578.
$$

削減数は

$$
128842-117578=11264.
$$

削減率は

$$
\begin{aligned}
R
&=1-\frac{117578}{128842}\\
&\approx0.087425\\
&\approx8.74\%.
\end{aligned}
$$

Conv2単体の61.11%削減よりモデル全体削減率が小さいのは、圧縮していない層が残るため。

---

## 15. Fine-tuning前後を算術で比較する

05の同条件比較では、test accuracyは

$$
\begin{aligned}
\Delta\mathrm{test}
&=0.7555-0.7508\\
&=0.0047.
\end{aligned}
$$

つまり0.47 percentage point。

seed 0の1runだけなので、統計的優位性とはしない。

weight errorは、HOSVDで

$$
\begin{aligned}
0.483660-0.449042
&=0.034618
\end{aligned}
$$

増え、HOOIで

$$
\begin{aligned}
0.481255-0.442504
&=0.038751
\end{aligned}
$$

増えた。

それでもaccuracyは大幅に回復した。

HOOIのHOSVDに対するweight-error優位は、fine-tuning前

$$
0.449042-0.442504=0.006538
$$

から、fine-tuning後

$$
0.483660-0.481255=0.002405
$$

へ縮小した。

これはfine-tuningが元weightへのFrobenius近似ではなくtask lossを最適化していることと整合する。

---

## 16. TensorLy timingの解釈

04では

```text
self HOOI  = 41.65 ms
TensorLy   = 83.83 ms
```

だったが、両者とも6 iterationで最終errorも一致した。

したがって「TensorLyは反復回数が多いから約2倍遅い」とは説明できない。

今回のweight要素数は

$$
64\times32\times3\times3
=18432
$$

と小さい。汎用backend/API、Python関数呼び出し、初回kernel、同期等の固定overheadが相対的に目立つ可能性があるため、この単発比率をライブラリ一般の速度差へ外挿しない。

---

## 17. まとめ

```text
unfold
→ shapeを式で確認
→ mode productを行列積として確認

HOSVD
→ 元Wからmodeごとに独立SVD
→ factor
→ factor^Tでcore
→ factorでreconstruction

Tucker-2 forward
→ input 1x1
→ core kxk
→ output 1x1
→ 3式を代入してeffective weightへ戻す

HOOI
→ HOSVD初期値
→ 他factorでproject
→ target modeをunfold
→ SVDで更新
→ 同一sweepでは更新済factorを使う
→ core/reconstruction/error
→ convergence
```

これらをTT/MPSへ進む前の基礎として固定する。
