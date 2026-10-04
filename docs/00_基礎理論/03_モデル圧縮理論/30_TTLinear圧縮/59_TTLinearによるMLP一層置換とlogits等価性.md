---
title: TTLinearによるMLP一層置換とlogits等価性
tags:
  - TT-matrix
  - TTLinear
  - MLP
  - logits
  - equivalence
---

# TTLinearによるMLP一層置換とlogits等価性

## 1. このノートで示すこと

学習済みMLPの一つのdense Linearを、打ち切りなしTT-SVDで初期化したTTLinearへ置き換える。TT-matrixが元の重みを厳密に表し、biasと後続計算を変えなければ、置換した層の出力だけでなく最終logitsまで一致する。

ここで示すのは関数等価性であり、圧縮成功ではない。打ち切りなしTTでは内部rankが大きくなり、dense重みよりパラメータ数が増えることもある。

## 2. MLPとshape

batch-first入力を

$$
X\in\mathbb{R}^{B\times n}
$$

とする。説明を簡潔にするため、まず2層MLPを

$$
\begin{aligned}
Z_1 &= XW_1^{\mathsf T}+b_1
&&\in\mathbb{R}^{B\times h},\\
H &= \operatorname{ReLU}(Z_1)
&&\in\mathbb{R}^{B\times h},\\
L &= HW_2^{\mathsf T}+b_2
&&\in\mathbb{R}^{B\times c}
\end{aligned}
$$

と書く。PyTorchの <code>nn.Linear(n, h)</code> は

$$
W_1\in\mathbb{R}^{h\times n},
\qquad
b_1\in\mathbb{R}^{h}
$$

を保持し、入力の右から転置重みを掛ける。

実際のネットワークが

$$
\operatorname{fc1}
\longrightarrow
\operatorname{ReLU}
\longrightarrow
\operatorname{fc2}
\longrightarrow
\operatorname{ReLU}
\longrightarrow
\operatorname{fc3}
$$

のように長くても議論は同じである。置換層より後ろの決定論的計算全体を $F$ と書けば、

$$
L=F(Z_1)
$$

とまとめられる。

## 3. dense重みからTT-matrixへ

出力次元と入力次元を

$$
h=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

と分解する。dense重み

$$
W_1\in\mathbb{R}^{h\times n}
$$

を

$$
\mathcal{W}_1
\in
\mathbb{R}^{m_1\times n_1\times\cdots\times m_d\times n_d}
$$

へ並べ替える。第 $k$ TT-matrix coreを

$$
G^{(k)}
\in
\mathbb{R}^{r_{k-1}\times m_k\times n_k\times r_k},
\qquad
r_0=r_d=1
$$

とすると、

$$
\begin{aligned}
&W_{1,\mathrm{TT}}
\bigl[
(i_1,\ldots,i_d),
(j_1,\ldots,j_d)
\bigr]\\
&=
\sum_{\alpha_1=0}^{r_1-1}
\cdots
\sum_{\alpha_{d-1}=0}^{r_{d-1}-1}
G^{(1)}_{0,i_1,j_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,j_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,j_d,0}.
\end{aligned}
$$

境界rankはサイズ1なので、0-based indexでは左右端のbond indexは0である。

打ち切りなしTT-SVDなら、丸め誤差を除いて

$$
\boxed{
W_{1,\mathrm{TT}}=W_1
}
$$

となる。

## 4. 第1 Linear出力の一致

TTLinearでも元と同じbias $b_1$ を使う。

$$
Z_{1,\mathrm{TT}}
=
XW_{1,\mathrm{TT}}^{\mathsf T}+b_1.
$$

$W_{1,\mathrm{TT}}=W_1$ を代入すると、

$$
\begin{aligned}
Z_{1,\mathrm{TT}}
&=
XW_{1,\mathrm{TT}}^{\mathsf T}+b_1\\
&=
XW_1^{\mathsf T}+b_1\\
&=
Z_1.
\end{aligned}
$$

成分ごとにも、batch index $b$ と出力index $i$ に対して

$$
\begin{aligned}
(Z_{1,\mathrm{TT}})_{b,i}
&=
\sum_{j=0}^{n-1}
X_{b,j}(W_{1,\mathrm{TT}})_{i,j}
+(b_1)_i\\
&=
\sum_{j=0}^{n-1}
X_{b,j}(W_1)_{i,j}
+(b_1)_i\\
&=
(Z_1)_{b,i}
\end{aligned}
$$

である。

## 5. activationとlogitsの一致

ReLUは同じ入力に同じ出力を返す決定論的関数なので、

$$
\begin{aligned}
H_{\mathrm{TT}}
&=
\operatorname{ReLU}(Z_{1,\mathrm{TT}})\\
&=
\operatorname{ReLU}(Z_1)\\
&=
H.
\end{aligned}
$$

ここではLipschitz上界を使っていない。入力テンソル自体が一致するため、関数適用後も一致するというだけである。

後続層の重み、bias、演算順序が同じなら、

$$
\begin{aligned}
L_{\mathrm{TT}}
&=
H_{\mathrm{TT}}W_2^{\mathsf T}+b_2\\
&=
HW_2^{\mathsf T}+b_2\\
&=
L.
\end{aligned}
$$

一般の後続写像 $F$ に対しても、

$$
Z_{1,\mathrm{TT}}=Z_1
\quad\Longrightarrow\quad
F(Z_{1,\mathrm{TT}})=F(Z_1)
$$

である。ただしDropoutを含む場合は乱数状態とtrain/eval mode、BatchNormを含む場合はmodeとrunning statisticsを一致させる必要がある。

## 6. 2-coreでの全添字

小さい例として

$$
B=5,
\qquad
n=12=3\cdot4,
\qquad
h=20=4\cdot5
$$

とし、

$$
(n_1,n_2)=(3,4),
\qquad
(m_1,m_2)=(4,5)
$$

を選ぶ。row-majorの複合添字は

$$
j=4j_1+j_2,
\qquad
i=5i_1+i_2.
$$

したがって、

$$
\mathcal{W}[i_1,j_1,i_2,j_2]
=
W[5i_1+i_2,\ 4j_1+j_2].
$$

core shapeは

$$
G^{(1)}\in\mathbb{R}^{1\times4\times3\times r},
\qquad
G^{(2)}\in\mathbb{R}^{r\times5\times4\times1}.
$$

入力を

$$
\mathcal{X}[b,j_1,j_2]
=
X[b,4j_1+j_2]
$$

とすると、biasを加える前の出力全要素は

$$
\begin{aligned}
\mathcal{Y}[b,i_1,i_2]
&=
\sum_{j_1=0}^{2}
\sum_{j_2=0}^{3}
\sum_{\alpha=0}^{r-1}
\mathcal{X}[b,j_1,j_2]\\
&\qquad\qquad\cdot
G^{(1)}[0,i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0].
\end{aligned}
$$

最後に

$$
Y[b,5i_1+i_2]
=
\mathcal{Y}[b,i_1,i_2]+b_{5i_1+i_2}
$$

と戻す。この式は、dense重みを明示的に再構成せずに $XW_{\mathrm{TT}}^{\mathsf T}+b$ を計算している。

## 7. 数値計算では近接判定を使う

SVDと縮約は浮動小数点演算なので、数学上の等号をbit単位の一致として検査してはいけない。要素ごとに

$$
|A-B|
\le
\mathrm{atol}
+
\mathrm{rtol}|B|
$$

を満たすかを確認する。

検証は次の順に分ける。

1. TT coreから復元した重みと元のdense重み
2. 置換対象Linear単体の出力
3. activation直後
4. 最終logits
5. 補助確認としてclass prediction

この段階分けにより、tensorization・縮約・層置換・後続計算のどこで差が生じたかを切り分けられる。

## 8. prediction一致の位置づけ

分類予測は

$$
\widehat c_b
=
\operatorname*{arg\,max}_{k}
L_{b,k}
$$

で求める。softmaxはlogitの大小関係を保存するため、予測classだけを見るならsoftmaxを先に計算する必要はない。

ただし、prediction一致だけではlogitsの等価性を証明できない。異なるlogitsでも最大要素の位置が同じ場合があるため、まずlogitsを近接判定し、その後にpredictionを補助確認する。

## 9. 等価性と圧縮を分ける

打ち切りなしTT-SVDの目的は、実装上の添字順序とforwardが正しいことを確認することである。

$$
\boxed{
\text{forward等価性}
\neq
\text{パラメータ圧縮}
}
$$

圧縮を評価する段階ではrankを制限し、

- 重み再構成誤差
- 層出力誤差
- logits誤差
- predictionとaccuracy
- パラメータ数、計算量、実測時間

を別々に測る。

打ち切りによるLinear出力誤差は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]を参照する。学習済みモデルでのPyTorch確認は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]を参照する。
