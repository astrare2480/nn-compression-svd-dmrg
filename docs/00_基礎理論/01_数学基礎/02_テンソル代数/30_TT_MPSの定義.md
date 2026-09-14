---
title: TT・MPSの定義
aliases:
  - Tensor Train
  - Matrix Product State
  - TTコア
  - MPSサイトテンソル
tags:
  - TT
  - MPS
  - TensorTrain
  - TensorNetwork
  - Schmidt分解
---

# TT・MPSの定義

## サマリー

Tensor Train（TT）は、高階テンソルを1個の巨大な配列として保持せず、各modeを担当する小さな3階テンソルを鎖状につないで表す形式である。物理では、開放境界条件の Matrix Product State（MPS）と同じ数学的構造を使う。

元テンソルを

$$
X\in\mathbb R^{n_1\times n_2\times\cdots\times n_d}
$$

とすると、第 $k$ TTコアは

$$
\boxed{
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
}
$$

で、両端は

$$
r_0=r_d=1
$$

とする。元テンソルの各成分は、内部bond indexを縮約して

$$
\boxed{
X_{i_1\cdots i_d}
=
\sum_{\alpha_1=1}^{r_1}\cdots
\sum_{\alpha_{d-1}=1}^{r_{d-1}}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,1}
}
$$

と復元する。

物理のMPS記法

$$
A^{[k]}_{\alpha_{k-1},s_k,\alpha_k}
$$

との対応は

$$
s_k\longleftrightarrow i_k,
\qquad
A^{[k]}\longleftrightarrow G^{(k)}
$$

である。

---

## 1. TTコアとは何か

TTコアは、元の大きなテンソルを復元するための「各mode担当の小テンソル」である。

3階テンソル

$$
X\in\mathbb R^{n_1\times n_2\times n_3}
$$

を考える。TTではこれを

$$
G^{(1)}\in\mathbb R^{1\times n_1\times r_1},
$$

$$
G^{(2)}\in\mathbb R^{r_1\times n_2\times r_2},
$$

$$
G^{(3)}\in\mathbb R^{r_2\times n_3\times1}
$$

という3個のコアへ分ける。

各コアは独立ではなく、内部添字

$$
\alpha_1,\alpha_2
$$

を共有する。

```text
G^(1) -- α1 -- G^(2) -- α2 -- G^(3)
```

この鎖構造が Tensor Train という名前の由来である。

3サイトの場合、元テンソルは

$$
\boxed{
X_{i_1i_2i_3}
=
\sum_{\alpha_1=1}^{r_1}
\sum_{\alpha_2=1}^{r_2}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
G^{(3)}_{\alpha_2,i_3,1}
}
$$

となる。

---

## 2. コアの3本の添字

physical indexとbond indexは同じ「番号」でも役割が違う。
元Tensorの一要素を求めるとき、physical index $i_1,\ldots,i_d$ は求めたい位置として固定する。
bond indexはコア間の全経路を足し合わせるための番号なので、最後の出力には残らない。
各physical位置に小さい行列を選び、その行列を鎖として掛けた結果が一つのスカラーになる。

例えば3サイトの一要素では

$$
g_{\alpha_1}:=G^{(1)}_{1,i_1,\alpha_1},
\qquad
H_{\alpha_1,\alpha_2}:=G^{(2)}_{\alpha_1,i_2,\alpha_2},
\qquad
q_{\alpha_2}:=G^{(3)}_{\alpha_2,i_3,1}
$$

と固定すれば、$X_{i_1,i_2,i_3}=g^{\mathsf T}Hq$ である。
$g,H,q$ は別々のサイトの元データを切り出したものではなく、テンソル全体を表現するために決めた係数である。
従って「Tensorを三つの部分配列に切って保管した」とは読まない。
また、任意のTTコアが最初から正規直交しているわけではない。TTという鎖の構造と、SVD・QRで選ぶ正準形の条件は別である。

一般の第 $k$ コア

$$
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

には3種類の添字がある。

| 軸 | 添字 | 意味 |
| --- | --- | --- |
| 左bond | $\alpha_{k-1}$ | 前のコアとの接続、左側に蓄積された情報 |
| physical / mode | $i_k$ | 元テンソルの第 $k$ mode |
| 右bond | $\alpha_k$ | 次のコアとの接続、右側へ渡す情報 |

物理のMPSでは $i_k$ を局所物理状態 $s_k$ と読む。左右のbond indexは仮想添字であり、局所自由度そのものではなく、左右の部分系を結ぶ相関チャネルを表す。

特定のphysical index $i_k$ を固定すると、中央コアは

$$
G^{(k)}_{:,i_k,:}
\in
\mathbb R^{r_{k-1}\times r_k}
$$

という行列として読める。したがってTTの1成分は

$$
\underbrace{G^{(1)}_{1,i_1,:}}_{1\times r_1}
\underbrace{G^{(2)}_{:,i_2,:}}_{r_1\times r_2}
\cdots
\underbrace{G^{(d)}_{:,i_d,1}}_{r_{d-1}\times1}
$$

という行列積で得られる1個のスカラーである。

---

## 3. なぜ端のbond dimensionは1か

第1コアの左側には前のコアがない。最後のコアの右側にも次のコアがない。そのため

$$
\boxed{r_0=r_d=1}
$$

と置く。

これは「新しい物理自由度を1個増やす」という意味ではなく、全コアを

$$
(r_{k-1},n_k,r_k)
$$

という同じ3軸形式で統一するためのサイズ1の境界軸である。

例えば第1 SVDで

$$
U^{(1)}\in\mathbb R^{n_1\times r_1}
$$

が得られたとする。この行列を第1TTコアとして

$$
G^{(1)}
=
\operatorname{reshape}
\left(U^{(1)},1,n_1,r_1\right)
$$

と書く。

成分ではただ

$$
\boxed{
G^{(1)}_{1,i_1,\alpha_1}
=
U^{(1)}_{i_1,\alpha_1}
}
$$

である。値も要素数も変化しない。

$$
n_1r_1
=
1\cdot n_1r_1.
$$

したがってこれは「4階テンソルにする」操作ではない。`reshape(U, 1, n_1, r_1)` の3個の数字は3本の軸のサイズを指定しており、結果は3階テンソルである。

### 小さい実行列で全要素を確認

$n_1=r_1=2$ とし、第1 SVDで得た列直交行列を

$$
U^{(1)}
=
\begin{pmatrix}
\dfrac{3}{5}&-\dfrac{4}{5}\\
\dfrac{4}{5}&\dfrac{3}{5}
\end{pmatrix}
$$

とする。PyTorchのrow-major順で全要素を並べると

$$
\left(
\dfrac{3}{5},
-\dfrac{4}{5},
\dfrac{4}{5},
\dfrac{3}{5}
\right)
$$

である。これを

$$
G^{(1)}
=
\operatorname{reshape}
\left(U^{(1)},1,2,2\right)
$$

とすると、shapeは $(2,2)$ から $(1,2,2)$ へ変わるが、唯一の左境界sliceは

$$
G^{(1)}_{1,:,:}
=
\begin{pmatrix}
\dfrac{3}{5}&-\dfrac{4}{5}\\
\dfrac{4}{5}&\dfrac{3}{5}
\end{pmatrix}
$$

のままである。各要素の対応を全て書けば

$$
\begin{aligned}
G^{(1)}_{1,1,1}
&=
U^{(1)}_{1,1}
=
\dfrac{3}{5},\\
G^{(1)}_{1,1,2}
&=
U^{(1)}_{1,2}
=
-\dfrac{4}{5},\\
G^{(1)}_{1,2,1}
&=
U^{(1)}_{2,1}
=
\dfrac{4}{5},\\
G^{(1)}_{1,2,2}
&=
U^{(1)}_{2,2}
=
\dfrac{3}{5}.
\end{aligned}
$$

したがって、格納される4個の値とその順序は変化せず、先頭にサイズ1の境界軸が追加されるだけである。

---

### 元資料の $2\times3$ 配列例：reshapeの説明とSVDの制約を分ける

元資料 `TT_MPS基礎理論.md` の1532〜1597行・2782〜2836行付近には、2行3列の値を境界軸付きコアへ移す説明がある。
その要素配置を省かず示す。ただし、ここでは任意の係数配列 $H$ と呼ぶ。

$$
H=\begin{pmatrix}
u_{1,1}&u_{1,2}&u_{1,3}\\
u_{2,1}&u_{2,2}&u_{2,3}
\end{pmatrix},\qquad
C=\operatorname{reshape}(H,1,2,3).
$$

$$
\begin{aligned}
C_{1,1,1}&=u_{1,1},&C_{1,1,2}&=u_{1,2},&C_{1,1,3}&=u_{1,3},\\
C_{1,2,1}&=u_{2,1},&C_{1,2,2}&=u_{2,2},&C_{1,2,3}&=u_{2,3}.
\end{aligned}
$$

行優先の値の列は、変更前後とも

$$
(u_{1,1},u_{1,2},u_{1,3},u_{2,1},u_{2,2},u_{2,3})
$$

であり、要素数も $2\cdot3=1\cdot2\cdot3=6$ のままである。
追加されるのは常に1を取る境界添字だけで、値の計算・並べ替え・近似は行わない。

一方、元資料はこの2行3列の配列を「SVDで得た $U^{(1)}$」と呼んでいるが、3本の正規直交列を2次元に置くことはできない。

$$
\operatorname{rank}(H^{\mathsf T}H)
\le\operatorname{rank}(H)\le2<3=\operatorname{rank}(I_3).
$$

従って $H^{\mathsf T}H=I_3$ は不可能であり、サイト1のrank-sized左特異ベクトル行列なら $r_1\le n_1=2$ が必要である。
上の例は**一般配列のreshapeとしては有効だが、列直交SVDコアの例ではない**。
任意のTT表現なら冗長なbondを持つ非直交コアとして $1\times2\times3$ を使うこと自体はできる。
直前の $2\times2$ 実行列例は、このSVDの制約も満たす例である。元資料の配列を別の例で置き換えず、要素配置と数学的な制約を区別して残す。

## 4. 3サイトのTTをSVDから作る見取り図

3サイトの係数テンソルを

$$
\Psi_{s_1s_2s_3}
\in
\mathbb C^{n_1\times n_2\times n_3}
$$

とする。

最初に

$$
s_1\mid(s_2,s_3)
$$

で行列化し、

$$
\Psi^{\langle1\rangle}
\in
\mathbb C^{n_1\times(n_2n_3)}
$$

へ変形する。reduced SVDを

$$
\Psi^{\langle1\rangle}
=
U^{(1)}\Sigma^{(1)}V^{(1)\dagger}
$$

とし、

$$
U^{(1)}\in\mathbb C^{n_1\times r_1},
$$

$$
\Sigma^{(1)}\in\mathbb R^{r_1\times r_1},
$$

$$
V^{(1)\dagger}\in\mathbb C^{r_1\times(n_2n_3)}
$$

とする。

左因子を

$$
G^{(1)}\in\mathbb C^{1\times n_1\times r_1}
$$

へreshapeし、残りを

$$
B^{(1)}
=
\Sigma^{(1)}V^{(1)\dagger}
\in
\mathbb C^{r_1\times(n_2n_3)}
$$

とする。

これを

$$
\mathcal B^{(1)}
\in
\mathbb C^{r_1\times n_2\times n_3}
$$

へ戻し、次は

$$
(\alpha_1,s_2)\mid s_3
$$

でSVDする。

この「SVDで左側を1サイトずつコアとして取り出し、$\Sigma V^\dagger$ を右へ渡す」操作を繰り返すことでTT/MPSが得られる。詳細は [[32_TT-SVD]] を参照する。

---


### 元資料の左特異ベクトルの全成分と物理状態

添付 `TT_MPS基礎理論.md` の20643–20822行では、
「1本の左特異ベクトルの成分数」と「残すベクトルの本数」を分けて説明している。
第1cutのreduced SVDなら $q=\min(n_1,n_2n_3)$ で

$$
U=(u_1\ u_2\ \cdots\ u_q),\qquad
u_\alpha=U(:,\alpha)=
\begin{pmatrix}U_{1,\alpha}\\U_{2,\alpha}\\\vdots\\U_{n_1,\alpha}\end{pmatrix}
\in\mathbb R^{n_1}.
$$

1列の長さは $n_1$ であり、右側の $n_2n_3$ ではない。
元資料の $n_1=2$ の例では

$$
u_1=\begin{pmatrix}U_{1,1}\\U_{2,1}\end{pmatrix},\qquad
u_2=\begin{pmatrix}U_{1,2}\\U_{2,2}\end{pmatrix},
$$

$$
|L_1\rangle=U_{1,1}|1\rangle+U_{2,1}|2\rangle,\qquad
|L_2\rangle=U_{1,2}|1\rangle+U_{2,2}|2\rangle.
$$

各状態は2個の物理基底の線形結合である。
非ゼロ特異値に対応して必要な状態の本数は $r_1=\operatorname{rank}(X^{\langle1\rangle})\le q$、
打ち切りで保持する本数はさらに $\widetilde r_1\le r_1$ となる。
列を減らしても、残した各列の成分数 $n_1$ は変わらない。
元資料で「固有ベクトル」と呼んだ左特異ベクトルは、
正確には $X^{\langle1\rangle}(X^{\langle1\rangle})^T$ の固有ベクトルでもある。
一般の長方形 $X^{\langle1\rangle}$ 自体の固有ベクトルではない。
## 5. 行列SVDからSchmidt分解へ

TTとMPSをつなぐ最初の重要点は、係数行列のSVDを状態ベクトルとして読み直すとSchmidt分解になることである。

3サイト状態を

$$
|\Psi\rangle
=
\sum_{s_1,s_2,s_3}
\psi_{s_1s_2s_3}
|s_1\rangle\otimes|s_2s_3\rangle
$$

とする。

右側の複合添字を

$$
\mu=(s_2,s_3)
$$

と置けば、

$$
|\Psi\rangle
=
\sum_{s_1,\mu}
\Psi^{\langle1\rangle}_{s_1,\mu}
|s_1\rangle\otimes|\mu\rangle.
$$

係数行列をSVDする。

$$
\Psi^{\langle1\rangle}_{s_1,\mu}
=
\sum_{\alpha_1=1}^{r_1}
U_{s_1,\alpha_1}
\lambda_{\alpha_1}
V^*_{\mu,\alpha_1}.
$$

これを状態展開へ代入する。

$$
\begin{aligned}
|\Psi\rangle
&=
\sum_{s_1,\mu}
\left(
\sum_{\alpha_1}
U_{s_1,\alpha_1}
\lambda_{\alpha_1}
V^*_{\mu,\alpha_1}
\right)
|s_1\rangle\otimes|\mu\rangle\\
&=
\sum_{\alpha_1}
\lambda_{\alpha_1}
\sum_{s_1,\mu}
U_{s_1,\alpha_1}
V^*_{\mu,\alpha_1}
|s_1\rangle\otimes|\mu\rangle\\
&=
\sum_{\alpha_1}
\lambda_{\alpha_1}
\left(
\sum_{s_1}U_{s_1,\alpha_1}|s_1\rangle
\right)
\otimes
\left(
\sum_{\mu}V^*_{\mu,\alpha_1}|\mu\rangle
\right).
\end{aligned}
$$

ここで

$$
|L_{\alpha_1}\rangle
:=
\sum_{s_1}U_{s_1,\alpha_1}|s_1\rangle,
$$

$$
|R_{\alpha_1}\rangle
:=
\sum_{\mu}V^*_{\mu,\alpha_1}|\mu\rangle
$$

と定義すると、

$$
\boxed{
|\Psi\rangle
=
\sum_{\alpha_1=1}^{r_1}
\lambda_{\alpha_1}
|L_{\alpha_1}\rangle
\otimes
|R_{\alpha_1}\rangle
}
$$

となる。これが切断 $1\mid23$ に対するSchmidt分解である。

SVDの列直交性から

$$
\langle L_{\beta_1}|L_{\alpha_1}\rangle
=
\delta_{\beta_1\alpha_1},
$$

$$
\langle R_{\beta_1}|R_{\alpha_1}\rangle
=
\delta_{\beta_1\alpha_1}.
$$

ここでは直交性も、係数を代入して確認しておく。物理基底の直交性を使うと、

$$
\begin{aligned}
\langle L_{\beta_1}|L_{\alpha_1}\rangle
&=\sum_{t_1,s_1}
U^*_{t_1,\beta_1}U_{s_1,\alpha_1}
\langle t_1|s_1\rangle\\
&=\sum_{t_1,s_1}
U^*_{t_1,\beta_1}U_{s_1,\alpha_1}\delta_{t_1,s_1}\\
&=\sum_{s_1}U^*_{s_1,\beta_1}U_{s_1,\alpha_1}\\
&=(U^\dagger U)_{\beta_1,\alpha_1}\\
&=\delta_{\beta_1,\alpha_1},
\end{aligned}
$$

$$
\begin{aligned}
\langle R_{\beta_1}|R_{\alpha_1}\rangle
&=\sum_{\nu,\mu}
V_{\nu,\beta_1}V^*_{\mu,\alpha_1}
\langle\nu|\mu\rangle\\
&=\sum_\mu V_{\mu,\beta_1}V^*_{\mu,\alpha_1}\\
&=(V^\dagger V)_{\alpha_1,\beta_1}\\
&=\delta_{\alpha_1,\beta_1}.
\end{aligned}
$$

右ketの係数が $V^*$ なので、braではさらに共役されて $V$ になる。右Schmidt状態の係数を、複素数の場合にも無条件で $V$ と置かない。

また、状態の規格化は

$$
\begin{aligned}
\langle\Psi|\Psi\rangle
&=\sum_{\alpha_1,\beta_1}
\lambda_{\beta_1}\lambda_{\alpha_1}
\langle L_{\beta_1}|L_{\alpha_1}\rangle
\langle R_{\beta_1}|R_{\alpha_1}\rangle\\
&=\sum_{\alpha_1,\beta_1}
\lambda_{\beta_1}\lambda_{\alpha_1}
\delta_{\beta_1,\alpha_1}\delta_{\beta_1,\alpha_1}\\
&=\sum_{\alpha_1}\lambda_{\alpha_1}^2
=\|\Psi^{\langle1\rangle}\|_F^2
\end{aligned}
$$

である。規格化された量子状態なら最後の値は1であり、一般の数値テンソルなら1とは限らない。

したがって対応は

$$
\boxed{
\text{SVDの左特異ベクトル}
\leftrightarrow
\text{左Schmidtベクトル}
}
$$

$$
\boxed{
\text{SVDの右特異ベクトル}
\leftrightarrow
\text{右Schmidtベクトル}
}
$$

$$
\boxed{
\text{特異値}
\leftrightarrow
\text{Schmidt係数}
}
$$

$$
\boxed{
\text{行列rank}
\leftrightarrow
\text{Schmidt rank}
}
$$

である。

### 小さい係数行列でSchmidt対応を確認

$s_1,\mu\in\{0,1\}$ とし、正規化された状態の係数行列を

$$
\Psi^{\langle1\rangle}
=
\dfrac{1}{\sqrt{5}}
\begin{pmatrix}
2&0\\
0&1
\end{pmatrix}
$$

とする。この行列のSVDは

$$
\Psi^{\langle1\rangle}
=
\underbrace{
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
}_{U}
\underbrace{
\begin{pmatrix}
\dfrac{2}{\sqrt{5}}&0\\
0&\dfrac{1}{\sqrt{5}}
\end{pmatrix}
}_{\Sigma}
\underbrace{
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
}_{V^T}
$$

である。係数を状態展開へ戻すと

$$
\begin{aligned}
|\Psi\rangle
&=
\sum_{s_1,\mu}
\Psi^{\langle1\rangle}_{s_1,\mu}
|s_1\rangle\otimes|\mu\rangle\\
&=
\dfrac{2}{\sqrt{5}}
|0\rangle\otimes|0\rangle
+
\dfrac{1}{\sqrt{5}}
|1\rangle\otimes|1\rangle.
\end{aligned}
$$

したがって、左・右特異ベクトルはそれぞれ左右のSchmidtベクトル、特異値 $2/\sqrt{5}$ と $1/\sqrt{5}$ はSchmidt係数であり、Schmidt rankは係数行列のrankと同じ2である。

---

## 6. TTとMPSで同じ対象をどう読むか

| 応用数学・数値線形代数 | 物理・MPS |
| --- | --- |
| Tensor Train | Matrix Product State |
| mode index $i_k$ | local physical index $s_k$ |
| TT core $G^{(k)}$ | site tensor $A^{[k]}$ |
| TT-rank $r_k$ | bond dimension |
| cut unfolding | bipartition |
| SVD | Schmidt decomposition |
| singular value | Schmidt coefficient |
| truncated SVD | 小さいSchmidt成分の切り捨て |

この対応を使うと、TTを「高階テンソルの低ランク表現」として読む視点と、MPSを「部分系間の相関をbondに保持する状態表現」として読む視点を同時に持てる。

---

## 7. TTパラメータ数

TTコア列が保持する総要素数は

$$
\boxed{
P_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}n_kr_k
}
$$

である。

元のdenseテンソルは

$$
P_{\mathrm{dense}}
=
\prod_{k=1}^{d}n_k
$$

個の値を持つ。TTが圧縮になるかどうかは、TT-rankが十分小さく

$$
P_{\mathrm{TT}}
<
P_{\mathrm{dense}}
$$

となるかで決まる。小さいテンソルでは、TT表現の方がパラメータ数が多くなる場合もある。

---

## 8. この章で固定する理解

- TTコアは各modeを担当する小さな3階テンソルである。
- 第 $k$ コアのshapeは $(r_{k-1},n_k,r_k)$。
- $n_k$ はphysical/mode index、$r_{k-1},r_k$ は左右bond dimension。
- 両端の $1$ は境界を統一形式で表すdummy axisであり、新しい自由度ではない。
- TTは内部bondを縮約して元テンソルを復元する。
- 開放境界MPSとTTは同じ鎖状テンソルネットワークである。
- 係数行列のSVDを状態として読み直すとSchmidt分解になる。

次は [[31_TT-rankとunfolding]] で、bond dimensionがなぜ元テンソルの各cut rankと結び付くかを整理する。
