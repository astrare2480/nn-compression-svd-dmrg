---
title: テンソルとmode演算
aliases:
  - Tensorの基礎
  - mode-n unfolding
  - mode product
  - unfoldとfold
tags:
  - Tensor
  - Tucker
  - HOSVD
  - HOOI
  - 線形代数
---

# テンソルとmode演算

## サマリー

Tucker分解・HOSVD・HOOIを理解するには、Tensorを「多次元配列」として見るだけでなく、**各軸をmodeとして取り出し、行列化（unfold）し、行列を特定modeへ掛ける**という操作を理解する必要がある。

$N$階Tensorを

$$
\mathcal{X}
\in
\mathbb{R}^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

とする。mode $n$ は第$n$軸を表し、mode-$n$ unfoldingはその軸を行方向へ移した行列

$$
X_{(n)}
\in
\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

を作る操作である。

mode-$n$ productは、行列

$$
A\in\mathbb{R}^{J\times I_n}
$$

をmode $n$へ掛け、

$$
\mathcal{Y}
=
\mathcal{X}\times_n A
$$

として第$n$軸だけを $I_n\rightarrow J$ に変換する。

この2操作が、

```text
Tensor
→ 対象modeをunfold
→ SVD
→ factorを得る
```

というHOSVDと、

```text
Tensor
→ 他modeのfactorで射影
→ 対象modeをunfold
→ SVD
→ factorを更新
```

というHOOIの共通部品になる。

---

## 1. Tensorとmode

ベクトルは1階、行列は2階、Conv2dのweightは4階Tensorである。

PyTorchの通常のConv2d weightは、

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

で、各modeは次の意味を持つ。

| mode | 軸 | 意味 |
| ---: | --- | --- |
| 0 | $C_{\mathrm{out}}$ | 出力channel |
| 1 | $C_{\mathrm{in}}$ | 入力channel |
| 2 | $K_h$ | kernel高さ |
| 3 | $K_w$ | kernel幅 |

今回のTucker-2では、channel方向のmode 0 / 1だけを圧縮し、空間mode 2 / 3は保持する。

---

## 2. mode-n unfolding

### 行を選ぶ軸と、列にまとめる軸は役割が違う

mode $n$ の一つの行を選ぶと、そのmodeの成分番号だけが固定され、残りの全添字の値が列に並ぶ。
一方、列を一つ選ぶと残りの添字が全て固定され、mode $n$ だけを動かす一本のベクトルが得られる。
左特異ベクトルはmode $n$ の成分番号で書かれた長さ $I_n$ のベクトルであり、$X_{(n)}$ の全列が主にどの方向へ広がるかを表す。
HOSVDで選ぶのは $X_{(n)}$ のcolumn space側の主要部分空間であって、残りのmodeをまとめたrow space側ではない。

3階Tensorのmode 1なら、列番号を $c(i_0,i_2)=(i_0-1)I_2+i_2$ と固定して

$$
[X_{(1)}]_{i_1,c(i_0,i_2)}=X_{i_0,i_1,i_2}
$$

と読む。列 $c$ の長さは $I_1$ であり、一つの $(i_0,i_2)$ の組に対する $X_{i_0,:,i_2}$ である。
全列が選んだfactorの列空間に入れば、そのmodeでの座標圧縮からexactに再構成できる。
入らない列があれば、その保持部分空間に直交する成分が近似誤差になる。
unfoldingそのものでは値も要素数も失われず、失われる可能性があるのは、その後に部分空間を小さく選ぶ段階である。

### 定義

Tensor

$$
\mathcal{X}
\in
\mathbb{R}^{I_0\times\cdots\times I_{N-1}}
$$

をmode $n$でunfoldすると、

$$
X_{(n)}
\in
\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

という行列になる。

重要なのは、**mode $n$の次元を行方向に置き、残りの軸を列方向へまとめる**こと。

Conv weight

$$
W\in\mathbb{R}^{64\times32\times3\times3}
$$

なら、mode 0 unfoldingは

$$
W_{(0)}
\in
\mathbb{R}^{64\times(32\cdot3\cdot3)}
=
\mathbb{R}^{64\times288}
$$

mode 1 unfoldingは

$$
W_{(1)}
\in
\mathbb{R}^{32\times(64\cdot3\cdot3)}
=
\mathbb{R}^{32\times576}
$$

になる。

この行列へ通常のSVDを行えば、mode 0では出力channel側、mode 1では入力channel側の主要部分空間を得られる。

### SVDとHOSVDの名前の違い

単一modeをunfoldして行うのは、あくまで**1回の行列SVD**である。

```text
mode 0だけunfold → SVD
= mode 0 unfoldingに対する行列SVD

mode 1だけunfold → SVD
= mode 1 unfoldingに対する行列SVD
```

複数modeについて元Tensorをそれぞれunfoldし、factor一式とcoreを組み立てるアルゴリズム全体をHOSVDと呼ぶ。

---

## 3. fold

`fold`はunfoldの逆操作である。

$$
\operatorname{fold}_n(X_{(n)})=\mathcal{X}
$$

したがって、正しく実装されているなら

$$
\operatorname{fold}_n(
\operatorname{unfold}_n(\mathcal{X})
)
=
\mathcal{X}
$$

が成立する。

自作srcでは、対象modeを先頭へ移してflattenし、foldでは元shapeを復元した後、先頭modeを元位置へ戻すことでこの対応を実装する。

参照実装：`src/nn_compression/tensor/operations.py`

---

## 4. mode-n product

### 他の添字を固定し、そのmodeに沿うベクトルだけを変換する

例えばmode 1では、$(i_0,i_2)$ を固定した $X_{i_0,:,i_2}$ に行列 $A$ を掛ける。
同じ $A$ を全ての $(i_0,i_2)$ の組に使い、変換後のベクトルを元の位置へ並べる。
従って対象modeだけが変わり、他のmodeが勝手に混ざったり消えたりすることはない。

$$
v^{(i_0,i_2)}_{i_1}:=X_{i_0,i_1,i_2},
\qquad
w^{(i_0,i_2)}_j:=\sum_{i_1=1}^{I_1}A_{j,i_1}v^{(i_0,i_2)}_{i_1},
\qquad
Y_{i_0,j,i_2}:=w^{(i_0,i_2)}_j.
$$

和を取る $i_1$ は行列 $A$ の列側の入力番号、新しく残る $j$ は $A$ の行側の出力番号である。
「Tensorへ行列を掛ける」という記号だけを見るより、この入力・出力の番号を照合すると転置の向きを判断しやすい。

行列

$$
A
\in
\mathbb{R}^{J\times I_n}
$$

をTensor $\mathcal{X}$ のmode $n$へ掛ける操作を

$$
\mathcal{Y}
=
\mathcal{X}\times_n A
$$

と書く。

結果のshapeは

$$
\mathcal{Y}
\in
\mathbb{R}^{
I_0\times\cdots\times I_{n-1}
\times J
\times I_{n+1}\times\cdots\times I_{N-1}
}
$$

となり、第$n$軸だけが $I_n\rightarrow J$ に変わる。

要素表示では、

$$
Y_{i_0,\ldots,i_{n-1},j,i_{n+1},\ldots,i_{N-1}}
=
\sum_{i_n=1}^{I_n}
A_{j,i_n}
X_{i_0,\ldots,i_n,\ldots,i_{N-1}}
$$

である。

unfoldを使えば、

$$
Y_{(n)}=AX_{(n)}
$$

と理解できる。

### 成分和と行列積・foldを同じ全要素で確認する

この具体例だけ、配列要素はPythonと同じ0始まりとする。

$$
X_{0,:,:}=\begin{pmatrix}1&0\\2&3\end{pmatrix},
\qquad
X_{1,:,:}=\begin{pmatrix}4&5\\6&7\end{pmatrix},
\qquad
A=\begin{pmatrix}2&-1\end{pmatrix}.
$$

mode 1を行へ移し、列の残りの添字を

$$
(i_0,i_2)=(0,0),(0,1),(1,0),(1,1)
$$

の順に並べる。unfolding・積・fold後の全要素は

$$
X_{(1)}=\begin{pmatrix}1&0&4&5\\2&3&6&7\end{pmatrix},
$$

$$
\begin{aligned}
Y_{(1)}
&=AX_{(1)}\\
&=\begin{pmatrix}2\cdot1-2&2\cdot0-3&2\cdot4-6&2\cdot5-7\end{pmatrix}\\
&=\begin{pmatrix}0&-3&2&3\end{pmatrix},
\end{aligned}
$$

$$
Y_{0,0,:}=\begin{pmatrix}0&-3\end{pmatrix},
\qquad
Y_{1,0,:}=\begin{pmatrix}2&3\end{pmatrix},
\qquad
\operatorname{shape}(Y)=(2,1,2)
$$

となる。例えば

$$
Y_{1,0,1}
=\sum_{i_1=0}^{1}A_{0,i_1}X_{1,i_1,1}
=2X_{1,0,1}-X_{1,1,1}
=2\cdot5-7
=3
$$

である。unfoldingは値を並べるだけ、mode productの行列積は値を線形結合し、foldはその結果の軸を戻す。この三つを一つのreshape操作として扱わない。

### 異なるmodeでは交換でき、同じmodeでは積の順序を保つ

ここからは一般式の1始まりの要素添字へ戻す。異なるmode 0 / 1へ作用させると、

$$
\begin{aligned}
[(\mathcal X\times_0A)\times_1B]_{j_0,j_1,i_2}
&=\sum_{i_1}B_{j_1,i_1}
\left(\sum_{i_0}A_{j_0,i_0}X_{i_0,i_1,i_2}\right)\\
&=\sum_{i_0,i_1}A_{j_0,i_0}B_{j_1,i_1}X_{i_0,i_1,i_2}\\
&=[(\mathcal X\times_1B)\times_0A]_{j_0,j_1,i_2}.
\end{aligned}
$$

一方、同じmodeへ $A$ の後に $B$ を作用させると、

$$
\begin{aligned}
[(\mathcal X\times_0A)\times_0B]_{k_0,i_1,i_2}
&=\sum_{j_0}B_{k_0,j_0}\sum_{i_0}A_{j_0,i_0}X_{i_0,i_1,i_2}\\
&=\sum_{i_0}\left(\sum_{j_0}B_{k_0,j_0}A_{j_0,i_0}\right)X_{i_0,i_1,i_2}\\
&=[\mathcal X\times_0(BA)]_{k_0,i_1,i_2}.
\end{aligned}
$$

同じmodeの場合は $BA$ であって $AB$ ではない。この区別により、factor転置で圧縮した後にfactorで展開すると、そのmodeへ $UU^{\mathsf T}$ が作用すると理解できる。

---

## 5. factorの転置が「圧縮方向」になる理由

shapeが合うことは必要条件だが、それだけで転置が適切な座標抽出になるわけではない。
HOSVD・HOOIのようにfactorの列 $u_\alpha$ が正規直交するとき、
対象modeに沿う一本のベクトル $v$ は、保持部分 $Uc$ とそれに直交する残差 $e$ に分けられる。

$$
v=Uc+e,\qquad U^{\mathsf T}e=0,\qquad U^{\mathsf T}U=I,
$$

$$
U^{\mathsf T}v
=U^{\mathsf T}Uc+U^{\mathsf T}e
=c+0=c.
$$

これが転置を使う理由である。再展開すると $UU^{\mathsf T}v=Uc$ であり、残差 $e$ は戻らない。
一般の非直交factorにまで「shapeが合うから転置でよい」と広げない。
具体的な座標・残差・二つのGramは [[22_Tucker分解とHOSVD#2. factorとcoreのshape]] で計算する。

Tuckerのfactorを

$$
U^{(n)}
\in
\mathbb{R}^{I_n\times R_n}
$$

とする。

元のmode次元 $I_n$ をrank次元 $R_n$ へ落とすには、

$$
U^{(n)\mathsf{T}}
\in
\mathbb{R}^{R_n\times I_n}
$$

を掛ける。

$$
I_n
\overset{U^{(n)\mathsf{T}}}{\longrightarrow}
R_n
$$

逆に、rank空間から元の空間へ戻すときは転置しないfactorを使う。

$$
R_n
\overset{U^{(n)}}{\longrightarrow}
I_n
$$

これはConv Tucker-2で、入力側に $U_{\mathrm{in}}^{\mathsf{T}}$、出力側に $U_{\mathrm{out}}$ を使う理由そのものである。

---

## 6. Conv weightでの具体例

$$
W
\in
\mathbb{R}^{64\times32\times3\times3}
$$

に対し、入力channel側factorを

$$
U_{\mathrm{in}}
\in
\mathbb{R}^{32\times R_{\mathrm{in}}}
$$

とする。

mode 1をrankへ圧縮すると、

$$
W\times_1 U_{\mathrm{in}}^{\mathsf{T}}
\in
\mathbb{R}^{64\times R_{\mathrm{in}}\times3\times3}
$$

になる。

出力channel側factor

$$
U_{\mathrm{out}}
\in
\mathbb{R}^{64\times R_{\mathrm{out}}}
$$

でも射影すれば、

$$
W
\times_0 U_{\mathrm{out}}^{\mathsf{T}}
\times_1 U_{\mathrm{in}}^{\mathsf{T}}
\in
\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times3\times3}
$$

となり、これがTucker-2のcore shapeになる。

---

## 7. 実装との対応

今回のsrcでは次を共通部品としている。

```text
src/nn_compression/tensor/operations.py
├─ unfold
├─ fold
└─ mode_dot
```

学習Notebook：

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb
```

次に読む：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
