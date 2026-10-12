---
title: TT-matrixのtensorization比較とcut別特異値エネルギー
tags:
  - TT-matrix
  - tensorization
  - unfolding
  - singular-value
  - retained-energy
  - parameter-budget
---

# TT-matrixのtensorization比較とcut別特異値エネルギー

## 1. このノートの到達点

同じdense weightを同程度のparameter数でTT-matrix化しても、tensorizationによって近似誤差が変わる理由を整理する。

このノートは一般論を扱う。Fashion-MNISTの特定checkpointで得たcut別保持率、tensorization間の順位、その数値からの考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/02_fc1_tensorization比較]]に分ける。一般論として成立する因果関係と、1回の実験から得た仮説を混同しない。

$$
\text{tensorization}
\longrightarrow
\text{各cutのunfolding}
\longrightarrow
\text{特異値分布}
\longrightarrow
\text{指定rankでの保持エネルギー}
\longrightarrow
\text{weight近似誤差}
$$

この連鎖はweight近似までの説明である。classification accuracyまでを直接保証するものではない。分類指標との関係は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/72_TT圧縮の重み近似指標と分類性能指標の読み分け]]で扱う。

固定tensorizationでrankだけを変える場合は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/70_TT-matrixのrank_sweepと圧縮率_再構成誤差]]を参照する。

## 2. tensorizationは単なるshape表記ではない

dense weightを

$$
W\in\mathbb R^{M\times N}
$$

とする。TT-matrixでは、

$$
M=\prod_{k=1}^{d}m_k,
\qquad
N=\prod_{k=1}^{d}n_k
$$

となるmode列を選び、行indexと列indexをsiteごとの組

$$
(i_1,j_1),\ldots,(i_d,j_d)
$$

へ割り当てる。各siteのphysical sizeは、

$$
q_k=m_kn_k
$$

である。

modeの順序や組合せを変えると、各cutの左側・右側に集められる元のindexが変わる。従って、同じ要素を保存していても、cutで作るunfoldingは異なる行列になる。

### 2.1 小さい数値例

4個の値

$$
(1,0,0,1)
$$

を、あるcutで

$$
A=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
$$

と並べると、$\operatorname{rank}(A)=2$である。一方、indexの割当てを変えて、

$$
B=
\begin{pmatrix}
1&1\\
0&0
\end{pmatrix}
$$

と並べると、$\operatorname{rank}(B)=1$になる。

元の値の集合は同じでも、unfoldingの行・列へどのindexを割り当てるかによってrankと特異値分布が変わる。TT-matrixのtensorization比較でも同じことが各cutで起こる。

## 3. 3-core TT-matrixのparameter数

physical sizeを$(q_1,q_2,q_3)$、TT-rankを

$$
(r_0,r_1,r_2,r_3)=(1,r_1,r_2,1)
$$

とすると、core parameter数は、

$$
P_{\mathrm{core}}
=
q_1r_1+r_1q_2r_2+r_2q_3
$$

である。

同じ共通rank$r_1=r_2=r$でも、$(q_1,q_2,q_3)$が変わればparameter数は変わる。反対に、同じparameter予算を与えても、各tensorizationで選べるrankは一般に一致しない。

従って公平な比較では、「同じrank」か「同じparameter数」か「同じparameter上限」かを明記する必要がある。離散的なrank候補から選ぶ場合、同じ予算上限でも実際のparameter数が完全には一致しないことがある。

## 4. 各cutで取り得る最大rank

physical tensorのshapeを$(q_1,\ldots,q_d)$とすると、cut $k$のunfolding shapeは、

$$
\left(
\prod_{j=1}^{k}q_j
\right)
\times
\left(
\prod_{j=k+1}^{d}q_j
\right)
$$

である。従って、取り得る最大rankは、

$$
r_k^{\max}
=
\min\left(
\prod_{j=1}^{k}q_j,
\prod_{j=k+1}^{d}q_j
\right)
$$

となる。

### 4.1 3-coreの数値例

$$
(q_1,q_2,q_3)=(64,64,98)
$$

なら、

$$
r_1^{\max}=\min(64,64\times98)=64
$$

$$
r_2^{\max}=\min(64\times64,98)=98
$$

である。実rankが$(1,64,64,1)$なら、cut 1はフルランクだが、cut 2は最大98に対して64で打ち切っている。全cutが最大rankでない限り、単に「rank 64だからフルランク」とは言えない。

## 5. cut別の特異値エネルギー保持率

cut $k$のSVDで得た特異値を、

$$
\sigma_1^{(k)}\ge\sigma_2^{(k)}\ge\cdots
$$

とする。rank $r_k$まで残すときの保持率を、

$$
\rho_k
=
\frac{
\sum_{j=1}^{r_k}\left(\sigma_j^{(k)}\right)^2
}{
\sum_j\left(\sigma_j^{(k)}\right)^2
}
$$

と定義する。

$\rho_k$が小さいcutほど、そのrank制約の下で多くのFrobeniusエネルギーを捨てている。ただし、後段のcutは前段までに圧縮された中間tensorへ作用する。cutごとの値は、互いに独立な元weightのSVDではない。

## 6. global保持率

逐次TT-SVDで、各cutの保持率をそのcutへ入る中間tensorのFrobeniusノルムで正規化すると、

$$
\rho_{\mathrm{global}}
=
\prod_{k=1}^{d-1}\rho_k
$$

と書ける。

TT-SVDの相対誤差上界を$\varepsilon_{\mathrm{rel}}$とすれば、同じ逐次打ち切りから、

$$
\rho_{\mathrm{global}}
=
1-\varepsilon_{\mathrm{rel}}^2
$$

と対応づけられる。

これは「元weightのFrobeniusエネルギーを、逐次打ち切り後にどれだけ保持したか」を読む診断量である。classificationに重要な方向へ重み付けした量ではない。

### 6.1 小さい数値例

2つのcutで、

$$
\rho_1=\frac{13}{14},
\qquad
\rho_2=\frac{16}{17}
$$

なら、

$$
\rho_{\mathrm{global}}
=
\frac{13}{14}\frac{16}{17}
=
\frac{208}{238}
\approx0.8739
$$

である。対応する相対誤差上界は、

$$
\varepsilon_{\mathrm{rel}}
=
\sqrt{1-\rho_{\mathrm{global}}}
\approx0.3551
$$

となる。

## 7. ボトルネックとなるcutはtensorizationで移動する

あるtensorizationでcut 1の保持率が低く、cut 2が1に近くても、別のtensorizationでは逆になることがある。これは、site順序を変えたことで、各cutをまたぐ構造を表現するために必要なbond dimensionが変わったことを表す。

物性物理のMPSになぞらえると、site順序の変更によって、相関をまたがせる境界が移動することに対応する。ただし、ニューラルネットワークのweight tensorに対して、量子状態のentanglement entropyと同一の物理的意味を与えるわけではない。

後段のcutが100%を保持しても、前段で捨てた成分は回復しない。従って、cut別保持率とglobal保持率を併せて読む。

## 8. 同じparameter数でも結果が変わる理由

同じparameter数が意味するのは、保存できるスカラーの個数が同程度ということだけである。異なるtensorizationが、同じ部分空間や同じ相関構造を表現することは意味しない。

$$
\text{同程度のparameter数}
\centernot\Longrightarrow
\text{同じunfolding}
\centernot\Longrightarrow
\text{同じ特異値分布}
$$

従って、同じparameter予算でも、保持率とweight近似誤差は変わり得る。

## 9. この理論から言えることと言えないこと

### 言えること

- tensorizationを変えると各cutのunfoldingが変わる
- unfoldingが変わると特異値分布と最大rankが変わる
- 同じrankやparameter予算でも、保持率とweight近似誤差は変わり得る
- cut別保持率から、rank制約が強く作用した境界を診断できる
- global保持率は逐次TT-SVD全体のFrobeniusエネルギー保持を表す

### これだけでは言えないこと

- global保持率が高ければclassification accuracyも必ず高い
- 特異値の大きい成分ほど、必ず分類に重要である
- あるtensorizationが異なるcheckpointや層でも常に最良である
- validationで選ばれた構成がtest setでも最良である

## 10. 実験への接続

Fashion-MNISTの保存結果と具体的なcut別保持率は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/02_fc1_tensorization比較]]で扱う。
