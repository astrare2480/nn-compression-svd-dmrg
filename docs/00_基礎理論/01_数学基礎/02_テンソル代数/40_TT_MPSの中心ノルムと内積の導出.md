---
title: TT・MPSの中心ノルムと内積の導出
aliases:
  - mixed-canonicalの中心ノルム
  - 中心コアの等長な埋め込み
tags:
  - TT
  - MPS
  - canonical
  - isometry
  - derivation
---

# TT・MPSの中心ノルムと内積の導出

## この章の範囲

中心コアの座標としての意味、ノルム・差分・内積の導出と非正準形との比較を扱う。
有限和の並べ替え、独立した二組のボンド添字、Gramからデルタへの置換、
デルタで和を消す手順を省略せず確認する。

[[39_TT_MPSの混合正準形への導入]] の後続理論であり、
同章の導入範囲やNotebookの学習進捗を変更するものではない。
中心移動、中心SVD、Schmidt係数、rounding、MPO、one-site/two-site最適化の
実装は扱わない。本章の数値例は、導出を検証するための独自の補足例である。

## 1. 出発点：左右の基底を固定した中心コア

実数の3階TTで、中心を第2サイトへ置く。
QR因子を両側から中心へ吸収するまでの導出は [[39_TT_MPSの混合正準形への導入]] にある。
ここでは、その結果を次の係数で書く。

$$
L(i_1,\alpha_1)=G_1^{[L]}(1,i_1,\alpha_1),\qquad
C(\alpha_1,i_2,\alpha_2)=G_2^{[C]}(\alpha_1,i_2,\alpha_2),\qquad
R(\alpha_2,i_3)=G_3^{[R]}(\alpha_2,i_3,1).
$$

$$
L\in\mathbb R^{n_1\times r_1},\qquad
C\in\mathbb R^{r_1\times n_2\times r_2},\qquad
R\in\mathbb R^{r_2\times n_3}.
$$

$$
X(i_1,i_2,i_3)
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3).
$$

左右のmixed-canonical条件は、行列表示と成分表示で

$$
L^T L=I_{r_1},\qquad R R^T=I_{r_2},
$$

$$
\sum_{i_1=1}^{n_1}L(i_1,\alpha_1)L(i_1,\alpha_1')
=\delta_{\alpha_1,\alpha_1'},\qquad
\sum_{i_3=1}^{n_3}R(\alpha_2,i_3)R(\alpha_2',i_3)
=\delta_{\alpha_2,\alpha_2'}
$$

である。中心 $C$ 自身には、左右いずれの直交性も要求しない。
中心が必ず非直交という意味ではなく、直交条件を課さず全体の係数を持たせるという意味である。

コアunfoldingの直交条件を、行列の略記へ対応させる。
境界軸を除いた $G_1^{[L]\langle L\rangle}=L$ と
$G_{3,\mathrm{right}}^{[R]}=R$ のGramだけがmixed-canonicalの条件である。
中心を左右それぞれにunfoldするなら

$$
C_L((\alpha_1,i_2),\alpha_2)=C(\alpha_1,i_2,\alpha_2),\qquad
C_L\in\mathbb R^{(r_1n_2)\times r_2},
$$

$$
C_R(\alpha_1,(i_2,\alpha_2))=C(\alpha_1,i_2,\alpha_2),\qquad
C_R\in\mathbb R^{r_1\times(n_2r_2)}.
$$

そのGramの要素は

$$
(C_L^T C_L)_{\alpha_2,\alpha_2'}
=\sum_{\alpha_1,i_2}C(\alpha_1,i_2,\alpha_2)C(\alpha_1,i_2,\alpha_2'),
$$

$$
(C_R C_R^T)_{\alpha_1,\alpha_1'}
=\sum_{i_2,\alpha_2}C(\alpha_1,i_2,\alpha_2)C(\alpha_1',i_2,\alpha_2).
$$

一般には $C_L^T C_L\ne I_{r_2}$、$C_R C_R^T\ne I_{r_1}$ でよい。
これらの不等式自体を必須条件とするのでもなく、偶然どちらかが単位行列でも構わない。
中心が保持するスケール・結合を、そのまま左右どちらのGramも単位行列へ揃える必要はない。

左の列は $|L_{\alpha_1}\rangle$、右の行は $|R_{\alpha_2}\rangle$ の係数を持つ。
物理基底が正規直交なら

$$
\begin{aligned}
|L_{\alpha_1}\rangle&=\sum_{i_1}L(i_1,\alpha_1)|i_1\rangle,\\
|R_{\alpha_2}\rangle&=\sum_{i_3}R(\alpha_2,i_3)|i_3\rangle,\\
|X\rangle&=\sum_{\alpha_1,i_2,\alpha_2}C(\alpha_1,i_2,\alpha_2)
|L_{\alpha_1}\rangle\otimes|i_2\rangle\otimes|R_{\alpha_2}\rangle.
\end{aligned}
$$

左右が直交する理由は、[[37_TT_MPSの左ブロックと直交性の導出]] と
[[38_TT_MPSの右QR直交化と右ブロック]] で、コアの直交条件からブロックのGramまで展開している。
中心が第2サイトなら左環境は第1サイト、右環境は第3サイトである。
左環境へ第2コアまで含めると中心を二重に数える。

「座標系を整える」とは、
この積基底が正規直交になり、$C$ が通常の直交座標の係数になることを意味する。
基底は全物理空間の基底とは限らず、左右のTTブロックが張る部分空間の基底である。

## 2. ノルム：二つの独立したTT収縮を掛ける

Frobeniusノルムの二乗の定義から出発する。

$$
\|X\|_F^2=\sum_{i_1=1}^{n_1}\sum_{i_2=1}^{n_2}\sum_{i_3=1}^{n_3}
X(i_1,i_2,i_3)^2.
$$

二つの $X$ へ独立したボンド添字を付けて、TTの式を代入する。
最初から $\alpha_1'=\alpha_1,\alpha_2'=\alpha_2$ と置いてはいけない。

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{i_1,i_2,i_3}
\left[\sum_{\alpha_1,\alpha_2}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3)\right]\\
&\quad\times
\left[\sum_{\alpha_1',\alpha_2'}
L(i_1,\alpha_1')C(\alpha_1',i_2,\alpha_2')R(\alpha_2',i_3)\right]\\
&=\sum_{i_1,i_2,i_3}\sum_{\alpha_1,\alpha_2}
\sum_{\alpha_1',\alpha_2'}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3)
L(i_1,\alpha_1')C(\alpha_1',i_2,\alpha_2')R(\alpha_2',i_3).
\end{aligned}
$$

すべて有限和なので和の順を交換できる。
$C$ は $i_1,i_3$ に依存しないため、左右の物理添字の和の外へ出す。

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2}\sum_{\alpha_1',\alpha_2'}
C(\alpha_1,i_2,\alpha_2)C(\alpha_1',i_2,\alpha_2')\\
&\quad\times
\left[\sum_{i_1}L(i_1,\alpha_1)L(i_1,\alpha_1')\right]
\left[\sum_{i_3}R(\alpha_2,i_3)R(\alpha_2',i_3)\right].
\end{aligned}
$$

角括弧はそれぞれ左・右のGramの成分である。
左右の直交条件をここで初めて使う。

$$
\begin{aligned}
\|X\|_F^2
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2}\sum_{\alpha_1',\alpha_2'}
C(\alpha_1,i_2,\alpha_2)C(\alpha_1',i_2,\alpha_2')
\delta_{\alpha_1,\alpha_1'}\delta_{\alpha_2,\alpha_2'}\\
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2}
C(\alpha_1,i_2,\alpha_2)
\left[\sum_{\alpha_1',\alpha_2'}
C(\alpha_1',i_2,\alpha_2')
\delta_{\alpha_1,\alpha_1'}\delta_{\alpha_2,\alpha_2'}\right]\\
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2}
C(\alpha_1,i_2,\alpha_2)C(\alpha_1,i_2,\alpha_2)\\
&=\sum_{\alpha_1=1}^{r_1}\sum_{i_2=1}^{n_2}\sum_{\alpha_2=1}^{r_2}
C(\alpha_1,i_2,\alpha_2)^2
=\|C\|_F^2.
\end{aligned}
$$

両ノルムは非負なので

$$
\boxed{\|X\|_F=\|C\|_F}.
$$

「クロス項が消える」とは、異なるボンド状態を比較する項には
$L(:,\alpha_1)^T L(:,\alpha_1')=0$ または
$R(\alpha_2,:)R(\alpha_2',:)^T=0$ が掛かるという意味である。
中心の異なる係数がどれもゼロになるのではなく、
ノルムの二重展開で異なる積基底間の寄与がゼロになり、各係数の二乗だけが残る。

全状態が規格化されていれば $\|C\|_F=1$ である。
左右コアのGramが単位行列という条件だけで、$\|X\|_F=1$ が保証されるわけではない。

## 3. 非正準形では環境のGramが残る

正準形と非正準形の違いを、同じ和の順序で確認する。
左右が直交化されていない表現を

$$
X(i_1,i_2,i_3)=\sum_{\alpha_1,\alpha_2}
\widetilde L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)
\widetilde R(\alpha_2,i_3)
$$

とし

$$
\begin{aligned}
N_L&=\widetilde L^T\widetilde L,&
N_L(\alpha_1,\alpha_1')&=\sum_{i_1}
\widetilde L(i_1,\alpha_1)\widetilde L(i_1,\alpha_1'),\\
N_R&=\widetilde R\widetilde R^T,&
N_R(\alpha_2,\alpha_2')&=\sum_{i_3}
\widetilde R(\alpha_2,i_3)\widetilde R(\alpha_2',i_3)
\end{aligned}
$$

と定義する。前節の角括弧はデルタではなく、この二つのGramになる。

$$
\boxed{
\|X\|_F^2
=\sum_{i_2}\sum_{\alpha_1,\alpha_1',\alpha_2,\alpha_2'}
C(\alpha_1,i_2,\alpha_2)
N_L(\alpha_1,\alpha_1')
C(\alpha_1',i_2,\alpha_2')
N_R(\alpha_2,\alpha_2')
}.
$$

左右の基底が重なったり大きくスケールしたりしていれば、中心の係数だけでは
全体のノルムを単純な二乗和で測れない。mixed-canonical化は
$N_L=I_{r_1},N_R=I_{r_2}$ とすることで、この環境の重なりを取り除く。
そのため、同じ左右環境を固定した局所計算では中心の変化量と全テンソルの変化量を同じFrobeniusノルムで測れる。これは余分な環境Gramによる尺度の歪みを避けるという数値上の利点であり、どんな最適化手順も自動的に安定・高精度になるという保証ではない。
非正準形だから必ず特定の $C$ で両ノルムが異なる、という主張ではない。
偶然一致する $C$ はあり得るが、任意の $C$ に対する等長性は一般に保証されない。

## 4. 中心だけを変えた差分：微小近似ではなく厳密な線形性

左右を固定し、中心だけを $C+\Delta C$ へ変える。
各physical sliceを $C_{i_2}=C(:,i_2,:)$ とすると

$$
\begin{aligned}
X(:,i_2,:)&=LC_{i_2}R,\\
\widetilde X(:,i_2,:)&=L(C_{i_2}+\Delta C_{i_2})R,\\
\Delta X(:,i_2,:)
&:=\widetilde X(:,i_2,:)-X(:,i_2,:)\\
&=L(C_{i_2}+\Delta C_{i_2})R-LC_{i_2}R\\
&=LC_{i_2}R+L\Delta C_{i_2}R-LC_{i_2}R
=L\Delta C_{i_2}R.
\end{aligned}
$$

成分に戻すと

$$
\Delta X(i_1,i_2,i_3)=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
L(i_1,\alpha_1)\Delta C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3).
$$

他のコアを固定したTT収縮は中心について線形なので、
$\Delta C$ が微小でなくてもこれは厳密である。
前のノルム導出で $C$ を $\Delta C$ へ置き換えれば

$$
\begin{aligned}
\|\Delta X\|_F^2
&=\sum_{i_2,\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
\Delta C(\alpha_1,i_2,\alpha_2)\Delta C(\alpha_1',i_2,\alpha_2')
\delta_{\alpha_1,\alpha_1'}\delta_{\alpha_2,\alpha_2'}\\
&=\sum_{\alpha_1,i_2,\alpha_2}\Delta C(\alpha_1,i_2,\alpha_2)^2
=\|\Delta C\|_F^2.
\end{aligned}
$$

したがって固定環境の写像 $\mathcal F:C\mapsto X$ は

$$
\boxed{\|\widetilde X-X\|_F=\|\Delta C\|_F},\qquad
\boxed{\|\mathcal F(C)\|_F=\|C\|_F}
$$

を満たす等長写像である。
中心の誤差を全状態の誤差として測れることが、正準化の実用的な利点である。
同時に左右も変更した場合にはこの「中心だけの差分」の式をそのまま使えない。
非正準形では、差分の二乗和にも前節の $N_L,N_R$ が残る。

差分に対する式を、$C$ の場合への参照だけでなく書くと、

$$
\begin{aligned}
\|\Delta X\|_F^2
&=\sum_{i_2,\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
\Delta C(\alpha_1,i_2,\alpha_2)\Delta C(\alpha_1',i_2,\alpha_2')\\
&\quad\times
\left[\sum_{i_1}\widetilde L(i_1,\alpha_1)\widetilde L(i_1,\alpha_1')\right]
\left[\sum_{i_3}\widetilde R(\alpha_2,i_3)\widetilde R(\alpha_2',i_3)\right]\\
&=\sum_{i_2,\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
\Delta C(\alpha_1,i_2,\alpha_2)
N_L(\alpha_1,\alpha_1')
\Delta C(\alpha_1',i_2,\alpha_2')
N_R(\alpha_2,\alpha_2').
\end{aligned}
$$

直交性を使えないのでprimed添字の和をデルタで消せず、
一般には $\|\Delta X\|_F=\|\Delta C\|_F$ とはならない。

## 5. 同じ左右基底を共有する二つのTTの内積

左右は**同じ** $L,R$、中心だけを $C,D$ とする。

$$
X(i_1,i_2,i_3)=\sum_{\alpha_1,\alpha_2}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3),
$$

$$
Y(i_1,i_2,i_3)=\sum_{\alpha_1',\alpha_2'}
L(i_1,\alpha_1')D(\alpha_1',i_2,\alpha_2')R(\alpha_2',i_3).
$$

実数テンソルのFrobenius内積へ、二つの式を独立に代入する。

$$
\begin{aligned}
\langle X,Y\rangle_F
&=\sum_{i_1,i_2,i_3}X(i_1,i_2,i_3)Y(i_1,i_2,i_3)\\
&=\sum_{i_1,i_2,i_3}
\left[\sum_{\alpha_1,\alpha_2}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3)\right]
\left[\sum_{\alpha_1',\alpha_2'}
L(i_1,\alpha_1')D(\alpha_1',i_2,\alpha_2')R(\alpha_2',i_3)\right]\\
&=\sum_{i_1,i_2,i_3}\sum_{\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3)
L(i_1,\alpha_1')D(\alpha_1',i_2,\alpha_2')R(\alpha_2',i_3)\\
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
C(\alpha_1,i_2,\alpha_2)D(\alpha_1',i_2,\alpha_2')
\left[\sum_{i_1}L(i_1,\alpha_1)L(i_1,\alpha_1')\right]
\left[\sum_{i_3}R(\alpha_2,i_3)R(\alpha_2',i_3)\right]\\
&=\sum_{i_2}\sum_{\alpha_1,\alpha_2,\alpha_1',\alpha_2'}
C(\alpha_1,i_2,\alpha_2)D(\alpha_1',i_2,\alpha_2')
\delta_{\alpha_1,\alpha_1'}\delta_{\alpha_2,\alpha_2'}\\
&=\sum_{\alpha_1=1}^{r_1}\sum_{i_2=1}^{n_2}\sum_{\alpha_2=1}^{r_2}
C(\alpha_1,i_2,\alpha_2)D(\alpha_1,i_2,\alpha_2)
=\langle C,D\rangle_F.
\end{aligned}
$$

2行目では独立した二つのボンド和を掛け、3行目で積を一つの全和へ展開している。
最後のデルタの和はノルムの場合と同じように、primed添字を消す。
$D=C$ とすれば $\langle X,X\rangle_F=\langle C,C\rangle_F$ なので、
ノルム保存は内積保存の特別な場合と確認できる。

## 6. 左右が違う場合には相互の重なりを省略できない

左右の基底が異なる場合は、内積の計算に注意が必要である。
$X=LCR$ に対して $Y=\widehat L D\widehat R$ なら、
各TTの左右がそれぞれ正規直交でも、相互Gramは単位行列とは限らない。

$$
M_L=L^T\widehat L,\qquad M_R=R\widehat R^T.
$$

前節の物理添字の和をこの場合にも実行すると

$$
\begin{aligned}
\langle X,Y\rangle_F
&=\sum_{i_2,\alpha_1,\alpha_2,\beta_1,\beta_2}
C(\alpha_1,i_2,\alpha_2)D(\beta_1,i_2,\beta_2)
\left[\sum_{i_1}L(i_1,\alpha_1)\widehat L(i_1,\beta_1)\right]
\left[\sum_{i_3}R(\alpha_2,i_3)\widehat R(\beta_2,i_3)\right]\\
&=\sum_{i_2,\alpha_1,\alpha_2,\beta_1,\beta_2}
C(\alpha_1,i_2,\alpha_2)
M_L(\alpha_1,\beta_1)D(\beta_1,i_2,\beta_2)
M_R(\alpha_2,\beta_2).
\end{aligned}
$$

相互Gramを使い、最終的な和まで展開した。
「二つともmixed-canonicalだから、中心の係数をそのまま内積すればよい」
とは言えない。同じ左右基底・同じボンド座標を共有していることが条件である。
別々の基底では、中心係数は別々の座標系で表されている。

## 7. 小さい全要素例でノルム・差分・内積を確認する

導出に対応させた小さい数値例を考える。
$n_1=n_3=2,r_1=r_2=1,n_2=2$ とする。

$$
L=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix},\qquad
R=\frac1{\sqrt5}\begin{pmatrix}1&2\end{pmatrix},\qquad
C(1,:,1)=\begin{pmatrix}3&4\end{pmatrix},\qquad
D(1,:,1)=\begin{pmatrix}1&-1\end{pmatrix}.
$$

$$
L^TL=\frac{1^2+1^2}{2}=1,\qquad
RR^T=\frac{1^2+2^2}{5}=1,\qquad
LR=\frac1{\sqrt{10}}\begin{pmatrix}1&2\\1&2\end{pmatrix}.
$$

中心sliceは $1\times1$ の行列なので、全physical sliceは

$$
X(:,1,:)=\frac1{\sqrt{10}}\begin{pmatrix}3&6\\3&6\end{pmatrix},\qquad
X(:,2,:)=\frac1{\sqrt{10}}\begin{pmatrix}4&8\\4&8\end{pmatrix},
$$

$$
Y(:,1,:)=\frac1{\sqrt{10}}\begin{pmatrix}1&2\\1&2\end{pmatrix},\qquad
Y(:,2,:)=-\frac1{\sqrt{10}}\begin{pmatrix}1&2\\1&2\end{pmatrix}.
$$

全8要素の二乗を足すと

$$
\begin{aligned}
\|X\|_F^2
&=\frac{3^2+6^2+3^2+6^2+4^2+8^2+4^2+8^2}{10}\\
&=\frac{90+160}{10}=25=3^2+4^2=\|C\|_F^2.
\end{aligned}
$$

内積でも全要素を掛け合わせる。

$$
\begin{aligned}
\langle X,Y\rangle_F
&=\frac{3\cdot1+6\cdot2+3\cdot1+6\cdot2
+4\cdot(-1)+8\cdot(-2)+4\cdot(-1)+8\cdot(-2)}{10}\\
&=\frac{30-40}{10}=-1=3\cdot1+4\cdot(-1)=\langle C,D\rangle_F.
\end{aligned}
$$

$C$ を $D$ へ変える場合、$\Delta C=D-C$ の2要素は $(-2,-5)$ であり

$$
\begin{aligned}
\|Y-X\|_F^2
&=\frac{(-2)^2+(-4)^2+(-2)^2+(-4)^2
+(-5)^2+(-10)^2+(-5)^2+(-10)^2}{10}\\
&=\frac{40+250}{10}=29=(-2)^2+(-5)^2=\|\Delta C\|_F^2.
\end{aligned}
$$

非正準形の比較として左だけを $\widetilde L=2L$ へ変えると、
$N_L=4,N_R=1$ になり、同じ $C$ で作る全テンソルは $\widetilde X=2X$ になる。

$$
\|\widetilde X\|_F^2=4\|X\|_F^2=100,\qquad
\|C\|_F^2=25,\qquad
\sum_{i_2}C(1,i_2,1)^2N_LN_R=(3^2+4^2)\cdot4\cdot1=100.
$$

これは単に中心の係数を小さく見積もっても、環境のスケールによって
全体の変化が大きくなる場合を示している。

## 8. 一般の中心サイトと学習の境界

一般 $d$ 階では、中心 $c$ より左を左直交、
右を右直交にすることである。$r_0=r_d=1$ とし、成分で書くと

$$
\sum_{\alpha_{k-1},i_k}
G_k^{[L]}(\alpha_{k-1},i_k,\alpha_k)
G_k^{[L]}(\alpha_{k-1},i_k,\alpha_k')
=\delta_{\alpha_k,\alpha_k'},\qquad k<c,
$$

$$
\sum_{i_k,\alpha_k}
G_k^{[R]}(\alpha_{k-1},i_k,\alpha_k)
G_k^{[R]}(\alpha_{k-1}',i_k,\alpha_k)
=\delta_{\alpha_{k-1},\alpha_{k-1}'},\qquad k>c.
$$

複数サイトのブロックでも、これらの条件を外側から順に縮約すると
左・右ブロックのGramが単位行列になる。
中心の物理添字 $i_c$ を残せば、本章と同じ二つのGramの消去が成立する。
これは一般式への接続であり、別のNotebookのTODOを実装したという意味ではない。

実用上の利点は、規格化の確認、中心だけを変えた誤差の測定、
局所計算を通常の直交座標で行えること、将来の中心移動・局所最適化の準備である。
「正準化しただけで低rankになった」「中心のノルムだけでNNの精度を保証できる」
という意味ではない。打ち切りと誤差は [[33_TT-SVDの打ち切りと誤差]]、
TTとDMRGの学習段階は [[17_TT_MPS学習ロードマップ]] と区別する。

局所最適化・中心移動の実装は次の学習段階とする。
理論を読んだことと、Notebookの実装・理解の完了を混同しない。

## 9. 次の学習段階と演習範囲

次の学習段階は、中心の意味、中心だけの最小二乗問題、
QR/SVDによる中心移動、Schmidt分解・roundingとの関係、DMRGへの接続である。
この章ではDMRGへの接続は扱わない。
これは詳細な導出を済ませた章ではなく、次の学習への予告である。
中心コアにQRを適用するときは、3階コアそのものを通常の行列QRへ渡すのではない。
左unfoldingして

$$
C^{\langle L\rangle}\in\mathbb R^{(r_1n_2)\times r_2},\qquad
C^{\langle L\rangle}=Q_2T_2
$$

と分け、$Q_2$ をコアへ戻し、$T_2$ を第3コアへ吸収すれば、
中心を右へ移す構成へ接続する、という予告として読む。
この予告の式は記録するが、中心移動のコードや最適化問題の解答までは追加していない。
中心が「唯一の非直交部分」とは、中心以外に片側の直交条件を課すという意味であり、
中心がたまたま直交条件を満たす可能性を否定するものではない。

演習では、実装なしで一問一論点を次の順で確認する。

1. $G_2^{[C]}=T_1G_2T_3^T$ の添字を展開する。
2. 各収縮の前後のshapeを確認する。
3. 左右の直交条件をKronecker deltaで書く。
4. $\|X\|_F^2=\|G_2^{[C]}\|_F^2$ の途中式を埋める。
5. 中心だけの差分 $\Delta X$ の式と $\|\Delta X\|_F=\|\Delta G_2\|_F$ を導く。

同じ左右コアを共有するTTの内積も演習範囲に含む。
物理的解釈の詳説・中心最適化・中心移動・SVD/Schmidt・rounding・MPO/TT-matrix・DMRGの演習は別の学習段階とする。
本章の式を読んだことを、これらの演習をユーザーが解いた記録として扱わない。

中心だけを変えた差分のPyTorch確認は [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]]、この先の中心移動は [[41_TT_MPSの直交中心の移動]]、移動操作のPyTorch確認は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] に分ける。本章の「ここでは扱わない」は本章の範囲を示すものであり、後続ノートで扱うことと矛盾しない。
