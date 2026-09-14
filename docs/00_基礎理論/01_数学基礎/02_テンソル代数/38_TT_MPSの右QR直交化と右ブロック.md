---
title: TT・MPSの右QR直交化と右ブロック
aliases:
  - MPSの右直交化
  - 右QRで転置する理由
  - 右ブロックのGram行列
tags:
  - TT
  - MPS
  - QR
  - orthogonality
---

# TT・MPSの右QR直交化と右ブロック

## サマリー

通常のQRは列を正規直交化する。TTコアを標準順 $(\alpha_{k-1},i_k,\alpha_k)$ で右unfoldingすると、右ブロック状態の係数は行に入る。そのため、転置してQRした後、$Q^T$ をコアへ戻し、三角因子 $T^T$ を左隣へ吸収する。

[[36_TT_MPSのGauge自由度と左QR直交化]] と [[37_TT_MPSの左ブロックと直交性の導出]] の右側に対応する章である。QRの三角因子 $T_k$ と右ブロック $\mathcal R_k$ を区別する。

## 1. なぜ右側は行直交条件なのか

3階TTの第3コアから境界軸だけを外す。

$$
A_3(\alpha_2,i_3)=G_3(\alpha_2,i_3,1),\qquad
A_3\in\mathbb R^{r_2\times n_3}.
$$

状態ラベル $\alpha_2$ を固定すると、第 $\alpha_2$ 行の全 $n_3$ 要素が右状態の係数になる。

$$
|R_{\alpha_2}\rangle=\sum_{i_3=1}^{n_3}A_3(\alpha_2,i_3)|i_3\rangle.
$$


元資料 `TT_MPS の学習を続けたいです。__前回までに、3階 TT-SVD の exact recons (1).md`
の1666–1679行、3161–3174行では右状態を行に並べている。
この図を係数の行列として書き直すと

$$
A_3=
\begin{pmatrix}
G_3(1,1,1)&G_3(1,2,1)&\cdots&G_3(1,n_3,1)\\
\vdots&\vdots&\ddots&\vdots\\
G_3(r_2,1,1)&G_3(r_2,2,1)&\cdots&G_3(r_2,n_3,1)
\end{pmatrix}.
$$

対応する右状態の係数columnを

$$
v_{\alpha_2}=
\begin{pmatrix}
G_3(\alpha_2,1,1)\\
G_3(\alpha_2,2,1)\\
\vdots\\
G_3(\alpha_2,n_3,1)
\end{pmatrix}
$$

と置くと、行配置と列配置の関係を省略せずに示せる。

$$
A_3=\begin{pmatrix}v_1^T\\v_2^T\\\vdots\\v_{r_2}^T\end{pmatrix},
\qquad
A_3^T=(v_1\ v_2\ \cdots\ v_{r_2}).
$$

元資料2174–2188行、2639–2653行、3201–3213行の右状態を列に並べた模式図は、後者である。
4476–4488行の $r_2=3,n_3=5$ の図は「5成分の右状態を3本、行へ並べた $3\times5$ 行列」を表す。
転置後は「同じ5成分を3本、列へ並べた $5\times3$ 行列」になる。
この並べ替えは状態の中身を変えない。

この元資料の模式図を実際の15要素で書くと、$a_{\alpha j}=G_3(\alpha,j,1)$ として

$$
A_3=\begin{pmatrix}
a_{11}&a_{12}&a_{13}&a_{14}&a_{15}\\
a_{21}&a_{22}&a_{23}&a_{24}&a_{25}\\
a_{31}&a_{32}&a_{33}&a_{34}&a_{35}
\end{pmatrix},\qquad
A_3^T=\begin{pmatrix}
a_{11}&a_{21}&a_{31}\\
a_{12}&a_{22}&a_{32}\\
a_{13}&a_{23}&a_{33}\\
a_{14}&a_{24}&a_{34}\\
a_{15}&a_{25}&a_{35}
\end{pmatrix}.
$$

例えば転置後の第4行・第2列は、元の第2行・第4列の $a_{24}$ である。
列を直交化した $Q_3$ の第2列は、$i_3=1,\ldots,5$ の5成分を持つ新しい右状態を表す。

複素係数では $v_\alpha^T$ とbraの係数 $v_\alpha^\dagger$ は異なるので、
元資料の行をbraとして描いた模式図は実数の場合の説明として読む。
実数係数でbraとketを展開すると

$$
\begin{aligned}
\langle R_{\beta_2}|R_{\alpha_2}\rangle
&=\sum_{j_3,i_3}A_3(\beta_2,j_3)A_3(\alpha_2,i_3)\langle j_3|i_3\rangle\\
&=\sum_{j_3,i_3}A_3(\beta_2,j_3)A_3(\alpha_2,i_3)\delta_{j_3,i_3}\\
&=\sum_{i_3}A_3(\beta_2,i_3)A_3(\alpha_2,i_3)
=(A_3A_3^T)_{\beta_2,\alpha_2}.
\end{aligned}
$$

従って、右状態を正規直交化する条件は

$$
\boxed{A_3A_3^T=I_{r_2}}.
$$

積の $(\beta_2,\alpha_2)$ を残し、$i_3$ を和で消すことが条件の由来である。$A_3^TA_3$ は $i_3,j_3$ を残し、ボンド添字を消す別の行列になる。

$r_2$ 本の正規直交な行を $n_3$ 次元に置くには $r_2\le n_3$ が必要である。なお、この条件は正規直交な右基底の条件であり、全状態 $X$ の規格化条件ではない。

## 2. 転置QRから $T_3^TQ_3^T$ へ

転置する理由は、右状態を物理的に左状態へ変えるためではない。
元の $A_3$ では「一つの右状態の係数」が1行に入っており、通常のQRが直交化する対象は列である。
そこで同じ係数を一時的に列へ置く。転置後も $i_3$ はサイト3の物理成分、$\alpha_2$ は右状態のラベルのままである。
サイト順を反転したり、切断の左右を交換したりしているわけではない。

QRの $Q_3$ の列は新しい正規直交な右状態を表す。
$T_3$ は、古い右状態を新しい状態の線形結合として表す係数と、その大きさを持つ。
従って、直交化後に $Q_3^T$ だけを残して $T_3^T$ を捨てると、一般には元の右状態も全テンソルも変わる。
次節で $T_3^T$ を左隣へ渡すのは、捨てた誤差を処理するためではなく、基底変更に伴う係数を保存するためである。

$n_3\ge r_2$・行フルランクの場合、転置した入力のreduced QRは

$$
A_3^T=Q_3T_3,\qquad Q_3\in\mathbb R^{n_3\times r_2},\qquad
T_3\in\mathbb R^{r_2\times r_2},\qquad Q_3^TQ_3=I_{r_2}.
$$

添字を固定して書けば

$$
A_3^T(i_3,\alpha_2)
=\sum_{\beta_2}Q_3(i_3,\beta_2)T_3(\beta_2,\alpha_2).
$$

入力の行 $i_3$ は $Q_3$ に、元の列 $\alpha_2$ は $T_3$ に残り、新しい基底が $\beta_2$ になる。元の向きへ転置すると、積の順序が反転する。

$$
\begin{aligned}
A_3&=(A_3^T)^T=(Q_3T_3)^T=T_3^TQ_3^T,\\
A_3(\alpha_2,i_3)
&=\sum_{\beta_2}T_3^T(\alpha_2,\beta_2)Q_3^T(\beta_2,i_3).
\end{aligned}
$$

新しい右直交コアは

$$
A_3^{[R]}=Q_3^T,\qquad
G_3^{[R]}(\beta_2,i_3,1)=Q_3(i_3,\beta_2).
$$

右Gramは

$$
A_3^{[R]}(A_3^{[R]})^T
=Q_3^T(Q_3^T)^T
=Q_3^TQ_3
=I_{r_2}.
$$

最後の積は $Q_3Q_3^T$ ではない。転置しても行列積のshapeと添字の役割を追う。

## 3. なぜ $T_3^T$ は左隣の右ボンドへ吸収するのか

第2コアのphysical sliceを $H_{i_2}=G_2(:,i_2,:)$ とする。元の局所積は

$$
H_{i_2}A_3
=H_{i_2}(T_3^TQ_3^T)
=(H_{i_2}T_3^T)Q_3^T.
$$

従って

$$
G_2^{\mathrm{right}}(\alpha_1,i_2,\beta_2)
=\sum_{\alpha_2}G_2(\alpha_1,i_2,\alpha_2)T_3^T(\alpha_2,\beta_2).
$$

消すのは古い $\alpha_2$、残すのは新しい $\beta_2$ である。成分を戻した局所不変性は

$$
\begin{aligned}
\sum_{\beta_2}G_2^{\mathrm{right}}(\alpha_1,i_2,\beta_2)G_3^{[R]}(\beta_2,i_3,1)
&=\sum_{\alpha_2,\beta_2}G_2(\alpha_1,i_2,\alpha_2)
T_3^T(\alpha_2,\beta_2)Q_3^T(\beta_2,i_3)\\
&=\sum_{\alpha_2}G_2(\alpha_1,i_2,\alpha_2)A_3(\alpha_2,i_3).
\end{aligned}
$$

左から $G_1$ を縮約すれば全テンソルも保存される。右直交化のsweepは右から左へ進む。右直交という名前は因子を送る向きが右向きという意味ではない。

## 4. 右QRを全要素で最後まで計算する

$n_1=n_2=r_1=r_2=2,n_3=3$ とし、左端は $A_1=I_2$、残りの全要素を

$$
A_3=\begin{pmatrix}2&0&0\\1&3&4\end{pmatrix},\qquad
H_1=\begin{pmatrix}1&2\\0&1\end{pmatrix},\qquad
H_2=\begin{pmatrix}2&0\\1&1\end{pmatrix}
$$

とする。

### 転置して列を正規直交化する途中式

$$
A_3^T=\begin{pmatrix}2&1\\0&3\\0&4\end{pmatrix},\qquad
a_1=\begin{pmatrix}2\\0\\0\end{pmatrix},\quad
a_2=\begin{pmatrix}1\\3\\4\end{pmatrix}.
$$

$$
t_{11}=\|a_1\|_2=2,\quad q_1=\begin{pmatrix}1\\0\\0\end{pmatrix},\quad
t_{12}=q_1^Ta_2=1,\quad
v_2=a_2-q_1t_{12}=\begin{pmatrix}0\\3\\4\end{pmatrix},
$$

$$
t_{22}=\sqrt{0^2+3^2+4^2}=5,\qquad
q_2=\begin{pmatrix}0\\3/5\\4/5\end{pmatrix}.
$$

対角を正に取ると

$$
Q_3=\begin{pmatrix}1&0\\0&3/5\\0&4/5\end{pmatrix},\qquad
T_3=\begin{pmatrix}2&1\\0&5\end{pmatrix},\qquad
Q_3T_3=\begin{pmatrix}2&1\\0&3\\0&4\end{pmatrix}=A_3^T.
$$

転置して元の順へ戻す積も全要素で

$$
T_3^TQ_3^T
=\begin{pmatrix}2&0\\1&5\end{pmatrix}
\begin{pmatrix}1&0&0\\0&3/5&4/5\end{pmatrix}
=\begin{pmatrix}2&0&0\\1&3&4\end{pmatrix}=A_3.
$$

### 新コアへの軸の戻し方

$$
A_3^{[R]}=\begin{pmatrix}1&0&0\\0&3/5&4/5\end{pmatrix},\qquad
G_3^{[R]}(:,:,1)=A_3^{[R]}.
$$

全6要素の対応は

$$
\begin{aligned}
G_3^{[R]}(1,1,1)&=Q_3(1,1)=1,&G_3^{[R]}(1,2,1)&=Q_3(2,1)=0,&G_3^{[R]}(1,3,1)&=Q_3(3,1)=0,\\
G_3^{[R]}(2,1,1)&=Q_3(1,2)=0,&G_3^{[R]}(2,2,1)&=Q_3(2,2)=3/5,&G_3^{[R]}(2,3,1)&=Q_3(3,2)=4/5.
\end{aligned}
$$

`Q3.T.unsqueeze(-1)` は $(3,2)\to(2,3)\to(2,3,1)$ であり、`Q3.unsqueeze(-1)` の $(3,2,1)$ とは違う。`unsqueeze` はサイズ1の軸を追加するだけで軸交換はしない。特に $n_3=r_2$ では間違えてもshapeが同じになるため、添字や値で確認する必要がある。

### 三角因子の吸収を各要素まで展開

一般の2行2列sliceなら

$$
\begin{pmatrix}h_{11}&h_{12}\\h_{21}&h_{22}\end{pmatrix}T_3^T
=\begin{pmatrix}
2h_{11}+h_{12}&5h_{12}\\
2h_{21}+h_{22}&5h_{22}
\end{pmatrix}.
$$

従って中央コアの全sliceは

$$
H_1^{\mathrm{right}}
=\begin{pmatrix}1\cdot2+2\cdot1&1\cdot0+2\cdot5\\0\cdot2+1\cdot1&0\cdot0+1\cdot5\end{pmatrix}
=\begin{pmatrix}4&10\\1&5\end{pmatrix},
$$

$$
H_2^{\mathrm{right}}
=\begin{pmatrix}2\cdot2+0\cdot1&2\cdot0+0\cdot5\\1\cdot2+1\cdot1&1\cdot0+1\cdot5\end{pmatrix}
=\begin{pmatrix}4&0\\3&5\end{pmatrix}.
$$

### 局所縮約と全12要素の一致

$$
H_1A_3=\begin{pmatrix}4&6&8\\1&3&4\end{pmatrix}
=H_1^{\mathrm{right}}A_3^{[R]},\qquad
H_2A_3=\begin{pmatrix}4&0&0\\3&3&4\end{pmatrix}
=H_2^{\mathrm{right}}A_3^{[R]}.
$$

例えば新しい第1sliceの $(1,2)$ 成分は $4\cdot0+10\cdot(3/5)=6$ で、元の $1\cdot0+2\cdot3=6$ と一致する。$A_1=I_2$ なので、行 $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ の全テンソルは

$$
\begin{aligned}
X^{\langle2\rangle}
&=\begin{pmatrix}1&2\\2&0\\0&1\\1&1\end{pmatrix}
\begin{pmatrix}2&0&0\\1&3&4\end{pmatrix}\\
&=\begin{pmatrix}4&10\\4&0\\1&5\\3&5\end{pmatrix}
\begin{pmatrix}1&0&0\\0&3/5&4/5\end{pmatrix}\\
&=\begin{pmatrix}4&6&8\\4&0&0\\1&3&4\\3&3&4\end{pmatrix}.
\end{aligned}
$$

更新前後の全要素を比較するので、直交性だけを満たして全体を変えてしまうミスも検出できる。

## 5. 右部分収縮とGram・射影

第2cutの右側は第3コアだけなので、右部分収縮は

$$
\boxed{\mathcal R_2(\beta_2,i_3)=G_3^{[R]}(\beta_2,i_3,1)=A_3^{[R]}(\beta_2,i_3)}.
$$

添付の右ブロック $R_2$ と同じものであり、$T_2$ というQR三角因子とは無関係である。3階TTでは新しい縮約やQRをする必要はなく、`G3_right.squeeze(-1)` で境界軸を外すだけでよい。同じ三角因子をもう一度吸収してはいけない。

前節の例で、直交化前後のGramは

$$
A_3A_3^T=\begin{pmatrix}4&2\\2&1+9+16\end{pmatrix}
=\begin{pmatrix}4&2\\2&26\end{pmatrix},
$$

$$
\mathcal R_2\mathcal R_2^T
=\begin{pmatrix}1^2+0^2+0^2&1\cdot0+0\cdot(3/5)+0\cdot(4/5)\\
0\cdot1+(3/5)\cdot0+(4/5)\cdot0&0^2+9/25+16/25\end{pmatrix}
=I_2.
$$

行に対応する正規直交状態は

$$
|R_1\rangle=|1\rangle,\qquad |R_2\rangle=\tfrac35|2\rangle+\tfrac45|3\rangle.
$$

反対順の積は物理空間で

$$
P_R=\mathcal R_2^T\mathcal R_2
=\begin{pmatrix}1&0&0\\0&9/25&12/25\\0&12/25&16/25\end{pmatrix}\ne I_3,
$$

$$
P_R^2=\mathcal R_2^T(\mathcal R_2\mathcal R_2^T)\mathcal R_2=P_R
$$

となる。これは右状態の張る部分空間への射影である。行が正規直交していない一般の元コアについてまで $A_3^TA_3$ を直交射影と呼んではいけない。

## 6. 左右の比較方法をそろえる

標準順では、左ブロック $\mathcal L_k$ は「物理配置が行・状態ラベルが列」、右ブロック $\mathcal R_k$ は「状態ラベルが行・物理配置が列」である。

$$
\mathcal L_k^T\mathcal L_k=I_{r_k},\qquad
\mathcal R_k\mathcal R_k^T=I_{r_k}.
$$

両方を列状態の形式で比べたいなら、別の行列 $C_{R,k}:=\mathcal R_k^T$ を定義する。

$$
C_{R,k}^TC_{R,k}=(\mathcal R_k^T)^T\mathcal R_k^T
=\mathcal R_k\mathcal R_k^T=I_{r_k}.
$$

ここでは $\mathcal R_k$ の定義を変えず、比較用の表現を転置しただけである。同じ種類の「ブロック状態の正規直交性」を表すが、左と右は異なる物理空間の基底である。同じコアの左直交条件と右直交条件が同一という意味でも、左直交なら自動的に右直交になるという意味でもない。

## 7. 内部コアと一般の右ブロック

第 $k$ コアの右unfoldingは

$$
B_k(\alpha_{k-1},(i_k,\alpha_k))=G_k(\alpha_{k-1},i_k,\alpha_k),\qquad
B_k\in\mathbb R^{r_{k-1}\times(n_kr_k)}.
$$

1始まりの列位置は $c=(i_k-1)r_k+\alpha_k$ である。右直交条件の成分は

$$
\sum_{i_k,\alpha_k}G_k^{[R]}(a,i_k,\alpha_k)G_k^{[R]}(b,i_k,\alpha_k)
=\delta_{a,b}.
$$

$B_k^T=Q_kT_k$ とQRし、$Q_k^T$ を $(q,n_k,r_k)$ にreshapeする。ただし $q=\min(n_kr_k,r_{k-1})$。元の左ボンドサイズを維持した右直交化には $r_{k-1}\le n_kr_k$ が必要であり、満たさなければ左ボンドを $q$ へ変えて因子を吸収する。

第 $k-1$ コアへの吸収は

$$
\widetilde G_{k-1}(\alpha_{k-2},i_{k-1},\beta_{k-1})
=\sum_{\alpha_{k-1}}G_{k-1}(\alpha_{k-2},i_{k-1},\alpha_{k-1})T_k^T(\alpha_{k-1},\beta_{k-1}).
$$

第 $k$ cutより右のブロックを

$$
\mathcal R_{k-1}(a,i_k,\ldots,i_d)
=\sum_b G_k^{[R]}(a,i_k,b)\mathcal R_k(b,i_{k+1},\ldots,i_d)
$$

と定義する。右ブロックの直交性を仮定すると、帰納の途中式は

$$
\begin{aligned}
(\mathcal R_{k-1}\mathcal R_{k-1}^T)_{a,c}
&=\sum_{i_k,b,e}G_k^{[R]}(a,i_k,b)G_k^{[R]}(c,i_k,e)
\left[\sum_{i_{k+1},\ldots,i_d}\mathcal R_k(b,i_{k+1},\ldots,i_d)\mathcal R_k(e,i_{k+1},\ldots,i_d)\right]\\
&=\sum_{i_k,b,e}G_k^{[R]}(a,i_k,b)G_k^{[R]}(c,i_k,e)\delta_{b,e}\\
&=\sum_{i_k,b}G_k^{[R]}(a,i_k,b)G_k^{[R]}(c,i_k,b)
=\delta_{a,c}.
\end{aligned}
$$

右端 $\mathcal R_d=(1)$ から繰り返せば、右直交化した範囲の全ブロックで行直交性が成立する。

## 8. 複素MPSでの対応と検証境界

複素係数では、QR入力を $A_3^\dagger$ にし、$A_3=T_3^\dagger Q_3^\dagger$、新行列を $Q_3^\dagger$ とする。PyTorchでは2階行列に `.mH` を使える。braの係数は共役なので

$$
\langle R_\beta|R_\alpha\rangle
=\sum_i\overline{\mathcal R_2(\beta,i)}\mathcal R_2(\alpha,i)
=(\mathcal R_2\mathcal R_2^\dagger)_{\alpha,\beta}.
$$

行Gramが単位行列なら、この内積も $\delta_{\beta,\alpha}$ になる。実数の場合の添字式を共役なしで複素状態へ流用しない。

これは複素MPSの理論とPyTorch単体の操作の対応であり、既存のTT public APIへ複素dtype対応を追加したという意味ではない。現在のproject contractは [[27_TT_MPS基礎のPyTorch実装]] のfloat32/float64に従う。

添付にはfloat64で直交性・局所縮約・全再構成の誤差が小さかったという実行報告がある。本章の具体例はそれを転載せず、別の小行列を独立に計算したもの。数値確認ではGram誤差、局所不変性、全体不変性を別々に見る。QRの列符号は実装・環境で変わり得るため、手計算の $Q,T$ の要素一致だけを合否にしない。

右ブロックの定義までで右直交化のテーマは一区切りになる。次の [[39_TT_MPSの混合正準形への導入]] は同じTTへ左右の因子を集める導入であり、中心移動や局所最適化の実装は別テーマとする。

### 元資料の数値報告を、今回の計算結果と区別して回収する

添付 `TT_MPS の学習を続けたいです。__前回までに、3階 TT-SVD の exact recons (1).md`
の3792–3878行、6413–6438行にある行列と誤差の報告を残す。
これは過去の資料内の実行報告であり、本章の手計算例や今回のNotebook再実行結果ではない。

$$
A_3^{[R]}(A_3^{[R]})^T
\simeq
\begin{pmatrix}
1&3.47\times10^{-17}&-7.63\times10^{-17}\\
3.47\times10^{-17}&1&-2.78\times10^{-17}\\
-7.63\times10^{-17}&-2.78\times10^{-17}&1
\end{pmatrix}.
$$

元資料が報告した三つの誤差は

$$
\left\|A_3^{[R]}(A_3^{[R]})^T-I_3\right\|_F=4.04\times10^{-16},
$$

$$
\|B_{\mathrm{after}}-B_{\mathrm{before}}\|_F=2.44\times10^{-15},\qquad
\|X_{\mathrm{after}}-X_{\mathrm{before}}\|_F=6.21\times10^{-15}
$$

である。ここで $B$ は第2・第3コアの局所縮約を表す。
元資料と同じ意味を添字で書くと

$$
\begin{aligned}
B_{\mathrm{before}}(\alpha_1,i_2,i_3)
&=\sum_{\alpha_2}G_2(\alpha_1,i_2,\alpha_2)G_3(\alpha_2,i_3,1),\\
B_{\mathrm{after}}(\alpha_1,i_2,i_3)
&=\sum_{\beta_2}G_2^{\mathrm{right}}(\alpha_1,i_2,\beta_2)
G_3^{[R]}(\beta_2,i_3,1).
\end{aligned}
$$

Gramだけでなく、因子吸収後の局所縮約、さらに全状態まで確認していた点が元資料の説明である。
ただし、表示行列の対角の `1` や非対角の数値は丸められている。
表示値を厳密値として二乗和を取り直しても、元資料の未丸めGramに対する
$4.04\times10^{-16}$ を完全には復元できない。
本章では誤差値を再計算して確定したとは扱わず、資料の報告値として保持する。
元資料がここで「右向きのQR sweep」と呼んだ操作は、右直交化を作る右から左への因子吸収である。
現在の実装や全環境について、この過去の誤差値だけから成功を保証しない。

## 参考資料

- [PyTorch：torch.linalg.qr](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)：列直交性、出力shape、符号とrank欠損入力の注意。
- [Schollwöck：DMRG in the age of matrix product states](https://arxiv.org/abs/1008.3477)：right-canonical MPSと右ブロック状態。
