---
title: TT・MPSの混合正準形への導入
aliases:
  - mixed-canonical formへの導入
  - MPSの第2中心コア
tags:
  - TT
  - MPS
  - canonical
  - QR
---

# TT・MPSの混合正準形への導入

## この章の範囲

添付の最後に示された $G_2^{[C]}=R_1G_2R_3^T$ を、QR因子の記号を $T_1,T_3$ に分けて導出する。左右のQRを**同じ元TT**へ組み合わせるところまで扱う。

mixed-canonicalの学習やNotebook実装を完了したという記録ではない。中心の物理的意味の詳説、中心移動、中心のSVD・Schmidt係数、rounding、MPO、one-site/two-site最適化はここでは実装・展開しない。

前提は [[36_TT_MPSのGauge自由度と左QR直交化]] と [[38_TT_MPSの右QR直交化と右ブロック]] である。

## 1. 左右を別に試したことと、一つにまとめることの違い

同じ元TTの第1・第3コアを行列化してQRする。

$$
A_1=Q_1T_1,\qquad A_3^T=Q_3T_3,\qquad A_3=T_3^TQ_3^T.
$$

左右の直交コアは $G_1^{[L]}(1,i_1,\beta_1)=Q_1(i_1,\beta_1)$、
$G_3^{[R]}(\beta_2,i_3,1)=Q_3^T(\beta_2,i_3)$ とする。

元資料 `canonical_form.md` の141–332行で個別に展開したshapeと収縮も、
ここで行列の略記だけにせず残す。
本節では $r_1\le n_1,r_2\le n_3$ とし、QRの三角因子が正方になる場合を扱う。

$$
Q_1\in\mathbb R^{n_1\times r_1},\quad
T_1\in\mathbb R^{r_1\times r_1},\quad
Q_3\in\mathbb R^{n_3\times r_2},\quad
T_3\in\mathbb R^{r_2\times r_2}.
$$

$$
\begin{aligned}
G_1(1,i_1,\alpha_1)
&=\sum_{\beta_1=1}^{r_1}
G_1^{[L]}(1,i_1,\beta_1)T_1(\beta_1,\alpha_1),\\
G_3(\alpha_2,i_3,1)
&=\sum_{\beta_2=1}^{r_2}
T_3^T(\alpha_2,\beta_2)G_3^{[R]}(\beta_2,i_3,1),\\
(T_1G_2)(\beta_1,i_2,\alpha_2)
&=\sum_{\alpha_1=1}^{r_1}
T_1(\beta_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2),\\
(G_2T_3^T)(\alpha_1,i_2,\beta_2)
&=\sum_{\alpha_2=1}^{r_2}
G_2(\alpha_1,i_2,\alpha_2)T_3^T(\alpha_2,\beta_2).
\end{aligned}
$$

どちらか片側だけを吸収しても、中央コアのshapeは $(r_1,n_2,r_2)$ のままである。
これは物理軸へ行列を掛ける通常の積ではなく、指定したボンド軸の和である。
また、元資料426–505行の可逆なゲージ変換として読むには三角因子の可逆性が必要である。
rank欠損で三角因子が特異でもQRの等式と因子吸収による再構成は成り立つが、
逆行列を使う可逆ゲージの説明はそのまま適用できない。
寸法が変わるreduced QRとの区別は [[36_TT_MPSのGauge自由度と左QR直交化]] を参照する。

元の中央コアのsliceを $H_{i_2}=G_2(:,i_2,:)$ とすると、各physical indexを固定した途中式は

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=A_1(i_1,:)H_{i_2}A_3(:,i_3)\\
&=[Q_1(i_1,:)T_1]H_{i_2}[T_3^TQ_3^T(:,i_3)]\\
&=Q_1(i_1,:)[T_1H_{i_2}T_3^T]Q_3^T(:,i_3).
\end{aligned}
$$

従って中央コアは

$$
\boxed{G_2^{[C]}(:,i_2,:)=T_1G_2(:,i_2,:)T_3^T}.
$$

成分を全て区別して書けば

$$
G_2^{[C]}(\beta_1,i_2,\beta_2)
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
T_1(\beta_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2)T_3^T(\alpha_2,\beta_2).
$$


元資料333–384行の、二つのQRを元TTへ代入した直後の四つのボンド添字の和は

$$
\begin{aligned}
X(i_1,i_2,i_3)
&=\sum_{\alpha_1,\alpha_2}
G_1(1,i_1,\alpha_1)G_2(\alpha_1,i_2,\alpha_2)G_3(\alpha_2,i_3,1)\\
&=\sum_{\beta_1,\alpha_1,\alpha_2,\beta_2}
G_1^{[L]}(1,i_1,\beta_1)T_1(\beta_1,\alpha_1)
G_2(\alpha_1,i_2,\alpha_2)T_3^T(\alpha_2,\beta_2)
G_3^{[R]}(\beta_2,i_3,1)\\
&=\sum_{\beta_1,\beta_2}G_1^{[L]}(1,i_1,\beta_1)
\left[\sum_{\alpha_1,\alpha_2}T_1(\beta_1,\alpha_1)
G_2(\alpha_1,i_2,\alpha_2)T_3^T(\alpha_2,\beta_2)\right]
G_3^{[R]}(\beta_2,i_3,1).
\end{aligned}
$$

角括弧が定義した $G_2^{[C]}$ である。
この全和の表示でも、両端に残る添字と中央の中で消える添字を確認できる。
古い $\alpha_1,\alpha_2$ は消え、新しい $\beta_1,\beta_2$ が残る。結果は

$$
X(i_1,i_2,i_3)
=\sum_{\beta_1,\beta_2}G_1^{[L]}(1,i_1,\beta_1)
G_2^{[C]}(\beta_1,i_2,\beta_2)G_3^{[R]}(\beta_2,i_3,1).
$$

片方のQRで更新した $G_2$ と別の枝の直交コアを無造作に混ぜると、因子の吸収が欠落または重複する。同じ元 $G_2$ へ $T_1$ と $T_3^T$ を各1回吸収するか、同じTT上で順に更新する必要がある。

## 2. 三角因子・中央slice・再構成を全要素で計算する

本章独自の $2\times2\times2$ の例を使う。元の全コアは

$$
A_1=\begin{pmatrix}0&3\\2&2\end{pmatrix},\qquad
H_1=\begin{pmatrix}1&0\\2&1\end{pmatrix},\qquad
H_2=\begin{pmatrix}0&1\\1&2\end{pmatrix},\qquad
A_3=\begin{pmatrix}0&2\\3&1\end{pmatrix}.
$$

第1QRは、第1列 $(0,2)^T$ の正規化が $(0,1)^T$、第2列からその射影を引いた残りが $(3,0)^T$ なので

$$
Q_1=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
T_1=\begin{pmatrix}2&2\\0&3\end{pmatrix},\qquad
Q_1T_1=\begin{pmatrix}0&3\\2&2\end{pmatrix}.
$$

右側の入力は $A_3^T=\begin{pmatrix}0&3\\2&1\end{pmatrix}$。第1列の正規化は同じで、第2列の射影係数は1、残りは $(3,0)^T$ なので

$$
Q_3=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
T_3=\begin{pmatrix}2&1\\0&3\end{pmatrix},\qquad
T_3^TQ_3^T=\begin{pmatrix}2&0\\1&3\end{pmatrix}\begin{pmatrix}0&1\\1&0\end{pmatrix}
=\begin{pmatrix}0&2\\3&1\end{pmatrix}=A_3.
$$

中央の第1sliceでは

$$
T_1H_1
=\begin{pmatrix}2\cdot1+2\cdot2&2\cdot0+2\cdot1\\0\cdot1+3\cdot2&0\cdot0+3\cdot1\end{pmatrix}
=\begin{pmatrix}6&2\\6&3\end{pmatrix},
$$

$$
H_1^{[C]}=(T_1H_1)T_3^T
=\begin{pmatrix}6\cdot2+2\cdot1&6\cdot0+2\cdot3\\6\cdot2+3\cdot1&6\cdot0+3\cdot3\end{pmatrix}
=\begin{pmatrix}14&6\\15&9\end{pmatrix}.
$$

第2sliceでも

$$
T_1H_2=\begin{pmatrix}2\cdot0+2\cdot1&2\cdot1+2\cdot2\\0\cdot0+3\cdot1&0\cdot1+3\cdot2\end{pmatrix}
=\begin{pmatrix}2&6\\3&6\end{pmatrix},
$$

$$
H_2^{[C]}=(T_1H_2)T_3^T
=\begin{pmatrix}2\cdot2+6\cdot1&2\cdot0+6\cdot3\\3\cdot2+6\cdot1&3\cdot0+6\cdot3\end{pmatrix}
=\begin{pmatrix}10&18\\12&18\end{pmatrix}.
$$

中央コアの全8要素が2枚のsliceとして求まった。両端は正規直交だが、中央には左・右どちらの直交条件も課していない。「必ず非直交」という意味ではなく、たまたま条件を満たす場合もあり得る。

再構成の全要素は、元の積と新しい積で

各sliceの二つの積を途中行列まで残すと、元のコア側では

$$
\begin{aligned}
A_1H_1
&=\begin{pmatrix}0\cdot1+3\cdot2&0\cdot0+3\cdot1\\
2\cdot1+2\cdot2&2\cdot0+2\cdot1\end{pmatrix}
=\begin{pmatrix}6&3\\6&2\end{pmatrix},\\
(A_1H_1)A_3
&=\begin{pmatrix}6\cdot0+3\cdot3&6\cdot2+3\cdot1\\
6\cdot0+2\cdot3&6\cdot2+2\cdot1\end{pmatrix}
=\begin{pmatrix}9&15\\6&14\end{pmatrix},\\
A_1H_2
&=\begin{pmatrix}0\cdot0+3\cdot1&0\cdot1+3\cdot2\\
2\cdot0+2\cdot1&2\cdot1+2\cdot2\end{pmatrix}
=\begin{pmatrix}3&6\\2&6\end{pmatrix},\\
(A_1H_2)A_3
&=\begin{pmatrix}3\cdot0+6\cdot3&3\cdot2+6\cdot1\\
2\cdot0+6\cdot3&2\cdot2+6\cdot1\end{pmatrix}
=\begin{pmatrix}18&12\\18&10\end{pmatrix}.
\end{aligned}
$$

直交化後のコア側では

$$
\begin{aligned}
Q_1H_1^{[C]}
&=\begin{pmatrix}
0\cdot14+1\cdot15&0\cdot6+1\cdot9\\
1\cdot14+0\cdot15&1\cdot6+0\cdot9
\end{pmatrix}
=\begin{pmatrix}15&9\\14&6\end{pmatrix},\\
(Q_1H_1^{[C]})Q_3^T
&=\begin{pmatrix}15\cdot0+9\cdot1&15\cdot1+9\cdot0\\
14\cdot0+6\cdot1&14\cdot1+6\cdot0\end{pmatrix}
=\begin{pmatrix}9&15\\6&14\end{pmatrix},\\
Q_1H_2^{[C]}
&=\begin{pmatrix}
0\cdot10+1\cdot12&0\cdot18+1\cdot18\\
1\cdot10+0\cdot12&1\cdot18+0\cdot18
\end{pmatrix}
=\begin{pmatrix}12&18\\10&18\end{pmatrix},\\
(Q_1H_2^{[C]})Q_3^T
&=\begin{pmatrix}12\cdot0+18\cdot1&12\cdot1+18\cdot0\\
10\cdot0+18\cdot1&10\cdot1+18\cdot0\end{pmatrix}
=\begin{pmatrix}18&12\\18&10\end{pmatrix}.
\end{aligned}
$$

したがって、次の等式は三つの因子をまとめて記した結果だけでなく、両側の中間積から確認できる。

$$
A_1H_1A_3=\begin{pmatrix}9&15\\6&14\end{pmatrix}
=Q_1H_1^{[C]}Q_3^T,\qquad
A_1H_2A_3=\begin{pmatrix}18&12\\18&10\end{pmatrix}
=Q_1H_2^{[C]}Q_3^T.
$$

行を $(i_1,i_2)=(1,1),(1,2),(2,1),(2,2)$ とした第2cutは、両方で

$$
X^{\langle2\rangle}=\begin{pmatrix}9&15\\18&12\\6&14\\18&10\end{pmatrix}.
$$

例えば最初の $(i_1,i_2,i_3)=(1,1,1)$ の値は

$$
\begin{pmatrix}0&1\end{pmatrix}
\begin{pmatrix}14&6\\15&9\end{pmatrix}
\begin{pmatrix}0\\1\end{pmatrix}=9
$$

である。この例は全状態を正規化していないが、QR前後で値とノルムを保存している。

## 3. ブロック状態で書く導入式

中心を第2サイトに置くとき、左ブロックは**第1コアだけ**である。[[37_TT_MPSの左ブロックと直交性の導出]] の2コアブロック $\mathcal L_2$ を左環境として使うと、中心サイトを二重に数えてしまう。

$$
|L_{\beta_1}\rangle=\sum_{i_1}G_1^{[L]}(1,i_1,\beta_1)|i_1\rangle,\qquad
|R_{\beta_2}\rangle=\sum_{i_3}G_3^{[R]}(\beta_2,i_3,1)|i_3\rangle.
$$

これらと第2サイトの基底を用いると

$$
|X\rangle=\sum_{\beta_1,i_2,\beta_2}G_2^{[C]}(\beta_1,i_2,\beta_2)
|L_{\beta_1}\rangle\otimes|i_2\rangle\otimes|R_{\beta_2}\rangle.
$$

第2コアがorthogonality centerとなり、左・右の基底が正規直交になる配置をmixed-canonical formと呼ぶ。これが次の学習への接続点である。本章では中心移動、中心のSVD、中心ノルムの詳細な導出や最適化コードへは進まない。

## 参考資料

- [Schollwöck：DMRG in the age of matrix product states](https://arxiv.org/abs/1008.3477)：mixed-canonical MPSとorthogonality center。
