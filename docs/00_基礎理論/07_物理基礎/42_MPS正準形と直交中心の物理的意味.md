---
title: MPS正準形と直交中心の物理的意味
aliases:
  - MPSの直交中心とブロック状態
  - orthogonality centerの物理解釈
tags:
  - MPS
  - canonical
  - Schmidt分解
  - 量子多体系
---

# MPS正準形と直交中心の物理的意味

## この章の範囲

[[16_TT_MPSの物理解釈_縮約密度行列と局所写像]] で導入した物理添字、ボンド状態、Schmidt分解を前提に、直交中心を左右ブロックの正規直交基底と中心係数で読む。QRによる中心移動とSchmidt分解は区別する。行列化・QR・因子吸収・全要素の数学的導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/41_TT_MPSの直交中心の移動]] に置き、本章では物理状態としての意味を扱う。

左側が左直交、右側が右直交し、第2サイトの中心コアには直交性を要求しない**全体の配置**を混合正準形（mixed-canonical form）と呼ぶ。その配置で係数を保持する**コア**が直交中心（orthogonality center）である。この二つは同義語ではない。一般のサイト $c$ での定義は [[00_基礎理論/01_数学基礎/02_テンソル代数/39_TT_MPSの混合正準形への導入]] に置く。

## 1. 直交中心をブロック状態で読む

第2サイト中心で、左・右ブロックの正規直交状態を $|L_{\alpha_1}\rangle$、$|R_{\alpha_2}\rangle$ と置く。状態は

$$
|\psi\rangle
=\sum_{\alpha_1,i_2,\alpha_2}
C(\alpha_1,i_2,\alpha_2)
|L_{\alpha_1}\rangle\otimes|i_2\rangle\otimes|R_{\alpha_2}\rangle.
$$

この表示で第2コアを直交中心と呼ぶのは、左右のコアがそれぞれ正規直交なブロック基底を作り、全状態の係数が $C$ に集まっているためである。サイト番号が中央だからという意味ではなく、中心は端点へも動かせる。左右と局所基底の内積を省略せずに取ると、

$$
\begin{aligned}
\langle\psi|\psi\rangle
&=\sum_{\substack{\alpha_1',i_2',\alpha_2'\\\alpha_1,i_2,\alpha_2}}
\overline{C(\alpha_1',i_2',\alpha_2')}C(\alpha_1,i_2,\alpha_2)
\langle L_{\alpha_1'}|L_{\alpha_1}\rangle
\langle i_2'|i_2\rangle
\langle R_{\alpha_2'}|R_{\alpha_2}\rangle\\
&=\sum_{\substack{\alpha_1',i_2',\alpha_2'\\\alpha_1,i_2,\alpha_2}}
\overline{C(\alpha_1',i_2',\alpha_2')}C(\alpha_1,i_2,\alpha_2)
\delta_{\alpha_1'\alpha_1}\delta_{i_2'i_2}\delta_{\alpha_2'\alpha_2}\\
&=\sum_{\alpha_1,i_2,\alpha_2}|C(\alpha_1,i_2,\alpha_2)|^2.
\end{aligned}
$$

正規化した状態なら最後の和は1である。この局所化は左右ブロックの直交性を使った結果であり、任意の非正準コアだけについて同じ等式を主張するものではない。全テンソルの差分・内積までの導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/40_TT_MPSの中心ノルムと内積の導出]] に置く。

係数 $C$ の左展開を $C_L=Q_LT_L$ と分解し、左の二つの因子をまとめた新しいブロック状態を

$$
|L'_{\beta_2}\rangle
=\sum_{\alpha_1,i_2}
Q_L((\alpha_1,i_2),\beta_2)
|L_{\alpha_1}\rangle\otimes|i_2\rangle
$$

と定める。複素状態も含めるため、内積では転置でなく随伴を使う。消える添字を明示すると、

$$
\begin{aligned}
\langle L'_{\gamma_2}|L'_{\beta_2}\rangle
&=\sum_{\alpha_1,i_2}\sum_{\alpha_1',i_2'}
\overline{Q_L((\alpha_1',i_2'),\gamma_2)}
Q_L((\alpha_1,i_2),\beta_2)
\underbrace{\langle L_{\alpha_1'}|L_{\alpha_1}\rangle}_{\delta_{\alpha_1'\alpha_1}}
\underbrace{\langle i_2'|i_2\rangle}_{\delta_{i_2'i_2}}\\
&=\sum_{\alpha_1,i_2}
\overline{Q_L((\alpha_1,i_2),\gamma_2)}Q_L((\alpha_1,i_2),\beta_2)\\
&=(Q_L^\dagger Q_L)_{\gamma_2\beta_2}
=\delta_{\gamma_2\beta_2}.
\end{aligned}
$$

右への中心移動では、$|L'_{\beta_2}\rangle$ が左二サイトの新しい正規直交基底になる。$C_L=Q_LT_L$ を状態へ代入し、二つのボンド添字を残して和を組み替えると、

$$
\begin{aligned}
|\psi\rangle
&=\sum_{\alpha_1,i_2,\alpha_2,\beta_2}
Q_L((\alpha_1,i_2),\beta_2)T_L(\beta_2,\alpha_2)
|L_{\alpha_1}\rangle\otimes|i_2\rangle\otimes|R_{\alpha_2}\rangle\\
&=\sum_{\beta_2,\alpha_2}T_L(\beta_2,\alpha_2)
|L'_{\beta_2}\rangle\otimes|R_{\alpha_2}\rangle.
\end{aligned}
$$

$T_L$ を右隣へ吸収した係数を $C_3(\beta_2,i_3)$ とすれば、

$$
\begin{aligned}
|\psi\rangle
&=\sum_{\beta_2,\alpha_2,i_3}
T_L(\beta_2,\alpha_2)G_3^{[R]}(\alpha_2,i_3,1)
|L'_{\beta_2}\rangle\otimes|i_3\rangle\\
&=\sum_{\beta_2,i_3}C_3(\beta_2,i_3)
|L'_{\beta_2}\rangle\otimes|i_3\rangle.
\end{aligned}
$$

これは物理状態 $|\psi\rangle$ を変えず、左側の座標基底と係数の置き場所を変えた表現である。

## 2. 左への中心移動では右ブロック基底を更新する

逆向きに第2サイトから第1サイトへ中心を動かすときは、同じ $C$ の行を $\alpha_1$、列を $(i_2,\alpha_2)$ として並べる。右側の局所添字と古い右ボンドをまとめ、新しい右ブロックの基底ラベル $\beta_1$ を残す。

$$
C_R(\alpha_1,(i_2,\alpha_2))=C(\alpha_1,i_2,\alpha_2),
\qquad C_R^{\mathsf T}=Q_RT_R,
\qquad C_R=T_R^{\mathsf T}Q_R^{\mathsf T}.
$$

$$
C_R\in\mathbb C^{r_1\times(n_2r_2)},\qquad
Q_R\in\mathbb C^{(n_2r_2)\times q_1},\qquad
T_R\in\mathbb C^{q_1\times r_1},\qquad
q_1=\min(n_2r_2,r_1).
$$

転置 $\mathsf T$ は行列の添字順を交換する操作である。複素係数でもQRの直交性は $Q_R^\dagger Q_R=I$ と随伴で確認する。$Q_R^{\mathsf T}$ の各行を係数として、

$$
|R'_{\beta_1}\rangle
=\sum_{i_2,\alpha_2}Q_R^{\mathsf T}(\beta_1,(i_2,\alpha_2))
|i_2\rangle\otimes|R_{\alpha_2}\rangle
$$

と置く。ここで $i_2$ と $\alpha_2$ は和で消え、$\beta_1$ が新しい右ブロック状態の番号になる。内積の全添字を展開すると、

$$
\begin{aligned}
\langle R'_{\gamma_1}|R'_{\beta_1}\rangle
&=\sum_{i_2',\alpha_2'}\sum_{i_2,\alpha_2}
\overline{Q_R^{\mathsf T}(\gamma_1,(i_2',\alpha_2'))}
Q_R^{\mathsf T}(\beta_1,(i_2,\alpha_2))
\underbrace{\langle i_2'|i_2\rangle}_{\delta_{i_2'i_2}}
\underbrace{\langle R_{\alpha_2'}|R_{\alpha_2}\rangle}_{\delta_{\alpha_2'\alpha_2}}\\
&=\sum_{i_2,\alpha_2}
\overline{Q_R((i_2,\alpha_2),\gamma_1)}Q_R((i_2,\alpha_2),\beta_1)\\
&=(Q_R^\dagger Q_R)_{\gamma_1\beta_1}
=\delta_{\gamma_1\beta_1}.
\end{aligned}
$$

係数行列の積を元の状態へ戻すと、各成分について

$$
C(\alpha_1,i_2,\alpha_2)
=\sum_{\beta_1}T_R^{\mathsf T}(\alpha_1,\beta_1)
Q_R^{\mathsf T}(\beta_1,(i_2,\alpha_2))
$$

である。これを代入して $i_2,\alpha_2$ の和を $|R'_{\beta_1}\rangle$ にまとめれば、

$$
\begin{aligned}
|\psi\rangle
&=\sum_{\alpha_1,\beta_1,i_2,\alpha_2}
T_R^{\mathsf T}(\alpha_1,\beta_1)Q_R^{\mathsf T}(\beta_1,(i_2,\alpha_2))
|L_{\alpha_1}\rangle\otimes|i_2\rangle\otimes|R_{\alpha_2}\rangle\\
&=\sum_{\alpha_1,\beta_1}T_R^{\mathsf T}(\alpha_1,\beta_1)
|L_{\alpha_1}\rangle\otimes|R'_{\beta_1}\rangle.
\end{aligned}
$$

$|L_{\alpha_1}\rangle=\sum_{i_1}G_1^{[L]}(1,i_1,\alpha_1)|i_1\rangle$ と展開し、$T_R^{\mathsf T}$ を第1コアへ吸収した係数を

$$
C_1(i_1,\beta_1)
=\sum_{\alpha_1}G_1^{[L]}(1,i_1,\alpha_1)T_R^{\mathsf T}(\alpha_1,\beta_1)
$$

と定めれば、$|\psi\rangle=\sum_{i_1,\beta_1}C_1(i_1,\beta_1)|i_1\rangle\otimes|R'_{\beta_1}\rangle$ となる。第1コアは新しい中心なので、それ自体に左直交性は要求しない。右向きと左向きで更新するブロックは逆だが、どちらも状態を変えない。コアのshapeと全要素の数値計算は [[00_基礎理論/01_数学基礎/02_テンソル代数/41_TT_MPSの直交中心の移動]] に置く。

## 3. QRによる正準化とSchmidt分解を区別する

一般には $C_3$ は対角行列ではない。QRで正準化しただけではSchmidt係数を得たとは言えず、そのcutのSchmidt分解には係数行列をさらにSVDで対角化する必要がある。正規化した状態なら、左ブロックの縮約密度行列はこの基底で $C_3C_3^\dagger$ と表せる。

この違いを、別の規格化された2状態ずつのcutで全要素確認する。元の係数行列を

$$
A=\frac1{\sqrt6}\begin{pmatrix}1&2\\1&0\end{pmatrix},
\qquad \|A\|_F^2=\frac{1+4+1+0}{6}=1
$$

とする。そのQR分解の一つは

$$
A=
\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}}_{Q}
\underbrace{\frac1{\sqrt3}\begin{pmatrix}1&1\\0&1\end{pmatrix}}_{T},
$$

であり、積の四成分は

$$
QT=\frac1{\sqrt6}
\begin{pmatrix}1\cdot1+1\cdot0&1\cdot1+1\cdot1\\
1\cdot1+(-1)\cdot0&1\cdot1+(-1)\cdot1\end{pmatrix}
=\frac1{\sqrt6}\begin{pmatrix}1&2\\1&0\end{pmatrix}.
$$

$Q$ が作る左基底は正規直交しているが、その基底での密度行列は

$$
TT^\dagger
=\frac13
\begin{pmatrix}1^2+1^2&1\cdot0+1\cdot1\\0\cdot1+1\cdot1&0^2+1^2\end{pmatrix}
=\frac13\begin{pmatrix}2&1\\1&1\end{pmatrix}
$$

であり、非対角成分が残る。したがって $T$ の個々の要素はSchmidt係数ではない。さらに対角化すると、固有値は $\det(TT^\dagger-\lambda I_2)=\lambda^2-\lambda+1/9=0$ から $(3\pm\sqrt5)/6$ となり、その平方根がSchmidt係数である。QRによる中心移動とSchmidt分解を同一視しない。

数学上の中心移動と、物理上の「状態不変・ブロック基底の更新」は相互に参照できるが、QR自体はエンタングルメント量を変える操作でも、Schmidt係数を打ち切る操作でもない。
