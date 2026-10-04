---
title: TT_MPSの加減算とrank増加
tags:
  - TT
  - MPS
  - tensor-train
  - addition
  - subtraction
  - block-diagonal
  - TT-rank
---

# TT/MPSの加減算とrank増加

このノートでは、2本のTTの和・差をコアのblock結合で正確に表現する方法と、構成直後のbond dimensionが増える理由を導く。

距離だけが必要なら [[52_TT_MPSのFrobeniusノルムと距離]] の内積公式を使い、差のTTを明示的に作らない方がよい。本ノートは、TT加算そのものが必要な場合の構成と、その後にTT-roundingが必要になる理由を扱う。

---

## 0. 前提と到達点

同じphysical shapeを持つ2本のTTを

$$
\mathcal A
=
\llbracket A^{(1)},\ldots,A^{(d)}\rrbracket,
\qquad
\mathcal B
=
\llbracket B^{(1)},\ldots,B^{(d)}\rrbracket
$$

とする。

$$
A^{(k)}
\in
\mathbb R^{r_{k-1}^A\times n_k\times r_k^A},
\qquad
B^{(k)}
\in
\mathbb R^{r_{k-1}^B\times n_k\times r_k^B}.
$$

和

$$
\mathcal C=\mathcal A+\mathcal B
$$

を表すコア列は、

1. 最初のコアを右bond方向へ連結する。
2. 中間コアをphysical indexごとのblock diagonalにする。
3. 最後のコアを左bond方向へ連結する。

ことで構成できる。

構成上の内部bond dimensionは

$$
\widetilde r_k=r_k^A+r_k^B
$$

だが、最小TT-rankについて言えるのは

$$
\boxed{
r_k(\mathcal A+\mathcal B)
\le
r_k^A+r_k^B
}
$$

であり、等号とは限らない。

---

## 1. まず $d=2$ で構成する

$d=2$ では

$$
\mathcal A_{i_1,i_2}
=
\sum_{\alpha=1}^{r_1^A}
A^{(1)}_{1,i_1,\alpha}
A^{(2)}_{\alpha,i_2,1},
$$

$$
\mathcal B_{i_1,i_2}
=
\sum_{\beta=1}^{r_1^B}
B^{(1)}_{1,i_1,\beta}
B^{(2)}_{\beta,i_2,1}.
$$

新しいbond indexを

$$
\gamma\in\{1,\ldots,r_1^A+r_1^B\}
$$

とする。前半をA、後半をBに割り当てる。

$$
C^{(1)}
=
\begin{bmatrix}
A^{(1)} & B^{(1)}
\end{bmatrix}
$$

はright bond方向の連結、

$$
C^{(2)}
=
\begin{bmatrix}
A^{(2)}\\
B^{(2)}
\end{bmatrix}
$$

はleft bond方向の連結を表す。

成分では

$$
C^{(1)}_{1,i_1,\gamma}
=
\begin{cases}
A^{(1)}_{1,i_1,\gamma},
&1\le\gamma\le r_1^A,\\
B^{(1)}_{1,i_1,\gamma-r_1^A},
&r_1^A<\gamma\le r_1^A+r_1^B,
\end{cases}
$$

$$
C^{(2)}_{\gamma,i_2,1}
=
\begin{cases}
A^{(2)}_{\gamma,i_2,1},
&1\le\gamma\le r_1^A,\\
B^{(2)}_{\gamma-r_1^A,i_2,1},
&r_1^A<\gamma\le r_1^A+r_1^B.
\end{cases}
$$

従って、

$$
\begin{aligned}
\mathcal C_{i_1,i_2}
&=
\sum_{\gamma=1}^{r_1^A+r_1^B}
C^{(1)}_{1,i_1,\gamma}
C^{(2)}_{\gamma,i_2,1}\\
&=
\sum_{\alpha=1}^{r_1^A}
A^{(1)}_{1,i_1,\alpha}
A^{(2)}_{\alpha,i_2,1}
+
\sum_{\beta=1}^{r_1^B}
B^{(1)}_{1,i_1,\beta}
B^{(2)}_{\beta,i_2,1}\\
&=
\mathcal A_{i_1,i_2}
+
\mathcal B_{i_1,i_2}.
\end{aligned}
$$

---

## 2. $d=3$ では中間コアをblock diagonalにする

3階TTの和では、最初のコアだけを連結した後、中間コアがA経路とB経路を混ぜないようにする必要がある。

最初のコアは

$$
C^{(1)}
=
\begin{bmatrix}
A^{(1)} & B^{(1)}
\end{bmatrix}.
$$

固定したphysical index $i_2$ ごとに、第2コアのmatrix sliceを

$$
\boxed{
C^{(2)}[:,i_2,:]
=
\begin{pmatrix}
A^{(2)}[:,i_2,:] & 0\\
0 & B^{(2)}[:,i_2,:]
\end{pmatrix}
}
$$

とする。最後のコアは

$$
C^{(3)}
=
\begin{bmatrix}
A^{(3)}\\
B^{(3)}
\end{bmatrix}.
$$

このblock diagonal構造により、bond経路は

$$
A\to A\to A
$$

または

$$
B\to B\to B
$$

のどちらかだけになる。従って最終的に $\mathcal A+\mathcal B$ が得られる。

---

## 3. block diagonalでないと何が壊れるか

中間コアのoff-diagonal blockを0にせず、例えば

$$
C^{(2)}[:,i_2,:]
=
\begin{pmatrix}
A^{(2)}[:,i_2,:] & X[:,i_2,:]\\
Y[:,i_2,:] & B^{(2)}[:,i_2,:]
\end{pmatrix}
$$

とすると、Aの第1コアからBの最終コアへ移る経路

$$
A^{(1)} X B^{(3)}
$$

や、Bの第1コアからAの最終コアへ移る経路

$$
B^{(1)} Y A^{(3)}
$$

が生じる。

その結果、

$$
\mathcal C
=
\mathcal A+
\mathcal B+
\text{不要な交差項}
$$

となる。中間コアの0 blockは、AとBの経路を最後まで分離するために必要である。

---

## 4. 一般の $d$ 次TT

一般の $d\ge2$ に対し、和TTのコアを次で定義する。

### 4.1 左端コア

$$
C^{(1)}
=
\begin{bmatrix}
A^{(1)} & B^{(1)}
\end{bmatrix}
\in
\mathbb R^{1\times n_1\times(r_1^A+r_1^B)}.
$$

### 4.2 中間コア

$k=2,\ldots,d-1$ では、各 $i_k$ に対し

$$
C^{(k)}[:,i_k,:]
=
\begin{pmatrix}
A^{(k)}[:,i_k,:] & 0\\
0 & B^{(k)}[:,i_k,:]
\end{pmatrix}.
$$

shapeは

$$
C^{(k)}
\in
\mathbb R^{(r_{k-1}^A+r_{k-1}^B)\times n_k\times(r_k^A+r_k^B)}.
$$

### 4.3 右端コア

$$
C^{(d)}
=
\begin{bmatrix}
A^{(d)}\\
B^{(d)}
\end{bmatrix}
\in
\mathbb R^{(r_{d-1}^A+r_{d-1}^B)\times n_d\times1}.
$$

この構成はexactであり、打ち切り誤差を導入しない。

---

## 5. $r^A=2$, $r^B=3$ のbond indexを全て対応させる

あるbondで

$$
r_k^A=2,
\qquad
r_k^B=3
$$

とする。新しいbond indexは

$$
\gamma_k\in\{1,2,3,4,5\}
$$

であり、対応は

$$
\begin{array}{c|c}
\gamma_k & \text{元の経路}\\
\hline
1 & \alpha_k=1\\
2 & \alpha_k=2\\
3 & \beta_k=1\\
4 & \beta_k=2\\
5 & \beta_k=3
\end{array}
$$

となる。

$d=2$ の1要素をこの5項で書けば、

$$
\begin{aligned}
\mathcal C_{i_1,i_2}
=
&C^{(1)}_{1,i_1,1}C^{(2)}_{1,i_2,1}
+C^{(1)}_{1,i_1,2}C^{(2)}_{2,i_2,1}\\
&+C^{(1)}_{1,i_1,3}C^{(2)}_{3,i_2,1}
+C^{(1)}_{1,i_1,4}C^{(2)}_{4,i_2,1}
+C^{(1)}_{1,i_1,5}C^{(2)}_{5,i_2,1}.
\end{aligned}
$$

定義を代入すると、

$$
\begin{aligned}
\mathcal C_{i_1,i_2}
=
&A^{(1)}_{1,i_1,1}A^{(2)}_{1,i_2,1}
+A^{(1)}_{1,i_1,2}A^{(2)}_{2,i_2,1}\\
&+B^{(1)}_{1,i_1,1}B^{(2)}_{1,i_2,1}
+B^{(1)}_{1,i_1,2}B^{(2)}_{2,i_2,1}
+B^{(1)}_{1,i_1,3}B^{(2)}_{3,i_2,1}\\
=
&\mathcal A_{i_1,i_2}
+\mathcal B_{i_1,i_2}.
\end{aligned}
$$

新しいbondはAの状態2個とBの状態3個を並べた直和空間になっている。

---

## 6. なぜrankは等号でなく上界か

block構成で得たTTのbond dimensionは確かに

$$
r_k^A+r_k^B
$$

である。しかしこれは「このサイズの表現を1つ構成できた」という意味であり、そのtensorの最小TT-rankが必ず同じとは限らない。

### 6.1 $\mathcal B=\mathcal A$ の場合

$$
\mathcal A+\mathcal B
=
2\mathcal A
$$

なので、Aの1つのコアを2倍すれば元と同じrankで表現できる。block構成のrank $2r_k^A$ は冗長である。

### 6.2 $\mathcal B=-\mathcal A$ の場合

$$
\mathcal A+\mathcal B=0.
$$

零tensorは物理shapeを保つrank-1零TTで表せる。構成直後のrankは大きくても、最小rankは1でよい。

従って、

$$
\boxed{
r_k(\mathcal A+\mathcal B)
\le
r_k^A+r_k^B
}
$$

である。加算後にTT-roundingを行うのは、このblock構成に含まれる線形従属や微小成分を除き、必要なrankへ戻すためである。

---

## 7. 差 $\mathcal A-\mathcal B$ の構成

差は和と同じblock構成で、B側のどれか1つのコアへ符号 $-1$ を吸収すればよい。例えば左端を

$$
C^{(1)}
=
\begin{bmatrix}
A^{(1)} & -B^{(1)}
\end{bmatrix}
$$

とし、中間と右端は和の場合と同じにする。B経路を通る積全体に $-1$ が一度だけ掛かるため、

$$
\mathcal C=\mathcal A-\mathcal B
$$

となる。

符号をB側の複数コアへ重ねて入れると、偶数個なら符号が戻るため注意する。1つのコアだけへ入れるのが明確である。

---

## 8. 距離計算ではblock差分を避ける

一様rank $r$ の2本から差TTを作ると、構成直後の内部rankは $2r$ である。中間コアの要素数は概算で

$$
n(2r)(2r)=4nr^2
$$

となり、元の1コア $nr^2$ の4倍である。

距離だけが目的なら、

$$
\|\mathcal A-\mathcal B\|_F^2
=
\langle\mathcal A,\mathcal A\rangle
+
\langle\mathcal B,\mathcal B\rangle
-
2\langle\mathcal A,\mathcal B\rangle
$$

を使えば、rankを増やさずに済む。

一方、加算結果を後続計算へ渡す必要がある場合は、block構成でexactな和を作り、必要に応じて [[48_TT-roundingの定義と正準化sweep|TT-rounding]] で再圧縮する。

---

## 9. まとめ

TT加算はbond空間の直和として構成できる。

$$
\boxed{
\begin{aligned}
C^{(1)}&=[A^{(1)}\ B^{(1)}],\\
C^{(k)}[:,i_k,:]
&=
\begin{pmatrix}
A^{(k)}[:,i_k,:] & 0\\
0 & B^{(k)}[:,i_k,:]
\end{pmatrix},\\
C^{(d)}&=
\begin{bmatrix}
A^{(d)}\\
B^{(d)}
\end{bmatrix}.
\end{aligned}
}
$$

- block diagonalはA/Bの交差経路を0にする。
- 構成はexactだが、bond dimensionは和になる。
- 最小TT-rankはその和以下であり、等号とは限らない。
- 加算結果を保持するならroundingで冗長rankを落とす。
- 距離だけなら差TTを作らず、3つの内積から求める。
