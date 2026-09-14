---
title: TT・MPSの物理解釈―縮約密度行列と局所写像
aliases:
  - TT-rankと縮約密度行列
  - MPSコアの物理解釈
  - Schmidt係数と密度行列
tags:
  - TT
  - MPS
  - Schmidt分解
  - 縮約密度行列
  - 量子多体系
---

# TT・MPSの物理解釈―縮約密度行列と局所写像

## サマリー

TTを数値線形代数として見ると、$r_k$ はcut unfoldingのrankである。MPSとして見ると、同じ量はその切断をまたぐSchmidt rankであり、縮約密度行列の非ゼロ固有値数でもある。

第1cutなら、

$$
\boxed{
r_1
=
\operatorname{rank}\left(X^{\langle1\rangle}\right)
=
\operatorname{rank}(\rho_1)
=
\text{非ゼロSchmidt係数の本数}
}
$$

と対応する。

またMPSコア

$$
A^{[k]}_{\alpha_{k-1},i_k,\alpha_k}
$$

は、物理的には「左から来た仮想相関チャネルと局所状態を受け取り、右へ新しい仮想チャネルを渡す局所写像」と読める。

---

## 1. 「サイト1の固有ベクトル」とは何か

資料中では、途中で「$r_1$ はサイト1の固有ベクトルを何本取った数なのか」という疑問が出た。

ここで「サイト1の固有ベクトル」という言い方は曖昧である。正確には、サイト1の縮約密度行列

$$
\boxed{
\rho_1
=
\operatorname{Tr}_{2\cdots d}
\left(|\psi\rangle\langle\psi|\right)
}
$$

の固有ベクトルを指す。

### partial traceを物理基底の成分まで展開する

右側の複合添字を $\mu=(i_2,\ldots,i_d)$ とし、係数行列を $C=X^{\langle1\rangle}$ と略記する。

$$
|\psi\rangle=\sum_{i_1,\mu}C_{i_1,\mu}|i_1\rangle\otimes|\mu\rangle.
$$

密度演算子には二つの係数が現れ、bra側は複素共役される。

$$
|\psi\rangle\langle\psi|
=\sum_{i_1,j_1}\sum_{\mu,\nu}
C_{i_1,\mu}C^*_{j_1,\nu}
|i_1\rangle\langle j_1|\otimes|\mu\rangle\langle\nu|.
$$

右側のpartial traceを定義から計算すると、

$$
\begin{aligned}
\operatorname{Tr}_{R}(|\mu\rangle\langle\nu|)
&=\sum_\eta\langle\eta|\mu\rangle\langle\nu|\eta\rangle\\
&=\sum_\eta\delta_{\eta,\mu}\delta_{\nu,\eta}\\
&=\delta_{\mu,\nu}.
\end{aligned}
$$

したがって、

$$
\begin{aligned}
\rho_1
&=\sum_{i_1,j_1}\sum_{\mu,\nu}
C_{i_1,\mu}C^*_{j_1,\nu}
\delta_{\mu,\nu}|i_1\rangle\langle j_1|\\
&=\sum_{i_1,j_1}
\left(\sum_\mu C_{i_1,\mu}C^*_{j_1,\mu}\right)
|i_1\rangle\langle j_1|,\\
(\rho_1)_{i_1,j_1}
&=\sum_\mu C_{i_1,\mu}C^*_{j_1,\mu}
=(CC^\dagger)_{i_1,j_1}.
\end{aligned}
$$

この計算が、第3節で用いる係数行列からの密度行列の式を与える。右基底を追跡しない「traceを取る」の一言だけではなく、二つの右添字の和をデルタで一つにしている。

サイト1の局所Hilbert空間の次元が $n_1$ なら、

$$
\rho_1\in\mathbb C^{n_1\times n_1}
$$

であるため、完全な固有基底としては $n_1$ 本の固有ベクトルを持つ。

しかしTT/MPS表現で状態に実際に寄与するのは、**非ゼロ固有値に対応する方向**である。

---

## 2. Schmidt分解から縮約密度行列へ

第1cut

$$
1\mid2\cdots d
$$

についてSchmidt分解を

$$
|\psi\rangle
=
\sum_{\alpha_1=1}^{r_1}
\lambda_{\alpha_1}
|L_{\alpha_1}\rangle
\otimes
|R_{\alpha_1}\rangle
$$

とする。

密度演算子は

$$
|\psi\rangle\langle\psi|
=
\sum_{\alpha_1,\beta_1}
\lambda_{\alpha_1}\lambda_{\beta_1}
|L_{\alpha_1}\rangle\langle L_{\beta_1}|
\otimes
|R_{\alpha_1}\rangle\langle R_{\beta_1}|.
$$

右側をpartial traceする。

$$
\begin{aligned}
\rho_1
&=
\operatorname{Tr}_{2\cdots d}
\left(|\psi\rangle\langle\psi|\right)\\
&=
\sum_{\alpha_1,\beta_1}
\lambda_{\alpha_1}\lambda_{\beta_1}
|L_{\alpha_1}\rangle\langle L_{\beta_1}|
\operatorname{Tr}
\left(
|R_{\alpha_1}\rangle\langle R_{\beta_1}|
\right).
\end{aligned}
$$

右Schmidtベクトルは直交規格化されているので、

$$
\operatorname{Tr}
\left(
|R_{\alpha_1}\rangle\langle R_{\beta_1}|
\right)
=
\langle R_{\beta_1}|R_{\alpha_1}\rangle
=
\delta_{\alpha_1\beta_1}.
$$

したがって、

$$
\boxed{
\rho_1
=
\sum_{\alpha_1}
\lambda_{\alpha_1}^2
|L_{\alpha_1}\rangle
\langle L_{\alpha_1}|
}
$$

となる。

つまり、

- $|L_{\alpha_1}\rangle$：$\rho_1$ の固有ベクトル
- $\lambda_{\alpha_1}^2$：$\rho_1$ の固有値

である。

---

## 3. SVDから同じ関係を直接見る

第1cutの係数行列を

$$
X^{\langle1\rangle}
=
U\Sigma V^\dagger
$$

とする。

サイト1の縮約密度行列は、係数行列から

$$
\rho_1
=
X^{\langle1\rangle}
\left(X^{\langle1\rangle}\right)^\dagger
$$

と書ける。

SVDを代入すると、

$$
\begin{aligned}
\rho_1
&=
U\Sigma V^\dagger
\left(U\Sigma V^\dagger\right)^\dagger\\
&=
U\Sigma V^\dagger
V\Sigma U^\dagger\\
&=
U\Sigma^2U^\dagger.
\end{aligned}
$$

したがって、

$$
\boxed{
U\text{ の列}
=
\rho_1\text{ の固有ベクトル}
}
$$

$$
\boxed{
\sigma_{\alpha_1}
=
\lambda_{\alpha_1}
}
$$

$$
\boxed{
\rho_1\text{ の固有値}
=
\sigma_{\alpha_1}^2
=
\lambda_{\alpha_1}^2
}
$$

である。

この意味で、SVDの左特異ベクトルは物理側ではSchmidtベクトルであり、縮約密度行列の固有ベクトルでもある。

固有ベクトルという結論も、列 $u_{\alpha_1}=Ue_{\alpha_1}$ を代入すると、

$$
\begin{aligned}
\rho_1u_{\alpha_1}
&=U\Sigma^2U^\dagger Ue_{\alpha_1}\\
&=U\Sigma^2e_{\alpha_1}\\
&=\sigma_{\alpha_1}^2Ue_{\alpha_1}\\
&=\sigma_{\alpha_1}^2u_{\alpha_1}
\end{aligned}
$$

と確認できる。rank-sized SVDなら $U$ の列は非ゼロ固有値の方向だけであり、零固有値を含めた完全な固有基底とは区別する。

また、

$$
\operatorname{tr}(\rho_1)
=\sum_{i_1,\mu}|C_{i_1,\mu}|^2
=\|C\|_F^2
=\sum_{\alpha_1}\sigma_{\alpha_1}^2
$$

である。規格化された状態ならtraceは1。規格化前の数値テンソルを量子状態として使う場合は、$CC^\dagger/\|C\|_F^2$ に規格化してから確率として読む。

### 非対角の密度行列を全要素から計算する

本節独自の、二つの非ゼロSchmidt係数を持つ規格化係数行列として

$$
C=\frac12\begin{pmatrix}1&1&0\\1&0&1\end{pmatrix}
$$

を使う。右側は次元3のまとめた空間とする。partial traceの全要素は

$$
\begin{aligned}
\rho_1
&=CC^T\\
&=\frac14\begin{pmatrix}
1^2+1^2+0^2&1\cdot1+1\cdot0+0\cdot1\\
1\cdot1+0\cdot1+1\cdot0&1^2+0^2+1^2
\end{pmatrix}\\
&=\frac14\begin{pmatrix}2&1\\1&2\end{pmatrix}.
\end{aligned}
$$

非対角要素も残る。固有値と固有ベクトルを最後まで求めると、

$$
\det(\rho_1-\lambda I_2)
=(1/2-\lambda)^2-\frac1{16}
=\left(\lambda-\frac34\right)\left(\lambda-\frac14\right)=0,
$$

$$
u_1=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix},
\qquad
u_2=\frac1{\sqrt2}\begin{pmatrix}1\\-1\end{pmatrix},
\qquad
\rho_1u_1=\frac34u_1,
\qquad
\rho_1u_2=\frac14u_2.
$$

したがって、Schmidt係数は $\sqrt3/2,1/2$、exact rankは2である。係数行列を最初から対角にして結論を読むのではなく、partial traceの成分和から固有値問題へ進んでいる。

---

## 4. 「全部で何本」と「状態に必要な本数」を分ける

$\rho_1$ が $n_1\times n_1$ なら、完全な固有基底としては $n_1$ 本の固有ベクトルがある。

しかし、例えば

$$
\lambda_1,\ldots,\lambda_{r_1}>0,
$$

$$
\lambda_{r_1+1}
=
\cdots
=
\lambda_{n_1}
=
0
$$

なら、状態に寄与するのは最初の $r_1$ 本だけである。

したがって、

$$
\boxed{
r_1
=
\operatorname{rank}(\rho_1)
=
\text{非ゼロ固有値を持つ固有ベクトルの本数}
}
$$

である。

右側のHilbert空間にも同数の直交Schmidt状態が必要なので、

$$
r_1\le n_1,
$$

$$
r_1\le n_2\cdots n_d,
$$

よって

$$
\boxed{
r_1
\le
\min\left(n_1,n_2\cdots n_d\right)
}
$$

となる。

---

## 5. 一般状態と積状態

例えば

$$
X\in\mathbb C^{2\times3\times4}
$$

を量子状態と見る。サイト1の空間は

$$
\mathcal H_1\simeq\mathbb C^2
$$

なので、

$$
\rho_1\in\mathbb C^{2\times2}.
$$

一般的に二つのSchmidt係数が非ゼロなら

$$
r_1=2
$$

で、

$$
|\psi\rangle
=
\lambda_1|L_1\rangle|R_1\rangle
+
\lambda_2|L_2\rangle|R_2\rangle.
$$

一方、積状態

$$
|\psi\rangle
=
|a\rangle_1\otimes|\phi\rangle_{23}
$$

では

$$
\rho_1=|a\rangle\langle a|
$$

でrank 1である。

完全な固有基底は2本あっても、固有値は例えば

$$
(1,0)
$$

であり、状態に必要なSchmidtチャネルは1本だけなので

$$
r_1=1.
$$

---

### 積状態の密度行列を要素まで展開する

上の積状態について、rankが1になる式も要素まで追っておく。
係数を $a_{i_1}$ と $\phi_\mu$ とすると

$$
C_{i_1,\mu}=a_{i_1}\phi_\mu,
$$

$$
\begin{aligned}
(\rho_1)_{i_1,j_1}
&=\sum_\mu C_{i_1,\mu}C^*_{j_1,\mu}\\
&=\sum_\mu a_{i_1}\phi_\mu a^*_{j_1}\phi_\mu^*\\
&=a_{i_1}a^*_{j_1}\sum_\mu|\phi_\mu|^2\\
&=\|\phi\|_2^2a_{i_1}a^*_{j_1}.
\end{aligned}
$$

$a,\phi$ がともに単位ベクトルなら $\rho_1=aa^\dagger$、
$\rho_1a=a(a^\dagger a)=a$ であり、$a$ に直交する全ての方向には $\rho_1z=0$ となる。
従って固有値は1が1本、残りが0でrankは1である。
非正規化の積状態では、非零固有値は $\|a\|_2^2\|\phi\|_2^2$ となる。

左と右に各2本の基底を使った、この節独自の非自明な積状態の例は

$$
a=\frac{1}{\sqrt5}\begin{pmatrix}1\\2\end{pmatrix},
\qquad
\phi=\frac{1}{\sqrt2}\begin{pmatrix}1\\-1\end{pmatrix},
\qquad
C=\frac{1}{\sqrt{10}}\begin{pmatrix}1&-1\\2&-2\end{pmatrix}.
$$

全要素の積から

$$
\begin{aligned}
CC^{\mathsf T}
&=\frac{1}{10}\begin{pmatrix}
1^2+(-1)^2&1\cdot2+(-1)\cdot(-2)\\
2\cdot1+(-2)\cdot(-1)&2^2+(-2)^2
\end{pmatrix}\\
&=\frac{1}{10}\begin{pmatrix}2&4\\4&8\end{pmatrix}
=\frac{1}{5}\begin{pmatrix}1&2\\2&4\end{pmatrix},\\
\det(\rho_1-\lambda I_2)
&=(1/5-\lambda)(4/5-\lambda)-(2/5)^2\\
&=\lambda^2-\lambda=\lambda(\lambda-1).
\end{aligned}
$$

見かけ上は非対角の密度行列でも、固有値は $(1,0)$ である。
右基底はサイト $2,3$ の複合基底のうち2本と考えてよく、サイト1とのcutが積状態であることだけを使った。

## 6. MPSコアの3本の添字を物理で読む

内部MPSコア

$$
A^{[k]}_{\alpha_{k-1},i_k,\alpha_k}
$$

の3本の添字は、物理的には

```text
左の仮想相関チャネル  ─  局所サイト k  ─  右の仮想相関チャネル
       α_{k-1}                 i_k                 α_k
```

と読む。

- $i_k$：サイト $k$ の実在する局所Hilbert空間のラベル
- $\alpha_{k-1}$：左cutをまたぐSchmidtチャネル
- $\alpha_k$：右cutをまたぐSchmidtチャネル

例えばspin-$\frac12$なら

$$
i_k\in\{\uparrow,\downarrow\},
\qquad
n_k=2.
$$

局所bosonなら

$$
i_k=0,1,\ldots,n_{\max},
\qquad
n_k=n_{\max}+1
$$

のように、そのサイトの局所状態をphysical indexが表す。

---

## 7. なぜ内部コアは3階なのか

内部サイト $k$ は二つのcutに挟まれている。

```text
1...k-1 | k...d       1...k | k+1...d
          ↑                      ↑
       α_{k-1}                 α_k
```

そのため1個のサイトテンソルは同時に、

1. 左から来る相関情報 $\alpha_{k-1}$
2. 自分自身の局所状態 $i_k$
3. 右へ渡す相関情報 $\alpha_k$

を持つ必要がある。

$$
\boxed{
\text{内部サイトには「左の相関」「局所状態」「右の相関」の3種類の自由度が集まる}
}
$$

これがMPSコアを3階テンソルとして書く物理的理由である。

---

## 8. physical indexを固定すると「行列の束」になる

$i_k$ を固定すると、

$$
A^{[k]}[i_k]
:=
A^{[k]}_{:,i_k,:}
\in
\mathbb C^{r_{k-1}\times r_k}
$$

という行列になる。

spin-$\frac12$なら各サイトに

```text
A^[k][↑]
A^[k][↓]
```

という2枚の行列があると見られる。

ある基底配置

$$
(i_1,\ldots,i_d)
$$

を固定すると、その波動関数振幅は

$$
\boxed{
\psi_{i_1\cdots i_d}
=
A^{[1]}[i_1]
A^{[2]}[i_2]
\cdots
A^{[d]}[i_d]
}
$$

で得られる。両端bond dimensionが1なので、最終結果はスカラーになる。

したがって各MPSコアは、

$$
\boxed{
\text{局所状態 }i_k\text{ ごとに選ばれる、左右bondを結ぶ行列の束}
}
$$

と読むこともできる。

---

## 9. コアを局所写像として読む

左・右の仮想空間を

$$
\mathcal V_{k-1},
\qquad
\mathcal V_k
$$

サイト $k$ の物理空間を

$$
\mathcal H_k
$$

とすると、コアは例えば

$$
A^{[k]}:
\mathcal V_{k-1}
\longrightarrow
\mathcal H_k\otimes\mathcal V_k
$$

という局所写像として読める。

向きを逆に見れば

$$
A^{[k]}:
\mathcal V_{k-1}\otimes\mathcal H_k
\longrightarrow
\mathcal V_k
$$

とも読める。

現段階ではcanonical formの方向付けまでは扱わず、本質を

```text
左の仮想状態
  +
局所物理状態
  ↓
右へ渡す仮想状態
```

という局所的な変換則として理解する。

---

## 10. 開放境界と両端bond dimension 1

左端には左隣がなく、右端には右隣がない。

$$
r_0=r_d=1.
$$

物理的には、鎖の外側にさらに仮想系を接続しないopen boundary conditionに対応する。

したがって

$$
A^{[1]}
\in
\mathbb C^{1\times n_1\times r_1},
$$

$$
A^{[d]}
\in
\mathbb C^{r_{d-1}\times n_d\times1}.
$$

---

## 11. 応用数学との対応を固定する

| 応用数学・TT | 物理・MPS |
| --- | --- |
| cut unfolding rank | Schmidt rank |
| TT-rank | bond dimension |
| SVDの左特異ベクトル | 左Schmidtベクトル / 縮約密度行列の固有ベクトル |
| singular value $\sigma_\alpha$ | Schmidt係数 $\lambda_\alpha$ |
| $\sigma_\alpha^2$ | 縮約密度行列の固有値 |
| bond index $\alpha_k$ | Schmidtチャネル番号 |
| mode index $i_k$ | 局所物理状態 |

この対応を固定すると、TT-SVDで行っている「数値線形代数上の基底選択」が、MPSでは「部分系間のSchmidt基底を選ぶ操作」と読める。

詳細なTT定義は [[30_TT_MPSの定義]]、cut rankは [[31_TT-rankとunfolding]]、逐次SVDは [[32_TT-SVD]] を参照する。
