---
title: TT-SVDの打ち切りと誤差
aliases:
  - truncated TT-SVD
  - TT圧縮
  - TT誤差評価
  - discarded singular values
tags:
  - TT
  - SVD
  - truncation
  - FrobeniusNorm
  - error
---

# TT-SVDの打ち切りと誤差

## サマリー

厳密TT-SVDでは各段階の非ゼロ特異値を全て残す。圧縮TT-SVDでは、厳密rank $r_k$ より小さい

$$
\widetilde r_k<r_k
$$

を選び、小さいが非ゼロの特異値を捨てる。

$$
\boxed{
\text{厳密rankの先の特異値は元から0}
}
$$

に対し、

$$
\boxed{
\text{圧縮rankの先は非ゼロ成分を人為的に0とみなす}
}
$$

という違いがある。

第 $k$ 段階で捨てた特異値の二乗和を

$$
\varepsilon_k^2
=
\sum_{\alpha=\widetilde r_k+1}^{\rho_k}
\left(\sigma_\alpha^{(k)}\right)^2
$$

とすると、資料では標準的な左から右へのTT-SVDを入れ子の直交射影として整理し、実際の局所残差 $\varepsilon_k$ を用いた場合

$$
\|X-\widehat X\|_F^2
=
\sum_{k=1}^{d-1}\varepsilon_k^2
$$

と読む。各段階について上界しか与えない場合は

$$
\|X-\widehat X\|_F^2
\le
\sum_{k=1}^{d-1}\varepsilon_k^2
$$

となる。

---

## 1. 厳密rankと圧縮rankを分ける

第 $k$ cutのunfoldingを

$$
X^{\langle k\rangle}
$$

とし、特異値を

$$
\sigma_1^{(k)}
\ge
\sigma_2^{(k)}
\ge
\cdots
\ge0
$$

とする。

厳密TT-rankは

$$
\boxed{
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right)
}
$$

であり、理論上

$$
\sigma_1^{(k)},\ldots,\sigma_{r_k}^{(k)}>0,
$$

$$
\sigma_{r_k+1}^{(k)}
=
\sigma_{r_k+2}^{(k)}
=
\cdots
=
0.
$$

一方、圧縮のために選ぶbond dimensionを

$$
\widetilde r_k\le r_k
$$

とする。

$\widetilde r_k<r_k$ なら、捨てる

$$
\sigma_{\widetilde r_k+1}^{(k)},
\ldots,
\sigma_{r_k}^{(k)}
$$

は一般には0ではない。

---

## 2. 1回のtruncated SVD

行列

$$
A=U\Sigma V^T
$$

のrank-$r$ 近似を

$$
A_r
=
U_r\Sigma_rV_r^T
$$

とする。

捨てた成分は

$$
A-A_r
=
\sum_{j=r+1}^{\rho}
\sigma_j u_jv_j^T
$$

である。特異ベクトル対がFrobenius内積で直交するため、

$$
\begin{aligned}
\|A-A_r\|_F^2
&=
\left\|
\sum_{j=r+1}^{\rho}
\sigma_j u_jv_j^T
\right\|_F^2\\
&=
\sum_{j=r+1}^{\rho}
\sigma_j^2
\|u_jv_j^T\|_F^2\\
&=
\sum_{j=r+1}^{\rho}
\sigma_j^2.
\end{aligned}
$$

したがって

$$
\boxed{
\|A-A_r\|_F^2
=
\sum_{j=r+1}^{\rho}\sigma_j^2
}
$$

である。

---

## 3. TT-SVDの各段階での打ち切り

第 $k$ TT-SVD段階でSVDする行列を

$$
M_k
=
U_k\Sigma_kV_k^T
$$

とする。

ここで注意するのは、$k\ge2$ では $M_k$ は一般に元テンソル $X$ の $k$ 番目cut unfoldingそのものではなく、**それ以前の打ち切り後に残ったremainderを行列化したもの**であることである。

上位 $\widetilde r_k$ 個だけ残し、

$$
M_{k,\widetilde r_k}
=
U_{k,\widetilde r_k}
\Sigma_{k,\widetilde r_k}
V_{k,\widetilde r_k}^T
$$

とする。

局所打ち切り誤差を

$$
\boxed{
\varepsilon_k^2
=
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2
}
$$

と定義する。

---

## 4. rankと保存量のtrade-off

小さいrankにすると、各コア

$$
G^{(k)}\in\mathbb R^{\widetilde r_{k-1}\times n_k\times\widetilde r_k}
$$

の要素数が減る。

TT全体の保存量は

$$
P_{\mathrm{TT}}
=
\sum_{k=1}^d
\widetilde r_{k-1}n_k\widetilde r_k.
$$

したがって一般に

```text
大きい bond rank
→ 表現力が高い
→ 誤差が小さい
→ 保存量・計算量が増える

小さい bond rank
→ 圧縮率が高い
→ 誤差が大きくなる
```

というtrade-offになる。

圧縮率を

$$
\text{parameter ratio}
=
\frac{P_{\mathrm{TT}}}{P_{\mathrm{dense}}}
$$

と定義するなら、1未満で圧縮、1を超えるとTT表現の方が大きい。

---

## 5. 具体例：厳密rankと圧縮rank

あるcutの特異値が

$$
(10,3,0.4,0.02,0,0,\ldots)
$$

だとする。

非ゼロ特異値は4個なので

$$
r_k=4.
$$

厳密表現では4個を全て残す。

$$
X^{\langle k\rangle}
=
\sum_{\alpha=1}^{4}
\sigma_\alpha u_\alpha v_\alpha^T.
$$

一方、

$$
\widetilde r_k=2
$$

と圧縮すれば、$0.4$ と $0.02$ を捨てる。

$$
\widehat X^{\langle k\rangle}
=
10u_1v_1^T+3u_2v_2^T.
$$

誤差は

$$
\begin{aligned}
\left\|X^{\langle k\rangle}-\widehat X^{\langle k\rangle}\right\|_F^2
&=0.4^2+0.02^2\\
&=0.1604,
\end{aligned}
$$

したがって

$$
\left\|X^{\langle k\rangle}-\widehat X^{\langle k\rangle}\right\|_F
\approx0.4005.
$$

---

## 6. TT全体の誤差：準備

ここから資料で扱った途中式を順に残す。

実数テンソル

$$
X\in\mathbb R^{n_1\times\cdots\times n_d}
$$

を考え、Frobenius内積を

$$
\langle A,B\rangle_F
=
\sum_{i_1,\ldots,i_d}
A_{i_1,\ldots,i_d}B_{i_1,\ldots,i_d}
$$

とする。

$$
\|A\|_F^2
=
\langle A,A\rangle_F.
$$

unfoldingは要素の並べ替えなので

$$
\boxed{
\|A\|_F
=
\|A^{\langle k\rangle}\|_F
}
$$

である。

第 $k$ 段階で残す左特異ベクトルを

$$
Q_k
:=
U_k(:,1:\widetilde r_k)
$$

とすると

$$
Q_k^TQ_k=I_{\widetilde r_k}.
$$

対応する直交射影は

$$
P_k:=Q_kQ_k^T
$$

で、

$$
P_k^T=P_k,
\qquad
P_k^2=P_k.
$$

局所残差を

$$
\boxed{
\varepsilon_k
:=
\|(I-P_k)M_k\|_F
}
$$

と定義すると、SVDから

$$
\varepsilon_k^2
=
\sum_{\alpha>\widetilde r_k}
\left(\sigma_\alpha^{(k)}\right)^2.
$$

---

## 7. 左interfaceと直交射影

第 $k$ 段階までに作った左側TTコアをまとめて、左interface行列

$$
L_k
\in
\mathbb R^{(n_1\cdots n_k)\times\widetilde r_k}
$$

とする。

標準的な左から右へのTT-SVDでは

$$
L_k^TL_k=I_{\widetilde r_k}
$$

である。

その列空間への射影を

$$
\Pi_k
:=
L_kL_k^T
$$

とする。

テンソル全体上では右側に何もしないので、

$$
\mathcal P_k
=
\Pi_k\otimes I_{n_{k+1}\cdots n_d}
$$

と読む。

資料ではTT-SVDを、左から右へ進むにつれて残存部分空間が入れ子になる直交射影として整理している。

$$
\operatorname{Ran}(\mathcal P_{k+1})
\subseteq
\operatorname{Ran}(\mathcal P_k),
$$

したがって

$$
\mathcal P_{k+1}\mathcal P_k
=
\mathcal P_{k+1},
$$

$$
\mathcal P_k\mathcal P_{k+1}
=
\mathcal P_{k+1}.
$$

---

## 8. 段階ごとの誤差テンソル

第 $k$ 段階までの近似を

$$
X_k
:=
\mathcal P_kX
$$

とし、

$$
X_0:=X.
$$

最終近似は

$$
\widehat X=X_{d-1}.
$$

第 $k$ 段階で新しく捨てた成分を

$$
\boxed{
E_k
:=
X_{k-1}-X_k
}
$$

とする。

入れ子関係から

$$
\begin{aligned}
E_k
&=
\mathcal P_{k-1}X-
\mathcal P_kX\\
&=
\mathcal P_{k-1}X-
\mathcal P_k\mathcal P_{k-1}X\\
&=
(I-\mathcal P_k)\mathcal P_{k-1}X.
\end{aligned}
$$

つまり $E_k$ は、「前段階まで残っていた成分のうち、第 $k$ 段階で直交補空間へ捨てられた成分」である。

望遠鏡和で

$$
\begin{aligned}
X-\widehat X
&=
X_0-X_{d-1}\\
&=(X_0-X_1)+(X_1-X_2)+\cdots+(X_{d-2}-X_{d-1})\\
&=
\sum_{k=1}^{d-1}E_k.
\end{aligned}
$$

---

## 9. 各段階誤差の直交性

$k<\ell$ とする。

後段階の残存成分は第 $k$ 段階で残した空間の中にある。一方、$E_k$ はその直交補空間にある。

$$
E_k
\in
\operatorname{Ran}(I-\mathcal P_k),
$$

$$
E_\ell
\in
\operatorname{Ran}(\mathcal P_k).
$$

直交射影に対し

$$
\operatorname{Ran}(\mathcal P_k)
\perp
\operatorname{Ran}(I-\mathcal P_k)
$$

なので、

$$
\boxed{
\langle E_k,E_\ell\rangle_F=0
\qquad(k\ne\ell)
}
$$

と整理できる。

---

## 10. Pythagorasで全体誤差を足す

$$
X-\widehat X
=
\sum_{k=1}^{d-1}E_k
$$

かつ $E_k$ が互いに直交するとして、

$$
\begin{aligned}
\|X-\widehat X\|_F^2
&=
\left\|
\sum_{k=1}^{d-1}E_k
\right\|_F^2\\
&=
\left\langle
\sum_kE_k,
\sum_\ell E_\ell
\right\rangle_F\\
&=
\sum_k\sum_\ell
\langle E_k,E_\ell\rangle_F\\
&=
\sum_{k=1}^{d-1}\|E_k\|_F^2.
\end{aligned}
$$

次に、前段の左interfaceは列直交なので、任意の $Z$ に対して

$$
\begin{aligned}
\|QZ\|_F^2
&=
\operatorname{tr}\left((QZ)^TQZ\right)\\
&=
\operatorname{tr}\left(Z^TQ^TQZ\right)\\
&=
\operatorname{tr}(Z^TZ)\\
&=
\|Z\|_F^2.
\end{aligned}
$$

したがって、局所SVD残差を元のテンソル空間へ戻してもnormは変わらず、

$$
\|E_k\|_F
=
\|(I-P_k)M_k\|_F
=
\varepsilon_k.
$$

資料で採用した整理では、実際に捨てた特異値から $\varepsilon_k$ を定義する場合

$$
\boxed{
\|X-\widehat X\|_F^2
=
\sum_{k=1}^{d-1}\varepsilon_k^2
}
$$

となる。

一方、各段階で

$$
\|E_k\|_F\le\delta_k
$$

という上界だけを設定したなら

$$
\boxed{
\|X-\widehat X\|_F^2
\le
\sum_{k=1}^{d-1}\delta_k^2
}
$$

したがって

$$
\boxed{
\|X-\widehat X\|_F
\le
\sqrt{\delta_1^2+\cdots+\delta_{d-1}^2}
}
$$

となる。

---

## 11. 直感：なぜ単純和ではなく二乗和か

普通なら

$$
\|E_1+E_2\|
\le
\|E_1\|+\|E_2\|
$$

としか言えない。

しかし資料ではTT-SVDの各段階を直交射影として整理しているため、各段階で新たに捨てる方向が互いに直交する。

$$
E_1\perp E_2\perp\cdots
$$

直交する長さ3と4のベクトルの合計長が

$$
3+4=7
$$

ではなく

$$
\sqrt{3^2+4^2}=5
$$

になるのと同じ直感で、TT-SVDの誤差も二乗和で読む。

---

## 12. 数値計算では「0」の判定に注意

数学上のrankは非ゼロ特異値の本数である。しかし浮動小数点計算では理論上0の特異値が非常に小さい非ゼロ値として現れる。

したがって実装では数値rankを使う。

このプロジェクトでは

```python
torch.linalg.matrix_rank(mat)
```

のdefault toleranceを使用するため、threshold付近では行列shapeなどによって数値rank判定が変わり得る。

この点は [[27_TT_MPS基礎のPyTorch実装]] に分離する。

---

## 13. truncation後のrank不変性で比較する対象

第1 SVDをrank $\widetilde r_1$ へ打ち切ると、その時点で表現対象は元の $X$ から近似

$$
\widehat X
$$

へ変わる。

その後

$$
\widehat X^{\langle2\rangle}
=
(\widehat U\otimes I_{n_2})
\widehat B_{\mathrm{cut2}}
$$

が成立するので、

$$
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
=
\operatorname{rank}
\left(\widehat B_{\mathrm{cut2}}\right)
$$

は保たれる。

しかし一般には

$$
\operatorname{rank}
\left(\widehat X^{\langle2\rangle}\right)
\ne
\operatorname{rank}
\left(X^{\langle2\rangle}\right)
$$

である。これはtruncationで元の情報を意図的に落としたためである。

---

## 14. この章で固定する理解

- 厳密rank $r_k$ の先の特異値は元から0。
- 圧縮rank $\widetilde r_k<r_k$ では、非ゼロだが小さい特異値を捨てる。
- 1回のtruncated SVD誤差は捨てた特異値の二乗和。
- TTでは各段階の局所誤差を、左直交構造を使って全体誤差へ結び付ける。
- rankを小さくすると保存量は減るが近似誤差は増える。
- truncation後にrank不変性を議論するときは、元の $X$ ではなく表現対象の $\widehat X$ と remainderを比較する。
