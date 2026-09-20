---
title: Eckart–Young–Mirsky定理とTT/MPS単一ボンド打ち切りの最適性
tags:
  - TT
  - MPS
  - SVD
  - EckartYoungMirsky
  - Frobenius
  - vonNeumann
  - Schmidt
  - truncation
---

# Eckart–Young–Mirsky定理とTT/MPS単一ボンド打ち切りの最適性

[[45_TT_MPSの単一ボンドSVD打ち切り]]（[Notebook 09](../../../../notebooks/30_tt_mps/00_fundamentals/09_truncated_svd_and_bond_rank_truncation.ipynb)）で、3階TT/MPSの第2サイトを直交中心とし、中心行列 $A=G_2^{[C]\langle L\rangle}\in\mathbb R^{(r_1n_2)\times r_2}$ を特異値 $(\sigma_1,\sigma_2,\sigma_3)=(6,2,0.25)$、$k=2$ で打ち切り、誤差

$$
\|X-\widetilde X_k\|_F=\sigma_3=0.25,
\qquad
\|X-\widetilde X_k\|_F^2=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2=0.0625
$$

を数値検証した。このノートでは、**この打ち切りがすべてのrank-$k$候補の中で本当に最良なのか**という問いに答えるEckart–Young–Mirsky（EYM）定理を、直感的な比較 → $\rho-k=1$の特殊ケースの証明 → 一般の$\rho-k\ge2$への拡張の失敗と成功、まで、途中の誤った試みや疑問点も含めて追う。

Notebook 09ではさらに、$\|X\|_F^2=\|\widetilde X_k\|_F^2+\|X-\widetilde X_k\|_F^2$（保持成分と捨てたSchmidt成分の直交分解）を丸め誤差内で確認した。この**特定の構成での誤差等式**と、任意のrank-$k$候補がそれより良くならないという**最適性の下界**は別の主張である。

## 1. 定理の主張

$A\in\mathbb R^{m\times n}$、$\rho=\operatorname{rank}(A)$、$0\le k<\rho\le n$ とする（$k<\rho$は「まだ捨てる特異値が少なくとも1個ある」ために必要、$\rho\le n$は$A$のshapeから自動的に成立）。reduced SVDを

$$
A=U\Sigma V^T=\sum_{\alpha=1}^{\rho}\sigma_\alpha u_\alpha v_\alpha^T,
\qquad
\sigma_1\ge\cdots\ge\sigma_\rho>0
$$

とする。$U=[u_1\ \cdots\ u_\rho]\in\mathbb R^{m\times\rho}$、$\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_\rho)\in\mathbb R^{\rho\times\rho}$、$V=[v_1\ \cdots\ v_\rho]\in\mathbb R^{n\times\rho}$ は $U^TU=V^TV=I_\rho$ を満たす。$u_\alpha^Tu_\beta=v_\alpha^Tv_\beta=\delta_{\alpha\beta}$ で、Kroneckerのデルタ$\delta_{\alpha\beta}$は同じ添字なら1、異なる添字なら0である。

truncated SVDを $A_k=\sum_{\alpha=1}^{k}\sigma_\alpha u_\alpha v_\alpha^T$ と定義する。ここで $B$ は、$A$の特異ベクトルとは無関係な**任意**の行列であり、制約は $\operatorname{rank}(B)\le k$ のみである（ランダムな低rank行列、特異ベクトルを回転させた行列、零行列なども候補に含まれる）。この任意の $B\in\mathbb R^{m\times n}$、$\operatorname{rank}(B)\le k$ に対して、

$$
\boxed{
\|A-B\|_F^2\ge\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2
}
$$

が成り立ち、$A_k$がこの下界を達成する。$p=\min(m,n)$ とすれば $\sigma_{\rho+1}=\cdots=\sigma_p=0$ である。$U_k\in\mathbb R^{m\times k}$、$\Sigma_k\in\mathbb R^{k\times k}$、$V_k\in\mathbb R^{n\times k}$ より $A_k=U_k\Sigma_kV_k^T\in\mathbb R^{m\times n}$、$\operatorname{rank}(A_k)\le k$ で、確かに候補に含まれる。最適値と最適解は

$$
\boxed{A_k\in\operatorname*{argmin}_{\operatorname{rank}(B)\le k}\|A-B\|_F}
$$

および

$$
\boxed{
\min_{\operatorname{rank}(B)\le k}\|A-B\|_F
=\|A-A_k\|_F
=\sqrt{\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2}
}
$$

同じ$A_k$は作用素ノルムでも最適である。

$$
\boxed{
\min_{\operatorname{rank}(B)\le k}\|A-B\|_2=\|A-A_k\|_2=\sigma_{k+1}
}
$$

文献では、truncated SVDの最適性をFrobenius・作用素ノルムより広い**ユニタリ不変ノルム**へ拡張した主張をSchmidt–Mirsky定理とも呼ぶ。このノートで証明するのはFrobenius版と、kernel法で示す作用素ノルム版である。

$\rho-k=1$（捨てる非ゼロ特異値が1個）のときのみ、両ノルムの最小値は $\sigma_{k+1}$ に一致する。$\rho-k\ge2$ なら

$$
\|A-A_k\|_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_\rho^2}
\;>\;
\sigma_{k+1}=\|A-A_k\|_2.
$$

## 2. なぜ最適性を与えるか：直感（成分ごとの比較）

$A=\sum_{\alpha=1}^\rho\sigma_\alpha u_\alpha v_\alpha^T$ の各成分 $u_\alpha v_\alpha^T$ はFrobenius内積で正規直交系である（3節で証明）。したがって成分ごとのFrobeniusノルムはその特異値そのものであり、

$$
\|\sigma_1u_1v_1^T\|_F=\sigma_1,\quad\|\sigma_2u_2v_2^T\|_F=\sigma_2,\quad\|\sigma_3u_3v_3^T\|_F=\sigma_3.
$$

Notebook 09の数値 $(\sigma_1,\sigma_2,\sigma_3)=(6,2,0.25)$ で、rank-2まで落とすときにどの1成分を捨てるかを比較すると、

| 残す特異値の組 | 捨てる成分 | 誤差 $\|A-B\|_F$ |
|---|---|---|
| $(6,2)$ | $\sigma_3=0.25$ | $0.25$ |
| $(6,0.25)$ | $\sigma_2=2$ | $2$ |
| $(2,0.25)$ | $\sigma_1=6$ | $6$ |

一番小さい成分 $\sigma_3$ を捨てるのが直感的にも最良に見える。ただしこれは「SVDの3成分をそのまま選ぶ候補」だけの比較であり、EYM定理はこれよりずっと広い**あらゆる**rank-2行列 $B$（$A$の特異ベクトルと無関係な形も含む）と比較しても、truncated SVDに勝てないことを主張する、より強い定理である。この広さの理解が3節以降の証明の出発点になる。

## 3. 上界：truncated SVDの誤差（Pythagoras）

$E_\alpha:=u_\alpha v_\alpha^T$ とおく。これは外積によるrank-1行列で、行要素は $(E_\alpha)_{ij}=u_\alpha(i)v_\alpha(j)$。Frobenius内積 $\langle P,Q\rangle_F:=\operatorname{tr}(P^TQ)=\sum_{i=1}^{m}\sum_{j=1}^{n}P_{ij}Q_{ij}$ と、$u_\alpha,v_\alpha$ の正規直交性から

$$
\langle E_\alpha,E_\beta\rangle_F
=\operatorname{tr}\!\left[(u_\alpha v_\alpha^T)^T(u_\beta v_\beta^T)\right]
=(\sum_i u_\alpha(i)u_\beta(i))(\sum_j v_\alpha(j)v_\beta(j))
=(u_\alpha^Tu_\beta)(v_\alpha^Tv_\beta)
=\delta_{\alpha\beta}\delta_{\alpha\beta}=\delta_{\alpha\beta}
$$

すなわち $\{E_\alpha\}$ はFrobenius内積で正規直交系である。したがって

$$
A-A_k=\sum_{\alpha=k+1}^{\rho}\sigma_\alpha E_\alpha
\quad\Longrightarrow\quad
\|A-A_k\|_F^2
=\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2\|E_\alpha\|_F^2
=\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2.
$$

これは「$A_k$を選べばこの誤差を達成できる」という**上界**である。最適性を言うには、任意の $B$ に対してこれ未満にできないことを示す必要がある。

同じ残差 $A-A_k=\sum_{\alpha=k+1}^{\rho}\sigma_\alpha u_\alpha v_\alpha^T$ は、非ゼロ特異値が $\sigma_{k+1},\ldots,\sigma_\rho$ のSVDでもある。従って作用素ノルムでは $\|A-A_k\|_2=\max_{\alpha>k}\sigma_\alpha=\sigma_{k+1}$。これは作用素ノルム版の**上界**であり、任意の$B$への下界は4–5節で示す。

## 4. $\rho-k=1$ の場合の下界証明（kernelを使う方法）

この節の証明は $k=\rho-1$、すなわち捨てる特異値が $\sigma_\rho$ 一つだけの場合に**限って**完全である。理由は5節で明らかにする。

### 4.1 rank-nullityの定理

**主張**：線形写像 $B:\mathbb R^n\to\mathbb R^m$ に対し $\dim\mathcal N(B)+\operatorname{rank}(B)=n$。

**証明**：$\mathcal N(B)=\{x\in\mathbb R^n\mid Bx=0\}$ の基底を $x_1,\ldots,x_r$（$r=\dim\mathcal N(B)$）とし、基底拡張定理でこれを $\mathbb R^n$ の基底 $x_1,\ldots,x_r,x_{r+1},\ldots,x_n$ に拡張する。$Bx_1=\cdots=Bx_r=0$ なので、任意の $v=\sum_ic_ix_i$ に対し $Bv=\sum_{i=r+1}^nc_iBx_i$ となり、

$$
\operatorname{Im}(B)=\operatorname{span}\{Bx_{r+1},\ldots,Bx_n\}.
$$

$Bx_{r+1},\ldots,Bx_n$ が一次独立であることを背理法で示す。もし $\sum_{i=r+1}^nc_iBx_i=0$（すべてゼロではない $c_i$）なら、$w:=\sum_{i=r+1}^nc_ix_i\in\mathcal N(B)$ なので $w=\sum_{i=1}^rd_ix_i$ とも書け、

$$
\sum_{i=1}^rd_ix_i-\sum_{i=r+1}^nc_ix_i=0.
$$

基底の一次独立性から全係数がゼロとなり、$c_i$ が非ゼロという仮定に矛盾する。よって一次独立であり、

$$
\operatorname{rank}(B)=n-r=n-\dim\mathcal N(B). \qquad\blacksquare
$$

### 4.2 次元カウントによる存在証明

$k=\rho-1$ とし、任意の $B\in\mathbb R^{m\times n}$、$\operatorname{rank}(B)\le k$ を取る。4.1より

$$
\dim\mathcal N(B)=n-\operatorname{rank}(B)\ge n-k=n-\rho+1. \tag{1}
$$

上位 $\rho$ 本の右特異ベクトル空間 $\mathcal V_\rho:=\operatorname{span}\{v_1,\ldots,v_\rho\}$ は直交性より $\dim\mathcal V_\rho=\rho$。

部分空間の次元公式 $\dim(S+T)=\dim S+\dim T-\dim(S\cap T)$ と $\dim(S+T)\le n$ から、

$$
\dim(S\cap T)\ge\dim S+\dim T-n.
$$

$S=\mathcal N(B)$, $T=\mathcal V_\rho$ を代入し(1)を使うと、

$$
\dim(\mathcal N(B)\cap\mathcal V_\rho)
\ge(n-\rho+1)+\rho-n=1. \tag{2}
$$

したがってゼロでない $z\in\mathcal N(B)\cap\mathcal V_\rho$ が存在し、$\|z\|_2=1$ と正規化できる。$z\in\mathcal N(B)$ より $Bz=0$。$z\in\mathcal V_\rho$ より

$$
z=\sum_{\alpha=1}^{\rho}c_\alpha v_\alpha,\qquad\sum_{\alpha=1}^{\rho}c_\alpha^2=1.
$$

SVDの直交性 $v_\beta^Tv_\alpha=\delta_{\alpha\beta}$、$u_\beta$の正規直交性より

$$
Az=\sum_{\alpha=1}^{\rho}\sigma_\alpha c_\alpha u_\alpha,
\qquad
\|Az\|_2^2=\sum_{\alpha=1}^{\rho}\sigma_\alpha^2c_\alpha^2
\ge\sigma_\rho^2\sum_{\alpha=1}^{\rho}c_\alpha^2=\sigma_\rho^2
$$

（$\sigma_1\ge\cdots\ge\sigma_\rho$ より各項を最小値 $\sigma_\rho^2$ で下から押さえた）。$Bz=0$ より $(A-B)z=Az$ なので $\|(A-B)z\|_2=\|Az\|_2\ge\sigma_\rho$。

ここで使う三つのノルムを区別する。$M\in\mathbb R^{m\times n}$、$z\in\mathbb R^n$ に対して、$\|Mz\|_2^2=\sum_{i=1}^{m}(Mz)_i^2$ は**特定の入力方向**でのベクトルノルム、$\|M\|_2=\max_{\|x\|_2=1}\|Mx\|_2$ は**全単位方向での最大伸長率**（作用素ノルム、spectral norm）、$\|M\|_F^2=\sum_{i=1}^{m}\sum_{j=1}^{n}M_{ij}^2$ は**全行列要素の二乗和**である。$\|z\|_2=1$ だから、定義から直ちに

$$
\|Mz\|_2\le\max_{\|x\|_2=1}\|Mx\|_2=\|M\|_2.
$$

$M=\sum_{i=1}^{p}\tau_i s_it_i^T$、$p=\min(m,n)$、$\tau_1\ge\cdots\ge\tau_p\ge0$ とする。右特異ベクトル$\{t_i\}$は正規直交系なので、$x_\parallel:=\sum_i(t_i^Tx)t_i$ と $x_\perp:=x-x_\parallel$ は直交する。従って $\|x\|_2^2=\|x_\parallel\|_2^2+\|x_\perp\|_2^2\ge\sum_i(t_i^Tx)^2$。これがここで使うBesselの不等式である。任意の単位ベクトル$x$について、左特異ベクトルの直交性も合わせると

$$
\|Mx\|_2^2
=\sum_{i=1}^{p}\tau_i^2(t_i^Tx)^2
\le\tau_1^2\sum_{i=1}^{p}(t_i^Tx)^2
\le\tau_1^2\|x\|_2^2=\tau_1^2.
$$

$x=t_1$ で等号を達成するので $\|M\|_2=\tau_1$。またSVD成分の直交性により $\|M\|_F^2=\sum_i\tau_i^2\ge\tau_1^2=\|M\|_2^2$ である。平方根を取って

$$
\|Mz\|_2\le\|M\|_2\le\|M\|_F.
$$

作用素ノルムは最も伸びる**1方向**、Frobeniusノルムは**全方向の二乗の合計**を測る。この鎖を下界として逆向きに読み、$M=A-B$ として

$$
\boxed{\|A-B\|_F\ge\|A-B\|_2\ge\|(A-B)z\|_2\ge\sigma_\rho.}
$$

### 4.3 結論とNotebook 09数値との対応

3節の上界と4.2の下界が一致するので、

$$
\min_{\operatorname{rank}(B)\le\rho-1}\|A-B\|_F=\|A-A_{\rho-1}\|_F=\sigma_\rho.
$$

Notebook 09では $\rho=3,\,k=2,\,\sigma_3=0.25$ なので、この一般形にちょうど当てはまり、

$$
\min_{\operatorname{rank}(B)\le2}\|A-B\|_F=0.25
$$

が理論的に確定する。

Notebook 09の教材設定は $n_1=4,\,n_2=3,\,n_3=5,\,r_1=2,\,r_2=3$ である。したがって $A$ のshapeは $(r_1n_2,r_2)=(6,3)$、reduced SVDのshapeは $U:(6,3)$、$S:(3,)$、$V^T:(3,3)$。$k=2$ では $U_k:(6,2)$、$S_k:(2,)$、$V_k^T:(2,3)$、$\widetilde G_2:(2,3,2)$、transfer $\Sigma_kV_k^T:(2,3)$、$\widetilde G_3:(2,5,1)$、$\widetilde X:(4,3,5)$ となる。保持ノルムと残差ノルムも

$$
\|X\|_F^2=6^2+2^2+0.25^2=40.0625,
\quad\|\widetilde X_2\|_F^2=6^2+2^2=40,
\quad\|X-\widetilde X_2\|_F^2=0.0625
$$

と一致する。添付資料に記録された浮動小数点の値は、テンソル誤差二乗 $0.06250000000000003$、捨てた特異値の二乗和 $0.0625000000000003$、誤差等式の差約 $2.78\times10^{-16}$、Pythagorasの差約 $2.84\times10^{-14}$ である。これらは資料に記録されたNotebook 09の結果であり、このノートを編集する際に再実行した値ではない。

### 4.4 補足：この証明で見落としていた前提

証明を最初に述べた際、$k<\rho\le n$ という前提を明示していなかった。$\dim\mathcal V_{k+1}=k+1$ を主張するには $v_{k+1}$（すなわち $k+1\le\rho$）が存在する必要があり、また $\mathcal V_{k+1}\subseteq\mathbb R^n$ が意味を持つには $k+1\le n$ も必要である。Notebook 09の $\rho=3,\,k=2,\,n=3$ では $k+1=n=\rho=3$ というちょうど境界的な場合になっており、$\mathcal V_{k+1}=\mathbb R^n$ 全体に一致する特殊な状況だった。一般の $k+1<n$ のケース（例えば $n=10,\,k=2$）で確認するとこの次元カウントの非自明さがよりはっきり見える。

## 5. 一般 $\rho-k\ge2$ への拡張の限界

### 5.1 目標と手持ちの下界のギャップ

$q:=\rho-k$ とする。目標は

$$
\|A-B\|_F^2\ge\sigma_{k+1}^2+\sigma_{k+2}^2+\cdots+\sigma_\rho^2 \tag{Goal}
$$

（$q$個の項の和）である。4.2は $k=\rho-1$ の場合だったので、ここで**一般の$k$に対する1本の証明を途中式から書く**。任意の$\operatorname{rank}(B)\le k$について

$$
\dim\mathcal N(B)=n-\operatorname{rank}(B)\ge n-k,
\qquad
\mathcal V_{1:k+1}:=\operatorname{span}\{v_1,\ldots,v_{k+1}\},
\qquad
\dim\mathcal V_{1:k+1}=k+1.
$$

部分空間の次元公式を使うと

$$
\dim\bigl(\mathcal N(B)\cap\mathcal V_{1:k+1}\bigr)
\ge\dim\mathcal N(B)+\dim\mathcal V_{1:k+1}-n
\ge(n-k)+(k+1)-n=1.
$$

よって交差内に単位ベクトル $z=\sum_{\alpha=1}^{k+1}c_\alpha v_\alpha$ があり、$Bz=0$、$\sum_\alpha c_\alpha^2=1$。SVDを右から掛けて左特異ベクトルの直交性を使えば

$$
Az=\sum_{\alpha=1}^{k+1}\sigma_\alpha c_\alpha u_\alpha,
\qquad
\|Az\|_2^2=\sum_{\alpha=1}^{k+1}\sigma_\alpha^2c_\alpha^2
\ge\sigma_{k+1}^2\sum_{\alpha=1}^{k+1}c_\alpha^2=\sigma_{k+1}^2.
$$

従って $(A-B)z=Az$ と4.2のノルム不等式から

$$
\|A-B\|_F\ge\|A-B\|_2
\ge\|(A-B)z\|_2=\|Az\|_2\ge\sigma_{k+1}.
$$

この証明から得られるFrobenius版の下界は

$$
\|A-B\|_F^2\ge\sigma_{k+1}^2 \tag{3}
$$

（1項のみ）。$\sigma_{k+2},\ldots,\sigma_\rho>0$ なので $q\ge2$ では

$$
\sigma_{k+1}^2<\sigma_{k+1}^2+\cdots+\sigma_\rho^2
$$

となり、(3)は(Goal)より真に弱い。例えば $\rho=4,\,k=2$ なら、捨てる特異値は $\sigma_3,\sigma_4$ の2個であり、(3)は $\sigma_3^2$ しか保証しないが、目標は $\sigma_3^2+\sigma_4^2$ である。二つの正の数の和は、そのうちの一方だけより必ず大きいため、これは常に不足する。

### 5.2 なぜ$q$本の$z$を同時に確保できないか

$q$本の直交単位ベクトル $z_1,\ldots,z_q$（$Bz_j=0$）が取れれば、$Z=[z_1\cdots z_q]$（$Z^TZ=I_q$）に対し

$$
\|M\|_F^2\ge\|MZ\|_F^2=\sum_{j=1}^q\|Mz_j\|_2^2 \tag{4}
$$

が成り立つ。実際、$MZ=[Mz_1\ \cdots\ Mz_q]$ なので等号はFrobeniusノルムの列ごとの定義である。また $ZZ^T$ は直交射影で、これを直交基底 $[Z\ Z_\perp]$ まで拡張すれば $\|M\|_F^2=\|MZ\|_F^2+\|MZ_\perp\|_F^2\ge\|MZ\|_F^2$ である。1本の議論はこの$q=1$の場合。この発想自体は正しい方向である。

kernelの中だけなら、$\dim\mathcal N(B)\ge n-k\ge\rho-k=q$（$\rho\le n$より）なので、正規直交な$q$本を確かに取れる。しかし**問題はその$q$本が$A$の残り特異方向を捉える保証がないこと**である。

任意の $z_j\in\mathcal N(B)$ は、$A$ の右特異ベクトルと零空間成分に分けて

$$
z_j=\sum_{\alpha=1}^{\rho}c_{\alpha j}v_\alpha+z_j^{(0)},
\qquad z_j^{(0)}\in\mathcal N(A),
\qquad
\|Az_j\|_2^2=\sum_{\alpha=1}^{\rho}\sigma_\alpha^2c_{\alpha j}^2
$$

となる。$Az_j^{(0)}=0$ なので、$Bz_j=0$ だけでは $\|Az_j\|_2^2\ge\sigma_{k+j}^2$ は導けない。$B$の核が十分大きくても、それが$A$の非ゼロ特異方向を含むとは限らない。

$\mathcal N(B)\cap\mathcal V_{1:k+1}$（次元 $k+1$ の空間との交差）から本数を増やそうとしても、次元公式は

$$
\dim(\mathcal N(B)\cap\mathcal V_{1:k+1})\ge(n-k)+(k+1)-n=1
$$

で、**常に1本しか保証されない**。二つの部分空間の次元の和がちょうど $n+1$ にしかならないため、1次元以上の重なりを強制できないからである。

小さい特異値側の空間 $\mathcal V_{k+1:\rho}$（次元 $q$）との交差を使っても、

$$
\dim(\mathcal N(B)\cap\mathcal V_{k+1:\rho})\ge(n-k)+q-n=q-k
$$

となり、$k\ge q$ なら非正で、非ゼロベクトルの存在すら保証できない。すなわち、**$B$は自分のkernelを捨てるべき方向から完全にそらすことができる**。

### 5.3 誤った拡張案（記録として残す）

検討の過程で「$\mathcal N(B)\cap\mathcal V_{k+1:\rho}$から捨てる本数 $q=\rho-k$ 本を常に取れる」と示唆したが、これは誤りだった。次元公式が保証する本数の下界は $\max(0,q-k)=\max(0,\rho-2k)$ であり、$k\ge q$ なら0本しか保証しない。$k>0$ で $\rho-2k>0$ の場合に数本取れても、必要な$q$本には届かない。**この単純な交差次元の評価だけでは、一般の捨てた特異値すべてを同時に強制できない**ことが、この試行錯誤で判明した限界である。

### 5.4 なぜ下界が「支えられない」のか：アドバーサリー的な直感

4.2の証明で見つかる $z$ は、$\mathcal N(B)\cap\mathcal V_{1:k+1}$ の中に「少なくとも1本存在する」ことしか保証されない。その中身は、

$$
z=c_1v_1+\cdots+c_{k+1}v_{k+1},\qquad\sum_\alpha c_\alpha^2=1
$$

という形だが、**係数 $c_\alpha$ の配分は、$B$（相手方）が実質的にコントロールできる**。$\rho=4,\,k=2$ の例で、捨てたい特異値が $\sigma_3,\sigma_4$ の2個あるとき、$B$ が自分のkernelをちょうど

$$
\mathcal N(B)\cap\mathcal V_{1:3}=\operatorname{span}\{v_3\}
$$

（$v_3$方向だけ）になるように仕込めば、見つかる $z$ は必然的に $z=v_3$（$c_3=1,\,c_4=0$に相当）となり、

$$
\|Az\|_2^2=\sigma_3^2\cdot1^2+\sigma_4^2\cdot0^2=\sigma_3^2.
$$

$\sigma_4$の寄与は、選んだ空間 $\mathcal V_{1:3}$ に $v_4$ が含まれないため、この**1本の評価式には**現れない。$B$の核とこの空間の交差を1次元にする選び方は可能であり、その場合この方法で得る下界は $\sigma_3^2$ にとどまる。ただし、行列全体の誤差が $\sigma_3^2$ だけで済むという意味ではない。

これが「**この交差空間から**複数の特異値を同時に強制できない」ことの具体的な理由である。ただし、$m=n=\rho=4,\,k=2$ のとき $\dim\mathcal N(B)\ge2$ なので、核そのものを「ちょうど1次元」と言うのは誤りである。ちょうど1次元にできるのは $\mathcal N(B)\cap\mathcal V_{1:3}$ のほうである。

実際、$\{v_1,v_2,v_3,v_4\}$ を直交基底とし、$w_+=(v_1+v_4)/\sqrt2$、$w_-=(v_1-v_4)/\sqrt2$ と置く。$B=v_2v_2^T+w_-w_-^T$ はrank $2$ の行列で、

$$
\mathcal N(B)=\operatorname{span}\{v_3,w_+\},
\qquad
\mathcal N(B)\cap\operatorname{span}\{v_1,v_2,v_3\}=\operatorname{span}\{v_3\},
\qquad v_4\notin\mathcal N(B).
$$

従って上位3方向との交差から取れるのは $z_1=v_3$ だけで、2本目の $v_4$ を核ベクトルとして**同じやり方で**追加できない。この例はkernel法の証明上の限界を示すのであり、EYMの正しい下界 $\sigma_3^2+\sigma_4^2$ を破る例ではない。

### 5.5 kernel法の適用範囲のまとめ

| 項目 | kernel法（4節）で示せること |
|---|---|
| $\|A-B\|_2\ge\sigma_{k+1}$（作用素ノルム版） | 任意の$k$で成立 |
| $\|A-B\|_F\ge\sigma_{k+1}$ | 任意の$k$で成立（$\|\cdot\|_F\ge\|\cdot\|_2$より） |
| $\|A-B\|_F^2\ge\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2$ | $\rho-k=1$のときのみ十分。$\rho-k\ge2$では不足 |

一般のFrobenius版には、個々の方向を追う方法ではなく、行列全体の特異値の並びを同時に比較する道具が必要になる。

## 6. von Neumannのトレース不等式

kernel法が1本の入力方向 $z$ を調べるのに対し、ここでは**両行列の特異値を順位ごとに同時比較**する。Frobenius内積 $\langle A,B\rangle_F=\sum_{ij}A_{ij}B_{ij}$、すなわち行列要素ごとの積の和は、特異値を大きい順に対応させた積の和を超えない。$\|A-B\|_F^2=\|A\|_F^2+\|B\|_F^2-2\langle A,B\rangle_F$ の負符号を通じて、内積の上界が誤差の下界になる。これにより、$\sigma_{k+1}^2$ という1項だけでなく、捨てた特異値の二乗和全体を押さえられる。

**主張**：$A,B\in\mathbb R^{n\times n}$（一般の$m\times n$も同様）の特異値を $\sigma_1(A)\ge\cdots\ge\sigma_n(A)\ge0$、$\sigma_1(B)\ge\cdots\ge\sigma_n(B)\ge0$ とすると、

$$
\boxed{
\operatorname{tr}(A^TB)\le\sum_{i=1}^{n}\sigma_i(A)\sigma_i(B)
}
$$

長方形 $A,B\in\mathbb R^{m\times n}$ の場合は $N=\max(m,n)$ とし、両者を右または下へゼロで埋めて $N\times N$ 行列 $\widehat A,\widehat B$ にする。$\operatorname{tr}(\widehat A^T\widehat B)=\operatorname{tr}(A^TB)$、非ゼロ特異値は同じで、追加された特異値は0である。したがって以下の正方行列の証明は、$p=\min(m,n)$ までの和として長方形にも適用できる。7節で $\sum_{i=1}^{n}$ と書くときは、$n>p$ の場合の特異値を0で延長したものとする。

### 6.1 Ky Fanの最大値原理（引用、証明は別立て）

任意の $M\in\mathbb R^{m\times n}$ と正規直交列 $X\in\mathbb R^{m\times k}$（$X^TX=I_k$）、$Y\in\mathbb R^{n\times k}$（$Y^TY=I_k$）に対して、

$$
\operatorname{tr}(X^TMY)\le\sum_{i=1}^{k}\sigma_i(M),
$$

かつ$X,Y$を$M$の上位$k$特異ベクトルに選ぶと等号成立。この定理はKy Fan (1949)の結果として引用し、本ノートでは再証明しない。

### 6.2 Abel和分による証明

$B=\sum_{j=1}^n\sigma_j(B)p_jq_j^T$（SVD）とし、部分和 $S_k=\sum_{j=1}^kp_jq_j^T$（$S_0=0$、$\sigma_{n+1}(B):=0$）を定義する。恒等式

$$
B=\sum_{k=1}^n\bigl(\sigma_k(B)-\sigma_{k+1}(B)\bigr)S_k
$$

が成り立つ（両辺で$p_jq_j^T$の係数を比較すると、右辺は望遠鏡和 $\sum_{k=j}^n(\sigma_k(B)-\sigma_{k+1}(B))=\sigma_j(B)$ に一致）。したがって

$$
\operatorname{tr}(A^TB)=\sum_{k=1}^n\bigl(\sigma_k(B)-\sigma_{k+1}(B)\bigr)\operatorname{tr}(A^TS_k).
$$

$S_k=P_kQ_k^T$（$P_k=[p_1,\ldots,p_k]$、$Q_k=[q_1,\ldots,q_k]$は正規直交列）とすると、トレースの巡回性より

$$
\operatorname{tr}(A^TS_k)=\operatorname{tr}(A^TP_kQ_k^T)
=\operatorname{tr}(Q_k^TA^TP_k)
=\operatorname{tr}((Q_k^TA^TP_k)^T)
=\operatorname{tr}(P_k^TAQ_k)
\le\sum_{i=1}^{k}\sigma_i(A)
$$

（Ky Fanの原理）。$\sigma_k(B)-\sigma_{k+1}(B)\ge0$なので不等号の向きを保ったまま代入でき、

$$
\operatorname{tr}(A^TB)\le\sum_{k=1}^n\bigl(\sigma_k(B)-\sigma_{k+1}(B)\bigr)\sum_{i=1}^{k}\sigma_i(A).
$$

右辺で和の順序を$i\le k$の範囲で入れ替えると、

$$
\sum_{k=1}^n\bigl(\sigma_k(B)-\sigma_{k+1}(B)\bigr)\sum_{i=1}^{k}\sigma_i(A)
=\sum_{i=1}^n\sigma_i(A)\sum_{k=i}^n\bigl(\sigma_k(B)-\sigma_{k+1}(B)\bigr)
=\sum_{i=1}^n\sigma_i(A)\sigma_i(B)
$$

（内側の和も望遠鏡和で$\sigma_i(B)$に一致）。以上より

$$
\operatorname{tr}(A^TB)\le\sum_{i=1}^n\sigma_i(A)\sigma_i(B). \qquad\blacksquare
$$

**等号成立**：$A$と$B$が同じ特異ベクトルを共有する場合（例えば$B=A_k$）、各$k$でKy Fanの等号が成立し、全体でも等号となる。

### 6.3 撤回した二重劣確率行列の試み：成立する途中式と破綻点

添付資料では、最初にBirkhoff–von Neumannの定理と並べ替え不等式を使う道も試した。$A=U\operatorname{diag}(a)V^T$、$B=P\operatorname{diag}(b)Q^T$（$a_i=\sigma_i(A)$、$b_j=\sigma_j(B)$、双方降順）とし、まず正方行列として $W=U^TP$、$Z=Q^TV$ と置く。トレースの巡回性と対角成分の展開は正しく、

$$
\begin{aligned}
\operatorname{tr}(A^TB)
&=\operatorname{tr}\!\left(V\operatorname{diag}(a)U^TP\operatorname{diag}(b)Q^T\right)\\
&=\operatorname{tr}\!\left(\operatorname{diag}(a)W\operatorname{diag}(b)Z\right)\\
&=\sum_{i=1}^{n}\sum_{j=1}^{n}a_i b_j W_{ij}Z_{ji}.
\end{aligned}
$$

$c_{ij}=W_{ij}Z_{ji}$、$D_{ij}=|c_{ij}|$ とすると、直交行列の行・列の二乗和が1であることとCauchy–Schwarz不等式から

$$
\sum_jD_{ij}=\sum_j|W_{ij}Z_{ji}|
\le\sqrt{\sum_jW_{ij}^2}\sqrt{\sum_jZ_{ji}^2}=1,
\qquad
\sum_iD_{ij}\le\sqrt{\sum_iW_{ij}^2}\sqrt{\sum_iZ_{ji}^2}=1.
$$

したがって $D$ は**二重劣確率行列**（非負で各行和・各列和が1以下）であり、$a_i,b_j\ge0$ から $\operatorname{tr}(A^TB)\le\sum_{ij}a_ib_jD_{ij}$ までは正しい。一方、資料の次の「不足分を1行1列足して、$D$を $(n+1)\times(n+1)$ の二重確率行列へ埋め込む」というステップは一般には成立しない。提案されたブロック行列は

$$
D'=\begin{pmatrix}D&r\\c^T&s\end{pmatrix},
\qquad r_i=1-\sum_jD_{ij},
\qquad c_j=1-\sum_iD_{ij}
$$

である。$D'=\text{二重確率行列}$ とするには最後の行和から $s=1-\sum_jc_j$ が必要。例えば $n=2, D=0$ では $c=(1,1)$ なので $s=-1$ となり非負性に反する。$D=0$ は $W=I_2$、$Z$を2列の交換行列とすれば実際に起こりうる。隅の成分を非負に選んでも行和1にはできない。

このため、その先で**二重確率行列**（各行・各列の和がちょうど1）を置換行列の凸結合 $D'=\sum_\ell\lambda_\ell\Pi_\ell$（$\lambda_\ell\ge0,\ \sum_\ell\lambda_\ell=1$）と表すBirkhoff–von Neumannの定理を使っても、元の試みは証明としてつながらない。二重確率行列全体の集合はBirkhoff多面体とも呼ばれ、その頂点が置換行列である。降順列について $\sum_i a_ib_{\pi(i)}\le\sum_i a_ib_i$ とする並べ替え不等式自体は、逆転 $i<j,\,\pi(i)>\pi(j)$ を交換すると

$$
\left(a_ib_{\pi(i)}+a_jb_{\pi(j)}\right)
-\left(a_ib_{\pi(j)}+a_jb_{\pi(i)}\right)
=(a_i-a_j)(b_{\pi(i)}-b_{\pi(j)})\le0
$$

なので正しい。有限回の逆転解消で恒等置換へ進めば和は減らない。仮に適切な$D'$が得られていたなら、

$$
\sum_{i,j}a_ib_jD'_{ij}
=\sum_\ell\lambda_\ell\sum_i a_ib_{\pi_\ell(i)}
\le\sum_\ell\lambda_\ell\sum_i a_ib_i
=\sum_i a_ib_i
$$

と結べたはずだが、この鎖の**入口である$D'$の存在**が上記の構成では示せない。そこで6.1–6.2節では**Ky Fanの最大値原理を引用し、Abel和分で証明する経路**を採用した。撤回した途中式を定理の証明と混同しない。

### 6.4 別ルートの存在について（比較のみ、本ノートでは未使用）

一般Frobenius版EYMの証明には、本ノートで採用したvon Neumann／Mirskyのルート以外に、特異値の不等式（Weyl型）

$$
\sigma_i(A-B)\ge\sigma_{i+k}(A),\qquad i=1,\ldots,p-k\quad(p=\min(m,n))
$$

を各$i$で示し、

$$
\|A-B\|_F^2=\sum_{i=1}^p\sigma_i(A-B)^2\ge\sum_{i=1}^{p-k}\sigma_{i+k}(A)^2=\sum_{\alpha=k+1}^{\rho}\sigma_\alpha(A)^2
$$

と結論するルートもある。どちらも最終結論は同じだが、本ノートでは前者（von Neumann経由）のみを完全に導出した。

## 7. 一般Frobenius版EYM定理の証明（Mirskyの不等式経由）

### 7.1 内積展開とMirskyの不等式

Frobenius内積の双線形性と対称性をまず1項ずつ使うと、

$$
\begin{aligned}
\langle A-B,A-B\rangle_F
&=\langle A,A\rangle_F-\langle A,B\rangle_F
-\langle B,A\rangle_F+\langle B,B\rangle_F\\
&=\|A\|_F^2-2\langle A,B\rangle_F+\|B\|_F^2,
\end{aligned}
$$

である。ここで $\langle A,B\rangle_F=\langle B,A\rangle_F=\operatorname{tr}(A^TB)$。

$$
\|A-B\|_F^2=\langle A-B,A-B\rangle_F=\|A\|_F^2+\|B\|_F^2-2\operatorname{tr}(A^TB). \tag{5}
$$

$\|A\|_F^2=\sum_i\sigma_i(A)^2$、$\|B\|_F^2=\sum_i\sigma_i(B)^2$（rank-1成分の直交性によるPythagoras、3節と同様）を代入し、6節のvon Neumann不等式 $-2\operatorname{tr}(A^TB)\ge-2\sum_i\sigma_i(A)\sigma_i(B)$ を(5)に使うと、

$$
\|A-B\|_F^2
\ge\sum_i\sigma_i(A)^2+\sum_i\sigma_i(B)^2-2\sum_i\sigma_i(A)\sigma_i(B)
=\sum_i\bigl(\sigma_i(A)-\sigma_i(B)\bigr)^2.
$$

$$
\boxed{\|A-B\|_F^2\ge\sum_{i=1}^{n}\bigl(\sigma_i(A)-\sigma_i(B)\bigr)^2} \tag{Mirsky}
$$

### 7.2 rank制約の代入

$\operatorname{rank}(B)\le k$ より $\sigma_i(B)=0\;(i>k)$。(Mirsky)の右辺を$i\le k$と$i>k$に分け、

$$
\sum_{i=1}^n(\sigma_i(A)-\sigma_i(B))^2
=\underbrace{\sum_{i=1}^k(\sigma_i(A)-\sigma_i(B))^2}_{\ge0}
+\sum_{i=k+1}^n\sigma_i(A)^2.
$$

第1項は非負なので取り除いても不等号は保たれ、

$$
\boxed{
\|A-B\|_F^2\ge\sum_{\alpha=k+1}^{\rho}\sigma_\alpha(A)^2
}
$$

（$A$の特異値は$\rho$より先はゼロなので和の上限を$\rho$にできる）。これが5節で不足していたFrobenius版の一般証明である。

### 7.3 等号成立と依存関係のまとめ

第1項が0となる条件は $\sigma_i(B)=\sigma_i(A)\;(i\le k)$ である。これは**特異値だけ**の条件であり、$B=A_k$ と同値ではない。例えば同じ特異値を持っていても特異ベクトルの向きが違えば $B\ne A_k$ となる。$B=A_k$ と選ぶと、

$$
\sigma_i(A_k)=
\begin{cases}
\sigma_i(A),&1\le i\le k,\\
0,&i>k,
\end{cases}
$$

なので第1項は0で、さらに $A$ と $B$ の特異ベクトルが揃うため von Neumannの不等式も等号となり、3節の上界と完全に一致する。退化した特異値が打ち切り境界にある場合、最小化解は複数存在し得る。証明の依存関係は

$$
\text{Ky Fanの原理（引用）}
\to
\text{von Neumannのトレース不等式（6節）}
\to
\text{Mirskyの不等式（7.1）}
\to
\text{一般Frobenius版EYM（7.2）}
$$

この証明の主要な外部定理はKy Fanの原理である。rank-nullityは4.1節で証明し、ノルム不等式やAbel和分の途中式も示した。基底拡張、Cauchy–Schwarzなど有限次元線形代数の基本事項は使用する。

## 8. TT/MPS単一ボンドtruncationへの翻訳

### 8.1 等長写像とノルム保存

mixed-canonical formの左右コアは等長写像である。

$$
H:=G_1^{[L]\langle R\rangle}\in\mathbb R^{n_1\times r_1},\quad H^TH=I_{r_1},
\qquad
R:=G_3^{[R]\langle L\rangle}\in\mathbb R^{r_2\times n_3},\quad RR^T=I_{r_2}.
$$

中心行列を任意の $B\in\mathbb R^{(r_1n_2)\times r_2}$ に差し替えたテンソルを $X^{(B)}$ とする。行要素を明示すると、$B=A$ で元の$X$に戻り、一般の$B$では

$$
X^{(B)}_{i_1i_2i_3}
=\sum_{\alpha_1=1}^{r_1}\sum_{\alpha_2=1}^{r_2}
H_{i_1\alpha_1}B_{(\alpha_1,i_2),\alpha_2}R_{\alpha_2 i_3}.
$$

切断行列の行を $(i_1,i_2)$、列を$i_3$とし、$T:=H\otimes I_{n_2}$、$T_{(i_1,i_2),(\alpha_1,j_2)}=H_{i_1\alpha_1}\delta_{i_2j_2}$ と定める。このとき

$$
\begin{aligned}
(TBR)_{(i_1,i_2),i_3}
&=\sum_{\alpha_1,j_2,\alpha_2}
T_{(i_1,i_2),(\alpha_1,j_2)}B_{(\alpha_1,j_2),\alpha_2}R_{\alpha_2i_3}\\
&=\sum_{\alpha_1,\alpha_2}H_{i_1\alpha_1}B_{(\alpha_1,i_2),\alpha_2}R_{\alpha_2i_3}
=X^{(B)}_{i_1i_2i_3},
\end{aligned}
$$

ゆえに $X^{(B)\langle2\rangle}=TBR$ である。$T^TT=(H^TH)\otimes I_{n_2}=I_{r_1n_2}$ であり、添字でも

$$
(T^TT)_{(\alpha_1,i_2),(\beta_1,j_2)}
=\sum_{i_1,\ell_2}H_{i_1\alpha_1}H_{i_1\beta_1}\delta_{\ell_2i_2}\delta_{\ell_2j_2}
=\delta_{\alpha_1\beta_1}\delta_{i_2j_2}.
$$

任意の等長写像 $Q$（$Q^TQ=I$）について

$$
\|QM\|_F^2=\operatorname{tr}(M^TQ^TQM)=\operatorname{tr}(M^TM)=\|M\|_F^2
$$

が成り立つので、これを両側から使うと

$$
\boxed{
\|X-X^{(B)}\|_F=\|T(A-B)R\|_F=\|A-B\|_F
}
$$

局所的な中心行列の差の**ノルム**が、そのまま全テンソルの差のノルムに等しい。右側も $\|MR\|_F^2=\operatorname{tr}(R^TM^TMR)=\operatorname{tr}(M^TMRR^T)=\|M\|_F^2$ であり、reshapeは行列要素の並べ替えにすぎない。

### 8.2 EYMの代入と最適性の結論

左右の環境 $H,R$ を固定して中心行列だけを変える候補では、切断行列 $TBR$ のrankは $\operatorname{rank}(B)$ と一致する。実際、$T^TT=I$、$RR^T=I$ なので $B=T^T(TBR)R^T$、両向きのrank不等式が成立する。よってbond rankを$r_2\to k$に制限する候補は $\operatorname{rank}(B)\le k$ を満たす全ての$B$。8.1の等式と7節のEYM定理から、

$$
\min_{\operatorname{rank}(B)\le k}\|X-X^{(B)}\|_F
=\min_{\operatorname{rank}(B)\le k}\|A-B\|_F
=\|A-A_k\|_F
=\sqrt{\sum_{\beta=k+1}^{\rho}\sigma_\beta^2}
$$

を達成するのがtruncated SVD由来の $\widetilde X_k$ である。Notebook 09の数値では

$$
\min_{\operatorname{rank}(B)\le2}\|X-X^{(B)}\|_F=\sigma_3=0.25
$$

であり、これは「たまたま$0.25$になった」のではなく、**bond rankを2に制限するあらゆる第2コアの選び方の中で、これより小さい誤差は存在しない**という意味を持つ。

さらにこの場合、**外側のコアも変更してよい全ての切断rank-$k$テンソル**まで最適性を強められる。$A=U\Sigma V^T$ なら

$$
X^{\langle2\rangle}=TAR=(TU)\Sigma(R^TV)^T,
\quad (TU)^T(TU)=I_\rho,
\quad(R^TV)^T(R^TV)=I_\rho.
$$

つまりこれは切断行列のreduced SVDで、$A$と$X^{\langle2\rangle}$ の非ゼロ特異値は同じ。$A_k$由来の $\widetilde X_k^{\langle2\rangle}=TA_kR$ に行列版EYMを適用すれば、任意の $Y$（$\operatorname{rank}(Y^{\langle2\rangle})\le k$）に対して

$$
\|X-Y\|_F=\|X^{\langle2\rangle}-Y^{\langle2\rangle}\|_F
\ge\sqrt{\sum_{\beta=k+1}^{\rho}\sigma_\beta^2}
=\|X-\widetilde X_k\|_F.
$$

したがって単一切断のrank制約については、左右コア固定という条件を外しても最適である。この結論は**その切断ひとつ**についてであり、複数bondを同時に打ち切るTT rounding全体の誤差評価とは区別する。

Schmidt channelの中身も添字で確認できる。$A=U\Sigma V^T$ とし、

$$
\Phi_\beta(i_1,i_2)
:=\sum_{\alpha_1=1}^{r_1}H_{i_1\alpha_1}U_{(\alpha_1,i_2),\beta}
=(TU)_{(i_1,i_2),\beta},
\qquad
\Psi_\beta(i_3)
:=\sum_{\alpha_2=1}^{r_2}V_{\alpha_2\beta}R_{\alpha_2i_3}
=(R^TV)_{i_3,\beta}
$$

と置けば、$X_{i_1i_2i_3}=\sum_{\beta=1}^{\rho}\sigma_\beta\Phi_\beta(i_1,i_2)\Psi_\beta(i_3)$。$\sum_{i_1,i_2}\Phi_\beta\Phi_\gamma=(U^TT^TTU)_{\beta\gamma}=\delta_{\beta\gamma}$、$\sum_{i_3}\Psi_\beta\Psi_\gamma=(V^TRR^TV)_{\beta\gamma}=\delta_{\beta\gamma}$ で、左右とも正規直交する。従って

$$
\widetilde X_{k,i_1i_2i_3}=\sum_{\beta=1}^{k}\sigma_\beta\Phi_\beta(i_1,i_2)\Psi_\beta(i_3),
\qquad
(X-\widetilde X_k)_{i_1i_2i_3}=\sum_{\beta=k+1}^{\rho}\sigma_\beta\Phi_\beta(i_1,i_2)\Psi_\beta(i_3)
$$

はそれぞれ保持・破棄したSchmidt成分で、互いのFrobenius内積は0。これがNotebook 09で確かめたPythagoras等式と捨てた特異値の二乗和の行要素レベルの理由である。exact SVDで全て保持すれば$X$は不変、truncated SVDで$\beta>k$を捨てれば$\widetilde X_k\ne X$ になる。

### 8.3 Notebook 09の圧縮コア構成との対応

truncated SVDを実際にコアへ吸収する際は、$A_k=U_k\Sigma_kV_k^T$ を

$$
\widetilde G_2^{[L]}(\alpha_1,i_2,\beta)=(U_k)_{(\alpha_1,i_2),\beta},
\qquad
\widetilde G_3^{[C]}(\beta,i_3,1)=\sum_{\alpha_2=1}^{r_2}(\Sigma_kV_k^T)_{\beta,\alpha_2}G_3^{[R]}(\alpha_2,i_3,1),
$$

のように分ける。コア全体のshapeは $\widetilde G_2\in\mathbb R^{r_1\times n_2\times k}$、$\widetilde G_3\in\mathbb R^{k\times n_3\times1}$。$\Sigma_kV_k^T$を右側の環境コアへ吸収して内部bondを$r_2\to k$とする点に注意。8.1の $X^{(B)}=G_1^{[L]}\,\operatorname{reshape}(B)\,G_3^{[R]}$ は**最適性を比較するための固定環境での表現**であり、実際に保存する圧縮コアは上式で作る。$B=A_k$ と置けば両表現は同じテンソルになる。この構成自体はNotebook 09で既に実施済みだが、今回の証明との対応（8.2節）は今回初めて明示した。

$\Sigma_k$ を左側に置き、$\operatorname{reshape}(U_k\Sigma_k)$ と $V_k^TR$ に分けても積は同じ $TA_kR$ となる。ただしこの配置では新しい第2コアの左展開について $(U_k\Sigma_k)^T(U_k\Sigma_k)=\Sigma_k^2$ となり、左直交条件 $I_k$ は一般に成立しない。Notebook 09の $U_k$ と $\Sigma_kV_k^T$ という配置は、左正準性を保つ選び方でもある（ゲージの違いは [[36_TT_MPSのGauge自由度と左QR直交化]] を参照）。

## 9. 手を動かして確かめる実験計画とNotebook 10

添付資料が提案した実験は以下の3つである。現在のリポジトリには[Notebook 10](../../../../notebooks/30_tt_mps/00_fundamentals/10_eckart_young_mirsky_and_single_bond_optimality.ipynb)があり、A・B・Cの演習セルと`TODO`が用意されている。このノートでは演習セルを実行していないため、実験の数値結果としては扱わない。

| 実験 | 確認する内容 | 使用ツール |
|---|---|---|
| A | ランダムなrank-2行列 $B_j=PQ^T$（$P\in\mathbb R^{6\times2}$、$Q\in\mathbb R^{3\times2}$）を多数生成し、$\|A-B_j\|_F\ge\|A-A_2\|_F$ が常に成り立つことを数値確認する | `torch.linalg.svd(A, full_matrices=False)` |
| B | von Neumannのトレース不等式 $\operatorname{tr}(A^TB)\le\sum_i\sigma_i(A)\sigma_i(B)$ をランダム行列で数値検証する | `torch.linalg.svdvals` |
| C | mixed-canonical TTで $\|X-\widetilde X_k\|_F=\|A-A_k\|_F$ が、$H^TH=I$、$RR^T=I$ を明示的に検査した上で一致することを確認する | テンソル縮約（`einsum`等） |

実験Aでは、浮動小数点誤差を考慮し、判定条件を

$$
\min_j\|A-B_j\|_F\ge\|A-A_2\|_F-\varepsilon,\qquad\varepsilon=10^{-12}\text{（float64の場合）}
$$

のように緩めるとよい。なお、ランダム探索によって最適解そのものを発見することが目的ではなく、「rank-2候補をいろいろ試してもEYMの下界を破れない」ことを確認するのが実験Aの趣旨である。

## 10. 議論の経緯と訂正履歴

このノートの内容は、以下の誤りと訂正を経て確定した。

- 当初、「$\mathcal N(B)\cap\mathcal V_{k+1:\rho}$から$\rho-k$本の$z$を同時に取れる」と示唆したが、交差次元の保証は $\max(0,\rho-2k)$ にとどまり誤りだった（5.3節参照）。
- kernel法（4節）は $\rho-k=1$ の特殊ケースでは完全な証明を与えるが、一般の $\rho-k\ge2$ には拡張できないことを、次元カウントの限界として5節で明示した。
- 最初にkernel証明を述べた際、$k<\rho\le n$ という前提を明示していなかった（4.4節）。Notebook 09の数値がちょうど $k+1=n=\rho$ という境界的な場合だったため、この前提の必要性が見えにくかった。
- von Neumannのトレース不等式の最初の試みは、二重劣確率行列を**1行1列だけ**足して二重確率行列へ埋め込む箇所が一般には成り立たないため撤回した。成立する成分展開と反例は6.3節に記録し、Ky Fanの最大値原理とAbel和分による経路（6.2節）に切り替えた。
- von Neumannのトレース不等式の証明（6.2節）は、Ky Fanの最大値原理を**引用のみ**で使用しており、完全に基礎から積み上げた証明ではない。Ky Fanの原理自体を証明したい場合は別ノートが必要。
- 「なぜ小さい特異値を捨てるのが最適なのか」という直感（2節：成分ごとのFrobeniusノルムの大小比較）と、「あらゆるrank-$k$候補の中での最適性」（EYM定理本体）は別レベルの主張であり、前者だけでは後者の証明にならないことを区別した。
- 「なぜ下界が支えられなかったか」について、単に次元カウントが不足するというだけでなく、$B$が自分のkernel内の係数配分を操作して小さい特異値方向の寄与をゼロにできる、というアドバーサリー的な具体的メカニズム（5.4節）を追加で確認した。

## 11. 今後の展望

このノートで扱ったのはEYM定理の一般証明と、TT/MPSの**単一bond**への翻訳、および実験内容の設計までである。次に進む場合の候補は以下（本ノートの範囲外）。

- Notebook 10の実験A・B・Cの`TODO`を埋めて検証すること
- tolerance／relative errorによるrankの自動決定
- 複数bondにまたがるTT rounding（sweep）の誤差評価
- entanglement entropyとSchmidt係数の物理的対応
- Ky Fanの最大値原理自体の証明
- 6.4節で触れたWeyl型不等式ルートによる別証明

## 12. 検算例と資料上の実施状況

添付資料で実測値が記録されているのは、4.3節のNotebook 09（$A\in\mathbb R^{6\times3}$、$\rho=3$、$k=2$）である。$2\times2$、特異値 $(2,1)$ の数値検算がNotebook 09で実施済みという記述は添付資料から確認できないため、実施済みとは扱わない。$\rho-k\ge2$ の場合は、例えば特異値 $(6,4,2,1)$ の $4\times4$ 行列を $k=2$ で打ち切ると、理論値は $\|A-A_2\|_F=\sqrt{2^2+1^2}=\sqrt5$、$\|A-A_2\|_2=2$ と**手計算で予測**できる。これに対する任意のrank-2候補の数値比較は、添付資料では今後の実験として提案され、現リポジトリのNotebook 10には演習として配置されている。

## 関連ノート

- SVDの定義と行列のshape：[[01_SVDとは]]、[[02_SVD数式の導出]]
- 低rank近似と誤差評価：[[03_SVDによる低ランク近似]]、[[06_誤差評価_数式導出補足]]
- 切断行列とTT-rank：[[31_TT-rankとunfolding]]、[[33_TT-SVDの打ち切りと誤差]]
- 等長写像、mixed-canonical form、中心ノルム：[[35_TT_MPSの等長写像と射影]]、[[39_TT_MPSの混合正準形への導入]]、[[40_TT_MPSの中心ノルムと内積の導出]]
- SVDでのSchmidt形と単一ボンド打ち切り：[[43_TT_MPSのSVD中心移動とSchmidt形]]、[[45_TT_MPSの単一ボンドSVD打ち切り]]

## 添付資料で参照された資料

- EYM定理と行列ノルム：[SJSU講義資料](https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec7matrixnorm.pdf)、[京都大学Yukawa Institute資料](https://www2.yukawa.kyoto-u.ac.jp/~ken.shiozaki/doc_EN/Eckart_Young_EN.pdf)
- 一般のユニタリ不変ノルムまでのSchmidt–Mirsky定理：[Ballani–Kressner（EPFL）](https://sma.epfl.ch/~anchpcommon/publications/cime.pdf)。添付資料では別の[EPFL講義資料](https://sma.epfl.ch/~anchpcommon/lecture1.pdf)も参照している。
- PyTorchのreduced SVD：[`torch.linalg.svd` 公式資料](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)。実数入力なら返り値 `U, S, Vh` の `Vh` が数式の $V^T$ に対応する。SVDは縮退特異値のもとで特異ベクトルが一意とは限らないため、因子の成分一致より再構成値と誤差を検証する。
