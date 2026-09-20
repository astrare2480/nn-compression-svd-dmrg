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

[[45_TT_MPSのSVD打ち切り]]（09_truncated_svd_and_bond_rank_truncation.ipynb）で、3階TT/MPSの第2サイトを直交中心とし、中心行列 $A=G_2^{[C]\langle L\rangle}\in\mathbb R^{(r_1n_2)\times r_2}$ を特異値 $(\sigma_1,\sigma_2,\sigma_3)=(6,2,0.25)$、$k=2$ で打ち切り、誤差

$$
\|X-\widetilde X_k\|_F=\sigma_3=0.25,
\qquad
\|X-\widetilde X_k\|_F^2=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2=0.0625
$$

を数値検証した。このノートでは、**この打ち切りがすべてのrank-$k$候補の中で本当に最良なのか**という問いに答えるEckart–Young–Mirsky（EYM）定理を、直感的な比較 → $\rho-k=1$の特殊ケースの証明 → 一般の$\rho-k\ge2$への拡張の失敗と成功、まで、途中の誤った試みや疑問点も含めて追う。

## 1. 定理の主張

$A\in\mathbb R^{m\times n}$、$\rho=\operatorname{rank}(A)$、$0\le k<\rho\le n$ とする（$k<\rho$は「まだ捨てる特異値が少なくとも1個ある」ために必要、$\rho\le n$は$A$のshapeから自動的に成立）。reduced SVDを

$$
A=U\Sigma V^T=\sum_{\alpha=1}^{\rho}\sigma_\alpha u_\alpha v_\alpha^T,
\qquad
\sigma_1\ge\cdots\ge\sigma_\rho>0
$$

とし、truncated SVDを $A_k=\sum_{\alpha=1}^{k}\sigma_\alpha u_\alpha v_\alpha^T$ と定義する。ここで $B$ は、$A$の特異ベクトルとは無関係な**任意**の行列であり、制約は $\operatorname{rank}(B)\le k$ のみである（ランダムな低rank行列、特異ベクトルを回転させた行列、零行列なども候補に含まれる）。この任意の $B\in\mathbb R^{m\times n}$、$\operatorname{rank}(B)\le k$ に対して、

$$
\boxed{
\|A-B\|_F^2\ge\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2
}
$$

が成り立ち、$A_k$がこの下界を達成する。すなわち

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

$\rho-k=1$（捨てる特異値が1個）のときのみ、両ノルムの最小値は $\sigma_{k+1}$ に一致する。一般には

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

$E_\alpha:=u_\alpha v_\alpha^T$ とおく。$u_\alpha,v_\alpha$ の正規直交性から

$$
\langle E_\alpha,E_\beta\rangle_F
=\operatorname{tr}\!\left[(u_\alpha v_\alpha^T)^T(u_\beta v_\beta^T)\right]
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

任意の行列 $M$ と単位ベクトル $z$ について $\|Mz\|_2\le\|M\|_2\le\|M\|_F$（作用素ノルムの定義：$\|M\|_2=\max_{\|x\|_2=1}\|Mx\|_2$ より特定の$z$での値は最大値を超えない。また特異値 $\tau_i$ を用いて $\|M\|_2=\tau_1$、$\|M\|_F^2=\sum_i\tau_i^2\ge\tau_1^2$）が成り立つので、$M=A-B$ として

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

### 4.4 補足：この証明で見落としていた前提

証明を最初に述べた際、$k<\rho\le n$ という前提を明示していなかった。$\dim\mathcal V_{k+1}=k+1$ を主張するには $v_{k+1}$（すなわち $k+1\le\rho$）が存在する必要があり、また $\mathcal V_{k+1}\subseteq\mathbb R^n$ が意味を持つには $k+1\le n$ も必要である。Notebook 09の $\rho=3,\,k=2,\,n=3$ では $k+1=n=\rho=3$ というちょうど境界的な場合になっており、$\mathcal V_{k+1}=\mathbb R^n$ 全体に一致する特殊な状況だった。一般の $k+1<n$ のケース（例えば $n=10,\,k=2$）で確認するとこの次元カウントの非自明さがよりはっきり見える。

## 5. 一般 $\rho-k\ge2$ への拡張の限界

### 5.1 目標と手持ちの下界のギャップ

$q:=\rho-k$ とする。目標は

$$
\|A-B\|_F^2\ge\sigma_{k+1}^2+\sigma_{k+2}^2+\cdots+\sigma_\rho^2 \tag{Goal}
$$

（$q$個の項の和）だが、4.2と同様の議論（$\mathcal V_{k+1}=\operatorname{span}\{v_1,\ldots,v_{k+1}\}$、$\dim=k+1$ との交差）から得られるのは

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

が成り立つ（列の直交射影はノルムを増やさない）。この発想自体は正しい方向である。

kernelの中だけなら、$\dim\mathcal N(B)\ge n-k\ge\rho-k=q$（$\rho\le n$より）なので、正規直交な$q$本を確かに取れる。しかし**問題はその$q$本が$A$の残り特異方向を捉える保証がないこと**である。

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

検討の過程で「$\mathcal N(B)\cap\mathcal V_{k+1:\rho}$から$\rho-2k$本取れる」と述べたが、これは誤りだった。実際の次元下界は $\max(0,\rho-2k)$ であり、$k$が大きい場合には0にしかならない。**次元カウントだけでは、複数の特異値を同時に強制する道具にならない**ことが、この試行錯誤で判明した本質的な限界である。

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

$\sigma_4$の寄与は係数がゼロなので現れない。$B$にとって都合がいいのは、なるべく小さい特異値方向（$\sigma_4$）に重みを置かず、少しでも大きい$\sigma_3$方向に重みを集中させることであり、これにより証明する側が要求する下界を最小限（$\sigma_3^2$ちょうど）に抑えられる。

これが「複数の特異値を同時に強制できない」ことの最も具体的な理由である。もう1本 $z_2$（$v_4$方向）を別に用意しようとしても、$B$のkernelがちょうど1次元しかない場合（4.2の下限どおり）、$z_1$と直交する都合のよいベクトルが$\mathcal N(B)$内に存在する保証はない。$B$の形は相手（最悪ケース）が決めるので、そのようなベクトルが存在しない$B$を相手が選べてしまう。

### 5.5 kernel法の適用範囲のまとめ

| 項目 | kernel法（4節）で示せること |
|---|---|
| $\|A-B\|_2\ge\sigma_{k+1}$（作用素ノルム版） | 任意の$k$で成立 |
| $\|A-B\|_F\ge\sigma_{k+1}$ | 任意の$k$で成立（$\|\cdot\|_F\ge\|\cdot\|_2$より） |
| $\|A-B\|_F^2\ge\sum_{\alpha=k+1}^{\rho}\sigma_\alpha^2$ | $\rho-k=1$のときのみ十分。$\rho-k\ge2$では不足 |

一般のFrobenius版には、個々の方向を追う方法ではなく、行列全体の特異値の並びを同時に比較する道具が必要になる。

## 6. von Neumannのトレース不等式

**主張**：$A,B\in\mathbb R^{n\times n}$（一般の$m\times n$も同様）の特異値を $\sigma_1(A)\ge\cdots\ge\sigma_n(A)\ge0$、$\sigma_1(B)\ge\cdots\ge\sigma_n(B)\ge0$ とすると、

$$
\boxed{
\operatorname{tr}(A^TB)\le\sum_{i=1}^{n}\sigma_i(A)\sigma_i(B)
}
$$

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
\operatorname{tr}(A^TS_k)=\operatorname{tr}(A^TP_kQ_k^T)=\operatorname{tr}(P_k^TAQ_k)
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

### 6.3 別ルートの存在について（比較のみ、本ノートでは未使用）

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

第1項が0になるのは $\sigma_i(B)=\sigma_i(A)\;(i\le k)$、すなわち $B=A_k$ のとき。このとき von Neumannの不等式も等号となり、3節の上界と完全に一致する。証明の依存関係は

$$
\text{Ky Fanの原理（引用）}
\to
\text{von Neumannのトレース不等式（6節）}
\to
\text{Mirskyの不等式（7.1）}
\to
\text{一般Frobenius版EYM（7.2）}
$$

Ky Fanの原理のみ引用であり、それ以外はすべて自己完結して導出している。

## 8. TT/MPS単一ボンドtruncationへの翻訳

### 8.1 等長写像とノルム保存

mixed-canonical formの左右コアは等長写像である。

$$
H:=G_1^{[L]\langle R\rangle}\in\mathbb R^{n_1\times r_1},\quad H^TH=I_{r_1},
\qquad
R:=G_3^{[R]\langle L\rangle}\in\mathbb R^{r_2\times n_3},\quad RR^T=I_{r_2}.
$$

中心行列を任意の $B\in\mathbb R^{(r_1n_2)\times r_2}$ に差し替えたテンソルを $X^{(B)}$、対応する切断行列を $X^{(B)\langle2\rangle}=TBR$（$T=H\otimes I_{n_2}$、$T^TT=I_{r_1n_2}$）とする。任意の等長写像 $Q$（$Q^TQ=I$）について

$$
\|QM\|_F^2=\operatorname{tr}(M^TQ^TQM)=\operatorname{tr}(M^TM)=\|M\|_F^2
$$

が成り立つので、これを両側から使うと

$$
\boxed{
\|X-X^{(B)}\|_F=\|T(A-B)R\|_F=\|A-B\|_F
}
$$

局所的な中心行列の差が、そのまま全テンソルの差に等しい。

### 8.2 EYMの代入と最適性の結論

bond rankを$r_2\to k$に制限する候補は $\operatorname{rank}(B)\le k$ を満たす全ての$B$。8.1の等式と7節のEYM定理から、

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

### 8.3 実装として構成する量（未実施、次の課題）

truncated SVDを実際にコアへ吸収する際は、$A_k=U_k\Sigma_kV_k^T$ を

$$
\widetilde G_2^{[L]}(\alpha_1,i_2,\beta)=(U_k)_{(\alpha_1,i_2),\beta}\in\mathbb R^{r_1\times n_2\times k},
\qquad
\widetilde G_3^{[C]}(\beta,i_3,1)=\sum_{\alpha_2}(\Sigma_kV_k^T)_{\beta,\alpha_2}G_3^{[R]}(\alpha_2,i_3,1)\in\mathbb R^{k\times n_3\times1}
$$

のように分ける（単純に中心行列を差し替えるのではなく、$\Sigma_kV_k^T$を右側の環境コアへ吸収する点に注意）。この構成自体はNotebook 09で既に実施済みだが、今回の証明との対応（8.2節）は今回初めて明示した。

## 9. 手を動かして確かめる実験計画（未実施）

理論の理解を検証するための実装として、以下3つを次回以降に実施する。

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

- 当初、「$\mathcal N(B)\cap\mathcal V_{k+1:\rho}$から$\rho-k$本の$z$を同時に取れる」と述べたが、次元下界は $\max(0,\rho-2k)$ にしかならず誤りだった（5.3節参照）。
- kernel法（4節）は $\rho-k=1$ の特殊ケースでは完全な証明を与えるが、一般の $\rho-k\ge2$ には拡張できないことを、次元カウントの限界として5節で明示した。
- 最初にkernel証明を述べた際、$k<\rho\le n$ という前提を明示していなかった（4.4節）。Notebook 09の数値がちょうど $k+1=n=\rho$ という境界的な場合だったため、この前提の必要性が見えにくかった。
- von Neumannのトレース不等式の証明の最初の試み（$W_1,W_2$を用いた二重直交行列の縮約によるアプローチ）は複雑になりすぎて撤回し、Ky Fanの最大値原理とAbel和分による、より簡潔な経路（6.2節）に切り替えた。
- von Neumannのトレース不等式の証明（6.2節）は、Ky Fanの最大値原理を**引用のみ**で使用しており、完全に基礎から積み上げた証明ではない。Ky Fanの原理自体を証明したい場合は別ノートが必要。
- 「なぜ小さい特異値を捨てるのが最適なのか」という直感（2節：成分ごとのFrobeniusノルムの大小比較）と、「あらゆるrank-$k$候補の中での最適性」（EYM定理本体）は別レベルの主張であり、前者だけでは後者の証明にならないことを区別した。
- 「なぜ下界が支えられなかったか」について、単に次元カウントが不足するというだけでなく、$B$が自分のkernel内の係数配分を操作して小さい特異値方向の寄与をゼロにできる、というアドバーサリー的な具体的メカニズム（5.4節）を追加で確認した。

## 11. 今後の展望

このノートで完了したのはEYM定理の一般証明と、TT/MPSの**単一bond**への翻訳、および実装計画の設計までである。次に進む場合の候補は以下（本ノートの範囲外）。

- 9節の実験A・B・Cの実際のPyTorch実装
- tolerance／relative errorによるrankの自動決定
- 複数bondにまたがるTT rounding（sweep）の誤差評価
- entanglement entropyとSchmidt係数の物理的対応
- Ky Fanの最大値原理自体の証明
- 6.3節で触れたWeyl型不等式ルートによる別証明

## 12. 検算例（最小例、未実施）

$A\in\mathbb R^{2\times2}$、$\sigma_1=2,\ \sigma_2=1$、$k=1$ とする単純な数値検算はNotebook側で実施済み（$\rho-k=1$のケースはNotebook 09と同型）。$\rho-k\ge2$の数値検証（例えば$\rho=4,\,k=2$でランダムなrank-2行列と比較する実験、9節の実験A）は未実施であり、次のノート・チャットでのPyTorch実装課題として残す。
