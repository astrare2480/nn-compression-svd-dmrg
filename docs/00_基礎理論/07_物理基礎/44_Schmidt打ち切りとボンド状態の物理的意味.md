---
title: Schmidt打ち切りとボンド状態の物理的意味
tags:
  - MPS
  - Schmidt
  - truncation
---

# Schmidt打ち切りとボンド状態の物理的意味

[[43_TT_MPSのSVD中心移動とSchmidt形]] のexact SVDは状態を変えず、切断のSchmidt係数を露出させる。ここでは、非零係数を実際に捨てると何が変わるかを、左ブロック $A=(i_1,i_2)$ と右ブロック $B=i_3$ の一つの切断で考える。コアshape・行要素・単一cut誤差の導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/45_TT_MPSの単一ボンドSVD打ち切り]]、複数cutのTT-SVD誤差は [[33_TT-SVDの打ち切りと誤差]] に置く。

## 1. 何を一本のボンド状態と呼ぶか

規格化された純粋状態のSchmidt分解を

$$
|\psi\rangle
=\sum_{\beta=1}^{\rho}\sigma_\beta
|L_\beta\rangle_A\otimes|R_\beta\rangle_B,
\qquad
\sum_{\beta=1}^{\rho}\sigma_\beta^2=1
$$

とする。左右の状態群はそれぞれ正規直交し、$\rho$ は非零係数の本数である。$\beta$ は単独の左サイトや右サイトの配置ではなく、切断をまたぐ対応づけられた一組のブロック状態を数える。係数を $\sigma_\beta$ から0へ変えると、その組の項全体が波動関数から消える。

$$
|\widetilde\psi_k\rangle
=\sum_{\beta=1}^{k}\sigma_\beta
|L_\beta\rangle_A\otimes|R_\beta\rangle_B,
\qquad k<\rho.
$$

$\sigma_{k+1}>0$ だから $|\widetilde\psi_k\rangle\ne|\psi\rangle$ である。exact SVDによる中心移動と異なり、これは単なるコア表示の変更ではない。ボンド次元を $\rho$ から $k$ へ下げると、保持できるSchmidtチャネルが $k$ 本になる。

## 2. 左ブロックへの射影を添字で確認する

残す左状態が張る部分空間への直交射影を

$$
P_k=\sum_{\gamma=1}^{k}|L_\gamma\rangle\langle L_\gamma|
$$

と定義する。右側には恒等写像を作用させる。途中の内積を省略せずに計算すると

$$
\begin{aligned}
(P_k\otimes I_B)|\psi\rangle
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}
\sigma_\beta
|L_\gamma\rangle
\langle L_\gamma|L_\beta\rangle
\otimes|R_\beta\rangle\\
&=\sum_{\gamma=1}^{k}\sum_{\beta=1}^{\rho}
\sigma_\beta\delta_{\gamma\beta}
|L_\gamma\rangle\otimes|R_\beta\rangle\\
&=|\widetilde\psi_k\rangle.
\end{aligned}
$$

落としたSchmidt状態は射影先の空間にない。射影後の状態を規格化し直しても、それらの成分は戻らない。

## 3. 特異値の二乗は何の重みか

右ブロックをトレースアウトした縮約密度行列は、右状態の正規直交性を用いて

$$
\begin{aligned}
\rho_A
&=\operatorname{Tr}_B(|\psi\rangle\langle\psi|)\\
&=\sum_{\beta,\gamma}\sigma_\beta\sigma_\gamma
|L_\beta\rangle\langle L_\gamma|
\underbrace{\langle R_\gamma|R_\beta\rangle}_{\delta_{\gamma\beta}}\\
&=\sum_{\beta=1}^{\rho}\sigma_\beta^2
|L_\beta\rangle\langle L_\beta|.
\end{aligned}
$$

従って $\rho_A|L_\beta\rangle=\sigma_\beta^2|L_\beta\rangle$ である。$\sigma_\beta^2$ は全状態のノルム二乗に対する直交成分の重みであり、左Schmidt状態の縮約密度行列における重みでもある。ただし、これを「任意の物理現象をその割合で再現できる」という保証とは読まない。

捨てた重みを

$$
w_{\mathrm{discard}}:=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2
$$

と書く。直交性から、単一cutで一回だけSchmidt打ち切りをした非規格化状態について

$$
\begin{aligned}
\|\psi-\widetilde\psi_k\|^2
&=\left\|\sum_{\beta=k+1}^{\rho}\sigma_\beta
|L_\beta\rangle\otimes|R_\beta\rangle\right\|^2\\
&=\sum_{\beta,\gamma>k}\sigma_\beta\sigma_\gamma
\langle L_\gamma|L_\beta\rangle
\langle R_\gamma|R_\beta\rangle\\
&=\sum_{\beta=k+1}^{\rho}\sigma_\beta^2
=w_{\mathrm{discard}}.
\end{aligned}
$$

また $\|\widetilde\psi_k\|^2=1-w_{\mathrm{discard}}$ である。再規格化した状態は $|\widetilde\psi_k^{\mathrm{norm}}\rangle=|\widetilde\psi_k\rangle/\sqrt{1-w_{\mathrm{discard}}}$ で、元状態との二乗距離は一般に $w_{\mathrm{discard}}$ とは異なる。この等式を、複数cutを順に打ち切るTT-SVD全体へそのまま流用しない。

## 4. 二つのスピン状態を全要素で見る

基底順を $|\uparrow\uparrow\rangle,|\uparrow\downarrow\rangle,|\downarrow\uparrow\rangle,|\downarrow\downarrow\rangle$ とする。具体例として

$$
|\psi\rangle
=\sqrt{0.99}|\uparrow\downarrow\rangle
+\sqrt{0.01}|\downarrow\uparrow\rangle
$$

を取る。係数行列と状態ベクトルの全要素は

$$
C=\begin{pmatrix}0&\sqrt{0.99}\\\sqrt{0.01}&0\end{pmatrix},
\qquad
[\psi]=\begin{pmatrix}0\\\sqrt{0.99}\\\sqrt{0.01}\\0\end{pmatrix}.
$$

右側を消去すると

$$
\rho_A=CC^\dagger
=\begin{pmatrix}0.99&0\\0&0.01\end{pmatrix}.
$$

rank 1に打ち切ると

$$
[\widetilde\psi_1]
=\begin{pmatrix}0\\\sqrt{0.99}\\0\\0\end{pmatrix},
\qquad
[\psi-\widetilde\psi_1]
=\begin{pmatrix}0\\0\\\sqrt{0.01}\\0\end{pmatrix},
\qquad
\|\psi-\widetilde\psi_1\|^2=0.01.
$$

再規格化後は $|\uparrow\downarrow\rangle$ という積状態になる。元の状態にあった $|\downarrow\uparrow\rangle$ との相関した組はなくなり、この切断のSchmidt rankは2から1へ変わる。「小さいから物理的に無意味」ではなく、**この状態のこの切断でノルム二乗の重みが小さい**という判断である。多数の係数が無視できない状態では、小さなボンド次元で重要な相関を失い得る。

## 5. Schmidt状態と有効Hilbert空間

一般の二部状態を $|a_m\rangle_A\otimes|b_n\rangle_B$ の基底で書くと、係数は $c_{mn}$ という二つの添字を持ち、左右のすべての組が現れ得る。Schmidt分解では左右をそれぞれ正規直交基底へ回転し、対角係数だけを残した

$$
|\psi\rangle=\sum_{\beta=1}^{\rho}
\sigma_\beta|L_\beta\rangle_A\otimes|R_\beta\rangle_B,
\qquad
\langle L_\beta|L_\gamma\rangle
=\langle R_\beta|R_\gamma\rangle
=\delta_{\beta\gamma}
$$

を得る。$|L_\beta\rangle_A$ と $|R_\beta\rangle_B$ が左右の**Schmidt状態**であり、$\beta$ は一つのサイトの配置ではなく、左右のブロック状態を結ぶラベルである。例えば左ブロックが二つのスピン1/2なら $\dim\mathcal H_A=2^2=4$ だが、波動関数に実際に現れる左Schmidt状態の本数 $\rho$ は4以下でよい。

打ち切り前後の有効部分空間は

$$
\begin{aligned}
\mathcal H_{A,\mathrm{eff}}
&=\operatorname{span}\{|L_1\rangle,\ldots,|L_\rho\rangle\},\\
\widetilde{\mathcal H}_{A,\mathrm{eff}}
&=\operatorname{span}\{|L_1\rangle,\ldots,|L_k\rangle\},
\qquad \dim\widetilde{\mathcal H}_{A,\mathrm{eff}}=k,
\end{aligned}
$$

であり、右側でも対応する $k$ 本を残す。元のHilbert空間から状態が物理法則上「存在しなくなる」わけではない。**近似MPSが採用する有効部分空間から、元の波動関数にあった方向を除外する**という意味である。DMRGのブロック基底という言葉も、この有効状態の選択として読む。

## 6. 左右の縮約密度行列まで展開する

第3節では左の縮約密度行列を求めた。右側も同じ添字で確かめると

$$
\begin{aligned}
\rho_B
&=\operatorname{Tr}_A(|\psi\rangle\langle\psi|)\\
&=\sum_{\beta,\gamma}
\sigma_\beta\sigma_\gamma
\underbrace{\langle L_\gamma|L_\beta\rangle}_{\delta_{\gamma\beta}}
|R_\beta\rangle\langle R_\gamma|\\
&=\sum_{\beta=1}^{\rho}\sigma_\beta^2
|R_\beta\rangle\langle R_\beta|.
\end{aligned}
$$

従って、非零固有値は左右とも $\sigma_\beta^2$ である。左Schmidt状態は $\rho_A$ の、右Schmidt状態は $\rho_B$ の固有状態となる。規格化した全状態で $p_\beta=\sigma_\beta^2$ と言う場合、それは**Schmidt基底への射影の重み**である。任意の局所観測量が同じ割合で保存されるという主張ではない。

第4節の二スピン例を右基底 $|\uparrow\rangle,|\downarrow\rangle$ の順に計算すれば

$$
\rho_B=C^\dagger C
=\begin{pmatrix}0.01&0\\0&0.99\end{pmatrix}.
$$

左では $|\uparrow\rangle$ が重み $0.99$、右では対応する $|\downarrow\rangle$ が同じ重み $0.99$ を持つ。もう一組は左 $|\downarrow\rangle$ と右 $|\uparrow\rangle$ の重み $0.01$ である。rank 1打ち切り後の非規格化密度行列は

$$
\widetilde\rho_A
=\begin{pmatrix}0.99&0\\0&0\end{pmatrix},
\qquad
\operatorname{Tr}\widetilde\rho_A=0.99,
$$

であり、再規格化後は $\operatorname{diag}(1,0)$ になる。再規格化は残った係数を拡大するだけで、落とした固有状態を復元しない。

## 7. 捨てた重みと再規格化後の距離を区別する

四つのSchmidt状態ペアの重みが

$$
(\sigma_1^2,\sigma_2^2,\sigma_3^2,\sigma_4^2)
=(0.70,0.25,0.04,0.01)
$$

なら合計は1で、$k=2$ の捨てた重みは $w_{\mathrm{discard}}=0.04+0.01=0.05$。残した非規格化状態は

$$
|\widetilde\psi_2\rangle
=\sigma_1|L_1\rangle|R_1\rangle
+\sigma_2|L_2\rangle|R_2\rangle,
\qquad
\|\widetilde\psi_2\|^2=0.70+0.25=0.95.
$$

この段階で $\|\psi-\widetilde\psi_2\|^2=0.05$ である。再規格化した $|\widehat\psi_2\rangle=|\widetilde\psi_2\rangle/\sqrt{0.95}$ に対しては、直交性から $\langle\psi|\widetilde\psi_2\rangle=0.95$ なので

$$
\begin{aligned}
\langle\psi|\widehat\psi_2\rangle&=\sqrt{0.95},\\
\|\psi-\widehat\psi_2\|^2
&=\|\psi\|^2+\|\widehat\psi_2\|^2
-2\operatorname{Re}\langle\psi|\widehat\psi_2\rangle\\
&=2-2\sqrt{0.95}\ne0.05.
\end{aligned}
$$

一般に、規格化された元状態で $w=w_{\mathrm{discard}}<1$ なら、再規格化後の二乗距離は $2-2\sqrt{1-w}$。discarded weightの値を、この距離と取り違えない。また「95%の重みを保持」は、元状態のノルム二乗の95%が残すSchmidt部分空間へ射影された意味であり、すべての物理現象が95%再現できるという保証ではない。

量子状態ではこの切断をまたぐ弱い相関やもつれの成分を削る。一般のTTテンソルなら、左右の添字ブロック間にある独立な結合モードを削ると読む。小さい重みでも観測したい物理量に効くことがあり、特に多くのSchmidt係数が無視できない場合には小さなbond dimensionが十分とは限らない。
