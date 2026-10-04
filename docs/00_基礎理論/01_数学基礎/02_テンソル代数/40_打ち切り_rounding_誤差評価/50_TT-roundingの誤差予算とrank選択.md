---
title: TT-roundingの誤差予算とrank選択
tags:
  - TT
  - MPS
  - TT-rounding
  - rank-selection
  - tolerance
  - truncation-error
  - Frobenius
  - numerical-analysis
---

# TT-roundingの誤差予算とrank選択

このノートでは、TT-roundingの全体相対誤差目標から、各bondの局所誤差予算と保持rankを決める。

アルゴリズムの2方向sweepは [[48_TT-roundingの定義と正準化sweep]]、局所実誤差が二乗和で全体誤差になる理由は [[49_TT-roundingの環境行列と誤差直交分解]]、PyTorchでの比較・ログ・検証は [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]] に分ける。

---

## 0. 最初に記号を固定する

入力TTが表すテンソルを $\mathcal X$、rounding後を $\widetilde{\mathcal X}$ とする。

$$
N
=
\|\mathcal X\|_F.
$$

このノートでは、次の量を混同しない。

- $\varepsilon$：ユーザーが指定する全体相対誤差上限。
- $T=\varepsilon N$：全体絶対誤差予算。
- $\delta_k$：第 $k$ bondへ事前に配る局所誤差上限。
- $e_k(r)$：保持rankを $r$ と仮定したとき、実際に捨てる局所尾部ノルム。
- $\widetilde r_k$：予算条件を満たすように採用した保持rank。
- $e_k=e_k(\widetilde r_k)$：採用rankで実際に生じた局所誤差。

順序は

$$
\varepsilon
\longrightarrow
N
\longrightarrow
T
\longrightarrow
\delta_k
\longrightarrow
\widetilde r_k
\longrightarrow
e_k
$$

である。特異値が先に $\delta_k$ を決めるのではない。

---

## 1. 全体相対誤差目標

非零テンソルに対して、

$$
\frac{
\|\mathcal X-\widetilde{\mathcal X}\|_F
}{
\|\mathcal X\|_F
}
\le
\varepsilon
$$

を目標とする。等価に、

$$
\boxed{
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\varepsilon N
}
$$

である。

例えば、

$$
N=20,
\qquad
\varepsilon=10^{-2}
$$

なら、全体で許す絶対誤差は

$$
T
=
\varepsilon N
=
0.01\cdot20
=
0.2
$$

である。

---

## 2. なぜ局所予算を二乗和で配るか

標準TT-roundingの実局所誤差は、exact arithmeticで

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\sum_{k=1}^{d-1}e_k^2
$$

と合成される。したがって、各段階で

$$
e_k
\le
\delta_k
$$

を満たし、事前予算が

$$
\sum_{k=1}^{d-1}
\delta_k^2
\le
T^2
$$

を満たせば、

$$
\begin{aligned}
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
&=
\sum_{k=1}^{d-1}e_k^2\\
&\le
\sum_{k=1}^{d-1}\delta_k^2\\
&\le
T^2\\
&=
\varepsilon^2N^2.
\end{aligned}
$$

両辺の平方根から、

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\varepsilon N
$$

が得られる。

---

## 3. 均等な局所予算

内部bondは $d-1$ 本である。すべてへ同じ予算

$$
\delta_1
=
\cdots
=
\delta_{d-1}
=
\delta
$$

を割り当てると、

$$
\sum_{k=1}^{d-1}
\delta_k^2
=
(d-1)\delta^2.
$$

これを全体予算 $T^2=\varepsilon^2N^2$ に合わせる。

$$
(d-1)\delta^2
=
\varepsilon^2N^2.
$$

両辺を $d-1$ で割ると、

$$
\delta^2
=
\frac{
\varepsilon^2N^2
}{
d-1
}.
$$

すべて非負なので平方根を取り、

$$
\boxed{
\delta_k
=
\frac{
\varepsilon\|\mathcal X\|_F
}{
\sqrt{d-1}
},
\qquad
k=1,\ldots,d-1
}
$$

となる。

### 3.1 数値例

$$
d=5,
\qquad
\varepsilon=10^{-2},
\qquad
N=20
$$

とする。bond数は

$$
d-1
=
4,
$$

全体絶対誤差予算は

$$
T
=
0.2.
$$

したがって、

$$
\delta_k
=
\frac{0.2}{\sqrt4}
=
0.1.
$$

各段階で $e_k\le0.1$ なら、

$$
\sum_{k=1}^{4}e_k^2
\le
4\cdot0.1^2
=
0.04,
$$

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\sqrt{0.04}
=
0.2.
$$

誤差を

$$
4\cdot0.1
=
0.4
$$

と単純加算するのではない。

---

## 4. 非均等な局所予算

均等配分は分かりやすい十分条件であり、唯一の配分ではない。

$$
w_k
\ge
0,
\qquad
\sum_{k=1}^{d-1}w_k^2
\le
1
$$

を満たす重みを選び、

$$
\delta_k
=
w_kT
$$

とすれば、

$$
\sum_{k=1}^{d-1}
\delta_k^2
=
T^2
\sum_{k=1}^{d-1}w_k^2
\le
T^2.
$$

bondごとのスペクトル減衰や計算コストを事前に知っているなら、非均等配分の方が効率的なことがある。しかし配分自体の最適化は、基本的なTT-roundingとは別問題である。

---

## 5. 第 $k$ 段階でSVDする行列

右から左へのQR sweep後、左から右へ進む。第 $k$ 段階の更新済みコアを

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

とする。局所行列は

$$
V_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1}n_k,r_k)
\right).
$$

economy SVDを

$$
V_k
=
U_k\Sigma_kW_k^T
$$

とする。特異値数は

$$
q_k
=
\min(r_{k-1}n_k,r_k),
$$

$$
\sigma_{k,1}
\ge
\sigma_{k,2}
\ge
\cdots
\ge
\sigma_{k,q_k}
\ge
0.
$$

---

## 6. rank $r$ を保持したときの実局所誤差

rank $r$ の打ち切りを

$$
V_k^{(r)}
=
\sum_{j=1}^{r}
\sigma_{k,j}u_{k,j}w_{k,j}^T
$$

とする。残差は

$$
E_k(r)
=
V_k-V_k^{(r)}
=
\sum_{j=r+1}^{q_k}
\sigma_{k,j}u_{k,j}w_{k,j}^T.
$$

異なるrank-one SVD成分はFrobenius内積で直交するため、

$$
\begin{aligned}
\|E_k(r)\|_F^2
&=
\left\langle
\sum_{a=r+1}^{q_k}
\sigma_{k,a}u_{k,a}w_{k,a}^T,
\sum_{b=r+1}^{q_k}
\sigma_{k,b}u_{k,b}w_{k,b}^T
\right\rangle_F\\
&=
\sum_{a=r+1}^{q_k}
\sum_{b=r+1}^{q_k}
\sigma_{k,a}\sigma_{k,b}
\langle u_{k,a},u_{k,b}\rangle
\langle w_{k,a},w_{k,b}\rangle\\
&=
\sum_{j=r+1}^{q_k}
\sigma_{k,j}^2.
\end{aligned}
$$

したがって、

$$
\boxed{
e_k(r)
=
\|E_k(r)\|_F
=
\left(
\sum_{j=r+1}^{q_k}
\sigma_{k,j}^2
\right)^{1/2}
}
$$

である。

$\sigma_{k,j}$ は1本のSVD成分の大きさであり、$e_k(r)$ は捨てた全成分を合成した誤差である。1個だけ捨てる場合に限り、$e_k(r)$ はその1個の特異値と等しい。

---

## 7. 保持rankの定義

局所予算を守る条件は

$$
e_k(r)^2
=
\sum_{j=r+1}^{q_k}
\sigma_{k,j}^2
\le
\delta_k^2.
$$

この条件を満たす最小のrankを採用する。

$$
\boxed{
\widetilde r_k
=
\min
\left\{
r\in\{1,\ldots,q_k\}
:
\sum_{j=r+1}^{q_k}
\sigma_{k,j}^2
\le
\delta_k^2
\right\}
}
\tag{1}
$$

$r=q_k$ なら尾部は空で0なので、集合は空にならない。

rankが小さいほど多くの特異値を捨て、以後の保存量と計算量を小さくできる。したがって、「条件を満たす最大rank」ではなく、「条件を満たす最小rank」を選ぶ。

$\widetilde r_k>1$ なら、理想的な比較では

$$
e_k(\widetilde r_k)^2
\le
\delta_k^2
<
e_k(\widetilde r_k-1)^2.
$$

つまり、採用rankでは予算内だが、もう1本多く捨てると予算を超える。

例えば $\delta_k=0.1$ で、rankを小さい方から試したときの候補誤差が

$$
0.19,
\quad
0.08,
\quad
0.03,
\quad
0
$$

なら、最初に予算を満たす $0.08$ に対応するrankを採用する。$0.03$ や $0$ も条件を満たすが、より大きいrankを残すため最小rank条件には合わない。

---

## 8. 具体的なrank選択例

$$
\sigma
=
(5,2,0.6,0.3),
\qquad
\delta_k
=
0.7,
\qquad
\delta_k^2
=
0.49
$$

とする。

保持rank 4では

$$
e_k(4)^2
=
0.
$$

保持rank 3では

$$
e_k(3)^2
=
0.3^2
=
0.09.
$$

保持rank 2では

$$
e_k(2)^2
=
0.6^2+0.3^2
=
0.36+0.09
=
0.45.
$$

保持rank 1では

$$
\begin{aligned}
e_k(1)^2
&=
2^2+0.6^2+0.3^2\\
&=
4+0.36+0.09\\
&=
4.45.
\end{aligned}
$$

予算条件を満たす最小rankは

$$
\widetilde r_k
=
2
$$

である。実局所誤差は

$$
e_k
=
\sqrt{0.45}
\approx
0.671
<
0.7
=
\delta_k.
$$

$e_k$ と $\delta_k$ は通常一致しない。rankが整数なので、予算を使い切らないことがある。

同じ特異値で $\delta_k=0.1$ なら、最小の非零特異値 $0.3$ も捨てられないため、

$$
\widetilde r_k
=
4
$$

である。

---

## 9. reverse cumulative sum

特異値二乗を

$$
s_j
=
\sigma_{k,j}^2
$$

とする。rank $r$ の尾部二乗和は

$$
t_k(r)
=
\sum_{j=r+1}^{q_k}s_j.
$$

小さい特異値側から累積すれば、すべての候補rankの尾部を一度に求められる。

上の例では、

$$
s
=
(25,4,0.36,0.09).
$$

候補rank $r=0,1,2,3,4$ に対応する尾部は

$$
\begin{aligned}
t_k(0)&=29.45,\\
t_k(1)&=4.45,\\
t_k(2)&=0.45,\\
t_k(3)&=0.09,\\
t_k(4)&=0.
\end{aligned}
$$

この列から、

$$
t_k(r)
\le
0.49
$$

を最初に満たす $r\ge1$ は2だと分かる。

### 9.1 末尾から逐次加算する方法

最小の特異値から二乗を1個ずつ足し、次の1個を足すと予算を超える位置で止めてもよい。

$$
0
\longrightarrow
\sigma_{q_k}^2
\longrightarrow
\sigma_{q_k}^2+\sigma_{q_k-1}^2
\longrightarrow
\cdots
$$

この方法は「小さい特異値を何個まで捨てられるか」という定義に直接対応する。

### 9.2 全候補を一括計算する方法

PyTorchでは逆順の累積和を使い、rank $r=0,1,\ldots,q_k$ に対する尾部を一括して作れる。0始まり添字では

$$
\texttt{tail\_sq[r]}
=
\sum_{j=r}^{q_k-1}
\texttt{S[j]}^2
$$

とすると、`r` 個保持したときの尾部に一致する。特に

$$
\texttt{tail\_sq[q\_k]}=0
$$

を末尾へ置く必要がある。先頭へ0を追加するとrankと尾部の対応が1つずれる。

例えば

$$
\sigma
=
(10,3,0.4,0.1)
$$

なら、

$$
\texttt{tail\_sq}
=
(109.17,9.17,0.17,0.01,0)
$$

である。$\delta=0.5$、$\delta^2=0.25$ なら、条件を最初に満たすのは

$$
\texttt{tail\_sq[2]}
=
0.17
\le
0.25
$$

なので、保持rankは2になる。

---

## 10. コア更新

採用rankを $\widetilde r_k$ としたとき、保持する因子は

$$
U_{k,\mathrm{keep}}
=
U_k(:,1:\widetilde r_k),
$$

$$
\Sigma_{k,\mathrm{keep}}
=
\Sigma_k(1:\widetilde r_k,1:\widetilde r_k),
$$

$$
W_{k,\mathrm{keep}}^T
=
W_k(:,1:\widetilde r_k)^T.
$$

新しい第 $k$ コアは

$$
\widetilde G^{(k)}
=
\operatorname{reshape}
\left(
U_{k,\mathrm{keep}},
(r_{k-1},n_k,\widetilde r_k)
\right).
$$

右へ送る係数は

$$
B_k
=
\Sigma_{k,\mathrm{keep}}
W_{k,\mathrm{keep}}^T
\in
\mathbb R^{\widetilde r_k\times r_k}.
$$

右隣コアへ

$$
\widetilde G^{(k+1)}
(\beta,i,\gamma)
=
\sum_{\alpha=1}^{r_k}
B_k(\beta,\alpha)
G^{(k+1)}(\alpha,i,\gamma)
$$

として吸収する。

---

## 11. $\varepsilon=0$ の扱い

$$
\varepsilon=0
$$

なら、目標は

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
=
0
$$

である。厳密算術では真のゼロ特異値だけを除去してもテンソルは変わらない。しかし浮動小数点では「数学的にゼロ」と「非常に小さい非ゼロ」を安易に区別できない。

学習用の安全な基本仕様では、

$$
\varepsilon=0
$$

なら意図的なSVD truncationを行わない。QRによるgauge整理は行ってもよいが、近似rank削減はしない。

この分岐へ、後述する浮動小数点比較用slackを流用してはいけない。正のslackを加えると、$\delta_k=0$ でも非ゼロ成分を捨てる可能性がある。

---

## 12. 零テンソルの扱い

$$
N
=
\|\mathcal X\|_F
=
0
$$

なら $\mathcal X$ は零テンソルである。相対誤差比

$$
\frac{
\|\mathcal X-\widetilde{\mathcal X}\|_F
}{N}
$$

は $0/0$ となり定義できない。

絶対誤差条件

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
=
0
$$

を守るため、出力も零テンソルとする。例えば、物理shapeを保ち、すべての内部rankを1とした

$$
\widetilde G^{(k)}
\in
\mathbb R^{1\times n_k\times1}
$$

の零コア列で表せる。

分岐順序は

$$
N=0
\quad\longrightarrow\quad
\varepsilon=0
\quad\longrightarrow\quad
\text{通常ケース}
$$

とする。

---

## 13. $d=1$ の扱い

$d=1$ では内部bondが存在せず、truncation回数は

$$
d-1
=
0
$$

である。$\sqrt{d-1}$ で割って局所予算を作ってはいけない。入力コアのコピーをそのまま返し、誤差0、truncationなしとする。

---

## 14. rank上限を別に課す場合

局所予算だけから決まるrankを

$$
\widetilde r_k^{(\mathrm{tol})}
$$

とする。別のrank上限

$$
r_{\max,k}
$$

を課すなら、実際に採用するrankは

$$
\widehat r_k
=
\min
\left(
\widetilde r_k^{(\mathrm{tol})},
r_{\max,k}
\right).
$$

上限が発動して

$$
\widehat r_k
<
\widetilde r_k^{(\mathrm{tol})}
$$

となれば、定義上

$$
e_k(\widehat r_k)
>
\delta_k
$$

である。したがって、その局所予算に基づく保証は失われる。

先ほどの例では、toleranceだけなら $\widetilde r_k^{(\mathrm{tol})}=2$ である。rank capを1にすれば、

$$
e_k(1)
=
\sqrt{4.45}
\approx
2.109
>
0.7.
$$

ただし、あるbondで予算を超えても、他bondの未使用予算が大きければ最終全体誤差が目標内に偶然収まることはある。

$$
\text{rank capが発動した}
$$

ことと、

$$
\text{実測全体誤差が目標を超えた}
$$

ことは別の判定である。

---

## 15. 相対toleranceと絶対tolerance

相対条件は

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\varepsilon_{\mathrm{rel}}
\|\mathcal X\|_F.
$$

絶対条件は

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\le
\tau_{\mathrm{abs}}.
$$

両方を同時に守るAPI契約なら、全体絶対予算は厳しい方を取る。

$$
T
=
\min
\left(
\varepsilon_{\mathrm{rel}}N,
\tau_{\mathrm{abs}}
\right).
$$

どちらか一方を満たせばよいという別契約なら、

$$
T
=
\max
\left(
\varepsilon_{\mathrm{rel}}N,
\tau_{\mathrm{abs}}
\right)
$$

となる。ただし後者は相対条件単独を必ず満たすという意味ではない。

基本的な学習用TT-roundingでは、相対toleranceだけを扱えばよい。

---

## 16. 浮動小数点比較の注意

理論上のrank条件は

$$
t_k(r)
\le
\delta_k^2
$$

である。境界比較を安定させる案として、

$$
t_k(r)
\le
\delta_k^2+\eta_k,
$$

$$
\eta_k
=
100u
\max
\left(
\delta_k^2,
\sum_j\sigma_{k,j}^2
\right)
$$

のようなadditive slackを考えることはできる。ここで $u$ はmachine epsilonである。

しかし、このslackは小さい $\varepsilon$ に対して誤差保証を実質的に緩めることがある。

例えば、

$$
\varepsilon
=
10^{-8},
\qquad
d=3
$$

なら、均等局所予算の二乗は

$$
\delta^2
=
\frac{10^{-16}N^2}{2}
=
5\times10^{-17}N^2.
$$

float64では

$$
u
\approx
2.22\times10^{-16}
$$

なので、

$$
100uN^2
\approx
2.22\times10^{-14}N^2.
$$

これは $\delta^2$ の約

$$
\frac{2.22\times10^{-14}}{5\times10^{-17}}
\approx
444
$$

倍である。もはや微小な等号判定補正ではない。

別の注意として、数学上

$$
0.4^2+0.1^2
=
0.17
$$

でも、浮動小数点では

$$
0.17000000000000004
$$

のように表現されることがある。このため、予算境界と数学上等しいrankを1つ多く残す場合がある。しかし、固定の `atol=10^{-12}` のような値を無条件に足すと、さらに小さい誤差予算を実質的に緩め得る。厳格な上限を優先するなら、境界で迷ったときに大きいrankを残し、最終実誤差も検証する。

したがって、厳密なtolerance契約を優先する基本実装では、

$$
t_k(r)
\le
\delta_k^2
$$

を保守的に比較し、最後に実測全体誤差を検証する。slackを使う場合は、それを理論予算内へ組み込んだことを明示し、`tolerance_met` を無条件に真としない。

---

## 17. 検証すべき量

小さいTTではdenseへ復元し、次を別々に確認する。

1. QR sweep前後でdense tensorが丸め誤差範囲で一致する。
2. right-canonicalコアの行Gram誤差が小さい。
3. QR後の $\|\mathcal X\|_F$ と第1コアのFrobeniusノルムが一致する。
4. 各bondで $e_k^2\le\delta_k^2$ が成り立つ。
5. $\widetilde r_k>1$ なら、1つ小さいrankでは予算を超える。
6. dense実測誤差と $\sqrt{\sum_ke_k^2}$ が丸め誤差範囲で一致する。
7. dense実測誤差が $\varepsilon N$ 以下である。
8. $d=1$、零TT、$\varepsilon=0$、圧縮が起きる例、起きない例を別々に確認する。

大規模TTでdenseへ戻せない場合は、TT内積を使って

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
+\|\widetilde{\mathcal X}\|_F^2
-2\langle
\mathcal X,
\widetilde{\mathcal X}
\rangle
$$

と計算できる。ただし非常に近いテンソル同士では、大きな3項の差による桁落ちに注意する。

---

## 18. 最終仕様

基本的なTT-roundingの数学仕様をまとめる。

1. 入力は実数TTコア列と $\varepsilon\ge0$。
2. $d=1$ ならコピーを返す。
3. $N=0$ なら物理shapeを保つrank-1零TTを返す。
4. 右から左へreduced QRを行い、right-canonical formを作る。
5. $N=\|G^{(1)}\|_F$ を得る。
6. $\varepsilon=0$ なら意図的なSVD truncationを行わない。
7. 通常ケースでは

$$
\delta_k
=
\frac{\varepsilon N}{\sqrt{d-1}}
$$

を使う。
8. 左から右へeconomy SVDし、式 (1) の最小rank $\widetilde r_k$ を選ぶ。
9. $U_{\mathrm{keep}}$ を現在コアに残し、$\Sigma_{\mathrm{keep}}W_{\mathrm{keep}}^T$ を右隣へ吸収する。
10. 各 $e_k$、入力・出力rank、全体誤差推定、tolerance判定を記録する。

最終的な誤差関係は

$$
\boxed{
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\sum_{k=1}^{d-1}e_k^2
\le
\sum_{k=1}^{d-1}\delta_k^2
\le
\varepsilon^2
\|\mathcal X\|_F^2
}
$$

である。

---

## 19. 間違えやすい論点

### 19.1 $\delta_k$ は特異値尾部そのものではない

$\delta_k$ は先に配る許容上限、$e_k$ は採用rankで実際に捨てた尾部ノルムである。

### 19.2 $e_k=\delta_k$ を狙うわけではない

rankは整数なので通常は

$$
e_k<\delta_k
$$

で未使用予算が残る。

### 19.3 最小rankは全特異値を捨てる意味ではない

rank下限を1とするため、少なくとも最大特異値に対応する成分を1本残す。

### 19.4 rank capとtolerance保証は競合する

capを優先して必要rankより小さくすれば、その局所予算保証を失う。

### 19.5 浮動小数点slackは常に安全ではない

特に非常に小さい $\varepsilon$ では、固定倍率のmachine-epsilon項が本来の予算を上回り得る。

### 19.6 零テンソルの相対誤差は定義しない

$0/0$ を0とみなして通常処理へ流さず、零テンソルとして分岐する。

---

## 次に読む

TT-rounding前後の誤差をdense tensorへ復元せずに評価するには、[[51_TT_MPSの内積とenvironment縮約]] で2本のTTの内積を導き、[[52_TT_MPSのFrobeniusノルムと距離]] でFrobenius距離へ接続する。差のTTを明示的に作る場合のrank増加は [[53_TT_MPSの加減算とrank増加]] を参照する。
