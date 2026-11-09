---
title: 計量付きTT-ALSの局所方程式と停留性
aliases:
  - metric-aware TT ALS
  - TT-ALSの局所方程式
  - block stationarity
tags:
  - TensorTrain
  - TT-matrix
  - ALS
  - Gram行列
  - 共役勾配法
  - 停留性
  - モデル圧縮
---

# 計量付きTT-ALSの局所方程式と停留性

## このノートの役割

このノートは、入力データのGram行列を使ってTT-matrixを最適化する1-site ALSについて、次を一般理論として整理する。

- 出力誤差から計量付き目的関数が現れる理由
- 他のTT coreを固定すると局所問題が二次関数になる理由
- 局所方程式 $H_k c_k=b_k$ の意味
- 共役勾配法で逆行列を作らずに解く方法
- 局所CG残差と、sweep終了時のblock stationarityの違い
- $s_k$ と $s_{\max}$ を停止判定へ使うときの注意
- 数学的収束と、有限予算での実用的な打ち切りの区別

Qwen3-0.6B-Baseの実測値は、[[60_LLM_TT_vs_Matrix/10_Qwen3_0p6B_Base/03_長いALSの収束確認と初期値依存]]へ分ける。

関連する実装は次である。

- `src/nn_compression/tt_mps/metric_tt_solver.py`
- `src/nn_compression/metrics/als_convergence.py`
- `tests/test_metric_tt_convergence.py`
- `tests/test_als_convergence.py`

---

## 1. 出発点は層出力の二乗誤差

入力を $x\in\mathbb{R}^{n}$、元の重みを $W\in\mathbb{R}^{m\times n}$、TT-matrixで近似した重みを $\widehat W$ とする。biasは重み近似の前後で変えないものとする。

元の出力と近似後の出力は、

$$
y=Wx,
\qquad
\widehat y=\widehat W x
$$

である。差を $E=W-\widehat W$ と置くと、1入力に対する出力誤差は、

$$
y-\widehat y=Ex
$$

となる。入力 $x_1,\ldots,x_N$ に対する二乗誤差の総和は、

$$
\begin{aligned}
Q
&=\sum_{t=1}^{N}\left\|Ex_t\right\|_2^2\\
&=\sum_{t=1}^{N}\operatorname{tr}\left(Ex_t x_t^{\mathsf T}E^{\mathsf T}\right)\\
&=\operatorname{tr}\left(EGE^{\mathsf T}\right),
\end{aligned}
$$

ただし、

$$
G=\sum_{t=1}^{N}x_t x_t^{\mathsf T}
$$

は入力の非中心化Gram行列である。$G$の固有値が大きい入力方向ほど、重み誤差を強く数える。

### 1.1 正則化した計量

$G$が低rankまたは悪条件の場合、次のように小さい正則化を加える。

$$
G_{\lambda}
=G+\lambda\frac{\operatorname{tr}(G)}{n}I
$$

目的関数は、

$$
\boxed{
Q_{\lambda}
=\operatorname{tr}\left[
(W-\widehat W)G_{\lambda}(W-\widehat W)^{\mathsf T}
\right]
}
$$

となる。$Q_{\lambda}$は重みのFrobenius誤差そのものではない。入力データがよく使う方向を重く数える、層出力に対応した誤差である。

---

## 2. ベクトル化した二次形式

$W$と$\widehat W$を同じ順序でベクトル化し、

$$
w=\operatorname{vec}(W),
\qquad
\widehat w=\operatorname{vec}(\widehat W)
$$

とする。行ごとに同じ $G_{\lambda}$ が作用するため、適切なvectorization順序では、

$$
M=I_m\otimes G_{\lambda}
$$

と置いて、

$$
Q_{\lambda}
=(w-\widehat w)^{\mathsf T}
M
(w-\widehat w)
$$

と書ける。

$M$は対称半正定値である。正則化によって $G_{\lambda}$ が正定値になれば、$M$も正定値になる。

---

## 3. 他のcoreを固定すると線形写像になる

TT-matrixの全coreを $C^{(0)},\ldots,C^{(d-1)}$ とする。core $k$ 以外を固定し、core $k$ の要素を1列に並べたベクトルを $c_k$ とする。

TTの成分は各coreについて多重線形である。他のcoreを固定すれば、$c_k$について線形になるため、

$$
\boxed{
\widehat w=P_k c_k
}
$$

と書ける。$P_k$は新しい学習parameterではない。固定した左右のcoreから決まる、「core $k$ から重み全体を組み立てる線形写像」である。

重み全体の要素数を $D=mn$、core $k$ の要素数を $p_k$ とすると、

$$
P_k\in\mathbb{R}^{D\times p_k},
\qquad
c_k\in\mathbb{R}^{p_k}
$$

である。

### 3.1 小さい例

固定行列と更新対象を、

$$
A=
\begin{pmatrix}
1\\
2
\end{pmatrix},
\qquad
C=
\begin{pmatrix}
u&v
\end{pmatrix}
$$

とする。$\widehat W=AC$なら、

$$
\widehat W=
\begin{pmatrix}
u&v\\
2u&2v
\end{pmatrix}
$$

である。行ごとに並べると、

$$
\underbrace{
\begin{pmatrix}
u\\v\\2u\\2v
\end{pmatrix}
}_{\widehat w}
=
\underbrace{
\begin{pmatrix}
1&0\\
0&1\\
2&0\\
0&2
\end{pmatrix}
}_{P}
\underbrace{
\begin{pmatrix}
u\\v
\end{pmatrix}
}_{c}
$$

となる。TTでは、$P_k$の中身を左右の複数coreの縮約で構成する。

---

## 4. 局所二次関数の導出

$\widehat w=P_kc_k$を目的関数へ代入する。

$$
Q_{\lambda}(c_k)
=(w-P_kc_k)^{\mathsf T}
M
(w-P_kc_k)
$$

展開すると、

$$
\begin{aligned}
Q_{\lambda}(c_k)
&=w^{\mathsf T}Mw
-w^{\mathsf T}MP_kc_k
-c_k^{\mathsf T}P_k^{\mathsf T}Mw\\
&\quad
+c_k^{\mathsf T}P_k^{\mathsf T}MP_kc_k.
\end{aligned}
$$

$M^{\mathsf T}=M$であり、中央の2項は同じスカラーなので、

$$
H_k=P_k^{\mathsf T}MP_k,
\qquad
b_k=P_k^{\mathsf T}Mw
$$

と置けば、

$$
\boxed{
Q_{\lambda}(c_k)
=c_k^{\mathsf T}H_kc_k
-2b_k^{\mathsf T}c_k
+w^{\mathsf T}Mw
}
$$

を得る。最後の項は $c_k$ に依存しない。

勾配は、

$$
\nabla_{c_k}Q_{\lambda}
=2H_kc_k-2b_k
$$

である。したがって、他のcoreを固定した局所最小化条件は、

$$
\boxed{
H_kc_k=b_k
}
$$

となる。

### 4.1 $H_k$と$b_k$の意味

- $H_k$：core $k$ を変えたときの、目的関数の局所的な曲率
- $b_k$：元の重みと入力計量を、core $k$ の空間へ引き戻した右辺
- $c_k$：更新前・更新後の両方を表せるcoreの変数

$b_k$は更新前のcoreではない。

---

## 5. 共役勾配法で局所方程式を解く

$H_k$は $p_k\times p_k$ の密行列であり、明示的に作るとメモリと計算量が大きい。実装では、任意のベクトル $v$ に対する、

$$
H_kv=P_k^{\mathsf T}MP_kv
$$

だけを縮約で計算する。これはHessian-vector productに相当する。

共役勾配法は、$H_k^{-1}$を作らず、$H_kv$と$b_k$を使って $H_kc_k=b_k$ を反復的に解く。

### 5.1 局所CG残差

局所solverが使う相対残差は、

$$
r_k=b_k-H_kc_k,
$$

$$
\boxed{
\frac{\|r_k\|_2}{\|b_k\|_2}
=
\frac{\|b_k-H_kc_k\|_2}{\|b_k\|_2}
}
$$

である。これは重み誤差率や層出力誤差率ではない。現在のcoreが局所方程式をどれだけ満たしているかを表す。

---

## 6. $s_k$とblock stationarity

局所更新の直後は、その時点の他のcoreに対して残差が小さい。しかし、その後のcoreを更新すると環境 $P_k$ が変わり、先に解いたcoreは再び局所最適点からずれる。

そこで、1 sweepの終了時点の同じTT状態について、coreを更新せずに各siteへ正準中心を移し、

$$
\boxed{
s_k
=
\frac{\|b_k-H_kc_k\|_2}{\|b_k\|_2}
}
$$

を測り直す。全siteの最大値を、

$$
\boxed{
s_{\max}=\max_k s_k
}
$$

とする。

$s_{\max}$が小さいことは、「どのblockについても、他のblockを固定した局所方程式をほぼ満たす」ことを意味する。これはblock coordinate stationaryであるための診断であり、非凸なTT全体の大域最適性を保証しない。

実装では $\|b_k\|_2=0$ の場合、ゼロ除算を避けて $s_k=0$ を返す。これは実装上の規約である。一般の問題で $b_k=0$ が起こり得る場合は、$\|H_kc_k\|_2$も確認し、残差が本当にゼロかを別に診断する必要がある。

### 6.1 正準中心を移す理由

TTにはgauge自由度がある。同じ $\widehat W$ を表していても、隣接core間でscaleを移せるため、coreの数値だけをそのまま比べられない。

core $k$ を混合正準形の中心にすると、左右のenvironmentが等長写像になる。残差を同じ正準条件で測ることで、単なるscale配分の違いを抑えて比較できる。

### 6.2 sweep順序による見かけの非対称性

3-coreを、

$$
0\rightarrow1\rightarrow2\rightarrow1\rightarrow0
$$

の順で更新すると、sweep終了時のcore 0は解いた直後である。そのため $s_0$ は局所solverの許容誤差程度まで小さくなりやすい。一方、core 2は更新後にcore 1と0が動くので、$s_2$が最大になりやすい。

したがって、site別の $s_k$ を読むときは、sweep順序を併記する必要がある。

---

## 7. 数値例

$$
H=
\begin{pmatrix}
2&0\\
0&1
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
4\\
3
\end{pmatrix},
\qquad
c^{\mathrm{before}}=
\begin{pmatrix}
1\\
2.5
\end{pmatrix}
$$

とする。更新前の残差は、

$$
r=b-Hc^{\mathrm{before}}
=
\begin{pmatrix}
2\\
0.5
\end{pmatrix}
$$

である。相対残差は、

$$
s
=
\frac{\sqrt{2^2+0.5^2}}{\sqrt{4^2+3^2}}
\approx0.412
$$

となる。

局所方程式の解は、

$$
c^{\mathrm{after}}=H^{-1}b
=
\begin{pmatrix}
2\\
3
\end{pmatrix}
$$

であり、厳密演算なら残差は0になる。実装では逆行列を作らず、共役勾配法でこの解へ近づける。

---

## 8. 内側の精度と外側の停留性は別である

次の2つを混同しない。

1. 局所CG残差
   - 1回のsite更新で $H_kc_k=b_k$ をどこまで正確に解いたか
2. sweep終了時の $s_{\max}$
   - すべてのcoreが、同じ最終状態で局所方程式をどこまで同時に満たすか

局所方程式を毎回 $10^{-8}$ まで解いても、その後に他のcoreが動けば $s_{\max}$ は $10^{-3}$ や $10^{-4}$ になり得る。反対に、外側の進みが遅い段階で、内側だけを極端に高精度で解くことが計算効率上適切とは限らない。

---

## 9. 停止理由を分ける

長期ALSでは、少なくとも次を分ける。

### 9.1 `converged`

目的値の相対減少が所定回数連続して小さく、かつ $s_{\max}$ も小さい状態である。

$$
\frac{Q_{t-1}-Q_t}{Q_{t-1}}<\varepsilon_Q
$$

だけでは、平坦な方向をゆっくり進んでいる状態と区別できない。そのため、停留性も併用する。

### 9.2 `max_sweeps`

収束条件を満たす前に、事前に定めたsweep上限へ到達した状態である。これは「発散」や「今後も収束しない」を意味しない。

### 9.3 `local_solver_unresolved`

局所CGを再試行しても残差基準を満たせなかった状態である。これは外側ALSの収束判定とは別の失敗である。

未解決の局所候補で目的値が下がっていても、正式な状態へ採用せず、最後に完了したsweepへ戻すと、成果物の意味が明確になる。

---

## 10. 初期値依存を判定する順序

二つの初期値を比較するときは、最終目的値の差だけで判定しない。先に停止理由を確認する。

1. どちらかが `local_solver_unresolved`
   - 局所問題未解決として保留
2. それ以外で、どちらかが `max_sweeps`
   - sweep上限による未解決
3. 両方が `converged`
   - 初めて最終目的値の相対差を評価する

相対差の一例は、

$$
\delta_Q
=
\frac{|Q_A-Q_B|}{\min(Q_A,Q_B)}
$$

である。$\delta_Q$が小さくても、未収束なら「同じ停留点へ収束した」とは言えない。

---

## 11. 数学的収束と実用的な打ち切り

数学的な収束条件を満たす前でも、目的値や検証誤差が実用上ほぼ変わらなくなる場合がある。

このときは、次を区別して記録する。

- 収束：事前固定した停留条件を満たした
- 打ち切り：計算予算、目的値変化、検証指標などを根拠に停止した
- 最終値の一定範囲内：後から得た最終値を基準にした診断であり、オンライン停止規則ではない

後から「最終値の0.1%以内へ何sweepで入ったか」を計算することは、将来の上限設計には有用である。ただし、その実行中には最終値が分からないため、そのまま停止条件にはできない。

---

## 12. 保存・再開で保持する情報

長期実行では、少なくとも次を保存する。

- 完了sweep数
- TT coresと正準中心
- $Q_{\lambda}$と正則化なしの$Q$
- site別 $s_k$ と $s_{\max}$
- 局所CGの反復数、残差、再試行
- 単調性確認
- 設定と入力tensorのhash
- 診断値と計算時間
- 失敗した局所候補の情報

再開結果と中断なし実行を照合するときは、時間列を除いた履歴と最終coreを比較する。

---

## 13. 読み取りの限界

- $s_{\max}$が小さくても大域最適性は保証しない。
- 一つの初期値で低い目的値を得ても、初期値一般への依存性は分からない。
- 層出力誤差が小さくても、end-to-endのperplexityやtask性能が保たれるとは限らない。
- Gram行列の固有値が小さい方向では、Frobenius距離が大きくても計量付き誤差が小さくなり得る。
- $s_{\max}$の経験的な減衰傾向を、非常に長いsweepまでそのまま外挿してはいけない。

---

## まとめ

計量付きTT-ALSでは、他のcoreを固定すると、目的関数は更新対象coreについて二次関数になる。

$$
Q_{\lambda}(c_k)
=c_k^{\mathsf T}H_kc_k-2b_k^{\mathsf T}c_k+\text{constant}
$$

その局所最適条件は、

$$
H_kc_k=b_k
$$

であり、局所CGはこの方程式を解く。sweep終了時に全coreの相対残差を測り直したものが $s_k$、その最大値が $s_{\max}$である。

$s_{\max}$は「どのcoreにも局所的な改善余地がほとんど残っていないか」を診断するが、大域最適性を保証しない。目的値の変化、局所solverの状態、停留性、計算予算を分けて記録することが、長期ALSを正しく評価するために必要である。
