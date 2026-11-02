---
title: TT-MLPのfull-batch共同更新と更新予算
aliases:
  - TT-MLP full-batch joint update
tags:
  - TTLinear
  - DMRG-like
  - full-batch gradient
  - joint update
  - update budget
  - canonicalization
---

# TT-MLPのfull-batch共同更新と更新予算

## このノートの役割

本ノートは、TT-MLPの隣接2-core局所更新を、全学習データ平均損失による左右sweepへ拡張するときの一般技術を整理する。Fashion-MNISTの個別結果は次へ分ける。

- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]

物理DMRGとの構造的な対応、2-core結合、方向付きSVD、切り捨て誤差と損失低下条件は[[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]を前提とする。

---

## 1. 初期化を段階に分ける

dense行列$W$をTT-matrixへ変換するとき、次を区別する。

```text
A: dense weight
↓ reshape・TT-SVD・rank打ち切り
B: rank制限TT
↓ gauge変換・正準中心移動
C: 正準化TT
```

### 1.1 TT表現とrank打ち切りは別である

打ち切りなしTT-SVDでは、TTは行列の別表現であり、丸め誤差内で$W$を再構成できる。rank capを指定すると、各unfoldingで保持しない特異値方向が捨てられる。

行列SVDの1 cutだけを考えると、rank $r$近似のFrobenius誤差は

$$
\left\lVert M-M_r\right\rVert_F^2
=
\sum_{i>r}\sigma_i^2
$$

である。TT-SVDではこれを複数cutで順に行うため、実効weightのnorm、方向、activation scaleが変化し得る。

### 1.2 正準化はgaugeを変える

隣接core間では、可逆行列$X$を使って

$$
G_k\mapsto G_kX,
\qquad
G_{k+1}\mapsto X^{-1}G_{k+1}
$$

としても全体の縮約は変わらない。QRやSVDを用いた正準化は、このgauge freedomを使って直交性と中心位置を整える。

正準化前後で実効weightとlogitsを比較する理由は、core normだけでは全体関数の不変性を判断できないからである。

### 1.3 数値例

Fashion-MNISTの例では、dense norm `13.0630`がrank制限後に`1.88249`へ下がったが、正準化前後は同じ`1.88249`だった。正準化前後のlogits最大差は約$1.5\times10^{-8}$だった。

この例では、スケール変化はrank打ち切り、表現不変な整理は正準化と切り分けられる。

---

## 2. weight scaleとactivation scale

第1 Linearを

$$
z_1=Wx+b,
\qquad
h_1=\operatorname{ReLU}(z_1)
$$

とする。$W$のFrobenius normが小さくなっても、個々の入力に対する$z_1$の変化は入力方向と特異ベクトルに依存する。norm比だけからすべてのactivationが同じ比率で縮むとは限らない。

そのため、初期化診断では少なくとも次を測る。

- 実効weightのFrobenius norm。
- fixed batch上の線形出力のmean、std、RMS。
- ReLU後のRMSとzero rate。
- 後段logitsのRMS。
- lossとaccuracy。

zero rateが約50%でも、非ゼロactivationの振幅が小さくなることはある。「dead ReLUか」と「activation scaleが縮んだか」は別の問いである。

---

## 3. 全データ平均損失をminibatchへ分けて計算する

training sample数を$N$、sampleごとのlossを$\ell_i(\theta)$とすると、目的関数は

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)
$$

である。batch $b$のsample数を$n_b$、batch mean lossを

$$
L_b(\theta)=\frac{1}{n_b}\sum_{i\in b}\ell_i(\theta)
$$

とすると、

$$
L(\theta)=\sum_b\frac{n_b}{N}L_b(\theta)
$$

である。よって勾配も

$$
\nabla_\theta L
=
\sum_b\frac{n_b}{N}\nabla_\theta L_b
$$

として蓄積できる。

### 3.1 なぜ単純なbatch平均の平均ではないか

最後のbatchが小さいとき、$L_b$をbatch数で単純平均すると各sampleの重みが等しくならない。$n_b/N$を掛ければ、batch sizeにかかわらず全sampleを$1/N$で重み付けできる。

### 3.2 minibatch SGDとの違い

ここでminibatchはmemoryへ収める計算分割であり、batchごとにparameterを更新しない。全batchのbackwardを終えてから1回だけ局所更新する。

```text
gradを初期化
for batch:
    weighted batch lossを計算
    backwardしてgradへ加算
全batch終了
parameter候補を1回作る
```

この方式では、各pairが同じtraining mean lossを目的にする。batchごとに目的が変わるオンラインSGDとは区別する。

---

## 4. 同じlossからTT側と通常parameter側の勾配を得る

現在更新する隣接2 coreを結合し、独立なleaf tensor$T$を作る。通常parameterをまとめて$\phi$とする。モデルのlossは

$$
L(T,\phi)
$$

という一つのスカラーである。

backwardにより、

$$
g_T=\frac{\partial L}{\partial T},
\qquad
g_\phi=\frac{\partial L}{\partial\phi}
$$

が同時に得られる。勾配が二種類あるのはlossが二つあるからではなく、同じlossを異なる変数で偏微分しているからである。

素の勾配stepなら候補は

$$
T_{\mathrm{cand}}=T-\eta_Tg_T,
$$

$$
\phi_{\mathrm{cand}}=\phi-\eta_\phi g_\phi
$$

となる。$\eta_T$と$\eta_\phi$を同じ値にしても、parameter normとgradient normが異なるので相対更新量は同じにならない。

相対更新量は

$$
\rho=\frac{\eta\lVert g\rVert}{\lVert\theta\rVert}
$$

で測る。これは「元のparameter normに対して候補更新がどれだけ大きいか」の尺度であり、loss低下や安定性を単独で保証しない。

---

## 5. 候補更新とatomic commit

1回の局所更新は次の順で行う。

1. 最新の2 coreから$T$を作る。
2. 全データ平均lossから$g_T$と$g_\phi$を蓄積する。
3. $T_{\mathrm{cand}}$と$\phi_{\mathrm{cand}}$を作る。
4. $T_{\mathrm{cand}}$を方向付きSVDで2 coreへ戻す。
5. 候補モデルへTT coreと通常parameter候補を入れる。
6. shape、bond、rank、dtype、device、有限性、diagnostic forwardを検査する。
7. 全検査が通った場合だけ実モデルへ一括反映する。

TT側だけ先に実モデルへ反映してから通常parameter勾配を計算すると、二つの勾配が別の更新前stateに対応してしまう。同じ更新前stateから候補全体を作り、まとめてcommitすることで意味を固定する。

有限であることはloss低下を意味しない。現在のcontractは、非有限値や構造違反を拒否するものであり、diagnostic batchのlossが上がったという理由だけでrollbackする設計ではない。

---

## 6. 方向付きSVDと正準中心

更新後のmerged tensorを行列$M$へreshapeし、

$$
M=U\Sigma V^{\mathsf T}
$$

とする。

right sweepでは

$$
M=U(\Sigma V^{\mathsf T})
$$

として特異値を右coreへ持たせる。左coreは列直交になり、正準中心は右へ移る。

left sweepでは

$$
M=(U\Sigma)V^{\mathsf T}
$$

として特異値を左coreへ持たせる。右coreは行直交になり、正準中心は左へ移る。

6 coreなら、pair順は

```text
right: 0, 1, 2, 3, 4
left : 4, 3, 2, 1, 0
```

であり、中心は0から5へ移動して0へ戻る。折り返しでpair 4を連続訪問するが、directionと特異値の吸収先が異なる。

---

## 7. 更新予算を数える

core数を$d$、往復数を$S$、各pairの局所step数を$L$とする。1往復のpair visit数は

$$
2(d-1)
$$

なので、成功局所更新数は

$$
N_{\mathrm{update}}
=
2(d-1)SL
$$

となる。$d=6$なら

$$
N_{\mathrm{update}}=10SL
$$

である。

### 7.1 数値例

- $S=5,L=1$：50更新。
- $S=5,L=4$：200更新。
- $S=20,L=1$：200更新。
- $S=20,L=4$：800更新。
- $S=40,L=8$：3,200更新。

$S=5,L=4$と$S=20,L=1$は総更新数が同じなので、同じpairに長く留まる配分と、多く往復する配分を比べられる。

ただし、validationを往復終了時に行うなら評価回数は$S$で決まる。同じ更新数でもwall timeは完全には同じにならない。

### 7.2 local stepの意味

現在の`local_steps_per_pair=L`は、$T$を$L$回連続更新して最後に1回だけSVDする方式ではない。次の一式を同じpairで$L$回繰り返す。

```text
最新coreを結合
→ 全データ勾配
→ 候補更新
→ 方向付きSVD
→ 検査
→ commit
```

SVD前inner step数を導入する場合は、別の実験因子として名前と通常parameterの更新時点を定義する必要がある。

---

## 8. final、best、固定予算、early stopping

### 8.1 finalとbest

`final`は実行した最後のstate、`best`は記録したvalidation基準が最小のstateである。

finalがbestでも、途中の指標が毎回単調だったとは限らない。逆に、途中がbestなら、最後まで更新したstateよりvalidation基準がよかったことを意味する。

### 8.2 固定予算比較

学習率や更新配分の影響を見る段階では、early stoppingを無効にし、全条件を同じ事前予算まで実行すると比較しやすい。途中で止まらないため、不安定化も観察できる。

一方、固定更新数は固定wall timeではない。評価回数、データ供給、更新対象の違いで1更新の費用が変わる。

### 8.3 best選択用データ

validationを繰り返し見てbestを選ぶと、そのsplitに対して選択が適応する。test setは最終評価まで使わない。rank選択、early stopping、最終testの役割を分ける。

---

## 9. 因果解釈の限界

joint updateで性能が改善しても、TT側だけの効果とは言えない。必要なcontrolled comparisonは、同じ初期state・データ・予算で、

- joint：TTと通常parameterを更新。
- TT-only：通常parameterを更新対象から外す。
- dense-only：TTを更新対象から外す。
- 必要ならno-update control。

を比較することである。

「固定」は学習率を0にするだけでは不十分な場合がある。学習率0でもSVD再分割や正準中心移動を実行すれば、純粋な無操作controlではない。何を固定するかを更新経路レベルで定義する。

また、単一seedの小差を一般化しない。最良学習率、更新配分、収束速度を主張するには、複数seedと同じ判断規則が必要である。

---

## 10. 物理DMRGとの違い

共通する構造は、隣接blockを結合し、他を固定して局所変数を更新し、SVDで再分割しながら左右へsweepする点である。

しかし、NN側では局所Hamiltonianの基底状態を厳密に解いていない。非線形networkのcross-entropyに対して、全データgradient stepを行っている。rank打ち切り後のloss単調減少や大域最適性は保証されない。

したがって、`DMRG-like`は更新構造の対応を表す名称であり、物理DMRGと同じ収束保証を意味しない。

---

## 11. 実装・実験チェックリスト

- 全条件が同じ初期checkpointから独立に開始する。
- checkpointのhash、data split、sample順を保存する。
- batch meanに$n_b/N$を掛ける。
- batch途中でparameterを更新しない。
- TT側と通常parameter側の学習率を別フィールドで記録する。
- 往復数、local step数、成功更新数を別々に保存する。
- SVD direction、center、retained rank、discarded normを保存する。
- 非有限候補を実モデルへcommitしない。
- finalとbestを別checkpointとして保存する。
- checkpoint再読込後の評価を確認する。
- joint updateの結果をTT-onlyの効果と書かない。
- 固定予算と固定wall timeを区別する。

## 関連

- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]

