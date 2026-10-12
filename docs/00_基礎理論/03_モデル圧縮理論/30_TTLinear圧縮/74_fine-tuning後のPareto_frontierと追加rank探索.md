---
title: Fine-tuning後のPareto frontierと追加rank探索
tags:
  - model-compression
  - TTLinear
  - fine-tuning
  - Pareto-frontier
  - rank-selection
  - validation
---

# Fine-tuning後のPareto frontierと追加rank探索

## 1. このノートの目的

圧縮モデルのrank選択では、accuracy最大の候補だけを探すのではなく、性能と資源コストのtrade-offを評価する。このノートでは、次の一般的な考え方を整理する。

- fine-tuning前後のPareto frontierを分ける理由
- frontierの端に最良候補がある場合の追加探索
- 微小なaccuracy差と許容差の扱い
- 共通学習条件の比較と、各rankの最良性能比較の違い
- validationで選択し、testを最終評価へ残す手順

Fashion-MNIST MLPの具体的な保存結果は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/04_fine-tuning後のTT-rank_trade-off]]で扱う。

## 2. Pareto dominance

候補$i$のparameter数を$P_i$、validation accuracyを$A_i$とする。parameter数を小さく、accuracyを大きくしたいとき、候補$j$が候補$i$を支配するとは、

$$
P_j \le P_i,
\qquad
A_j \ge A_i
$$

が成り立ち、少なくとも一方が狭義不等号になることである。

他の候補に支配されない点の集合がPareto frontierである。frontier上の点同士には、通常、accuracyとmodel sizeのtrade-offが残る。

### 小さい数値例

| rank | parameter数 | accuracy |
| ---: | ---: | ---: |
| 12 | 120,000 | 0.8880 |
| 16 | 150,000 | 0.8890 |
| 32 | 210,000 | 0.8850 |

rank 32はrank 16より大きく、accuracyも低いため支配される。rank 12と16では、12が小ささで勝ち、16がaccuracyで勝つため、両方がfrontierに残り得る。

## 3. fine-tuning前後のfrontierは別々に求める

圧縮直後とfine-tuning後は、異なる問いに答える。

- before：dense weightをそのまま近似したとき、性能をどれだけ保存できるか
- after：圧縮構造の下で再最適化したとき、どこまでタスク性能を得られるか

beforeとafterの点を1つの候補集合へ混ぜると、rank間のtrade-offとfine-tuningの効果が混ざる。従って、stageごとにfrontierを求め、frontierの構造がどう変わったかを比較する。

例えば、beforeで高rankほどaccuracyが高く、afterで最低rankが全候補を支配したなら、fine-tuningによって「高rankへparameterを追加するaccuracy上の利点」が、評価した範囲では消えたと解釈できる。

## 4. frontierは候補集合と評価軸に依存する

rank 16、32、64だけを入力してrank 16だけがfrontierに残っても、rank 16が連続的なrank空間全体で最適だとは分からない。Pareto判定は未評価のrankを推測しない。

また、parameter数とaccuracyの2目的で支配されても、別の目的を加えれば関係が変わる可能性がある。

- 推論latency
- peak memory
- 消費電力
- denseとのprediction agreement
- logitsまたは確率分布の再現性
- calibration
- モデル作成に必要なfine-tuning・探索コスト

ただし、「別指標なら価値があるかもしれない」と「その別指標で実際に優れていた」は区別する。測っていない指標上の優位性は仮説であり、実験結果ではない。

## 5. 最良点が探索範囲の端に来た場合

探索済みrankを、

$$
r_1 < r_2 < \cdots < r_K
$$

とする。最良候補が内部の$r_k$なら、その近傍$r_{k-1}$と$r_{k+1}$を細かくする選択が考えられる。一方、最良候補が最小rank$r_1$なら、探索範囲の外側、つまりさらに低rank側に未探索領域がある。

この場合の自然な次の問いは、

> どこまでrankを下げても、許容できる性能を維持できるか。

である。最小rankがfrontierに残ったことは探索終了ではなく、低rank側へ探索範囲を広げる根拠になり得る。

## 6. 中間rankと外挿rankは目的が違う

- rank 24：16と32の間の曲線を細かくする
- rank 48：32と64の間の曲線を細かくする
- rank 12：現在の最小rank 16より強い圧縮側を調べる

中間rankは既存区間の急変やkneeを調べる。外側のrankは、現在の探索境界が十分だったかを調べる。

既存候補だけで実用上の選択が明確なら、すべての中間点を埋める必要はない。逐次探索では、現在の意思決定を変える可能性が高い候補を優先する。

## 7. 許容accuracy差による選択

圧縮が目的なら、観測accuracy最大の候補を自動的に採用する必要はない。最高accuracyを$A_{\max}$、許容差を$\varepsilon$とすると、

$$
\mathcal{C}_{\varepsilon}
=
\left\{
i
\mid
A_i \ge A_{\max}-\varepsilon
\right\}
$$

を許容候補集合とし、その中でparameter数や実測コストが最小の候補を選ぶ方法がある。

例えば、

| rank | accuracy | parameter数 |
| ---: | ---: | ---: |
| 12 | 0.8880 | 120,000 |
| 16 | 0.8890 | 150,000 |

で$arepsilon=0.001$なら、rank 12も許容集合に入る。accuracyを最優先するなら16、`0.001`の低下を許容して軽さを重視するなら12を選べる。

ただし、$\varepsilon$は統計的同等性を自動的に保証しない。用途上許容できる低下量として、結果を見る前に決めるのが望ましい。結果を見た後に都合よく変更した場合は、その選択を探索的と明記する。

## 8. 微小なaccuracy差には対応関係が必要

同じ$N$件で2モデルを評価するとき、

- $b$：model Aは不正解、model Bは正解
- $c$：model Aは正解、model Bは不正解

とすると、accuracy差は、

$$
\Delta A
=
\frac{b-c}{N}
$$

である。差だけから$b$と$c$は分からない。

例えば$N=5{,}000$で差が3件なら、$b-c=3$である。$b=3,c=0$でも、$b=103,c=100$でも同じaccuracy差になる。固定した2モデルの同一sample上の正誤差を調べるなら、McNemar検定のような対応のある比較を使う。

最も一方に有利な$b=3,c=0$でも、両側の正確な検定では、

$$
p
=
2\left(\frac{1}{2}\right)^3
=
0.25
$$

となる。従って3件差だけでは、一般的な優位性を主張できない。

さらに、固定された2モデル間の差と、rank設定としての再現性は別である。後者にはfine-tuning seedやdense checkpointを変えた反復が必要になる。

## 9. 表現能力、最適化、汎化

rankを上げると、一般に表現可能なweightの集合は広がる。しかし、次の含意は成立しない。

$$
\text{高rank}
\not\Rightarrow
\text{共通学習条件で高いvalidation accuracy}
$$

理由は3つに分ける。

1. 表現能力：どのweightを表せるか
2. 最適化：初期値からどのweightへ到達するか
3. 汎化：得られたweightが未学習データでどの程度機能するか

optimizerがtrain lossを下げても、validation accuracyの改善は保証しない。また、学習が落ち着いたように見えても、局所最小点へ到達したと証明したことにはならない。

## 10. dense再現とタスク再最適化

正解labelへのcross-entropyを、

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}
$$

とする。この損失にdense logitsとの差が含まれなければ、fine-tuning後のモデルがdenseから離れることはあり得る。

denseのlogitsを$z_i^{\mathrm{dense}}$、圧縮モデルを$z_i^{\mathrm{TT}}$として、

$$
\left\lVert
z_i^{\mathrm{TT}}
-
z_i^{\mathrm{dense}}
\right\rVert_2^2
$$

を小さくすることは、分類lossの最小化とは別の目的である。dense出力の保持も必要なら、distillation型の損失を追加する。

また、生のlogits RMSEは共通offsetやscale変化にも反応する。RMSEの原因を分解するなら、centered logits、logits norm、class margin、softmax確率も確認する。ただし、生logitsの再現が目的なら、元のRMSEを消して別指標だけに置き換えない。

## 11. fine-tuning前も候補に含める

圧縮直後から性能が高いrankでは、fine-tuningしない状態が最良である可能性がある。従ってepoch 0も候補に含める。

これは、

> 圧縮後は必ずfine-tuningしたmodelを採用する。

という暗黙の制約を外すことに相当する。分類用途ならvalidation accuracy、dense再現用途ならlogits RMSEなど、選択指標を目的ごとに事前に定める。

## 12. 公平性には2種類ある

### 12.1 共通protocolへの反応を比べる

全rankへ同じcheckpoint、seed、mini-batch順、optimizer、学習率、early-stopping規則を与える。これは、同じ手順に対する反応を比較する意味で公平である。

### 12.2 各rankの最良性能を比べる

rankごとに適切な学習率や停止時点が違う可能性がある。各rankの潜在性能を比較するなら、同じ学習率候補、同じ試行数、または同じ計算予算を与える。

前者の実験からは、

> 共通条件ではrank 16が最良だった。

とは言える。しかし、

> 各rankを最適に調整してもrank 16が最良である。

とは限らない。

## 13. 段階的な追加実験

目的を混ぜず、次の順で進める。

### 13.1 探索境界を広げる

最小rankが有力なら、まず1つ低いrankを評価する。accuracyだけでなく、parameter数、MACs、latency、memoryを分けて記録する。

### 13.2 学習条件との相性を調べる

各rankへ、例えば、

$$
\eta_0,
\qquad
\frac{\eta_0}{3},
\qquad
\frac{\eta_0}{10}
$$

の同じlearning-rate gridを与える。epoch 0と各epochの指標を保存する。

### 13.3 更新対象をablationする

- 全parameter更新
- TT coresのみ更新
- 必要ならTT coresとbiasのみ更新

を比較し、回復が低rank層自身から来たのか、後段の再適応に依存したのかを調べる。

### 13.4 複数seedで再現性を調べる

条件選択後、同じrank間でfine-tuning seedをそろえて反復し、平均、標準偏差、各試行値を報告する。別のdense checkpointでも比較すれば、圧縮対象となる元モデルへの依存も確認できる。

seedはhyperparameterのように、test結果を見て最もよいものを採用しない。最終checkpointを1つ必要とする場合は、例えば次のどちらかをtest評価前に固定する。

- 代表モデルには事前指定したseedを使い、複数seedは安定性評価に使う
- 固定したvalidation基準でseedも選び、その選択手順を明記する

複数seedをすべて最終評価する場合も、そのtest結果から採用seedや条件を選び直さず、平均、ばらつき、各試行値として報告する。

### 13.5 testで最終評価する

rank、学習率、更新対象、checkpoint選択をvalidationで固定してから、未使用testを評価する。test結果を見て再び選択を変更すると、そのtestは最終評価ではなくなる。

## 14. モデル選択の最終単位

選ぶ対象はrankだけではない。実際には、

$$
\text{model construction procedure}
=
\text{rank}
+
\text{initialization}
+
\text{updated parameters}
+
\text{training protocol}
+
\text{selection metric}
$$

を選んでいる。

同じrankでも、圧縮直後を採用するか、分類lossでfine-tuningするか、dense再現を含む損失で学習するかによって、得られるモデルは異なる。rank単独の順位へ単純化せず、用途に必要な一連の作成手順として報告する。
