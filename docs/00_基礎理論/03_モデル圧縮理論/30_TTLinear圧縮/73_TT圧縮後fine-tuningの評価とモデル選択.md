---
title: TT圧縮後fine-tuningの評価とモデル選択
tags:
  - TTLinear
  - fine-tuning
  - early-stopping
  - model-selection
  - validation
---

# TT圧縮後fine-tuningの評価とモデル選択

## 1. このノートの範囲

このノートは、TT-SVDで圧縮したモデルをfine-tuningするときの一般的な読み方を整理する。特定のcheckpointで得たrank 16、32、64の数値と、その結果からの考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/03_fc1_TT-fine-tuning]]へ分ける。

区別する中心的な論点は次である。

- 圧縮直後の性能とfine-tuning後の性能
- accuracy、cross-entropy、prediction agreement、logits RMSE
- recoveryの大きさとfine-tuningへの反応性
- 共通学習条件での比較と、各候補の最良性能の比較
- parameter数、実測学習時間、理論計算量、推論時間
- validationによるモデル選択とtestによる最終評価

## 2. fine-tuningはdenseモデルの再現を目的にしているとは限らない

正解label$y_i$に対するcross-entropyでfine-tuningする場合、最適化しているのは、

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}
$$

である。denseモデルのlogitsを$z_i^{\mathrm{dense}}$、圧縮モデルのlogitsを$z_i^{\mathrm{TT}}$としても、

$$
\left\lVert
z_i^{\mathrm{TT}}-z_i^{\mathrm{dense}}
\right\rVert_2^2
$$

は目的関数へ直接入っていない。従って、fine-tuning後のモデルは、dense logitsを再現するのではなく、低rank制約の下で正解labelに適合する別の解へ移動し得る。

dense出力への忠実度も目的にするなら、cross-entropyだけでなく、logitsや確率分布をteacherへ近づけるdistillation型の目的が別途必要になる。

## 3. logits RMSEが大きくてもaccuracyを維持できる理由

accuracyが使うのはlogitsの値全体ではなく、

$$
\arg\max_c z_c
$$

である。一方、logits RMSEは全classの数値差を平均する。

### 3.1 正のスケール変化

正解classが先頭にあるとして、

~~~text
dense logits      = [3, 2, 0]
fine-tuned logits = [30, 20, 0]
~~~

では、数値差は大きいがargmaxは同じである。ただし、softmax確率の集中度は変化するため、cross-entropyやcalibrationまで同じとは限らない。

### 3.2 全class共通のオフセット

~~~text
dense logits      = [3, 2, 0]
fine-tuned logits = [13, 12, 10]
~~~

では、全classへ10を加えただけなので、argmaxだけでなくsoftmax確率も変わらない。一方、生のlogits RMSEは10になる。

従って、logits RMSEの増加だけから、分類境界が悪化した、過信が強まった、または過学習したとは断定できない。原因を分けるには、少なくとも次を確認する。

- sampleごとにclass平均を引いたcentered logitsのRMSE
- logitsの標準偏差やノルム
- 1位と2位のclass margin
- softmax確率とsample別cross-entropy
- fine-tuning前後のprediction agreement

## 4. accuracyとcross-entropyは同じ変化をしない

accuracyは正解classが1位かだけを見る。cross-entropyは正解classへ割り当てた確率も評価する。

例えば、正解classの確率が、

~~~text
0.20 -> 0.01
~~~

と変化したsampleは、どちらも誤分類のままでも、後者のcross-entropyが大きい。このため、accuracyがほぼ変わらなくてもvalidation lossは悪化し得る。

validation lossの上昇を解釈するときは、次を分ける。

- 誤分類数が増えた
- 同じ誤分類でも誤ったclassへの確信が強まった
- 正解を維持したsampleでも正解classの確率が下がった

## 5. train lossとvalidation lossの乖離

train lossが下がり続ける一方、validation lossが最小値を過ぎて上昇する場合、後半の更新が訓練データへの適合を改善しても、未使用データへ一般化していない可能性がある。これは過学習と整合する典型的な兆候である。

ただし、学習曲線だけから特定sampleの暗記や原因となるparameterを確定することはできない。モデル容量、初期状態、学習率、正則化、mini-batch順などが同時に影響する。

early stoppingでは、最終epochのモデルではなく、監視するvalidation lossが最良だったepochの重みへ復元する。従って、終盤のvalidation loss悪化と、復元後best modelのaccuracyを同一視しない。

## 6. recoveryと最終性能を分ける

accuracy recoveryを、

$$
\Delta A
=
A_{\mathrm{after}}
-
A_{\mathrm{before}}
$$

とする。$\Delta A$が大きいことは、fine-tuning前後の絶対改善量が大きいことを示す。しかし候補ごとの出発点が違う場合、$\Delta A$だけから「fine-tuningへの反応性」を比較できない。

例えば、

~~~text
候補A: 0.50 -> 0.88  recovery = 0.38
候補B: 0.87 -> 0.89  recovery = 0.02
~~~

では、Aの改善余地がもともと大きい。比較では少なくとも次を併記する。

- fine-tuning前の性能
- recovery
- fine-tuning後の最終性能
- parameter数と計算コスト

## 7. 圧縮直後の性能とfine-tuning後の性能

TT-SVD直後のaccuracyは、学習済みdense weightを指定rankで近似した直後に、性能をどれだけ維持できるかを測る。

fine-tuning後のaccuracyは、そのrank制約の下でparameterを再最適化した後に、どこまでタスク性能を得られるかを測る。

従って、fine-tuningを前提とするモデル選択では、圧縮直後のaccuracyだけを使うと、初期性能は低くても再最適化で回復できる低rank候補を除外し、高rankを選びすぎる場合がある。ただし、低rankほど常にfine-tuning後に優れるという意味ではない。

また、低rankモデルが高いaccuracyへ回復しても、denseと同じ関数や同じdecision boundaryを表現できると証明したことにはならない。その評価データ上で、高いaccuracyを与えるparameterが見つかったことまでが直接の結論である。

## 8. 比較の公平性には2種類ある

### 8.1 共通protocolへの反応を比べる

全候補に同じcheckpoint、mini-batch順、optimizer、学習率、early-stopping規則を与える比較は、共通手順に対する反応を比べる意味で公平である。

### 8.2 各候補の最良性能を比べる

同じ学習率が、parameter数や初期性能の異なる全rankに最適とは限らない。各候補の最良性能を比較するには、同じ候補集合、同じ試行数、または同じ計算予算でhyperparameter探索の機会を揃える必要がある。

従って、固定protocolの結果からは、

> 共通条件下で候補Aが最良だった。

とは言えるが、

> 各候補を最適に調整しても候補Aが最良である。

とは限らない。

## 9. model selectionでは最高値と採用モデルを分ける

圧縮の目的が、accuracyを維持しながらparameter数を減らすことであれば、観測されたaccuracyの最大値をそのまま採用基準にする必要はない。

例えば、

~~~text
rank 16: accuracy = 0.9000, parameter数が少ない
rank 24: accuracy = 0.9002, parameter数が多い
~~~

なら、差は0.02パーセントポイントである。5,000件では正解数の差し引き1件に相当する。この差がseed変動より十分大きいか、追加parameterに見合うかを確認する。

選択規則の例は、

> validation accuracyが最高値から許容差$\varepsilon$以内の候補のうち、parameter数が最小のものを選ぶ。

である。$\varepsilon$は普遍的な値ではなく、用途と許容リスクに応じて事前に決める。parameter数とaccuracyのPareto frontierを使う方法もある。

## 10. 微小なaccuracy差と対応のある比較

同じ5,000件でモデルAが4,442件、モデルBが4,445件正解しても、「Bが3件だけ予測を改善した」とは限らない。

- $b$：Aは不正解、Bは正解
- $c$：Aは正解、Bは不正解

とすると、集計accuracyから分かるのは、

$$
b-c=3
$$

だけである。例えば$b=10,c=7$でも同じ差になる。固定した2モデルの対応のある正誤差を検定する場合はMcNemar検定を利用できるが、異なるcheckpointやseedでも順位が再現するかは別の問題である。

## 11. validationとtestの役割

- train：weightを更新する
- validation：rank、学習率、early stopping、許容差を決める
- test：すべての選択を固定した後、未知データ上の性能を評価する

testで勾配を計算しなくても、test結果を見てrank追加、学習率変更、seed選択を行えば、開発手順がtestへ適合する。従って、最終候補と評価手順をvalidationで固定するまでtestを見ない。

複数モデルを事前に決めた手順でtest評価すること自体ではなく、test結果を使って候補を選び直すことが問題である。

## 12. parameter数と計算コストを分ける

次は異なる量である。

- parameter数：保存するscalar数
- early stoppingまでの総学習時間：学習手順全体の実測コスト
- 1 epochまたは1 step当たりの時間：同じ処理量に対する実測速度
- FLOPs・MACs：理論的な演算回数
- 推論latency：実運用時の応答時間
- peak memory：最大メモリ使用量

総学習時間には実行epoch数の違いが含まれる。候補の処理速度を比較するなら、同じbatch size、固定step数、warm-up、複数反復を用い、中央値やばらつきを報告する。fine-tuning時間だけから推論高速化は結論づけられない。

## 13. 一般化に必要な検証

1つのcheckpoint、fine-tuning seed、validation split、共通学習条件から得たrank順位は、一般的な最適rankを意味しない。段階的に次を確認する。

1. 同じcheckpointとsplitでfine-tuning seedを変える。
2. denseモデルの学習seedを変え、別checkpointでも比較する。
3. 必要ならデータ分割を変え、dense学習から選択までを再実行する。
4. 全rankへ同じ探索機会を与え、学習率や正則化を比較する。
5. 未評価rankを追加し、近傍と圧縮限界を調べる。
6. 最終候補を固定した後、test setで一度だけ最終評価する。

分割を変える場合、元のdenseモデルが学習したsampleを新しいvalidationへ混ぜてはならない。

## 14. 実験への接続

Fashion-MNIST MLPのrank 16、32、64を通常fine-tuningした保存結果と、その結果からの考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/03_fc1_TT-fine-tuning]]で扱う。
