---
title: CNNのLinear SVD
aliases:
  - CNN fc1 SVD
  - Fashion-MNIST CNN Linear圧縮
  - CNN Linear rank selection
tags:
  - Fashion-MNIST
  - CNN
  - Linear
  - SVD
  - FineTuning
  - RankSelection
---

# CNNのLinear SVD

## サマリー

02/03ではCNN後段の

```text
fc1 = Linear(3136, 128)
```

をSVDして、

```text
Linear(3136, r, bias=False)
Linear(r, 128, bias=True)
```

へ置換した。02でrank sweepとknee選択、03でknee±1をFine-tuningし、最終的に **fc1 rank 24** を採用した。

```text
Parameters: 421,642 → 98,570  (-76.62%)
Test acc:   91.75%  → 91.87%   (+0.12 percentage point)
```

この実験では差が小さいため、accuracyが「改善した」とは扱わず、**大幅なparameter削減後も精度を維持した**と解釈する。

---

## なぜfc1を圧縮するか

`fc1.weight` は `(128, 3136)` の2次元行列であり、MLPで使ったLinear SVDをそのまま適用できる。

また、bias込みのparameter数は、

```text
fc1 = 401,536
CNN全体 = 421,642
```

で、全体の約95.23%を占める。したがってparameter数を減らす効果が非常に大きい。

### rankとparameter数

元のfc1：

$$
3136\times128+128
$$

低rank2層：

$$
3136r+128r+128
$$

したがって、fc1単体のparameter数が元より減る条件は、

$$
r < \frac{3136\times128}{3136+128}\approx122.98
$$

となる。

---

## 02: rank sweep

候補rank：

```text
16, 20, 24, 28, 32, 36, 42, 44, 48, 64, 80, 96, 112, 128
```

`parameters` と `validation_loss` のPareto frontierを作り、正規化したfrontierのkneeを求めた。

![[20_FashionMNIST/assets/cnn_linear_rank_pareto.png]]

```text
Pareto points = 10
knee          = rank 24
knee neighbors = 20 / 24 / 28
```

### knee±1

| Candidate | fc1 rank | Parameters | Param reduction | Val loss | Val acc | Agreement | Logits RMSE | Retained energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggressive | 20 | 85,514 | 79.72% | 0.250269 | 91.42% | 0.9568 | 1.347940 | 0.827196 |
| Balanced / knee | **24** | **98,570** | **76.62%** | **0.233418** | **91.72%** | **0.9754** | **0.855327** | **0.856742** |
| Conservative | 28 | 111,626 | 73.53% | 0.231134 | 91.66% | 0.9736 | 0.723372 | 0.877729 |

rankを増やすとretained energyは増えるが、Validation accuracyは単調には増加していない。retained energyは重み近似の指標であり、分類性能そのものではない。

---

## Notebook内MACsの読み方

02/03の `compressed_macs` は**CNN全体ではなく、圧縮対象のfc1とfc2を合わせたLinear部分**の値。

rank 24では、

```text
factorized fc1 = 3136*24 + 24*128 = 78,336 MACs
fc2            = 128*10           = 1,280 MACs
------------------------------------------------
Linear部分                           79,616 MACs
```

Notebookに表示される `compute_reduction = 80.23%` はこのLinear部分を分母にした削減率。

全CNNで比較するためにconv1/conv2も含めると、

```text
Baseline total MACs = 4,241,152
fc1 rank24 total    = 3,918,080
Total reduction     = 7.62%
```

となる。

---

## 03: knee±1 Fine-tuning

Fine-tuning条件：

```text
optimizer     = Adam
learning rate = 3e-4
max epoch     = 30
patience      = 3
seed          = 0
```

各rank候補の前にseedとDataLoader用generatorを同じ状態へ戻して比較した。

全候補ともepoch 4でEarly Stoppingし、best epochは1。

| Candidate | rank | Acc before | Acc after | Δacc | Loss before | Loss after | Δloss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggressive | 20 | 91.42% | 92.16% | +0.74pt | 0.250269 | 0.220859 | -0.029409 |
| Balanced | **24** | 91.72% | **92.26%** | +0.54pt | 0.233418 | **0.220129** | -0.013290 |
| Conservative | 28 | 91.66% | 92.20% | +0.54pt | 0.231134 | 0.220493 | -0.010640 |

![[20_FashionMNIST/assets/cnn_linear_finetuning_delta.png]]

![[20_FashionMNIST/assets/cnn_linear_finetuning_after.png]]

Fine-tuning後のRank/Final Validationではrank 24が3候補中で最小lossかつ最高accuracyとなり、最終モデルに選択した。

---

## 最終Test

| Model | fc1 rank | Parameters | Test loss | Test acc | ΔTest acc |
|---|---:|---:|---:|---:|---:|
| Baseline | — | 421,642 | **0.234131** | 91.75% | — |
| fc1 SVD + FT | **24** | **98,570** | 0.238414 | **91.87%** | **+0.12pt** |

```text
Parameter reduction = 76.62%
Test loss change     = +0.004283
```

Test accuracyはほぼ同じだがlossは少し増えているため、正解数が近くてもlogits・予測確率まで同じとは限らない。

## rank 24の意味

rank 24は「Validation accuracyが最大になるrank」そのものではなく、`parameters` と `validation_loss` のPareto frontier上で求めた**圧縮率と性能の折衷点（knee）**である。実際、rank 32以上ではValidation性能がBaselineへさらに近づく一方、parameter削減率は小さくなる。

したがって今回の `rank=24` は唯一の最適解ではなく、**今回採用した評価軸とknee規則に対するBalancedな選択**と解釈する。

また、full-rankの `rank=128` ではagreementが1.0まで戻る一方、2層へ分解するためCNN全体のParametersは `438,026` となり、Baselineの `421,642` を上回る。

```text
rankを大きくする
→ 近似誤差は減る
→ しかし2層化のオーバーヘッドが増える
→ 一定rank以上では「圧縮」ではなくなる
```

これは理論上のbreak-even `r < 122.98` とも整合する。

## Fine-tuningから分かること

3候補ともbest epochは1で、SVD直後から短いFine-tuningでValidation lossが大きく回復した。結果からは、SVDで得た因子がランダム初期化ではなく、元の学習済み重みの情報を保持した有効な初期値になっていると考えられる。

ただし、1 runの結果だけから「SVDが正則化として働いてaccuracyを改善した」とまでは結論しない。ここで確認できたのは、**低rank化による近似誤差をFine-tuningでかなり補正できた**ことまでである。

## 結論

- `fc1` SVDはCNN全体のparameter削減に非常に効いた。
- fc1 rank 24でparameterを76.62%削減した。
- CNN全体MACsの削減は7.62%で、parameter削減ほど大きくない。
- SVD直後の性能低下はFine-tuningで回復した。
- 最終Test accuracyはBaselineと同程度に維持された。

## 関連

- [[20_FashionMNIST/07_CNN実験]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
