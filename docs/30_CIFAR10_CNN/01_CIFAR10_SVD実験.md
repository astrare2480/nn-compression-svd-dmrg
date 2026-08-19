---
title: CIFAR-10 CNNのSVD圧縮実験
aliases:
  - CIFAR10 SVD
  - CIFAR10 CNN SVD圧縮
  - CIFAR10 model-wide rank allocation
tags:
  - CIFAR10
  - CNN
  - SVD
  - NN圧縮
  - RankSelection
  - FineTuning
---

# CIFAR-10 CNNのSVD圧縮実験

## サマリー

CIFAR-10用CNNの `conv1` / `conv2` / `conv3` を対象に、SVDによる低rank化を行った。

この実験の目的は、単一層のrank sweepだけではなく、**複数Conv層を同時に圧縮するときのrank配分**を扱うことである。

探索は全rank組合せの総当たりではない。

```text
各Convを単独rank sweep
        ↓
各層のPareto frontier / knee
        ↓
knee近傍を候補化
        ↓
候補組合せだけmodel-wide評価
        ↓
Fine-tuning
        ↓
最終model選択
```

したがって、本実験は

> **knee近傍に探索空間を制約したmodel-wide rank allocation**

として扱う。

正式結果は `notebooks/30_cifar10/02_svd_global_compression_using_src_corrected.ipynb` と、対応するcorrected resultsを基準とする。

---

## 最終結果

### Baseline

| 項目 | 値 |
|---|---:|
| Validation accuracy | 0.7344 |
| Validation loss | 0.7602 |
| Test accuracy | **0.7327** |
| Test loss | 0.7556 |
| Parameters | **128,842** |
| MACs | **10,357,248** |
| Latency | **約0.531 ms/batch** |

### Final compressed + Fine-tuning

| 項目 | 値 |
|---|---:|
| conv1 rank | **9** |
| conv2 rank | **32** |
| conv3 rank | **48** |
| Validation accuracy | **0.7444** |
| Validation loss | **0.7469** |
| Test accuracy | **0.7343** |
| Test loss | **0.7510** |
| Parameters | **81,405** |
| Parameter reduction | **36.82%** |
| MACs | **5,625,344** |
| MAC reduction | **45.69%** |
| Latency | **約0.537 ms/batch** |

Test accuracy差は、

$$
0.7343 - 0.7327 = 0.0016
$$

で、**+0.16 percentage point**。

ただしsingle seedの小差なので、accuracy改善とは断定せず、**約37%のparameter削減・約46%のMACs削減後も精度をほぼ維持した**と解釈する。

---

# 1. モデル

CIFAR-10用CNNは、3つのConv層とGAP、2つのLinear層からなる。

```text
Input (3, 32, 32)
  ↓
conv1: 3 → 32, 3x3
ReLU + MaxPool
  ↓
conv2: 32 → 64, 3x3
ReLU + MaxPool
  ↓
conv3: 64 → 128, 3x3
ReLU + MaxPool
  ↓
Global Average Pooling
  ↓
fc1: 128 → 256
  ↓
fc2: 256 → 10
```

Baseline parameter数は `128,842`。

Conv重みのbiasを除く主な内訳は、

```text
conv1 =    864
conv2 = 18,432
conv3 = 73,728
fc1   = 32,768
fc2   =  2,560
```

で、特に `conv3` が大きい。

GAPを使うため、Fashion-MNIST CNNのように巨大なfc1がparameterを支配する構造ではない。したがって、CIFAR-10ではConv層そのものの圧縮がモデル全体へより直接効く。

---

# 2. Conv SVD

Conv2d重み、

$$
W \in \mathbb{R}^{C_{out}\times C_{in}\times k_H\times k_W}
$$

を、

$$
W_{mat} \in \mathbb{R}^{C_{out}\times(C_{in}k_Hk_W)}
$$

へ行列化し、truncated SVDを適用する。

```text
Conv2d(C_in → C_out, kxk)
        ↓
Conv2d(C_in → r, kxk, bias=False)
        ↓
Conv2d(r → C_out, 1x1, bias=元bias)
```

この方式では、元Convのstride / padding / dilation / padding_modeを前段へ引き継ぎ、biasは最終出力側へ置く。

現在の実装は `groups=1` のConv2dを対象とする。

---

# 3. 単一層rank sweep

最初に、`conv1` / `conv2` / `conv3` を1層ずつ圧縮してValidation性能を評価した。

これは、モデル全体のrank配分を直接総当たりする前に、各層の圧縮感度を見るためである。

各層について、

```text
rank
parameters
validation loss
validation accuracy
MACs
latency
```

などを記録し、`parameters` と `validation_loss` のPareto frontierからkneeを求めた。

重要なのは、full-rank近くまでrankを増やせば必ず「圧縮」になるわけではないこと。

SVD factorizationでは元1層を2層にするため、高rankでは元Convよりparameter数やMACsが増える場合がある。

---

# 4. Knee近傍だけをmodel-wide探索

各層の単独sweepから得たknee周辺rankを使って、3層の組合せを評価した。

この探索は全rank空間のglobal optimum探索ではない。

```text
全rank候補
  ↓
各層単独sweepで絞る
  ↓
knee近傍だけ残す
  ↓
その直積をmodel-wide評価
```

という**制約付き探索**である。

そのため、採用された `9 / 32 / 48` は、

> 今回の候補集合と評価規則の中で選ばれたrank配分

であり、考えられる全rank組合せの大域最適解とは言わない。

---

# 5. Fine-tuning前後

最終的にFine-tuning候補として、knee周辺から代表的な3構成を比較した。

| Candidate | conv1 | conv2 | conv3 | Val acc after FT | Val loss after FT | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Aggressive | 6 | 32 | 32 | 0.7336 | 0.7802 | 69,964 |
| Balanced | 9 | 32 | 32 | 0.7360 | 0.7715 | 70,141 |
| Conservative | **9** | **32** | **48** | **0.7444** | **0.7469** | **81,405** |

最終選択は `9 / 32 / 48`。

この構成は、Fine-tuning前にはValidation accuracyが約0.4792まで低下していたが、Fine-tuning後に0.7444まで回復した。

```text
SVD direct
Val acc  = 0.4792
Val loss = 1.7082

Fine-tuning後
Val acc  = 0.7444
Val loss = 0.7469
```

この結果は、強い低rank近似で一度性能が大きく落ちても、元の学習済み重みをSVDで分解した因子を初期値にして追加学習することで、低rank制約内でtask性能を大きく回復できる場合があることを示す。

ただし、Fine-tuningが常に元性能まで回復する保証はない。

---

# 6. Test結果

TestはrankやFine-tuning候補の選択には使わず、最終model決定後に確認した。

| Model | Test loss | Test accuracy |
|---|---:|---:|
| Baseline | 0.755618 | 0.7327 |
| Final compressed + FT | **0.750995** | **0.7343** |

差は小さい。

```text
accuracy: +0.16 pt
loss    : -0.00462
```

single seedの1回の結果なので、「SVD圧縮で精度が上がった」とは結論しない。

ここで重要なのは、

```text
Parameters -36.82%
MACs       -45.69%
Test acc   ほぼ同水準
```

を同時に達成できたこと。

---

# 7. MACsとlatency

理論MACsは大幅に減った。

```text
Baseline   10,357,248
Compressed  5,625,344
Reduction      45.69%
```

一方、同一input batch、同一batch size、同一warmup/repeatsで測ったGPU latencyは、

```text
Baseline   約0.531 ms/batch
Compressed 約0.537 ms/batch
```

で、短縮しなかった。

これは重要な結果である。

```text
MACs reduction
≠
wall-clock speedup
```

低rank化すると1つのConvが複数演算へ分かれるため、実速度には、

- kernel launch
- 中間Tensor
- memory access
- 演算shape
- backend最適化

なども影響し得る。

ただし、今回それぞれの要因を分離して計測したわけではないため、「SVD Convは一般に遅くなる」とは結論しない。

今回のGPU・batch=256・この実装で、**MACs削減がlatency短縮へ直結しなかった**という観測として扱う。

---

# 8. Fashion-MNISTからCIFAR-10へ進んだ意味

Fashion-MNISTでは、

```text
Linear SVD
Conv SVD
Conv + Linear同時圧縮
```

までを確認した。

CIFAR-10ではさらに、

```text
1層のrankを選ぶ
        ↓
複数層へrank budgetをどう配るか
```

という問題へ進んだ。

したがってCIFAR-10実験の主題は、SVDの式をもう一度確認することではなく、**複数層を持つ実モデルでrank allocationをどう設計するか**にある。

---

# 9. 実験上の制約

## Single seed

corrected実験は `SEED=0` のsingle run。

複数seedによる平均・標準偏差は取っていないため、0.16ptのtest accuracy差を統計的改善とは解釈しない。

## 探索空間

knee近傍に制約した探索であり、全rank空間のglobal optimumではない。

## Latency

同条件比較には修正したが、平均値1つを主に記録しており、分散・信頼区間までは評価していない。

したがって速度差が極小の場合は、強い結論を出さない。

---

# 10. 結論

- 3つのConv層を対象にmodel-wide rank allocationを行った。
- 最終rankは `conv1=9, conv2=32, conv3=48`。
- Parametersを **36.82%** 削減した。
- 理論MACsを **45.69%** 削減した。
- Test accuracyは `73.27% → 73.43%` で、ほぼ維持された。
- 強い圧縮直後の性能低下をFine-tuningで大きく回復できた。
- MACs削減はGPU latency短縮には直結しなかった。
- 探索は全rank空間のglobal optimumではなく、knee近傍に制約したmodel-wide探索である。

この結果を、行列化したConv SVDの到達点とし、次はテンソルの多モード構造をより直接扱うTucker decompositionへ進む。

---

# 関連

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
