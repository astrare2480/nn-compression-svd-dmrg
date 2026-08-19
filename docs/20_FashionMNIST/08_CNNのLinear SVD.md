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

CNN後段の、

```text
fc1 = Linear(3136, 128)
```

へLinear SVDを適用し、

```text
Linear(3136 → r, bias=False)
Linear(r → 128, bias=True)
```

へ置換した。

`02_cnn_linear_svd.ipynb` でrank sweep、`03_cnn_linear_svd_finetuning.ipynb` でknee近傍をFine-tuningし、

```text
fc1 rank = 24
```

を採用した。

この02/03 runはレビュー時点でbenchmark条件とcandidate seed条件が揃っていたため、04/05のようなcorrectedコピーは作っていない。

```text
Parameters 421,642 → 98,570  (-76.62%)
Test acc   0.9175  → 0.9187
```

小差なのでaccuracy改善とは扱わず、**大幅なparameter削減後も精度を維持した**と解釈する。

> [!important]
> この02/03 Linear-only runと、corrected 04/05 Conv系runはBaselineを別々に学習している。
> したがって `0.9175` とcorrected 04/05のBaseline `0.9128` を同一runの値として比較しない。

---

# 1. なぜfc1を圧縮するか

`fc1.weight` は、

```text
(128, 3136)
```

の2次元行列なので、MLPで使ったLinear SVDをそのまま適用できる。

また、

```text
fc1 parameters = 401,536
CNN total       = 421,642
```

で、fc1が全parameterの約95%を占める。

したがって、model sizeを減らしたい場合はfc1が非常に効率のよい圧縮対象。

---

# 2. rankとparameter数

元fc1：

$$
3136\times128+128
$$

低rank2層：

$$
3136r+128r+128
$$

fc1単体のparameter数が元より減る条件は、

$$
r<\frac{3136\times128}{3136+128}
\approx122.98
$$

となる。

数学的な最大rankは128だが、

```text
数学的に有効なrank
≠
圧縮になるrank
```

である。

rank 128ならweight写像はfull-rankで再構成できる一方、2層化のparameter overheadでBaselineより大きくなる。

---

# 3. rank sweep

候補：

```text
16, 20, 24, 28, 32, 36, 42, 44, 48, 64, 80, 96, 112, 128
```

`parameters` と `validation_loss` のPareto frontierを作り、frontier上のkneeを求めた。

```text
knee = rank 24
neighbors = 20 / 24 / 28
```

代表値：

| Candidate | rank | Parameters | Val loss | Val acc | Retained energy |
|---|---:|---:|---:|---:|---:|
| Aggressive | 20 | 85,514 | 0.250269 | 0.9142 | 0.827196 |
| Balanced | **24** | **98,570** | **0.233418** | **0.9172** | **0.856742** |
| Conservative | 28 | 111,626 | 0.231134 | 0.9166 | 0.877729 |

retained energyはrankとともに増えるが、Validation accuracyは必ずしも単調ではない。

```text
retained energy
→ weight approximation

accuracy / loss
→ task performance
```

を区別する。

---

# 4. MACsの読み方

このNotebookで表示する `compressed_macs` はCNN全体ではなく、Linear部分の比較値。

rank 24では、

```text
factorized fc1 = 3136*24 + 24*128 = 78,336 MACs
fc2            = 128*10           = 1,280 MACs
Linear total                           79,616 MACs
```

一方、Convも含めたCNN全体では、

```text
Baseline total MACs = 4,241,152
fc1 rank24 total    ≈ 3,918,080
reduction           ≈ 7.62%
```

となる。

parameter削減は76%以上だが、CNN全体MACsの削減は約8%に留まる。

ここから、

```text
parameterを最も持つ層
≠
MACsを最も使う層
```

が分かる。

---

# 5. Fine-tuning

候補：

```text
rank 20 / 24 / 28
```

Fine-tuning条件：

```text
optimizer = Adam
learning rate = 3e-4
max epoch = 30
patience = 3
seed = 0
```

この03 runでは候補ごとに同じseed / shuffle条件へ戻して比較した。

Fine-tuning後：

| Candidate | rank | Acc before | Acc after | Loss before | Loss after |
|---|---:|---:|---:|---:|---:|
| Aggressive | 20 | 0.9142 | 0.9216 | 0.250269 | 0.220859 |
| Balanced | **24** | 0.9172 | **0.9226** | 0.233418 | **0.220129** |
| Conservative | 28 | 0.9166 | 0.9220 | 0.231134 | 0.220493 |

rank 24が3候補中で最小Validation lossかつ最高accuracyとなり、最終候補にした。

---

# 6. 最終Test

| Model | fc1 rank | Parameters | Test loss | Test acc |
|---|---:|---:|---:|---:|
| Baseline | — | 421,642 | 0.234131 | 0.9175 |
| fc1 SVD + FT | **24** | **98,570** | 0.238414 | **0.9187** |

```text
Parameter reduction = 76.62%
Accuracy delta      = +0.12pt
Test loss change    = +0.004283
```

accuracy差は小さく、lossは少し増えている。

したがって中心結論は、

> **parameterを約77%削減してもTest accuracyをほぼ維持した**

とする。

---

# 7. Conv SVDとの違い

fc1 SVDでは、

```text
大きなparameter削減
小さめのCNN全体MACs削減
```

となった。

一方、conv2 SVDでは、

```text
小さなparameter削減
大きなMACs削減
```

となる。

この違いを確認したことが、Conv + Linear同時圧縮へ進む理由。

---

# 結論

- CNNでもMLPと同じLinear SVDを適用できる。
- fc1 rank 24でparameter数を76.62%削減した。
- CNN全体MACsの削減は約7.62%で、parameter削減ほど大きくない。
- Fine-tuningでSVD直後の性能を回復した。
- Test accuracyはほぼ維持した。
- Linear-only 02/03は有効な独立runだが、corrected 04/05のBaseline値とは混ぜない。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/07_CNN実験]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
