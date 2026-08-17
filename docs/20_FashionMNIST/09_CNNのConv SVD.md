---
title: CNNのConv SVD
aliases:
  - CNN conv2 SVD
  - Fashion-MNIST Conv圧縮
  - Conv2d rank selection
tags:
  - Fashion-MNIST
  - CNN
  - Conv2d
  - SVD
  - FineTuning
  - RankSelection
---

# CNNのConv SVD

## サマリー

04では、

```text
conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
```

をSVD圧縮した。

```text
conv2.weight: (64, 32, 3, 3)
        ↓ flatten(start_dim=1)
W_mat:       (64, 288)
        ↓ truncated SVD rank=r
Conv2d(32 → r, 3x3, bias=False)
Conv2d(r  → 64, 1x1, bias=True)
```

rank sweep → Pareto → knee±1 → Fine-tuningを行い、最終的に **conv2 rank 28** を採用した。

```text
Parameters: 421,642 → 413,066  (-2.03%)
Test acc:   91.75%  → 91.97%   (+0.22 percentage point)
```

parameter削減は小さいが、Convは同じ重みを多数の空間位置で使うため、MACs削減への寄与が大きい。

---

## 対象とした重み

SVDするのは特徴マップ `(N, 32, H, W)` ではなく、学習済みの `conv2.weight`。

```text
(64, 32, 3, 3)
→ (64, 288)
```

詳細は以下。

- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]

### conv1を今回の対象にしなかった理由

`conv1.weight` は `(32, 1, 3, 3)` で288 weights、bias込みでも320 parameters。行列化しても `(32, 9)` で最大rankは9となる。今回のモデルでは圧縮余地が小さいため、Convでは `conv2` を対象にした。

---

## rankとparameter数

conv2の元parameter数：

$$
64\times32\times3\times3+64=18496
$$

低rank化後：

$$
r(32\times3\times3)+64r+64=352r+64
$$

元よりparameterが減る条件は、

$$
352r+64 < 18496
$$

より、

$$
r < 52.36\ldots
$$

となる。full-rankの64では元の写像を再現できるが、2層化によりparameter数とMACsは元Convより増える。

---

## rank sweep

候補：

```text
16, 20, 24, 28, 32, 36, 42, 44, 48, 64
```

![[20_FashionMNIST/assets/cnn_conv_rank_pareto.png]]

```text
Pareto points  = 9
knee           = rank 28
knee neighbors = 24 / 28 / 32
```

### knee±1

| Candidate | conv2 rank | Parameters | Param reduction | Val loss | Val acc | Conv2 MACs | Conv2 reduction | Agreement | Logits RMSE | Retained energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggressive | 24 | 411,658 | 2.37% | 0.249816 | 91.02% | 1,655,808 | 54.17% | 0.9680 | 0.776895 | 0.825743 |
| Balanced / knee | **28** | **413,066** | **2.03%** | **0.230316** | **92.16%** | **1,931,776** | **46.53%** | **0.9796** | **0.578214** | **0.863418** |
| Conservative | 32 | 414,474 | 1.70% | 0.227031 | 92.06% | 2,207,744 | 38.89% | 0.9836 | 0.418914 | 0.892792 |

Notebookの `compute_reduction` はconv2単体に対する削減率。

全CNNでrank 28を使うと、

```text
Baseline total MACs   = 4,241,152
conv2 rank28 total    = 2,560,256
Total MAC reduction   = 39.63%
```

となる。

---

## Fine-tuning

条件：

```text
optimizer     = Adam
learning rate = 3e-4
max epoch     = 30
patience      = 3
seed          = 0
```

候補ごとに同じseed・shuffle順へ戻して比較した。全候補ともepoch 5でEarly Stoppingし、best epochは2。

| Candidate | rank | Acc before | Acc after | Δacc | Loss before | Loss after | Δloss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggressive | 24 | 91.02% | 92.50% | +1.48pt | 0.249816 | 0.210216 | -0.039600 |
| Balanced | **28** | 92.16% | **92.60%** | +0.44pt | 0.230316 | **0.209334** | -0.020982 |
| Conservative | 32 | 92.06% | 92.40% | +0.34pt | 0.227031 | 0.209582 | -0.017449 |

![[20_FashionMNIST/assets/cnn_conv_finetuning_delta.png]]

![[20_FashionMNIST/assets/cnn_conv_finetuning_after.png]]

Early-Stopping Validationのbest lossだけではrank32がわずかに小さいが、最終候補比較は別のRank/Final Validationで行っており、そこでrank28が `acc=0.9260`, `loss=0.209334` となったためrank28を選択した。

---

## 最終Test

| Model | conv2 rank | Parameters | Test loss | Test acc | ΔTest acc |
|---|---:|---:|---:|---:|---:|
| Baseline | — | 421,642 | **0.234131** | 91.75% | — |
| conv2 SVD + FT | **28** | **413,066** | 0.236488 | **91.97%** | **+0.22pt** |

```text
Parameter reduction = 2.03%
Total MAC reduction = 39.63%  # 全CNN換算
Test loss change     = +0.002358
```

## なぜparameter削減は小さく、MACs削減は大きいか

`conv2` のweightはbiasを除くと `18,432` 個しかないが、その同じweightを `14×14=196` 個の空間位置で繰り返し使用する。

```text
conv2 weights = 64 * 32 * 3 * 3 = 18,432
spatial positions = 14 * 14 = 196

18,432 * 196 = 3,612,672 MACs
```

一方、`fc1` はweightが `401,408` 個あり、1サンプルのforwardでは各weightを基本的に1回ずつ使うため `401,408 MACs` となる。

このモデルでは、

```text
parameterを多く持つ層  ≠  MACsを多く使う層
```

である。したがって、圧縮対象は「モデルサイズを減らしたいか」「演算量を減らしたいか」で変わる。

## full-rankでも圧縮になるとは限らない

`rank=64` は行列化したconv2のfull-rankであり、agreementは1.0まで戻る。しかし2層化後は、

```text
CNN全体 Parameters = 425,738  > Baseline 421,642
conv2 MACs          = 4,415,488 > 元conv2 3,612,672
```

となる。つまりSVD分解そのものが圧縮なのではなく、**十分小さいrankへtruncationすることで初めて圧縮になる**。今回のparameter break-even `r < 52.36` と実験結果が一致している。

## rank選択の解釈

Fine-tuningのEarly-Stopping側ではrank32のbest lossがわずかに小さい一方、最終候補比較用のRank/Final Validationではrank28が最小lossかつ最高accuracyとなった。

このためrank選択は単一の数値だけで自動的に決まるものではなく、

- どのValidation splitを最終比較に使うか
- parameter/MACsをどこまで重視するか
- lossとaccuracyのどちらを重視するか

という評価設計にも依存する。今回のrank28は、その設計の下で採用したBalancedな点である。

## 結論

- `conv2` SVDはparameter削減への効果は小さい。
- 一方で、CNN全体の理論MACsを約39.63%削減できる。
- Fine-tuningでSVD直後のValidation性能が回復した。
- rank28でTest accuracyをほぼ維持した。
- `fc1` と `conv2` では、圧縮によって得られる効果の種類が異なる。

## 関連

- [[20_FashionMNIST/07_CNN実験]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
