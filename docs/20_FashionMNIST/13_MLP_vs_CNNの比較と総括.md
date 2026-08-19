---
title: MLP vs CNNの比較と総括
aliases:
  - Fashion-MNIST MLP CNN 比較
  - Linear SVD Conv SVD 比較
  - SVD圧縮 総括
  - CIFAR-10への移行
tags:
  - Fashion-MNIST
  - MLP
  - CNN
  - SVD
  - NN圧縮
  - FineTuning
  - CIFAR-10
---

# MLP vs CNNの比較と総括

## サマリー

Fashion-MNISTでは、

```text
MLP
→ Linear SVD
→ Fine-tuning

CNN
→ Linear SVD
→ Conv SVD
→ Conv + Linear同時圧縮
```

まで進んだ。

この章の目的は、raw accuracyでMLPとCNNの優劣を決めることではない。

**ネットワーク構造が変わると、どの層を圧縮すべきか、ParametersとMACsがどう変わるかが変わる**ことを整理する。

正式結果を引用するときはcorrected版を優先する。

---

# 1. canonical結果

## MLP

```text
Baseline Parameters = 535,818
Final rank = fc1=32, fc2=16
Compressed Parameters = 57,098
Parameter reduction = 89.34%
Test acc = 88.19% → 88.53%
```

## CNN Conv + Linear

```text
Baseline Parameters = 421,642
Baseline MACs = 4,241,152
Final rank = conv2=28, fc1=24
Compressed Parameters = 89,994
Parameter reduction = 78.66%
Compressed MACs = 2,237,184
MAC reduction = 47.25%
Test acc = 91.28% → 91.33%
```

> [!important]
> MLPとCNNではmodel、run、baselineが異なる。`88.53%` と `91.33%` を直接比較して「SVD手法としてCNNの方が優れる」とは言わない。それぞれのrun内で、圧縮前後の差を見る。

---

# 2. MLPで学んだこと

MLPはほぼLinear層で構成される。

```text
784 → 512 → 256 → 10
```

大きなweight matrixを持つ `fc1` / `fc2` をSVDで低rank化すると、モデル全体のparameter数を大きく削減できる。

corrected最終rankは、

```text
fc1=32
fc2=16
```

で、Parametersを約89%削減した。

SVD直後の性能低下はFine-tuningで回復し、最終Test accuracyはbaselineとほぼ同水準だった。

ここで重要だったのは、

```text
低rank化
→ weight approximation

Fine-tuning
→ task lossへの再適応
```

を分けて考えること。

またcandidate比較では、seed・DataLoader Generator条件を揃える必要があることも確認した。

関連：

- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]

---

# 3. CNNではParametersとMACsの支配層が違う

Fashion-MNIST CNN：

```text
conv1
conv2
Flatten
fc1
fc2
```

このmodelでは、

```text
fc1
→ Parametersを非常に多く持つ

conv2
→ 同じweightを多数の空間位置で使う
→ MACsを非常に多く使う
```

という違いがある。

つまり、

```text
一番weightが多い層
=
一番計算量が多い層
```

ではない。

---

# 4. Linear SVDとConv SVDの共通点

どちらも学習済みweightへtruncated SVDを適用する。

$$
W
\approx
U_r\Sigma_rV_r^{\mathsf{T}}
$$

SVD後にdense weightへ戻すだけでは、保存parameter数は減らない。

実際の圧縮では、因子を分解したlayerとして保持する。

```text
Linear
→ Linear + Linear

Conv2d
→ spatial Conv + 1×1 Conv
```

また、truncated SVDが直接最小化するのはweight近似誤差であり、classification lossではない。

そのためFine-tuningが重要になる。

---

# 5. Linear SVD

Linear weightは最初から2次元。

```text
W: (out_features, in_features)
```

rank `r` なら、

```text
Linear(in → out)
↓
Linear(in → r, bias=False)
Linear(r → out, bias=元bias)
```

へ置換する。

Linearでは1サンプルのforwardでweightが概ね1回ずつ使われるため、parameter数とMACsは比較的似た方向に変化する。

Fashion-MNIST CNNの `fc1` も、このLinear SVDで大きくparameter数を削減できた。

---

# 6. Conv SVD

Conv2d weightは4次元Tensor。

$$
W
\in
\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

今回のSVDでは、

$$
W_{mat}
\in
\mathbb{R}^{C_{out}\times(C_{in}K_hK_w)}
$$

へ行列化する。

その後、

```text
Conv2d(C_in → C_out, K×K)
↓
Conv2d(C_in → r, K×K)
Conv2d(r → C_out, 1×1)
```

へ置換する。

Convでは同じweightを空間位置ごとに繰り返し使うため、parameter数が小さい層でもMACsが大きいことがある。

Fashion-MNISTの `conv2` がその例。

---

# 7. CNN単独圧縮から分かった役割分担

## fc1 Linear SVD

```text
fc1 rank=24
Parameters: 421,642 → 98,570
```

parameter削減に非常に強い。

## conv2 Conv SVD corrected

```text
conv2 rank=28
Parameters: 421,642 → 413,066
```

モデル全体parameter削減は約2%と小さい。

一方、Convの理論演算量を大きく減らせる。

したがって、

```text
fc1
→ model size

conv2
→ compute
```

という役割が見えた。

---

# 8. Conv + Linear同時圧縮

両方を組み合わせると、

```text
conv2=28
fc1=24
```

で、

```text
Parameters -78.66%
MACs       -47.25%
Test acc   91.28% → 91.33%
```

となった。

大幅にサイズと理論演算量を減らしながら、accuracyをほぼ維持した。

ただし、`(28,24)` は、

```text
conv2 rank × fc1 rank
```

の全組合せを探索した結果ではない。

各層単独で選んだrankを固定して組み合わせた実験なので、global optimumとは呼ばない。

関連：

- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]

---

# 9. MACsとlatencyは別物

corrected benchmarkでは、

```text
same input_batch
same batch size
same warmup
same repeats
```

へ揃えた。

Conv+Linearでは、

```text
MACs
4,241,152 → 2,237,184
-47.25%
```

にもかかわらず、latencyは、

```text
約0.378 → 0.399 ms/batch
```

で短縮しなかった。

ここから、

```text
MACs reduction
≠
wall-clock speedup
```

を明確に分ける必要がある。

低rank化ではlayer数や中間Tensorも変わるため、実速度は理論積和数だけでは決まらない。

---

# 10. retained energyとtask性能

SVDでは、

$$
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_i\sigma_i^2}
$$

で特異値エネルギー保持率を見られる。

しかし、

```text
retained energy
→ weight approximation

accuracy / loss
→ task performance
```

は別指標。

rankを増やしてretained energyが単調に増えても、validation accuracyが完全に単調増加するとは限らない。

---

# 11. Fine-tuningの共通知識

MLP / CNNの両方で、Fine-tuningによりSVD直後の性能低下を回復できた。

```text
学習済みweight
↓ SVD
低rankだがtask lossには未調整
↓ Fine-tuning
task lossへ再適応
```

SVD因子は元の学習済みweightから作るため、ランダム初期化より情報を保持した初期値になる。

ただしFine-tuningが必ずbaselineを超える保証はない。

---

# 12. single seedの制約

corrected実験は基本的にsingle seed。

```text
MLP +0.34pt
CNN Conv-only +0.47pt
CNN Combined +0.05pt
```

のような小差を、統計的なaccuracy改善とは扱わない。

主結論は、

> **圧縮後もtask accuracyをほぼ維持した**

とする。

---

# 13. Fashion-MNISTからCIFAR-10へ

Fashion-MNISTでは、

```text
single-layer rank selection
+
個別に選んだConv / Linear rankの組合せ
```

まで進んだ。

次に問題になるのは、

```text
複数の圧縮対象層へ
rankをどう配分するか
```

である。

CIFAR-10では、

```text
conv1
conv2
conv3
```

の各層を単独sweepし、Pareto / knee近傍を作り、その候補だけを組み合わせるmodel-wide rank allocationへ進んだ。

また、CIFAR-10では、

- RGB入力
- RandomCrop / HorizontalFlip
- train / evaluation transform分離
- GAP
- Dropout
- DataLoader Generator管理

といった、より現実的なCNN実験設計も学ぶ。

関連：

- [[00_基礎理論/18_CIFAR10の前処理とDataLoader]]
- [[00_基礎理論/19_再現性と乱数管理]]
- [[00_基礎理論/20_Global Average PoolingとCIFAR10モデル設計]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]

---

# 14. 結論

Fashion-MNISTから得た主要な学び：

1. MLPのLinear SVDはmodel全体のparameter削減へ直結しやすい。
2. CNNではparameter数とMACsを支配する層が異なる。
3. Linear SVDとConv SVDでは、圧縮効果の種類が異なる。
4. Fine-tuningはSVD直後のtask性能を回復する重要な工程。
5. retained energyだけではrankを決められない。
6. Pareto / kneeは圧縮率とValidation性能の折衷候補を選ぶ規則。
7. MACs削減は実測latency短縮を保証しない。
8. single seedの小さなaccuracy差は過大解釈しない。
9. 次の課題はsingle-layer rank selectionからmodel-wide rank allocationへの拡張。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[SVD実験まとめ]]
- [[30_CIFAR10_CNN/README]]
