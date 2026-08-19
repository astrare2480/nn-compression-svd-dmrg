---
title: SVD実験まとめ
aliases:
  - SVD最終まとめ
  - SVD experiment summary
  - SVD圧縮まとめ
tags:
  - SVD
  - NN圧縮
  - MNIST
  - Fashion-MNIST
  - CIFAR10
---

# SVD実験まとめ

## 位置付け

このノートは、SVD編で得た**正式結果と実験上の教訓を横断的にまとめる入口**である。

個別の式・実装・実験履歴は各ノートへ分ける。

```text
理論
→ 00_基礎理論

実装の検証・修正理由
→ 05_SVD基礎実装検証

MNIST
→ 10_MNIST_MLP_SVD

Fashion-MNIST
→ 20_FashionMNIST

CIFAR-10
→ 30_CIFAR10_CNN
```

正式結果を引用するときは `*_corrected.ipynb` と対応するcorrected resultsを優先する。

original / using_src / before_srcはhistorical recordとして残す。

---

# 1. SVD編の流れ

```text
MNIST MLP
Linear SVDの基本
validationでrank選択
        ↓
Fashion-MNIST MLP
Pareto / knee
Fine-tuning
        ↓
Fashion-MNIST CNN
Linear SVD
Conv SVD
Conv + Linear同時圧縮
        ↓
CIFAR-10 CNN
複数Convのmodel-wide rank allocation
        ↓
Tucker decomposition
```

SVDの数式を繰り返すのではなく、データセットとモデルを複雑にしながら、**圧縮対象・rank選択・評価設計**を発展させた。

---

# 2. 正式結果

## MNIST MLP

```text
Selected rank: fc1=128, fc2=128
Parameters: 約50%削減
Test acc: 98.17% → 98.08%
Δ = -0.09pt
```

rankはvalidationで選び、testは最終選択した1組だけ評価した。

[[10_MNIST_MLP_SVD/04_MNISTでの実験結果]]

## Fashion-MNIST MLP

```text
Selected rank: fc1=32, fc2=16
Parameters: 535,818 → 57,098
Reduction: 89.34%
Test acc: 88.19% → 88.53%
Δ = +0.34pt
```

single seedの小差なので、accuracy改善とは主張せず、**約89%圧縮後も精度維持**と解釈する。

[[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]

## Fashion-MNIST CNN Conv-only

```text
Selected rank: conv2=28
Parameters: 421,642 → 413,066
Reduction: 2.03%
Test acc: 91.28% → 91.75%
Δ = +0.47pt
Latency: 約0.367 → 0.370 ms/batch
```

Convはモデルparameter数への寄与が小さくても、空間位置で同じweightを繰り返し使うため、MACsへの効果が大きい。

[[20_FashionMNIST/09_CNNのConv SVD]]

## Fashion-MNIST CNN Conv + Linear

```text
Ranks: conv2=28, fc1=24
Parameters: 421,642 → 89,994
Reduction: 78.66%
MACs: 4,241,152 → 2,237,184
Reduction: 47.25%
Test acc: 91.28% → 91.33%
Δ = +0.05pt
Latency: 約0.378 → 0.399 ms/batch
```

このrank組は、単独実験で選んだrankを組み合わせたもの。`conv2 rank × fc1 rank` 全組合せの最適点ではない。

[[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]

## CIFAR-10 CNN

```text
Selected ranks: conv1=9, conv2=32, conv3=48
Parameters: 128,842 → 81,405
Reduction: 36.82%
MACs: 10,357,248 → 5,625,344
Reduction: 45.69%
Test acc: 73.27% → 73.43%
Δ = +0.16pt
Latency: 約0.531 → 0.537 ms/batch
```

探索は、各Conv単独sweepのPareto / knee近傍へ候補を絞ってから組み合わせる**制約付きmodel-wide rank allocation**。全rank空間のglobal optimumではない。

[[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]

---

# 3. 一番重要な結論

SVD編全体で確認できた中心的な傾向は、

> **適切にrankを選び、必要に応じてFine-tuningを行うことで、parameter数や理論MACsを大きく減らしながらtask accuracyをほぼ維持できる場合がある**

ということ。

一方、

```text
MACsを減らす
=
実測推論時間が短くなる
```

ではなかった。

Fashion-MNIST CNNでもCIFAR-10でも、公平化したGPU benchmarkではMACs削減がlatency短縮へ直結しなかった。

したがって、

```text
モデルサイズ
理論演算量
wall-clock latency
accuracy / loss
```

を別々の指標として測る必要がある。

---

# 4. rank選択で学んだこと

## retained energyだけでは決めない

特異値エネルギーはweight近似の指標。

```text
retained energyが高い
≠
validation accuracyが必ず高い
```

なので、task loss / accuracyも実データで評価する。

## Pareto / kneeは選択規則

Pareto frontierとkneeは、

```text
圧縮率
vs
validation性能
```

の折衷候補を絞る方法。

唯一絶対のrankを数学的に証明するものではない。

## model-wideでは探索空間が急増する

CIFAR-10では3層rankの全組合せを総当たりせず、各層単独knee近傍から候補を作った。

この経験が、Tucker以降で複数rankを扱うときの実験設計につながる。

---

# 5. Fine-tuningで学んだこと

truncated SVDは、

$$
\lVert W-W_r\rVert_F
$$

を小さくするが、classification lossを直接最適化しない。

したがって、

```text
SVD
→ weight近似として良い低rank初期値

Fine-tuning
→ task lossに対して低rank構造を再最適化
```

と分ける。

特にCIFAR-10の最終rankでは、SVD直後のValidation accuracyが `0.4792` まで落ちても、Fine-tuning後に `0.7444` まで回復した。

これはFine-tuningの役割を最も明確に示した例。

---

# 6. corrected実験で直したもの

SVDの式そのものではなく、**実験設計と実装契約**を主に修正した。

```text
test leakage
candidate間seed不公平
DataLoader Generator消費
benchmark条件不一致
baseline / compressed Parameter共有
rank / device / dtype / requires_grad契約
sweepでcandidate modelを保持しすぎる設計
CIFAR探索をglobal optimumと呼ぶ表現
```

詳しくは、

[[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

を参照。

---

# 7. 今後も守る評価規則

```text
train
→ parameter学習

validation
→ rank選択
→ candidate選択
→ Early Stopping
→ Fine-tuning条件

test
→ 最終モデル決定後だけ
```

candidate間ではseed / DataLoader Generator条件を揃える。

latency benchmarkでは、

```text
same input batch
same batch size
same warmup
same repeats
proper device synchronization
```

を守る。

---

# 8. single seedの制約

corrected実験は基本的にsingle seed。

したがって、

```text
+0.05pt
+0.16pt
+0.34pt
+0.47pt
```

などの小さなaccuracy差を「改善」と強く主張しない。

中心表現は、

> **精度をほぼ維持した**

とする。

複数seed平均・標準偏差は、より厳密な統計的比較を行う場合の次の課題。

---

# 9. Tuckerへ進む理由

Conv SVDでは、4次元weight、

$$
W \in \mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

を、

$$
W_{mat} \in \mathbb{R}^{C_{out}\times(C_{in}K_hK_w)}
$$

へ行列化した。

これはSVDを使うために複数modeを1つへまとめている。

次のTucker decompositionでは、Conv weightをテンソルのまま見て、

```text
out channel mode
in channel mode
spatial mode
```

のような多モード構造をより直接扱う。

SVD編で学んだ、

```text
rank
圧縮率
近似誤差
Fine-tuning
Pareto / knee
validation / test分離
MACs / latency
```

は、そのままTucker実験の評価基盤になる。

---

# 関連

- [[00_基礎理論/README]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[10_MNIST_MLP_SVD/04_MNISTでの実験結果]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
