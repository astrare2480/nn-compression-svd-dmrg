---
title: CNNでの実験結果
aliases:
  - Fashion-MNIST CNN SVD圧縮結果
  - CNN SVD最終結果
  - CNN圧縮比較
tags:
  - Fashion-MNIST
  - CNN
  - SVD
  - NN圧縮
  - 実験結果
---

# CNNでの実験結果

## サマリー

Fashion-MNIST CNNでは、段階的に、

```text
Linear-only SVD
↓
Conv-only SVD
↓
Conv + Linear同時圧縮
```

を行った。

重要なのは、**3実験を1つの同一runとして混ぜないこと**。

Linear-onlyの02/03は、それ自体のbaselineとの比較として有効である。

Conv-only / Conv+Linearは、benchmark条件などを修正したcorrected 04/05を正式結果とする。

```text
Linear-only
→ 03_cnn_linear_svd_finetuning のrun内で比較

Conv-only
→ 04_cnn_conv_svd_corrected のrun内で比較

Conv+Linear
→ 05_cnn_conv_linear_svd_corrected のrun内で比較
```

各実験のabsolute accuracyを横並びにして「どれが最も高性能」と比較するより、**各run内のbaselineとの差と圧縮量**を見る。

---

# 1. モデル構造

Fashion-MNIST CNN：

```text
Input (1, 28, 28)
↓
conv1: 1 → 32, 3×3
ReLU + MaxPool
↓
conv2: 32 → 64, 3×3
ReLU + MaxPool
↓
Flatten: 64×7×7 = 3136
↓
fc1: 3136 → 128
ReLU
↓
fc2: 128 → 10
```

総Parametersは、

```text
421,642
```

fc1がparameter数の大半を持ち、conv2がMACsの大半を使う。

この構造により、

```text
Linear圧縮
→ モデルサイズ削減

Conv圧縮
→ 理論演算量削減
```

の違いを観測できた。

---

# 2. Linear-only SVD

対象：

```text
fc1 = Linear(3136 → 128)
```

採用rank：

```text
fc1 rank = 24
```

この実験では、03のrun内で、

```text
Parameters: 421,642 → 98,570
Parameter reduction: 76.62%

Test acc: 91.75% → 91.87%
```

となった。

差は `+0.12pt` と小さいため、accuracy改善ではなく**大幅parameter削減後も精度を維持した**と解釈する。

Linear-onlyのbenchmarkはbaseline / compressedで同じwarmup / repeatsを使っており、旧04/05で見つかったbenchmark不整合の対象ではない。

詳細は [[20_FashionMNIST/08_CNNのLinear SVD]] を参照。

> [!important]
> Linear-only runのbaseline Test acc `91.75%` と、corrected 04/05のbaseline `91.28%` は別runの値。1つの共通baselineとして混ぜない。

---

# 3. Conv-only corrected

対象：

```text
conv2 = Conv2d(32 → 64, 3×3)
```

corrected direct sweepのknee近傍：

```text
20 / 24 / 28
```

Fine-tuning後に最終採用：

```text
conv2 rank = 28
```

### Validation after FT

| rank | Val acc | Val loss |
|---:|---:|---:|
| 20 | 0.9252 | 0.212728 |
| 24 | **0.9254** | 0.210294 |
| **28** | 0.9250 | **0.209190** |

rank28はaccuracy最大ではなく、最終比較でloss最小だったcandidateとして採用した。

### Test

| Model | Parameters | Test loss | Test acc |
|---|---:|---:|---:|
| Baseline | 421,642 | 0.238263 | 0.9128 |
| conv2 SVD + FT | **413,066** | **0.232226** | **0.9175** |

```text
Parameter reduction = 2.03%
Test acc delta       = +0.47pt
```

single seedなので改善とは断定しない。

### Latency

corrected benchmark：

```text
batch size = 256
same input_batch
warmup = 20
repeats = 2000

Baseline   ≈ 0.367 ms/batch
Compressed ≈ 0.370 ms/batch
```

理論MACsは減るが、wall-clock latencyは短くならなかった。

---

# 4. Conv + Linear corrected

単独実験で採用したrankを固定して同時に適用した。

```text
conv2 rank = 28
fc1 rank   = 24
```

これは2次元rank sweepのglobal optimumではない。

### 圧縮量

| 項目 | Baseline | Compressed |
|---|---:|---:|
| Parameters | 421,642 | **89,994** |
| Parameter reduction | — | **78.66%** |
| MACs | 4,241,152 | **2,237,184** |
| MAC reduction | — | **47.25%** |

### Validation

| Stage | Val loss | Val acc |
|---|---:|---:|
| Baseline | **0.220664** | **0.9214** |
| SVD direct | 0.241923 | 0.9116 |
| SVD + FT | **0.225938** | **0.9188** |

Fine-tuningにより、SVD directのaccuracy低下を `-0.98pt → -0.26pt` まで回復した。

### Test

| Model | Test loss | Test acc |
|---|---:|---:|
| Baseline | **0.238263** | 0.9128 |
| Conv2 + fc1 SVD + FT | 0.241242 | **0.9133** |

差は `+0.05pt`。

大幅圧縮後も**ほぼ同じaccuracyを維持した**と解釈する。

---

# 5. ParameterとMACsの役割分担

このCNNでは、

```text
fc1
→ 巨大なweight matrix
→ Parametersへの寄与が大きい

conv2
→ weight自体はfc1より少ない
→ 同じweightを14×14の多数位置で使う
→ MACsへの寄与が大きい
```

したがって、

```text
parameterを減らしたい
→ Linear SVDが効きやすい

演算量を減らしたい
→ Conv SVDが効きやすい

両方減らしたい
→ Conv + Linear
```

という整理になる。

圧縮対象は「一番parameterが多い層」だけで決めるのではなく、目的指標によって選ぶ。

---

# 6. Fine-tuningの意味

SVDは重み行列に対する最良低rank近似を与えるが、task lossそのものを直接最小化していない。

```text
学習済みweight
↓ truncated SVD
低rank初期値
↓ Fine-tuning
低rank制約の中でtask lossへ再適応
```

Fashion-MNIST CNNでは、Conv-only / CombinedともにFine-tuningでValidation性能が回復した。

ただし、single seedの小さなaccuracy差から、

```text
SVDに正則化効果がある
SVDで精度が上がる
```

とは一般化しない。

---

# 7. MACsとlatency

Conv+Linear corrected：

```text
MACs
4,241,152 → 2,237,184
-47.25%
```

一方、同条件latencyは、

```text
Baseline   ≈ 0.378 ms/batch
Compressed ≈ 0.399 ms/batch
```

で改善しなかった。

この結果は、

```text
理論演算量
≠
実装上の実行時間
```

を示す。

低rank因子化では、

- layer数
- kernel launch
- 中間Tensor
- memory access
- GEMM / convolution shape
- backend最適化

なども影響し得る。

ただし各要因を分離して測定したわけではないので、一般的な速度低下と結論しない。

---

# 8. correctedで直した実験設計

Conv-only / Combinedのcorrected版では、主に、

```text
baseline / compressedで同じinput batchを使用
warmup / repeatsを統一
train指標評価でshuffle Generatorを進めない
current src APIを使用
corrected専用resultsへ保存
```

を徹底した。

SVDの数式を変更したのではなく、**比較条件と再現性を正した**。

詳細は、

- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

を参照。

---

# 9. Single seedの制約

今回の実験はsingle seed。

```text
Linear-only  +0.12pt
Conv-only    +0.47pt
Combined     +0.05pt
```

という小差を、統計的な性能改善とは扱わない。

主結論は、

> **SVDで大きくparameter / MACsを削減しても、Fine-tuningによりtask accuracyをほぼ維持できた**

である。

---

# 10. 次のCIFAR-10へ何を持ち越したか

Fashion-MNISTでは、

```text
単一Linearのrank選択
単一Convのrank選択
個別に選んだConv + Linearの組合せ
```

まで確認した。

次のCIFAR-10では、

```text
conv1
conv2
conv3
```

の複数Convへ、rankをどう配るかを扱う。

つまり、

```text
single-layer rank selection
↓
model-wide rank allocation
```

へ進む。

---

# 11. 結論

1. Linear SVDはparameter削減に非常に効いた。
2. Conv SVDはMACs削減に効いた。
3. Conv+LinearではParametersを約79%、MACsを約47%削減できた。
4. Fine-tuningでSVD直後のtask性能を大きく回復できた。
5. corrected結果では大幅圧縮後もaccuracyをほぼ維持した。
6. MACs削減はGPU latency短縮を保証しなかった。
7. Linear-onlyとcorrected 04/05は別runなのでabsolute baselineを混ぜない。
8. `(conv2=28, fc1=24)` は同時圧縮のglobal optimumではない。

---

# 関連

- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
