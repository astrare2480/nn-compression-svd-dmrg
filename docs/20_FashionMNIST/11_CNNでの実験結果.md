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

Fashion-MNIST CNNでは、Linear圧縮・Conv圧縮・Conv+Linear同時圧縮を比較した。

現在、最終結果を引用するときはcorrected Notebookを優先する。

特に、

```text
04_cnn_conv_svd_corrected.ipynb
05_cnn_conv_linear_svd_corrected.ipynb
```

では、旧04/05で不揃いだったlatency benchmark条件を修正し、baseline / compressedを同一 `input_batch`、同一warmup/repeatsで比較した。

---

# 1. Baseline

圧縮実験側で共通に使うCNN：

```text
Parameters = 421,642
MACs       = 4,241,152
Test loss  = 0.238263
Test acc   = 0.9128
```

このモデルでは、parameter数とMACsを支配する層が異なる。

```text
fc1
→ parameter数を大きく持つ

conv2
→ 同じweightを多くの空間位置で使うためMACsが大きい
```

この違いが、Linear SVDとConv SVDの役割分担につながる。

---

# 2. Conv-only corrected

対象：

```text
conv2 = Conv2d(32 → 64, 3x3)
```

最終rank：

```text
conv2 rank = 28
```

### 最終Test

| Model | Parameters | Test loss | Test acc |
|---|---:|---:|---:|
| Baseline | 421,642 | 0.238263 | 0.9128 |
| conv2 SVD + FT | **413,066** | **0.232226** | **0.9175** |

Parameter reductionは約 **2.03%**。

Accuracy差は、

```text
+0.47 percentage point
```

だが、single seedなので性能改善とは断定しない。

### Latency

corrected版では、

```text
batch        = 256
warmup       = 20
repeats      = 2000
same input_batch
```

へ条件を統一した。

```text
Baseline   ≈ 0.367 ms/batch
Compressed ≈ 0.370 ms/batch
```

MACsは減っているが、実測latencyは短くならなかった。

---

# 3. Conv + Linear corrected

単独実験で選んだrankを固定して同時に適用した。

```text
conv2 rank = 28
fc1 rank   = 24
```

これは `conv2 rank × fc1 rank` の全組合せを探索した結果ではない。

> **個別に選んだrankを組み合わせた複合圧縮実験**

として扱う。

### 圧縮量

| 項目 | Baseline | Conv2 + fc1 SVD |
|---|---:|---:|
| Parameters | 421,642 | **89,994** |
| Parameter reduction | — | **78.66%** |
| MACs | 4,241,152 | **2,237,184** |
| MAC reduction | — | **47.25%** |

### Validation

Fine-tuning前：

```text
Val acc  = 0.9116
Val loss = 0.241923
```

Fine-tuning後：

```text
Val acc  = 0.9188
Val loss = 0.225938
```

Fine-tuningにより、同時圧縮で生じた性能低下をかなり回復した。

### 最終Test

| Model | Test loss | Test acc |
|---|---:|---:|
| Baseline | **0.238263** | 0.9128 |
| Conv2 + fc1 SVD + FT | 0.241242 | **0.9133** |

Accuracy差：

```text
+0.05 percentage point
```

この差は極小であり、**精度維持**と解釈する。

---

# 4. Corrected latency

Conv+Linear correctedの公平なbenchmark：

```text
Baseline   ≈ 0.378 ms/batch
Compressed ≈ 0.399 ms/batch
```

一方、理論MACsは、

```text
4,241,152 → 2,237,184
-47.25%
```

まで減っている。

つまり、今回の実装・GPU・batchでは、

```text
MACs -47%
でも
wall-clock latencyは改善しない
```

という結果になった。

これは失敗として隠すのではなく、重要な実験結果として扱う。

ただし、

> SVD圧縮は一般に遅くなる

とは結論しない。

低rank2層化では、kernel launch、中間Tensor、memory access、行列shapeなどの影響も受けるため、**理論演算量と実速度は別に測定する必要がある**という結論までに留める。

---

# 5. Parameter削減とMACs削減の役割分担

このCNNでは、

```text
fc1 SVD
→ parameter削減に大きく効く

conv2 SVD
→ MACs削減に大きく効く
```

という違いがある。

そのため、Conv+Linear同時圧縮では、

```text
Parameters -78.66%
MACs       -47.25%
```

を同時に実現できた。

ここから、圧縮対象層は、

```text
モデルサイズを減らしたいのか
演算量を減らしたいのか
```

によって変わることが分かる。

---

# 6. Fine-tuningの意味

SVD直後の低rankモデルは元の学習済みweightを近似しているが、task lossを直接最適化した低rankモデルではない。

Fine-tuningでは、

```text
SVDで低rank制約を入れる
        ↓
その低rankモデルを初期値にする
        ↓
task lossで再最適化
```

を行う。

Fashion-MNISTでは、SVD直後の性能低下を短いFine-tuningで大きく回復できた。

ただしFine-tuning後の小さなaccuracy上昇を、SVDの正則化効果などとして一般化しない。

---

# 7. Single seedの制約

corrected実験は `SEED=0` のsingle run。

Conv-onlyの `+0.47pt`、Conv+Linearの `+0.05pt` は、複数seedで再現確認していない。

したがって、本ノートでは、

> **大きな圧縮後もaccuracyをほぼ維持した**

ことを中心結論とする。

---

# 8. Historical 04/05との違い

旧04/05では、baselineとcompressedでbenchmarkのwarmup / repeatsが揃っていなかった。

そのため旧latency値はhistorical recordとして残すが、正式な速度比較には使わない。

corrected版では、

```text
same input_batch
same batch size
warmup=20
repeats=2000
```

へ統一した。

詳細は [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] を参照。

---

# 9. 結論

Fashion-MNIST CNNから得た主要な結果は次のとおり。

1. LinearとConvでは、parameter数・MACsへの効き方が異なる。
2. Conv-onlyではparameter削減は小さいが、理論演算量を大きく減らせる。
3. Conv+LinearではParametersを約79%、MACsを約47%削減できた。
4. Fine-tuningにより低rank近似後のtask性能をかなり回復できる。
5. corrected Testでは大幅圧縮後もaccuracyをほぼ維持した。
6. MACs削減はGPU latency短縮を保証しなかった。
7. `(conv2=28, fc1=24)` は同時圧縮の全rank空間における最適解ではない。

この段階までで、単一層圧縮から複数種類の層の同時圧縮まで確認した。次のCIFAR-10では、複数Conv層の**model-wide rank allocation**へ進む。

---

# 関連

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
