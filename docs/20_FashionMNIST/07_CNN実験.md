---
title: Fashion-MNIST CNN実験
aliases:
  - Fashion-MNIST CNN
  - CNN SVD実験
  - Fashion-MNIST CNN SVD圧縮
tags:
  - Fashion-MNIST
  - CNN
  - SVD
  - PyTorch
  - NN圧縮
---

# Fashion-MNIST CNN実験

## 目的

MLPで確認したLinear SVDをCNNへ拡張し、

```text
CNN Baseline
→ fc1 Linear SVD
→ conv2 Conv SVD
→ conv2 + fc1 同時圧縮
```

まで検証した。

この段階で特に重要だったのは、**parameter数を支配する層とMACsを支配する層が一致しない**こと。

```text
fc1
→ parameter数を大きく持つ

conv2
→ 同じweightを多数の空間位置で使う
→ MACsが大きい
```

そのため、Linear SVDとConv SVDでは圧縮効果の性質が異なる。

---

# 1. CNN構造

```text
Input (N, 1, 28, 28)
  ↓ Conv2d(1, 32, 3x3, padding=1)
  ↓ ReLU
  ↓ MaxPool2d(2)
(N, 32, 14, 14)
  ↓ Conv2d(32, 64, 3x3, padding=1)
  ↓ ReLU
  ↓ MaxPool2d(2)
(N, 64, 7, 7)
  ↓ Flatten
(N, 3136)
  ↓ Linear(3136, 128)
  ↓ ReLU
  ↓ Linear(128, 10)
Output (N, 10)
```

Parameter内訳：

| Parameter | shape | 数 |
|---|---:|---:|
| `conv1.weight` | `(32, 1, 3, 3)` | 288 |
| `conv1.bias` | `(32,)` | 32 |
| `conv2.weight` | `(64, 32, 3, 3)` | 18,432 |
| `conv2.bias` | `(64,)` | 64 |
| `fc1.weight` | `(128, 3136)` | 401,408 |
| `fc1.bias` | `(128,)` | 128 |
| `fc2.weight` | `(10, 128)` | 1,280 |
| `fc2.bias` | `(10,)` | 10 |
| **合計** | — | **421,642** |

`fc1` は全parameterの約95%を占める。

一方、畳み込みはweightを空間位置ごとに再利用するため、`conv2` はparameter数以上にMACsへ効く。

---

# 2. Notebook世代と位置づけ

CNN実験には次のNotebookがある。

| Notebook | 内容 | 現在の扱い |
|---|---|---|
| `01_cnn_baseline.ipynb` | CNN構造・Baseline確認 | 学習用Baseline run |
| `02_cnn_linear_svd.ipynb` | `fc1` rank sweep | 有効な独立run |
| `03_cnn_linear_svd_finetuning.ipynb` | 旧`fc1` Fine-tuning | historical（旧共有train評価処理） |
| `03_cnn_linear_svd_finetuning_rerun.ipynb` | 現行`fc1` Fine-tuning | **canonical、最終rank=28** |
| `04_cnn_conv_svd.ipynb` | 旧Conv-only | historical。latency条件に問題あり |
| `04_cnn_conv_svd_corrected.ipynb` | Conv-only corrected | 従来corrected run（保存） |
| `04_cnn_conv_svd_corrected_rerun.ipynb` | Conv-only corrected rerun | **canonical** |
| `05_cnn_conv_linear_svd.ipynb` | 旧Conv+Linear | historical。latency条件に問題あり |
| `05_cnn_conv_linear_svd_corrected.ipynb` | Conv+Linear corrected | 従来corrected run（保存） |
| `05_cnn_conv_linear_svd_corrected_rerun.ipynb` | Conv+Linear corrected rerun | **canonical** |

現在の数値は2026-09-12の独立rerunを出典とする。元02/03・corrected 04/05も残す。各rerunでbaselineを学習・保存し、値が一致しても同一runのcheckpointとして流用しない。出典は [[20_FashionMNIST/11_CNNでの実験結果]] を参照。

---

# 3. 01 Baselineは構造確認用

`01_cnn_baseline.ipynb` では、

```text
Train       55,000
Validation   5,000
Test        10,000
```

を使い、CNNが正常に学習できることを確認した。

代表値：

```text
Validation accuracy ≈ 91.94%
Test accuracy       ≈ 90.90%
Parameters          = 421,642
```

このrunは後続の圧縮前後比較Baselineとは学習条件が異なるため、圧縮効果を計算する基準にはしない。

---

# 4. Linear-only 03 rerun

CNN後段の、

```text
fc1 = Linear(3136, 128)
```

をSVDで2層へ置換した。

```text
Linear(3136 → r, bias=False)
Linear(r → 128, bias=True)
```

rank sweepとFine-tuningの結果、

```text
fc1 rank = 28
Parameters 421,642 → 111,626
Test acc   0.9128  → 0.9149
```

となった。

direct sweepのkneeは24、FT候補は20/24/28。FT後の最小rank-selection validation lossにより最終rank=28となった。旧02/03のrank=24と性能値はhistoricalとして残す。

詳細：[[20_FashionMNIST/08_CNNのLinear SVD]]

---

# 5. Conv-only 04 corrected

対象：

```text
conv2 = Conv2d(32, 64, 3x3, padding=1)
```

行列化：

```text
conv2.weight
(64, 32, 3, 3)
→
(64, 288)
```

corrected final：

```text
conv2 rank = 28
Parameters 421,642 → 413,066  (-2.03%)
Test acc   0.9128 → 0.9175
```

parameter削減は小さいが、ConvのMACs削減へ大きく効く。

詳細：[[20_FashionMNIST/09_CNNのConv SVD]]

---

# 6. Conv + Linear 05 corrected

従来の単独実験で採用した固定条件、

```text
conv2 rank = 28
fc1 rank   = 24
```

を同時に適用した。新Linear-onlyの最終fc1=28へ変更せず、元の固定fc1=24を維持した。

これは `conv2 rank × fc1 rank` の全組合せ探索ではない。

```text
個別に選択したrank
→ 同時適用
```

という実験。

corrected final：

```text
Parameters 421,642 → 89,994      (-78.66%)
MACs       4,241,152 → 2,237,184 (-47.25%)
Test acc   0.9128 → 0.9133
```

大きなparameter / MACs削減後もaccuracyをほぼ維持した。

詳細：[[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]

---

# 7. corrected 04 / 05で直したbenchmark

旧04 / 05では、Baselineとcompressedでlatency benchmarkの条件が揃っていなかった。

例：

```text
Baseline
warmup=20
repeats=2000

Compressed
warmup=5
repeats=200
```

correctedでは、

```text
same input_batch
same batch size = 256
warmup = 20
repeats = 2000
```

へ統一した。

以下は最終FT後と同一run baselineの3 trial平均時間の中央値であり、FT前sweep時間とは区別する。

### Conv-only

```text
Baseline   ≈ 0.344271 ms/batch
Final FT   ≈ 0.353625 ms/batch
```

### Conv + Linear

```text
Baseline   ≈ 0.346444 ms/batch
Final FT   ≈ 0.383861 ms/batch
```

理論MACsは減ったが、実測latencyは短縮しなかった。

したがって、

```text
MACs reduction
≠
wall-clock speedup
```

を明確に分ける。

---

# 8. Fine-tuning

SVD直後の因子はweight近似としては合理的だが、task lossを直接最適化した低rank modelではない。

```text
SVD
→ 低rank構造 + 学習済みweight由来の初期値

Fine-tuning
→ task lossへ再適応
```

CNNでも、SVD直後の性能低下を短いFine-tuningで回復できた。

ただし、single seedの小さなaccuracy上昇を「SVDにより性能改善」とは一般化しない。

---

# 9. Fashion-MNIST CNNで得たこと

1. LinearとConvでparameter / MACsへの効き方が異なる。
2. `fc1` SVDはmodel size削減へ強く効く。
3. `conv2` SVDはparameter削減は小さいがMACs削減へ効く。
4. 同時圧縮でParametersとMACsを両方大きく削減できる。
5. Fine-tuningでSVD直後の性能を回復できる。
6. MACs削減はlatency短縮を保証しない。
7. 正式結果はcorrected Notebookを優先する。
8. 同時圧縮 `28/24` はglobal optimumではない。

この次のCIFAR-10では、単一層rank選択から、**複数Conv層へrankをどう配るか**というmodel-wide rank allocationへ進む。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
- [[20_FashionMNIST/13_MLP_vs_CNNの比較と総括]]
