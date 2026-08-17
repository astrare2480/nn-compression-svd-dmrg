---
title: CNNのConvとLinear同時圧縮
aliases:
  - CNN Conv Linear SVD
  - CNN同時圧縮
  - conv2 fc1 SVD
tags:
  - Fashion-MNIST
  - CNN
  - SVD
  - Conv2d
  - Linear
  - FineTuning
---

# CNNのConvとLinear同時圧縮

## サマリー

05では、03と04で選んだrankを固定して同時に適用した。

```text
conv2 rank = 28
fc1 rank   = 24
```

```text
conv2:
Conv2d(32 → 64, 3x3)
→ Conv2d(32 → 28, 3x3)
→ Conv2d(28 → 64, 1x1)

fc1:
Linear(3136 → 128)
→ Linear(3136 → 24)
→ Linear(24 → 128)
```

最終結果：

```text
Parameters: 421,642 → 89,994     (-78.66%)
MACs:       4,241,152 → 2,237,184 (-47.25%)
Test acc:   91.75% → 91.80%       (+0.05 percentage point)
```

大幅な圧縮後もTest accuracyはほぼ同じ水準を維持した。

---

## なぜ同時圧縮するか

単独実験で役割が分かれた。

```text
fc1 SVD
→ parameter削減に大きく効く

conv2 SVD
→ MACs削減に大きく効く
```

したがって両方を組み合わせることで、**モデルサイズ側と理論計算量側を同時に減らす**ことを狙った。

---

## SVD直後

| Metric | Baseline | conv2 r28 + fc1 r24 | 変化 |
|---|---:|---:|---:|
| Parameters | 421,642 | **89,994** | **-78.66%** |
| Total MACs | 4,241,152 | **2,237,184** | **-47.25%** |
| Validation loss | 0.215624 | 0.246371 | +0.030747 |
| Validation acc | 92.54% | 91.24% | -1.30pt |
| Agreement | — | 0.9674 | — |
| Logits RMSE | — | 0.873615 | — |
| conv2 retained energy | — | 0.863418 | — |
| fc1 retained energy | — | 0.856742 | — |

単独圧縮よりも近似誤差が重なり、SVD直後のValidation accuracyはBaselineから1.30pt低下した。

### なぜ同時圧縮では誤差が重なりやすいか

今回のCNNでは `conv2` の出力が後段の `fc1` へ渡る。したがって同時圧縮では、

```text
conv2の低rank近似
→ fc1へ入る特徴量自体がBaselineから変化
→ さらにfc1も低rank近似されている
→ 2か所の近似誤差が最終logitsへ伝播
```

という構造になる。SVD directで単独圧縮よりValidation低下が大きかった結果は、このように**上流と下流の両方を同時に近似した影響が重なった可能性**と整合する。

ただし、これは今回の結果とモデル構造からの解釈であり、各層の誤差寄与を個別に分解して測定したわけではない。

### 理論MACs内訳

Baseline：

```text
conv1 =   225,792
conv2 = 3,612,672
fc1   =   401,408
fc2   =     1,280
-----------------
Total = 4,241,152
```

同時圧縮後：

```text
conv1              =   225,792
factorized conv2   = 1,931,776
factorized fc1     =    78,336
fc2                =     1,280
-----------------------------
Total              = 2,237,184
```

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

SVDで作った2層Conv/Linearの重みを初期値として使用し、SVD後に新しく作られたParameter群に対してoptimizerを作り直した。

```text
Early stopping = epoch 4
Best epoch     = 1
Best ES val loss = 0.220585
```

### 学習履歴

| epoch | Train acc | Train loss | ES Val acc | ES Val loss |
|---:|---:|---:|---:|---:|
| 1 | 95.530% | 0.130555 | **92.22%** | **0.220585** |
| 2 | 94.716% | 0.141647 | 91.56% | 0.234376 |
| 3 | 95.452% | 0.126422 | 91.86% | 0.233680 |
| 4 | 95.384% | 0.127972 | 91.76% | 0.239803 |

![[20_FashionMNIST/assets/cnn_combined_finetuning_history.png]]

---

## 同じFinal Validationで3段階比較

| Stage | Val loss | Val acc | Δacc vs Baseline | Parameters | MACs |
|---|---:|---:|---:|---:|---:|
| Baseline | **0.215624** | **92.54%** | — | 421,642 | 4,241,152 |
| SVD direct | 0.246371 | 91.24% | -1.30pt | 89,994 | 2,237,184 |
| SVD + FT | **0.222078** | **92.04%** | **-0.50pt** | 89,994 | 2,237,184 |

Fine-tuningにより、SVD直後のaccuracy低下 `-1.30pt` が `-0.50pt` まで回復した。lossも `0.246371 → 0.222078` まで回復した。

![[20_FashionMNIST/assets/cnn_combined_validation_delta.png]]

![[20_FashionMNIST/assets/cnn_combined_reduction.png]]

---

## 最終Test

| Model | Parameters | MACs | Test loss | Test acc | ΔTest acc |
|---|---:|---:|---:|---:|---:|
| Baseline | 421,642 | 4,241,152 | **0.234131** | 91.75% | — |
| conv2 r28 + fc1 r24 + FT | **89,994** | **2,237,184** | 0.240347 | **91.80%** | **+0.05pt** |

```text
Parameter reduction = 78.66%
MAC reduction       = 47.25%
Test loss change    = +0.006217
```

0.05 percentage pointはTest 10,000件なら正解数5件分に相当する。単一seedの今回の実験では、accuracy改善と主張するより**圧縮後も精度を維持した**と扱う。

---

## 固定rank同時圧縮の限界

05では、03で選んだ `fc1=24` と04で選んだ `conv2=28` をそのまま組み合わせた。したがって、

```text
conv2 rank × fc1 rank
```

の2次元rank sweepを行って同時圧縮としてのPareto最適点を探索したわけではない。

各層を単独で選んだrankが、同時圧縮時にも最良の組み合わせになる保証はない。今回の05が示したのは、**個別実験で選んだrankを組み合わせても大幅圧縮と精度維持が可能だった**ことであり、`(28, 24)` が全組み合わせ中の最適解であることではない。

---

## MACsと実測時間

05で記録された推論時間：

```text
Baseline   = 0.384253 ms
Compressed = 0.416497 ms
```

ただし、この2値は**厳密な速度比較には使わない**。05のNotebookではbenchmark条件が、

```text
Baseline   : warmup=20, repeats=2000
Compressed : warmup=5,  repeats=200
```

と揃っていないためである。したがって「圧縮モデルの方が遅い」とは結論できず、この値は参考記録として扱う。

一方で、MACsとlatencyを別指標として扱う必要がある点は変わらない。低rank化では元の1層を2層へ分割するため、実測時間には理論MACs以外にも影響しうる要素がある。

```text
1層 → 2層
→ 演算呼び出し回数の増加
→ 中間Tensorの生成・読み書き
→ メモリアクセスや実装backendの影響
→ 小規模モデルでは固定オーバーヘッドの割合が大きくなりやすい
```

これらは今回個別に計測した原因ではなく、**MACs削減がそのままlatency削減にならない場合の候補要因**である。実速度を結論するには、同じwarmup/repeats・同じbatch size・同じdeviceで複数回測定して比較する必要がある。

---

## 結論

- `fc1` と `conv2` の同時圧縮でParametersを78.66%削減した。
- CNN全体MACsを47.25%削減した。
- SVD直後の性能低下はFine-tuningで大きく回復した。
- 最終Test accuracyは91.75% → 91.80%で、ほぼ維持された。
- Test lossは少し増加した。
- latencyはbenchmark条件が揃っていないため未確定で、理論MACsとは別に同条件で再評価する必要がある。

## 関連

- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
