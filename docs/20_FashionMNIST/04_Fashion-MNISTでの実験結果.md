---
title: Fashion-MNISTでの実験結果
aliases:
  - Fashion-MNIST SVD圧縮結果
  - Fashion-MNIST最終結果
  - Fashion-MNIST rank selection結果
tags:
  - Fashion-MNIST
  - SVD
  - NN圧縮
  - FineTuning
  - 実験結果
  - PyTorch
---

# Fashion-MNISTでの実験結果

## サマリー

Fashion-MNIST用MLP `784 -> 512 -> 256 -> 10` の `fc1` / `fc2` をSVD低rank2層へ置き換えた。

最新の03 runでは、SVD直後のknee ±1、

```text
Aggressive   16/16
Balanced     32/16
Conservative 64/16
```

をFine-tuningした。

Fine-tuning後に `parameters + validation_loss` でPareto判定するとAggressiveだけが残り、最終モデルになった。

## 最終結果

| 項目 | Baseline | Final: SVD + Fine-tuning |
|---|---:|---:|
| rank | — | **16 / 16** |
| Test loss | **0.333783** | 0.366167 |
| Test accuracy | **88.71%** | 87.14% |
| Parameters | 535,818 | **36,362** |
| Parameter reduction | — | **93.2137%** |
| Compression factor | — | **14.74x** |
| 理論MACs | 535,040 | **35,584** |
| MACs reduction | — | **93.3493%** |

Test accuracy差：

$$
0.8714-0.8871=-0.0157
$$

**-1.57 percentage point**。

---

## 1. Notebookごとの最新結果

| Notebook | 主な結果 |
|---|---|
| 00 fixed 50 epochs | epoch50: Test loss 0.6842 / Test acc 88.44% |
| 01 Early Stopping | best epoch 10 / val loss 0.2767 / Test acc 89.07% |
| 02 Rank Selection | best epoch 12 / Early-Stop val loss 0.2997 / knee **64/16** |
| 03 Fine-tuning | best epoch 11 / Early-Stop val loss 0.3010 / knee **32/16** / final **16/16** |

各Notebookは独立runとして扱う。

---

## 2. 50 epoch固定学習

```text
Train loss: 0.4983 -> 0.0650
Train acc : 81.93% -> 97.51%
Test loss at epoch50: 0.6842
Test acc  at epoch50: 88.44%
```

![[20_FashionMNIST/assets/baseline_fixed50_learning_curves.png]]

後半ではTrain lossが低下し続ける一方でTest lossが増加し、Train/Testの乖離が明確になった。
これが01以降でEarly Stoppingを導入する動機になっている。

この値は、shape確認用の `next(iter(train_loader))` セルを削除した最新版00を先頭から再実行した結果。

---

## 3. Early Stopping

最新01：

```text
Best epoch = 10
Best validation loss = 0.2767
Test loss = 0.3212
Test accuracy = 0.8907
```

旧結果ではなく、shape確認セル削除後の最新Notebook出力を採用する。

---

## 4. 最新02 rank selection

02のglobal kneeは **64/16**。

```text
Aggressive   32/32
Balanced     64/16
Conservative 64/32
```

| Candidate | Parameters | Val loss | Val acc |
|---|---:|---:|---:|
| Aggressive 32/32 | 69,386 | 0.376715 | 86.46% |
| Balanced 64/16 | 98,570 | 0.304215 | 89.20% |
| Conservative 64/32 | 110,858 | 0.300669 | 89.30% |

Paretoは11点。

![[20_FashionMNIST/assets/rank_selection_pareto.png]]

---

## 5. 最新03 SVD直後

03は独立runで、global kneeが **32/16** になった。

| Candidate | Parameters | Val loss before FT | Val acc before FT |
|---|---:|---:|---:|
| Aggressive 16/16 | 36,362 | 0.858264 | 71.76% |
| Balanced 32/16 | 57,098 | 0.364576 | 87.18% |
| Conservative 64/16 | 98,570 | 0.311055 | 88.86% |

02と03でkneeが一致しないことから、kneeは学習済み重み/runに依存することが分かる。

---

## 6. Fine-tuningの効果

| candidate    |   fc1_rank |   fc2_rank |   acc_before |   acc_after |   delta_acc |   loss_before |   loss_after |   delta_loss |
|:-------------|-----------:|-----------:|-------------:|------------:|------------:|--------------:|-------------:|-------------:|
| Aggressive   |         16 |         16 |     0.717600 |    0.883200 |    0.165600 |      0.858264 |     0.322567 |    -0.535697 |
| Balanced     |         32 |         16 |     0.871800 |    0.885800 |    0.014000 |      0.364576 |     0.328358 |    -0.036218 |
| Conservative |         64 |         16 |     0.888600 |    0.886200 |   -0.002400 |      0.311055 |     0.335646 |     0.024591 |

最も大きく回復したのはAggressive。
一方Conservativeは今回のRank-Selection ValidationではFine-tuning後にaccuracyもlossも悪化した。

この結果は、

> Fine-tuningは低rank制約内で再最適化するが、別Validation上で必ず改善する保証はない。

ことを示す。

---

## 7. Fine-tuning後のPareto

| model               |   fc1_rank |   fc2_rank |   validation_acc |   validation_loss |   parameters |   parameters_reduction |
|:--------------------|-----------:|-----------:|-----------------:|------------------:|-------------:|-----------------------:|
| Aggressive after FT |         16 |         16 |         0.883200 |          0.322567 |        36362 |               0.932137 |

Aggressiveが、

- 最小parameters
- 最小validation loss

の両方を満たし、他2候補を支配した。

validation accuracyだけなら他候補がわずかに高いが、今回の主目的は事前に `parameters + validation_loss` と定義している。

---

## 8. 最終Test

| model                   | fc1_rank   | fc2_rank   |   parameters |   test_loss |   test_acc |   delta_test_acc |   delta_test_loss |
|:------------------------|:-----------|:-----------|-------------:|------------:|-----------:|-----------------:|------------------:|
| Baseline                | -          | -          |       535818 |    0.333783 |   0.887100 |         0.000000 |          0.000000 |
| Final (compressed + FT) | 16         | 16         |        36362 |    0.366167 |   0.871400 |        -0.015700 |          0.032384 |

約93.2%のparameter削減と引き換えに、Test accuracyは1.57pt低下した。
以前の結果より精度維持幅は小さくなったが、圧縮率はさらに大きい。

「十分に性能維持できた」と言い切るかは用途依存であり、研究上は、

```text
93.2% parameter reduction
vs
1.57 percentage point accuracy drop
```

というトレードオフとして報告するのが適切。

---

## 9. 今回の主要な考察

1. SVD直後の極端な性能低下は、Fine-tuningで大幅に回復し得る。
2. retained energyだけではFine-tuning後のtask性能は決められない。
3. Fine-tuningは常に改善ではなく、Conservativeのように別Validationで悪化する場合がある。
4. AccuracyとCross Entropy Lossでは候補順位が異なる。
5. `parameters + validation_loss` という事前規則ではAggressiveが一意に選ばれた。
6. 02と03でkneeが変動したため、rank選択の安定性は複数seedで検証する余地がある。
7. 最終Testはモデル選択後に使用し、Test結果を見てrankを変更していない。

---

## 10. 実装上の確認

- `evaluate()` は `model.eval()` + `torch.no_grad()`。
- Agreement / logits RMSEは `torch.inference_mode()`。
- Fine-tuning候補は `copy.deepcopy()` してから更新。
- optimizer / Early-Stopping stateは候補ごとに独立。
- shape確認セルは01/02/03の最新runから削除し、実験手順には含めない。
- `best_validation_acc` 列の命名上の問題は残るが、最終比較はbest state復元後に再評価しているため最終選択へ影響しない。

---

## 完全な生結果

- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]
