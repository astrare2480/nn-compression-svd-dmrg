---
title: Fashion-MNISTのFine-tuning
aliases:
  - Fashion-MNIST SVD fine-tuning
  - Fashion-MNIST 低rank再学習
  - SVD後の性能回復
tags:
  - Fashion-MNIST
  - SVD
  - FineTuning
  - Pareto
  - NN圧縮
---

# Fashion-MNISTのFine-tuning

## サマリー

最新 `03_mlp_svd_finetuning(3).ipynb` はBaseline学習からrank sweepまで独立に再実行し、03内で得たknee ±1をFine-tuningする。

03内の候補：

```text
Aggressive   = 16 / 16
Balanced     = 32 / 16  <- SVD直後のknee
Conservative = 64 / 16
```

Fine-tuning後、`parameters + validation_loss` で再度Pareto支配を判定した結果、**Aggressiveだけが残った**。

最終モデル：

```text
Aggressive
fc1_rank = 16
fc2_rank = 16
```

---

## 1. 03は独立run

Baseline Early Stopping：

| 項目 | 値 |
|---|---:|
| Early stopping | epoch 16 |
| Best epoch | 11 |
| Best Early-Stopping Validation loss | 0.3010 |

Best state復元後、Rank-Selection ValidationでBaselineを評価すると、

| 指標 | 値 |
|---|---:|
| validation accuracy | 89.42% |
| validation loss | 0.299003 |
| parameters | 535,818 |

となった。

`02` と `03` のrank結果を混ぜず、03の最終モデルは03内のValidationだけで決める。

---

## 2. 03のSVD直後rank選択

Pareto frontierは **9点**。

![[20_FashionMNIST/assets/finetuning_rank_selection_pareto.png]]

knee：

```text
distance = 0.5862092326788827
parameters_normalize      = 0.05222437137330754
validation_loss_normalize = 0.11875058138389116
```

|   fc1_rank |   fc2_rank |   parameters |   parameters_reduction |   validation_loss |   validation_acc |   accuracy_drop |   compressed_macs |   compute_reduction |   baseline_time_ms |   compressed_time_ms |   agreement |   logits_rmse |   retained_energy_fc1 |   retained_energy_fc2 |   parameters_normalize |   validation_loss_normalize |
|-----------:|-----------:|-------------:|-----------------------:|------------------:|-----------------:|----------------:|------------------:|--------------------:|-------------------:|---------------------:|------------:|--------------:|----------------------:|----------------------:|-----------------------:|----------------------------:|
|  32.000000 |  16.000000 | 57098.000000 |               0.893438 |          0.364576 |         0.871800 |        0.022400 |      56320.000000 |            0.894737 |           0.136975 |             0.275393 |    0.930800 |      2.979520 |              0.709006 |              0.607805 |               0.052224 |                    0.118751 |

knee ±1：

|   fc1_rank |   fc2_rank |   parameters |   parameters_reduction |   validation_loss |   validation_acc |   accuracy_drop |   compressed_macs |   compute_reduction |   baseline_time_ms |   compressed_time_ms |   agreement |   logits_rmse |   retained_energy_fc1 |   retained_energy_fc2 |   parameters_normalize |   validation_loss_normalize |
|-----------:|-----------:|-------------:|-----------------------:|------------------:|-----------------:|----------------:|------------------:|--------------------:|-------------------:|---------------------:|------------:|--------------:|----------------------:|----------------------:|-----------------------:|----------------------------:|
|  16.000000 |  16.000000 | 36362.000000 |               0.932137 |          0.858264 |         0.717600 |        0.176600 |      35584.000000 |            0.933493 |           0.136975 |             0.179334 |    0.753400 |      6.248245 |              0.549914 |              0.607805 |               0.000000 |                    1.000000 |
|  32.000000 |  16.000000 | 57098.000000 |               0.893438 |          0.364576 |         0.871800 |        0.022400 |      56320.000000 |            0.894737 |           0.136975 |             0.275393 |    0.930800 |      2.979520 |              0.709006 |              0.607805 |               0.052224 |                    0.118751 |
|  64.000000 |  16.000000 | 98570.000000 |               0.816038 |          0.311055 |         0.888600 |        0.005600 |      97792.000000 |            0.817225 |           0.136975 |             0.188748 |    0.961600 |      1.592894 |              0.837559 |              0.607805 |               0.156673 |                    0.023214 |

Fine-tuning対象を要約すると、

| Candidate | fc1 | fc2 | Parameters | Val acc before FT | Val loss before FT |
|---|---:|---:|---:|---:|---:|
| Aggressive | 16 | 16 | 36,362 | 71.76% | 0.858264 |
| Balanced | 32 | 16 | 57,098 | 87.18% | 0.364576 |
| Conservative | 64 | 16 | 98,570 | 88.86% | 0.311055 |

となる。

---

## 3. Fine-tuning条件と再現性

```text
optimizer     = Adam
learning rate = 0.001
MAX_EPOCHS    = 50
PATIENCE      = 5
MIN_DELTA     = 1e-4
```

候補モデルは必ず、

```python
pareto_model = copy.deepcopy(
    pareto_row["model"]
).to(device)
```

として元のSVD候補を壊さない。

さらに最新03では候補ごとに、

```python
set_seed(SEED + 1000 * fc1_rank + fc2_rank)
```

を設定し、各候補のFine-tuningを再現可能な別seedで実行する。

---

## 4. `eval()` / `no_grad()` の確認

評価用 `evaluate()` の内部で、

```python
model.eval()
with torch.no_grad():
    ...
```

を使っている。

Fine-tuning中のEarly-Stopping Validation、Train再評価、Fine-tuning前後比較、最終Testはいずれもこの `evaluate()` を通るので、セルごとに `model.eval()` / `torch.no_grad()` が見えなくても評価自体は正しい。

Agreement / logits RMSEは `torch.inference_mode()`、推論時間計測も `eval()` + `no_grad()`。

---

## 5. Fine-tuning中のBest epoch

NotebookのhistoryからBest epoch時点のaccuracyを取り直すと、

|   fc1_rank |   fc2_rank |   best_epoch |   best_validation_loss |   validation_acc_at_best_epoch |   notebook_best_validation_acc_column |
|-----------:|-----------:|-------------:|-----------------------:|-------------------------------:|--------------------------------------:|
|  16.000000 |  16.000000 |     4.000000 |               0.332432 |                       0.878400 |                              0.878800 |
|  32.000000 |  16.000000 |     6.000000 |               0.316080 |                       0.884600 |                              0.888600 |
|  64.000000 |  16.000000 |     2.000000 |               0.329689 |                       0.882600 |                              0.881400 |

> [!warning]
> Notebook内の `best_validation_acc` 列は現在も `re_validation_acc` の最終値を入れているため、列名どおりの「Best epoch accuracy」ではない。最終モデル選択にはこの列を使っていないため、最終結果への影響はない。

---

## 6. Fine-tuning前後の正式比較

同じRank-Selection ValidationでBefore / Afterを比較する。

| candidate    |   fc1_rank |   fc2_rank |   acc_before |   acc_after |   delta_acc |   loss_before |   loss_after |   delta_loss |
|:-------------|-----------:|-----------:|-------------:|------------:|------------:|--------------:|-------------:|-------------:|
| Aggressive   |         16 |         16 |     0.717600 |    0.883200 |    0.165600 |      0.858264 |     0.322567 |    -0.535697 |
| Balanced     |         32 |         16 |     0.871800 |    0.885800 |    0.014000 |      0.364576 |     0.328358 |    -0.036218 |
| Conservative |         64 |         16 |     0.888600 |    0.886200 |   -0.002400 |      0.311055 |     0.335646 |     0.024591 |

![[20_FashionMNIST/assets/finetuning_delta.png]]

![[20_FashionMNIST/assets/finetuning_after.png]]

解釈：

- **Aggressive**: accuracy +16.56pt、loss -0.535697。SVD直後の大幅劣化を大きく回復。
- **Balanced**: accuracy +1.40pt、loss -0.036218。小幅だが両指標改善。
- **Conservative**: accuracy -0.24pt、loss +0.024591。今回のrunではFine-tuning後にRank-Selection Validationが悪化。

Fine-tuningは必ず性能を改善するわけではない。
Early-Stopping Validationを使ってbest stateを選んでも、別のRank-Selection Validationでの性能が必ず改善するとは限らないことが実測で確認できた。

---

## 7. Fine-tuning後のモデル比較

| model                 | fc1_rank   | fc2_rank   |   validation_acc |   validation_loss |   parameters |   parameters_reduction |
|:----------------------|:-----------|:-----------|-----------------:|------------------:|-------------:|-----------------------:|
| Baseline              | -          | -          |         0.894200 |          0.299003 |       535818 |               0.000000 |
| Aggressive after FT   | 16         | 16         |         0.883200 |          0.322567 |        36362 |               0.932137 |
| Balanced after FT     | 32         | 16         |         0.885800 |          0.328358 |        57098 |               0.893438 |
| Conservative after FT | 64         | 16         |         0.886200 |          0.335646 |        98570 |               0.816038 |

Accuracyだけを見るとConservativeが3候補中わずかに高いが、事前に定めた主目的は、

```text
parameters ↓
validation loss ↓
```

である。

Aggressiveは、

```text
parameters      36,362  < 57,098 < 98,570
validation loss 0.322567 < 0.328358 < 0.335646
```

なのでBalanced / Conservativeを両方支配する。

---

## 8. 最終モデル選択

Fine-tuning後Pareto：

| model               |   fc1_rank |   fc2_rank |   validation_acc |   validation_loss |   parameters |   parameters_reduction |
|:--------------------|-----------:|-----------:|-----------------:|------------------:|-------------:|-----------------------:|
| Aggressive after FT |         16 |         16 |         0.883200 |          0.322567 |        36362 |               0.932137 |

最終選択：

| model               |   fc1_rank |   fc2_rank |   validation_acc |   validation_loss |   parameters |   parameters_reduction |
|:--------------------|-----------:|-----------:|-----------------:|------------------:|-------------:|-----------------------:|
| Aggressive after FT |         16 |         16 |         0.883200 |          0.322567 |        36362 |               0.932137 |

今回はPareto候補が1つだけなので、第2基準のvalidation loss優先まで使う必要なく、**Aggressiveが一意に選ばれた**。

---

## 9. 最終Test

| model                   | fc1_rank   | fc2_rank   |   parameters |   test_loss |   test_acc |   delta_test_acc |   delta_test_loss |
|:------------------------|:-----------|:-----------|-------------:|------------:|-----------:|-----------------:|------------------:|
| Baseline                | -          | -          |       535818 |    0.333783 |   0.887100 |         0.000000 |          0.000000 |
| Final (compressed + FT) | 16         | 16         |        36362 |    0.366167 |   0.871400 |        -0.015700 |          0.032384 |

```text
Baseline:
  loss = 0.333783
  acc  = 0.8871

Final Aggressive (16/16):
  loss = 0.366167
  acc  = 0.8714

accuracy delta = -0.0157 = -1.57 percentage point
loss delta     = +0.032384
```

サイズ：

```text
Parameters: 535,818 -> 36,362
reduction = 93.2137%
compression factor = 14.74x

MACs: 535,040 -> 35,584
reduction = 93.3493%
```

Testを見てから別rankへ変更しない。

---

## 10. 考察

### Aggressiveが大きく回復した

`16/16` はSVD直後にはaccuracy 71.76%、loss 0.858264と大きく崩れたが、Fine-tuning後は88.32% / 0.322567まで回復した。

したがって、SVD直後の悪化すべてを「rank不足」と解釈するのは不適切。
低rank構造自体には分類性能を取り戻す余地があり、SVDで得た初期配置がtask lossに最適ではなかった影響が大きい可能性がある。

### ただし元モデルには届いていない

最終Test accuracyはBaselineより1.57pt低く、lossも0.032384悪化した。
93%以上のparameter削減とのトレードオフとして評価する必要がある。

### AccuracyとLossで順位が異なる

Fine-tuning後、validation accuracyはConservativeの方が高いが、validation lossはAggressiveの方が小さい。
今回の機械選択は `parameters + validation_loss` を主目的に固定しているためAggressiveとなった。
選択目的をaccuracy中心に変えれば結果は変わり得るが、Testを見た後に目的関数を変更しない。

### 02と03でkneeが異なる

最新02は `64/16`、最新03は `32/16` がSVD直後knee。
独立runでPareto / kneeが動くこと自体が、単一seed・単一runのkneeを絶対視できないことを示す。
より強い結論には複数seedで平均・分散を確認する必要がある。

---

## 関連

- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
