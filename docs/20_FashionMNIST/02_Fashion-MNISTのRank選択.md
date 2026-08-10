---
title: Fashion-MNISTのRank選択
aliases:
  - Fashion-MNIST Pareto
  - Fashion-MNIST knee point
  - Fashion-MNIST rank sweep
tags:
  - Fashion-MNIST
  - SVD
  - Rank
  - Pareto
  - KneePoint
  - NN圧縮
---

# Fashion-MNISTのRank選択

## サマリー

最新 `02_mlp_svd_rank_selection.ipynb` では30個のrank pairを評価し、

```text
parameters ↓
validation loss ↓
```

の2目的でPareto frontierを作った。
Pareto点だけをMin-Max正規化し、両端を結ぶ直線から最も遠い点をglobal kneeとした。

最新 `02` の結果：

```text
Aggressive   = (32, 32)
Balanced     = (64, 16)  <- knee
Conservative = (64, 32)
```

> [!important]
> これは**02 run内の候補名**。03はBaselineから独立に再実行するので、03のknee ±1は別の組になる。

---

## 1. Baseline学習

```text
Train                       50,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test                        10,000
```

| 項目 | 値 |
|---|---:|
| Early stopping | epoch 17 |
| Best epoch | 12 |
| Best Early-Stopping Validation loss | 0.2997 |
| Baseline parameters | 535,818 |

---

## 2. rank候補

```python
fc1: [16, 32, 64, 128, 256, 512]
fc2: [16, 32, 64, 128, 256]
```

直積30条件。

---

## 3. 評価指標

rank pairごとに、

- validation loss / accuracy
- accuracy drop
- parameters / parameter reduction
- MACs / compute reduction
- 推論時間
- prediction agreement
- logits RMSE
- retained energy

を計算した。

`evaluate()` は `model.eval()` + `torch.no_grad()`、Agreement / logits RMSEは `torch.inference_mode()` を使う。

---

## 4. retained energy

### fc1

|   fc1_rank |   retained_energy_fc1 |
|-----------:|----------------------:|
|  16.000000 |              0.545871 |
|  32.000000 |              0.706678 |
|  64.000000 |              0.837824 |
| 128.000000 |              0.927048 |
| 256.000000 |              0.979764 |
| 512.000000 |              1.000000 |

### fc2

|   fc2_rank |   retained_energy_fc2 |
|-----------:|----------------------:|
|  16.000000 |              0.595510 |
|  32.000000 |              0.717894 |
|  64.000000 |              0.836574 |
| 128.000000 |              0.942325 |
| 256.000000 |              1.000000 |

![[20_FashionMNIST/assets/rank_selection_retained_energy.png]]

retained energyは重み再構成の補助指標であり、rank選択の主目的にはしない。

---

## 5. Pareto frontier

最新runのParetoは **11点**。

|   fc1_rank |   fc2_rank |    parameters |   validation_loss |   parameters_normalize |   validation_loss_normalize |
|-----------:|-----------:|--------------:|------------------:|-----------------------:|----------------------------:|
|  16.000000 |  16.000000 |  36362.000000 |          0.629381 |               0.000000 |                    1.000000 |
|  32.000000 |  16.000000 |  57098.000000 |          0.380187 |               0.052224 |                    0.273265 |
|  32.000000 |  32.000000 |  69386.000000 |          0.376715 |               0.083172 |                    0.263139 |
|  64.000000 |  16.000000 |  98570.000000 |          0.304215 |               0.156673 |                    0.051704 |
|  64.000000 |  32.000000 | 110858.000000 |          0.300669 |               0.187621 |                    0.041361 |
|  64.000000 |  64.000000 | 135434.000000 |          0.297295 |               0.249516 |                    0.031521 |
|  64.000000 | 128.000000 | 184586.000000 |          0.297146 |               0.373308 |                    0.031089 |
| 128.000000 |  32.000000 | 193802.000000 |          0.291303 |               0.396518 |                    0.014048 |
| 128.000000 |  64.000000 | 218378.000000 |          0.288164 |               0.458414 |                    0.004894 |
| 128.000000 | 128.000000 | 267530.000000 |          0.287246 |               0.582205 |                    0.002215 |
| 256.000000 | 128.000000 | 433418.000000 |          0.286486 |               1.000000 |                    0.000000 |

![[20_FashionMNIST/assets/rank_selection_pareto.png]]

---

## 6. knee point

```text
distance = 0.5597619332048335
parameters_normalize      = 0.15667311411992263
validation_loss_normalize = 0.05170396824161927
knee_pos = 3
```

knee：

|   fc1_rank |   fc2_rank |   parameters |   parameters_reduction |   validation_loss |   validation_acc |   accuracy_drop |   compressed_macs |   compute_reduction |   baseline_time_ms |   compressed_time_ms |   agreement |   logits_rmse |   retained_energy_fc1 |   retained_energy_fc2 |   parameters_normalize |   validation_loss_normalize |
|-----------:|-----------:|-------------:|-----------------------:|------------------:|-----------------:|----------------:|------------------:|--------------------:|-------------------:|---------------------:|------------:|--------------:|----------------------:|----------------------:|-----------------------:|----------------------------:|
|  64.000000 |  16.000000 | 98570.000000 |               0.816038 |          0.304215 |         0.892000 |        0.003400 |      97792.000000 |            0.817225 |           0.135121 |             0.162425 |    0.955600 |      1.807795 |              0.837824 |              0.595510 |               0.156673 |                    0.051704 |

knee ±1：

|   fc1_rank |   fc2_rank |    parameters |   parameters_reduction |   validation_loss |   validation_acc |   accuracy_drop |   compressed_macs |   compute_reduction |   baseline_time_ms |   compressed_time_ms |   agreement |   logits_rmse |   retained_energy_fc1 |   retained_energy_fc2 |   parameters_normalize |   validation_loss_normalize |
|-----------:|-----------:|--------------:|-----------------------:|------------------:|-----------------:|----------------:|------------------:|--------------------:|-------------------:|---------------------:|------------:|--------------:|----------------------:|----------------------:|-----------------------:|----------------------------:|
|  32.000000 |  32.000000 |  69386.000000 |               0.870505 |          0.376715 |         0.864600 |        0.030800 |      68608.000000 |            0.871770 |           0.135121 |             0.155571 |    0.917600 |      2.657922 |              0.706678 |              0.717894 |               0.083172 |                    0.263139 |
|  64.000000 |  16.000000 |  98570.000000 |               0.816038 |          0.304215 |         0.892000 |        0.003400 |      97792.000000 |            0.817225 |           0.135121 |             0.162425 |    0.955600 |      1.807795 |              0.837824 |              0.595510 |               0.156673 |                    0.051704 |
|  64.000000 |  32.000000 | 110858.000000 |               0.793105 |          0.300669 |         0.893000 |        0.002400 |     110080.000000 |            0.794258 |           0.135121 |             0.154388 |    0.968200 |      1.173226 |              0.837824 |              0.717894 |               0.187621 |                    0.041361 |

したがって02内では、

| Candidate | fc1 | fc2 | Parameters | Val loss | Val acc |
|---|---:|---:|---:|---:|---:|
| Aggressive | 32 | 32 | 69,386 | 0.376715 | 86.46% |
| Balanced | 64 | 16 | 98,570 | 0.304215 | 89.20% |
| Conservative | 64 | 32 | 110,858 | 0.300669 | 89.30% |

となった。

---

## 7. Reference Test

02にはknee ±1をTestでも確認するセルが残るが、**この結果はrank選択に使わない**。

|   fc1_rank |   fc2_rank |   test_acc |   test_loss |   accuracy_drop_raw |    parameters |   compressed_macs |   compressed_time_ms |   agreement |
|-----------:|-----------:|-----------:|------------:|--------------------:|--------------:|------------------:|---------------------:|------------:|
|  32.000000 |  32.000000 |   0.856100 |    0.419000 |            0.039300 |  69386.000000 |      68608.000000 |             0.134000 |    0.916500 |
|  64.000000 |  16.000000 |   0.877900 |    0.349700 |            0.017500 |  98570.000000 |      97792.000000 |             0.223000 |    0.950800 |
|  64.000000 |  32.000000 |   0.877800 |    0.348800 |            0.017600 | 110858.000000 |     110080.000000 |             0.172000 |    0.962700 |

> [!warning]
> 参考セルの `accuracy_drop_raw` はRank-Selection ValidationのBaseline accuracyとTest accuracyを引いているため、厳密な同一dataset上のdropではない。正式な選択・考察には使わない。

---

## 8. 02から分かること

- `16/16` は93%以上圧縮できる一方、SVD直後のvalidation lossは0.629381まで悪化した。
- `64/16` ではparameterを約81.6%削減しつつvalidation accuracy 89.20%、loss 0.304215まで回復する。
- `64/32` は少し大きいがloss 0.300669で、kneeの右隣として残る。
- 高rankほど再構成は良くなるが、最大rankの2因子表現は元モデルよりparameterが増えるため圧縮ではない。
- global kneeはrun依存であり、03では別のkneeになる。単一runのkneeを普遍的最適rankとみなさない。

---

## 関連

- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[00_基礎理論/05_圧縮率とRank]]
