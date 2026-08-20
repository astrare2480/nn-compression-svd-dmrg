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
  - Historical
---

# Fashion-MNISTのRank選択

> [!warning]
> このノートは **旧 `02_mlp_svd_rank_selection.ipynb` のhistorical run** を整理したもの。
> 現在のMLP最終rankはここから引用しない。formal resultは [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]] を参照する。

## このrunで行ったこと

`fc1` / `fc2` のrank pairを30条件評価し、

```text
parameters ↓
validation loss ↓
```

の2目的でPareto frontierを作った。

Pareto点だけをMin-Max正規化し、frontier両端を結ぶ直線から最も遠い点をkneeとした。

このrun内では、

```text
Aggressive   = (32, 32)
Balanced     = (64, 16)  <- knee
Conservative = (64, 32)
```

となった。

ただし、これは**このrunのBaseline weight・Validation結果に対するknee**であり、普遍的な最適rankではない。

---

# 1. データ分割

```text
Train                       50,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test                        10,000
```

この役割分離自体は現在も重要。

```text
Early-Stopping Validation
→ epoch選択

Rank-Selection Validation
→ rank選択

Test
→ 最終model決定後にだけ使う
```

旧02にはknee近傍をTestでも表示する参考セルが残るが、Test値はrank選択へ使わない。

---

# 2. rank候補

```python
fc1: [16, 32, 64, 128, 256, 512]
fc2: [16, 32, 64, 128, 256]
```

直積30条件。

rank pairごとに、

- Validation loss / accuracy
- Parameters / reduction
- MACs / reduction
- inference latency
- prediction agreement
- logits RMSE
- retained energy

を記録した。

この段階で、**rank選択をaccuracyだけで決めない**という考え方を導入した。

---

# 3. retained energyはtask性能ではない

旧02では、rankを上げると `fc1` / `fc2` のretained energyが上昇することを確認した。

しかし、

```text
retained energy
→ weight近似の指標

Validation loss / accuracy
→ task性能の指標
```

なので同一視しない。

特異値エネルギーが高いrankが、必ずしもValidation accuracyで最良になるわけではない。

---

# 4. Pareto / kneeの意味

Pareto frontierでは、ある候補Aが別候補Bに対して、

```text
Parametersが少ない
かつ
Validation lossも小さい
```

なら、Bは選択候補から外せる。

そのうえでkneeを使い、

```text
圧縮率をさらに上げると
性能悪化が急になる境界付近
```

を候補化した。

旧02では、

```text
knee = fc1 64 / fc2 16
```

となった。

ここで重要なのは、**kneeはデータ・Baseline weight・候補rank・評価軸に依存する**こと。

---

# 5. full-rankでも圧縮とは限らない

SVDを使って2因子へ分解しても、rankが大きすぎるとparameter数は元層より増える。

Linearの元weightを、

$$
W \in \mathbb{R}^{D_{out}\times D_{in}}
$$

とすると、低rank2層のweight数は、

$$
r(D_{in}+D_{out})
$$

となる。

したがって、

```text
数学的に許されるrank
≠
parameter削減になるrank
```

である。

旧02の大rank候補では、agreementが1へ近づいてもparameter reductionが負になる条件が確認できた。

---

# 6. 旧runの数値を残す意味

旧02で得た代表値は、

| Candidate | fc1 | fc2 | Parameters | Val loss | Val acc |
|---|---:|---:|---:|---:|---:|
| Aggressive | 32 | 32 | 69,386 | 0.376715 | 86.46% |
| Balanced | 64 | 16 | 98,570 | 0.304215 | 89.20% |
| Conservative | 64 | 32 | 110,858 | 0.300669 | 89.30% |

だった。

これらは現在のformal resultではないが、

- Pareto frontierの作り方
- kneeのrun依存性
- retained energyとtask性能の違い
- full-rankでも圧縮にならない場合

を学んだ過程として残す。

全30条件のhistorical raw tableは [[20_FashionMNIST/05_全RankSweep結果]] を参照する。

---

# 7. corrected版で変わったこと

後のレビューでは、Fine-tuning candidate比較について、

- candidateごとのseedを同じ条件へ戻す
- DataLoader Generatorも同じ状態へ戻す
- train metricsでshuffle付きtraining loaderを再走査しない
- sweep DataFrameへ全candidate modelを保持しない
- baseline / compressedのParameter共有を避ける

といった実験契約を明確化した。

その結果、MLPのcanonical finalは、

```text
fc1 = 32
fc2 = 16
Parameters = 57,098
Test acc = 0.8853
```

となった。

したがって旧02の `64/16` kneeは、**rank選択手法を学ぶためのhistorical result** として扱う。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/05_全RankSweep結果]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
