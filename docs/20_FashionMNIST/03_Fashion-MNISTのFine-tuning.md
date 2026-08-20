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
  - Historical
---

# Fashion-MNISTのFine-tuning

> [!warning]
> このノートは旧 `03_mlp_svd_finetuning` 系runで得たFine-tuning結果と、そこから学んだことを残すhistorical note。
> 現在の正式結果は [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]] のcorrected版を参照する。

## サマリー

旧03ではBaseline学習からrank sweepまで独立に実行し、SVD直後のknee近傍をFine-tuningした。

旧runの候補：

```text
Aggressive   = 16 / 16
Balanced     = 32 / 16
Conservative = 64 / 16
```

Fine-tuningにより低rank化直後の性能を大きく回復できることを確認した一方、後のレビューで**candidate間の乱数条件を同一にして比較すべき**ことが分かった。

そのため旧runの最終 `16/16` はcanonical resultではない。

---

# 1. Fine-tuningは何をしているか

SVDで作った低rank modelは、元weightをFrobeniusノルムの意味でよく近似するよう初期化される。

しかし、

```text
weight approximation
≠
task loss optimum
```

である。

Fine-tuningでは、

```text
学習済みweight
↓ truncated SVD
低rank因子を初期値にする
↓ task lossで追加学習
低rank構造を保ったまま分類性能へ再適応
```

を行う。

この違いは、Tucker / TTへ進んだあとも重要になる。

---

# 2. 旧runで見えた回復

旧03では、Fine-tuning前後で次のような変化があった。

| Candidate | rank | Val acc before | Val acc after | Val loss before | Val loss after |
|---|---|---:|---:|---:|---:|
| Aggressive | 16 / 16 | 0.7176 | 0.8832 | 0.858264 | 0.322567 |
| Balanced | 32 / 16 | 0.8718 | 0.8858 | 0.364576 | 0.328358 |
| Conservative | 64 / 16 | 0.8886 | 0.8862 | 0.311055 | 0.335646 |

特に `16/16` はSVD直後には大きく崩れたが、Fine-tuningで大幅に回復した。

ここから、

> SVD直後の大きなaccuracy低下だけを見て「このrankでは表現能力が足りない」と即断しない

ことを学んだ。

一方でConservativeでは別Validation上の性能が悪化しており、**Fine-tuningは必ず改善するわけではない**ことも分かった。

---

# 3. 旧runの問題 — candidateごとに別seedを使っていた

旧03では、Fine-tuning candidateごとにrankからseedを作る実装を使っていた。

```python
set_seed(SEED + 1000 * fc1_rank + fc2_rank)
```

これは各candidateを再実行可能にはするが、**candidate間の比較条件は同じではない**。

```text
Candidate A
→ rank A + 乱数条件 A

Candidate B
→ rank B + 乱数条件 B
```

となるため、Validation差にrank以外の学習乱数差が混ざる。

rank差を比較したいなら、Fine-tuning直前に、

```python
set_seed(SEED)
loader_generator.manual_seed(SEED)
```

として候補間の乱数条件を揃える方がよい。

corrected版ではこの方針へ変更した。

---

# 4. DataLoader Generatorも比較条件に含める

seedを同じ整数にするだけでは十分ではない。

`shuffle=True` のDataLoaderは内部のGenerator状態に依存するため、先に何回iteratorを作ったか、評価で同じloaderを再走査したかでも次のmini-batch順が変わり得る。

そのためcorrected実装では、

```text
candidate開始前
→ global seedを戻す
→ DataLoader Generatorも戻す

train metrics
→ shuffle=Falseのtrain_eval_loaderで計測
```

とした。

評価処理そのものが後続学習のshuffle順を変えないようにする。

詳細：

- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

---

# 5. baseline / compressed modelを共有しない

Fine-tuning candidateはbaselineから独立したmodelでなければならない。

もし未圧縮層をbaselineとcompressedで同じModule / Parameterとして共有すると、compressed modelをFine-tuningしただけでbaselineも変化する。

corrected実装では `deepcopy` と新規factor layer生成により、

```text
baseline parameters
≠
compressed parameters
```

を保証する。

この契約は `src/nn_compression` 側にもコメントとして残している。

---

# 6. sweepでは全modelを保持しない

旧実験ではrank sweepのrecordへcandidate model本体を保持する構成があった。

学習目的では分かりやすいが、候補数が増えると、

- メモリを余計に使う
- DataFrameへPyTorch objectが混ざる
- CSVへ保存しにくい

という問題がある。

現在は、

```text
sweep
→ rankとmetricsを保存

Fine-tuning対象
→ 必要なrankだけmodelを再構築
```

を基本とする。

`include_model=False` はこのための契約。

---

# 7. corrected版の結果

公平性を修正したcanonical Notebook：

```text
notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb
```

Fine-tuning候補：

| Candidate | fc1 | fc2 | Val acc before | Val acc after | Val loss after | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Aggressive | 16 | 16 | 0.7822 | 0.8850 | 0.318649 | 36,362 |
| Balanced | **32** | **16** | 0.8694 | **0.8924** | **0.307956** | **57,098** |
| Conservative | 32 | 32 | 0.8694 | 0.8856 | 0.318636 | 69,386 |

最終選択：

```text
fc1 = 32
fc2 = 16
```

最終Test：

```text
Baseline
loss = 0.329024
acc  = 0.8819

Compressed + FT
loss = 0.333776
acc  = 0.8853
```

Parameter reduction：

```text
535,818 → 57,098
-89.34%
```

`+0.34pt` はsingle seedの小差なので、accuracy改善ではなく**大幅圧縮後も精度をほぼ維持した**と表現する。

---

# 8. 旧runとcorrectedをどう読むか

旧03から得られた学びそのものは残る。

- SVD直後の性能低下はFine-tuningで大きく回復し得る
- 低rank因子はランダム初期化ではなく、学習済みweight由来の初期値
- Fine-tuningが常にValidation性能を改善するわけではない

ただし最終rank・accuracyを引用するときは、

```text
旧 16/16
ではなく
corrected 32/16
```

を使う。

旧runの詳細なepoch / rank表は、

- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]

にhistorical raw dataとして残す。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
