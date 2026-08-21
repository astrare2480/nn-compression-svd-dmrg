---
title: Fashion-MNIST実験
aliases:
  - Fashion-MNIST SVD実験
  - Fashion-MNIST MLP圧縮
  - Fashion-MNIST baseline
tags:
  - Fashion-MNIST
  - PyTorch
  - MLP
  - SVD
  - NN圧縮
  - EarlyStopping
---

# Fashion-MNIST実験

> [!important]
> このノートは、Fashion-MNIST MLP実験を組み立てていった**学習履歴・設計経緯**を残す。
> 最終rank・Test accuracyなどの正式値は [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]] のcorrected結果を参照する。

## サマリー

Fashion-MNIST分類用のMLPを学習し、学習済み `fc1` / `fc2` のweightをSVDで低rank化した。

```mermaid
flowchart LR
    D["Fashion-MNIST"] --> B["Baseline MLP"]
    B --> ES["Early Stopping"]
    ES --> S["fc1 / fc2をSVD"]
    S --> R["rank pair sweep"]
    R --> P["Pareto / knee"]
    P --> F["Fine-tuning"]
    F --> V["Validationで選択"]
    V --> T["Testで最終確認"]
```

SVD対象は画像ではなく、学習済みMLPのweight matrixである。

---

# 1. データとモデル

Fashion-MNISTは28×28のグレースケール衣類画像を10クラスへ分類するデータセット。

今回のMLPは、

```text
784 → 512 → 256 → 10
```

```text
fc1 = Linear(784, 512)
fc2 = Linear(512, 256)
fc3 = Linear(256, 10)
```

で、SVD対象は `fc1` / `fc2`。`fc3` はそのまま残す。

Baseline parameter数：

```text
535,818
```

---

# 2. 固定50 epoch学習で過学習を確認した

最初のNotebookでは、Early Stoppingを入れず50 epoch学習し、Train lossが下がり続ける一方でTest lossが後半に悪化することを確認した。

```text
最終epoch付近
Train acc ≈ 97.5%
Test acc  ≈ 88.4%
Test loss は後半で上昇
```

このrunの目的は最終modelを作ることではなく、**固定epochで学習し続けるとValidation / Test性能が悪化し得る**ことを確認することだった。

---

# 3. Early Stoppingを導入した

次に、trainの一部をValidationへ分け、Validation lossが改善しなくなった時点で学習を止めるようにした。

重要なのは、最後のepochのweightではなく、

```text
best validation lossを出したepoch
```

の `state_dict` を保存・復元すること。

これにより、

```text
train lossを最小化し続ける
```

のではなく、Validation性能を基準にmodelを選択する流れへ移った。

---

# 4. Validationを用途別に分けた

SVD rank選択まで行うと、1つのValidationを、

```text
Early Stopping
+
rank選択
+
Fine-tuning後の最終候補選択
```

に何度も使うことになる。

そのため実験では、

```text
Train                       50,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test                        10,000
```

へ役割を分離した。

```text
Early-Stopping Validation
→ 学習を止めるepochを決める

Rank-Selection Validation
→ rank / candidateを選ぶ

Test
→ 最終モデル決定後に確認する
```

この考え方はMNIST correctedで明確化した「Testをrank選択に使わない」という原則と同じ。

---

# 5. DataLoaderを読むときの注意

学習用DataLoaderは通常、

```text
shuffle=True
```

を使う。

そのため、学習前に、

```python
next(iter(train_loader))
```

などで確認用batchを取得すると、DataLoader Generatorの状態を進める可能性がある。

shape確認自体が悪いわけではないが、再現性を重視する実験では、

```text
確認用処理
≠
学習用Generatorを進める処理
```

として分離する。

corrected実装ではさらに、train metricsを計算するときも `shuffle=True` のtraining loaderを再走査せず、`train_eval_loader` を分けている。

詳細：

- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

---

# 6. 旧02 / 03 runとcorrectedの関係

historicalなMLP 02 / 03では、rank sweepとFine-tuningまで実施した。

ただし後のレビューで、candidateごとに異なるseedを使うなど、比較条件を揃える余地が見つかった。

そのため、旧runのrankやaccuracyは履歴として残し、現在の正式結果はcorrected Notebookを使う。

```text
notebooks/10_svd/20_fashion_mnist_mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb
```

canonical result：

```text
fc1 rank = 32
fc2 rank = 16
Parameters = 57,098
Test acc   = 0.8853
```

Baseline Test accuracyは `0.8819`。

single seedなので、`+0.34pt` を性能改善とは断定せず、**約89%のparameter削減後も精度を維持した**と解釈する。

---

# 7. この実験設計から学んだこと

1. SVD対象は入力画像ではなく学習済みweight。
2. 固定epochよりValidationを使ったmodel選択が重要。
3. Early Stopping用Validationとrank選択用Validationを分けると役割が明確になる。
4. Testは最終model選択後まで温存する。
5. DataLoaderの乱数状態も実験条件の一部。
6. Fine-tuning candidate間では比較対象以外の乱数条件を揃える。
7. historical runを残しつつ、formal resultはcorrectedへ一本化する。

---

# 関連

- [[20_FashionMNIST/README]]
- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
