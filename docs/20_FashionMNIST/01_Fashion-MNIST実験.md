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

## サマリー

Fashion-MNIST分類用のMLPを学習し、学習済み `fc1` と `fc2` の重みをSVDで低ランク2層へ置き換え、圧縮率と分類性能の関係を調べた。

```mermaid
flowchart LR
    D["Fashion-MNIST"] --> B["Baseline MLP"]
    B --> ES["Early Stopping"]
    ES --> S["fc1 / fc2をSVD"]
    S --> R["rank pair sweep"]
    R --> P["Pareto frontier"]
    P --> K["knee ± 1"]
    K --> F["3候補をFine-tuning"]
    F --> V["Validationで最終モデル選択"]
    V --> T["Testで最終評価"]
```

今回SVDをかけた対象は画像そのものではなく、学習済みMLPの重み行列。

---

## 1. 目的

> 学習済みMLPの重み行列にどれだけ冗長性があり、低rank表現へ置き換えても分類性能をどこまで維持できるかを調べる。

さらにSVD直後に落ちた性能を、rankを固定したFine-tuningでどこまで回復できるかを見る。

---

## 2. データとモデル

Fashion-MNISTは28×28グレースケール画像、10クラス、train 60,000 / test 10,000。

モデル：

```text
784 -> 512 -> 256 -> 10
```

```text
fc1 = Linear(784, 512)
fc2 = Linear(512, 256)
fc3 = Linear(256, 10)
```

SVD圧縮対象は `fc1` と `fc2`。`fc3` は非圧縮。
Baseline parameter数は **535,818**。

---

## 3. 00: 50 epoch固定学習

条件：

| 項目 | 値 |
|---|---:|
| Train | 60,000 |
| Test | 10,000 |
| batch size | 64 |
| optimizer | Adam |
| learning rate | 0.001 |
| epoch | 50固定 |

epoch 50：

| 指標 | Train | Test |
|---|---:|---:|
| loss | 0.0650 | 0.6842 |
| accuracy | 97.51% | 88.44% |

![[20_FashionMNIST/assets/baseline_fixed50_learning_curves.png]]

Train lossは全体として低下する一方、Test lossは後半で大きく上昇しており、過学習を確認できる。
このNotebookは過学習確認用であり、正式なモデル選択には使わない。

shape確認用の `next(iter(train_loader))` セルは削除し、最新版00を先頭から再実行した結果へ更新した。

---

## 4. 01: Early Stopping

分割：

```text
Train      55,000
Validation  5,000
Test       10,000
```

設定：

```text
MAX_EPOCHS = 50
PATIENCE   = 5
MIN_DELTA  = 1e-4
```

最新run：

| 項目 | 値 |
|---|---:|
| Early stopping | epoch 15 |
| Best epoch | 10 |
| Best validation loss | 0.2767 |
| Test loss | 0.3212 |
| Test accuracy | 89.07% |

![[20_FashionMNIST/assets/baseline_early_stopping_learning_curves.png]]

Best epochの `state_dict` を `deepcopy` で保存し、学習終了後に復元してTest評価する。

---

## 5. 02 / 03: Validationを2つへ分離

SVD rank選択以降はtrain 60,000を、

```text
Train                       50,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test                        10,000
```

へ分ける。

役割：

```text
Early-Stopping Validation
-> 学習を止めるepochを決める

Rank-Selection Validation
-> rank / Pareto / knee / Fine-tuning後の最終モデルを選ぶ

Test
-> 最終モデル決定後にだけ評価する
```

---

## 6. shape確認セルを実験手順から外した理由

`next(iter(train_loader))` でshapeを見る処理は、データ形式の初期確認には使えるが、モデル学習・評価には不要。

さらに `train_loader` が `shuffle=True` なので、学習前にiteratorを作ると乱数状態を進め、後続epochのバッチ順を変える可能性がある。
今回の最新 `01` / `02` / `03` ではこのセルを削除したため、旧runの数値ではなく**今回添付されたNotebook出力を正本**とする。

> [!warning]
> 最新 `00_mlp_baseline_fixed_50epochs.ipynb` にはshape確認セルがまだ残る。ここで記録した00の値はその添付Notebookの出力に合わせている。00からもセルを削除して再実行した場合は、00の履歴を再更新する必要がある。

---

## 7. 評価関数

最新Notebookの `evaluate()` は、

```python
model.eval()
with torch.no_grad():
    ...
```

を内部で実行する。
そのため各評価セルに `eval()` / `no_grad()` を直接書かなくても、Validation / Testのlossとaccuracyは評価モードで計算される。

---

## 関連

- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
