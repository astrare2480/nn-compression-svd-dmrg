# Fashion-MNIST MLP SVD実験

Fashion-MNIST分類用MLPを対象に、過学習確認、Early Stopping、SVD rank sweep、Pareto / kneeによるrank選択、Fine-tuning、最終Test評価までをまとめる。

## この版の正本

2026-08-10に添付された次のNotebookを数値の正本とする。

| Notebook | 役割 |
|---|---|
| `00_mlp_baseline_fixed_50epochs.ipynb` | 50 epoch固定学習で過学習を確認 |
| `01_mlp_baseline_early_stopping(3).ipynb` | Early Stopping導入 |
| `02_mlp_svd_rank_selection.ipynb` | 30 rank pairのSVD sweep、Pareto、knee、3候補選択 |
| `03_mlp_svd_finetuning(3).ipynb` | 3候補Fine-tuning、最終モデル選択、Test評価 |

## ノート

- [[20_FashionMNIST/01_Fashion-MNIST実験]]
- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]

## 1バッチshape確認セルの扱い

次のセルは**実験結果を得るためには不要**なので、ノートの実験手順から外した。

```python
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
```

最新の `01` / `02` / `03` ではこのセルは削除済み。
一方、添付された最新 `00` にはまだ残っているため、`00` の数値はそのNotebookの実行結果をそのまま採用している。

> [!note]
> `train_loader` が `shuffle=True` の場合、学習前に `next(iter(train_loader))` を呼ぶと乱数状態を進め、後続のシャッフル順を変える可能性がある。そのため旧Notebookの数値と今回の再実行結果を混ぜない。

## 共通理論

一般理論は `00_基礎理論` 側へ分離する。

- [[00_基礎理論/03_SVDによる低ランク近似]]
- [[00_基礎理論/05_圧縮率とRank]]
- [[00_基礎理論/06_誤差評価]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/12_PyTorch学習と評価の基礎]]

## 現在の最終結果

```text
Final model: Aggressive
fc1_rank = 16
fc2_rank = 16

Parameters: 535,818 -> 36,362  (93.2137% reduction)
MACs:       535,040 -> 35,584  (93.3493% reduction)

Test accuracy: 0.8871 -> 0.8714  (-1.57 percentage point)
Test loss:     0.333783 -> 0.366167
```


## 最新00について

`00_mlp_baseline_fixed_50epochs` は、学習前に1バッチを取り出すshape確認セルを削除して再実行した最新版へ更新した。

```text
Epoch 50
Train loss = 0.0650
Train acc  = 0.9751
Test loss  = 0.6842
Test acc   = 0.8844
```

全50 epochは [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]] を参照。
