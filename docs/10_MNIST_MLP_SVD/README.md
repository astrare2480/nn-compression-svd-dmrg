# MNIST MLP SVD実験

MNIST用MLPで、Linear SVDの基本、rank sweep、精度・圧縮率・MACs・latencyの評価を確認する章。

## 読む順番

1. [[10_MNIST_MLP_SVD/01_MNIST実験]]
2. [[10_MNIST_MLP_SVD/02_MNIST評価で使用した指標と実装]]
3. [[10_MNIST_MLP_SVD/03_MNISTのMACs訂正と実測時間]]
4. [[10_MNIST_MLP_SVD/04_MNISTでの実験結果]]

## canonical結果

正式なrank sweep結果は、

```text
notebooks/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun.ipynb
results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/
```

を優先する。2026-09-12の完全保存runを現行出典とし、複製元correctedの記録も残す。

```text
Selected rank: fc1=128, fc2=128
Parameters: 約50%削減
Test acc: 98.17% → 98.08%
```

corrected版ではrankをvalidationで選び、testは最終選択した1構成だけ評価した。

旧rank sweepでtestをrank選択に使っていた点はhistoricalな制約として残すが、正式なrank選択根拠には使わない。

詳細：

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

## historical docs

01〜03には初期実装・旧Notebookの数値や訂正履歴も含まれる。

それらは学習履歴として有用だが、最終rank・accuracyを引用するときは04のcorrected rerun結果を使う。

## 画像

`assets/` に元ZIPへ含まれていた画像を配置している。historical Notebook由来の図は、corrected結果と数値が一致しない場合があるため、どのrunの図かを確認して読む。
