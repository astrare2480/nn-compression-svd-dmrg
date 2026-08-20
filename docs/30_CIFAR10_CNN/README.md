# CIFAR-10 CNN / SVD

Fashion-MNISTで確認したLinear / Conv SVDを、RGB自然画像と複数Conv層へ拡張する章。

## 読む順番

1. [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]]
2. [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
3. [[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]
4. [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
5. [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
6. [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]

## この章で進んだ点

```text
Fashion-MNIST
single-layer rank selection
+ fixed-rank Conv/Linear combination
        ↓
CIFAR-10
conv1 / conv2 / conv3 の model-wide rank allocation
```

CIFAR-10では、学習時だけRandomCrop / HorizontalFlipを使い、評価側では入力条件を固定する。GAPで巨大な全結合層を避け、複数Conv層の圧縮を主題にした。

## canonicalデータ分割

corrected Notebookでは公式train 50,000枚を、

```text
Train                       40,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test（公式test）            10,000
```

へ分ける。

PerplexityでDataset / Subset / Generatorを学んだ際の `45,000 / 5,000` は基礎説明用の単純例であり、**canonical実験条件は40,000 / 5,000 / 5,000**。

データ取得が極端に遅い場合の `download=False`、展開先、MD5確認などの実務メモも [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]] に残している。

## 正式結果

正式に引用する結果はcorrected版。

```text
Notebook:
notebooks/30_cifar10/02_svd_global_compression_using_src_corrected.ipynb

Results:
results/30_cifar10/02_svd_global_compression_using_src_corrected/
```

最終rank：

```text
conv1 = 9
conv2 = 32
conv3 = 48
```

```text
Parameters: 128,842 → 81,405       (-36.82%)
MACs:       10,357,248 → 5,625,344 (-45.69%)
Test acc:   73.27% → 73.43%
```

Fine-tuning候補：

```text
Aggressive   6 / 32 / 32 : 0.3994 → 0.7336
Balanced     9 / 32 / 32 : 0.4354 → 0.7360
Conservative 9 / 32 / 48 : 0.4792 → 0.7444
```

この `before → after` は **Rank-Selection Validation accuracy** のFine-tuning前後を表す。Early-Stopping Validationやtestの値ではない。

single seedなので、小さいaccuracy差は改善と断定せず、**圧縮後も精度をほぼ維持した**と解釈する。

探索は全rank空間のglobal optimumではなく、各層のPareto / knee近傍に候補を制約したmodel-wide rank allocationである。

## MACsとlatency

corrected benchmarkでは、

```text
batch size = 256
same input_batch
warmup = 20
repeats = 2000
```

へ揃えた。

```text
MACs:    -45.69%
Latency: 約0.531 → 0.537 ms/batch
```

理論演算量削減はwall-clock speedupを保証しなかった。

## 修正履歴を読む

旧実験の問題とcorrected版での修正理由は、

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

へまとめている。

original / using_src / before_srcはhistorical recordとして残し、正式結果はcorrectedを優先する。
