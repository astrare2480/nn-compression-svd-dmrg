# CIFAR-10 CNN / SVD・Tucker

Fashion-MNISTで確認したLinear / Conv SVDを、RGB自然画像と複数Conv層へ拡張し、その同じCIFAR-10 CNNをTucker / HOSVD / HOOIの実験基盤にも使う章。

```text
Fashion-MNIST
single-layer rank selection
+ fixed-rank Conv / Linear combination
        ↓
CIFAR-10 / SVD
conv1 / conv2 / conv3 の model-wide rank allocation
        ↓
CIFAR-10 / Tucker
conv2のmode 0 / 1をTucker-2圧縮
        ↓
HOOI
同rankでfactorを反復精密化
```

## 読む順番

### CIFAR-10 / SVD

1. [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]]
2. [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
3. [[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]
4. [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
5. [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
6. [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]

### Tucker / HOOI

1. [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
2. [[00_基礎理論/01_数学基礎/02_テンソル代数/21_テンソルとmode演算]]
3. [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
4. [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
5. [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
6. [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]
7. [[06_Tucker基礎実装検証/README]]

## canonicalデータ分割

SVD corrected Notebookでは公式train 50,000枚を、

```text
Train                       40,000
Early-Stopping Validation    5,000
Rank-Selection Validation    5,000
Test（公式test）            10,000
```

へ分ける。

Dataset / Subset / Generatorの基礎説明で使う `45,000 / 5,000` は単純例であり、**canonical SVD実験条件は40,000 / 5,000 / 5,000**。

データ取得が極端に遅い場合の `download=False`、展開先、MD5確認などの実務メモも [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]] に残している。

## SVDのcanonical result

正式に引用するSVD結果はcorrected版。

```text
Notebook:
notebooks/10_svd/40_cifar10_cnn/02_svd_global_compression_using_src_corrected.ipynb

Results:
results/10_svd/40_cifar10_cnn/02_svd_global_compression_using_src_corrected/
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

この `before → after` はRank-Selection Validation accuracyのFine-tuning前後。single seedなので小差を性能改善とは断定せず、圧縮後も精度をほぼ維持したと解釈する。

探索は全rank空間のglobal optimumではなく、各層のPareto / knee近傍へ候補を制約したmodel-wide rank allocationである。

## SVDのMACsとlatency

2026-09-12のcorrected保存checkpoint再計測では、

```text
batch size = 256
same input_batch
warmup = 20
repeats = 2000
```

へ揃えた。

```text
MACs:    -45.69%
Latency: 約0.575 → 0.551 ms/batch（3 trialの平均時間の中央値）
```

今回は約4.09%短縮したが、MACsの45.69%削減に比例した短縮ではない。GPUはRTX 5070 Ti、PyTorchは2.11.0+cu128。元の学習とは別の計測runで、再学習は行っていない。生データ・checkpoint/入力hashと、出典不明の旧値の扱いは [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]] の「MACsとlatency」を参照。

## Tucker / HOSVD / HOOIへの拡張

SVDではConv weight

```text
(C_out, C_in, K_h, K_w)
```

を行列へreshapeして2次元rankで近似した。

Tuckerでは同じ4階weightのmode構造を保持し、channel mode 0 / 1へ別々のrankを持たせる。

主対象は学習済み

```text
conv2.weight = (64, 32, 3, 3)
```

で、Tucker-2では

```text
C_in
→ 1x1 / U_in^T
→ R_in
→ 3x3 / core
→ R_out
→ 1x1 / U_out
→ C_out
```

へ置換した。

Tucker/HOOIのcanonical sourceは、

```text
notebooks/20_tucker/
results/20_tucker/
```

で、結果の整理は [[30_CIFAR10_CNN/02_Tucker2_Convとrank_sweepの確認結果]]、[[30_CIFAR10_CNN/03_HOOIとTensorLy照合]]、[[30_CIFAR10_CNN/04_HOSVD_HOOI_FineTuning比較]] を参照する。

### balanced rank `(32,16)`

baseline：

```text
validation accuracy = 0.7344
test accuracy       = 0.7327
parameters          = 128,842
```

Tucker-2後：

```text
parameters             = 117,578
Conv2 MAC reduction    = 61.11%
model MAC reduction    = 27.84%
```

分解直後のweight relative errorは、

```text
HOSVD = 0.449042
HOOI  = 0.442504
```

でHOOIの方が小さかった。一方、validation accuracyは

```text
HOSVD = 0.6280
HOOI  = 0.6202
```

であり、**weight Frobenius誤差を小さくすることがtask accuracy改善を保証しない**ことを確認した。

TensorLy `partial_tucker` は、自作HOOIと最終weight error・圧縮直後accuracyで整合した。

## HOSVD / HOOI同条件Fine-tuning

seed 0の同条件比較では、

```text
post validation
HOSVD = 0.7528
HOOI  = 0.7528

post test
HOSVD = 0.7508
HOOI  = 0.7555
```

となった。

差は0.0047で1 seedのみなので、HOOIの一般的な最終accuracy優位性とは結論しない。

さらにFine-tuning後には元weightへのrelative errorが増えながらaccuracyが改善し、

```text
元weightへの近さ
≠
taskにとって最適なlow-rank weight
```

であることも実測した。

## canonical / historical

SVDでは `*_corrected.ipynb` と対応するcorrected resultsを正式結果として優先する。

Tucker / HOOIでは `notebooks/20_tucker/` と `results/20_tucker/` の現行Notebook / CSVを優先する。

original / using_src / before_srcはhistorical recordとして残し、別runのabsolute値を同一実験として混ぜない。

## 修正履歴・関連

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[06_Tucker基礎実装検証/README]]
- [[README_実装編]]
