# 基礎理論・共通実装

データセットに依存しないSVD、Linear / Conv2dの低ランク近似、評価設計、PyTorch実装、理論計算量、ベンチマークを置く。
Fashion-MNIST実験で整理したPareto / knee、厳密なValidation分離、Fine-tuningの再現性、CNN / Conv SVDに加え、CIFAR-10で整理したデータ前処理、DataLoader、乱数管理、GAPと複数Convのrank allocationへつながる知識も共通知識として反映する。

## ノート

- [[00_基礎理論/01_SVDとは]]
- [[00_基礎理論/02_nn.Linearとは]]
- [[00_基礎理論/03_SVDによる低ランク近似]]
- [[00_基礎理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/05_圧縮率とRank]]
- [[00_基礎理論/06_誤差評価]]
- [[00_基礎理論/07_PyTorch実装]]
- [[00_基礎理論/08_Linear層のSVD実装]]
- [[00_基礎理論/09_Linear層の2層置換_実装]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]
- [[00_基礎理論/12_PyTorch学習と評価の基礎]]
- [[00_基礎理論/13_PandasとPython実装メモ]]
- [[00_基礎理論/14_低ランク学習からテンソルネットワークへの発展]]
- [[00_基礎理論/15_CNNとConv2dの基礎]]
- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/18_CIFAR10の前処理とDataLoader]]
- [[00_基礎理論/19_再現性と乱数管理]]
- [[00_基礎理論/20_Global Average PoolingとCIFAR10モデル設計]]

## Linear-SVDの読み順

```text
01_SVDとは
↓
02_nn.Linearとは
↓
03_SVDによる低ランク近似
↓
04_Linear層を2層へ置き換える
↓
05_圧縮率とRank
↓
06_誤差評価
↓
08_Linear層のSVD実装
↓
09_Linear層の2層置換_実装
```

## CNN / Conv SVDの読み順

```text
15_CNNとConv2dの基礎
↓
16_Conv2d重みの行列化とSVD
↓
17_Conv2dの低ランク2層置換
↓
11_理論計算量とベンチマーク
↓
10_SVD圧縮モデルの評価設計
```

Linear-SVDとの対応は、

```text
04_Linear層を2層へ置き換える
↕
17_Conv2dの低ランク2層置換
```

として読む。

## CIFAR-10へ進むときの読み順

Fashion-MNISTでLinear / Conv SVDを確認した後は、次の順で読む。

```text
18_CIFAR10の前処理とDataLoader
↓
19_再現性と乱数管理
↓
20_Global Average PoolingとCIFAR10モデル設計
↓
15_CNNとConv2dの基礎（shapeを再確認）
↓
16 / 17 Conv SVD
↓
10 評価設計
↓
30_CIFAR10_CNN/01_CIFAR10_SVD実験
```

ここで重要なのは、CIFAR-10ではSVDの式自体を変えるのではなく、

```text
自然画像向けの前処理
train / validation / test分離
乱数とDataLoaderの再現性
GAPを使ったモデル設計
複数Convへのrank allocation
```

へ実験設計を発展させたことである。

## 現在の共通実装

主要な汎用処理は `src/nn_compression` へ分離している。

```text
compression/
├─ svd.py         # モデル非依存SVD / retained energy
├─ linear_svd.py  # 任意のnn.Linear
├─ conv_svd.py    # 任意のgroups=1 nn.Conv2d
└─ mlp_svd.py     # 現在のMLP構造専用

metrics/
├─ macs.py
├─ mlp_macs.py
├─ cnn_macs.py
└─ model_comparison.py

models/
├─ mlp.py
├─ cnn.py
└─ cifar10.py
```

現在のsrcでは、corrected実験で重要になった、

- baseline / compressedのParameter非共有
- rank上限
- device / dtype / requires_grad保持
- training DataLoader Generatorを評価で進めない
- benchmarkで同一input batchを使う
- rank sweepでcandidate modelを保持しすぎない

といった不変条件を実装・コメントとして残している。

詳細は [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] を参照。

## 実験ノート

具体的なrank・accuracy・loss・選択結果は実験側へ分離する。

- [[10_MNIST_MLP_SVD/README]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]

正式結果を参照するときは、historical original / using_srcではなくcorrected Notebook / corrected resultsを優先する。
