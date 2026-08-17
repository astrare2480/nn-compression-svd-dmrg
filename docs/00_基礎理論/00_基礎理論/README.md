# 基礎理論・共通実装

データセットに依存しないSVD、Linear / Conv2dの低ランク近似、評価設計、PyTorch実装、理論計算量、ベンチマークを置く。
Fashion-MNIST実験で整理したPareto / knee、厳密なValidation分離、fine-tuning再現性、CNN / Conv-SVDも共通知識として反映した。

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

## CNN / Conv-SVDの読み順

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

## 現在の共通実装

主要な汎用処理は `src/nn_compression` へ分離している。

```text
compression/
├─ svd.py         # モデル非依存SVD / retained energy
├─ linear_svd.py  # 任意のnn.Linear
├─ conv_svd.py    # 任意のgroups=1 nn.Conv2d
└─ mlp_svd.py     # 現在のMLP構造専用

metrics/
├─ macs.py        # Linear / Conv2dの層単位MACs
├─ mlp_macs.py    # MLP全体
├─ cnn_macs.py    # FashionMNISTCNN全体
└─ model_comparison.py

models/
├─ mlp.py
└─ cnn.py
```

理論ノートでは、汎用処理とFashion-MNIST固有処理を区別する。

## 実験ノート

具体的なrank・accuracy・loss・選択結果は実験側へ分離する。

- [[10_MNIST_MLP_SVD/README]]
- [[20_FASHION_MNIST_MLP_SVD/README]]

Fashion-MNIST CNN実験結果は、CNN実験ノート側へ追加する。
