# TT-Conv圧縮

Conv2dのweightをTT-matrixとして表し、dense kernel再構成経路、TT coresによる直接計算、`TTConv2d`のModule設計、学習済みモデルへの層置換、rank sweep、fine-tuningと評価方法を整理する。

## 技術ノート

- [[00_基礎理論/03_モデル圧縮理論/31_TTConv圧縮/79_Conv2dのTT-matrix表現と直接forward]]

## 対応する実装確認

- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/README]]
