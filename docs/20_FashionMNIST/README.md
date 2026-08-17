# Fashion-MNIST SVD圧縮実験

Fashion-MNISTを使ったSVDによるニューラルネットワーク圧縮実験をまとめる。

このディレクトリには、先に行った **MLP編** と、その後に行った **CNN編** の結果を置く。

```text
Fashion-MNIST
├─ MLP
│  ├─ Baseline / Early Stopping
│  ├─ Linear SVD rank sweep
│  ├─ Pareto / knee
│  └─ Fine-tuning / Test
│
└─ CNN
   ├─ CNN Baseline
   ├─ fc1 Linear SVD
   ├─ conv2 Conv SVD
   └─ conv2 + fc1 同時SVD
```

## MLP編

- [[20_FashionMNIST/01_Fashion-MNIST実験]]
- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]

MLP最終結果：

```text
fc1 rank = 16
fc2 rank = 16

Parameters: 535,818 → 36,362  (-93.2137%)
MACs:       535,040 → 35,584   (-93.3493%)
Test acc:   88.71% → 87.14%    (-1.57pt)
Test loss:  0.333783 → 0.366167
```

## CNN編

- [[20_FashionMNIST/07_CNN実験]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]] — 最終比較、考察、実験上の限界、今後の追加検証
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]

CNN最終結果：

```text
conv2 rank = 28
fc1 rank   = 24

Parameters: 421,642 → 89,994      (-78.66%)
MACs:       4,241,152 → 2,237,184 (-47.25%)
Test acc:   91.75% → 91.80%       (+0.05pt)
Test loss:  0.234131 → 0.240347
```

CNNの最終Test accuracy差は小さいため、本ノートでは「圧縮により精度が改善した」ではなく、**大幅な圧縮後も精度を維持した**と解釈する。

## CNNの比較で重要な点

`fc1` と `conv2` は圧縮効果の性質が異なる。

```text
fc1 SVD
→ Parametersの削減に効く
→ rank24で全体Parameters -76.62%
→ 全CNN MACsでは -7.62%

conv2 SVD
→ Parameters削減は小さい
→ 全CNN MACsでは -39.63%

conv2 + fc1
→ Parameters -78.66%
→ 全CNN MACs -47.25%
```

また05では、MACsを47.25%削減しても、保存されたbenchmarkでは推論時間は `0.384253 ms → 0.416497 ms` で短縮しなかった。理論演算量と実測latencyは分けて記録する。

## MLP vs CNN 比較・総括

- [[20_FashionMNIST/13_MLP_vs_CNNの比較と総括]]

MLP編とCNN編をまとめ、Linear SVDとConv SVDの違い、ParametersとMACsの違い、Fine-tuningの役割を整理する。

また、Fashion-MNISTで確立した実験手順を次のCIFAR-10へ引き継ぐためのロードマップを記載する。

## 共通理論

一般理論は `00_基礎理論` 側へ分離する。

### Linear / SVD

- [[00_基礎理論/03_SVDによる低ランク近似]]
- [[00_基礎理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/05_圧縮率とRank]]
- [[00_基礎理論/06_誤差評価]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]

### CNN / Conv SVD

- [[00_基礎理論/15_CNNとConv2dの基礎]]
- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]

### PyTorch / 学習

- [[00_基礎理論/12_PyTorch学習と評価の基礎]]
- [[00_基礎理論/13_PandasとPython実装メモ]]
