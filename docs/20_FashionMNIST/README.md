# Fashion-MNIST SVD圧縮実験

Fashion-MNISTでは、MNISTから一段進めて、

```text
MLP Fine-tuning
CNN
Linear SVD
Conv SVD
Conv + Linear同時圧縮
```

を扱う。

このディレクトリにはhistorical runの詳細も残しているため、**最終数値を引用するときはcorrected結果を優先する**。

---

## MLP編

読む順番：

1. [[20_FashionMNIST/01_Fashion-MNIST実験]]
2. [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
3. [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
4. [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]

raw / historical結果：

- [[20_FashionMNIST/05_全RankSweep結果]]
- [[20_FashionMNIST/06_学習履歴とFine-tuning履歴]]

### canonical MLP結果

正式結果：

```text
notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb
results/20_fashion_mnist/03_mlp_svd_finetuning_using_src_corrected/
```

```text
fc1 rank = 32
fc2 rank = 16

Parameters: 535,818 → 57,098  (-89.34%)
Test acc:   88.19% → 88.53%   (+0.34pt)
```

single seedの小差なので、accuracy改善とは主張せず、**約89%のparameter削減後も精度をほぼ維持した**と解釈する。

旧MLP docsには `16/16` を最終rankとしたhistorical runも残る。現在のcanonicalな最終rankは `32/16`。

---

## CNN編

読む順番：

1. [[20_FashionMNIST/07_CNN実験]]
2. [[20_FashionMNIST/08_CNNのLinear SVD]]
3. [[20_FashionMNIST/09_CNNのConv SVD]]
4. [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
5. [[20_FashionMNIST/11_CNNでの実験結果]]

raw / historical結果：

- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
- [[20_FashionMNIST/13_MLP_vs_CNNの比較と総括]]

### Linear-only

Linear-onlyの03 runでは、

```text
fc1 rank = 24
Parameters: 421,642 → 98,570  (-76.62%)
Test acc:   91.75% → 91.87%   (+0.12pt)
```

となった。

このrunはcorrected 04/05とは別runなので、absolute baseline accuracyを混ぜず、**run内のbaseline差**として読む。

### Conv-only corrected

正式結果：

```text
notebooks/20_fashion_mnist/cnn/04_cnn_conv_svd_corrected.ipynb
results/20_fashion_mnist/04_cnn_conv_svd_corrected/
```

```text
conv2 rank = 28
Parameters: 421,642 → 413,066  (-2.03%)
Test acc:   91.28% → 91.75%   (+0.47pt)
Latency:    約0.367 → 0.370 ms/batch
```

### Conv + Linear corrected

正式結果：

```text
notebooks/20_fashion_mnist/cnn/05_cnn_conv_linear_svd_corrected.ipynb
results/20_fashion_mnist/05_cnn_conv_linear_svd_corrected/
```

```text
conv2 rank = 28
fc1 rank   = 24

Parameters: 421,642 → 89,994      (-78.66%)
MACs:       4,241,152 → 2,237,184 (-47.25%)
Test acc:   91.28% → 91.33%       (+0.05pt)
Latency:    約0.378 → 0.399 ms/batch
```

`(conv2=28, fc1=24)` は単独実験で選んだrankを組み合わせたもの。`conv2 rank × fc1 rank` の全空間を探索したglobal optimumではない。

---

## CNNの比較で重要な点

`fc1` と `conv2` は圧縮効果の性質が異なる。

```text
fc1 SVD
→ Parameters削減に効く

conv2 SVD
→ MACs削減に効く

conv2 + fc1
→ ParametersとMACsを同時に減らす
```

またcorrected 04/05ではbenchmarkを、

```text
same input_batch
same batch size
warmup=20
repeats=2000
```

へ公平化した。

その結果、MACsを大きく減らしてもGPU実測latencyは短縮しなかった。

```text
MACs reduction
≠
wall-clock speedup
```

を重要な結果として残す。

---

## Fine-tuning

SVDはweightの低rank近似を作るが、task lossを直接最適化しているわけではない。

```text
SVD
→ 低rank初期値

Fine-tuning
→ task lossへ再適応
```

Fashion-MNIST MLP / CNNとも、SVD直後の性能低下をFine-tuningで大きく回復できた。

ただし、single seedの小さなaccuracy上昇を「SVDにより性能改善」と一般化しない。

---

## correctedで直した主な点

- candidate間のseed条件
- DataLoader Generatorの公平性
- train評価でshuffle loaderを再走査しない
- baseline / compressedのParameter非共有
- benchmarkのsame input batch / warmup / repeats
- current `src` APIへの統一

詳細：

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[00_基礎理論/19_再現性と乱数管理]]

---

## 共通理論

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

---

## 次

Fashion-MNISTでは、single-layer rank選択と、個別に選んだConv + Linear rankの組合せまで確認した。

次のCIFAR-10では、複数Conv層へrankを配るmodel-wide rank allocationへ進む。

- [[30_CIFAR10_CNN/README]]
- [[SVD実験まとめ]]
