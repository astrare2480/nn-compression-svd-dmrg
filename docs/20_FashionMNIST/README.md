# Fashion-MNIST SVD圧縮実験

Fashion-MNISTでは、MNISTから一段進めて、MLP Fine-tuning、CNN、Linear SVD、Conv SVD、Conv + Linear同時圧縮まで扱う。

このディレクトリには学習途中の試行錯誤とraw resultsも残している。**最終数値を引用するときはcorrected結果を優先する。**

## まず読むノート

現在の正式結果を確認する場合は、次を優先する。

- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]] — MLP corrected結果
- [[20_FashionMNIST/11_CNNでの実験結果]] — CNN corrected結果
- [[20_FashionMNIST/13_MLP_vs_CNNの比較と総括]] — MLPからCNN、CIFAR-10への接続

実験設計の修正理由は次を参照する。

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]

## docsの位置づけ

| docs | 位置づけ |
|---|---|
| `01_Fashion-MNIST実験.md` | MLP実験設計の学習履歴。初期BaselineやValidation分離の経緯を読むためのhistorical note |
| `02_Fashion-MNISTのRank選択.md` | 旧MLP rank-sweep runの記録。現在の最終rank根拠には使わない |
| `03_Fashion-MNISTのFine-tuning.md` | 旧MLP Fine-tuning run。candidateごとに別seedを使っていたhistorical result |
| `04_Fashion-MNISTでの実験結果.md` | **MLP correctedのcanonical result** |
| `05_全RankSweep結果.md` | 旧02/03のraw result保存用。historical |
| `06_学習履歴とFine-tuning履歴.md` | 旧MLPのepoch履歴保存用。historical |
| `07_CNN実験.md` | CNN実験全体の設計とNotebook世代の整理 |
| `08_CNNのLinear SVD.md` | `02/03` のLinear-SVD実験。独立runとして有効 |
| `09_CNNのConv SVD.md` | **Conv-only correctedを基準に整理** |
| `10_CNNのConvとLinear同時圧縮.md` | **Conv+Linear correctedを基準に整理** |
| `11_CNNでの実験結果.md` | **CNN全体のcanonical summary** |
| `12_CNN全RankSweepと学習履歴.md` | 01〜05のraw / historical record。旧04/05の数値も含む |
| `13_MLP_vs_CNNの比較と総括.md` | corrected結果を基準に構造差を総括 |

## MLPのcanonical result

```text
notebooks/10_svd/20_fashion_mnist_mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb
results/20_fashion_mnist/03_mlp_svd_finetuning_using_src_corrected/
```

```text
fc1 rank = 32
fc2 rank = 16

Parameters 535,818 → 57,098  (-89.34%)
Test acc   0.8819  → 0.8853
```

single seedの小差なので、accuracy改善ではなく、**約89%圧縮後も精度を維持した**と解釈する。

旧03では `16/16` を最終rankとしたrunが残っているが、candidate間でseed / DataLoader条件が揃っていなかった。現在のcanonicalな最終rankは `32/16`。

## CNNのcanonical result

### Linear-only 02/03

```text
fc1 rank = 24
Parameters 421,642 → 98,570  (-76.62%)
Test acc   0.9175  → 0.9187
```

`02_cnn_linear_svd.ipynb` / `03_cnn_linear_svd_finetuning.ipynb` は、レビュー時点でBaseline / compressedのbenchmark条件が揃っており、candidate Fine-tuningのseed条件も揃っていたため、04/05のようなcorrectedコピーは作っていない。

この結果は独立runとして有効。ただし、corrected 04/05とはBaselineの再学習runが異なるためabsolute値を混ぜない。

### Conv-only corrected

```text
notebooks/10_svd/30_fashion_mnist_cnn/04_cnn_conv_svd_corrected.ipynb
results/20_fashion_mnist/04_cnn_conv_svd_corrected/
```

```text
conv2 rank = 28
Parameters 421,642 → 413,066  (-2.03%)
Test acc   0.9128  → 0.9175
Latency    約0.367 → 0.370 ms/batch
```

### Conv + Linear corrected

```text
notebooks/10_svd/30_fashion_mnist_cnn/05_cnn_conv_linear_svd_corrected.ipynb
results/20_fashion_mnist/05_cnn_conv_linear_svd_corrected/
```

```text
conv2 rank = 28
fc1 rank   = 24

Parameters 421,642 → 89,994      (-78.66%)
MACs       4,241,152 → 2,237,184 (-47.25%)
Test acc   0.9128 → 0.9133
Latency    約0.378 → 0.399 ms/batch
```

`(conv2=28, fc1=24)` は各層を単独で選んだrankを組み合わせた実験で、`conv2 rank × fc1 rank` の全空間を探索したglobal optimumではない。

## CNNで分かった役割分担

```text
fc1 SVD
→ Parameters削減に効く

conv2 SVD
→ MACs削減に効く

conv2 + fc1
→ ParametersとMACsを同時に減らす
```

corrected 04/05ではbenchmarkを、

```text
same input_batch
same batch size
warmup=20
repeats=2000
```

へ統一した。

その結果、理論MACsを大きく減らしてもGPU実測latencyは短縮しなかった。

```text
MACs reduction
≠
wall-clock speedup
```

を重要な実験結果として残す。

## historical raw dataを残す理由

旧Notebookやraw docsは削除しない。

今回の学習対象にはSVDの式だけでなく、

- Testをrank選択に使わない
- Candidate間で乱数条件を揃える
- train評価でDataLoader Generatorを進めない
- benchmark条件を揃える
- baselineとcompressedでParameterを共有しない
- MACs削減とlatency短縮を同一視しない

という実験設計上の修正も含まれるためである。

正式な数値を引用するときだけ、correctedを優先する。

## 次

Fashion-MNISTではsingle-layer rank選択と、個別に選んだConv / Linear rankの組合せまで確認した。

次のCIFAR-10では、複数Conv層へrankを配るmodel-wide rank allocationへ進む。

- [[30_CIFAR10_CNN/README]]
- [[SVD実験まとめ]]
