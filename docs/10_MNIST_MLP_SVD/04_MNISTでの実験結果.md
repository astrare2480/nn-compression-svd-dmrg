---
title: MNISTでの実験結果
aliases:
  - MNIST SVD圧縮結果
  - MNIST rank sweep
  - SVD圧縮最終結果
tags:
  - MNIST
  - SVD
  - NN圧縮
  - 実験結果
  - PyTorch
  - 低ランク近似
---

# MNISTでの実験結果

## サマリー

MNIST分類用MLPの `fc1` / `fc2` をSVDで低rank2層へ置換し、rankとaccuracy・parameter数・MACs・latencyの関係を確認した。

現在の正式結果は、

```text
notebooks/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun.ipynb
results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/
```

を基準とする。

run ID: `rerun_20260912T074153Z`（2026-09-12）。現在の数値は独立した新runを出典とし、旧runの欠落値の復元ではない。途中表は本文で丸めているが、元CSVの値は変更していない。

出典：[実行済みNotebook](../../notebooks/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun.ipynb)、[Test比較](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/test_comparison.csv)、[学習・FTの要約](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/fit_summary.csv)、[最終benchmark](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/final_model_benchmark.csv)、[manifest](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/run_manifest.json)。

旧Notebookは学習・試行錯誤のhistorical recordとして残す。

2026-09-12の初回監査で、保存済みSVDモデルだけでは同じrunのbaseline重みを復元できないことを確認した。次回実行からbaseline checkpointと学習履歴を保存する処理をcorrected Notebookへ追加したが、初回監査では再学習していなかった。既存成果物のhash・未保存重みの制約は [複製元runのartifact_provenance.json](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected/artifact_provenance.json) に記録した。保存出力の整合性と現行コードのRun All成功は別の判定である。

### 承認後の独立した再実行（2026-09-12）

別名Notebook `02_rank_accuracy_tradeoff_corrected_rerun.ipynb` を全セル実行し、同一runのbaseline/最終重み・全16候補の丸め前の指標・baseline全5 epochの履歴・測定条件を新規ディレクトリへ保存した。保存重みを読み直したtest評価も一致した。従来runのNotebook・CSV・重みは保存したまま、本ページの正式な出典を新runへ切り替えた。最終rankとTest値は従来correctedと一致する。新しい記録は [完全保存runの一覧と検証](../../results/10_svd/rerun_20260912T074153Z/README.md) に分けた。過去runの欠落値が復元されたという意味ではない。

---

## corrected版で直した最重要点

旧rank sweepではtest setをrank選択に使っていた。

```text
旧
rank候補
→ testを見る
→ rankを選ぶ
```

これはtest leakageになるため、corrected版では、

```text
Train
→ Early-Stopping Validation
→ Rank-Selection Validation
→ rank選択
→ Testは選んだ1構成だけ最終確認
```

へ分離した。

さらに、

- candidate比較のseed / DataLoader Generator条件を固定
- train指標は `shuffle=False` の `train_eval_loader` で評価
- baseline / compressedで同じ `input_batch` を使用
- benchmarkを同じ `warmup=20`, `repeats=200` へ統一
- baseline / compressedでParameterを共有しない
- rank sweep結果へmodel本体を保持しない

という実験契約も揃えた。

詳細は [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] を参照。

---

# 1. モデル

Baseline MLP：

```text
784 → 512 → 256 → 10
```

圧縮対象：

```text
fc1: Linear(784, 512)
fc2: Linear(512, 256)
```

SVD後：

```text
fc1
784 → r1 → 512

fc2
512 → r2 → 256
```

`fc3 = Linear(256, 10)` は非圧縮。

Baseline parameter数は `535,818`。

---

# 2. Rank sweep

候補：

```python
R1_LIST = [16, 32, 64, 128]
R2_LIST = [16, 32, 64, 128]
```

16通りをRank-Selection Validationで比較した。

validation上で選択された最終rankは、

```text
fc1 = 128
fc2 = 128
```

である。

### 選択rankのvalidation結果

| 項目 | 値 |
|---|---:|
| fc1 rank | 128 |
| fc2 rank | 128 |
| Validation loss | 0.077302 |
| Validation accuracy | 0.9776 |
| Parameters | 267,530 |
| Parameter reduction | 50.07% |
| MACs | 266,752 |
| MAC reduction | 50.14% |
| Baseline latency（rank sweep・単回） | 0.251118 ms/batch |
| Compressed latency（rank sweep・単回） | 0.208188 ms/batch |

このrankは「最も小さいrank」ではなく、今回のvalidation選択規則で採用された構成である。

![MNIST MLPの16通りのrankとvalidation性能](assets/02_rank_accuracy_tradeoff_corrected_rerun/rank_accuracy_tradeoff_validation.png)

掲載画像はdocs内表示用の複製で、[出典runの元画像](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/) とバイト同一。旧runの画像は上書きしていない。

図：上記rerunのSVD直後・FTなしのrank sweep。左はfc1 rankを行、fc2 rankを列とするRank-Selection Validation accuracy、右はモデル全体のparametersと同validation loss。[元データ](../../results/10_svd/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected_rerun/rank_sweep_validation.csv) の16構成を表示しており、最終Test accuracyや推論時間の図ではない。

---

# 3. 最終Test

Testはrank選択には使用せず、最終選択後にbaselineと選択済み1構成を評価した。保存checkpointからの再評価でも同じ値を確認した。

| Model | fc1 rank | fc2 rank | Test loss | Test accuracy |
|---|---:|---:|---:|---:|
| Baseline | — | — | **0.065442** | **0.9817** |
| Selected SVD | **128** | **128** | 0.065758 | **0.9808** |

Accuracy差：

$$
0.9808 - 0.9817 = -0.0009
$$

**-0.09 percentage point**。

約50%のparameter削減後も、Test accuracyはほぼ同水準を維持した。

---

# 4. Rankと圧縮率

Linear層、

$$
W \in \mathbb{R}^{D_{out}\times D_{in}}
$$

をrank $r$ で、

```text
Linear(D_in → r, bias=False)
Linear(r → D_out, bias=元bias)
```

へ分けると、weight parameter数は概ね、

$$
r(D_{in}+D_{out})
$$

となる。

rankを小さくするとparameter数・MACsは減るが、近似誤差は大きくなる。

MNIST rank sweepでも、`r1=16` まで下げるとaccuracyが大きく低下した一方、32以上ではかなり回復した。

したがって、

```text
rankを小さくするほど良い
```

ではなく、**サイズ削減とtask性能のトレードオフ**として選ぶ必要がある。

---

# 5. Retained energyとtask性能

選択rank `128 / 128` のretained energyは、

```text
fc1 ≈ 0.8872
fc2 ≈ 0.9506
```

である。

ただし、retained energyは重み行列のSVD近似を表す指標であり、分類accuracyそのものではない。

```text
重み近似誤差
≠
logits誤差
≠
classification accuracy
```

として区別する。

---

# 6. MACsとlatency

corrected benchmarkの条件は、

```text
same input_batch
batch size = 64
input shape = (64, 1, 28, 28)
warmup = 20
repeats = 200
```

である。

選択rankでは、理論MACsは約50%減った。

```text
Baseline   ≈ 535,040 MACs
Compressed = 266,752 MACs
```

最終選択後の同一runのbaseline / selected SVDを3 trialで計測し、各trialの平均時間の中央値を採用した。実測latencyは、

```text
Baseline   ≈ 0.119641 ms/batch
Compressed ≈ 0.195747 ms/batch
```

で、短縮しなかった。`eval()` / `no_grad()`、CUDA同期あり、host→device転送を除外した。

rank sweepの単回値 `0.251118 → 0.208188 ms/batch` は探索時の補助値であり、最終3 trial計測とは測定段階・反復集計が異なる。両者を混ぜず、速度の最終記載は `final_model_benchmark.csv` を出典とする。

小規模MLPを2層へ分割すると、行列積そのもののMACs以外の固定オーバーヘッドが相対的に効く可能性がある。

ただし個々の要因は分離測定していないため、一般にSVDモデルが遅いとは結論しない。

この実験では、

> **MACs削減とwall-clock latencyは別に測る必要がある**

ことを確認した。

---

# 7. Historical結果の扱い

旧Notebook / 旧docsには、`64 / 64` や別runのTest accuracy・latencyが残っている。

それらは当時の実験記録として残すが、現在のcanonical resultではない。

特に旧rank sweepはtestをrank選択へ利用していたため、最終rankの根拠としては使用しない。

現在の優先順位：

```text
corrected rerun（今回の正式出典）
→ 複製元corrected（従来run）
→ using_src
→ original / before_src
```

---

# 8. このMNIST実験で確認できたこと

1. 学習済みLinear重みへSVDを適用できる。
2. 低rank2層として保持すればparameter圧縮になる。
3. rankを下げるほど圧縮率は上がるが、task性能とのトレードオフがある。
4. retained energyだけではtask accuracyを決められない。
5. Testはrank選択に使わず、最終確認へ分離する必要がある。
6. benchmarkは同じ入力・warmup・repeatsで比較する必要がある。
7. MACs削減は実測latency短縮を保証しない。

MNISTでSVD圧縮と公平なrank評価の基本を確認し、次にFashion-MNISTでFine-tuning・CNN・Conv SVDへ拡張する。

---

# 関連

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
