---
title: SVD実験で修正した問題と設計原則
aliases:
  - SVD corrected 実験の修正点
  - SVD実験設計の注意点
  - SVD experiment corrections
  - SVD experiment invariants
tags:
  - SVD
  - NN圧縮
  - 実験設計
  - 再現性
  - PyTorch
  - Benchmark
---

# SVD実験で修正した問題と設計原則

## このノートの目的

このプロジェクトでは、SVDによるニューラルネットワーク圧縮を MNIST、Fashion-MNIST、CIFAR-10 の順に拡張する過程で、SVDの数式そのものではなく、**実験の公平性・再現性・モデル非破壊性・benchmark条件・API契約**に複数の問題が見つかった。

それらは `*_corrected.ipynb` / `*_using_src_corrected.ipynb` と `src/nn_compression/` で修正した。

このノートでは、

```text
何が問題だったか
→ なぜ問題か
→ どう直したか
→ 結果の解釈へどう影響したか
→ 今後の実験で守るルール
```

をまとめる。

ここで扱う修正は、原則として**SVDの数学を変更したものではない**。主に、実験設計と実装契約を正しくするための修正である。

---

# 1. Notebookの位置づけ

SVD実験には、試行錯誤と共通化の履歴を残すため複数世代のNotebookが存在する。

| 種別 | 位置づけ |
|---|---|
| original | 初期実験・学習履歴。historical record |
| `before_src` | `src/` 共通化以前のsnapshot |
| `using_src` | `src/` 共通化時点のsnapshot |
| `corrected` / `using_src_corrected` | 実験設計・API契約を修正後に実行したcanonicalな正式結果 |

今後、rank・精度・loss・parameters・MACs・latencyを引用するときは **correctedを最優先**する。

original / `before_src` / `using_src` に既知の問題が残っていても、履歴を消さないため原則として改変しない。

---

# 2. Test setをrank選択に使っていた

## 何が問題だったか

旧MNIST rank sweepでは、候補rankの比較にtest setを使用していた。

```text
rank候補を作る
→ test accuracyを見る
→ rankを選ぶ
```

という流れになっていた。

## なぜ問題か

Test setは、モデル・rank・ハイパーパラメータの選択がすべて終わったあとに、最終性能を確認するための独立データとして残す必要がある。

Test結果を見てrankを選ぶと、testがモデル選択へ参加するため、最終性能評価として独立ではなくなる。

## どう直したか

corrected MNISTでは、役割を分離した。

```text
Train
  ↓
Early-Stopping Validation
  ↓
Rank-Selection Validation
  ↓
rankを決定
  ↓
Testは選んだ1構成だけ最終確認
```

## 結果への影響

corrected MNISTではvalidationで `fc1=128, fc2=128` を選択し、testはその1構成だけ評価した。

正式結果は、

```text
Baseline test accuracy = 0.9817
Selected SVD           = 0.9808
```

である。

旧rank sweepのtest値はhistoricalな記録として残すが、正式なrank選択根拠には使わない。

## 今後のルール

> Testは最終モデル選択後に1回だけ使う。rank・候補・学習条件の選択には使わない。

---

# 3. Candidate間で乱数条件が揃っていなかった

## 何が問題だったか

一部の旧実験では、候補ごとに異なるseedを与えたり、DataLoader Generatorの内部状態が候補間で一致していなかった。

この場合、比較している差はrankだけではない。

```text
Candidate A
rank差 + 学習乱数差 + mini-batch順序差

Candidate B
rank差 + 別の学習乱数差 + 別のmini-batch順序差
```

となる。

## なぜ問題か

候補間の性能差をrankや圧縮率の違いとして解釈したいなら、Fine-tuning条件はできるだけ揃える必要がある。

特に短いFine-tuningでは、初期乱数やmini-batch順の差がvalidation結果へ混ざる可能性がある。

## どう直したか

候補ごとのFine-tuning直前に、同じ状態へ戻す。

```python
set_seed(SEED)
loader_generator.manual_seed(SEED)
```

corrected実験では基本的に `SEED = 0` を使う。

## 今後のルール

> Candidate比較では、比較対象以外の乱数条件を揃える。

---

# 4. shuffle付きtrain_loaderを評価で再走査していた

## 何が問題だったか

学習時に使う `shuffle=True` のDataLoaderを、そのままtrain accuracyやtrain lossの再評価にも使うと、DataLoader Generatorがさらに進む。

```text
trainingでtrain_loaderを走査
→ Generator状態が進む
→ train metric計算でも同じtrain_loaderを走査
→ Generator状態がさらに進む
→ 次epochのshuffle順が変わる
```

## なぜ問題か

評価処理そのものが後続の学習順を変えてしまう。

そのため、同じseedを設定していても、評価を入れるかどうかで次epochや次candidateのmini-batch順が変わり得る。

これは再現性と候補比較の公平性の問題になる。

## どう直したか

`src/nn_compression/training/fit.py` の `non_shuffling_loader()` を使い、train再評価用に `shuffle=False` のloaderを分ける。

```text
train_loader
  shuffle=True
  学習専用

train_eval_loader
  shuffle=False
  train metrics専用
```

`fit_with_early_stopping()` も、train再評価が必要な場合はshuffle付きtraining loaderを再走査しない契約にした。

## 今後のルール

> 学習順序を決めるGeneratorを、単なる評価処理で進めない。

---

# 5. Latency benchmarkの条件が揃っていなかった

## 何が問題だったか

旧Fashion-MNIST CNN実験の一部では、baselineとcompressedでbenchmark条件が異なっていた。

例：

```text
Baseline
warmup = 20
repeats = 2000

Compressed
warmup = 5
repeats = 200
```

また、baselineとcompressedがDataLoaderから別々のbatchを取得する実装もあった。

## なぜ問題か

速度比較では、モデル以外の条件を揃える必要がある。

- 入力Tensor
- batch size
- warmup
- repeats
- device
- CUDA synchronize

が異なると、測定差を純粋なモデル差として解釈できない。

## どう直したか

`take_inference_batch()` で比較用Tensorを1回だけ取り、baseline / compressedで共有する。

```text
same input_batch
same batch size
same warmup
same repeats
same device
correct CUDA synchronize
```

corrected Fashion CNN / CIFAR-10では、主要比較を `warmup=20, repeats=2000` に統一した。

## 結果への影響

公平な条件で最終モデルを再測定しても、parameter / MACs削減率とGPU latency短縮率は一致しなかった。以下の正式な速度値は2026-09-12の3 trial平均時間の中央値。Fashion-MNISTは新しい学習run、CIFAR-10は既存保存重みの再計測runであり、FT前sweep時間へ転用しない。詳細条件は各実験章を参照する。

Fashion-MNIST Conv+Linear corrected：

```text
MACs      4,241,152 → 2,237,184  (-47.25%)
latency   約0.346444 ms → 約0.383861 ms（最終FT後・3 trial中央値）
```

CIFAR-10 corrected：

```text
MACs      10,357,248 → 5,625,344 (-45.69%)
latency   約0.574813 ms → 約0.551291 ms（保存済み最終FT後重みの再計測・3 trial中央値）
```

したがって、このプロジェクトでは

> **理論MACs削減とwall-clock latencyは別指標**

として扱う。

低rank化では1層を複数層へ分割するため、kernel launch、中間Tensor、memory access、演算shapeなどの影響を受ける。ただし、今回それぞれの原因を分離測定したわけではないため、「SVDは一般に遅い」とは結論しない。

## 今後のルール

> MACsを減らしただけで高速化を主張しない。速度を論じる場合は同一条件の実測benchmarkを別途行う。

---

# 6. Baselineとcompressed modelがParameterを共有する危険

## 何が問題だったか

圧縮モデルを作るとき、baselineの一部moduleをそのまま流用すると、2モデルが同じParameter objectを共有する可能性がある。

その状態でcompressed modelをFine-tuningすると、baseline側まで変化し得る。

## なぜ問題か

比較対象であるbaselineがFine-tuningによって書き換わると、圧縮前後の比較そのものが無効になる。

特に非圧縮層を「そのまま使う」実装は、値が同じこととParameter objectを共有しないことを区別する必要がある。

## どう直したか

現在は、モデル単位の置換では `copy.deepcopy()` を基本とし、Linear / Convのfactor layerは新しいmoduleとして生成する。

```text
baseline model
      │
      ├── deepcopy
      ↓
compressed model
      ↓
factorized layersへ置換
```

`fc3` のような非圧縮層も含め、baselineとcompressedのParameterを共有しない。

この契約はtestsでも確認している。

## 今後のルール

> 圧縮モデルはbaselineを破壊しない。Fine-tuningしてもbaselineのParameterが変化しないことを保証する。

---

# 7. SVD layerのAPI契約が不足していた

SVDの数式が正しくても、PyTorch layerとして置換するときには追加の契約がある。

## Rank

rankは正整数というだけでは不十分で、対象行列の数学的最大rank以下でなければならない。

Linearでは、

$$
r \leq \min(D_{out}, D_{in})
$$

である。

現在は範囲外rankを暗黙に切り詰めず、明示的にエラーにする。

## device / dtype

factorized layerは元layerと同じdevice / dtypeで作る。

そうしないと、例えばCUDAモデルの一部だけCPUへ戻ったり、float64のlayerがfloat32へ変わる可能性がある。

## requires_grad

元layerがfrozenなら、factorized layerも同じtrainabilityを維持する。

`requires_grad=False` の層を分解した結果、勝手にtrainableへ戻してはいけない。

## bias

2層分解ではbiasを中間層へ入れず、最終出力側へ配置する。

## Conv2d

現在のConv SVDは、

```text
W: (C_out, C_in, kH, kW)
→ (C_out, C_in*kH*kW)
```

へ行列化する方式で、`groups=1` を前提とする。

そのため、未対応のgrouped convolutionを曖昧に処理せず明示的にrejectする。

元Convのsemantic contractを維持するため、stride / padding / dilation / padding_mode / biasを適切なfactor側へ引き継ぐ。

## Nested module

`layer1.0.conv` のようなnested pathも既存moduleだけを正確に置換する。

`set_submodule(..., strict=True)` を使い、存在しないpathを暗黙に新規作成しない。

---

# 8. Rank sweepで全candidate modelを保持していた

## 何が問題だったか

旧設計では、rank sweep結果のrecord / DataFrameにcandidate model本体を残すことがあった。

これは、

- candidate数に応じてメモリを消費する
- DataFrameにPyTorch objectが混入する
- CSVへそのまま保存できない
- `row["model"]` へNotebookが依存する

という問題を生む。

## どう直したか

現在の `collect_compression_metrics()` は原則、

```python
include_model=False
```

で数値結果だけを保持する。

Fine-tuningが必要になった候補だけ、保存したrankからモデルを再構築する。

```text
rank sweep
→ rank + metricsだけ保存
→ 候補を選択
→ 選択候補だけmodelを再構築
→ Fine-tuning
```

Fine-tuning済みmodelや最終modelなど、保持する意味がある場合だけmodel objectを持つ。

## 今後のルール

> Sweep recordは可能な限りnumeric / serializableに保つ。モデル本体は必要になった時点で再構築する。

---

# 9. CIFAR-10の探索をglobal optimumと誤読できる表現

## 実際に行った探索

CIFAR-10では、全rank組合せを総当たりしたわけではない。

```text
conv1 single-layer rank sweep
conv2 single-layer rank sweep
conv3 single-layer rank sweep
        ↓
各層のPareto frontier / knee
        ↓
knee近傍を候補化
        ↓
候補組合せだけmodel-wide評価
        ↓
Fine-tuning
```

したがって、これは

> **knee近傍に探索空間を制約したmodel-wide rank allocation**

である。

## 何を意味しないか

`conv1=9, conv2=32, conv3=48` が、考えられる全rank組合せのglobal optimumであることは示していない。

ここでいうkneeは、正規化したPareto frontier上の幾何学的折れ曲がりを使った選択規則であり、探索空間全体に対する「大域最適解」という意味ではない。

## 今後のルール

> `global` という語を使う場合は、Pareto curve内の幾何学的kneeなのか、探索空間全体のglobal optimumなのかを区別する。

---

# 10. Single seedと微小なaccuracy差

corrected実験は乱数条件を固定して再現可能にしたが、複数seedの平均・標準偏差までは取得していない。

そのため、例えば、

```text
Fashion-MNIST MLP      +0.34 pt
Fashion-MNIST Conv     +0.47 pt
Fashion Conv+Linear    +0.05 pt
CIFAR-10               +0.16 pt
```

のような微小なtest accuracy差を、SVDによる統計的な性能向上とは扱わない。

今回の中心的な結論は、

> **大きくparameter / MACsを削減してもtask accuracyをほぼ維持できる構成が存在した**

ことである。

複数seedを実行して平均・標準偏差を比較するまでは、小さいaccuracy差の符号を過度に解釈しない。

---

# 11. Corrected実験から得た実験設計上の原則

今回の修正を一般化すると、今後Tucker / TT / MPS / DMRGへ進む際も次を守る。

```text
1. Testはモデル選択に使わない
2. Candidate比較ではseedとDataLoader条件を揃える
3. 評価処理でtraining Generatorを進めない
4. Baselineを破壊・共有しない
5. Layer置換ではdevice/dtype/requires_grad/semantic contractを保つ
6. Sweep結果には原則model objectを残さない
7. 理論MACsと実測latencyを分ける
8. 探索範囲以上の最適性を主張しない
9. Single seedの微小差を性能向上と断定しない
10. Historical Notebookとcanonical Notebookを区別する
```

これらはSVD特有というより、**NN圧縮実験全般で守るべき実験契約**として扱う。

---

# 関連

- [[05_SVD基礎実装検証/01_SVD実装の確認結果]]
- [[05_SVD基礎実装検証/02_Linear層2層置換の確認結果]]
- [[10_MNIST_MLP_SVD/04_MNISTでの実験結果]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
