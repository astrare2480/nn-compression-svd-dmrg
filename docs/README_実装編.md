---
title: SVD実装編 目次
aliases:
  - SVD実装編
  - NN圧縮実装ロードマップ
  - PyTorch SVD圧縮実装
tags:
  - PyTorch
  - SVD
  - NN圧縮
  - Linear
  - Conv2d
  - 実装
---

# SVD実装編 目次

## サマリー

現在のSVD実装の正本は、旧 `code/` 内のNotebook関数ではなく、

```text
src/nn_compression/
```

へ分離した共通実装である。

Notebookは、

```text
実験条件
rank候補
選択規則
結果の解釈
```

を担当し、再利用可能な処理は `src` に置く。

SVD編のcorrected実験とCodex reviewを通して、SVDの数式だけでなく、

- model非共有
- rank validation
- device / dtype / requires_grad
- DataLoader Generator
- benchmark公平性
- nested module replacement
- Pareto / knee

まで実装契約として整理した。

---

# 1. 現在のディレクトリ構成

```text
src/nn_compression/
├─ compression/
├─ datasets/
├─ metrics/
├─ models/
├─ selection/
├─ training/
└─ utils/
```

大きく、

```text
圧縮
データ・分割
評価
モデル
rank選択
学習
共通utility
```

へ責務を分けている。

---

# 2. `compression/`

SVD圧縮の中心。

```text
compression/
├─ svd.py
├─ linear_svd.py
├─ conv_svd.py
├─ mlp_svd.py
├─ named_layers.py
└─ rank_sweep.py
```

## `svd.py`

モデルに依存しないSVD処理。

主な役割：

```text
truncated SVD
rank validation
retained energy
```

rankは、

```text
1 <= rank <= mathematical max rank
```

を要求する。

上限を超えたrankを黙ってclipせず、誤った実験条件として明示的に扱う。

## `linear_svd.py`

任意の `nn.Linear` を低rank2層へ変換する。

```text
Linear(D_in → D_out)
↓
Linear(D_in → r, bias=False)
Linear(r → D_out, bias=元bias)
```

分解後も、

```text
device
dtype
requires_grad
```

を元layerから引き継ぐ。

named layerを置換するときはmodelを `deepcopy` し、baselineとParameterを共有しない。

## `conv_svd.py`

`groups=1` の `nn.Conv2d` を対象に、

```text
Conv2d(C_in → C_out, K×K)
↓
Conv2d(C_in → r, K×K, bias=False)
Conv2d(r → C_out, 1×1, bias=元bias)
```

へ分解する。

元Convの、

```text
stride
padding
dilation
padding_mode
```

は空間畳み込みを行う前段へ引き継ぐ。

biasは最終出力側へ置く。

現在のweight flatteningは `groups=1` 前提なので、grouped convolutionは明示的にrejectする。

## `mlp_svd.py`

現在のMLP実験向けhelper。

`fc1_rank` / `fc2_rank` を主APIとし、旧Notebook互換の `r1` / `r2` aliasも扱う。

返す圧縮modelはbaselineとParameterを共有しない。

## `named_layers.py`

複数のnamed Conv / Linearをまとめて圧縮するときの共通処理。

元のmodelを変更せず、コピー側だけを置換する。

## `rank_sweep.py`

層ごとのrank sweepを共通化する。

rank sweep中は、原則としてcandidate model本体を結果表へ保持しない。

```text
include_model=False
```

を基本とし、Fine-tuningするcandidateだけrankから再構築する。

これにより、

- 不要なmodel保持
- DataFrameへのPyTorch object混入
- CSVへ保存できないrecord

を避ける。

---

# 3. `models/`

```text
models/
├─ mlp.py
├─ cnn.py
└─ cifar10.py
```

## `mlp.py`

MNIST / Fashion-MNIST MLPの共通model。

## `cnn.py`

Fashion-MNIST CNNのmodel。

## `cifar10.py`

CIFAR-10用の、

```text
Conv×3
+ GAP
+ fc1
+ Dropout
+ fc2
```

構造を持つ。

既定構成：

```text
conv1 3 → 32
conv2 32 → 64
conv3 64 → 128
GAP
fc1 128 → 256
Dropout 0.5
fc2 256 → 10
```

GAPを使うことで巨大なFlatten→Linearを避け、CIFAR-10実験ではConv圧縮を主題にする。

---

# 4. `training/`

学習・評価・Early Stoppingを共通化する。

重要な契約は、**学習用shuffle DataLoaderをtrain metricsの再評価で再走査しない**こと。

```text
train_loader
→ shuffle=True
→ 学習専用

train_eval_loader
→ shuffle=False
→ train loss / accuracy確認用
```

学習用loaderを評価で追加走査すると、Generator状態が進み、次epochや次candidateのmini-batch順を変える可能性がある。

このため、現在の実装ではnon-shuffling loaderを評価用に使う。

詳細：

- [[00_基礎理論/19_再現性と乱数管理]]

---

# 5. `metrics/`

```text
parameters
accuracy / loss
agreement
logits RMSE
retained energy
MACs
latency
```

などを扱う。

## latency benchmark

corrected実験では、baselineとcompressedの速度比較条件を揃える。

```text
same input_batch
same batch size
same warmup
same repeats
proper CUDA synchronize
```

別々にDataLoaderからbatchを取ると、入力条件差がbenchmarkへ混ざる。

したがって比較用 `input_batch` を一度作り、両modelへ同じTensorを渡す。

## `include_model=False`

metrics収集は数値recordを基本とし、rank sweepの全candidate modelを保持しない。

必要なmodelだけ再構築する。

---

# 6. `selection/`

Pareto frontierとknee計算を共通化する。

重要なのは、

```text
Pareto / kneeの数式処理
```

と、

```text
どのmetricを使うか
どの候補をFine-tuningするか
最終rankをどう決めるか
```

を分けること。

前者は `src/selection`、後者は実験Notebookへ残す。

kneeは唯一絶対のrankを証明するものではなく、指定した評価軸に対する折衷点を作る選択規則。

---

# 7. `datasets/`

Fashion-MNIST用loaderやsplit helperを置く。

CIFAR-10の前処理・splitは現在Notebook側の実験設計として残っている部分もある。

データ処理で重要なのは、

```text
train augmentation
validation / test fixed transform
split RNG
training RNG
```

を区別すること。

- [[00_基礎理論/18_CIFAR10の前処理とDataLoader]]
- [[00_基礎理論/19_再現性と乱数管理]]

---

# 8. `utils/`

named moduleの取得・置換などを扱う。

例えば、

```text
layer1.0.conv
```

のようなnested pathでも既存moduleを置換できるようにする。

`set_submodule(..., strict=True)` を使い、存在しないpathを暗黙に新規作成しない。

このAPIを使用するため、現在の `pyproject.toml` は、

```text
torch>=2.7
```

を最低versionとしている。

---

# 9. model非共有が重要な理由

圧縮modelをFine-tuningしたとき、baselineのParameterまで変わってはいけない。

NG：

```text
baseline
└─ Parameter A

compressed
└─ 同じ Parameter A
```

これではbaseline比較が壊れる。

現在は、

```text
deepcopy
+
新しいfactor layer生成
```

でbaselineとcompressedを分離する。

---

# 10. rank / device / dtype / requires_grad

factorization helperは、shapeだけ正しければよいわけではない。

## rank

対象行列の数学的最大rank以下。

## device

CUDA modelの一部だけCPUへ戻さない。

## dtype

float64等の元dtypeを勝手にfloat32へ落とさない。

## requires_grad

frozen parameterを分解した結果、trainableに変えない。

これらをlayer factorizationのAPI contractとして扱う。

---

# 11. Notebookとの役割分担

現在の方針：

```text
src
→ 再利用可能な実装
→ 守るべきcontract

Notebook
→ 実験固有rank
→ candidate集合
→ 学習条件
→ 選択規則
→ 結果と考察
```

したがって、`src` へ、

```text
CIFARの最終rank=9/32/48
```

のような実験固有値をhard-codeしない。

---

# 12. corrected Notebook

正式結果として扱うcorrected Notebook：

```text
MNIST
notebooks/10_mnist_mlp/02_rank_accuracy_tradeoff_corrected.ipynb

Fashion-MNIST MLP
notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning_using_src_corrected.ipynb

Fashion-MNIST CNN Conv
notebooks/20_fashion_mnist/cnn/04_cnn_conv_svd_corrected.ipynb

Fashion-MNIST CNN Conv + Linear
notebooks/20_fashion_mnist/cnn/05_cnn_conv_linear_svd_corrected.ipynb

CIFAR-10
notebooks/30_cifar10/02_svd_global_compression_using_src_corrected.ipynb
```

historical Notebookは削除せず、正式な数値引用ではcorrectedを優先する。

---

# 13. tests

SVD実装の最終review時点で、

```text
python -m pytest tests -q --tb=short

49 passed
```

を確認している。

テスト対象には、

- rank contract
- Linear / Conv factorization
- device / dtype
- requires_grad
- Parameter非共有
- nested module replacement
- benchmark helper
- knee edge case

などが含まれる。

---

# 14. 旧 `code/` の位置付け

旧 `code/` のNotebookやartifactは、初期実装・学習履歴として残っている。

現在は、

```text
code/
= historical

src/nn_compression/
= current reusable implementation
```

と考える。

legacyな `code/data/MNIST` や `code/mnist_mlp_baseline.pth` はGit追跡から解除済みで、今後のcanonical実装ではない。

---

# 15. 修正履歴を読む

何が問題で、なぜ今の実装になったかは、

- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

にまとめている。

srcには長いバグ履歴を書かず、

```text
なぜ現在のcontractを守る必要があるか
```

だけを短いdocstring / commentとして残している。

---

# 16. Tuckerへ持ち越すもの

次のTucker decompositionで再利用するのは、SVD専用factorizationそのものではなく、

```text
models
training
metrics
benchmark
selection
named module操作
実験結果保存
```

の基盤。

Tucker用の巨大な万能decomposition frameworkを先回りして作らず、必要になった段階で抽象化する。

---

# 関連

- [[README]]
- [[SVD実験まとめ]]
- [[00_基礎理論/README]]
- [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
