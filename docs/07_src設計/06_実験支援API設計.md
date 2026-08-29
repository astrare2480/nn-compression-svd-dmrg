---
title: 実験支援API設計
---

# 実験支援API設計

## 1. 対象

数学コアではないが、既存Notebook・再現実験・rank選択が依存するPublic APIをまとめる。

対象：

```text
models/
datasets/
selection/
utils/
compression/rank_sweep.py
compression/mlp_svd.py
metrics/mlp_macs.py
metrics/cnn_macs.py
```

これらはCore API v1では **Experiment Support Stable API** として扱う。

---

# 2. models

## `MNISTMLP`

```text
784
→ fc1 512
→ ReLU
→ fc2 256
→ ReLU
→ fc3 10
```

Public layer names：

```text
fc1
relu1
fc2
relu2
fc3
```

MNIST / Fashion-MNIST MLPのSVD実験が `fc1 / fc2 / fc3` 名へ依存するため、Core API v1中は主要layer名とdefault shapeを変更しない。

入力はbatch先頭次元を残してflattenする。

---

## `FashionMNISTCNN`

```text
Conv 1→32, 3x3
→ ReLU
→ MaxPool 28→14
→ Conv 32→64, 3x3
→ ReLU
→ MaxPool 14→7
→ Flatten 64*7*7 = 3136
→ Linear 3136→128
→ ReLU
→ Linear 128→10
```

Public layer names：

```text
conv1
conv2
fc1
fc2
```

SVD実験・MACs計算がこれらの名前へ依存する。

`inspect_shapes()` は学習用補助メソッドで、DataLoaderを消費せずshape確認する。

---

## `CIFAR10CNN`

既定architecture：

```text
Conv 3→32
→ pool
Conv 32→64
→ pool
Conv 64→128
→ pool
AdaptiveAvgPool 1x1
→ Linear 128→256
→ Dropout 0.5
→ Linear 256→10
```

constructor：

```python
CIFAR10CNN(
    *,
    in_channels=3,
    conv_channels=(32, 64, 128),
    hidden_dim=256,
    num_classes=10,
    dropout=0.5,
)
```

Public layer names：

```text
conv1
conv2
conv3
fc1
fc2
```

CIFAR-10 SVD / Tucker実験のcheckpoint・named compressionがこれらへ依存するため、既定値と名前を安定化する。

新しいarchitectureが必要なら、既存defaultを変更して過去結果を再解釈させず、新classまたは明示的引数で追加する。

---

# 3. datasets

## Fashion-MNIST取得

```python
get_fashion_mnist_datasets(
    data_dir,
    *,
    transform=None,
    download=True,
)
```

transform未指定時は `ToTensor()` を使う。

返り値：

```text
full_train_dataset
test_dataset
```

Dataset取得とtrain/validation分割を分離する。

---

## Fashion-MNIST split

```python
split_fashion_mnist_dataset(
    dataset,
    split_lengths,
    *,
    seed,
)
```

`random_split` + 専用 `torch.Generator` を使い、同seedなら同じ分割を再現する。

Early-Stopping ValidationとRank-Selection Validationを分ける実験を支援する。

---

## DataLoader作成

```python
make_fashion_mnist_loaders(...)
```

shuffle方針：

```text
train_loader
→ shuffle=True

validation_loader
→ shuffle=False

test_loader
→ shuffle=False

train_eval_loader
→ shuffle=False

validation_loader_rank
→ 指定時のみ作成、shuffle=False
```

学習用shuffleのGeneratorと評価走査を分離する。

---

## generic index split

```python
shuffled_index_splits(n, lengths, *, seed)
```

0..n-1をseed固定でshuffleし、指定lengthごとにindex listへ分ける。

```text
sum(lengths) <= n
length >= 0
```

余ったindexは使わない。

trainとvalidationで異なるtransformを持つDatasetへ同じindexを使う場合等に利用する。

---

# 4. seed utilities

## `set_seed`

```python
set_seed(seed=0)
```

固定対象：

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
CUBLAS_WORKSPACE_CONFIG
cudnn benchmark=False
cudnn deterministic=True
torch.use_deterministic_algorithms(..., warn_only=True)
```

ただし、完全なbit-level再現はhardware / CUDA / PyTorch / operatorに依存する。

---

## `make_torch_generator`

```python
make_torch_generator(seed=0)
```

DataLoader shuffle / random_split等の専用Generatorを返す。

重要なcontract：

```text
set_seed()
≠
既に作成したDataLoader専用Generatorの状態reset
```

候補比較前に必要なら同じGeneratorへ再度 `manual_seed` する。

---

# 5. path utilities

## `find_project_root`

上位directoryを辿り、`.git` が存在する場所をproject rootとする。

Notebookのcurrent working directoryに依存せず、project-root基準で成果物pathを作るためのAPI。

`.git` が無い展開物等ではFileNotFoundErrorになる。

---

## `get_experiment_dirs`

```python
get_experiment_dirs(
    project_root,
    method_name,
    case_name,
    experiment_name,
    *,
    create=True,
)
```

返り値：

```text
data_dir
models_dir
results_dir
```

models/resultsの分類軸：

```text
method
→ case
→ experiment
```

例：

```text
results/
└─ 20_tucker/
   └─ 10_cifar10_cnn/
      └─ 05_hosvd_vs_hooi_finetuning/
```

今後TT/MPSでも同じ分類規則を使う。

---

# 6. named module utilities

## `get_named_module`

PyTorch `get_submodule` を使い、

```text
conv2
block.0
block1.conv
```

のようなdot pathを扱う。

## `set_named_module`

既存pathにmoduleを置換する。

置換前に `get_named_module` でpath存在を確認し、`set_submodule(..., strict=True)` を使う。

存在しないpathを暗黙に作らない。

このutilityはSVD/Tucker/TT等のnamed layer replacementで再利用できる。

---

# 7. Pareto frontier

```python
extract_pareto_frontier(df, x_column, y_column)
is_pareto_candidate(df, row, x_column, y_column)
```

現在は**2目的とも小さいほど良い**前提。

候補 $a$ が、別候補 $b$ に対して、

```text
b.x <= a.x
and
b.y <= a.y
and
少なくとも一方はstrictly smaller
```

なら $a$ はdominated。

Fashion-MNIST等では、

```text
parameters
validation_loss
```

へ使用する。

maximize目的を自動変換するAPIではない。

---

# 8. knee selection

Public API：

```text
get_first_point
get_endpoints
line_equation
find_knee_point
find_knee_point_numpy
```

### input validation

- empty DataFrame拒否
- x/y column存在確認
- NaN拒否
- inf拒否

### line representation

通常：

```text
slope, intercept
```

垂直線：

```text
(inf, x_constant)
```

同一点：

```text
(0.0, y0)
```

### knee

端点を結ぶ直線から距離最大の点をkneeとする。

1点だけならその点、距離0。

全点が直線上なら先頭の最大距離点が返る。

`find_knee_point` とNumPy版は同じ意味を持つ。

---

# 9. rank sweep API

`compression/rank_sweep.py` はPublic export上はcompression packageに属するが、設計責務としては実験支援。

## generic sweep

```python
sweep_layer_ranks(...)
```

呼び出し側から、

```text
factorize function
baseline metrics
optional MACs function
```

を注入する。

各rankについてbaselineをdeepcopyし、対象named layerだけを置換する。

結果recordにmodel objectは既定で含めない。

---

## Conv2d sweep

```python
sweep_conv2d_ranks(...)
```

Conv SVDと `estimate_conv2d_macs` を組み合わせるconvenience API。

`out_hw` は呼び出し側が与える。

モデルforwardから自動推論する仕様にはしていない。

---

# 10. model-specific MACs

## MLP

`estimate_mlp_macs` は、

```text
784 → 512 → 256 → 10
```

を前提とするhistorical experiment-support API。

generic MLP estimatorではない。

## CNN

`estimate_cnn_linear_macs` はFashion-MNIST CNNのLinear部向け。

`estimate_cnn_conv2_macs` はdefaultではconv2 / 14×14だが、layer_name/out_hwを変更できる。

`estimate_cnn_macs` は、

```text
conv_output_hw
conv_ranks
linear_ranks
linear_layer_names
```

を引数化しており、現在のmodel-specific集計の中では最も一般化されている。

---

# 11. historical compatibility

既存Notebookを読める状態を保つため、次のcompatibilityをCore API v1中は維持する。

```text
make_*_svd_modelの r1 / r2
sweep_conv_svd_ranks
旧SVD alias群
```

ただし新しいNotebookではprimary API名へ寄せる。

historical Notebookを一括書換えして互換APIを削除することをCore API v1の目的にはしない。

---

# 12. TT/MPS追加時の方針

実験支援層はできるだけ再利用する。

再利用候補：

```text
models
training
benchmark
agreement/logits_rmse
Pareto/knee
seed
paths
named module utility
```

TT/MPS固有のDatasetやmodelが不要なら新規追加しない。

圧縮rankの探索は、既存Pareto/kneeを使える場合でも、TT rankが複数bondを持つためcandidate recordの列設計は新しく行う。

SVD用 `rank_sweep` の引数を無理に多次元TT rankへ拡張せず、必要ならTT専用sweep APIを追加する。
