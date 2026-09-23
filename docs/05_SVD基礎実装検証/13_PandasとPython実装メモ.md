---
title: PandasとPython実装メモ
aliases:
  - Pareto実装のPandas操作
  - DataFrame操作メモ
  - rank選択実装メモ
tags:
  - Python
  - Pandas
  - NumPy
  - 実装
  - Pareto
  - DataFrame
---

# PandasとPython実装メモ

## サマリー

rank sweep、Pareto frontier、knee point、Fine-tuning候補管理のような小規模な実験集計では、

- **PyTorch**：モデル学習、SVD、推論、loss / accuracy
- **Pandas**：rankごとの結果表、条件抽出、並べ替え、CSV保存
- **NumPy / Python**：Pareto・kneeの幾何計算
- **Matplotlib**：可視化

と役割を分けると見通しがよい。

候補が数十〜数百程度なら、Pareto判定をPandasのBoolean Seriesとループで素直に書いても十分扱いやすい。

関連：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]、[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]、[[05_SVD基礎実装検証/12_PyTorch学習と評価の基礎]]

---

## 1. DataFrameの「行数」と「要素数」を区別する

候補モデル数を知りたい場合は、通常 `len(df)` または `df.shape[0]` を使う。

```python
n_rows = len(df)
n_rows = df.shape[0]
```

`df.size` は **行数 × 列数の全要素数** なので、候補数とは意味が異なる。

```text
shape = (10, 8)
len(df) = 10

df.size = 80
```

---

## 2. `iterrows()` は index と1行分のSeriesを返す

```python
for index, row in df.iterrows():
    print(index)
    print(row["parameters"])
```

`row` は1行分の `Series` であり、列名で値を取り出せる。

rank候補が30個程度のような小規模な表なら、可読性を優先して `iterrows()` を使って問題ない。
大量の行を処理する場合は、ベクトル化や `itertuples()` を検討する。

---

## 3. Pandasの条件式では `and / or` ではなく `& / |`

Pythonの単一Booleanなら、

```python
a and b
a or b
```

を使う。

一方、PandasのBoolean Series同士を要素ごとに比較するときは、

```python
condition_a & condition_b
condition_a | condition_b
```

を使う。

また、演算子の優先順位で意図しない評価にならないよう、**各条件を括弧で囲む**。

```python
mask = (
    (df["parameters"] <= row["parameters"])
    & (df["validation_loss"] <= row["validation_loss"])
)
```

---

## 4. `.any()` と `.all()`

Boolean Seriesに対して、

```python
mask.any()
```

は「1つでも `True` があるか」を返す。

```python
mask.all()
```

は「すべて `True` か」を返す。

Pareto支配判定では、

> 自分を支配する別候補が **1つでも存在するか**

を確認するため、`.any()` が自然である。

---

## 5. Pareto支配をPandasで表す

2目的を、

- `parameters`：小さいほど良い
- `validation_loss`：小さいほど良い

とする。

ある行 `row` を別の候補が支配する条件は、

1. parametersが `row` 以下
2. validation lossが `row` 以下
3. 少なくとも片方は厳密に小さい

である。

```python
dominated = (
    (df["parameters"] <= row["parameters"])
    & (df["validation_loss"] <= row["validation_loss"])
    & (
        (df["parameters"] < row["parameters"])
        | (df["validation_loss"] < row["validation_loss"])
    )
)
```

3番目の条件があるため、自分自身は「自分を厳密に支配する候補」にはならない。

```python
if not dominated.any():
    keep_indices.append(index)
```

この判定の数学的意味は [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]] を参照。

---

## 6. `.loc` と `.iloc` の違い

### `.loc`

**ラベル**またはBoolean Seriesで選択する。

```python
selected = df.loc[df["fc1_rank"] == 32]
```

### `.iloc`

**位置**で選択する。

```python
first = df.iloc[0]
previous = df.iloc[knee_pos - 1]
next_row = df.iloc[knee_pos + 1]
```

Pareto frontierをparameter順に並べ、kneeの「前後の行」を取る場合は、ラベルではなく**位置**が欲しいので `.iloc` が自然である。

```python
df = df.sort_values("parameters").reset_index(drop=True)

left = max(knee_pos - 1, 0)
right = min(knee_pos + 1, len(df) - 1)

neighbors = df.iloc[left : right + 1]
```

---

## 7. 文字列と列を取り違えない

次の式は、DataFrameの列を比較していない。

```python
"parameters_normalize" == p[0]
```

左辺は単なる文字列なので、数値 `p[0]` と比較して `False` になる。

列の値と比較するなら、

```python
df["parameters_normalize"] == p[0]
```

と書く。

複数条件なら、

```python
mask = (
    (df["parameters_normalize"] == p[0])
    & (df["validation_loss_normalize"] == p[1])
)

knee_row = df.loc[mask]
```

とする。

---

## 8. `sort_values()` と `reset_index(drop=True)`

```python
df = (
    df
    .sort_values("parameters")
    .reset_index(drop=True)
)
```

`sort_values()` は行を並べ替えるが、元のindexラベルはそのまま残る。

`reset_index(drop=True)` を使うと、

```text
0, 1, 2, 3, ...
```

と位置に対応したindexへ振り直せる。

`drop=True` は、古いindexを新しい列として残さず捨てる指定である。

kneeの前後を `.iloc` で扱う場合、先にこれを行うと混乱しにくい。

---

## 9. `.drop()` は行・列を落とす

列を落とす例：

```python
df_view = df.drop(columns=["model"])
```

行をindexで落とす例：

```python
df_new = df.drop(index=[3, 5])
```

通常は新しいDataFrameが返るため、必要なら代入する。

PyTorchの `nn.Module` を一時的にDataFrameへ保持している場合、表示やCSV保存では `model` 列を除くと扱いやすい。

---

## 10. 1つの値・1行を取り出す

Boolean抽出の結果は、1件だけであってもDataFrameである。

```python
matched = df.loc[
    (df["fc1_rank"] == r1)
    & (df["fc2_rank"] == r2)
]
```

1行だけと分かっているなら、

```python
row = matched.iloc[0]
model = matched.iloc[0]["model"]
```

のように取り出す。

「本当に1行だけであるべき」処理なら、先に件数を検証すると安全である。

```python
if len(matched) != 1:
    raise ValueError("候補が一意に見つからない")
```

---

## 11. `.at` は1セルを扱う

`.at` は、**1つの行ラベル + 1つの列ラベル**で1セルを読み書きするためのもの。

```python
df.at[index, "validation_loss"] = loss
```

複数条件で行を探して結果を追加する用途には向かない。

実験結果を何行も作る場合は、後述の `list[dict]` が扱いやすい。

---

## 12. 結果は `list[dict]` に貯めて最後にDataFrame化する

学習・rank sweep・Fine-tuning結果は、ループ中にDataFrameへ無理に1セルずつ書くより、レコードをappendする方が単純である。

```python
records = []

for ...:
    records.append(
        {
            "fc1_rank": fc1_rank,
            "fc2_rank": fc2_rank,
            "validation_loss": validation_loss,
            "validation_acc": validation_acc,
        }
    )

df_results = pd.DataFrame(records)
```

この形なら、後から列を追加・並べ替え・CSV保存しやすい。

---

## 13. DataFrameにモデル本体を一時保持してもよい

小規模なNotebook実験では、

```python
{
    "fc1_rank": fc1_rank,
    "fc2_rank": fc2_rank,
    "model": compressed_model,
}
```

のように `nn.Module` をobject列として保持すると、rankとモデルの対応を管理しやすい。

ただし、CSVにはPyTorchモデルを保存できないため、保存前に除く。

```python
def drop_model_col(df):
    return df.drop(columns=["model"], errors="ignore")


drop_model_col(df_results).to_csv(path, index=False)
```

モデル重みは別途 `torch.save(model.state_dict(), path)` で保存する。

---

## 14. `MinMaxScaler` は各列を独立に0〜1へ変換する

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[[
    "parameters_normalize",
    "validation_loss_normalize",
]] = scaler.fit_transform(
    df[["parameters", "validation_loss"]]
)
```

これは各列について、

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

を計算している。

Pareto / kneeでは、**Pareto frontier抽出後の点群**に対して正規化し、parameterとlossの尺度を揃える。

詳細：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]

---

## 15. kneeの端点と前後候補は意味を分ける

knee計算の端点は、Pareto frontierの両極端である。

一方、Aggressive / Balanced / Conservativeを作るときの「左右」は、

```text
parameters 小 ←────────────→ parameters 大
```

とparameter昇順に並べたPareto frontier上の**隣接点**を意味する。

したがって、rank値そのものの「1段階前後」とは限らない。

---

## 16. NumPy / Pythonで距離計算する

knee pointの候補数は通常少ないため、幾何計算をGPUへ載せる必要はほぼない。

点と直線の距離のような計算は、NumPyやPythonのscalar演算で十分である。

```python
import numpy as np

point = np.array([x, y], dtype=float)
p0 = np.array([x0, y0], dtype=float)
p1 = np.array([x1, y1], dtype=float)
```

モデル学習や重みSVDはPyTorch、候補表と小さな幾何計算はPandas / NumPy、と分けると責務が明確になる。

---

## 17. `Path` で実験ディレクトリを管理する

Notebookの実行位置に依存しにくくするには `pathlib.Path` が便利である。

```python
from pathlib import Path

project_root = Path.cwd()
results_dir = project_root / "results" / "20_fashion_mnist"
results_dir.mkdir(parents=True, exist_ok=True)
```

プロジェクトルートを `.git` のある親ディレクトリとして探索する方法もある。

```python
def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()

    for candidate in [start_path, *start_path.parents]:
        if (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError("project root not found")
```

これにより、Notebook、models、resultsの保存先を一貫させやすい。

---

## 18. Notebookを再実行するときの注意

DataFrameの中に保持したモデルをそのままFine-tuningすると、同じセルを再実行したときに「元の圧縮モデル」ではなく「すでにFine-tuning済みのモデル」から再学習を始めることがある。

そのため、学習対象は独立コピーにする。

```python
pareto_model = copy.deepcopy(
    pareto_row["model"]
).to(device)
```

さらに乱数状態もセル実行ごとに進むため、正式な比較結果を得るときは、

```text
Restart Kernel
→ Run All
```

のように実行経路を固定する。

詳細：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]、[[05_SVD基礎実装検証/12_PyTorch学習と評価の基礎]]

---

## 19. 実験コードで残すべきもの／残さなくてよいもの

### 残す価値が高い

- rankごとの指標を作るロジック
- Pareto支配の条件
- kneeの数学的定義
- Fine-tuning前後を対応付けるキー
- モデル選択規則
- CSV / modelの保存規則

### 機械的に任せやすい

- 表の見栄え調整
- グラフの装飾
- 列の並べ替え
- 型ヒントや関数分割
- CSV出力の定型処理

実験の意味を決めるロジックと、単なる整形処理を分けることが重要である。

---

## 20. `list` / `tuple` / `set` / `dict` の最小整理

CNN実装中にshapeや戻り値の型を読むため、Pythonの基本containerを区別する。

| 型 | 例 | 順序 | 変更 | 主な用途 |
|---|---|---|---|---|
| `list` | `[1, 2, 3]` | あり | 可能 | 可変な値の並び |
| `tuple` | `(1, 2, 3)` | あり | 不可 | 固定的な組・shape表現 |
| `set` | `{1, 2, 3}` | 順序を前提にしない | 可能 | 重複なし集合 |
| `dict` | `{"rank": 16}` | 挿入順を保持 | 可能 | keyとvalueの対応 |

実験結果を貯めるときは、

```python
records: list[dict] = []
```

が扱いやすい。

一方、Tensorのshapeは、

```python
weight.shape
```

で取得する。

PyTorchでは `torch.Size` が返り、tupleのように、

```python
out_ch, in_ch, k_h, k_w = weight.shape
```

とunpackできる。

### container操作を、実際の値で確認する

型名だけでなく、変更・追加・重複・添字の違いも確認する。

```python
# listは位置で取得でき、変更・追加・重複を許す。
filters = [32, 64, 128]
filters[0] = 16
filters.append(256)
assert filters == [16, 64, 128, 256]

# tupleは位置の意味が固定した組。要素そのものの再代入はできない。
shape = (64, 32, 3, 3)
out_ch, in_ch, k_h, k_w = shape
assert (out_ch, in_ch, k_h, k_w) == (64, 32, 3, 3)
assert not isinstance((3), tuple)
assert isinstance((3,), tuple)  # 1要素tupleにはカンマが必要。

# setは重複が消える。表示順・位置による取得を前提にしない。
classes = {"T-shirt", "Sandal", "Sandal", "Bag"}
assert classes == {"T-shirt", "Sandal", "Bag"}
assert "Sandal" in classes

# dictはkeyで値を取得する。現行Pythonでは挿入順を保持する。
layer_info = {
    "in_channels": 32,
    "out_channels": 64,
    "kernel_size": 3,
    "padding": 1,
}
assert layer_info["out_channels"] == 64

# 入れ子tupleを平らにする操作はTensor.flattenとは別である。
nested = ((1, 2), (3, 4))
flat = tuple(value for group in nested for value in group)
assert flat == (1, 2, 3, 4)
```

`shape[0] = 128` はtupleの変更エラー、`classes[0]` はsetの添字エラーになる。`torch.Size` はtupleのsubclassで、取得・unpack・`tuple(weight.shape)` が使える。`shape.flatten()` はsize情報へTensorメソッドを呼ぶ誤りであり、行列化する対象は `weight.flatten(start_dim=1)` である。

`for i = 0 in shape[0]:` はPythonの文法ではない。`for i in range(shape[0]):` なら $i=0,\ldots,63$ を順に使える。

[Pythonのcontainerの定義](https://docs.python.org/3/library/stdtypes.html)

[torch.Sizeの定義](https://docs.pytorch.org/docs/stable/size.html)

---

## 21. Tensor / Parameter / shapeを区別する

### Tensor

```python
W_mat = conv.weight.detach().flatten(start_dim=1)
```

の `W_mat` は `torch.Tensor` である。

### Parameter

```python
conv.weight
```

は通常 `nn.Parameter` であり、Tensorのsubclassとして学習対象parameterとしてModuleへ登録されている。

### shape

```python
conv.weight.shape
```

は値そのものではなくshape情報である。

```text
conv.weight
→ weight値

conv.weight.shape
→ (64, 32, 3, 3) というshape情報
```

である。

`flatten()` / `reshape()` を行う対象はTensorであり、shapeそのものではない。

```python
# 正しい
W_mat = conv.weight.detach().flatten(start_dim=1)

# shapeだけをflattenしてweight行列にするわけではない
shape = conv.weight.shape
```

### objectの型・dtype・shape・deviceを別々に確認する

「Tensorの型」という表現では、Python objectの型と内部scalarの型を分ける必要がある。

```python
import torch
from torch import nn

# データ取得なしで、Conv weightの型と配置を確認する。
conv = nn.Conv2d(32, 64, kernel_size=3, dtype=torch.float32)
weight = conv.weight
weight_matrix: torch.Tensor = weight.detach().flatten(start_dim=1)

assert isinstance(weight, nn.Parameter)
assert isinstance(weight, torch.Tensor)
assert type(weight_matrix) is torch.Tensor
assert isinstance(weight.shape, torch.Size)
assert isinstance(weight.shape, tuple)
assert weight.ndim == 4 and weight.shape == (64, 32, 3, 3)
assert weight_matrix.ndim == 2 and weight_matrix.shape == (64, 288)
assert weight.dtype == weight_matrix.dtype == torch.float32
assert weight.device == weight_matrix.device == torch.device("cpu")

# 後で因子をkernelへ戻すため、shapeの整数の組を保持する。
original_shape = tuple(weight.shape)
out_ch, in_ch, k_h, k_w = original_shape
assert out_ch * in_ch * k_h * k_w == weight_matrix.numel() == 18_432
```

`type(weight)` はParameterのclass、`weight.dtype` はscalarの数値型である。典型的なdtypeにはfloat32、float16、bfloat16、分類labelのint64、maskのboolがあるが、いずれもobjectのclassとは別の情報である。default dtypeは設定で変わり、SVDに使用できるdtypeも演算とdeviceに依存する。

`weight_matrix: torch.Tensor = ...` は型ヒント付き代入であり、annotationだけで値や型を変換する操作ではない。型ヒントは実行時の型・ndim検査にもならない。

### DataFrameへ表示用に変換する場合も、元weightを守る

```python
import pandas as pd
import torch
from torch import nn

conv = nn.Conv2d(1, 2, kernel_size=2, dtype=torch.float64)
before = conv.weight.detach().clone()
weight_matrix = conv.weight.detach().flatten(start_dim=1)

# CPU Tensorと値を共有しない表示・加工用arrayを作る。
display_array = weight_matrix.cpu().numpy().copy()
weight_frame: pd.DataFrame = pd.DataFrame(display_array)
assert weight_frame.shape == (2, 4)

# 表を編集しても学習用Parameterへは伝わらない。
weight_frame.iloc[0, 0] = 999
torch.testing.assert_close(conv.weight, before)
```

`detach` はgrad追跡を外し、`cpu` は配置を合わせ、`numpy` は配列へ変換する。既にCPUにあるTensorでは `cpu()` だけで独立copyになるとは限らない。SVDやGPU演算のためにDataFrameへ変換する必要はなく、ここでは表示・表操作のためだけに変換している。

[Tensor.numpyの変換条件と共有storage](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)

---

## 22. `flatten(start_dim=1)` と手動loop

```python
W_mat = tensor.flatten(start_dim=1)
```

は、

```text
先頭軸を残す
残りを1軸へまとめる
```

操作である。

一般に、

```text
(N, d1, d2, ...)
→
(N, d1×d2×...)
```

となる。概念的には、各 `tensor[i]` をflattenして `torch.stack(rows, dim=0)` で積む操作と同じだが、実際にはloopして集める必要はない。Conv2d weightについて値・行順・shapeを確認する具体例は [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]] で扱う。

また、

```python
tensor.flatten()
```

と全軸をflattenすると、

```text
(64×32×3×3,)
=
(18432,)
```

という1次元Tensorになり、今回のConv-SVDで欲しい `(64,288)` とは異なる。

### 型ヒント

Tensorを受け取ってTensorを返す関数なら、

```python
def as_matrix(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return tensor.flatten(start_dim=1)
```

のように書ける。

Pandasの `DataFrame` は表形式の実験結果整理用であり、SVDへ渡すweightの型ではない。

```text
weight / SVD因子
→ torch.Tensor

rank sweepの結果表
→ pandas.DataFrame
```

と役割を分ける。

### 2階以上で使う一般形と、4階Conv weight専用の検査

先頭軸を残す行列化は、入力が2階以上であることを前提にする。

$$
(d_0,d_1,\ldots,d_k)
\longrightarrow
\left(d_0,\prod_{j=1}^{k}d_j\right),
\qquad k\ge1.
$$

例えば $(10,2,3,4,5)\to(10,120)$、$(8,100)\to(8,100)$ である。1階の $(100,)$ と0階scalarにはaxis 1がないため、この `start_dim=1` はエラーになる。引数なしの `flatten()` がscalarも1階へ変換できることとは別である。

```python
import torch


def flatten_keep_first_dim_checked(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """2階以上を受け、先頭軸を残して残りの連続軸をまとめる。"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("shapeのtupleではなくTensorを渡してください。")
    if tensor.ndim < 2:
        raise ValueError("axis 1を持つ2階以上のTensorを渡してください。")
    return tensor.flatten(start_dim=1)


assert flatten_keep_first_dim_checked(
    torch.zeros(10, 2, 3, 4, 5)
).shape == (10, 120)
assert flatten_keep_first_dim_checked(
    torch.zeros(8, 100)
).shape == (8, 100)
```

Conv2d weight専用の関数なら、さらに `weight.ndim != 4` を検査する。`torch.Tensor` という型ヒントだけでは軸数を保証しない。一度だけ行列化するなら1行の `flatten` で足り、繰り返す処理や検査を名前にしたい場合に関数化する。

[flattenの指定軸とview/copyの条件](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html)

---

## 23. PyTorchモデルを読むためのクラス記法

### `self` の意味

```python
# 現在のモデル自身にfc1という子Moduleを登録する。
self.fc1 = nn.Linear(784, 512)
```

`self` は、現在操作しているモデルのインスタンス自身を表す。

```text
self.fc1
=
このモデルが持っているfc1
```

`self.fc1` として登録した層は `forward()` から利用できるだけでなく、`nn.Module` の子Moduleとして認識される。そのため、次の処理にも含まれる。

```python
model.parameters()
model.state_dict()
model.to(device)
model.train()
model.eval()
```

### `__init__()` の意味

```python
def __init__(self):
    # インスタンス作成時の初期化をここへ書く。
    ...
```

`__init__()` は、インスタンスを作るときに実行される初期化メソッドである。モデルではLinear層、活性化関数、Dropout、BatchNormなどの子Moduleを主に定義する。

### `super().__init__()` の意味

```python
class MNISTMLP(nn.Module):
    def __init__(self):
        # 親クラスnn.Moduleの初期化を実行する。
        super().__init__()
```

`super().__init__()` は、親クラスである `nn.Module` の初期化を呼ぶ。これにより、PyTorchが子ModuleやParameterを正しく登録できる。

### `model(images)` と `forward(images)`

```python
# nn.Moduleの呼出し機構を経由してforwardを実行する。
outputs = model(images)
```

概念上は `model.forward(images)` に近いが、通常は `model(images)` を使う。hookなどを含む `nn.Module` の呼出し機構が正しく働くためである。

---

## 24. 未使用値を受ける `_`

`_` はPythonの特別な破棄構文ではなく普通の変数名である。ただし、値を受け取る必要はあるが、その後使わない場合に `_` と書く慣習がある。値は実際には代入されるが、未使用という意図を示すため通常は参照しない。

### 複数の戻り値の一部を使わない

```python
# DataLoaderが返すラベルは、この処理では使わない。
images, _ = next(iter(data_loader))
```

### ループ番号を使わない

```python
# warm-up回数だけ繰り返し、ループ番号は使わない。
for _ in range(warmup):
    model(images)
```

### 戻り値を使わない

```python
# forwardは実行するが、出力値自体は使わない。
_ = model(images)
```

### 一部の戻り値だけを使う

```python
# 3つの戻り値のうち、先頭だけを使わない。
_, compressed_macs, compute_reduction = MACs(
    model,
    fc1_rank,
    fc2_rank,
    verbose=False,
)
```

単独の `_` と、Pythonの特殊メソッド名に含まれる二重アンダースコアは別の意味である。

```text
_          → 慣習的な未使用変数名
__init__   → Pythonの特殊メソッド名
```

---

## 25. `torch.round()` と表示桁数

```python
import torch

x = torch.tensor([1.234, 1.235, 1.236])
rounded = torch.round(x, decimals=2)
```

`torch.round()` は、ちょうど中間の値で偶数側へ丸める方式を採用する。

```text
2.5 → 2
3.5 → 4
```

常に5を切り上げる意味での四捨五入とは異なる。実験結果を表示するだけなら、計算に使うTensorを丸めず、f-stringで表示桁数だけを指定する。

```python
# 計算値を変えず、表示だけを整える。
print(f"{accuracy * 100:.2f}%")
print(f"{latency_ms:.3f} ms/batch")
```

## 26. rank sweepの直積を、meshgridの全要素と対応させる

複数のrank候補リストから全組合せを作る操作は直積であり、二つのリストを一対一に対応させる `zip` とは異なる。小さい候補

$$
r_1\in\{8,16\},
\qquad
r_2\in\{4,12,20\}
$$

では、全組合せは $2\cdot3=6$ 通りである。

$$
\mathcal R
=\{(8,4),(8,12),(8,20),(16,4),(16,12),(16,20)\}.
$$

`indexing="ij"` のmeshgridを全要素で書くと、

$$
R_1
=\begin{pmatrix}8&8&8\\16&16&16\end{pmatrix},
\qquad
R_2
=\begin{pmatrix}4&12&20\\4&12&20\end{pmatrix}.
$$

row-major順に `ravel()` して2列へ並べると、

$$
\begin{pmatrix}
8&4\\
8&12\\
8&20\\
16&4\\
16&12\\
16&20
\end{pmatrix}
$$

となり、`product(r1_list, r2_list)` と同じ順序になる。`indexing="xy"` では最初の二つの軸が入れ替わり、同じ組合せ集合でもflatten後の順序が変わる。

```python
from itertools import product

import numpy as np
import pandas as pd

# rank pairを生成するだけで、学習や評価は行わない。
r1_list = [8, 16]
r2_list = [4, 12, 20]
rank_pairs = list(product(r1_list, r2_list))

R1, R2 = np.meshgrid(r1_list, r2_list, indexing="ij")
pairs_from_grid = np.stack([R1.ravel(), R2.ravel()], axis=1)
assert pairs_from_grid.tolist() == [list(pair) for pair in rank_pairs]

rank_table = pd.DataFrame(rank_pairs, columns=["r1", "r2"])
assert len(rank_table) == 6

# 最終層をrank 10へ固定しても、各候補は三つのrankを持つ。
rank_triples = list(product(r1_list, r2_list, [10]))
assert len(rank_triples) == 6
assert rank_triples[0] == (8, 4, 10)
assert rank_triples[-1] == (16, 20, 10)
```

各層に4候補 `[16, 32, 64, 128]` があれば2層で16通り、最終層rankも1から10まで振るなら $4\cdot4\cdot10=160$ 通りとなる。候補の重複や各層の最大rank超過は事前に検査する。rank選択はvalidation、候補固定後の最終報告はtestで行い、全候補を同じbaseline checkpointから独立に作る。

[productの直積仕様](https://docs.python.org/3/library/itertools.html#itertools.product)

[meshgridのij/xy規約](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)

## 27. 全ペア比較を、比較件数と削除まで追う

`itertools.combinations(df.index, 2)` は自己ペアを作らず、$(i,j)$ と $(j,i)$ を二重に列挙しない。候補数 $N$ のペア数は

$$
\binom N2
=\frac{N(N-1)}2,
\qquad
\binom42=6.
$$

```python
from itertools import combinations

import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "parameters": [1_000_000, 500_000, 250_000, 400_000],
        "validation_loss": [0.300, 0.302, 0.310, 0.315],
    },
    index=[10, 20, 30, 40],
)
assert df.index.is_unique
assert np.isfinite(df[["parameters", "validation_loss"]].to_numpy()).all()


def dominates(a: pd.Series, b: pd.Series) -> bool:
    """全目的で悪化せず、少なくとも一つでstrictに改善するか。"""
    return (
        a["parameters"] <= b["parameters"]
        and a["validation_loss"] <= b["validation_loss"]
        and (
            a["parameters"] < b["parameters"]
            or a["validation_loss"] < b["validation_loss"]
        )
    )


pairs = list(combinations(df.index, 2))
assert len(pairs) == len(df) * (len(df) - 1) // 2 == 6

dominated_labels = set()
for label_a, label_b in pairs:
    a, b = df.loc[label_a], df.loc[label_b]
    if dominates(a, b):
        dominated_labels.add(label_b)
    elif dominates(b, a):
        dominated_labels.add(label_a)

# 比較中はdfを変更せず、全判定後に一度だけ落とす。
assert dominated_labels == {40}
pareto = df.drop(index=list(dominated_labels)).sort_values("parameters")
assert pareto.index.tolist() == [30, 20, 10]
```

indexが重複すると `df.loc[label]` が複数行を返す。欠損・無限大も事前に検査する。全ペア法は $O(N^2)$ なので、少数候補を明示的に追う用途に向く。

## 28. knee計算を、行位置とモデル取得までつなぐ

```python
import numpy as np
import pandas as pd

pareto = pd.DataFrame(
    {
        "parameters": [1000, 2000, 3000],
        "validation_loss": [0.8, 0.4, 0.35],
        "model": ["small_model", "middle_model", "large_model"],
    },
    index=[12, 4, 9],
).sort_values("parameters")

metrics = pareto[["parameters", "validation_loss"]].to_numpy(dtype=float)
lo, span = metrics.min(axis=0), np.ptp(metrics, axis=0)
if len(pareto) < 3 or np.any(span == 0):
    raise ValueError("距離kneeを検討できる候補数・変動幅が必要")

points = (metrics - lo) / span
start, end = points[0], points[-1]
line = end - start
offset = points - start
denominator = np.linalg.norm(line)
distances = np.abs(
    line[0] * offset[:, 1] - line[1] * offset[:, 0]
) / denominator
if np.allclose(distances, 0):
    raise ValueError("全点がほぼ一直線")

pareto["distance"] = distances
knee_pos = int(np.argmax(distances))  # 行labelではなく行位置。
knee_row = pareto.iloc[knee_pos]
knee_label = pareto.index[knee_pos]
knee_model = knee_row["model"]

assert knee_pos == 1 and knee_label == 4
assert knee_model == "middle_model"
assert pareto.loc[knee_label, "model"] == knee_model
```

`pareto.loc[knee_pos]` はlabel 1を探すため誤りである。距離最大が同率なら `argmax` は最初の位置を返す。kneeは座標・候補・端点に依存する経験則である。

## 29. scalar条件分岐とSeries条件を区別する

scalar比較を組み合わせる `if` では `and / or` を使う。一方、列全体のSeries条件は `& / |` と括弧を使い、最後に `any()` / `all()` で一つの真偽値へ集約する。二つの独立した `if` は両方実行され得るが、`if` / `elif` / `else` は最初に成立した枝だけを実行する。

### Python組込みのany/allを、全条件と一つの改善へ対応させる

Python組込みの `any(iterable)` と `all(iterable)` はscalarの比較結果をまとめられる。

```python
all_not_worse = [100 <= 200, 0.30 <= 0.30]
one_strictly_better = [100 < 200, 0.30 < 0.30]

assert all(all_not_worse) and any(one_strictly_better)
assert any([False, True, False])
assert not all([False, True, False])
assert all([]) and not any([])
```

## 30. 平方根・vector norm・vector化を実際の値で区別する

$$
\sqrt{(1,4,9)}=(1,2,3),
\qquad
\|(3,4)\|_2
=\sqrt{3^2+4^2}
=5.
$$

```python
import numpy as np
import torch

np.testing.assert_allclose(np.sqrt([1.0, 4.0, 9.0]), [1, 2, 3])
torch.testing.assert_close(
    torch.sqrt(torch.tensor([1.0, 4.0, 9.0])),
    torch.tensor([1.0, 2.0, 3.0]),
)
assert np.linalg.norm([3.0, 4.0]) == 5
assert torch.linalg.vector_norm(torch.tensor([3.0, 4.0])).item() == 5
assert torch.isnan(torch.sqrt(torch.tensor(-1.0))).item()

x = np.arange(5, dtype=float)
loop_result = np.array([2 * value for value in x])
np.testing.assert_array_equal(loop_result, 2 * x)
```

平方根は各要素への演算、normはvector全体の二乗和の平方根である。Python loopとNumPy/PyTorchの一括演算は、同じ数式を異なる実行方法で計算している。

## 31. 列追加・行追加を、ラベル対応まで確認する

```python
import pandas as pd

df = pd.DataFrame(
    {"rank": [8, 16], "parameters": [1000, 2000]},
    index=[0, 2],
)
df["is_knee"] = False
df["distance"] = [0.1, 0.3]  # 現在の行位置順。
df["aligned"] = pd.Series([20, 10], index=[2, 0])  # labelでalign。
df.insert(0, "candidate", ["small", "large"])

assert df["aligned"].tolist() == [10, 20]
assert df.columns[0] == "candidate"

# label 2は既存なので、df.loc[len(df)]は新規追加にならない。
assert len(df) in df.index
new_row = {
    "candidate": "extra",
    "rank": 32,
    "parameters": 3000,
    "is_knee": False,
    "distance": 0.0,
    "aligned": 30,
}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
assert df["rank"].tolist() == [8, 16, 32]
assert df.index.tolist() == [0, 1, 2]
```

Series代入はindex labelでalignする。新しい未使用labelへの `.loc` は追加、既存labelなら更新である。多数の行を追加するときは `list[dict]` を作って最後にDataFrame化する。

## 32. 重み辞書をobject列へ保持する

DataFrameへ格納しただけでは、辞書内のTensorまでdeep copyされない。保存時点で独立snapshotを作る。

```python
import copy

import pandas as pd
import torch

model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.float64))

df = pd.DataFrame({"fc1_rank": [16], "fc2_rank": [32]}, index=[7])
df["before_ft_state"] = pd.Series([None], index=df.index, dtype=object)
df.at[7, "before_ft_state"] = copy.deepcopy(model.state_dict())

with torch.no_grad():
    model.weight.zero_()

torch.testing.assert_close(
    df.at[7, "before_ft_state"]["weight"],
    torch.tensor([[2.0, 3.0]], dtype=torch.float64),
)

index = pd.MultiIndex.from_tuples(
    [(16, 32)], names=["fc1_rank", "fc2_rank"]
)
by_rank = pd.DataFrame(index=index)
by_rank["before_ft_state"] = pd.Series([None], index=index, dtype=object)
by_rank.at[(16, 32), "before_ft_state"] = copy.deepcopy(
    df.at[7, "before_ft_state"]
)
assert isinstance(by_rank.at[(16, 32), "before_ft_state"], dict)
```

`.at` の座標は「行label、列名」の二つである。object列をnumeric計算へ混ぜず、永続保存ではweightを別ファイルへ保存してCSVにはpathを置く。

## 33. Python・NumPy・Pandas・PyTorchを処理の役割で使い分ける

どれが常に速いかではなく、処理の種類、配列の大きさ、device転送、自動微分の要否で選ぶ。

- Python：`if`、`for`、関数、`list`、`dict`による制御。
- Pandas：rank候補、parameter数、validation指標の表形式での整理・抽出・CSV保存。
- NumPy：CPU上の配列演算、正規化、Pareto／kneeの幾何計算。
- PyTorch：NNのforward、loss、backward、optimizer、SVD、GPU Tensor演算。
- Matplotlib：Pandas／NumPyで整理した結果の可視化。

候補数が数十から数千程度のPareto／knee計算では、PandasとNumPyが自然である。小さい配列をGPUへ移すと、転送とkernel起動のoverheadが演算時間を上回ることがある。反対に、大きな行列積やNN学習ではPyTorchとGPUの並列処理を利用しやすい。Python loopで要素を一つずつ更新せず、可能ならNumPyまたはPyTorchの一括演算を使う。

PyTorch TensorをNumPyへ渡すときは、計算graphから切り離し、CPUへ移してから変換する。逆にNumPy配列をPyTorch Tensorへ変換するときは、必要なdtypeとdeviceを明示する。

```python
import torch

layer = torch.nn.Linear(4, 3)

# 学習用Parameterを変更しない、集計・表示用のNumPy配列を作る。
weight_np = layer.weight.detach().cpu().numpy().copy()

# NumPy配列をTensorへ戻し、元の層と同じdtype・deviceへ置く。
weight_tensor = torch.from_numpy(weight_np).to(
    device=layer.weight.device,
    dtype=layer.weight.dtype,
)

torch.testing.assert_close(weight_tensor, layer.weight.detach())
```

速度比較ではwarm-up、反復回数、入力shape、dtype、deviceを固定する。CUDA演算は非同期なので、計測区間の前後で同期が必要である。具体的な計測コードは [[05_SVD基礎実装検証/11_ベンチマークのPyTorchコード]] を参照する。

## 34. Cursorでインデント解除と置換を行う

複数行のインデントを一段戻すときは、対象行を選択して `Shift + Tab` を使う。右へ一段進める操作は `Tab` である。これはコードの文字列を変更する編集操作であり、Pythonの実行時処理ではない。

現在のファイル内を置換するショートカットは、Windows／Linuxでは `Ctrl + H`、macOSでは `Cmd + Option + F` である。プロジェクト全体を検索・置換する場合は、Windows／Linuxでは `Ctrl + Shift + H`、macOSでは `Cmd + Shift + H` を使う。全置換の前に検索結果と対象ファイルを確認し、変数名の一部やMarkdownリンクまで意図せず変更しないようにする。

---

## 関連ノート

- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[05_SVD基礎実装検証/12_PyTorch学習と評価の基礎]]
- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[20_FashionMNIST/02_Fashion-MNISTのRank選択]]
- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
