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

関連：[[00_基礎理論/05_圧縮率とRank]]、[[00_基礎理論/10_SVD圧縮モデルの評価設計]]、[[00_基礎理論/12_PyTorch学習と評価の基礎]]

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

この判定の数学的意味は [[00_基礎理論/05_圧縮率とRank]] を参照。

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

詳細：[[00_基礎理論/05_圧縮率とRank]]

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

詳細：[[00_基礎理論/10_SVD圧縮モデルの評価設計]]、[[00_基礎理論/12_PyTorch学習と評価の基礎]]

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

## 関連ノート

- [[00_基礎理論/05_圧縮率とRank]]
- [[00_基礎理論/06_誤差評価]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]
- [[00_基礎理論/12_PyTorch学習と評価の基礎]]
- [[20_FASHION_MNIST_MLP_SVD/02_Fashion-MNISTのRank選択]]
- [[20_FASHION_MNIST_MLP_SVD/03_Fashion-MNISTのFine-tuning]]
