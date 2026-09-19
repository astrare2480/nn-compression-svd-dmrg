# rank指標のPython集計と可視化

数式と考え方は [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## 51. 圧縮指標を計算するPython関数

この節までに導出した式は、実装では `compression_stats` にまとめられる。関数は

```text
in_features, out_features, rank
→ 元の重みパラメータ数
→ 低ランク重みパラメータ数
→ 保持率・削減率・圧縮倍率
→ 損益分岐rank・圧縮可能な最大整数rank
```

を一度に計算する。

完成した `CompressionStats` と `compression_stats` の実装は [[05_SVD基礎実装検証/07_PyTorch実装#6. `compression_stats`]] を正本とする。ここでは、各出力が本ノートのどの式に対応するかを確認する。

---

## 52. 使用例

```python
# 実装の正本は07_PyTorch実装を参照する。
stats = compression_stats(
    in_features=784,
    out_features=512,
    rank=64,
)

print(stats)
```

個別に表示する。

```python
print(
    "original:",
    stats.original_weight_parameters,
)

print(
    "low rank:",
    stats.low_rank_weight_parameters,
)

print(
    "keep ratio:",
    stats.parameter_keep_ratio,
)

print(
    "reduction ratio:",
    stats.parameter_reduction_ratio,
)

print(
    "compression factor:",
    stats.compression_factor,
)

print(
    "maximum compressing rank:",
    stats.maximum_compressing_rank,
)
```

---

### 重みだけの指標とbias込みのfield名を区別する

上の個別表示コードと、[[07_PyTorch実装#3. 実装全体]] の `CompressionStats` ではfield名が異なる。後者は重みだけの数とbias込みの総数を区別するため、次の対応で読む。

- `low_rank_weight_parameters` に対応するのは `compressed_weight_parameters`。
- `parameter_keep_ratio` に対応するのは `weight_keep_ratio`。
- `parameter_reduction_ratio` に対応するのは `weight_reduction_ratio`。
- `compression_factor` に対応するのは `weight_compression_factor`。
- `original_weight_parameters` と `maximum_compressing_rank` は同名。
- bias込みの比率は `total_keep_ratio`、`total_reduction_ratio`、`total_compression_factor`。

上の関数は重みだけを数えるので、`parameter_keep_ratio` を `total_keep_ratio` と対応付けてはいけない。`CompressionStats` を使う場合は、上の表示ブロックではなく、次を使う。後続の表・グラフも、使用するデータクラスのfield名へ合わせる。

```python
# 07の実装全体にあるcompression_statsを定義済みとして使用する。
stats = compression_stats(in_features=784, out_features=512, rank=64)
print("original:", stats.original_weight_parameters)
print("low rank:", stats.compressed_weight_parameters)
print("keep ratio:", stats.weight_keep_ratio)
print("reduction ratio:", stats.weight_reduction_ratio)
print("compression factor:", stats.weight_compression_factor)
print("maximum compressing rank:", stats.maximum_compressing_rank)
```

上の次元・rankを代入すると、

$$
\begin{aligned}
P_{\mathrm{original,weight}}
&=784\cdot512=401408,\\
P_{\mathrm{lowrank,weight}}
&=64(784+512)=64\cdot1296=82944,\\
q_{\mathrm{weight}}
&=\frac{82944}{401408}
=\frac{81}{392}
\approx0.206633,\\
1-q_{\mathrm{weight}}
&=\frac{311}{392}
\approx0.793367,\\
c_{\mathrm{weight}}
&=\frac{401408}{82944}
=\frac{392}{81}
\approx4.839506,\\
r_{\mathrm{break}}
&=\frac{401408}{1296}
=\frac{25088}{81}
\approx309.728395,\\
r_{\mathrm{max,compress}}
&=\left\lceil\frac{25088}{81}\right\rceil-1\\
&=310-1=309
\end{aligned}
$$

となる。元biasを後段に残す場合は両方へ $512$ を加え、$P_{\mathrm{original,total}}=401920$、$P_{\mathrm{lowrank,total}}=83456$ となる。重み保持率と総数保持率 $83456/401920$ は別であり、biasを含めるかをそろえて比較する。

## 53. bias込みのパラメータ数

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TotalParameterStats:
    original_total: int
    low_rank_total: int
    keep_ratio: float
    reduction_ratio: float
    compression_factor: float


def total_parameter_stats(
    in_features: int,
    out_features: int,
    rank: int,
    bias: bool,
) -> TotalParameterStats:
    original_weight = (
        in_features
        * out_features
    )

    low_rank_weight = (
        rank
        * (
            in_features
            +
            out_features
        )
    )

    bias_parameters = (
        out_features
        if bias
        else 0
    )

    original_total = (
        original_weight
        +
        bias_parameters
    )

    low_rank_total = (
        low_rank_weight
        +
        bias_parameters
    )

    keep_ratio = (
        low_rank_total
        /
        original_total
    )

    return TotalParameterStats(
        original_total=original_total,
        low_rank_total=low_rank_total,
        keep_ratio=keep_ratio,
        reduction_ratio=1.0 - keep_ratio,
        compression_factor=(
            original_total
            / low_rank_total
        ),
    )
```

biasは大きなLinear層では割合が小さいが、小さい層では無視できない場合がある。

---

## 54. rank候補表を作る

```python
from __future__ import annotations


def build_rank_table(
    in_features: int,
    out_features: int,
    ranks: list[int],
) -> list[dict[str, float | int | bool]]:
    rows: list[
        dict[str, float | int | bool]
    ] = []

    for rank in ranks:
        stats = compression_stats(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
        )

        rows.append(
            {
                "rank": rank,
                "original_parameters": (
                    stats.original_weight_parameters
                ),
                "low_rank_parameters": (
                    stats.low_rank_weight_parameters
                ),
                "keep_ratio": (
                    stats.parameter_keep_ratio
                ),
                "reduction_ratio": (
                    stats.parameter_reduction_ratio
                ),
                "compression_factor": (
                    stats.compression_factor
                ),
                "is_compression": (
                    stats.low_rank_weight_parameters
                    <
                    stats.original_weight_parameters
                ),
            }
        )

    return rows
```

### 使用例

```python
ranks = [
    8,
    16,
    32,
    64,
    128,
    256,
    309,
    310,
]

rows = build_rank_table(
    in_features=784,
    out_features=512,
    ranks=ranks,
)

for row in rows:
    print(row)
```

---

## 55. pandasで表にする

```python
import pandas as pd

table = pd.DataFrame(
    rows
)

table["keep_percent"] = (
    100
    * table["keep_ratio"]
)

table["reduction_percent"] = (
    100
    * table["reduction_ratio"]
)

print(table)
```

CSVへ保存する。

```python
table.to_csv(
    "rank_compression_table.csv",
    index=False,
)
```

実験結果と同じrank表を使うと、圧縮率と精度を結び付けて管理しやすい。

---

## 56. rankとパラメータ数を可視化する

```python
import matplotlib.pyplot as plt

ranks = list(
    range(
        1,
        min(784, 512) + 1,
    )
)

low_rank_parameters = [
    rank
    * (
        784
        +
        512
    )
    for rank in ranks
]

original_parameters = (
    784
    *
    512
)

plt.figure()
plt.plot(
    ranks,
    low_rank_parameters,
    label="Low-rank parameters",
)
plt.axhline(
    original_parameters,
    linestyle="--",
    label="Original parameters",
)
plt.xlabel("Rank")
plt.ylabel("Weight parameter count")
plt.title("Rank and Parameter Count")
plt.grid(True)
plt.legend()
plt.show()
```

元パラメータ数との交点付近が損益分岐rankである。

---

## 57. rankと保持率を可視化する

```python
import matplotlib.pyplot as plt

keep_ratios = [
    (
        rank
        * (
            784
            +
            512
        )
        /
        (
            784
            *
            512
        )
    )
    for rank in ranks
]

plt.figure()
plt.plot(
    ranks,
    keep_ratios,
)
plt.axhline(
    1.0,
    linestyle="--",
)
plt.xlabel("Rank")
plt.ylabel("Parameter keep ratio")
plt.title("Rank and Parameter Keep Ratio")
plt.grid(True)
plt.show()
```

保持率が1未満ならパラメータ数は減っている。

---

## 59. rankと精度のグラフ

```python
import matplotlib.pyplot as plt

ranks = [
    8,
    16,
    32,
    64,
    128,
    256,
]

accuracies = [
    # 実験結果を入力する
]

plt.figure()
plt.plot(
    ranks,
    accuracies,
    marker="o",
)
plt.xlabel("Rank")
plt.ylabel("Test accuracy")
plt.title("Rank and Test Accuracy")
plt.grid(True)
plt.show()
```

圧縮率を横軸にする方法もある。

```python
keep_ratios = [
    compression_stats(
        784,
        512,
        rank,
    ).parameter_keep_ratio
    for rank in ranks
]
```

---

## 60. パラメータ数と精度のグラフ

rank自体より、実際のパラメータ数を横軸にすると、異なる層や異なる手法と比較しやすい。

```python
parameter_counts = [
    compression_stats(
        784,
        512,
        rank,
    ).low_rank_weight_parameters
    for rank in ranks
]

plt.figure()
plt.plot(
    parameter_counts,
    accuracies,
    marker="o",
)
plt.xlabel("Weight parameter count")
plt.ylabel("Test accuracy")
plt.title("Parameter Count and Accuracy")
plt.grid(True)
plt.show()
```

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（3. 元のLinear層のパラメータ数）

```python
nn.Linear(
    in_features=D_in,
    out_features=D_out,
    bias=True,
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（4. 低ランク2層のパラメータ数）

```python
nn.Linear(
    D_in,
    rank,
    bias=False,
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（4. 低ランク2層のパラメータ数）

```python
nn.Linear(
    rank,
    D_out,
    bias=True,
)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（14. `Linear(784, 512)` の元パラメータ数）

```python
nn.Linear(
    in_features=784,
    out_features=512,
)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（18. 出力層 `Linear(512, 10)` の例）

```python
nn.Linear(
    in_features=512,
    out_features=10,
)
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（21. 小さいLinear層は圧縮効果が限定的）

```python
nn.Linear(
    16,
    8,
)
```

## PyTorch確認-007

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（36. 実測時間の比較条件）

```python
torch.cuda.synchronize()
```

## PyTorch確認-008

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（SD許容幅を、全候補の数値まで計算する）

```python
import numpy as np
import pandas as pd

# rank 16の3値から標本SDまで検算する。
losses = np.array([0.41, 0.39, 0.40])
np.testing.assert_allclose(losses.mean(), 0.40)
np.testing.assert_allclose(losses.std(ddof=1), 0.01)

# 他の2候補は説明用の要約値であり、実際のrun履歴ではない。
summary = pd.DataFrame({"rank": [8, 16, 32],
                        "mean_loss": [0.42, 0.40, 0.39],
                        "sample_sd": [0.015, 0.010, 0.009]})
best_pos = summary["mean_loss"].to_numpy().argmin()
best = summary.iloc[best_pos]
summary["candidate_sd_allowed"] = (
    summary["mean_loss"] <= best["mean_loss"] + summary["sample_sd"])
summary["max_sd_allowed"] = (
    summary["mean_loss"] <= best["mean_loss"] + summary["sample_sd"].max())
# 3 runという追加仮定の下でのみSEを計算する。
threshold = best["mean_loss"] + best["sample_sd"] / np.sqrt(3)
summary["one_se_allowed"] = summary["mean_loss"] <= threshold
assert summary.loc[summary["candidate_sd_allowed"], "rank"].tolist() == [16, 32]
assert summary.loc[summary["max_sd_allowed"], "rank"].tolist() == [16, 32]
assert summary.loc[summary["one_se_allowed"], "rank"].tolist() == [32]
```

## PyTorch確認-009

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]（62. 実験でのrank候補）

```python
nn.Linear(
    784,
    512,
)
```

