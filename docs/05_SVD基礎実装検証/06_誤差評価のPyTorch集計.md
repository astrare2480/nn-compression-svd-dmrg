# 誤差評価のPyTorch集計

数式と考え方は [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## 48. 低ランク2層から重みを再構成する

圧縮後の前段重みを $A$、後段重みを $B$ とする。

再構成重みは、

$$
W_r=BA
$$

である。

PyTorchでは、

```python
with torch.no_grad():
    reconstructed_weight = (
        compressed.second.weight
        @
        compressed.first.weight
    )
```

と計算する。

その後、

```python
weight_metrics = compare_weight_matrices(
    original=original.weight,
    approximation=reconstructed_weight,
)
```

で評価できる。

---

## 49. 層出力を比較する関数

実装では、比較する2層を同じ評価状態にし、同一入力から出力を得て `compare_tensors` へ渡す。

```python
original_layer.eval()
compressed_layer.eval()

with torch.inference_mode():
    original_output = original_layer(inputs)
    compressed_output = compressed_layer(inputs)

output_metrics = compare_tensors(
    reference=original_output,
    approximation=compressed_output,
)
```

`compare_tensors` の完成実装は [[05_SVD基礎実装検証/07_PyTorch実装#15. rankを下げた出力誤差]] を参照する。ここでは、圧縮前後で入力と評価条件をそろえることが要点である。

---

## 50. ReLU前後を同時に比較する

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActivationComparison:
    pre_activation: TensorErrorMetrics
    post_activation: TensorErrorMetrics
    relu_mask_match_ratio: float


def compare_with_relu(
    original_layer: nn.Module,
    compressed_layer: nn.Module,
    inputs: torch.Tensor,
) -> ActivationComparison:
    original_layer.eval()
    compressed_layer.eval()

    with torch.inference_mode():
        z_original = original_layer(
            inputs
        )

        z_compressed = compressed_layer(
            inputs
        )

        h_original = torch.relu(
            z_original
        )

        h_compressed = torch.relu(
            z_compressed
        )

        mask_match_ratio = (
            (
                (z_original > 0)
                ==
                (z_compressed > 0)
            )
            .float()
            .mean()
        )

    return ActivationComparison(
        pre_activation=compare_tensors(
            z_original,
            z_compressed,
        ),
        post_activation=compare_tensors(
            h_original,
            h_compressed,
        ),
        relu_mask_match_ratio=float(
            mask_match_ratio.item()
        ),
    )
```

---

## 53. ストリーミング誤差集計器

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ErrorAccumulator:
    element_count: int = 0
    signed_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    reference_squared_sum: float = 0.0
    maximum_absolute_error: float = 0.0

    def update(
        self,
        reference: torch.Tensor,
        approximation: torch.Tensor,
    ) -> None:
        if reference.shape != approximation.shape:
            raise ValueError(
                "tensor shapes must match."
            )

        difference = (
            reference
            -
            approximation
        )

        self.element_count += (
            difference.numel()
        )

        self.signed_error_sum += float(
            difference.sum().item()
        )

        self.absolute_error_sum += float(
            difference
            .abs()
            .sum()
            .item()
        )

        self.squared_error_sum += float(
            difference
            .square()
            .sum()
            .item()
        )

        self.reference_squared_sum += float(
            reference
            .square()
            .sum()
            .item()
        )

        batch_maximum = float(
            difference
            .abs()
            .max()
            .item()
        )

        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            batch_maximum,
        )

    def compute(
        self,
    ) -> TensorErrorMetrics:
        if self.element_count == 0:
            raise ValueError(
                "no elements were accumulated."
            )

        mean_error = (
            self.signed_error_sum
            /
            self.element_count
        )

        mae = (
            self.absolute_error_sum
            /
            self.element_count
        )

        mse = (
            self.squared_error_sum
            /
            self.element_count
        )

        rmse = mse ** 0.5

        reference_rms = (
            self.reference_squared_sum
            /
            self.element_count
        ) ** 0.5

        relative_rmse = (
            None
            if reference_rms == 0.0
            else rmse / reference_rms
        )

        return TensorErrorMetrics(
            element_count=self.element_count,
            mean_error=mean_error,
            mae=mae,
            mse=mse,
            rmse=rmse,
            maximum_absolute_error=(
                self.maximum_absolute_error
            ),
            reference_rms=reference_rms,
            relative_rmse=relative_rmse,
            cosine_similarity_mean=None,
        )
```

この集計器では、コサイン類似度は積算していない。

必要なら、サンプル単位の総和と件数を別途保持する。

---

## 54. DataLoader全体で層出力を比較する

DataLoaderが画像とラベルを返すとする。

対象層の入力を直接作れる場合の例を示す。

```python
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_mnist_first_layer_error(
    original_layer: nn.Module,
    compressed_layer: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> TensorErrorMetrics:
    original_layer.eval()
    compressed_layer.eval()

    accumulator = ErrorAccumulator()

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                device
            )

            inputs = images.flatten(
                start_dim=1
            )

            original_output = (
                original_layer(
                    inputs
                )
            )

            compressed_output = (
                compressed_layer(
                    inputs
                )
            )

            accumulator.update(
                reference=original_output,
                approximation=compressed_output,
            )

    return accumulator.compute()
```

MNISTの第1Linear層では、画像をFlattenしたテンソルが層入力になる。

---

## 55. モデル全体の評価結果

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    average_loss: float
    accuracy: float
```

---

## 56. モデルのlossとaccuracyを計算する

```python
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> ClassificationMetrics:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                inputs
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = (
                labels.shape[0]
            )

            total_loss += (
                float(
                    loss.item()
                )
                *
                batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            total_correct += int(
                (
                    predictions
                    ==
                    labels
                )
                .sum()
                .item()
            )

            total_samples += (
                batch_size
            )

    if total_samples == 0:
        raise ValueError(
            "loader produced no samples."
        )

    return ClassificationMetrics(
        sample_count=total_samples,
        average_loss=(
            total_loss
            /
            total_samples
        ),
        accuracy=(
            total_correct
            /
            total_samples
        ),
    )
```

`CrossEntropyLoss` の既定 `reduction="mean"` を想定し、バッチlossへバッチサイズを掛けてから全サンプル数で割る。

---

## 57. 圧縮前後のlogitsを比較する

```python
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ModelComparisonMetrics:
    logits_error: TensorErrorMetrics
    prediction_agreement: float


def compare_model_outputs(
    original_model: nn.Module,
    compressed_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ModelComparisonMetrics:
    original_model.eval()
    compressed_model.eval()

    accumulator = ErrorAccumulator()

    agreement_count = 0
    sample_count = 0

    with torch.inference_mode():
        for inputs, _ in loader:
            inputs = inputs.to(
                device
            )

            original_logits = (
                original_model(
                    inputs
                )
            )

            compressed_logits = (
                compressed_model(
                    inputs
                )
            )

            accumulator.update(
                original_logits,
                compressed_logits,
            )

            original_prediction = (
                original_logits.argmax(
                    dim=1
                )
            )

            compressed_prediction = (
                compressed_logits.argmax(
                    dim=1
                )
            )

            agreement_count += int(
                (
                    original_prediction
                    ==
                    compressed_prediction
                )
                .sum()
                .item()
            )

            sample_count += (
                inputs.shape[0]
            )

    if sample_count == 0:
        raise ValueError(
            "loader produced no samples."
        )

    return ModelComparisonMetrics(
        logits_error=accumulator.compute(),
        prediction_agreement=(
            agreement_count
            /
            sample_count
        ),
    )
```

---

## 60. `eval()` を使う

DropoutやBatchNormを含むモデルでは、学習モードと評価モードで出力が変わる。

圧縮前後の比較では、

```python
model.eval()
```

を使う。

LinearとReLUだけの単純MLPでも、評価コードの習慣として統一する。

---

## 61. `torch.inference_mode()` を使う

評価時は勾配が不要である。

```python
with torch.inference_mode():
    output = model(inputs)
```

とすると、

- 勾配計算を行わない
- メモリ使用量を減らせる
- 評価を高速化しやすい

`torch.no_grad()` でもよいが、純粋な推論では `torch.inference_mode()` が使える。

---

## 63. ランダム性を抑える

再現性を高めるため、必要に応じて乱数seedを固定する。

```python
import random
import numpy as np
import torch

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        seed
    )
```

ただし、GPU演算には完全な決定性を保証できない処理もある。

評価値の差が非常に小さい場合は、複数回測定も検討する。

---

## 68. 誤差とaccuracyの関係を可視化する

rankごとの結果を、

```text
相対重み誤差 → accuracy
層出力RMSE   → accuracy
logits RMSE  → accuracy
```

として散布図にできる。

```python
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(
    relative_weight_errors,
    accuracies,
)
plt.xlabel("Relative weight error")
plt.ylabel("Test accuracy")
plt.title("Weight Error and Accuracy")
plt.grid(True)
plt.show()
```

重み誤差とタスク性能がどの程度対応するかを観察する。

---

## 69. rankと複数誤差を可視化する

尺度が異なるため、同じ縦軸へ無理に重ねず、個別のグラフを作る方が読みやすい。

```python
plt.figure()
plt.plot(
    ranks,
    output_rmses,
    marker="o",
)
plt.xlabel("Rank")
plt.ylabel("Layer output RMSE")
plt.title("Rank and Layer Output Error")
plt.grid(True)
plt.show()
```

```python
plt.figure()
plt.plot(
    ranks,
    accuracies,
    marker="o",
)
plt.xlabel("Rank")
plt.ylabel("Test accuracy")
plt.title("Rank and Accuracy")
plt.grid(True)
plt.show()
```

---

## 70. サンプルごとの誤差を見る

全体平均だけでなく、各サンプルのRMSEを計算できる。

出力次元を $D$ とすると、サンプル $n$ のRMSEは、

$$
\operatorname{RMSE}_n
=
\sqrt{
\frac{1}{D}
\sum_{j=1}^{D}
\left(
Y_{nj}
-
\hat{Y}_{nj}
\right)^2
}
$$

PyTorchでは、

```python
per_sample_rmse = torch.sqrt(
    (
        original_output
        -
        compressed_output
    )
    .square()
    .mean(
        dim=1
    )
)
```

と計算できる。

誤差の大きいサンプルを画像として確認すると、圧縮が苦手なパターンを分析できる。

---

## 71. 出力ユニットごとの誤差

出力次元 $j$ ごとのMSEは、

$$
\operatorname{MSE}_j
=
\frac{1}{N}
\sum_{n=1}^{N}
E_{nj}^2
$$

である。

PyTorchでは、

```python
per_feature_mse = (
    difference
    .square()
    .mean(
        dim=0
    )
)
```

と計算する。

特定ユニットだけ誤差が大きいかを確認できる。

---

## 72. 分布を確認する

平均値だけでなく、次も確認できる。

- 中央値
- 90パーセンタイル
- 95パーセンタイル
- 99パーセンタイル
- ヒストグラム

```python
absolute_error = (
    difference
    .abs()
    .flatten()
)

quantiles = torch.quantile(
    absolute_error,
    torch.tensor(
        [0.5, 0.9, 0.95, 0.99],
        device=absolute_error.device,
    ),
)
```

大部分の誤差は小さいが、一部だけ大きい場合を検出しやすい。

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（1. まず差分を定義する）

```python
difference = (
    y_original
    -
    y_approx
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（3. MAE）

```python
mae = torch.mean(
    difference.abs()
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（4. MSE）

```python
mse = torch.mean(
    difference ** 2
)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（4. MSE）

```python
mse = torch.mean(
    difference.square()
)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（6. RMSE）

```python
rmse = torch.sqrt(mse)
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（6. RMSE）

```python
rmse = torch.sqrt(
    torch.mean(
        difference.square()
    )
)
```

## PyTorch確認-007

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（7. 最大絶対誤差）

```python
maximum_absolute_error = (
    difference
    .abs()
    .max()
)
```

## PyTorch確認-008

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（8. 平均誤差）

```python
mean_error = difference.mean()
```

## PyTorch確認-009

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（9. ベクトル・行列に対するMSE）

```python
torch.mean(
    (Y - Y_hat).square()
)
```

## PyTorch確認-010

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（10. フロベニウスノルム）

```python
frobenius_error = (
    torch.linalg.matrix_norm(
        error_matrix,
        ord="fro",
    )
)
```

## PyTorch確認-011

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（10. フロベニウスノルム）

```python
frobenius_like_error = (
    error_tensor
    .square()
    .sum()
    .sqrt()
)
```

## PyTorch確認-012

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（12. 相対誤差）

```python
relative_error = (
    torch.linalg.matrix_norm(
        W - W_rank,
        ord="fro",
    )
    /
    torch.linalg.matrix_norm(
        W,
        ord="fro",
    )
)
```

## PyTorch確認-013

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（15. コサイン類似度）

```python
cosine_similarity = (
    torch.nn.functional.cosine_similarity(
        y_original,
        y_approx,
        dim=-1,
    )
)
```

## PyTorch確認-014

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（上界だけでなく、上界を達成する入力を示す）

```python
spectral_error = (
    torch.linalg.matrix_norm(
        W - W_rank,
        ord=2,
    )
)
```

## PyTorch確認-015

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（20. 理論誤差と実測誤差を一致させる）

```python
import torch

torch.manual_seed(0)

W = torch.randn(
    8,
    5,
    dtype=torch.float64,
)

rank = 2

U, S, Vh = torch.linalg.svd(
    W,
    full_matrices=False,
)

W_rank = (
    U[:, :rank]
    * S[:rank].unsqueeze(0)
) @ Vh[:rank, :]

measured_frobenius = (
    torch.linalg.matrix_norm(
        W - W_rank,
        ord="fro",
    )
)

theoretical_frobenius = torch.sqrt(
    torch.sum(
        S[rank:].square()
    )
)

measured_spectral = (
    torch.linalg.matrix_norm(
        W - W_rank,
        ord=2,
    )
)

theoretical_spectral = S[rank]

print(
    measured_frobenius.item(),
    theoretical_frobenius.item(),
)

print(
    measured_spectral.item(),
    theoretical_spectral.item(),
)
```

## PyTorch確認-016

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（26. 1層の出力誤差を測る）

```python
from __future__ import annotations

import torch
from torch import nn


def layer_output_rmse(
    original: nn.Module,
    compressed: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    original.eval()
    compressed.eval()

    with torch.no_grad():
        y_original = original(x)
        y_compressed = compressed(x)

    return torch.sqrt(
        torch.mean(
            (
                y_original
                -
                y_compressed
            ).square()
        )
    )
```

## PyTorch確認-017

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（31. 活性状態の一致率）

```python
mask_match_ratio = (
    (z_original > 0)
    ==
    (z_compressed > 0)
).float().mean()
```

## PyTorch確認-018

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（37. 予測一致率）

```python
agreement = (
    logits_original.argmax(dim=1)
    ==
    logits_compressed.argmax(dim=1)
).float().mean()
```

## PyTorch確認-019

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（38. Cross Entropy Loss）

```python
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(
    logits,
    labels,
)
```

## PyTorch確認-020

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（誤った可能性がある計算）

```python
mse_values = []

for batch in loader:
    mse_values.append(
        batch_mse
    )

dataset_mse = sum(
    mse_values
) / len(mse_values)
```

## PyTorch確認-021

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]（65. `torch.allclose` は誤差指標ではない）

```python
is_close = torch.allclose(
    original,
    approximation,
    rtol=1e-5,
    atol=1e-6,
)
```

