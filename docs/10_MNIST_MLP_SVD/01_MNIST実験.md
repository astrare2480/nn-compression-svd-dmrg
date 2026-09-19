---
title: MNIST実験
aliases:
  - MNISTでSVD圧縮
  - SVD圧縮実験
  - MNIST低ランク近似
tags:
  - MNIST
  - PyTorch
  - SVD
  - 低ランク近似
  - NN圧縮
  - 実験
---

# MNIST実験

## サマリー

この章では、MNIST分類用のMLPを学習し、学習済みLinear層へSVDを適用して、rankごとの圧縮率・誤差・分類精度を比較する。

```mermaid
flowchart LR
    D["MNISTを読み込む"] --> T["未圧縮MLPを学習"]
    T --> B["baselineを評価・保存"]
    B --> S["学習済みfc1をSVD"]
    S --> R["複数rankで2層Linearへ置換"]
    R --> E["圧縮直後を評価"]
    E --> F["fine-tuning"]
    F --> C["圧縮率・誤差・精度を比較"]
```

対象モデルは、次の2層MLPとする。

```text
入力画像 (N, 1, 28, 28)
          ↓
Flatten → (N, 784)
          ↓
Linear(784, 512)
          ↓
ReLU
          ↓
Linear(512, 10)
          ↓
logits (N, 10)
```

最初に圧縮するのは `Linear(784, 512)` である。この層はモデル全体のパラメータの大部分を占めるため、SVD圧縮の効果を確認しやすい。

この実験では、次を比較する。

- 特異値スペクトルと累積エネルギー
- パラメータ保持率・削減率・圧縮倍率
- 相対フロベニウス誤差
- 対象Linear層の出力RMSE
- logits RMSEと予測一致率
- test lossとtest accuracy
- fine-tuning後のaccuracy
- 実測推論時間

---

## 1. 実験の目的

目的は、単にMNISTで高精度を出すことではない。

> 学習済みニューラルネットワークの重みをSVDで低ランク化したとき、圧縮率と性能の間にどのような関係が現れるかを理解する。

rank $r$ を下げると、圧縮後の重み数は減る。

$$
P_{\mathrm{lowrank}}
=
r
\left(
D_{\mathrm{in}}
+
D_{\mathrm{out}}
\right)
$$

一方、捨てる特異値が増えるため、重み誤差は増える。

$$
\lVert W-W_r\rVert_F^2
=
\sum_{i=r+1}^{k}
\sigma_i^2
$$

ただし、重み誤差と分類精度低下は同じものではない。実データの分布、ReLU、後続層、分類marginにも依存する。

---

## 2. 実験仮説

### 仮説1

rankを下げるほどパラメータ数は減る。

### 仮説2

rankを下げるほど重み誤差と層出力誤差は増える。

### 仮説3

重み誤差が増えても、accuracyは同じ割合では低下しない。

### 仮説4

圧縮直後にaccuracyが低下しても、fine-tuningで一部回復する。

---

## 3. 実験条件

比較では、rank以外の条件をできるだけ固定する。

- 同じ学習済みbaseline
- 同じデータ分割
- 同じ前処理
- 同じ評価用DataLoader
- 同じloss関数
- 同じdeviceとdtype
- 同じfine-tuning epoch数と学習率
- 同じ推論時間測定条件

rank候補は、最初は次でよい。

$$
r
\in
\{8,16,32,64,128,256\}
$$

---

## 4. Notebook名と構成

推奨Notebook名：

```text
08_mnist_svd_experiment.ipynb
```

推奨セル構成：

```text
Cell 1：import
Cell 2：seed・device
Cell 3：Dataset・DataLoader
Cell 4：モデル定義
Cell 5：学習・評価関数
Cell 6：baseline学習
Cell 7：baseline保存
Cell 8：特異値解析
Cell 9：rankごとの圧縮
Cell 10：誤差・accuracy評価
Cell 11：fine-tuning
Cell 12：表・グラフ・考察
```

---

## 5. import

```python
from __future__ import annotations

from pathlib import Path
import copy
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
```

[[05_SVD基礎実装検証/07_PyTorch実装]] のコードを `svd_compression.py` として保存した場合は、次もimportする。

```python
from svd_compression import (
    LowRankLinear,
    benchmark_module,
    compare_tensors,
    compare_weight_matrices,
    compression_stats,
    retained_energy,
)
```

---

## 6. seedとdevice

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SEED = 0
set_seed(SEED)
```

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)
```

seedを固定しても、すべてのGPU演算が完全に同じになるとは限らない。ただし、比較の再現性を高めるために固定する。

---

## 7. 保存先

```python
project_root = Path(".")
checkpoint_dir = project_root / "checkpoints"
result_dir = project_root / "results"
figure_dir = result_dir / "figures"

checkpoint_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)
```

---

## 8. MNISTを読み込む

最初の実験では、`ToTensor()` だけでもよい。

```python
transform = transforms.ToTensor()
```

標準化を使う場合は、baselineと圧縮後で統一する。

```python
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ]
)
```

Dataset：

```python
data_root = project_root / "data"

full_train_dataset = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root=data_root,
    train=False,
    download=True,
    transform=transform,
)
```

---

## 9. train・validationへ分割する

```python
train_size = 55_000
validation_size = 5_000

generator = torch.Generator().manual_seed(SEED)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    lengths=[train_size, validation_size],
    generator=generator,
)
```

rank選択にはvalidationを使い、testは最終評価に使うのが基本である。

---

## 10. DataLoader

```python
BATCH_SIZE = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
```

WindowsやNotebook環境では、最初は `num_workers=0` がトラブルを避けやすい。

---

## 11. データ形状を確認する

```python
images, labels = next(iter(train_loader))

print(tuple(images.shape))
print(tuple(labels.shape))
print(images.dtype)
print(labels.dtype)
```

期待される形状：

```text
images: (batch_size, 1, 28, 28)
labels: (batch_size,)
```

---

## 12. baselineモデル

```python
class MnistMLP(nn.Module):
    def __init__(
        self,
        hidden_features: int = 512,
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, 10)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
```

```python
baseline_model = MnistMLP(
    hidden_features=512,
).to(device)

print(baseline_model)
```

形状確認：

```python
dummy_input = torch.randn(
    32,
    1,
    28,
    28,
    device=device,
)

with torch.inference_mode():
    dummy_output = baseline_model(dummy_input)

print(tuple(dummy_output.shape))
```

期待値は `(32, 10)` である。

---

## 13. パラメータ数

```python
def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


baseline_parameter_count = count_parameters(
    baseline_model
)

print(baseline_parameter_count)
```

このモデルのパラメータ数は、

$$
401{,}920+5{,}130=407{,}050
$$

である。

---

## 14. 学習関数

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int(
            (logits.argmax(dim=1) == labels)
            .sum()
            .item()
        )
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )
```

---

## 15. 評価関数

```python
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int(
                (logits.argmax(dim=1) == labels)
                .sum()
                .item()
            )
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )
```

---

## 16. baselineを学習する

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    baseline_model.parameters(),
    lr=1e-3,
)

BASELINE_EPOCHS = 5
history = []

for epoch in range(1, BASELINE_EPOCHS + 1):
    train_loss, train_accuracy = train_one_epoch(
        model=baseline_model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    validation_loss, validation_accuracy = (
        evaluate_classifier(
            model=baseline_model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
    )

    print(
        f"epoch={epoch} "
        f"train_loss={train_loss:.4f} "
        f"train_accuracy={train_accuracy:.4f} "
        f"validation_loss={validation_loss:.4f} "
        f"validation_accuracy={validation_accuracy:.4f}"
    )
```

rankごとにbaselineを再学習しない。同じ学習済み重みを全rankの出発点にする。

---

## 17. baselineをテスト・保存する

```python
baseline_test_loss, baseline_test_accuracy = (
    evaluate_classifier(
        model=baseline_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
)

print(baseline_test_loss)
print(baseline_test_accuracy)
```

```python
baseline_path = (
    checkpoint_dir
    / "mnist_mlp_baseline.pt"
)

torch.save(
    {
        "model_state_dict": baseline_model.state_dict(),
        "hidden_features": 512,
        "seed": SEED,
        "test_loss": baseline_test_loss,
        "test_accuracy": baseline_test_accuracy,
    },
    baseline_path,
)
```

---

## 18. 圧縮対象層を確認する

```python
target_layer = baseline_model.fc1

print(target_layer)
print(tuple(target_layer.weight.shape))
print(target_layer.bias is not None)
```

期待される重み形状は `(512, 784)` である。

---

## 19. 特異値を計算する

```python
target_weight = target_layer.weight.detach()

singular_values = torch.linalg.svdvals(
    target_weight
)

print(tuple(singular_values.shape))
```

特異値は512個で、大きい順に並ぶ。

---

## 20. 特異値スペクトル

```python
values = singular_values.cpu().numpy()

plt.figure()
plt.plot(
    range(1, len(values) + 1),
    values,
)
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.title("fc1 Singular Value Spectrum")
plt.grid(True)
plt.show()
```

対数軸：

```python
plt.figure()
plt.semilogy(
    range(1, len(values) + 1),
    values,
)
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.title("fc1 Singular Value Spectrum")
plt.grid(True)
plt.show()
```

観察する点：

- 上位成分への集中
- 折れ曲がり位置
- 後半特異値の減衰
- 低rank近似が有効そうか

---

## 21. 累積エネルギー

```python
squared_values = singular_values.square()

cumulative_energy = (
    torch.cumsum(squared_values, dim=0)
    / squared_values.sum()
)

energy_values = cumulative_energy.cpu().numpy()

plt.figure()
plt.plot(
    range(1, len(energy_values) + 1),
    energy_values,
)
plt.xlabel("Rank")
plt.ylabel("Retained energy")
plt.title("fc1 Cumulative Singular Value Energy")
plt.ylim(0.0, 1.01)
plt.grid(True)
plt.show()
```

エネルギー保持率はrank候補を考える参考になるが、accuracyを保証しない。

---

## 22. エネルギー閾値rank

```python
for target_energy in [
    0.90,
    0.95,
    0.99,
    0.999,
]:
    rank = int(
        torch.searchsorted(
            cumulative_energy,
            torch.tensor(
                target_energy,
                device=cumulative_energy.device,
                dtype=cumulative_energy.dtype,
            ),
        ).item()
    ) + 1

    print(target_energy, rank)
```

---

## 23. rank候補

```python
RANKS = [
    8,
    16,
    32,
    64,
    128,
    256,
]
```

`Linear(784, 512)` では、rank 309以下なら重み数は減る。しかし、rank 309付近は削減量が小さいため、最初の比較では不要である。

---

## 24. 圧縮モデルを作る

```python
def make_compressed_model(
    baseline: MnistMLP,
    rank: int,
) -> MnistMLP:
    compressed = copy.deepcopy(baseline)

    compressed.fc1 = LowRankLinear.from_linear(
        layer=compressed.fc1,
        rank=rank,
        placement="second",
    )

    return compressed
```

元から `fc1` の後ろにあるReLUは残す。

```text
LowRankLinear
├─ Linear(784, rank, bias=False)
└─ Linear(rank, 512, bias=True)
        ↓
元のReLU
```

2つのLinearの間にReLUを入れない。

---

## 25. 最大rankで実装確認

```python
maximum_rank = min(
    target_layer.in_features,
    target_layer.out_features,
)

full_rank_model = make_compressed_model(
    baseline=baseline_model,
    rank=maximum_rank,
)
```

```python
images, _ = next(iter(test_loader))
images = images.to(device)

baseline_model.eval()
full_rank_model.eval()

with torch.inference_mode():
    baseline_logits = baseline_model(images)
    full_rank_logits = full_rank_model(images)

print(
    torch.allclose(
        baseline_logits,
        full_rank_logits,
        rtol=1e-4,
        atol=1e-5,
    )
)
```

最大rankで大きな差が出る場合は、`Vh`、特異値の掛け方、bias、前段と後段の順序を確認する。

---

## 26. テンソル誤差集計器

データセット全体のRMSEは、バッチRMSEの単純平均では求めない。二乗誤差和と要素数を積算する。

```python
class TensorErrorAccumulator:
    def __init__(self) -> None:
        self.element_count = 0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.maximum_absolute_error = 0.0
        self.reference_squared_sum = 0.0

    def update(
        self,
        reference: torch.Tensor,
        approximation: torch.Tensor,
    ) -> None:
        if reference.shape != approximation.shape:
            raise ValueError("tensor shapes must match")

        difference = reference - approximation

        self.element_count += difference.numel()
        self.squared_error_sum += float(
            difference.square().sum().item()
        )
        self.absolute_error_sum += float(
            difference.abs().sum().item()
        )
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(difference.abs().max().item()),
        )
        self.reference_squared_sum += float(
            reference.square().sum().item()
        )

    def compute(self) -> dict[str, float]:
        if self.element_count == 0:
            raise ValueError("no values were accumulated")

        mse = self.squared_error_sum / self.element_count
        rmse = mse ** 0.5
        mae = self.absolute_error_sum / self.element_count
        reference_rms = (
            self.reference_squared_sum
            / self.element_count
        ) ** 0.5

        relative_rmse = (
            rmse / reference_rms
            if reference_rms > 0
            else float("nan")
        )

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "maximum_absolute_error": (
                self.maximum_absolute_error
            ),
        }
```

---

## 27. 対象層の出力誤差

第1Linear層の入力は、画像をFlattenしたものなので直接作れる。

```python
def evaluate_first_layer_error(
    baseline: MnistMLP,
    compressed: MnistMLP,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    baseline.eval()
    compressed.eval()

    accumulator = TensorErrorAccumulator()

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            flattened = images.flatten(start_dim=1)

            reference = baseline.fc1(flattened)
            approximation = compressed.fc1(flattened)

            accumulator.update(
                reference=reference,
                approximation=approximation,
            )

    return accumulator.compute()
```

---

## 28. logits誤差と予測一致率

```python
def evaluate_logits_error(
    baseline: nn.Module,
    compressed: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    baseline.eval()
    compressed.eval()

    accumulator = TensorErrorAccumulator()
    agreement_count = 0
    sample_count = 0

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            reference = baseline(images)
            approximation = compressed(images)

            accumulator.update(
                reference=reference,
                approximation=approximation,
            )

            agreement_count += int(
                (
                    reference.argmax(dim=1)
                    == approximation.argmax(dim=1)
                )
                .sum()
                .item()
            )
            sample_count += images.shape[0]

    return (
        accumulator.compute(),
        agreement_count / sample_count,
    )
```

---

## 29. rankごとに圧縮直後を評価する

```python
rank_results = []

for rank in RANKS:
    compressed_model = make_compressed_model(
        baseline=baseline_model,
        rank=rank,
    )

    stats = compression_stats(
        in_features=784,
        out_features=512,
        rank=rank,
        bias=True,
    )

    energy = retained_energy(
        singular_values=singular_values,
        rank=rank,
    )

    weight_metrics = compare_weight_matrices(
        original=baseline_model.fc1.weight,
        approximation=(
            compressed_model
            .fc1
            .reconstructed_weight
        ),
    )

    layer_error = evaluate_first_layer_error(
        baseline=baseline_model,
        compressed=compressed_model,
        loader=test_loader,
        device=device,
    )

    logits_error, agreement = evaluate_logits_error(
        baseline=baseline_model,
        compressed=compressed_model,
        loader=test_loader,
        device=device,
    )

    test_loss, test_accuracy = evaluate_classifier(
        model=compressed_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    rank_results.append(
        {
            "rank": rank,
            "model_parameters": count_parameters(
                compressed_model
            ),
            "fc1_weight_parameters": (
                stats.compressed_weight_parameters
            ),
            "fc1_keep_ratio": stats.weight_keep_ratio,
            "fc1_reduction_ratio": (
                stats.weight_reduction_ratio
            ),
            "fc1_compression_factor": (
                stats.weight_compression_factor
            ),
            "retained_energy": energy,
            "relative_weight_error": (
                weight_metrics.relative_frobenius_error
            ),
            "spectral_error": (
                weight_metrics.spectral_error
            ),
            "layer_output_rmse": layer_error["rmse"],
            "layer_output_relative_rmse": (
                layer_error["relative_rmse"]
            ),
            "logits_rmse": logits_error["rmse"],
            "prediction_agreement": agreement,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "accuracy_drop": (
                baseline_test_accuracy
                - test_accuracy
            ),
        }
    )

    print(
        f"rank={rank} "
        f"accuracy={test_accuracy:.4f} "
        f"drop={baseline_test_accuracy - test_accuracy:.4f} "
        f"keep={stats.weight_keep_ratio:.4f}"
    )
```

---

## 30. DataFrameとCSV

```python
results_df = pd.DataFrame(rank_results)
results_df
```

```python
results_df["fc1_keep_percent"] = (
    100 * results_df["fc1_keep_ratio"]
)

results_df["fc1_reduction_percent"] = (
    100 * results_df["fc1_reduction_ratio"]
)

results_df.to_csv(
    result_dir
    / "rank_results_before_finetuning.csv",
    index=False,
)
```

---

## 31. rankとaccuracy

```python
plt.figure()
plt.plot(
    results_df["rank"],
    results_df["test_accuracy"],
    marker="o",
)
plt.axhline(
    baseline_test_accuracy,
    linestyle="--",
    label="Baseline",
)
plt.xlabel("Rank")
plt.ylabel("Test accuracy")
plt.title("Rank and Test Accuracy")
plt.grid(True)
plt.legend()
plt.show()
```

---

## 32. rankと重み誤差

```python
plt.figure()
plt.plot(
    results_df["rank"],
    results_df["relative_weight_error"],
    marker="o",
)
plt.xlabel("Rank")
plt.ylabel("Relative Frobenius error")
plt.title("Rank and Weight Error")
plt.grid(True)
plt.show()
```

---

## 33. rankと層出力RMSE

```python
plt.figure()
plt.plot(
    results_df["rank"],
    results_df["layer_output_rmse"],
    marker="o",
)
plt.xlabel("Rank")
plt.ylabel("Layer output RMSE")
plt.title("Rank and Layer Output Error")
plt.grid(True)
plt.show()
```

---

## 34. パラメータ数とaccuracy

```python
plt.figure()
plt.plot(
    results_df["model_parameters"],
    results_df["test_accuracy"],
    marker="o",
)
plt.xlabel("Model parameter count")
plt.ylabel("Test accuracy")
plt.title("Parameter Count and Accuracy")
plt.grid(True)
plt.show()
```

rankではなく、実際のパラメータ数を横軸にすると、別手法とも比較しやすい。

---

## 35. fine-tuningするrank

最初は代表rankだけでもよい。

```python
FINE_TUNE_RANKS = [
    32,
    64,
    128,
]
```

- rank 32：強い圧縮
- rank 64：中程度
- rank 128：比較的弱い圧縮

厳密にはvalidation accuracyを見て選ぶ。

---

## 36. fine-tuning関数

```python
def fine_tune_model(
    baseline: MnistMLP,
    rank: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 1e-4,
) -> tuple[MnistMLP, list[dict[str, float | int]]]:
    model = make_compressed_model(
        baseline=baseline,
        rank=rank,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = (
            evaluate_classifier(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_accuracy
                ),
            }
        )

    return model, history
```

optimizerは低ランク層へ置換した後に作る。

---

## 37. fine-tuningを実行する

```python
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4

fine_tuned_results = []

for rank in FINE_TUNE_RANKS:
    fine_tuned_model, fine_tune_history = (
        fine_tune_model(
            baseline=baseline_model,
            rank=rank,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            device=device,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LR,
        )
    )

    test_loss, test_accuracy = evaluate_classifier(
        model=fine_tuned_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    fine_tuned_results.append(
        {
            "rank": rank,
            "fine_tuned_test_loss": test_loss,
            "fine_tuned_test_accuracy": test_accuracy,
            "fine_tune_epochs": FINE_TUNE_EPOCHS,
            "fine_tune_learning_rate": FINE_TUNE_LR,
        }
    )

    torch.save(
        {
            "rank": rank,
            "model_state_dict": (
                fine_tuned_model.state_dict()
            ),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": fine_tune_history,
        },
        checkpoint_dir
        / f"mnist_mlp_rank{rank}_finetuned.pt",
    )
```

---

## 38. fine-tuning結果を結合する

```python
fine_tuned_df = pd.DataFrame(fine_tuned_results)

combined_df = results_df.merge(
    fine_tuned_df,
    on="rank",
    how="left",
)

combined_df.to_csv(
    result_dir
    / "rank_results_with_finetuning.csv",
    index=False,
)
```

---

## 39. 圧縮直後とfine-tuning後

```python
plt.figure()
plt.plot(
    combined_df["rank"],
    combined_df["test_accuracy"],
    marker="o",
    label="Before fine-tuning",
)
plt.plot(
    combined_df["rank"],
    combined_df["fine_tuned_test_accuracy"],
    marker="o",
    label="After fine-tuning",
)
plt.axhline(
    baseline_test_accuracy,
    linestyle="--",
    label="Baseline",
)
plt.xlabel("Rank")
plt.ylabel("Test accuracy")
plt.title("Accuracy Before and After Fine-tuning")
plt.grid(True)
plt.legend()
plt.show()
```

fine-tuning後も、実効重みは、

$$
W_{\mathrm{effective}}=BA
$$

である。中間次元がrank $r$ なので、

$$
\operatorname{rank}(BA)\leq r
$$

が保たれる。

---

## 40. 推論時間を測定する

```python
example_images = torch.randn(
    BATCH_SIZE,
    1,
    28,
    28,
    device=device,
)

baseline_benchmark = benchmark_module(
    module=baseline_model,
    example_input=example_images,
)

print(baseline_benchmark)
```

rankごと：

```python
benchmark_rows = []

for rank in RANKS:
    compressed_model = make_compressed_model(
        baseline=baseline_model,
        rank=rank,
    )

    result = benchmark_module(
        module=compressed_model,
        example_input=example_images,
    )

    benchmark_rows.append(
        {
            "rank": rank,
            "median_ms": result.median_milliseconds,
            "minimum_ms": result.minimum_milliseconds,
            "maximum_ms": result.maximum_milliseconds,
        }
    )
```

測定条件を記録する。

- device
- dtype
- batch size
- input shape
- ウォームアップ回数
- 測定回数

パラメータ数が減っても、2層化のオーバーヘッドで速くならない場合がある。

---

## 41. 結果表

| rank | モデルパラメータ数 | fc1保持率 | エネルギー保持率 | 相対重み誤差 | 層出力RMSE | logits RMSE | 予測一致率 | 圧縮直後accuracy | fine-tuning後accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 407,050 | 100% | 100% | 0 | 0 | 0 | 100% |  |  |
| 8 |  |  |  |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |  |  |  |
| 32 |  |  |  |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |  |  |  |
| 128 |  |  |  |  |  |  |  |  |  |
| 256 |  |  |  |  |  |  |  |  |  |

---

## 42. 最初に見るべき結果

1. baseline accuracyが十分か
2. 最大rankで元出力を再構成できるか
3. 特異値がどのように減衰するか
4. rank低下に対して重み誤差が増えるか
5. 重み誤差とaccuracy低下がどの程度対応するか
6. fine-tuningでどこまで回復するか
7. 理論圧縮率と実測速度が一致するか

---

## 43. 結果の読み方

### rank 64でもaccuracyがほぼ低下しない

主要な分類情報が64以下の方向で十分表現されている可能性がある。

### エネルギー99%でもaccuracyが低下する

小さい特異方向が分類境界に重要だった可能性がある。SVDの重みノルム最適性とタスク最適性の違いを示す。

### 重み誤差は大きいがaccuracyが保たれる

捨てた方向が実際のMNIST入力であまり使われていない可能性がある。

### fine-tuningで大きく回復する

低rank構造には必要な表現能力が残っているが、切り詰めSVD直後の因子が分類タスクへ最適ではなかった可能性がある。

### fine-tuningでも回復しない

rankが小さすぎ、必要な表現能力を失った可能性がある。

---

## 44. 公平な比較のための注意

### baselineをrankごとに再学習しない

同じ学習済みbaselineから全rankを作る。

### テストデータでrankを選ばない

validationでrankを選び、testは最終評価に使う。

### fine-tuning条件を揃える

rank以外のepoch数、optimizer、学習率、データを固定する。

### 同じ入力で比較する

圧縮前後の出力誤差は、同じミニバッチで測る。

### 圧縮直後とfine-tuning後を分ける

SVD近似そのものと再学習による回復を混同しない。

---

## 45. 実験メタデータを保存する

```python
metadata = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "baseline_epochs": BASELINE_EPOCHS,
    "baseline_learning_rate": 1e-3,
    "fine_tune_epochs": FINE_TUNE_EPOCHS,
    "fine_tune_learning_rate": FINE_TUNE_LR,
    "ranks": RANKS,
    "device": str(device),
    "torch_version": torch.__version__,
}

with open(
    result_dir / "experiment_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )
```

さらに、Python、torchvision、GPU名も記録する。

---

## 46. 最初の実験でやる範囲

最初は次までで十分である。

1. baselineを学習
2. `fc1`の特異値を可視化
3. rank 8、16、32、64、128、256で圧縮
4. パラメータ数を比較
5. 相対重み誤差を比較
6. 層出力RMSEを比較
7. test accuracyを比較
8. rank 32、64、128をfine-tuning
9. CSVとグラフを保存
10. 結果を文章で考察

複数seed、`fc2`圧縮、量子化、pruningは追加実験へ回す。

---

## 47. 完了条件

- baseline checkpointを保存した
- baseline test accuracyを記録した
- `fc1`の特異値スペクトルを描いた
- 最大rankで再構成を確認した
- 各rankの圧縮率を計算した
- 各rankの重み誤差を計算した
- 各rankの層出力誤差を計算した
- 各rankのtest accuracyを計算した
- 代表rankをfine-tuningした
- CSVとグラフを保存した
- 考察を書いた

---

## 48. よくある実装ミス

### 学習前の重みを圧縮する

基本実験では学習済み重みを対象にする。

### 2層の間にReLUを入れる

元のLinear層の低ランク近似ではなくなる。

### 元のReLUを削除する

低ランク2層の後ろへ元のReLUを残す。

### optimizerを置換前から使い続ける

新しい低ランクパラメータが更新されない可能性がある。

### 最大rankで再構成を確認しない

`Vh`、bias、形状の実装ミスを見逃す。

### バッチRMSEを単純平均する

データセット全体のRMSEと一致しない。

### accuracyだけを見る

logits、loss、予測一致率、内部誤差も確認する。

### パラメータ数だけで高速化を判断する

実測時間はハードウェアと2層化オーバーヘッドに依存する。

---

## 49. よくある誤解

### rank 64なら64分の1になる

圧縮後重み数は、

$$
64(784+512)
$$

である。

### エネルギー99%ならaccuracyも99%残る

保証されない。

### MSEが小さければaccuracyも必ず高い

分類marginなどに依存する。

### fine-tuningで必ず元精度へ戻る

rankが小さすぎると戻らない。

### MNISTで成功すれば大規模モデルでも同じ

MNIST MLPは原理確認用の小規模実験である。

---

## 50. レポート構成

### 目的

学習済みMNIST MLPの第1Linear層をSVD低ランク化し、rankと圧縮率・誤差・精度の関係を調べる。

### モデル

```text
784 → 512 → 10
```

活性化関数はReLU。

### 圧縮方法

$$
W_r
=
U_r\Sigma_rV_r^{\mathsf{T}}
$$

を2つのLinear層として実装する。

### 評価指標

- パラメータ数
- 相対フロベニウス誤差
- 層出力RMSE
- logits RMSE
- test loss
- test accuracy
- fine-tuning後accuracy

### 考察

- 特異値減衰
- 精度維持可能なrank
- 重み誤差とaccuracyの関係
- fine-tuning効果
- 推論時間との関係

---

## 51. 考察文テンプレート

> 学習済み第1Linear層の特異値は、上位成分ほど大きく、rankの増加とともに累積エネルギーが単調に増加した。rankを下げるとパラメータ数は線形に減少し、相対フロベニウス誤差と層出力RMSEは増加した。一方、分類精度は重み誤差と同じ割合では低下せず、一定rankまではほぼ維持された。この結果は、学習済み重みに分類タスクへ直接寄与しにくい冗長な方向が含まれる可能性を示す。ただし、切り詰めSVDが最適化しているのは重み行列のノルム誤差であり、分類精度ではない。fine-tuning後に精度が回復したrankでは、低rank構造のままタスクへ再適応できたと考えられる。

数値を入れて具体化する。

---

## 52. 結論文テンプレート

> 第1Linear層をrank 64へ低ランク化した結果、同層の重みパラメータ数は401,408から82,944へ減少し、約79.34%削減できた。圧縮直後のtest accuracy低下はXポイントで、Y epochのfine-tuning後にはZポイントまで回復した。したがって、本実験条件ではrank 64が圧縮率と精度の妥協点となった。ただし、実測推論時間は理論MAC削減率と同じ割合では改善しなかった。

---

## 53. 実験の限界

- MNISTは比較的簡単なデータセット
- MLPは小規模
- 1つのモデル構造だけ
- 1つのseedだけでは学習ばらつきを評価できない
- 推論時間は環境依存
- SVDはタスク誤差を直接最小化しない
- Linear層だけを対象にしている
- fine-tuning条件を十分探索していない

この限界を明記し、原理確認の基準実験として位置付ける。

---

## 54. 次の追加実験

- 複数seedで平均と標準偏差を報告
- 固定rankとエネルギー基準rankを比較
- 特異値配置 `first`・`second`・`balanced` を比較
- `fc1`だけ・`fc2`だけ・両方を比較
- 最初から低ランク構造で学習
- 隠れ次元128・256・1024を比較
- 量子化やpruningとの比較

---

## 55. DMRGへつなげるために残す結果

将来の比較基準として、次を保存する。

- 元モデルaccuracy
- rankごとのパラメータ数
- rankごとのaccuracy
- fine-tuning前後の差
- 推論時間
- 特異値スペクトル
- 圧縮対象層
- 実験コード・seed・環境情報

単純SVDが十分高性能なら、より複雑な方法を使う必要性は小さい。逆に、同じパラメータ数でSVDより高精度な方法が得られれば、追加手法の価値を示せる。

---

## 56. 研究・面接で説明するなら

### 30秒程度の説明

> MNISTの2層MLPを学習し、パラメータの大部分を占める第1Linear層へSVDを適用しました。学習済み重みを複数rankで切り詰め、2つの小さいLinear層へ置き換えました。各rankについて、パラメータ削減率、相対重み誤差、層出力RMSE、test accuracyを比較し、代表rankではfine-tuning後の回復も評価します。この実験により、SVDが最小化する重み誤差と、実際の分類性能の関係を確認できます。

---

## 57. このノートで押さえるポイント

- 同じ学習済みbaselineを全rankの出発点にする。
- 最初の圧縮対象は `Linear(784, 512)` とする。
- rank候補は8、16、32、64、128、256が使いやすい。
- baselineを学習・評価・保存してから圧縮する。
- 特異値スペクトルと累積エネルギーを確認する。
- 最大rankで元の出力を再構成できることを確認する。
- 2層の間にReLUを入れない。
- 元から後ろにあるReLUは残す。
- rankごとに圧縮率、重み誤差、層出力誤差、accuracyを測る。
- データセット全体のRMSEは二乗誤差和を積算して求める。
- test accuracyだけでなくlossと予測一致率も見る。
- 圧縮直後とfine-tuning後を分けて記録する。
- optimizerは低ランク層へ置換した後に作る。
- rank選択にはvalidationを使う。
- パラメータ削減率と推論時間改善率は同じではない。
- 結果はCSV、checkpoint、グラフとして保存する。
- MNIST実験は原理確認であり、大規模モデルへそのまま一般化しない。
- 単純SVDの結果を、将来のDMRG・MPO型圧縮の比較基準として残す。

---

## 58. 次に読むノート

次は、SVDとDMRGに共通する「特異値を基準に重要な状態を残す」という考え方を、混同しないように整理する。

- [[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/02_nn.Linearとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[05_SVD基礎実装検証/07_PyTorch実装]]
- [[50_DMRG/01_SVDからDMRGへのつながり]]

---

## 59. 実装プロファイルと結果ノート

このノートの最小実験は、

```text
プロファイルA
784 → 512 → 10
```

を基準とする。

今回の実装検討では、複数層を扱う、

```text
プロファイルB
784 → 512 → 256 → 10
```

も追加する。

プロファイルBでは、`fc1` と `fc2` を主な圧縮対象とし、`fc3` は標準では圧縮しない。`fc3 = Linear(256, 10)` の最大rankは10だが、rank 10で2層化するとパラメータ数が元より増えるためである。

実装詳細と実測結果は次を参照する。

- [[README_実装編]]
- [[10_MNIST_MLP_SVD/04_MNISTでの実験結果]]

このノートは実験設計を扱い、結果ノートは実際に得た数値だけを扱う。
