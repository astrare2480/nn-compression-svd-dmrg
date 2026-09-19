# 19 再現性のPyTorchコード

数式と意味は各例にリンクした基礎理論ノートを参照する。ここにはPyTorchで確認するコードだけを置く。複数の例は独立実行を前提とせず、各例の前提条件を理論ノートで確認する。

---

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（2. データ分割用Generator）

```python
split_generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    dataset_size,
    generator=split_generator,
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（3. DataLoader用Generator）

```python
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（3. DataLoader用Generator）

```python
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=loader_generator,
    ...,
)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（複数workerの乱数を初期化する）

```python
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_worker(_worker_id: int) -> None:
    """PyTorchが各workerへ割り当てたseedをPythonとNumPyへ渡す。"""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=loader_generator,
)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（3 seedの値から、平均と2種類の標準偏差を計算する）

```python
import pandas as pd

# 集計式を確認するための仮のaccuracyであり、実験runの値ではない。
accuracies = pd.Series(
    [0.901, 0.887, 0.906],
    index=[42, 43, 44],
    name="validation_accuracy",
)
mean_accuracy = accuracies.mean()
population_std = accuracies.std(ddof=0)  # 分母はseed数。
sample_std = accuracies.std(ddof=1)      # 分母はseed数-1。

assert abs(mean_accuracy - 0.898) < 1e-12
assert abs(population_std - 0.00804155872120988) < 1e-12
assert abs(sample_std - 0.0098488578017961) < 1e-12
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（10. 同じseedでも完全に同一とは限らない）

```python
torch.use_deterministic_algorithms(True)
```

## PyTorch確認-007

対応する数式・説明：[[00_基礎理論/04_実験設計/19_再現性と乱数管理]]（10. 同じseedでも完全に同一とは限らない）

```python
import torch

if torch.cuda.is_available():
    # 入力shapeに対する高速algorithmのbenchmark探索を無効化する。
    torch.backends.cudnn.benchmark = False

    # cuDNN convolutionでは決定論的algorithmを選ぶ。
    torch.backends.cudnn.deterministic = True

# PyTorch全体で、利用可能なら決定論的実装を選ぶ。
# 決定論的実装がない対象演算はerrorにして検出する。
torch.use_deterministic_algorithms(True)
```

