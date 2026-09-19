# Conv2d行列化のAutograd境界

数式と考え方は [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## 14. `detach()` と `torch.no_grad()`

### `detach()`

SVDへ渡す元weightを、元のautograd graphから切り離す。

```python
W_mat = conv.weight.detach().flatten(
    start_dim=1,
)
```

### `torch.no_grad()`

新しく作った層へSVD因子を `copy_()` するときに、重み代入をautogradで追跡しないために使う。

```python
with torch.no_grad():
    new_layer.weight.copy_(...)
```

役割は異なる。

```text
detach
→ SVDへ渡すTensorを計算graphから切る

no_grad
→ 新しいParameterへの代入を追跡しない
```

### 元weightとのメモリ共有

別のTensor objectであることと、値が独立したstorageへコピーされたことは同じではない。`detach()` は計算graphを切るが、値のstorageは共有する。`flatten()` はshapeやstrideに応じて元object・view・copyのいずれかを返すため、新しいshapeだけを見て独立copyと判断しない。

書き換えの影響を、独立した小さいParameterで確認する。

```python
import torch
from torch import nn

# (out=2, in=1, kernel=2×2) の小さい例。
weight = nn.Parameter(
    torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2)
)
matrix_view = weight.detach().flatten(start_dim=1)
matrix_copy = weight.detach().clone().flatten(start_dim=1)

assert matrix_view.shape == matrix_copy.shape == (2, 4)
assert not matrix_view.requires_grad
assert not matrix_copy.requires_grad

# このcontiguousな例では、detach→flattenは元weightとstorageを共有する。
assert matrix_view.data_ptr() == weight.data_ptr()
assert matrix_copy.data_ptr() != weight.data_ptr()

matrix_view[0, 2] = 99
assert weight[0, 0, 1, 0].item() == 99  # 元Parameterも変化する。
assert matrix_copy[0, 2].item() == 2   # cloneした値には伝わらない。

matrix_copy[1, 3] = -7
assert weight[1, 0, 1, 1].item() == 7
```

ここでの更新は共有を調べるためだけであり、圧縮準備時に元weightを書き換える推奨ではない。読み取り専用のSVDなら `weight.detach().flatten(start_dim=1)` でよいが、作業用行列をin-placeで加工するなら `clone()` で独立させる。`no_grad()` の中でも共有storageへの代入は元weightを変更するため、「graphを追跡しない」と「元の値を守る」は別の問題である。

`W_mat` を作るだけでは元のConv Moduleを差し替えていない。元層の設定を取り出し、SVD因子を新しいConvの4階weightへ戻してからModuleを置換する。

[detachの公式定義](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

[flattenの公式定義](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html)

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（対応コード）

```python
from nn_compression.compression import (
    conv2d_weight_matrix,
    retained_energy,
    retained_energy_from_matrix,
    truncated_svd,
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（3. PyTorchでは1行で行列化できる）

```python
W_mat = conv.weight.detach().flatten(
    start_dim=1,
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（3. PyTorchでは1行で行列化できる）

```python
W_mat = conv.weight.detach().reshape(
    conv.weight.shape[0],
    -1,
)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（手動loopと`torch.stack`の対応）

```python
import torch

# Conv weightと同じ4階Tensorを、値の追跡ができる連番で作る。
weight = torch.arange(64 * 32 * 3 * 3).reshape(64, 32, 3, 3)

# 各出力filterを1行へflattenし、第0軸へ順番に積む。
rows = []
for out_channel in range(weight.shape[0]):
    row = weight[out_channel].flatten()
    rows.append(row)

weight_matrix_loop = torch.stack(rows, dim=0)
weight_matrix_direct = weight.flatten(start_dim=1)

# 値・行順・shapeが1行で行列化した結果と一致する。
assert torch.equal(weight_matrix_loop, weight_matrix_direct)
assert weight_matrix_direct.shape == (64, 288)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（4. `shape` とTensorを混同しない）

```python
conv.weight.shape
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（4. `shape` とTensorを混同しない）

```python
conv.weight
```

## PyTorch確認-007

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（4. `shape` とTensorを混同しない）

```python
W_mat = conv.weight.detach().flatten(start_dim=1)
```

## PyTorch確認-008

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（7. Reduced SVD）

```python
U, S, Vh = torch.linalg.svd(
    W_mat,
    full_matrices=False,
)
```

## PyTorch確認-009

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（8. rank $r$ へ切り詰める）

```python
U_r = U[:, :rank]
S_r = S[:rank]
Vh_r = Vh[:rank, :]
```

## PyTorch確認-010

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（10. denseに復元するとshapeは元へ戻る）

```python
W_approx_mat = (
    U_r @ torch.diag(S_r) @ Vh_r
)
```

## PyTorch確認-011

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（10. denseに復元するとshapeは元へ戻る）

```python
W_approx_4d = W_approx_mat.reshape(
    64,
    32,
    3,
    3,
)
```

## PyTorch確認-012

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（13. Conv2dのretained energyでは先に行列化する）

```python
torch.linalg.svdvals(
    conv.weight,
)
```

## PyTorch確認-013

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（13. Conv2dのretained energyでは先に行列化する）

```python
singular_values[:rank]
```

## PyTorch確認-014

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]（13. Conv2dのretained energyでは先に行列化する）

```python
weight_matrix = conv.weight.detach().reshape(
    conv.weight.shape[0],
    -1,
)

singular_values = torch.linalg.svdvals(
    weight_matrix,
)
```

