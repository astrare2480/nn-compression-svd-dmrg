# 00 SVD理論のPyTorch確認コード

数式と意味は各例にリンクした基礎理論ノートを参照する。ここにはPyTorchで確認するコードだけを置く。複数の例は独立実行を前提とせず、各例の前提条件を理論ノートで確認する。

---

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（3. Reduced SVD）

```python
torch.linalg.svd(W, full_matrices=False)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（15. 実装上のSVD）

```python
U, S, Vh = np.linalg.svd(W, full_matrices=False)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（15. 実装上のSVD）

```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（15. 実装上のSVD）

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
W_r = (U_r * S_r) @ Vh_r
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（$5\times3$ の再構成例を手順ごとに確認する）

```python
import numpy as np

# 5×3の行列。seedと生成法を固定して再構成を確認する。
rng = np.random.default_rng(seed=0)
W = rng.standard_normal((5, 3))
U, S, Vh = np.linalg.svd(W, full_matrices=False)

print("W :", W.shape)
print("U :", U.shape)
print("S :", S.shape)
print("Vh:", Vh.shape)
Sigma = np.diag(S)
W_restored = U @ Sigma @ Vh
assert np.allclose(W, W_restored, rtol=1e-10, atol=1e-10)
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]（$5\times3$ の再構成例を手順ごとに確認する）

```python
import torch

# 全成分を残すreduced SVDは打ち切り近似ではない。
torch.manual_seed(0)
W = torch.randn(5, 3, dtype=torch.float64)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

print("W :", tuple(W.shape))
print("U :", tuple(U.shape))
print("S :", tuple(S.shape))
print("Vh:", tuple(Vh.shape))
W_restored = U @ torch.diag(S) @ Vh
assert torch.allclose(W, W_restored, rtol=1e-10, atol=1e-10)
```

## PyTorch確認-007

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]（19. 実装との対応）

```python
U, S, Vh = torch.linalg.svd(
    weight,
    full_matrices=False,
)
```

## PyTorch確認-008

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]（19. 実装との対応）

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
```

## PyTorch確認-009

対応する数式・説明：[[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]（19. 実装との対応）

```python
W_r = (U_r * S_r.unsqueeze(0)) @ Vh_r
```

