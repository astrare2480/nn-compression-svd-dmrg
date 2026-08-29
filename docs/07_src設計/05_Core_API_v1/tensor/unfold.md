# `unfold`

**Stability:** A  
**定義:** `src/nn_compression/tensor/operations.py`

## 責務

Tensorの指定modeを行方向へ移し、mode-n unfoldingの2次元行列を作る。

## Signature

```python
unfold(X: torch.Tensor, mode: int) -> torch.Tensor
```

## 引数

- `X`: 展開対象Tensor。2階以上、各dimensionは正。
- `mode`: 行方向に置くaxis。bool以外のintで `0 <= mode < X.ndim`。

## 戻り値

shapeが次の2次元Tensor。

```text
(X.shape[mode], product(X.shape[m] for m != mode))
```

## 使用場面

HOSVD/HOOIで、特定modeに対してSVDを行う前処理。

## ざっくりした処理

```text
X
→ modeを先頭axisへ movedim
→ 残りaxisをflatten
→ 2次元行列
```

## 主なcontract / 注意事項

- `torch.Tensor.unfold()` のsliding-window処理とは別物。
- `fold()` とaxis規約を必ず一致させる。

## 関連API

`fold`, `mode_dot`, `hosvd`, `hooi_sweep`
