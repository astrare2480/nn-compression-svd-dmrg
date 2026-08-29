# `mode_dot`

**Stability:** A  
**定義:** `src/nn_compression/tensor/operations.py`

## 責務

Tensorの指定modeへ2次元行列を作用させるn-mode productを計算する。

## Signature

```python
mode_dot(
    X: torch.Tensor,
    matrix: torch.Tensor,
    mode: int,
) -> torch.Tensor
```

## 引数

- `X`: 2階以上のTensor。
- `matrix`: `(new_dim, X.shape[mode])` の2次元行列。
- `mode`: 行列を作用させるaxis。

## 戻り値

`X` の対象modeだけが `matrix.shape[0]` に置き換わったTensor。

## 使用場面

Tucker core生成、factorによるprojection、Tucker再構成。

## ざっくりした処理

```text
X
→ unfold(X, mode)
→ matrix @ unfolded
→ 出力shapeを作る
→ fold(...)
```

## 主なcontract / 注意事項

`matrix.shape[1] == X.shape[mode]` が必要。

## 関連API

`unfold`, `fold`, `hosvd`, `reconstruct_tucker`, `core_from_factors`
