# `core_from_factors`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

現在のfactor集合を使って、元TensorからTucker coreを計算する。

## Signature

```python
core_from_factors(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor
```

## 引数

- `X`: 2階以上の実数浮動小数点Tensor。
- `factors`: `{mode: U}`。

## 戻り値

各指定modeをfactor転置でprojectionしたcore Tensor。

## 使用場面

HOOI sweep後に新しいfactorからcoreを再計算するとき。

## ざっくりした処理

```text
X
→ dtype/ndim validation
→ 各(mode, U)について U.T をmode_dot
→ core
```

## 主なcontract / 注意事項

このutility単独では`ranks`を受け取らないため、factor key集合とrank key集合の一致までは検証しない。

## 関連API

`hooi_sweep`, `hooi`, `reconstruct_tucker`
