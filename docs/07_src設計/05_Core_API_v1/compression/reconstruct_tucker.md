# `reconstruct_tucker`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

Tucker coreへfactorを掛け戻し、元空間の近似Tensorを再構成する。

## Signature

```python
reconstruct_tucker(
    core: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor
```

## 引数

- `core`: 2階以上のcore Tensor。
- `factors`: `{mode: U}`。

## 戻り値

factorを掛け戻した再構成Tensor。

## 使用場面

HOSVD/HOOIのrelative error評価、Tucker-2 effective weight再構成。

## ざっくりした処理

```text
core
→ factorsの各(mode, U)を順にmode_dot
→ reconstructed Tensor
```

## 主なcontract / 注意事項

- 空`factors={}`ならidentityとしてcoreを返す。
- このutility自体は分解入口ではないためcomplex guardを持たない。

## 関連API

`hosvd`, `core_from_factors`, `relative_frobenius_error`, `tucker2_effective_weight`
