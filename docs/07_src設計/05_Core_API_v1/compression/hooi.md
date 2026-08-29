# `hooi`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

HOSVDを初期値としてHOOIを反復し、最終core/factorsと再構成誤差履歴を返す。

## Signature

```python
hooi(
    X: torch.Tensor,
    ranks: dict[int, int],
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]
```

## 引数

- `X`: 2階以上の実数浮動小数点Tensor。zero tensor不可。
- `ranks`: 空でない `{mode: rank}` Mapping。
- `max_iter`: 最大sweep数。0以上。
- `abs_tol`, `rel_tol`: 収束判定tolerance。

## 戻り値

- `core`: 最終core。
- `factors`: 最終factor dict。
- `history`: relative Frobenius errorのlist。

## 使用場面

HOSVDよりweight再構成誤差を反復改善したいとき。

## ざっくりした処理

```text
Public API validation
→ HOSVD初期化
→ factor/rank feasibility検証
→ history[0]を計算
→ hooi_sweep
→ core_from_factors
→ reconstruct_tucker
→ relative error
→ has_converged
→ 収束またはmax_iterまで反復
```

## 主なcontract / 注意事項

- `history[0]` はHOSVD初期誤差。
- `max_iter=0`でも初期値・feasibilityを検証して返す。
- zero tensorは相対誤差が定義できないため拒否。

## 関連API

`hosvd`, `hooi_sweep`, `core_from_factors`, `has_converged`, `relative_frobenius_error`
