# `core_from_factors`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

現在のfactor集合を使って、元TensorからTucker coreを再計算する。

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

各指定modeを`U.T`でprojectionしたcore Tensor。

## 使用場面

HOOI sweepでfactorを更新した後、そのfactor集合に対応するcoreを求め直すとき。

## 処理の流れ（日本語）

1. **`X`が2階以上か確認する。**
2. **HOSVD/HOOIと同じdtype contractを確認する。**  
   現行実装では実数浮動小数点Tensorのみを正式対応とする。
3. **coreの初期値を元Tensor`X`にする。**
4. **各`(mode, U)`についてfactor転置を掛ける。**  
   `core = mode_dot(core, U.T, mode)`として、そのmodeをfactorのrank dimensionまで縮める。
5. **すべてのfactorを適用したcoreを返す。**

### 処理フロー（短縮版）

```text
X
→ ndim / dtype検証
→ core = X
→ 各factorについて U.T をmode_dot
→ Tucker core
```

## 主なcontract / 注意事項

このutility単独では`ranks`を引数に取らないため、「factor key集合とrank key集合が一致するか」までは検証しない。その整合性は`hooi_sweep()`/`hooi()`側で保証する。

## 関連API

`hooi_sweep`, `hooi`, `reconstruct_tucker`
