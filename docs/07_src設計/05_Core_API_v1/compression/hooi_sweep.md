# `hooi_sweep`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

現在のfactorを使い、`ranks`で指定された各modeを1回ずつ更新するHOOI 1 sweepを実行する。

## Signature

```python
hooi_sweep(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> dict[int, torch.Tensor]
```

## 引数

- `X`: 2階以上の実数浮動小数点Tensor。
- `factors`: 現在の `{mode: U}`。
- `ranks`: 更新対象modeとrank。

## 戻り値

更新済みfactor dict。

## 使用場面

HOOI反復の1単位を検証したいとき、Tucker-2 HOOIをgeneric処理へ委譲するとき。

## ざっくりした処理

```text
入力validation
→ factorsをclone
→ ranksの挿入順でtarget modeを選ぶ
→ target以外の最新factorでXをproject
→ unfold
→ truncated_svd
→ target factor更新
```

## 主なcontract / 注意事項

- 入力`factors`を破壊しない。
- sweep内で既に更新したfactorを後続modeが使うGauss-Seidel型。
- factor keys/shape/dtype/device、projected rank feasibilityを検証。

## 関連API

`hooi`, `tucker2_hooi_sweep`, `core_from_factors`
