# `hosvd`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

指定したmodeだけをtruncated HOSVDで低rank化し、Tucker coreとfactorを返す。

## Signature

```python
hosvd(
    X: torch.Tensor,
    ranks: dict[int, int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]
```

## 引数

- `X`: 2階以上の**実数浮動小数点Tensor**。
- `ranks`: `{mode: rank}` Mapping。未指定modeは圧縮しない。

## 戻り値

- `core`: 指定modeだけrankへ縮めたcore Tensor。
- `factors`: `{mode: U}` のfactor辞書。

## 使用場面

generic Tucker分解、Conv weightのTucker-2 HOSVD、HOOIの初期値生成。

## ざっくりした処理

```text
各modeについて元Xをunfold
→ truncated_svd
→ Uをfactorへ保存
→ coreへ U.T をmode_dot
```

## 主なcontract / 注意事項

- factorは**逐次更新したcoreではなく元Xのunfolding**から独立に求める。
- `ranks={}` はidentityとして許可。
- rank上限はmode-n unfoldingの `min(rows, cols)`。
- complex/integer/bool Tensorは分解入口で拒否。

## 関連API

`unfold`, `mode_dot`, `reconstruct_tucker`, `hooi`, `tucker2_decompose_conv_weight`
