# `tt_physical_shape`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb`

## 責務

検証済みTT core列から、各siteの物理次元だけを順序どおり取得する。

## Signature

```python
tt_physical_shape(
    cores: list[torch.Tensor],
) -> tuple[int, ...]
```

## 引数

- `cores`: 各要素がshape $(r_{k-1},n_k,r_k)$ のTT core列。境界rank、隣接bond、dtype、deviceの整合を要求する。

## 戻り値

各coreの中央axisを並べた物理shape

$$
(n_1,n_2,\ldots,n_d)
$$

を`tuple[int, ...]`で返す。

## 使用場面

- 2つのTTが同じdense shapeを表すか確認するとき。
- dense化せずに入出力shapeを記録するとき。
- TT-matrix等の後続表現へ渡すshape metadataを得るとき。

## 処理概要

1. `validate_tt_cores`でcore chain全体を検証する。
2. 各`core.shape[1]`をPython整数へ変換する。
3. site順のtupleとして返す。

## 主なcontract / 注意事項

- 無効なcore列から部分的なshapeを返さず、chain全体を先に検証する。
- TT-rankは戻り値に含めない。
- dense Tensorを構築せず、入力coreを変更しない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_ranks]]、[[90_src設計/05_Core_API_v1/compression/tt_reconstruct]]、[[90_src設計/05_Core_API_v1/Internal_API]]
