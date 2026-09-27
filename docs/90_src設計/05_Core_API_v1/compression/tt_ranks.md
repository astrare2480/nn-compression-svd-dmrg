# `tt_ranks`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb`

## 責務

TT core列から、左右の境界rankを含むTT-rank列を取得する。

## Signature

```python
tt_ranks(
    cores: list[torch.Tensor],
) -> tuple[int, ...]
```

## 引数

- `cores`: shape $(r_{k-1},n_k,r_k)$ のTT core列。`validate_tt_cores`の構造contractを満たす必要がある。

## 戻り値

$d$ coreに対し、長さ$d+1$の

$$
(1,r_1,\ldots,r_{d-1},1)
$$

を返す。

## 使用場面

- rounding前後のrank変化を比較するとき。
- TT parameter数や圧縮率の説明用metadataを作るとき。
- bondごとのrank上限を設計するとき。

## 処理概要

1. `validate_tt_cores`でTT構造を検証する。
2. 各coreのleft bond dimensionをsite順に取得する。
3. 最終coreのright bond dimensionを末尾に追加する。

## 主なcontract / 注意事項

- 内部bondだけでなく、先頭と末尾の境界rank 1も含める。
- Notebook内の`cores[:-1]`から内部bondだけを取り出す補助処理とは戻り値の長さが異なる。
- 入力coreを変更せず、dense化しない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_physical_shape]]、[[90_src設計/05_Core_API_v1/compression/tt_num_parameters]]、[[90_src設計/05_Core_API_v1/compression/tt_round]]
