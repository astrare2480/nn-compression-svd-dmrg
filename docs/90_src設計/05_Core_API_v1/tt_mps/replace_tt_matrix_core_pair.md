# `replace_tt_matrix_core_pair`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

TT-matrix core列の隣接pairだけを差し替えた新しいlistを返す。モジュールのParameterは変更しない。

## Signature

```python
replace_tt_matrix_core_pair(
    cores: Sequence[torch.Tensor],
    pair_index: int,
    left_core: torch.Tensor,
    right_core: torch.Tensor,
) -> list[torch.Tensor]
```

## 引数

- `cores`: 元のTT-matrix core列。listも各Tensorも変更しない。
- `pair_index`: 差し替える左coreのsite index。
- `left_core`: 新しい左core。外向きのbondが左隣と一致する必要がある。
- `right_core`: 新しい右core。外向きのbondが右隣と一致する必要がある。

## 戻り値

長さが`len(cores)`の新しいlist。pair以外の要素は元のTensorを参照する。構造はmodeなしの`validate_tt_matrix_cores`を満たす。

## 使用場面

- SVDで分け直した2 coreを、層へ書き戻す前にforwardやdense診断へ渡すとき。

## 処理概要

1. `pair_index`が隣接pairであることを確認する。
2. 指定位置の参照だけを新しいcoreへ差し替えたlistを作る。
3. bond、境界rank、dtype、deviceが列として一致することを確認する。

## 主なcontract / 注意事項

- pair以外のcoreはコピーも`detach`もしない。学習グラフが付いているcoreを、意図せず切らない。
- この関数は`nn.Module`へ代入しない。モジュールへ書くときは`assign_tt_linear_cores`を使う。

## 関連API

[[90_src設計/05_Core_API_v1/compression/assign_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/compression/module_forward_with_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
