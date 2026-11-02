# `assign_tt_linear_cores`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_linear_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

`TTLinear`のcore Parameterへ、別のTT-matrix core列の値を書き込む。biasと他の層には触らない。

## Signature

```python
assign_tt_linear_cores(
    layer: TTLinear,
    cores: Sequence[torch.Tensor],
    *,
    requires_grad: bool | None = None,
) -> None
```

## 引数

- `layer`: 書き込み先の`TTLinear`。`out_modes`と`in_modes`は変えない。
- `cores`: `layer`のmodeと一致するTT-matrix core列。
- `requires_grad`: `None`なら各coreの現在の`requires_grad`を保つ。boolならすべてのcoreをその値にする。

## 戻り値

戻り値はない。`layer.cores`の値、またはbond次元が変わる場合はそのParameter自体が更新される。

## 使用場面

- 正準化で得たcoreを、同じ層のParameterへ戻すとき。
- SVD後にbond次元が変わったcoreを、同じ`TTLinear`へ載せるとき。

## 処理概要

1. `layer`が`TTLinear`であること、core列がmodeと一致すること、dtypeとdeviceが層と一致することを確認する。
2. shapeが同じcoreは、`torch.no_grad`の中で既存Parameterへ`copy_`する。
3. shapeが違うcoreは、`detach().clone()`した新しいleaf Parameterに差し替える。

## 主なcontract / 注意事項

- これはgauge変換やSVD結果の値の代入であり、学習グラフは保持しない。
- 同じshapeの`copy_`は既存Parameterのidentityを保つ。shapeが変わるときだけParameterを差し替える。
- bias、`out_modes`、`in_modes`、他の層は変更しない。
- shapeが変わるcoreは、既存の`nn.Parameter`を新しいParameterへ差し替える。差し替え前のParameterをoptimizerの`param_groups`が参照している場合、そのoptimizerは古いParameterを見続け、新しいParameter用のstateは作られない。この関数はoptimizer stateを移行しない。shapeが変わる書き戻しのあとで同じoptimizerを使うときは、呼び出し側でoptimizerを作り直す。同じshapeの`copy_`ではParameterのidentityが保たれるので、既存optimizerの参照はそのまま有効である。

## 関連API

[[90_src設計/05_Core_API_v1/compression/TTLinear]]、[[90_src設計/05_Core_API_v1/tt_mps/canonicalize_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/replace_tt_matrix_core_pair]]、[[90_src設計/05_Core_API_v1/Internal_API]]
