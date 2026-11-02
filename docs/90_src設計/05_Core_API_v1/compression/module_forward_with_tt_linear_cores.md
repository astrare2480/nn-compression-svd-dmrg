# `module_forward_with_tt_linear_cores`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_linear_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

モデル全体のforwardのうち、指定した`TTLinear`だけを明示したcore列で計算する。SVD後のcoreをParameterへ書き戻す前の比較に使う。

## Signature

```python
module_forward_with_tt_linear_cores(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    cores: Sequence[torch.Tensor],
) -> torch.Tensor
```

## 引数

- `model`: 対象層を含むモデル。
- `inputs`: `model.forward`へ渡す入力。
- `layer_name`: 差し替える`TTLinear`のモジュール名。空文字は不可。
- `cores`: その層の計算に使うTT-matrix core列。層のmodeとbiasはそのまま使う。

## 戻り値

`model(inputs)`と同じshapeの出力。`cores`は`detach`しない。

## 使用場面

- 再分割した2 coreを含むcore列で、モデル出力が結合テンソル経由の出力と一致するかを確認するとき。

## 処理概要

1. `layer_name`のモジュールが`TTLinear`であることを確認する。
2. その層の`forward`だけを、渡された`cores`で`tt_linear_forward`を呼ぶ関数へ一時的に差し替える。
3. `model(inputs)`を実行し、例外の有無にかかわらず元の`forward`へ戻す。

## 主なcontract / 注意事項

- dense重みは再構成しない。`tt_linear_forward`を使う。
- `torch.no_grad`では囲まない。記録用にグラフを切るかどうかは呼び出し側が決める。
- モジュールが持つcore Parameterは変更しない。
- `cores`は対象層の`out_modes`、`in_modes`と一致している必要がある。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_linear_forward]]、[[90_src設計/05_Core_API_v1/tt_mps/replace_tt_matrix_core_pair]]、[[90_src設計/05_Core_API_v1/compression/module_forward_with_merged_tt_linear_pair]]、[[90_src設計/05_Core_API_v1/Internal_API]]
