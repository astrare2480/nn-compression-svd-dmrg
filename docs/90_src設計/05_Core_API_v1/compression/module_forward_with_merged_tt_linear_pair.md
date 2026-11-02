# `module_forward_with_merged_tt_linear_pair`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_linear_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

モデル全体のforwardのうち、指定した`TTLinear`の隣接pairだけを結合テンソルで計算する。活性化や後続層はモデル自身の`forward`を使う。

## Signature

```python
module_forward_with_merged_tt_linear_pair(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    pair_index: int,
    merged: torch.Tensor,
) -> torch.Tensor
```

## 引数

- `model`: 対象層を含むモデル。
- `inputs`: `model.forward`へ渡す入力。
- `layer_name`: 差し替える`TTLinear`のモジュール名。空文字は不可。
- `pair_index`: 結合した左coreのsite index。
- `merged`: そのpairの代わりに使う6階テンソル。

## 戻り値

`model(inputs)`と同じshapeの出力。`merged`から出力までのautograd graphは残る。

## 使用場面

- TT-MLPの分類損失を、対象層の結合テンソルに関して微分するとき。
- 元のcore列による出力と、結合テンソルによる出力を同じモデル経路で比べるとき。

## 処理概要

1. `layer_name`のモジュールが`TTLinear`であることを確認する。
2. その層の`forward`だけを、`tt_linear_forward_with_merged_pair`を呼ぶ関数へ一時的に差し替える。
3. `model(inputs)`を実行し、例外の有無にかかわらず元の`forward`へ戻す。

## 主なcontract / 注意事項

- dense重みは再構成しない。
- `torch.no_grad`では囲まない。
- 対象層のParameter値は変更しない。pairのcore値は縮約に使わない。
- モデル構造そのものは複製しない。`forward`属性の差し替えは呼び出しの間だけである。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_linear_forward_with_merged_pair]]、[[90_src設計/05_Core_API_v1/compression/module_forward_with_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/compression/TTLinear]]、[[90_src設計/05_Core_API_v1/Internal_API]]
