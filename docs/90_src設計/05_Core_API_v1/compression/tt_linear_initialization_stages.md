# `tt_linear_initialization_stages`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_linear.py`  
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/01_tt_initialization_scale_diagnostics.ipynb`

## 責務

同じdense初期値から、TT化前、TT-SVDとrank上限の直後、site中心の正準化後、の3つの独立したmodelを返す。

## Signature

```python
tt_linear_initialization_stages(
    model: nn.Module,
    layer_name: str,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
    canonical_center: int = 0,
) -> tuple[nn.Module, nn.Module, nn.Module]
```

## 引数

- `model`: 変換元。対象層は`nn.Linear`。このmodelは変更しない。
- `layer_name`: 置換する層の名前。
- `out_modes`: 出力mode。積は対象層の`out_features`と一致する。
- `in_modes`: 入力mode。積は対象層の`in_features`と一致する。
- `max_rank`: TT-SVDのbond上限。`None`なら打ち切りなし。TT-SVDとこの上限は`factorize_named_tt_linear`の中で一度に適用する。
- `canonical_center`: 段階Cの直交中心。既定は0。

## 戻り値

`(stage_a, stage_b, stage_c)`。Aはdenseのままのdeepcopy、Bは正準化前のTT model、CはBをdeepcopyしてから正準coreを書き戻したmodel。3つはstorageを共有しない。

## 使用場面

- 初期化のどの段階で実効重みや出力が変わるかを、同じ初期値から比べるとき。

## 処理概要

1. `model`をdeepcopyしてAにする。
2. Aを`factorize_named_tt_linear`してBにする。ここでは正準化しない。
3. Bをdeepcopyし、`canonicalize_tt_matrix_cores`の結果をCの対象層へ書き戻す。

## 主なcontract / 注意事項

- 対象層以外のparameterとbuffer、および対象層のbiasはAからコピーした値である。
- Cを作るときBのcoreは上書きしない。
- 正準化はgauge変換であり、実効重みの一致は丸めの範囲で別途確認する。

## 関連API

[[90_src設計/05_Core_API_v1/compression/factorize_named_tt_linear]]、[[90_src設計/05_Core_API_v1/tt_mps/canonicalize_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/compression/assign_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
