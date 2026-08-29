# `rebuild_linear_from_svd`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

truncated SVD成分から、元layerと同じ入出力shapeの単一`nn.Linear`を再構築する。

## Signature

```python
rebuild_linear_from_svd(
    U_r,
    S_r,
    Vh_r,
    layer: nn.Linear,
) -> nn.Linear
```

## 引数

- `U_r`, `S_r`, `Vh_r`: truncated SVD成分。
- `layer`: shape・bias・device/dtype・trainabilityの基準となる元Linear。

## 戻り値

元と同じ `in_features/out_features` の新しい`nn.Linear`。

## 使用場面

パラメータ削減をせず、「低rank近似weightに置き換えたら精度がどう変わるか」を確認するとき。

## ざっくりした処理

```text
U_r @ diag(S_r) @ Vh_r
→ 元と同shapeの新Linear作成
→ weight/biasをcopy
→ requires_gradを継承
```

## 主なcontract / 注意事項

- 新しいlayerを返し、入力layerは変更しない。
- 元biasがある場合のみbiasを持つ。
- device/dtype/requires_gradを維持。

## 関連API

`truncated_svd`, `factorize_linear_layer`, `make_one_layer_svd_model`
