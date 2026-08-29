# `factorize_named_layers`

**Stability:** A  
**定義:** `src/nn_compression/compression/named_layers.py`

## 責務

複数のnamed Conv2d/Linearを、指定rankでまとめてSVD圧縮したmodel copyを作る。

## Signature

```python
factorize_named_layers(
    model: nn.Module,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
) -> nn.Module
```

## 引数

- `model`: baseline model。
- `conv_ranks`: `{layer_name: rank}`。
- `linear_ranks`: `{layer_name: rank}`。

## 戻り値

指定した複数layerを置換したdeepcopy model。

## 使用場面

CIFAR-10等で複数Conv/Linearを同時圧縮するとき。

## ざっくりした処理

```text
modelを1回deepcopy
→ 各Conv名を元modelから取得して分解
→ copy側へ置換
→ 各Linearも同様に置換
```

## 主なcontract / 注意事項

各分解元は**圧縮途中のcopyではなく未分解baseline model**から取得する。

## 関連API

`factorize_conv2d_layer`, `factorize_linear_layer`, `get_named_module`, `set_named_module`
