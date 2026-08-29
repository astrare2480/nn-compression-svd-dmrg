# `factorize_named_linear`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

model内の指定named LinearだけをSVD 2層分解したmodel copyを作る。

## Signature

```python
factorize_named_linear(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module
```

## 引数

- `model`: baseline model。
- `layer_name`: `fc1`, `head.fc` 等のdot path。
- `rank`: 分解rank。

## 戻り値

対象layerだけを置換したdeepcopy model。

## 使用場面

特定Linearだけを圧縮してbaselineと比較するとき。

## ざっくりした処理

```text
modelをdeepcopy
→ 元modelからlayer_nameを取得
→ Linear型を確認
→ factorize_linear_layer
→ copy側の同pathへ置換
```

## 主なcontract / 注意事項

baseline modelとcompressed modelでParameterを共有しない。

## 関連API

`factorize_linear_layer`, `get_named_module`, `set_named_module`
