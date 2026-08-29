# `factorize_named_conv2d`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_svd.py`

## 責務

model内の指定Conv2dだけをSVD分解したmodel copyを返す。

## Signature

```python
factorize_named_conv2d(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module
```

## 引数

- `model`: baseline model。
- `layer_name`: `conv2`, `block.conv`等。
- `rank`: 分解rank。

## 戻り値

対象Convだけを置換したdeepcopy model。

## 使用場面

conv1/conv2等を1層ずつrank sweep・比較するとき。

## ざっくりした処理

```text
modelをdeepcopy
→ 元modelから対象Conv取得
→ 型確認
→ factorize_conv2d_layer
→ copy側へ置換
```

## 主なcontract / 注意事項

元modelを変更しない。

## 関連API

`factorize_conv2d_layer`, `factorize_named_layers`
