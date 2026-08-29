# `factorize_named_conv2d`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_svd.py`

## 責務

model内の指定named `nn.Conv2d`だけをSVD分解し、baselineとは独立した圧縮model copyを返す。

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
- `layer_name`: `"conv2"`, `"block.conv"`などのdot path。
- `rank`: 対象ConvのSVD rank。

## 戻り値

指定Convだけが2層factorized Convへ置換されたmodel copy。

## 使用場面

1つのConvだけを圧縮してrank sweepやFine-tuningを行うとき。

## 処理概要

1. **baseline modelを`deepcopy`する。**  
   圧縮modelとbaselineのParameter共有を避ける。
2. **元modelから対象submoduleを取得する。**  
   `layer_name`をdot pathとして解決する。
3. **対象が`nn.Conv2d`か確認する。**  
   名前だけ一致して別型の層を誤って圧縮しないよう型を検証する。
4. **元Convを`factorize_conv2d_layer()`へ渡す。**  
   SVD・2層構築・spatial semantics・bias処理は層単位APIへ委譲する。
5. **copy側modelの同じpathを新Moduleへ置換する。**  
   元model側の層は変更しない。
6. **独立したcompressed modelを返す。**

## 主なcontract / 注意事項

baseline非破壊。対象Conv自体の対応範囲は`factorize_conv2d_layer()`のcontractに従う。

## 関連API

`factorize_conv2d_layer`, `get_named_module`, `set_named_module`, `factorize_named_layers`
