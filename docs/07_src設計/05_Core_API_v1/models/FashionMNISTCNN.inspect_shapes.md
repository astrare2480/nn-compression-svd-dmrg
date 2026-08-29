# `FashionMNISTCNN.inspect_shapes`

**Stability:** B  
**定義:** `src/nn_compression/models/cnn.py`

## 責務

FashionMNISTCNNの各主要層の出力shapeを表示し、Flatten後の特徴数を返す。

## Signature

```python
FashionMNISTCNN.inspect_shapes(
    x: torch.Tensor | None = None,
) -> int
```

## 引数

- `x`: optionalな1 batch入力。省略時はCPU上の`(1,1,28,28)` zero Tensor。

## 戻り値

Flatten後のfeature数。default architectureでは3136。

## 使用場面

Linear `in_features`を決める前のshape確認、学習用Notebookで層ごとのshapeを追うとき。

## ざっくりした処理

forwardと同じConv/Pool/Flattenまでを順に通し、各shapeをprintする。

## 主なcontract / 注意事項

- train_loaderを消費せずshape確認できる補助method。
- modelをGPU等へ移した後に使う場合、default dummyはCPUなので、modelと同じdeviceの`x`を明示して渡す。

## 関連API

`FashionMNISTCNN`
