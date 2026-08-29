# `FashionMNISTCNN.inspect_shapes`

**Stability:** B  
**定義:** `src/nn_compression/models/cnn.py`

## 責務

FashionMNISTCNNのConv/Pool/Flattenまでの各出力shapeを表示し、Flatten後のfeature数を返す学習・確認用method。

## Signature

```python
FashionMNISTCNN.inspect_shapes(
    x: torch.Tensor | None = None,
) -> int
```

## 引数

- `x`: optionalな1 batch入力。省略時はCPU上に`torch.zeros(1,1,28,28)`を作る。

## 戻り値

Flatten後のfeature数。default architecture・28×28入力では3136。

## 使用場面

Linear `in_features`を決める前のshape確認、CNN各段で空間サイズ・channel数がどう変わるかをNotebookで追うとき。

## 処理の流れ（日本語）

1. **入力`x`が省略されているか確認する。**  
   `None`ならダミー入力`(1,1,28,28)`のzero TensorをCPU上に作る。
2. **入力shapeを表示する。**
3. **`conv1`を通し、その出力shapeを表示する。**
4. **`relu1 → pool1`を通し、pool後shapeを表示する。**
5. **`conv2`を通し、その出力shapeを表示する。**
6. **`relu2 → pool2`を通し、pool後shapeを表示する。**
7. **`flatten`を通し、flatten後shapeを表示する。**
8. **`x.shape[1]`をFlatten後feature数として取得する。**
9. **feature数も表示し、整数として返す。**

### 処理フロー（短縮版）

```text
x（省略時はCPU dummy）
→ input shape表示
→ conv1 shape表示
→ pool1 shape表示
→ conv2 shape表示
→ pool2 shape表示
→ flatten shape表示
→ flatten_dimを返す
```

## 主なcontract / 注意事項

- train_loaderを消費せずshape確認できる補助method。
- modelをGPU等へ移した後に使う場合、default dummyはCPUなのでdevice不一致になる。**modelと同じdeviceの`x`を明示して渡す。**
- forward全体ではなく、Linearへ入る直前までのshape確認が主目的。

## 関連API

`FashionMNISTCNN`
