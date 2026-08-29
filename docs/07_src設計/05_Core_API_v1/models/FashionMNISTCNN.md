# `FashionMNISTCNN`

**Stability:** B  
**定義:** `src/nn_compression/models/cnn.py`

## 責務

Fashion-MNIST CNN SVD実験用のbaseline architectureを提供する。

## Signature

```python
FashionMNISTCNN()
```

## 引数

なし。

## 戻り値

`nn.Module` instance。

## 使用場面

Fashion-MNIST Conv/Linear SVD実験、MACs比較。

## ざっくりした処理

```text
Conv 1→32 → ReLU → MaxPool
→ Conv 32→64 → ReLU → MaxPool
→ Flatten 3136
→ Linear 3136→128 → ReLU
→ Linear 128→10
```

## 主なcontract / 注意事項

主要layer名を固定する。

```text
conv1, conv2, fc1, fc2
```

`forward()`は通常のPyTorch呼出し`model(x)`で使用する。

## 関連API

`FashionMNISTCNN.inspect_shapes`, `estimate_cnn_macs`
