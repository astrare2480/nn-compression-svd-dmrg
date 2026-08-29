# `get_fashion_mnist_datasets`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNISTの学習用全Datasetとtest Datasetを取得する。

## Signature

```python
get_fashion_mnist_datasets(
    data_dir: Path | str,
    *,
    transform=None,
    download: bool = True,
)
```

## 引数

- `data_dir`: Dataset保存先。
- `transform`: torchvision transform。省略時は`ToTensor()`。
- `download`: 未取得データをdownloadするか。

## 戻り値

`(full_train_dataset, test_dataset)`。

## 使用場面

Fashion-MNIST MLP/CNN実験のDataset準備。

## ざっくりした処理

必要ならdefault transform作成 → torchvision FashionMNISTのtrain/testを取得。

## 主なcontract / 注意事項

train/validation分割はこの関数では行わず`split_fashion_mnist_dataset()`へ分離する。

## 関連API

`split_fashion_mnist_dataset`, `make_fashion_mnist_loaders`
