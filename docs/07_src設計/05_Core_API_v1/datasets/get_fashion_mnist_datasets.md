# `get_fashion_mnist_datasets`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNISTの学習用全Datasetとtest Datasetを、共通transform方針で取得する。

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

- `data_dir`: Fashion-MNISTファイルを保存・読み込みするroot directory。
- `transform`: 各画像を取り出すときに適用するtorchvision transform。`None`なら`ToTensor()`を使い、画像をTensorへ変換する。
- `download`: `data_dir`にDatasetが無い場合にtorchvisionからdownloadしてよいかを指定するフラグ。

## 戻り値

`(full_train_dataset, test_dataset)`の2要素tuple。

- `full_train_dataset`: torchvisionのFashion-MNIST train split全体。ここではまだtrain/validationへ分割しない。
- `test_dataset`: 最終評価用のFashion-MNIST test split。

両方に同じ`transform`方針が設定される。

## 使用場面

Fashion-MNIST MLP/CNN実験のDataset準備。

## 処理概要

1. **torchvisionのdatasets/transformsを関数内でimportする。**
2. **transform未指定ならdefault transformを作る。**  
   `ToTensor()`でPIL imageを`(1,28,28)` Tensorへ変換する。
3. **train split全体を`datasets.FashionMNIST(train=True)`で取得する。**
4. **test splitを`train=False`で取得する。**
5. **同じtransform設定を両Datasetへ適用する。**
6. **full trainとtestをtupleで返す。**

## 主なcontract / 注意事項

train/validation/rank-validationへの分割はこの関数では行わず、`split_fashion_mnist_dataset()`等へ分離する。

## 関連API

`split_fashion_mnist_dataset`, `make_fashion_mnist_loaders`
