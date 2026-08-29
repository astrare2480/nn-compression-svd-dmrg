# `make_torch_generator`

**Stability:** B  
**定義:** `src/nn_compression/utils/seed.py`

## 責務

DataLoader shuffleやrandom_split用のseed固定`torch.Generator`を作る。

## Signature

```python
make_torch_generator(
    seed: int = 0,
) -> torch.Generator
```

## 引数

`seed`: Generator seed。

## 戻り値

`manual_seed(seed)`済み`torch.Generator`。

## 使用場面

candidate間でsplitやmini-batch順を再現したいとき。

## ざっくりした処理

Generator生成 → `manual_seed(seed)` → return。

## 主なcontract / 注意事項

`next(iter(loader))`等でshuffle付きloaderを先読みするとGenerator stateを消費する。

## 関連API

`set_seed`, `take_inference_batch`, `split_fashion_mnist_dataset`
