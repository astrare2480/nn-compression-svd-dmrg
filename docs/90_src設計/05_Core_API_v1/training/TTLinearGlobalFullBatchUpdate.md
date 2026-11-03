# `TTLinearGlobalFullBatchUpdate`

**Stability:** A
**定義:** `src/nn_compression/training/tt_linear_global_full_batch.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/11_full_batch_global_vs_local_update_control.ipynb`

## 責務

全 TT core と通常 parameter を同時に 1 歩更新した結果を保持する。

## Signature

```python
TTLinearGlobalFullBatchUpdate(
    full_batch_loss: float,
    processed_samples: int,
    parameter_norm: float,
    gradient_norm: float,
    relative_update: float,
    core_parameter_norm: float,
    core_gradient_norm: float,
    core_relative_update: float,
    dense_parameter_norm: float,
    dense_gradient_norm: float,
    dense_relative_update: float,
    core_shapes: tuple[tuple[int, ...], ...],
) -> None
```

## 引数

- `full_batch_loss`: 全 sample の平均損失。
- `processed_samples`: 集計した sample 数。
- `parameter_norm`: 更新前の全 parameter を一つの群にした L2 ノルム。
- `gradient_norm`: その群の勾配の L2 ノルム。
- `relative_update`: `learning_rate * ||g|| / ||parameter||`。
- `core_parameter_norm`: TT core だけの parameter ノルム。
- `core_gradient_norm`: TT core だけの勾配ノルム。
- `core_relative_update`: TT core だけの relative update。
- `dense_parameter_norm`: TT core 以外の parameter ノルム。
- `dense_gradient_norm`: TT core 以外の勾配ノルム。
- `dense_relative_update`: TT core 以外の relative update。
- `core_shapes`: 更新後の core shape。実効 weight の数学的最小 TT-rank ではない。

## 戻り値

インスタンス。失敗した更新では作らない。

## 使用場面

- 全 core 同時の full-batch GD が 1 歩成功したあと、損失、勾配規模、core shape を記録するとき。

## 処理概要

1. `update_tt_linear_global_full_batch` が、書き戻しに成功したあとで各値を入れる。
2. core と通常 parameter の内訳は診断であり、別の更新 step ではない。

## 主なcontract / 注意事項

- 非有限値で失敗したときは、このオブジェクトを返さず、parameter も書き戻さない。
- `core_shapes` は core の shape を保ったことの記録であり、最小 TT-rank の保証ではない。

## 関連API

- `accumulate_tt_linear_global_full_batch_gradients`
- `update_tt_linear_global_full_batch`
- `GpuResidentTensorBatches`
