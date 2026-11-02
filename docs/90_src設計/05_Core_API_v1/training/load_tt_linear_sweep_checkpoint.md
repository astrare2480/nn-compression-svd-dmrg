# `load_tt_linear_sweep_checkpoint`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

`save_tt_linear_full_batch_sweep_run`が書いたcheckpointを、別のモデルインスタンスへ読み、通常の`forward`で評価できる状態にする。

## Signature

```python
load_tt_linear_sweep_checkpoint(
    path: Path | str,
    model: nn.Module,
) -> nn.Module
```

## 引数

- `path`: `initial_model.pt`、`final_model.pt`、`best_model.pt`のいずれか。
- `model`: 同じ`layer_name`の`TTLinear`を持つ読み込み先。coreのbondが違ってもよい。

## 戻り値

値を書き込んだ`model`。

## 使用場面

- 保存したbestを、学習中の最終モデルとは別のインスタンスで再評価するとき。

## 処理概要

1. checkpointからcore列を取り、`assign_tt_linear_cores`でbondが変わっても差し替える。
2. それ以外のparameterは、shapeとdtypeが一致するときだけ`copy_`する。

## 主なcontract / 注意事項

- core以外のshapeが違う場合は`ValueError`で、非有限扱いにはしない。
- 読み込み後の指標の一致は、同じdevice・同じ評価関数で確認する。

## 関連API

[[90_src設計/05_Core_API_v1/training/save_tt_linear_full_batch_sweep_run]]、[[90_src設計/05_Core_API_v1/compression/assign_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
