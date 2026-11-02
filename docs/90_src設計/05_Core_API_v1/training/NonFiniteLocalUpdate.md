# `NonFiniteLocalUpdate`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

全データ勾配の局所更新で、損失・勾配・更新候補・SVD結果・診断batchの出力が有限でないことを表す。この例外は実モデルへ候補を書く前に上げる。

## Signature

```python
NonFiniteLocalUpdate(
    message: str,
) -> None
```

## 引数

- `message`: どの量が有限でなかったかの説明。

## 戻り値

例外インスタンス。正常な更新結果は返さない。

## 使用場面

- batch損失、`T`の勾配、通常parameterの勾配、更新候補、SVD結果、固定診断batchのforwardが有限でないとき。

## 処理概要

1. `RuntimeError`としてメッセージを保持する。
2. 呼び出し側は、この例外を`svd_failure`や`shape_or_assertion`と別の停止理由にする。

## 主なcontract / 注意事項

- shape不整合、`assert`、SVDの`LinAlgError`は、この例外へ変換しない。
- 例外を上げた時点では、実モデルのparameter値は更新していない。

## 関連API

[[90_src設計/05_Core_API_v1/training/update_tt_linear_two_core_pair]]、[[90_src設計/05_Core_API_v1/training/train_tt_linear_full_batch_sweeps]]、[[90_src設計/05_Core_API_v1/Internal_API]]
