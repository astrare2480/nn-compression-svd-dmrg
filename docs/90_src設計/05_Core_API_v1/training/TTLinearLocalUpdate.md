# `TTLinearLocalUpdate`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

成功した1回の隣接2-core局所更新の記録を保持する。失敗した候補はこのオブジェクトにしない。

## Signature

```python
TTLinearLocalUpdate(
    pair_index: int,
    direction: str,
    local_step: int,
    center: int,
    retained_rank: int,
    discarded_squared_norm: float,
    diagnostic_batch_loss: float,
    tt_parameter_norm: float,
    tt_gradient_norm: float,
    tt_relative_update: float,
    dense_parameter_norm: float,
    dense_gradient_norm: float,
    dense_relative_update: float,
    bond_rank_decision: BondRankDecision | None = None,
) -> None
```

## 引数

- `pair_index`: 更新した左site。
- `direction`: `"right"`または`"left"`。
- `local_step`: そのpair内の0始まりの局所step。
- `center`: 反映後の直交中心。右向きなら`pair_index + 1`、左向きなら`pair_index`。
- `retained_rank`: SVDで保持したbond次元。
- `discarded_squared_norm`: 捨てた特異値の二乗和。
- `diagnostic_batch_loss`: 固定診断batch上の候補loss。全学習データの有限性は表さない。
- `tt_parameter_norm`: 更新前の結合テンソルの L2 norm `||T||`。
- `tt_gradient_norm`: `||grad_T||`。
- `tt_relative_update`: `tt_pair_learning_rate * ||grad_T|| / ||T||`。診断値であり、受理条件ではない。
- `dense_parameter_norm`: 更新対象の通常parameterを連結した L2 norm。この実験では `fc1.bias`、`fc2`、`fc3`。
- `dense_gradient_norm`: その通常parameterの勾配を同じ順に連結した L2 norm。
- `dense_relative_update`: `dense_parameter_learning_rate * dense_gradient_norm / dense_parameter_norm`。診断値であり、受理条件ではない。
- `bond_rank_decision`: 閾値でrankを選んだときの`BondRankDecision`。固定rank分割では`None`。

## 戻り値

上記フィールドを持つfrozen dataclass。

## 使用場面

- `update_tt_linear_two_core_pair`が実モデルへ反映した直後の記録。

## 処理概要

1. 成功時の位置、rank、切り捨て量、診断lossをまとめる。
2. 非有限やSVD失敗では生成しない。

## 主なcontract / 注意事項

- `diagnostic_batch_loss`は固定batchの値であり、全データ損失ではない。
- 損失が下がったことは、この記録の成立条件ではない。
- 相対更新量は診断値である。候補の受理や停止には使わない。
- 分母の parameter norm が 0 で勾配 norm も 0 のとき、相対更新量は 0。分母だけが 0 のときは `ValueError` とし、NaN や無限大は記録しない。

## 関連API

[[90_src設計/05_Core_API_v1/training/update_tt_linear_two_core_pair]]、[[90_src設計/05_Core_API_v1/training/NonFiniteLocalUpdate]]、[[90_src設計/05_Core_API_v1/Internal_API]]
