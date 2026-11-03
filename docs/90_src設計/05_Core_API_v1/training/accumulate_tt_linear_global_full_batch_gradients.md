# `accumulate_tt_linear_global_full_batch_gradients`

**Stability:** A
**定義:** `src/nn_compression/training/tt_linear_global_full_batch.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/11_full_batch_global_vs_local_update_control.ipynb`

## 責務

全 sample の平均 cross-entropy と、全 TT core・通常 parameter の勾配を、parameter を変えずに返す。

## Signature

```python
accumulate_tt_linear_global_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    batch_size: int | None,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[dict[str, torch.Tensor], float, int]
```

## 引数

- `model`: 勾配を取るモデル。集計の前後で値は変えない。
- `source`: `shuffle=False` かつ `drop_last=False` の DataLoader、または `GpuResidentTensorBatches`。
- `layer_name`: 対象の `TTLinear`。
- `batch_size`: 常駐 source の slice 幅。DataLoader では loader の batch size と一致させる。
- `device`: 計算 device。常駐 source の device と一致している必要がある。
- `criterion`: 平均 cross-entropy。`None` ならそれを作る。

## 戻り値

`(grads, full_batch_loss, processed_samples)`。`grads` は parameter 名から detach した勾配。`full_batch_loss` は全 sample の平均損失。`processed_samples` は全 sample 数と一致する。

## 使用場面

- 全 core 同時 GD の 1 歩で、書き戻す前に全データ平均勾配を確認するとき。

## 処理概要

1. すべての named parameter について、集計中だけ `requires_grad` を有効にする。
2. batch 平均損失 `L_b`、batch の sample 数 `n_b`、全 sample 数 `N` に対し、`(n_b / N) * L_b` の勾配を足す。`L_b` は平均なので、さらに `N` では割らない。
3. 最後の chunk が短くても残りの sample を含め、欠落なく `N` 件を集計する。
4. 勾配を clone し、`requires_grad` と既存の `.grad` を戻す。

## 主なcontract / 注意事項

- 全 chunk の集計が終わるまで parameter の値は変えない。SVD、再正準化、pair 巡回はしない。
- 非有限な損失、logits、勾配では `NonFiniteLocalUpdate` を上げ、値は戻したままにする。
- `GpuResidentTensorBatches` は、事前配置した device と `device` が一致するときだけ使える。
- core shape はこの関数では変わらない。数学的最小 TT-rank は計算しない。

## 関連API

- `update_tt_linear_global_full_batch`
- `TTLinearGlobalFullBatchUpdate`
- `GpuResidentTensorBatches`
- `NonFiniteLocalUpdate`
