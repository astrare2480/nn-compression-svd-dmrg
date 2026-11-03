# `update_tt_linear_global_full_batch`

**Stability:** A
**定義:** `src/nn_compression/training/tt_linear_global_full_batch.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/11_full_batch_global_vs_local_update_control.ipynb`

## 責務

全 TT core と通常 parameter を、全 sample 平均勾配による単純 GD で同時に 1 歩更新する。

## Signature

```python
update_tt_linear_global_full_batch(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    learning_rate: float,
    batch_size: int | None,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> TTLinearGlobalFullBatchUpdate
```

## 引数

- `model`: 更新先。候補がすべて有限になるまで値は変えない。
- `source`: `shuffle=False` かつ `drop_last=False` の DataLoader、または `GpuResidentTensorBatches`。
- `layer_name`: 対象の `TTLinear`。
- `learning_rate`: 0 以上の有限値。0 は値を変えない診断用の歩幅であり、parameter の凍結ではない。
- `batch_size`: 常駐 source の slice 幅。DataLoader では loader の batch size と一致させる。
- `device`: 計算 device。
- `criterion`: 平均 cross-entropy。`None` ならそれを作る。

## 戻り値

書き戻しに成功した `TTLinearGlobalFullBatchUpdate`。

## 使用場面

- TT core を結合せず、全 core と通常 parameter を同じ全データ勾配で同時に 1 歩動かすとき。

## 処理概要

1. `accumulate_tt_linear_global_full_batch_gradients` で全 parameter の平均勾配を取る。この間、値は変えない。
2. 各 parameter について `parameter - learning_rate * gradient` を候補にする。
3. 候補の値と shape がすべて有限かつ元と一致するときだけ、まとめて書き戻す。
4. core shape は更新前と同じであることを確認して返す。

## 主なcontract / 注意事項

- 更新対象は、その `TTLinear` の全 core と、core 以外の通常 parameter である。どちらかが無いモデルは対象外。
- momentum、Adam、weight decay、SVD、再正準化、pair 巡回は行わない。
- core の shape は維持する。実効 weight の数学的最小 TT-rank が不変であるとは保証しない。
- 損失、勾配、候補値、書き戻し結果のいずれかが非有限なら `NonFiniteLocalUpdate` を上げ、部分的には書き戻さない。
- 呼び出し前の `requires_grad` が `False` でも、集計中だけ勾配を有効にして更新できる。
- `GpuResidentTensorBatches` を source にできる。device は常駐済みの device と一致させる。

## 関連API

- `accumulate_tt_linear_global_full_batch_gradients`
- `TTLinearGlobalFullBatchUpdate`
- `GpuResidentTensorBatches`
- `NonFiniteLocalUpdate`
