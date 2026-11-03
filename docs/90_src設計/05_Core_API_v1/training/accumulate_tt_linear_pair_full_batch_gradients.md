# `accumulate_tt_linear_pair_full_batch_gradients`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

隣接pairを結合テンソル`merged`で置き換えたforwardについて、全学習データの平均cross-entropyの勾配をbatchごとに足す。parameterも`merged`も更新しない。

## Signature

```python
accumulate_tt_linear_pair_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    layer_name: str,
    pair_index: int,
    merged: torch.Tensor,
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

## 引数

- `model`: 対象`TTLinear`を含むモデル。TT coreは凍結されている前提。
- `source`: `shuffle=False`かつ`drop_last=False`の`DataLoader`、または`GpuResidentTensorBatches`。
- `layer_name`: 対象`TTLinear`の名前。
- `pair_index`: 結合する左site。
- `merged`: `requires_grad=True`のleaf。対象pairの結合テンソル。
- `device`: batchの計算device。常駐sourceはすでにこのdeviceにあること。
- `criterion`: `reduction="mean"`の損失。`None`なら平均cross-entropy。
- `batch_size`: 常駐sourceのslice幅。`DataLoader`では`None`ならloaderのbatch sizeを使う。

## 戻り値

`(grad_T, dense_grads)`。どちらも`detach().clone()`した勾配。`dense_grads`はTT core以外で`requires_grad=True`のparameter名から勾配へのdict。

## 使用場面

- 1回の局所更新で、全データ勾配をminibatch SGDと区別して集計するとき。
- batchサイズが不揃いでも、一括計算と同じ勾配になることを確認するとき。

## 処理概要

1. `merged`とモデルparameterの勾配を消す。
2. 各batchの平均損失`L_b`に`(n_b / N)`を掛けて`backward`し、グラフはbatchごとに捨てる。
3. 処理sample数が`N`と一致することを確認する。
4. 勾配を複製して返す。

## 主なcontract / 注意事項

- 平均損失に`(n_b / N)`を掛けるので、さらに`N`で割らない。
- 最終batchのサイズが違っても、同じ重みで足す。
- 集計中にparameter値は変えない。
- 有限でない損失や勾配は`NonFiniteLocalUpdate`。sample数の不一致は`ValueError`であり、非有限へは変換しない。

## 関連API

[[90_src設計/05_Core_API_v1/training/update_tt_linear_two_core_pair]]、[[90_src設計/05_Core_API_v1/compression/module_forward_with_merged_tt_linear_pair]]、[[90_src設計/05_Core_API_v1/training/NonFiniteLocalUpdate]]、[[90_src設計/05_Core_API_v1/Internal_API]]
