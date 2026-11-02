# `accumulate_scoped_full_batch_gradients`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

全学習 sample の平均損失と、指定した scope が更新する対象だけの勾配を、batch に分けて足す。

## Signature

```python
accumulate_scoped_full_batch_gradients(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    batch_size: int,
    device: torch.device,
    pair_index: int | None = None,
    criterion: nn.Module | None = None,
) -> tuple
```

## 引数

- `model`: 対象 `TTLinear` を含むモデル。
- `source`: `shuffle=False` かつ `drop_last=False` の DataLoader、または GPU 常駐 batch。
- `scope`: 更新対象。
- `layer_name`: 対象 `TTLinear` の名前。
- `batch_size`: 全データ走査の batch サイズ。DataLoader の batch サイズと一致させる。
- `device`: 計算に使う device。常駐 batch はすでにこの device にある必要がある。
- `pair_index`: 結合する左 site。dense-only では `None`。
- `criterion`: `reduction="mean"` の損失。`None` なら平均 cross-entropy。

## 戻り値

`(grad_T, dense_grads, full_batch_loss)`。dense-only の `grad_T` は `None`。tt-only の `dense_grads` は空 dict。損失は sample 数で重み付けした全データ平均。

## 使用場面

- DataLoader と GPU 常駐で、同じ全データ勾配の定義を使うとき。
- 固定する parameter の勾配を保存しないとき。

## 処理概要

1. scope に応じて、勾配を持たせる parameter を選ぶ。TT core は常に勾配対象から外す。
2. joint と tt-only では対象 pair を結合する。dense-only では結合しない。
3. 各 batch の平均損失 `L_b` に `(n_b / N)` を掛けて `backward` する。
4. 処理 sample 数が `N` と一致することを確認する。
5. scope が更新する勾配だけを複製して返す。

## 主なcontract / 注意事項

- 平均損失に `(n_b / N)` を掛けるので、さらに `N` で割らない。最後の batch が小さくても定義は同じ。
- tt-only でも、固定した通常 parameter を通る forward は残す。勾配は結合テンソルへ流れる。
- GPU 常駐 batch は batch ごとに CPU から転送しない。
- 有限でない損失や勾配は `NonFiniteLocalUpdate`。sample 数の不一致は `ValueError` であり、非有限へは変換しない。
- parameter は更新しない。

## 関連API

[[90_src設計/05_Core_API_v1/training/accumulate_tt_linear_pair_full_batch_gradients]]、[[90_src設計/05_Core_API_v1/training/GpuResidentTensorBatches]]、[[90_src設計/05_Core_API_v1/training/TTUpdateScope]]、[[90_src設計/05_Core_API_v1/training/NonFiniteLocalUpdate]]、[[90_src設計/05_Core_API_v1/Internal_API]]
