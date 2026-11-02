# `measure_tt_pair_update_scales`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`  
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/01_tt_initialization_scale_diagnostics.ipynb`

## 責務

1つのTT pairについて、全データ平均損失の勾配と、更新しないままの相対更新量を返す。

## Signature

```python
measure_tt_pair_update_scales(
    model: nn.Module,
    loader: DataLoader,
    *,
    layer_name: str,
    pair_index: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float | int | bool]
```

## 引数

- `model`: 対象`TTLinear`を含むmodel。値は変えない。
- `loader`: `shuffle=False`かつ`drop_last=False`の全データloader。
- `layer_name`: 対象層。
- `pair_index`: 結合する左site。
- `tt_pair_learning_rate`: 結合テンソル`T`に対する学習率。更新には使わない。
- `dense_parameter_learning_rate`: TT core以外の通常parameterに対する学習率。更新には使わない。
- `device`: batchの移動先。
- `criterion`: 平均cross-entropy。`None`ならそれを作る。

## 戻り値

診断dict。`tt_parameter_norm`は`||T||`、`tt_gradient_norm`は`||grad_T||`、`tt_relative_update`は`tt_pair_learning_rate * ||grad_T|| / ||T||`。通常parameterは、勾配を計算したtensorを連結したL2 normとその相対更新量。`num_samples`はloaderのdataset長。

## 使用場面

- 学習を始める前に、指定pairの初期勾配と、既存学習率での相対更新量を見るとき。

## 処理概要

1. `requires_grad`と既存の`.grad`を保存する。
2. 対象pairを結合し、`accumulate_tt_linear_pair_full_batch_gradients`で全データ勾配を集計する。
3. 相対更新量を計算して返す。
4. parameter値、`requires_grad`、`.grad`を戻す。

## 主なcontract / 注意事項

- parameter更新もSVDもしない。
- 相対更新量は診断値であり、学習性能の保証ではない。
- `T`の勾配ノルムとdense重み`W`の勾配ノルムは、変数が違うので同じ尺度として比較しない。
- 分母が0で勾配も0の相対更新量は0。分母だけが0のときは`ValueError`とし、無限大は返さない。

## 関連API

[[90_src設計/05_Core_API_v1/training/accumulate_tt_linear_pair_full_batch_gradients]]、[[90_src設計/05_Core_API_v1/tt_mps/merge_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
