# `sweep_tt_linear_pairs_left_to_right`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

site 0側から、pair `(0,1), (1,2), ...` を右向きSVDで順に更新する。各pairでは`local_steps_per_pair`回、最新coreから勾配を取り直す。

## Signature

```python
sweep_tt_linear_pairs_left_to_right(
    model: nn.Module,
    candidate: nn.Module,
    loader: DataLoader,
    *,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    local_steps_per_pair: int,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    cursor: _SweepCursor,
    logs: list[dict],
    criterion: nn.Module | None = None,
    observer: object | None = None,
) -> _SweepCursor
```

## 引数

- `model`: 更新先。
- `candidate`: 候補検証用の独立モジュール。
- `loader`: 全データ勾配loader。
- `layer_name`: 対象`TTLinear`。
- `rank_cap`: bond上限。
- `tt_pair_learning_rate`: `T`の学習率。
- `dense_parameter_learning_rate`: 通常parameterの学習率。
- `local_steps_per_pair`: 1 pairに留まる更新回数。1以上。
- `diagnostic_images`: 固定診断batch。
- `diagnostic_labels`: そのラベル。
- `device`: 計算device。
- `cursor`: 中心と成功回数を保持する進行状態。この関数が更新する。
- `logs`: 成功した局所更新を追記するlist。
- `criterion`: 平均損失。`None`なら平均cross-entropy。
- `observer`: 各更新の直前に呼ばれる任意の関数。

## 戻り値

更新後の`cursor`。失敗時は例外が伝播し、それまでの成功だけが`cursor`と`model`に残る。

## 使用場面

- 1往復の前半として、中心をsite 0側から最終siteへ進めるとき。

## 処理概要

1. pair indexを`0, 1, ..., d-2`の順に取る。
2. 各pairで、指定回数だけ最新coreから局所更新する。向きは`"right"`。
3. 成功のたびに中心をpairの右siteへ置く。

## 主なcontract / 注意事項

- 毎pairでTT列全体を正準化し直さない。中心は方向付き分割で引き継ぐ。
- 2 step目以降、中心はpairの進入側ではなく、pair内の右siteにあってよい。
- 失敗した候補は成功数に含めない。

## 関連API

[[90_src設計/05_Core_API_v1/training/update_tt_linear_two_core_pair]]、[[90_src設計/05_Core_API_v1/training/sweep_tt_linear_pairs_right_to_left]]、[[90_src設計/05_Core_API_v1/training/train_tt_linear_full_batch_sweeps]]、[[90_src設計/05_Core_API_v1/Internal_API]]
