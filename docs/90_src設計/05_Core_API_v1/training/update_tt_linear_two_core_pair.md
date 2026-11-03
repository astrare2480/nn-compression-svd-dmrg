# `update_tt_linear_two_core_pair`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

1つの隣接pairについて、全データ勾配から`T`と通常parameterの更新候補を作り、方向付きSVDと診断batchの確認が通ったときだけ実モデルへ反映する。

## Signature

```python
update_tt_linear_two_core_pair(
    model: nn.Module,
    candidate: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    *,
    layer_name: str,
    pair_index: int,
    direction: str,
    center: int,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    local_step: int = 0,
    criterion: nn.Module | None = None,
    observer=None,
    batch_size: int | None = None,
    discarded_weight_threshold: float | None = None,
    min_bond_rank: int = 1,
) -> TTLinearLocalUpdate
```

## 引数

- `model`: 更新先。成功するまで値は変えない。
- `candidate`: 検証専用の別モジュール。実モデルとstorageを共有してはならない。
- `source`: 全データ勾配を集計する`shuffle=False`の`DataLoader`、または`GpuResidentTensorBatches`。
- `layer_name`: 対象`TTLinear`。
- `pair_index`: 左site。
- `direction`: `"right"`または`"left"`。
- `center`: 現在の直交中心。このpairの2 siteのどちらか。進入側でなくてよい。
- `rank_cap`: SVDで保持するbondの上限。1以上。
- `tt_pair_learning_rate`: `T`だけの学習率。
- `dense_parameter_learning_rate`: TT core以外のparameterの学習率。`T`の学習率とは別。
- `diagnostic_images`: 固定診断batchの入力。loaderの先頭を消費して作らない。
- `diagnostic_labels`: そのラベル。
- `device`: 計算device。
- `local_step`: pair内の0始まりのstep。既定は0。
- `criterion`: 平均cross-entropy。`None`ならそれを作る。
- `observer`: 更新前のpair coreを受け取る任意の関数。`None`なら呼ばない。
- `batch_size`: 常駐sourceのslice幅。`DataLoader`では省略できる。
- `discarded_weight_threshold`: `None`なら`rank_cap`による固定rank分割のまま。有限の0以上の値なら、相対discarded weightがその値以下になる最小rankを`min_bond_rank`と`rank_cap`の間で選ぶ。
- `min_bond_rank`: 閾値選択の下限。既定は1。`discarded_weight_threshold`が`None`のときは使わない。

## 戻り値

反映に成功した`TTLinearLocalUpdate`。失敗時は例外で、このオブジェクトは返さない。

## 使用場面

- 左右スイープの各pairで、全データ勾配の1歩と方向付き再分割を行うとき。

## 処理概要

1. 最新coreから独立なleaf `T`を作り、全データ勾配を集計する。この間、値は変えない。
2. `T - tt_pair_learning_rate * grad_T`と、通常parameterの`parameter - dense_parameter_learning_rate * grad`を候補にする。
3. 候補`T`をdetachしてから方向付きSVDする。`discarded_weight_threshold`が`None`なら`rank_cap`までの固定rank、指定されていればdiscarded weightによるrank選択を行う。SVDは逆伝播経路に入れない。
4. 候補を`candidate`だけに載せて、診断batchのforwardとlossが有限かを見る。
5. すべて有限なら`candidate`の値を`model`へコピーする。右向きのあと中心は`pair_index + 1`、左向きのあと中心は`pair_index`。

## 主なcontract / 注意事項

- 通常parameterだけを先に実モデルへ書いてからSVDに失敗する、という状態は作らない。
- 診断batchの有限性は、そのbatchの確認であり、全学習データの有限性ではない。
- 損失減少は受理条件ではない。
- 非有限は`NonFiniteLocalUpdate`。不正な`direction`やpair外の`center`は`ValueError`。SVDの`LinAlgError`はそのまま伝播する。

## 関連API

[[90_src設計/05_Core_API_v1/training/accumulate_tt_linear_pair_full_batch_gradients]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/compression/assign_tt_linear_cores]]、[[90_src設計/05_Core_API_v1/training/TTLinearLocalUpdate]]、[[90_src設計/05_Core_API_v1/Internal_API]]
