# `sweep_layer_ranks`

**Stability:** B  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

任意named layerを複数rankで置換し、共通validation/benchmark metricsを収集するgeneric rank sweep。

## Signature

```python
sweep_layer_ranks(
    model,
    layer_name,
    ranks,
    loader,
    criterion,
    device,
    *,
    factorize,
    baseline_acc,
    baseline_time_s,
    macs_fn=None,
    warmup=5,
    repeats=200,
    verbose=True,
    input_batch=None,
) -> list[dict]
```

## 引数

- `factorize(original_layer, rank)`: 置換Moduleを返すcallback。
- `macs_fn`: optionalな理論MACs callback。
- その他はbaseline model・評価条件・rank候補。

## 戻り値

rankごとのmetrics record `list[dict]`。

## 使用場面

新しいfactorization方式でも同じ候補評価枠組みを再利用したいとき。

## ざっくりした処理

```text
元named layer取得
→ 各rankでmodel deepcopy
→ factorize callbackで1層置換
→ collect_compression_metrics
→ layer/rank/retained_energy付加
```

## 主なcontract / 注意事項

- 各candidateは同じbaselineから作る。
- model本体は既定でrecordへ保持しない。
- loaderは`collect_compression_metrics()`の制約上re-iterable前提。

## 関連API

`sweep_conv2d_ranks`, `collect_compression_metrics`, `retained_energy`
