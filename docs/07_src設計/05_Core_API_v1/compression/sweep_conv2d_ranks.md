# `sweep_conv2d_ranks`

**Stability:** B  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

1つのnamed Conv2dについて、SVD rank候補を評価するための`factorize_conv2d_layer`＋MACs付きwrapper。

## Signature

```python
sweep_conv2d_ranks(
    model,
    layer_name,
    ranks,
    out_hw,
    loader,
    criterion,
    device,
    *,
    baseline_acc,
    baseline_time_s,
    warmup=5,
    repeats=200,
    verbose=True,
    input_batch=None,
) -> list[dict]
```

## 引数

`model`, 対象Conv名、rank列、対象Convの出力空間`out_hw`、評価/benchmark条件。

## 戻り値

rank候補ごとのmetrics record list。

## 使用場面

Conv SVDのrank sweepをNotebookから簡潔に実行するとき。

## 処理概要

1. **Conv SVD用のMACs callbackを内部で定義する。**  
   元Convとrankを受け取り、`estimate_conv2d_macs()`でcompressed MACsと削減率を返す形にする。
2. **generic `sweep_layer_ranks()`へ処理を委譲する。**
3. **factorize callbackとして`factorize_conv2d_layer`を渡す。**  
   各rankで同じConv SVD実装を使う。
4. **対象Convの`out_hw`をMACs計算へ渡す。**  
   理論計算量が実際のfeature map sizeと対応するようにする。
5. **generic sweepが返したrecord listをそのまま返す。**

## 主なcontract / 注意事項

`out_hw`は対象Convの実際の出力空間sizeを呼び出し側が与える。

## 関連API

`sweep_layer_ranks`, `factorize_conv2d_layer`, `estimate_conv2d_macs`
