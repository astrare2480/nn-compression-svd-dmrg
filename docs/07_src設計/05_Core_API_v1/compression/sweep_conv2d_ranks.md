# `sweep_conv2d_ranks`

**Stability:** B  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

Conv2d SVD用に、factorizationとMACs計算を組み合わせたrank sweepを実行する。

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

- `out_hw`: 対象Convの出力空間 `(H, W)`。
- その他はrank候補・評価条件。

## 戻り値

rankごとの評価record list。

## 使用場面

Conv SVDのrank候補比較。

## ざっくりした処理

`estimate_conv2d_macs`用callbackを作り、`sweep_layer_ranks()`へ処理を委譲する。

## 主なcontract / 注意事項

出力空間sizeは自動推定せず呼び出し側が与える。

## 関連API

`sweep_layer_ranks`, `factorize_conv2d_layer`, `estimate_conv2d_macs`
