# `sweep_conv_svd_ranks`

**Stability:** C  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

旧Notebookの`rank_list`引数名を維持し、primary API `sweep_conv2d_ranks()`へ委譲するcompatibility wrapper。

## Signature

```python
sweep_conv_svd_ranks(
    model,
    layer_name,
    rank_list,
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

## 引数 / 戻り値

意味は`sweep_conv2d_ranks()`と同じ。`rank_list`だけhistorical名。

## 使用場面

既存Notebookをそのまま再実行するとき。

## ざっくりした処理

引数名を読み替えて`sweep_conv2d_ranks()`を呼ぶだけ。

## 主なcontract / 注意事項

新規コードでは`sweep_conv2d_ranks()`を使用する。

## 関連API

`sweep_conv2d_ranks`, `Compatibility_API`
