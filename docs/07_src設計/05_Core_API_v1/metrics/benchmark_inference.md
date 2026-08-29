# `benchmark_inference`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

固定した1 batchに対するmodel forwardの平均推論時間を測定する。

## Signature

```python
benchmark_inference(
    model,
    data_loader=None,
    device=None,
    warmup=10,
    repeats=5000,
    *,
    input_batch=None,
    return_details: bool = False,
)
```

## 引数

- `model`: 計測対象。
- `data_loader`: `input_batch`未指定時の取得元。
- `device`: 必須。
- `warmup`: 計測前forward回数。0以上。
- `repeats`: 計測forward回数。1以上。
- `input_batch`: 推奨される固定入力。
- `return_details`: 詳細dictを返すか。

## 戻り値

- 既定: 1 batch平均秒数`float`。
- 詳細時: `time_s/time_ms/batch_size/input_shape/warmup/repeats`を含むdict。

## 使用場面

理論MACsとは別にwall-clock latencyを比較するとき。

## ざっくりした処理

```text
引数validation
→ input batch決定
→ model状態保存
→ eval
→ warmup
→ CUDAなら同期
→ repeats回forwardを計時
→ CUDA同期
→ 平均時間計算
→ 状態復元
```

## 主なcontract / 注意事項

- `warmup/repeats`はbool/Tensor scalar不可。
- baseline/compressed比較では同じ`input_batch`を共有する。
- timingはhardware/frameworkに依存し、MACs削減を直接意味しない。

## 関連API

`take_inference_batch`, `benchmark_inference_print`, `collect_compression_metrics`
