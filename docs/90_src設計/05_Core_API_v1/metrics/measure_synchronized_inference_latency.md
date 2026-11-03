# `measure_synchronized_inference_latency`

**Stability:** A
**定義:** `src/nn_compression/metrics/inference_latency.py`

## 責務

eager 推論の wall-clock を、反復ごとに同期して測り、中央値と四分位を返す。

## Signature

```python
measure_synchronized_inference_latency(
    model: nn.Module,
    images: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]
```

## 引数

- `model`: 測定するモデル。`model.modules()` の各 training フラグを保存して `eval` で測り、成功時も例外時も各 module のフラグを個別に戻す。
- `images`: モデルと同じ device に既にある入力 batch。
- `warmup`: 計測前に捨てる反復回数。
- `repetitions`: 記録する反復回数。

## 戻り値

`median_seconds`、`q1_seconds`、`q3_seconds`、`repetitions`、`warmup`、`batch_size` を持つ辞書。

## 使用場面

- CPU から GPU への転送を含めず、同じ入力で複数モデルの推論時間を比べるとき。

## 処理概要

1. 入力がモデルと同じ device にあることを確認する。`.to()` は呼ばない。
2. `torch.inference_mode` で warmup する。
3. 各反復の前後で device を同期し、その区間の秒数を記録する。
4. 昇順の標本から Q1、中央値、Q3 を線形補間で返す。

## 主なcontract / 注意事項

- `torch.compile` は使わない。
- 学習時の peak memory は測らない。呼び出し側が推論の前後で別に読む。
- 平均だけの `benchmark_inference` とは集計が違う。

## 関連API

- `benchmark_inference`
