# `benchmark_inference_print`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelを同じinput batch・同じbenchmark条件で連続計測し、必要なら比較結果を表示するconvenience API。

## Signature

```python
benchmark_inference_print(
    baseline_model,
    compressed_model,
    data_loader,
    device,
    verbose=True,
    *,
    warmup=10,
    repeats=5000,
    input_batch=None,
)
```

## 引数

2モデル、batch取得元、device、表示有無、benchmark条件。

## 戻り値

`(baseline_time_s, compressed_time_s)`。

## 使用場面

Notebook上で圧縮前後latencyを同条件で簡単に比較したいとき。

## 処理概要

1. **両modelのtraining状態を保存する。**
2. **共通input batchを1つ決める。**  
   `input_batch`が未指定なら`take_inference_batch(data_loader)`を1回だけ呼ぶ。
3. **baseline modelを`benchmark_inference()`で計測する。**  
   共通input、warmup、repeatsを渡し、詳細dictを受け取る。
4. **compressed modelもまったく同じ条件で計測する。**
5. **`verbose=True`ならbatch size・shape・warmup/repeats・両時間を表示する。**
6. **秒単位の2つの平均時間をtupleで返す。**
7. **処理前の両modelのtraining状態を復元する。**

## 主なcontract / 注意事項

入力差をlatency差へ混ぜないため、両modelへ同一Tensorを渡す。

## 関連API

`benchmark_inference`, `take_inference_batch`
