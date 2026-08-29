# `logits_rmse`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelのlogits全要素についてRMSE（root mean squared error）を計算する。

## Signature

```python
logits_rmse(
    baseline_model,
    compressed_model,
    loader,
    device,
) -> float
```

## 引数

2モデル、評価loader、device。

## 戻り値

全sample・全logit要素をまとめたRMSE。

## 使用場面

argmaxが同じかだけでは分からない出力分布のずれを、baselineとの差として測りたいとき。

## 処理概要

1. **両modelのtraining状態を保存し、eval modeへ切り替える。**
2. **`torch.inference_mode()`でloaderを走査する。**
3. **各batchをdeviceへ移す。**
4. **baseline logitsとcompressed logitsを計算する。**
5. **要素ごとの差`baseline - compressed`を求め、二乗和を累積する。**
6. **logitsの総要素数も累積する。**  
   class数を固定値として仮定せず、実際の`difference.numel()`を使う。
7. **要素数が0ならempty loaderとして`ValueError`を送出する。**
8. **二乗誤差平均の平方根を取る。**
9. **両modelのtraining状態を復元してRMSEを返す。**

## 主なcontract / 注意事項

empty loader拒否。accuracyやagreementよりも細かくlogit値の変化を見る指標。

## 関連API

`agreement`, `collect_compression_metrics`
