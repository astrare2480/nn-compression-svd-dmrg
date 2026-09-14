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

- `baseline_model`: 圧縮前の基準model。各入力に対するlogitsを比較基準として使う。
- `compressed_model`: 圧縮後の比較対象model。baselineと同じ入力に対するlogitsとの差を測る。
- `loader`: logits差をdataset全体で集計する評価loader。
- `device`: 両modelの推論と入力Tensor配置に使うdevice。

## 戻り値

全sample・全logit要素に対する`baseline_logits - compressed_logits`のRMSEを表す`float`。`0.0`なら比較した全logit値が一致し、値が大きいほどbaselineからの出力値のずれが大きい。

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

- empty loaderを拒否する。
- accuracyやagreementよりも細かくlogit値の変化を見る指標である。
- 入力だけを`device`へ移す。baseline/compressed modelは[[07_src設計/05_Core_API_v1/README#共通device contract|共通device contract]]に従い、呼び出し前に同じdeviceへ配置する。

## 関連API

`agreement`, `collect_compression_metrics`
