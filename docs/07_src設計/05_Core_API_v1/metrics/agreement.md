# `agreement`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelが同じ入力に対して同じ予測classを出す割合をloader全体で計算する。

## Signature

```python
agreement(
    baseline_model,
    compressed_model,
    loader,
    device,
) -> float
```

## 引数

- `baseline_model`: 圧縮前の基準model。同じ入力に対する予測classを比較の正本として使う。
- `compressed_model`: 圧縮後の比較対象model。baselineと同じ入力で推論し、予測classが一致するかを数える。
- `loader`: 比較に使う入力sampleを供給する評価loader。ground-truth labelは一致率そのものには使わない。
- `device`: 両modelへ入力を渡して推論するdevice。

## 戻り値

loader全sampleのうち、baselineとcompressedが同じ`argmax`予測classを出したsampleの割合を表す`float`。値域は`[0, 1]`で、`1.0`なら全sampleで予測classが一致する。

## 使用場面

ground-truth accuracyとは別に、「圧縮後modelがbaselineの判断をどれだけ保持したか」を測るとき。

## 処理概要

1. **両modelのroot・全submoduleのtraining状態を保存する。**
2. **両modelを一時的に`eval()`へ切り替える。**
3. **`torch.inference_mode()`でloaderを走査する。**  
   gradientを作らず推論だけを行う。
4. **各batchの入力をdeviceへ移す。**
5. **baselineとcompressedのlogitsからそれぞれ`argmax`予測classを求める。**
6. **同じclassになったsample数を加算する。**
7. **sample総数も加算する。**
8. **空loaderなら明示的に`ValueError`を送出する。**
9. **一致数 / 総sample数を返す。**
10. **正常終了・例外のどちらでも両modelの全module training状態を復元する。**

## 主なcontract / 注意事項

- empty loaderは`ValueError`。
- root + 全submoduleのtraining状態を復元する。
- 正解ラベルは一致率計算には使わない。
- 入力だけを`device`へ移す。baseline/compressed modelは[[07_src設計/05_Core_API_v1/README#共通device contract|共通device contract]]に従い、呼び出し前に同じdeviceへ配置する。

## 関連API

`logits_rmse`, `collect_compression_metrics`
