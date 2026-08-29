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

2モデル、評価loader、device。

## 戻り値

予測class一致率 `[0, 1]`。

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

## 関連API

`logits_rmse`, `collect_compression_metrics`
