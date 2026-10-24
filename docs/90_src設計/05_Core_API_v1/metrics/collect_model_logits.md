# `collect_model_logits`

**Stability:** A
**定義:** `src/nn_compression/metrics/classification_outputs.py`

## 責務

DataLoaderを一度走査し、model logitsとlabelsを同じsample順で収集する。

## Signature
```python
collect_model_logits(model, data_loader, device, *, output_device="cpu")
```

## 引数

- `model`: shape `(batch, num_classes)`のlogitsを返す`nn.Module`。
- `data_loader`: 各batchが`(inputs, labels)`であるiterable。
- `device`: inputsを転送してforwardするdevice。
- `output_device`: 連結後のlogits/labelsを置くdevice。既定はCPU。

## 戻り値

`(logits, labels)`。logitsはshape `(N,C)`、labelsは先頭次元`N`で、loader順を保持する通常Tensor。

## 使用場面

同一の固定loaderについて、複数modelのlogitsを後から比較・保存するとき。

## 処理概要

modelと全submoduleのtraining状態を保存し、`eval()+inference_mode()`で走査する。batchごとの出力を指定deviceへ移し、inference modeを抜けてから`torch.cat`する。

## 主なcontract / 注意事項

- model自体は暗黙にdevice移動しない。空loader、不正batch、非2階出力、batch数不一致を拒否する。
- shuffle loaderを別々に走査するとsample対応が崩れるため、比較には固定順loaderを使う。
- 例外時も各submoduleのtraining状態を個別に復元する。

## 関連API

`classification_metrics_from_logits`, `compare_logits`, `non_shuffling_loader`
