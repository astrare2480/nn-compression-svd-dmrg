# `mlp_sample_logits_error_bounds`

**Stability:** A
**定義:** `src/nn_compression/metrics/error_bounds.py`

## 責務

第1層だけを近似した2層MLPについて、sampleごとのlogits誤差上界を返す。

## Signature
```python
mlp_sample_logits_error_bounds(X, delta_W1, W2)
```

## 引数

- `X`: 各行がsample $x_n$である入力行列。
- `delta_W1`: 第1層weight差。
- `W2`: 変更しない第2層weight。

## 戻り値

shape `(batch,)`のTensor。要素$n$は

$$
U_n=\lVert x_n\rVert_2\lVert\Delta W_1\rVert_2\lVert W_2\rVert_2
$$

で、$\lVert\Delta l_n\rVert_2$の上界となる。

## 使用場面

sample単位のprediction stability certificateへ上界を渡すとき。

## 処理概要

共有する2つのspectral normを一度計算し、`X`の行normへ掛ける。共有因子が`inf`でも零入力行は0として分岐する。

## 主なcontract / 注意事項

- 引数contractは`mlp_logits_error_bound`と同じ。
- `delta_W1`または`W2`が零なら全sampleが0。
- 零行と極端な非零行が同じbatchにあっても`inf*0`によるNaNを作らない。

## 関連API

`mlp_logits_error_bound`, `sample_stability_certificates`
