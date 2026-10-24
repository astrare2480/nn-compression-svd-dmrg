# `sweep_tt_conv2d_weight_ranks`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_conv.py`

## 責務

Conv2d weightを行列化して既存のTT-matrix rank sweepへ委譲し、weight単位とbias込み層単位のparameter指標を返す。

## Signature
```python
sweep_tt_conv2d_weight_ranks(conv, out_modes, in_modes, ranks)
```

## 引数

- `conv`: 評価対象の`nn.Conv2d`。変更しない。
- `out_modes`: 出力channelのmode列。
- `in_modes`: 入力channelとkernel軸をまとめたmode列。
- `ranks`: 正整数rank上限の非空列。入力順と重複を保持する。

## 戻り値

各rankの`list[dict]`。主なkeyは`requested_rank`, `actual_ranks`, `tt_core_parameters`, `dense_weight_parameters`, `bias_parameters`, `dense_layer_parameters`, `tt_layer_parameters`, `weight_compression_ratio`, `layer_compression_ratio`, `absolute_error`, `relative_error`, `is_weight_compressed`, `is_layer_compressed`。

## 使用場面

CNNのTT rank候補を、モデル評価の前にweight誤差とparameter予算で比較するとき。

## 処理概要

Conv weightを`(out, in*kH*kW)`へreshapeし、`sweep_tt_matrix_weight_ranks`を呼ぶ。そのrecordをConv向け名称へ変換し、bias込みの層指標を加える。

## 主なcontract / 注意事項

- TT-SVDと誤差計算を再実装しない。autograd graphは作らない。
- Conv対応範囲は`build_tt_conv2d`と同じ。零weight、NaN/infを拒否する。
- weight圧縮と層全体圧縮はbiasの分だけ異なり得る。

## 関連API

`sweep_tt_matrix_weight_ranks`, `build_tt_conv2d`, `tt_matrix_num_parameters`
