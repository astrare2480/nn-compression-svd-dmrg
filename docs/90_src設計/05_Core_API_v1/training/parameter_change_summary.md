# `parameter_change_summary`

**Stability:** A
**定義:** `src/nn_compression/training/parameter_scope.py`

## 責務

学習前stateと現在のmodelをparameter単位で比較し、凍結対象が変化していないか、学習対象がどれだけ変化したかを記録する。

## Signature
```python
parameter_change_summary(before_state, model)
```

## 引数

- `before_state`: `copy.deepcopy(model.state_dict())`等で保存した`Mapping[str, Tensor]`。全parameterを含める。modelに存在するbuffer keyは余分に含めてもよい。
- `model`: 比較時点の`nn.Module`。変更しない。

## 戻り値

登録順の`list[dict]`。各recordは`name`, `requires_grad`, `numel`, `exactly_equal`, `max_abs_change`を持ち、そのまま`pandas.DataFrame`へ渡せる。

## 使用場面

cores-only fine-tuning後に、biasや他層がbitwiseに不変であることを検証するとき。

## 処理概要

key、shape、dtype、deviceを検証し、`torch.equal`とdtype別の安全な最大絶対差を計算する。完全一致なら差分を計算せず0を返す。

## 主なcontract / 注意事項

- floatはfloat64差分、complexはcomplex128の複素絶対値、integerはint64 overflowとfloat64丸めを避けて差を計算する。
- 同じ符号の同位置infは変化0、finite↔infと`+inf`↔`-inf`は`inf`。
- NaNを拒否せず、含まれる場合は不一致かつ`max_abs_change=nan`。NaNのない入力からNaNを作らない。
- `exactly_equal=True`なら必ず`max_abs_change==0.0`。

## 関連API

`set_trainable_parameter_prefixes`, `trainable_named_parameters`
