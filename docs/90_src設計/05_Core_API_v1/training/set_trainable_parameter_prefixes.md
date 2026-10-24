# `set_trainable_parameter_prefixes`

**Stability:** A
**定義:** `src/nn_compression/training/parameter_scope.py`

## 責務

parameter名のprefixを使って学習対象を設定し、実際に一致した名前を返す。TT cores onlyなどの更新範囲をmodel種類に依存せず表現する。

## Signature
```python
set_trainable_parameter_prefixes(model, prefixes, *, freeze_others=True)
```

## 引数

- `model`: `requires_grad`を設定する`nn.Module`。parameter値は変更しない。
- `prefixes`: 非空文字列からなるlistまたはtuple。文字列単体、空列、空文字列は拒否する。
- `freeze_others`: `True`なら一致しない全parameterを凍結し、`False`なら既存状態を残して一致parameterだけ有効化する。

## 戻り値

一致したparameter名を登録順に並べた`tuple[str, ...]`。

## 使用場面

`("fc1.cores.",)`や`("conv2.cores.",)`を指定し、TT coreだけをfine-tuningするとき。

## 処理概要

1. modelとprefix列を検証する。
2. 変更前に全parameter名との一致集合を求める。
3. 1件も一致しなければmodelを変更せず失敗する。
4. `freeze_others`に従って`requires_grad_`を設定する。

## 主なcontract / 注意事項

- `"fc1"`は`fc1.weight`に一致するが、`fc10.*`や`fc1_aux.*`には一致しない。
- `"fc1.cores."`のように末尾が`.`のprefixは単純前方一致として扱う。
- 共有Parameterは`named_parameters()`と同様、最初の登録名だけを扱う。

## 関連API

`trainable_named_parameters`, `parameter_change_summary`, `TTLinear`, `TTConv2d`
