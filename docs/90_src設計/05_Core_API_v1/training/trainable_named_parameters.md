# `trainable_named_parameters`

**Stability:** A
**定義:** `src/nn_compression/training/parameter_scope.py`

## 責務

現在`requires_grad=True`であるParameterを、optimizer設定と監査の両方に使える名前付きlistとして返す。

## Signature
```python
trainable_named_parameters(model)
```

## 引数

- `model`: 学習対象Parameterを列挙する`nn.Module`。

## 戻り値

登録順の`list[tuple[str, nn.Parameter]]`。各tupleはparameter名と同一Parameter objectを含む。

## 使用場面

更新範囲を設定した後、optimizerへ渡すParameter集合と記録する名前を同じ正本から作るとき。

## 処理概要

`model.named_parameters()`を走査し、`requires_grad`が真の要素だけを残す。

## 主なcontract / 注意事項

- model、Parameter値、`requires_grad`を変更しない。
- 空listも有効な結果であり、optimizer作成前の判断は呼び出し側が行う。
- 共有Parameterの重複除去はPyTorchの`named_parameters()`規約に従う。

## 関連API

`set_trainable_parameter_prefixes`, `parameter_change_summary`
