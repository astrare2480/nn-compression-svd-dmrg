# `get_named_module`

**Stability:** B  
**定義:** `src/nn_compression/utils/modules.py`

## 責務

dot区切りpathを使ってmodel内の既存submoduleを取得する。

## Signature

```python
get_named_module(
    model: nn.Module,
    name: str,
) -> nn.Module
```

## 引数

- `model`: 検索対象model。
- `name`: `"conv2"`, `"block.0"`, `"block1.conv"`などのsubmodule path。

## 戻り値

指定pathに存在する`nn.Module`。

## 使用場面

層種別に依存しないnamed compression、MACs集計、層置換前の対象取得。

## 処理概要

1. **呼び出し側からmodelとdot pathを受け取る。**
2. **PyTorch標準の`model.get_submodule(name)`へ委譲する。**
3. **PyTorchがdot pathを親moduleから順に辿る。**  
   直下属性だけでなくnested moduleやSequentialの番号付き子も扱える。
4. **見つかったsubmoduleをそのまま返す。**
5. **pathが存在しない場合はPyTorch側の例外をそのまま利用する。**

## 主なcontract / 注意事項

このutilityは層種別を判定しない。取得後にLinear/Conv2d等の型確認が必要なら呼び出し側が行う。

## 関連API

`set_named_module`, `factorize_named_layers`
