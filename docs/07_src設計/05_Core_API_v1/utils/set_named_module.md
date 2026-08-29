# `set_named_module`

**Stability:** B  
**定義:** `src/nn_compression/utils/modules.py`

## 責務

model内の既存named submoduleを新しいModuleへ置換する。

## Signature

```python
set_named_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
) -> None
```

## 引数

置換対象model、既存path、新Module。

## 戻り値

なし。`model`をin-place変更する。

## 使用場面

圧縮model copy内の特定layerをfactorized Moduleへ差し替えるとき。

## ざっくりした処理

path存在確認 → `model.set_submodule(name, module, strict=True)`。

## 主なcontract / 注意事項

存在しないpathを暗黙作成せず、既存moduleの置換だけを許可する。

## 関連API

`get_named_module`, `factorize_named_linear`, `factorize_named_conv2d`
