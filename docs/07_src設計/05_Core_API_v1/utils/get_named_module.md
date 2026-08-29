# `get_named_module`

**Stability:** B  
**定義:** `src/nn_compression/utils/modules.py`

## 責務

dot pathでmodel内の既存submoduleを取得する。

## Signature

```python
get_named_module(
    model: nn.Module,
    name: str,
) -> nn.Module
```

## 引数

modelと`conv2`, `block.0`, `block1.conv`等のpath。

## 戻り値

指定pathのsubmodule。

## 使用場面

層種別に依存しないnamed layer compression/metrics処理。

## ざっくりした処理

`model.get_submodule(name)`へ委譲する。

## 主なcontract / 注意事項

存在しないpathはPyTorch側のエラーになる。

## 関連API

`set_named_module`, `factorize_named_layers`
