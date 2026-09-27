# `set_named_module`

**Stability:** B  
**定義:** `src/nn_compression/utils/modules.py`

## 責務

model内の**既に存在する**named submoduleを新しいModuleへ置換する。入力modelをin-placeで変更する低レベルutility。

## Signature

```python
set_named_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
) -> None
```

## 引数

- `model`: 置換対象model。
- `name`: 置換する既存submodule path。
- `module`: 新しく配置する`nn.Module`。

## 戻り値

なし。`model`自体をその場で変更する。

## 使用場面

`deepcopy`済み圧縮model内の特定layerをfactorized Moduleへ差し替えるとき。

## 処理概要

1. **`get_named_module(model, name)`を先に呼ぶ。**  
   置換先pathが本当に既存submoduleを指しているか確認する。
2. **存在しないpathならここで失敗させる。**  
   typo等で新しい属性を暗黙作成しないため。
3. **`model.set_submodule(name, module, strict=True)`を呼ぶ。**
4. **既存位置のsubmoduleを新Moduleへ置換する。**  
   `model`はin-placeで変化する。
5. **戻り値は返さない。**

## 主なcontract / 注意事項

- 存在しないpathを暗黙作成しない。
- この関数自体はdeepcopyしない。baseline非破壊性が必要な場合、呼び出し側がcopyを作ってから利用する。

## 関連API

`get_named_module`, `factorize_named_linear`, `factorize_named_conv2d`, `factorize_named_layers`
