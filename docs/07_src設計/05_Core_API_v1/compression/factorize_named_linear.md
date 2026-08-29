# `factorize_named_linear`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

model内の指定named `nn.Linear`だけをSVD分解し、元modelとは独立した圧縮model copyを返す。

## Signature

```python
factorize_named_linear(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module
```

## 引数

- `model`: baseline model。
- `layer_name`: `"fc1"`, `"head.fc"`などのdot path。
- `rank`: 対象Linearの分解rank。

## 戻り値

指定Linearだけが2層factorized Linearへ置換されたmodel copy。

## 使用場面

Notebookや実験コードから特定の1層だけを圧縮し、baselineと比較するとき。

## 処理の流れ（日本語）

1. **baseline model全体を`deepcopy`する。**  
   圧縮後modelをFine-tuningしても元modelのParameterへ影響しないよう、最初に独立copyを作る。
2. **元modelから`layer_name`のsubmoduleを取得する。**  
   分解元はcopy途中の層ではなく、未圧縮のbaseline layerを使う。
3. **対象が`nn.Linear`か確認する。**  
   名前が存在しても型が違う場合は`TypeError`とし、誤った層への圧縮を防ぐ。
4. **`factorize_linear_layer()`で2層Linearを作る。**  
   SVD・bias配置・device/dtype・requires_grad処理は層単位APIへ委譲する。
5. **copy側modelの同じpathを置換する。**  
   `set_named_module()`で既存submoduleをfactorized Moduleへ差し替える。
6. **圧縮model copyを返す。**  
   baseline modelは一切変更しない。

### 処理フロー（短縮版）

```text
baseline model
→ deepcopy
→ 元modelからnamed Linear取得
→ 型確認
→ factorize_linear_layer
→ copy側の同pathを置換
→ compressed model
```

## 主なcontract / 注意事項

- baseline非破壊。
- `layer_name`は既存Linearを指す必要がある。
- nested pathも`get_named_module` / `set_named_module`の規約で扱う。

## 関連API

`factorize_linear_layer`, `get_named_module`, `set_named_module`, `factorize_named_layers`
