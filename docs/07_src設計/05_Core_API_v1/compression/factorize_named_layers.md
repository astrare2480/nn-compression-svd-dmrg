# `factorize_named_layers`

**Stability:** A  
**定義:** `src/nn_compression/compression/named_layers.py`

## 責務

model内の複数named Conv2d / Linearを、層種別ごとのrank指定に従って一度に低rank分解したmodel copyを作る。

## Signature

```python
factorize_named_layers(
    model: nn.Module,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
) -> nn.Module
```

## 引数

- `model`: baseline model。
- `conv_ranks`: `{layer_name: rank}`形式のConv圧縮指定。
- `linear_ranks`: `{layer_name: rank}`形式のLinear圧縮指定。

## 戻り値

指定された複数層をfactorized Moduleへ置換したmodel copy。

## 使用場面

CIFAR-10などで複数Convを同時に圧縮するmodel-wide SVD、Conv+Linear同時圧縮。

## 処理の流れ（日本語）

1. **baseline modelを`deepcopy`する。**  
   すべての置換はcopy側だけへ行う。
2. **Conv圧縮指定を順に処理する。**  
   各`layer_name`について、**元baseline model**からsubmoduleを取得する。
3. **対象がConv2dか確認する。**  
   正しい型なら`factorize_conv2d_layer()`で2層Convを作り、copy側の同pathを置換する。
4. **Linear圧縮指定を順に処理する。**  
   同様に元baseline modelから対象を取り、型を確認して`factorize_linear_layer()`で置換する。
5. **すべての置換が終わったmodel copyを返す。**

### なぜ「元baseline layer」から毎回分解するか

圧縮途中のcopyから次の分解元を取ると、先に置換した構造の影響やnested path変化が混ざる可能性がある。各指定層は未分解のbaseline layerを正本として独立に分解する。

### 処理フロー（短縮版）

```text
baseline model
→ deepcopy
→ conv_ranks: 元modelから取得 → 型確認 → Conv SVD → copyを置換
→ linear_ranks: 元modelから取得 → 型確認 → Linear SVD → copyを置換
→ compressed model
```

## 主なcontract / 注意事項

- baselineとParameterを共有しない。
- Conv/Linearの分解contractはそれぞれの層単位APIへ委譲する。

## 関連API

`factorize_conv2d_layer`, `factorize_linear_layer`, `get_named_module`, `set_named_module`
