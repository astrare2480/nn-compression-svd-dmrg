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

## 処理概要

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

### フローチャート

```mermaid
flowchart TD
    A["baseline model"] --> B["deepcopy して compressed model を作る"]
    B --> C{"未処理の conv_ranks がある?"}
    C -- Yes --> D["元 baseline から named Conv2d を取得"]
    D --> E["型確認 → factorize_conv2d_layer"]
    E --> F["copy 側の同 path を置換"]
    F --> C
    C -- No --> G{"未処理の linear_ranks がある?"}
    G -- Yes --> H["元 baseline から named Linear を取得"]
    H --> I["型確認 → factorize_linear_layer"]
    I --> J["copy 側の同 path を置換"]
    J --> G
    G -- No --> K["compressed model を返す"]
```

## 主なcontract / 注意事項

- baselineとParameterを共有しない。
- Conv/Linearの分解contractはそれぞれの層単位APIへ委譲する。

## 関連API

`factorize_conv2d_layer`, `factorize_linear_layer`, `get_named_module`, `set_named_module`
