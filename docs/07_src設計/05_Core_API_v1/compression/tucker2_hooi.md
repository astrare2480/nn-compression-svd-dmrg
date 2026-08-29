# `tucker2_hooi`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

Conv2d weightのchannel mode 0/1へpartial HOOIを適用し、Tucker-2 core・factor・誤差履歴を返す。

## Signature

```python
tucker2_hooi(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]
```

## 引数

- `weight`: `(C_out,C_in,kH,kW)`の実数浮動小数点Tensor。
- `rank_out`, `rank_in`: channel ranks。
- `max_iter`, `abs_tol`, `rel_tol`: generic `hooi()`と同じ反復条件。

## 戻り値

`(core, factors, history)`。`factors`のkeyは0/1、`history[0]`はHOSVD初期誤差。

## 使用場面

同rankのHOSVD Tucker-2よりweight再構成誤差を反復改善したいとき。

## 処理の流れ（日本語）

1. **Conv weightが4階で、channel rankが基本範囲内か検証する。**
2. **Tucker-2用rank Mappingを作る。**  
   `{0: rank_out, 1: rank_in}`として空間modeは圧縮しない。
3. **generic `hooi()`へweightをそのまま渡す。**  
   **この関数では`detach()`しない。** Tensor-level decompositionとして、呼び出し側が必要ならautograd graphを保持できるようにする。
4. **generic HOOIがHOSVD初期化・feasibility検証・sweep・収束判定を行う。**
5. **`core, factors, history`をそのまま返す。**

### Tensor-levelとModule-levelの境界

```text
tucker2_hooi(weight)
→ detachしないTensor計算

build_tucker2_conv(conv)
→ detachして新Module Parameterを作る
```

この2つを分離することで、分解アルゴリズム自体のautograd可能性と、学習済みModuleを独立Parameterへ置換する処理を混同しない。

### 処理フロー（短縮版）

```text
4D weight / ranks検証
→ ranks={0: rank_out, 1: rank_in}
→ generic hooiへ委譲（detachなし）
→ core, factors, history
```

## 主なcontract / 注意事項

- **Tensor-level APIなので入力weightをdetachしない。**
- Moduleへ変換するときのdetachは`build_tucker2_conv()`側の責務。
- `history[0]`はHOSVD初期誤差。

## 関連API

`hooi`, `tucker2_hooi_sweep`, `build_tucker2_conv_from_components`
