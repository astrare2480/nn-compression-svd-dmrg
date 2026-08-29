# `tucker2_effective_weight`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

3層Tucker-2 Conv `nn.Sequential`から、その3層と等価な元空間の4階Conv weightを再構成する評価用utility。

## Signature

```python
tucker2_effective_weight(
    seq: nn.Sequential,
) -> torch.Tensor
```

## 引数

`seq`: `input 1x1 → core Conv → output 1x1`のTucker-2構造を持つ`nn.Sequential`。

## 戻り値

`(C_out, C_in, kH, kW)`のeffective weight Tensor。

## 使用場面

圧縮直後・Fine-tuning後のTucker-2 Moduleを元Conv weight空間へ戻し、relative Frobenius errorを測るとき。

## 処理の流れ（日本語）

1. **SequentialがTucker-2の3層構造か検証する。**  
   3層すべてがConv2dであること、入力/出力projectionが1x1・stride1・padding0・groups1であること、bias位置やchannel接続が整合することを確認する。
2. **各layer weightを評価用Tensorとして取り出す。**  
   元Moduleへgradientを戻す用途ではないため、各weightを`detach()`して扱う。
3. **入力projectionから`U_in`を復元する。**  
   stored weightは`U_in.T`なので、1x1 dimensionを外して転置し、`(C_in, R_in)`へ戻す。
4. **中央layerからcoreを取得する。**  
   shapeは`(R_out, R_in, kH, kW)`。
5. **出力projectionから`U_out`を取得する。**  
   1x1 dimensionを外し、`(C_out, R_out)`とする。
6. **`reconstruct_tucker(core, {0: U_out, 1: U_in})`で元空間へ戻す。**
7. **4階effective weightを返す。**

### 処理フロー（短縮版）

```text
3層Sequential
→ 構造contract検証
→ input weightからU_in復元
→ core weight取得
→ output weightからU_out取得
→ reconstruct_tucker
→ effective 4D weight
```

## 主なcontract / 注意事項

- evaluation utilityなので内部layer weightはdetachして再構成する。
- biasはeffective **weight**には含めない。
- 任意の3層Convではなく、Tucker-2 builderが作る構造contractを要求する。

## 関連API

`build_tucker2_conv_from_components`, `reconstruct_tucker`, `relative_frobenius_error`
