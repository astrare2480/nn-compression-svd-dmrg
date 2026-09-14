# `sweep_conv_svd_ranks`

**Stability:** C  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

旧Notebookの引数名`rank_list`を維持したまま、現行の`sweep_conv2d_ranks()`を呼び出すcompatibility wrapper。

## Signature

```python
sweep_conv_svd_ranks(
    model,
    layer_name,
    rank_list,
    out_hw,
    loader,
    criterion,
    device,
    *,
    baseline_acc,
    baseline_time_s,
    warmup=5,
    repeats=200,
    verbose=True,
    input_batch=None,
) -> list[dict]
```

## 引数

- `model`: candidate作成元となるbaseline model。
- `layer_name`: rank sweep対象Conv2dのnamed path。
- `rank_list`: 評価するSVD rank候補。現行`sweep_conv2d_ranks()`の`ranks`と同じ意味を持つhistorical引数名。
- `out_hw`: 対象Convの出力feature map size`(H, W)`。
- `loader`, `criterion`, `device`: candidateのvalidation・比較評価条件。
- `baseline_acc`, `baseline_time_s`: accuracy dropとlatency比較のbaseline値。
- `warmup`, `repeats`: latency benchmarkの反復条件。
- `verbose`: 下位APIの表示を有効にするか。
- `input_batch`: candidate間で固定利用するoptional benchmark入力。

## 戻り値

`sweep_conv2d_ranks()`と同じrank sweep recordの`list[dict]`。wrapper独自の列や変換は追加せず、旧`rank_list`を現行`ranks`へ読み替えた結果をそのまま返す。

## 使用場面

historical Notebookを変更せず再実行するとき。新規コードでは`sweep_conv2d_ranks()`を使用する。

## 処理概要

1. **旧引数名`rank_list`を受け取る。**
2. **値を変更せず`sweep_conv2d_ranks()`の`ranks`位置へ渡す。**
3. **その他の評価・benchmark引数もそのまま転送する。**
4. **primary APIの戻り値をそのまま返す。**  
   wrapper自身は分解・評価ロジックを持たない。

## 主なcontract / 注意事項

- Compatibility APIへ新しい機能や独自挙動を追加しない。
- device配置は委譲先の[[07_src設計/05_Core_API_v1/compression/sweep_conv2d_ranks]]と[[07_src設計/05_Core_API_v1/README#共通device contract|共通device contract]]に従う。

## 関連API

`sweep_conv2d_ranks`, `Compatibility_API`
