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

意味は`sweep_conv2d_ranks()`と同じ。`rank_list`だけが旧名。

## 戻り値

`sweep_conv2d_ranks()`が返すrank sweep record list。

## 使用場面

historical Notebookを変更せず再実行するとき。新規コードでは`sweep_conv2d_ranks()`を使用する。

## 処理概要

1. **旧引数名`rank_list`を受け取る。**
2. **値を変更せず`sweep_conv2d_ranks()`の`ranks`位置へ渡す。**
3. **その他の評価・benchmark引数もそのまま転送する。**
4. **primary APIの戻り値をそのまま返す。**  
   wrapper自身は分解・評価ロジックを持たない。

## 主なcontract / 注意事項

Compatibility APIへ新しい機能や独自挙動を追加しない。

## 関連API

`sweep_conv2d_ranks`, `Compatibility_API`
