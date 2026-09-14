# `sweep_conv2d_ranks`

**Stability:** B  
**定義:** `src/nn_compression/compression/rank_sweep.py`

## 責務

1つのnamed Conv2dについて、SVD rank候補を評価するための`factorize_conv2d_layer`＋MACs付きwrapper。

## Signature

```python
sweep_conv2d_ranks(
    model,
    layer_name,
    ranks,
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

- `model`: 各rank candidateを作るbaseline model。対象Convはこのmodelの未圧縮layerを正本にする。
- `layer_name`: SVD rank sweep対象となるConv2dのnamed path。
- `ranks`: 順番に評価するSVD rank候補。
- `out_hw`: 対象Convの実際の出力feature map size`(H, W)`。各rankの理論MACs計算に使う。
- `loader`: candidateのvalidation性能・agreement・logits RMSEを評価するloader。
- `criterion`: validation loss計算に使うloss関数。
- `device`: candidate評価とlatency benchmarkを実行するdevice。
- `baseline_acc`: 圧縮前modelのaccuracy。candidateごとのaccuracy dropの基準。
- `baseline_time_s`: 圧縮前modelの推論時間。latency比較の基準。
- `warmup`: latency本計測前のwarmup forward回数。
- `repeats`: latency平均を求める本計測forward回数。
- `verbose`: 下位評価APIの表示を有効にするか。
- `input_batch`: 全rankで共通利用するoptional benchmark入力。指定するとcandidate間で入力batchを固定できる。

## 戻り値

rank候補ごとの`list[dict]`。各recordはgeneric `sweep_layer_ranks()`の共通metricsに加え、対象`layer`、評価した`rank`、retained energy、Conv用MACs情報を持つ。

## 使用場面

Conv SVDのrank sweepをNotebookから簡潔に実行するとき。

## 処理概要

1. **Conv SVD用のMACs callbackを内部で定義する。**  
   元Convとrankを受け取り、`estimate_conv2d_macs()`でcompressed MACsと削減率を返す形にする。
2. **generic `sweep_layer_ranks()`へ処理を委譲する。**
3. **factorize callbackとして`factorize_conv2d_layer`を渡す。**  
   各rankで同じConv SVD実装を使う。
4. **対象Convの`out_hw`をMACs計算へ渡す。**  
   理論計算量が実際のfeature map sizeと対応するようにする。
5. **generic sweepが返したrecord listをそのまま返す。**

## 主なcontract / 注意事項

- `out_hw`は対象Convの実際の出力空間sizeを呼び出し側が与える。
- device配置は[[07_src設計/05_Core_API_v1/README#共通device contract|共通device contract]]に従う。baseline modelと、Parameter / bufferを持つcriterionは呼び出し前に指定deviceへ配置する。

## 関連API

`sweep_layer_ranks`, `factorize_conv2d_layer`, `estimate_conv2d_macs`
