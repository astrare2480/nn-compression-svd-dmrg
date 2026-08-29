# `take_inference_batch`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

benchmark比較用に固定した1 input batchをDataLoaderから取得する。

## Signature

```python
take_inference_batch(
    data_loader: DataLoader,
) -> torch.Tensor
```

## 引数

`data_loader`: input batch取得元。

## 戻り値

画像等のinput Tensor。`(inputs, labels)`型batchなら先頭要素を返す。

## 使用場面

baseline/compressed benchmarkへ**同じ入力Tensor**を渡すとき。

## ざっくりした処理

```text
sampler確認
→ next(iter(loader))
→ Tensorまたはtuple/listの先頭をinputとして抽出
```

## 主なcontract / 注意事項

- `RandomSampler`はshuffle Generator消費を避けるため拒否。
- empty loaderは`ValueError`。
- custom samplerの乱数消費までは自動判定しない。

## 関連API

`benchmark_inference`, `benchmark_inference_print`, `collect_compression_metrics`
