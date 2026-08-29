# `take_inference_batch`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

latency benchmarkでbaseline/compressedへ同じ入力を渡すため、DataLoaderから固定した1 input batchを取得する。

## Signature

```python
take_inference_batch(
    data_loader: DataLoader,
) -> torch.Tensor
```

## 引数

`data_loader`: input batch取得元。

## 戻り値

入力Tensor。batchが`(inputs, labels)`などtuple/listなら先頭要素を返す。

## 使用場面

baseline/compressed modelのbenchmark条件から「入力batchの違い」を排除したいとき。

## 処理概要

1. **DataLoaderのsamplerを確認する。**
2. **`RandomSampler`なら取得を拒否する。**  
   shuffle付きloaderから`next(iter(loader))`すると専用Generator状態を消費し、その後の学習順やcandidate比較条件を変える可能性があるため。
3. **`next(iter(data_loader))`で1 batch取得する。**
4. **StopIterationならempty loaderとして`ValueError`へ変換する。**
5. **batch形式を入力Tensorへ正規化する。**  
   batchがTensorならそのまま、tuple/listなら先頭要素をinputとして取り出す。
6. **固定input batchを返す。**  
   baselineとcompressedの両方へ同じTensorを渡すのは呼び出し側の責務。

## 主なcontract / 注意事項

- `RandomSampler`はshuffle Generator消費を避けるため拒否。
- empty loaderは`ValueError`。
- 任意custom samplerの乱数消費までは自動判定しない既知制約。

## 関連API

`benchmark_inference`, `benchmark_inference_print`, `collect_compression_metrics`
