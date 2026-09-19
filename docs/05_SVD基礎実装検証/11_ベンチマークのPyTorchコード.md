# ベンチマークのPyTorchコード

数式と考え方は [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## ベンチマークコードでの `_` の使い方

`_` の一般的な意味と、ループ・複数戻り値での使い方は [[05_SVD基礎実装検証/13_PandasとPython実装メモ#24. 未使用値を受ける `_`]] に集約する。ベンチマークでは、主に次の3箇所で値を使わないことを表す。

```python
# ラベルを使わない。
images, _ = next(iter(data_loader))

# ループ番号を使わない。
for _ in range(warmup):
    model(images)

# 推論出力の値を使わない。
_ = model(images)
```

### `iter()` と `next()`

```python
iterator = iter(data_loader)
images, labels = next(iterator)
```

- `iter(data_loader)`：DataLoaderを順番に取り出すためのiteratorを作る
- `next(iterator)`：iteratorから次の1バッチを取り出す

したがって、

```python
next(iter(data_loader))
```

は、DataLoaderから最初の1バッチだけを取得する書き方である。

ベンチマークでは同じ入力を繰り返し使い、DataLoaderの読込み時間をforward時間へ混ぜないために1バッチだけ取り出す。

ただし、`shuffle=True` の学習用DataLoaderから

```python
images, _ = next(iter(train_loader))
```

と取り出すと、iterator生成とbatch取得によって、そのDataLoaderが参照する乱数Generatorの状態が進む。その後の学習順序へ影響し得るため、正式な比較では学習用loaderをshape確認やbenchmark入力取得へ流用しない。

```python
from nn_compression.metrics import benchmark_inference

# shuffle=Falseの評価用loaderから、比較用batchを1回だけ固定する。
images, _ = next(iter(evaluation_loader))
images = images.to(device)

# 圧縮前後で同じimagesを繰り返し使用する。
baseline_time = benchmark_inference(
    baseline_model,
    device=device,
    input_batch=images,
)
compressed_time = benchmark_inference(
    compressed_model,
    device=device,
    input_batch=images,
)
```

または、学習開始前に比較用Tensorを別途用意する。これにより、DataLoaderの読込み時間を除外しつつ、学習shuffleの再現性も壊さない。

---

## ベンチマーク処理の全体

```python
def benchmark_inference(
    model,
    data_loader,
    device,
    warmup=5,
    repeats=50,
):
    model.eval()

    images, _ = next(iter(data_loader))
    images = images.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeats):
            _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / repeats
```

処理順序：

```text
1. evalモードにする
2. DataLoaderから1バッチだけ取得する
3. deviceへ移す
4. 勾配計算を無効にする
5. warm-upを行う
6. GPU処理を同期する
7. perf_counterで計測開始
8. repeats回forwardする
9. GPU処理を同期する
10. 平均時間を返す
```

今回のMLPにはDropoutとBatchNormがないため、`eval()` によるforward計算の違いはほぼない。
それでも一般的な推論コードとして `model.eval()` を呼ぶ。

`torch.no_grad()` でもよいが、純粋な推論では `torch.inference_mode()` を使う選択肢もある。

---

## 表示値の丸め

`torch.round()` の丸め規則は [[05_SVD基礎実装検証/13_PandasとPython実装メモ#25. `torch.round()` と表示桁数]] に集約する。ベンチマーク結果は、元の計算値を変更せず、f-stringで表示だけを整える。

```python
print(f"{accuracy * 100:.2f}%")
print(f"{latency_ms:.3f} ms/batch")
```

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]（対応コード）

```python
from nn_compression.metrics import (
    linear_macs,
    compressed_linear_macs,
    conv2d_macs,
    compressed_conv2d_macs,
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]（1. 元Linear層のパラメータ数）

```python
nn.Linear(
    in_features=D_in,
    out_features=D_out,
    bias=True,
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]（16. CUDA同期）

```python
torch.cuda.synchronize()
```

