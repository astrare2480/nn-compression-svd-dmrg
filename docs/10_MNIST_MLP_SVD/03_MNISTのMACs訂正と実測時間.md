---
title: MNISTのMACs訂正と実測時間
tags:
  - SVD
  - 実装検証
  - 実験記録
  - MNIST
---

# MNISTのMACs訂正と実測時間

> [!info] このノートは、旧 `10_SVD実装/04_理論計算量と実測時間.md` に混在していた実測・実行結果を分離したもの。
> 一般理論と実装方針は [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]] を参照する。

## 今回の実測値とMACsの訂正

### Baseline

対象モデル：

```text
784 → 512 → 256 → 10
```

biasを除いた1サンプル当たりのMACsは、

$$
784\times512
+
512\times256
+
256\times10
=
535,040
$$

である。

### `fc1_rank=64`、`fc2_rank=64`

`fc1`：

$$
784\times64
+
64\times512
=
82,944
$$

`fc2`：

$$
512\times64
+
64\times256
=
49,152
$$

非圧縮の `fc3`：

$$
256\times10
=
2,560
$$

したがって、モデル全体は、

$$
82,944
+
49,152
+
2,560
=
134,656
$$

MACsである。

削減率は、

$$
1-
\frac{134,656}{535,040}
=
0.748325\ldots
$$

より、約74.83%である。

> [!danger] MACsの修正
> `05_mnist_mlp_svd_compression.ipynb` の表示値 `134,756` は100多い。  
> 原因は、非圧縮の `fc3` を低ランク層として計算した式を使っていたためである。  
> `06_mnist_rank_accuracy_tradeoff.ipynb` では `linear_macs(model.fc3)` を使い、正しい `134,656` へ修正されている。

### 最終確認実行の推論時間

バッチサイズ1000、同じdevice、ウォームアップ後に5000回測定した結果：

| モデル | 推論時間 |
|---|---:|
| Baseline | 0.122 ms/batch |
| rank 64 / 64 SVD | 0.132 ms/batch |

理論MACsは約74.83%減ったが、実測では約8%遅くなった。

### `05` Notebook内の3モデル比較

別実行では次の結果だった。

| モデル | 推論時間 |
|---|---:|
| Baseline | 0.126 ms/batch |
| dense再構成SVD | 0.125 ms/batch |
| 2層SVD | 0.144 ms/batch |

dense再構成SVDはBaselineとほぼ同じであり、2層SVDだけが遅くなった。  
この結果は、遅延の主因が低ランク近似誤差ではなく、Linear呼出し回数の増加、小行列演算、GPU kernel起動などのオーバーヘッドであることを示している。

### ベンチマーク実装の要点

```python
images, _ = next(iter(data_loader))
```

`_` はラベルを使わないことを示す捨て変数である。

```python
for _ in range(warmup):
    _ = model(images)
```

ウォームアップでは初回実行の初期化やキャッシュ未使用状態を測定から除外する。

GPUでは処理が非同期なので、計測前後に、

```python
torch.cuda.synchronize()
```

を置く。

純粋な推論だけなら、

```python
model.eval()
with torch.inference_mode():
    ...
```

も利用できる。

### rank sweep内の時間値の扱い

`06_mnist_rank_accuracy_tradeoff.ipynb` では、Baselineと圧縮モデルで `repeats` が統一されていない箇所がある。  
そのため、rank sweep表のlatencyは補助情報とし、速度の最終結論には同一条件で測ったrank 64 / 64比較を使う。

