---
title: MNIST評価で使用した指標と実装
tags:
  - SVD
  - 実装検証
  - 実験記録
  - MNIST
---

# MNIST評価で使用した指標と実装

> [!info] このノートは、旧 `10_SVD実装/03_SVD圧縮モデルの評価.md` に混在していた実測・実行結果を分離したもの。
> 一般理論と実装方針は [[00_基礎理論/10_SVD圧縮モデルの評価設計]] を参照する。

## 今回の実験で実際に使ったモデル忠実度指標

最終版PDFの検討を受け、accuracy以外に次の2指標を追加した。

### Baselineとの予測一致率

同じ入力に対するBaselineの予測を $\hat{y}_{\mathrm{base}}$、圧縮モデルの予測を $\hat{y}_{\mathrm{svd}}$ とする。

$$
\operatorname{Agreement}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\hat{y}_{\mathrm{base},i}
=
\hat{y}_{\mathrm{svd},i}
\right]
$$

accuracyが同じでも、両モデルが同じサンプルで同じ判断をしているとは限らない。  
Agreementは、圧縮モデルがBaselineの判断をどれだけ維持しているかを測る。

実コードでは、

```python
agreement(...)
```

として実装されている。

### logits RMSE

Baselineのlogitsを $z_{\mathrm{base}}$、圧縮モデルのlogitsを $z_{\mathrm{svd}}$ とする。

$$
\operatorname{LogitsRMSE}
=
\sqrt{
\frac{1}{N C}
\sum_{i=1}^{N}
\sum_{c=1}^{C}
\left(
z_{\mathrm{base},ic}
-
z_{\mathrm{svd},ic}
\right)^2
}
$$

予測クラスが同じでも、クラス間のmarginや出力分布が変化している場合がある。  
logits RMSEは、その変化を連続値として捉える。

実コードでは、

```python
logits_rmse(...)
```

として実装されている。

### rank sweepで観測した傾向

- rankを上げるほどAgreementは1へ近づいた
- rankを上げるほどlogits RMSEは小さくなった
- `fc1_rank=16` では、`fc2_rank` を増やしてもaccuracyは約87%のままだった
- `fc1_rank=64` 以上ではaccuracyはBaseline付近まで回復した

この結果から、今回のモデルでは `fc2` よりも `fc1` のrankが分類性能を強く制限していることが分かった。

### 評価設計上の注意

`06_mnist_rank_accuracy_tradeoff.ipynb` はtest set上でrankを比較している。  
探索実験としては有用だが、厳密にはvalidation setでrankを選び、test setは最後の1回だけ使うべきである。

また、fine-tuningは実装していない。  
したがって、今回報告するaccuracyはすべて**SVD置換直後**の値である。

## 学習・評価ループの読み方

添付PDFでは、MNIST実験コードの各行が何をしているかも確認した。

### 学習の基本順序

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

役割は次のとおりである。

```text
zero_grad()
前回までの勾配を消す

model(images)
forwardを実行してlogitsを得る

criterion(outputs, labels)
予測と正解のずれをlossにする

loss.backward()
計算グラフを逆向きにたどり、各Parameterのgradを計算する

optimizer.step()
gradを使ってParameterを実際に更新する
```

`backward()` は勾配を計算するだけであり、重みを変更するのは `optimizer.step()` である。

PyTorchの勾配は既定では加算されるため、通常はバッチごとに `zero_grad()` を実行する。

### `model.train()` と `model.eval()`

```python
model.train()
```

は、モデル自身と子Moduleの `training` フラグを `True` にする。

```python
model.eval()
```

は評価モードへ切り替え、実質的には `model.train(False)` に相当する。

今回のMLPはLinearとReLUだけなので、forward結果はtrainとevalで基本的に変わらない。  
ただし、DropoutやBatchNormを追加した場合は挙動が変わるため、学習時と評価時で明示的に切り替える。

自作 `forward()` の中でも、

```python
if self.training:
    ...
else:
    ...
```

と書けば、モードによって処理を分岐できる。

### logitsと予測クラス

MNISTモデルの出力shapeは、

```text
(batch_size, 10)
```

である。

各行には数字0から9に対応する10個のlogitsが入る。  
これはSoftmax適用前の生スコアであり、確率そのものではない。

```python
predicted = torch.argmax(outputs, dim=1)
```

`dim=1` は、各画像に対応する行の中で最大のクラス番号を選ぶという意味である。

`CrossEntropyLoss` には、Softmaxを手動で適用せずlogitsをそのまま渡す。

### `images.size(0)` とloss集計

MNISTのバッチが、

```text
(64, 1, 28, 28)
```

なら、

```python
images.size(0)
```

は先頭次元である64を返す。

`CrossEntropyLoss` の既定設定では、`loss.item()` はそのバッチの平均lossである。

そのため、

```python
total_loss += loss.item() * images.size(0)
```

としてバッチ内のサンプル数を掛け、最後に全サンプル数で割る。

最終バッチは必ずしも設定したbatch sizeと同じとは限らないため、固定値64ではなく `images.size(0)` や `labels.size(0)` を使う。

### 今回使用したoptimizer

実験コードではAdamを使用した。

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

SGD、SGD with momentum、RMSprop、AdamWなども選択できるが、optimizerを変更すると実験条件も変わる。  
rank比較ではoptimizer、学習率、epoch数を固定する。

---

## rank候補の全組合せ

`fc1` と `fc2` のrank候補を手作業で16個列挙する必要はない。

実コードではPython標準の `itertools.product` を使った。

```python
import itertools

r1_list = [16, 32, 64, 128]
r2_list = [16, 32, 64, 128]

rank_configs = list(
    itertools.product(
        r1_list,
        r2_list,
    )
)
```

これは2つの候補集合の直積を作る。

```text
(16, 16)
(16, 32)
...
(128, 128)
```

NumPyの `meshgrid` でも作れるが、単純なrank組合せの列挙には `itertools.product` の方が直接的である。

実コードでは、

```python
for fc1_rank, fc2_rank in rank_configs:
    ...
```

として全条件を評価した。

