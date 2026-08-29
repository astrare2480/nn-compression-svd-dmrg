---
title: training・metrics設計
---

# training・metrics設計

## 1. 役割

`training` は学習・評価ループ、`metrics` は圧縮前後の比較と理論計算量を担当する。

```text
training
├─ train_one_epoch
├─ evaluate
├─ fit_with_early_stopping
└─ non_shuffling_loader

metrics
├─ parameter count
├─ accuracy drop
├─ prediction agreement
├─ logits RMSE
├─ inference benchmark
├─ compression metric aggregation
├─ Linear / Conv2d MACs
└─ tensor relative Frobenius error
```

圧縮方式に依存しないため、TT/MPS・DMRGでも可能な限り再利用する。

---

# 2. `train_one_epoch`

```python
train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
) -> tuple[float, float]
```

処理：

```text
model.train()
→ batch loop
→ zero_grad
→ forward
→ loss
→ backward
→ optimizer.step
→ loss / accuracy集計
```

criterionがbatch meanを返す前提で、

```python
total_loss += loss.item() * images.size(0)
```

とサンプル数で重み付けし、最後に全サンプル数で割る。

返り値は、

```text
(avg_loss, accuracy)
```

の2-tuple。

### empty loader

`len(loader)` に依存せず、実際に走査した `total == 0` を確認する。

emptyなら、

```text
ValueError("空の DataLoader では学習できません。")
```

を送出する。

### mode semantics

学習関数なので `model.train()` を設定する。評価APIのように呼び出し前のtraining modeへ戻す契約ではない。

---

# 3. `evaluate`

```python
evaluate(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float]
```

```text
一時的にmodel.eval()
torch.no_grad()
loader全走査
loss / accuracy
元のtraining状態を復元
```

返り値は `(avg_loss, accuracy)`。

### 全submodule状態復元

rootの `model.training` だけを保存して `model.train(was_training)` で戻すと、全submoduleが一律同じ状態になる。

```text
root = train
BatchNorm = eval
```

のような個別設定を壊さないため、現行実装は全 `model.modules()` の `.training` を個別に保存し、`finally` で直接復元する。

正常終了・例外発生のどちらでも元状態へ戻す。このcontractはCore API v1で固定する。

### empty loader

`len(loader)` ではなく、実際のtotalを確認する。emptyならValueError。

---

# 4. IterableDataset方針

`evaluate`、`agreement`、`logits_rmse` は `len(loader)` を前提にしない。

そのため `IterableDataset` を持つDataLoaderでも、通常の1回走査として利用できる。

ただし、`collect_compression_metrics()` は同じloaderを、

```text
evaluate
agreement
logits_rmse
```

等で複数回走査する。

したがってone-shot iteratorではなく**再iterableなloader**を前提とする。詳細は [[07_src設計/07_既知の制約と拡張方針]]。

---

# 5. `fit_with_early_stopping`

```python
fit_with_early_stopping(...)
```

基本フロー：

```text
epoch
→ train_one_epoch
→ validation evaluate
→ 必要ならtrain metric再評価
→ best validation判定
→ state_dict deepcopy
→ patience判定
...
→ best state restore
```

改善判定：

```python
best_validation_loss - min_delta > validation_loss
```

best stateは `copy.deepcopy(model.state_dict())` で保存する。

### return structure

返り値dictのキーはCore API v1で維持する。

```text
model
best_epoch
best_validation_loss
train_loss_history
validation_loss_history
history
```

`history` の各要素：

```text
epoch
train_acc
train_loss
validation_loss
validation_acc
```

### train metric再評価

shuffle付き `train_loader` を評価で再走査すると、専用Generatorの状態が進み、次epochのmini-batch順を変える。

そのため、

- `train_eval_loader` を明示的に渡す
- または `non_shuffling_loader(train_loader)` を作る

という方針を取る。

`reevaluate_train=False` なら `train_one_epoch` の戻り値を履歴へ使う。

---

# 6. `non_shuffling_loader`

元DataLoaderと同じDatasetを `shuffle=False` で再走査するためのutility。

主用途：

```text
training順序を変えずtrain metricsを取り直す
```

任意DataLoader構成を完全cloneするAPIではない。custom sampler / batch_sampler等の完全再現は保証しない。

---

# 7. model comparison metrics

## `count_parameters`

```python
sum(p.numel() for p in model.parameters())
```

`requires_grad` に関係なく、モデルが保持する全Parameter要素数を数える。「学習可能parameter数」とは別定義。

## `parameters_reduction`

$$
1-\frac{N_{compressed}}{N_{baseline}}
$$

負なら圧縮後の方がparameter数が多い。

## `accuracy_drop`

$$
Acc_{baseline}-Acc_{compressed}
$$

正なら精度低下、負ならcompressedのaccuracyが高い。single seedの微小負値を性能改善と一般化するかは実験側の責務。

---

# 8. `agreement`

baselineとcompressedの予測クラス一致率。

$$
\frac{1}{N}\sum_i\mathbf 1[
\arg\max f(x_i)=\arg\max \hat f(x_i)
]
$$

- 正解ラベルとのaccuracyとは別
- loader全体で集計
- empty loaderはValueError
- 両modelを一時的にeval
- 両modelのroot + 全submodule training stateを復元

---

# 9. `logits_rmse`

baseline / compressedの出力Tensor全要素について、

$$
\sqrt{\frac{\sum (z-\hat z)^2}{\text{num elements}}}
$$

を計算する。

クラス数10を固定しない。batchごとのRMSEを平均せず、全要素の二乗誤差和からdataset全体のRMSEを計算する。empty loaderはValueError。

---

# 10. inference benchmark

## `take_inference_batch`

baseline / compressedを同じinput Tensorで比較するため、1batchを取得する。

shuffle付き `RandomSampler` はGeneratorを消費するため拒否する。

```text
input_batchを明示的に渡す
or
shuffle=False loaderから取る
```

を基本とする。empty loaderはValueError。

---

## `benchmark_inference`

```python
benchmark_inference(
    model,
    data_loader=None,
    device=None,
    warmup=10,
    repeats=5000,
    *,
    input_batch=None,
    return_details=False,
)
```

### benchmark contract

```text
device必須
warmup >= 0
repeats >= 1
warmup/repeatsはbool拒否・整数必須
input_batch優先
data_loaderはinput_batchが無いときだけ使用
modelを一時的にeval
終了後に全submodule状態復元
```

CUDAでは計測区間の前後で対象deviceを明示して `torch.cuda.synchronize(device)` を行う。

既定returnは1batchあたり平均秒数 `float`。

`return_details=True` では、

```text
time_s
time_ms
batch_size
input_shape
warmup
repeats
```

をdictで返す。

---

## `benchmark_inference_print`

baseline / compressedを同じinput batch、warmup、repeatsで順に計測するconvenience API。

返り値：

```text
baseline_time_s, compressed_time_s
```

---

# 11. `collect_compression_metrics`

圧縮candidateの共通評価recordを作る。

基本キー：

```text
parameters
parameters_reduction
validation_loss
validation_acc
accuracy_drop
baseline_time_ms
compressed_time_ms
benchmark_batch_size
benchmark_input_shape
benchmark_warmup
benchmark_repeats
agreement
logits_rmse
```

必要に応じて、

```text
baseline_macs
compressed_macs
compute_reduction
model
```

を追加する。

`include_model=False` が既定。rank sweepで全candidate modelをDataFrameへ保持しないための設計。

higher-level API全体としてbaseline/compressedのtraining stateを呼出前へ復元する。

---

# 12. Linear MACs

baseline：

$$
MACs=D_{in}D_{out}
$$

factorized：

$$
MACs=D_{in}r+rD_{out}
$$

`compressed_linear_macs` のrankは、

$$
1\le r\le\min(D_{in},D_{out})
$$

を要求する。

---

# 13. Conv2d MACs

## output spatial contract

`conv2d_macs`、`compressed_conv2d_macs`、`estimate_conv2d_macs` で使う出力空間sizeは、

```text
out_h / out_w
→ bool以外の正の整数scalar
```

を要求する。

floatや0/負数を暗黙に計算へ流さない。

## baseline

PyTorch weight shapeを、

```text
(C_out, C_in/groups, kH, kW)
```

として、

$$
H_{out}W_{out}C_{out}\frac{C_{in}}{groups}K_hK_w
$$

を計算する。

`conv2d_macs` 自体はgrouped Convのweight shapeにも対応する。

## SVD factorized Conv

現在のConv SVDは `groups=1` のみなので、compressed MACsもgrouped Convをrejectする。

$$
H_{out}W_{out}\left(rC_{in}K_hK_w+C_{out}r\right)
$$

rank上限：

$$
\min(C_{out},C_{in}K_hK_w)
$$

分解実装と同じrank contractを使う。

---

# 14. model-specific MACs

`estimate_mlp_macs`、`estimate_cnn_linear_macs`、`estimate_cnn_conv2_macs`、`estimate_cnn_macs` は既存実験構造に合わせた集計API。

層単位の式・input validationは `macs.py` を正本とする。

`estimate_cnn_macs` は、

- `conv_output_hw`
- `conv_ranks`
- `linear_ranks`
- `linear_layer_names`

を引数化している。

一方 `estimate_mlp_macs` は現行784→512→256→10 MLPに強く依存するためexperiment-support APIとする。

---

# 15. Tensor relative error

```python
relative_frobenius_error(X, X_hat)
```

$$
\frac{\|X-X_{hat}\|_F}{\|X\|_F}
$$

- shape一致必須
- `X` がzero tensorならValueError
- Tensorを返す
- 関数内でdetachしない

HOOIではerror historyをPython floatへ保存する箇所で明示的にdetachする。

---

# 16. metricsとcompressionの依存制約

現在、

```text
compression.hooi
→ metrics.relative_frobenius_error
```

という依存がある。

そのため `metrics → compression` を追加すると循環importの危険がある。このためrank/integer validationが一部局所実装されている。

単純なDRY化より依存方向を優先する。

---

# 17. TT/MPS・DMRGでの再利用

新方式でも、

- `train_one_epoch`
- `evaluate`
- `fit_with_early_stopping`
- `count_parameters`
- `agreement`
- `logits_rmse`
- `benchmark_inference`
- `relative_frobenius_error`

は原則再利用する。

新しい圧縮表現のMACsだけは既存SVD/Tucker式へ無理に押し込まず、`tt_parameter_count` / `tt_linear_macs` 等の専用関数を追加する。
