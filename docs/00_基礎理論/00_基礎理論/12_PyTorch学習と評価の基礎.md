---
title: PyTorch学習と評価の基礎
aliases:
  - DatasetとDataLoader
  - epochとmini-batch
  - Early Stopping
  - Train Validation Test
  - state_dictとdeepcopy
  - PyTorch再現性
tags:
  - PyTorch
  - 機械学習
  - 学習
  - EarlyStopping
  - Validation
  - 再現性
---

# PyTorch学習と評価の基礎

## サマリー

ニューラルネットワークの実験では、モデル構造だけでなく、データの流し方・学習単位・検証・checkpoint・再現性を分けて理解する必要がある。

このノートでは、Fashion-MNIST実験で確認した一般的な学習概念を、データセット非依存の知識として整理する。

```text
Dataset
↓
DataLoader
↓
mini-batchごとにforward / backward / optimizer.step
↓
1周すると1 epoch
↓
ValidationでEarly Stopping
↓
best stateを復元
↓
モデル選択終了後にTest
```

---

## 1. DatasetとDataLoader

`Dataset` は、サンプルとラベルを保持・取得するための入れ物である。
`DataLoader` はDatasetからサンプルをまとめて取り出し、mini-batchとして学習ループへ渡す。

主な役割：

- `batch_size` ごとにまとめる
- `shuffle=True` で学習時の順序を入れ替える
- 必要に応じてworkerで並列読み込みする

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)
```

---

## 2. `ToTensor()` と `Compose`

画像Datasetでは、PIL ImageやNumPy配列をPyTorch Tensorへ変換する。
`transforms.ToTensor()` は典型的に、画像を `(C,H,W)` Tensorへ変換し、8bit画像なら画素値を0〜1へスケーリングする。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
])
```

`Compose` は複数のtransformを順番につなぐ器である。
`ToTensor()` だけの場合でも、後でNormalizeやaugmentationを追加しやすい。

`ToTensor()` による0〜1化と、平均0・分散1などへ変換する標準化は別である。

---

## 3. mini-batchと1 epoch

学習データ数を $N$、batch sizeを $B$ とすると、1 epochのbatch数はおおよそ、

$$
\left\lceil \frac{N}{B} \right\rceil
$$

である。

1 mini-batchごとに典型的には、

```text
forward
↓
loss
↓
zero_grad
↓
backward
↓
optimizer.step
```

を行う。

1 epochは「学習データを1回すべて見た」という単位であり、学習完了の意味ではない。
複数epochでは、更新後の重みで同じデータを再び学習し、誤差を段階的に減らす。

---

## 4. epochを増やす理由と過学習

1 epochだけでは、ランダム初期値から十分に最適化できていないことが多い。
一方、epochを増やし続けるとTrain性能だけ改善し、未知データの性能が悪化することがある。

典型的な学習曲線は、

- train loss
- validation loss
- train accuracy
- validation accuracy

をepochに対して描く。

```text
train lossは低下し続ける
validation lossは途中から上昇
```

となれば過学習の兆候である。

---

## 5. Train / Validation / Testの役割

### Train

`loss.backward()` と `optimizer.step()` により重みを更新する。

### Validation

重みを更新せず、Early Stopping、hyperparameter、モデル候補の選択に使う。

### Test

学習とモデル選択がすべて終わった後に、最終モデルの汎化性能を測る。

```text
Train = 学ぶ
Validation = 止める・選ぶ
Test = 最後に測る
```

Testを途中で何度も見てモデルを変更すると、Testへ選択過適合する。

---

## 6. Validationを2段階に分ける

モデル選択が多段階の場合、Validationをさらに、

```text
Early-Stopping Validation
Rank-Selection Validation
```

へ分けられる。

前者はepochの選択、後者はrankなどのモデル構造選択へ使う。
この方が、同じValidationへ何度も適応する影響を減らせる。

---

## 7. Early Stopping

最大epoch数を十分大きく設定し、validation lossが一定回数改善しなければ停止する。

```python
MAX_EPOCHS = 50
PATIENCE = 5
MIN_DELTA = 1e-4
```

改善条件の例：

```python
if best_validation_loss - MIN_DELTA > validation_loss:
    ...
```

- `PATIENCE`：何回改善なしを許すか
- `MIN_DELTA`：改善とみなす最小差

Early Stoppingは最後のepochを採用する仕組みではない。
**validation lossが最良だった時点の重みを保存し、最後に復元する。**

---

## 8. `state_dict()`

PyTorchの `state_dict()` は、モデルの学習可能parameterとbufferを保持する辞書である。
モデル構造そのものではない。

```python
state = model.state_dict()
```

保存される典型例：

```text
fc1.weight
fc1.bias
fc2.weight
fc2.bias
...
```

`state_dict` を別構造のモデルへそのままloadすることはできない。
keyとshapeが対応する必要がある。

---

## 9. `copy.deepcopy(model.state_dict())`

best checkpointをメモリ上へ保持するときは、後の学習更新から独立させるためdeep copyする。

```python
best_model_state = copy.deepcopy(
    model.state_dict()
)
```

これはモデル本体のコピーではなく、「その瞬間のparameter値の独立コピー」である。

---

## 10. `load_state_dict()`

最良重みをモデルへ戻すには、

```python
model.load_state_dict(best_model_state)
```

を使う。

```text
model = ネットワーク構造を持つ器
state_dict = 重み・bias等の辞書
load_state_dict = 器へparameterを読み戻す
```

`model = copy.deepcopy(best_model_state)` では、辞書をモデル変数へ入れるだけなのでモデル復元にならない。

---

## 11. モデル本体をコピーする場合

fine-tuningなどで元モデルを壊したくない場合は、state_dictではなくモデル本体をdeep copyする。

```python
new_model = copy.deepcopy(old_model)
```

これは、

```python
state = copy.deepcopy(model.state_dict())
```

とは目的が異なる。

- モデルdeepcopy：独立したモデルオブジェクトを作る
- state_dict deepcopy：ある時点のparameter値を保存する

---

## 12. optimizerはモデルのparameterに結び付く

optimizerは作成時に渡したparameterへの参照を持つ。

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
)
```

別モデルや、Linear / Conv2d置換後の新しい層へ同じoptimizerを使い回さない。
候補モデルごとに新しいoptimizerを作る。

### SVD前後では「weight値は引き継ぐ」がParameterは新しい

SVD圧縮では、学習済みweightをランダムに初期化し直すのではない。

```text
学習済みweight
↓ SVD
低rank因子
↓ copy_
新しいLinear / Conv2dのweight
```

と値を引き継ぐ。

しかし、層を1つから2つへ置換すると、PyTorch上では別の `Parameter` オブジェクトが作られる。

```text
SVD前
1つのLinear / Conv2dのParameter

SVD後
1層目のParameter
+
2層目のParameter
```

元optimizerは古いParameterへの参照を持つため、新しい因子weightを知らない。

Adam / AdamWはparameterごとに、

```text
step
exp_avg
exp_avg_sq
```

などの内部状態も持つ。

したがって、構造変更後は、

```python
optimizer = torch.optim.Adam(
    compressed_model.parameters(),
    lr=...,
)
```

と再作成する。

```text
weightはSVDから継承
optimizerは継承しない
```

と覚えると分かりやすい。

同じParameterを保ったまま構造を変えない学習フェーズなら、optimizerを再利用できる場合もある。ただしlearning rateやoptimizer stateを新しいphaseとしてリセットしたい場合は作り直す。

---

## 13. `random_split` とseed

Datasetを重ならないSubsetへ分けるには `random_split` を使える。

```python
train, val = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(0),
)
```

分割generatorを固定すると、毎回同じサンプル集合へ分割できる。

---

## 14. 再現性のために固定する乱数源

典型的には次を固定する。

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

CUDAではさらに、

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

などを使う。
完全なbit-level再現はhardware・PyTorch/CUDA version・非決定論的operatorにも依存する。

`DataLoader(shuffle=True)` も独自Generatorを渡すと、シャッフル順を管理しやすい。

### `set_seed()` とDataLoader専用Generatorは別

たとえば、

```python
loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=loader_generator,
    ...,
)
```

とした場合、`train_loader` は専用の `loader_generator` を参照する。

一般的な、

```python
set_seed(SEED)
```

でPython / NumPy / PyTorchのglobalな乱数状態を戻しても、**すでに作った専用 `loader_generator` の現在位置は自動では巻き戻らない**。

候補rankを公平に比較し、各候補で同じshuffle系列から始めたい場合は、候補学習の直前に、

```python
set_seed(SEED)
loader_generator.manual_seed(SEED)
```

の両方を行う。

DataLoaderそのものを作り直す必要はない。同じGeneratorオブジェクトをDataLoaderが参照しているため、そのGeneratorの乱数状態をresetすれば、次にiteratorを作るときからreset後の系列を使う。

### 毎epochでresetしない

候補モデルの学習開始前に一度resetした後は、epochごとにGeneratorが進んでよい。

```text
候補A epoch1 → shuffle A1
候補A epoch2 → shuffle A2
候補A epoch3 → shuffle A3

候補B開始前にmanual_seed(SEED)
↓
候補B epoch1 → A1と同じ系列
候補B epoch2 → A2と同じ系列
候補B epoch3 → A3と同じ系列
```

毎epoch `manual_seed(SEED)` すると、各epochで同じ順番を繰り返すため、通常の比較実験では行わない。

`num_workers=0` ではこの管理が比較的単純である。workerを複数使い、`__getitem__` 内でNumPy / Python randomやランダムaugmentationを使う場合は、worker seedも別途考える。

---

## 15. Notebook再実行と乱数状態

Notebook冒頭でseedを1回固定しても、同じKernel内で学習セルだけ再実行すると乱数状態はすでに進んでいる。

正式な再現結果を得るときは、

```text
Restart Kernel
→ Run All
```

を基本にする。

fine-tuning前のモデルを直接更新するコードでは、セル再実行が「同じ実験の再実行」ではなく「追加fine-tuning」になることがある。
モデルをdeepcopyしてから学習する。

---

## 16. AccuracyとCross Entropy Loss

Accuracyは最終予測クラスが正解かだけを見る離散指標である。
Cross Entropy Lossは正解クラスへどれだけ確信を置いているかまで反映する連続指標である。

したがって、

```text
Accuracyが高いモデル
≠
必ずlossも最小のモデル
```

である。
モデル選択規則では、どちらを主指標にするかを事前に決める。

---

## 17. 実験コードの役割分担

Fashion-MNISTのSVD実験では、次の分担が扱いやすかった。

| 処理 | 主に使うもの |
|---|---|
| Dataset / NN学習 / SVD | PyTorch |
| rank候補・結果表 | pandas |
| Pareto / kneeの小規模幾何計算 | pandas / NumPy |
| グラフ | Matplotlib |

候補点が数十個程度のPareto/knee計算をGPU Tensorへ移す利点は小さい。
GPUへの転送やkernel起動のoverheadの方が大きくなりうる。

---

## 18. 関連ノート

- [[00_基礎理論/05_圧縮率とRank]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]
- [[20_FASHION_MNIST_MLP_SVD/01_Fashion-MNIST実験]]
