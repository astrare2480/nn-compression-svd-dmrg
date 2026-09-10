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

### `num_workers`・`pin_memory`・`persistent_workers`

DataLoaderの並列数とCPUからGPUへの転送条件まで含めると、例えば次のように書ける。

```python
from torch.utils.data import DataLoader

NUM_WORKERS = 0
use_cuda = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda,
    persistent_workers=(NUM_WORKERS > 0),
)
```

- `num_workers=0`：main processがDataset取得・transform・batch化を行う
- `num_workers>0`：worker processがデータ準備を並列化する
- `pin_memory=True`：CPU側batchをpage-locked memoryへ置き、CUDA転送を効率化しやすくする
- `persistent_workers=True`：epoch終了後もworkerを維持する。`num_workers=0`では指定できない

`pin_memory=True`と対応させる場合、学習ループ側は

```python
images = images.to(device, non_blocking=use_cuda)
labels = labels.to(device, non_blocking=use_cuda)
```

と書ける。ただし非同期転送の効果はhardware・処理順・batch sizeにも依存する。`num_workers`も大きいほど必ず速いわけではない。Windows / Jupyterで最初に動作確認するときは `num_workers=0` から始め、必要ならスクリプト実行で2、4などを実測する。

### `drop_last`とdefault collate

データ数$N$がbatch size $B$で割り切れない場合、`drop_last`は最後の端数batchを使うか捨てるかを決める。例えば$N=100$、$B=32$なら、

```text
drop_last=False
→ 32, 32, 32, 4
→ 100 sampleすべてを使う

drop_last=True
→ 32, 32, 32
→ 最後の4 sampleを使わない
```

となる。学習時に極端に小さい最終batchを避けたい場合は `True` が候補になる。一方、validation / testでは全sampleを評価するため、通常は `False` とする。`drop_last=True` を使ったときは、1 epochで実際に学習へ使ったsample数も区別する。

Datasetが1 sampleごとに

```text
(image, label)
```

を返すと、DataLoaderのdefault collateは複数sampleを自動的にまとめる。例えば各画像が `(3, 32, 32)` なら、

```text
32個のimage
→ images.shape = (32, 3, 32, 32)

32個のscalar label
→ labels.shape = (32,)
```

となる。`default_collate`の動作は、小さい入力なら次のように直接確認できる。

```python
import torch
from torch.utils.data import default_collate

# Datasetが返す32 sample分の(image, label)を用意する。
samples = [
    (
        torch.full((3, 32, 32), float(index)),
        index,
    )
    for index in range(32)
]

images, labels = default_collate(samples)

# 同じshapeの画像には新しいbatch軸が追加される。
assert images.shape == (32, 3, 32, 32)
assert labels.shape == (32,)
assert torch.equal(images[5], samples[5][0])
assert labels[5].item() == samples[5][1]
```

画像Tensorをまとめる部分は、概念的には `torch.stack([image0, image1, ...], dim=0)` に対応する。可変長系列や画像shapeがsampleごとに異なる場合は、そのまま `torch.stack` できないため、paddingなどを行う独自の `collate_fn` が必要になる。

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

### Dataset classとtransformを対応させる

Dataset classは `torchvision.datasets`、画像変換は `torchvision.transforms` から読み込む。

```python
from torchvision import datasets, transforms

to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

fashion_train = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=to_tensor,
    download=True,
)
```

Dataset classが変わっても、`transform`を渡し、各indexから基本的に `(image, label)` を受け取る構造は共通である。CIFAR-10固有の画像shape、データ分割、augmentation、ダウンロードと整合性確認は [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]] で扱う。

### `Normalize`の計算と値の選び方

`transforms.Normalize(mean, std)`は、channel $c$ ごとに

$$
x'_{c,h,w}
=
\frac{
x_{c,h,w}-\mu_c
}{
\sigma_c
}
$$

を適用する。1 channel画像で

```python
mean = (0.5,)
std = (0.5,)
```

なら、各channelで

$$
x'
=
\frac{x-0.5}{0.5}
=
2x-1
$$

となり、`ToTensor()` 後の代表値は

```text
x = 0.0  → x' = -1.0
x = 0.5  → x' =  0.0
x = 1.0  → x' =  1.0
```

へ移る。これは$[0,1]$を$[-1,1]$へ移す分かりやすい固定変換であり、訓練集合の実測平均を厳密に0、標準偏差を1へする指定ではない。複数channelではchannelごとに $\mu_c,\sigma_c$ を指定する。統計量は算出方法や前処理に依存するため、採用値を実験条件として記録し、train / validation / testで同じ値を使う。

### 入力の`Normalize`とモデル内の`BatchNorm2d`は別

`transforms.Normalize`はDataset側で、固定したmean・stdを使って入力画素を変換する。一方、`nn.BatchNorm2d`はモデル内部の特徴量へ作用し、学習時のbatch統計と評価時のrunning statisticsを使い分ける。両者は置き換え関係ではない。

```text
transforms.Normalize
→ モデルへ入る前の入力前処理

nn.BatchNorm2d
→ モデル内部の学習可能な層
```

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

### `backward()`と`optimizer.step()`を数式へ対応させる

重みをまとめて$W$、損失を$L(W)$とする。$W$を$\Delta W$だけ動かしたとき、1次のTaylor展開は

$$
L(W+\Delta W)
\approx
L(W)
+
\left\langle
\nabla L(W),
\Delta W
\right\rangle_F
$$

である。勾配の反対方向へ

$$
\Delta W
:=
-\eta\nabla L(W),
\qquad
\eta>0
$$

と動かすと、

$$
\begin{aligned}
L(W+\Delta W)
&\approx
L(W)
-
\eta
\left\langle
\nabla L(W),
\nabla L(W)
\right\rangle_F\\
&=
L(W)
-
\eta
\lVert\nabla L(W)\rVert_F^2\\
&\le
L(W).
\end{aligned}
$$

学習率が局所近似に対して十分小さければ、負号によって損失が下がる方向へ進むことが分かる。PyTorchでは、

```text
forward
→ lossを計算して計算グラフを作る
→ zero_gradで前回のgradを消す
→ backwardで∂L/∂Wを各Parameter.gradへ計算する
→ optimizer.stepでgradを使ってParameterを更新する
```

と対応する。`backward()`は勾配を計算する処理であり、実際に重みを変更するのは`optimizer.step()`である。

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

上のコードは絶対改善量

$$
\Delta_{\mathrm{abs}}
:=
L_{\mathrm{best}}-L_t
$$

について

$$
\Delta_{\mathrm{abs}}>\delta_{\mathrm{abs}}
$$

を改善条件にする。lossのscaleが実験間でほぼ同じなら、この条件は読みやすい。

lossのscale自体が大きく変わる比較では、相対改善率

$$
\Delta_{\mathrm{rel}}
:=
\frac{
L_{\mathrm{best}}-L_t
}{
\max\left(
|L_{\mathrm{best}}|,
\epsilon
\right)
}
$$

を使い、

$$
\Delta_{\mathrm{rel}}>\delta_{\mathrm{rel}}
$$

と判定する方法もある。学習のEarly Stoppingはbest lossからの改善を判定する。一方、HOOIの収束判定は連続するiteration間の誤差変化を判定するため、同じ「停止条件」でも比較対象が異なる。

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

### 新しいfine-tuningと中断した学習の再開は別

rank候補を独立に比較するfine-tuningでは、候補ごとに新しいoptimizerとEarly Stopping状態を作る。一方、同じ学習を中断地点から継続する場合は、モデルのweightだけでなくoptimizerの内部状態も復元する。

```text
新しいrank候補のfine-tuning
→ 同じbaselineからモデルを構築
→ 新しいoptimizerを作る
→ Early Stopping状態も初期化

中断した同一学習の再開
→ model stateを復元
→ optimizer stateを復元
→ epoch位置を復元
```

再開用checkpointは、例えば次のように保存する。

```python
# 同じ学習を中断地点から再開するための状態をまとめて保存する。
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
    },
    checkpoint_path,
)
```

読み込み時は、同じモデル構造とoptimizerを作ってから状態を戻す。

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])

# 保存時と同じ種類のoptimizerを作成してから内部状態を復元する。
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

start_epoch = int(checkpoint["epoch"]) + 1
```

learning-rate schedulerやAMPの `GradScaler` を使っていた場合は、それらのstateも保存・復元する。厳密な再開が必要なら、Python・NumPy・PyTorchおよびDataLoader Generatorの乱数状態もcheckpoint対象になる。最良モデルを評価・推論へ使うだけならmodel stateの復元でよく、「同じ学習履歴を継続する場合」と区別する。

### transfer learningのfreeze / unfreeze

transfer learningでは、事前学習済みbackboneを一時的に固定し、新しい分類headだけを先に学習することがある。例えば `model.features` をbackbone、`model.classifier` を分類headとすると、

```python
# backboneでは勾配を計算せず、分類headだけを学習対象にする。
for parameter in model.features.parameters():
    parameter.requires_grad = False

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=1e-3,
)
```

と書ける。`requires_grad=False`にしたParameterでは、通常のbackwardで勾配を計算しない。

分類headの学習後にbackboneの最後のblockも調整するなら、そのblockをunfreezeする。

```python
# backboneの最後のblockを再び学習可能にする。
for parameter in model.features[-1].parameters():
    parameter.requires_grad = True

# 新しい学習phaseとしてoptimizerを作り直し、
# 分類headとbackboneで異なるlearning rateを設定する。
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.classifier.parameters(),
            "lr": 1e-3,
        },
        {
            "params": model.features[-1].parameters(),
            "lr": 1e-4,
        },
    ],
)
```

最初のoptimizerへ分類headしか渡していなかった場合、`requires_grad=True`へ戻すだけではunfreezeしたParameterはoptimizerへ登録されない。作り直さない場合は、

```python
# 既存optimizerへ、unfreezeしたParameterを新しいgroupとして追加する。
optimizer.add_param_group(
    {
        "params": model.features[-1].parameters(),
        "lr": 1e-4,
    }
)
```

と追加できる。一方、最初から `model.parameters()` 全体をoptimizerへ渡していた場合は、Parameter object自体は登録済みなので、`requires_grad=True`へ戻すと更新対象になり得る。ただし学習phaseごとにlearning rateやAdamWの内部状態を分離したいなら、optimizerを作り直す方が意図を明確にできる。

SVD / Tucker圧縮との違いは、Parameter objectが同じかどうかである。

```text
freeze / unfreeze
→ 同じParameter objectのrequires_gradを切り替える

SVD / Tuckerによる層置換
→ 新しい層と新しいParameter objectを作る
→ 原則として置換後にoptimizerを作り直す
```

したがって、transfer learningのoptimizer再設定と圧縮後のoptimizer再作成は、見た目は似ていても理由が異なる。

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

- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[00_基礎理論/05_PyTorch実装/09_Linear層の2層置換_実装]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]
- [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]]
- [[20_FashionMNIST/01_Fashion-MNIST実験]]
