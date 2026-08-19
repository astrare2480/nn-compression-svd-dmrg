---
title: CIFAR-10の前処理とDataLoader
aliases:
  - CIFAR10前処理
  - CIFAR10 DataLoader
  - Data Augmentation
  - ToTensorとNormalize
  - train validation test分離
tags:
  - CIFAR10
  - CNN
  - PyTorch
  - DataLoader
  - DataAugmentation
  - Normalize
  - 再現性
---

# CIFAR-10の前処理とDataLoader

## サマリー

CIFAR-10は、32×32 pixel・RGB 3チャネルの画像を10クラスへ分類するデータセットである。

```text
1画像: (C, H, W) = (3, 32, 32)
mini-batch: (N, 3, 32, 32)
```

SVD圧縮実験では、モデルだけでなく**入力側の条件を固定すること**が重要になる。

今回のCIFAR-10実験では、

```text
train
  RandomCrop(32, padding=4)
  RandomHorizontalFlip()
  ToTensor()
  Normalize((0.5,)*3, (0.5,)*3)

validation / test
  ToTensor()
  Normalize((0.5,)*3, (0.5,)*3)
```

とした。

学習時だけランダムaugmentationを入れ、validation / testでは評価入力を固定する。

---

# 1. CIFAR-10とは

CIFAR-10は10クラスのカラー画像分類データセットである。

| 項目 | 内容 |
|---|---|
| 画像サイズ | 32×32 pixel |
| チャネル | RGB 3チャネル |
| 総画像数 | 60,000 |
| 公式train | 50,000 |
| 公式test | 10,000 |
| クラス数 | 10 |

クラスは、

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

である。

Fashion-MNISTと比べると、背景・色・向き・位置のばらつきが大きく、自然画像に近い。

```text
Fashion-MNIST
28×28 / grayscale / 商品画像中心

CIFAR-10
32×32 / RGB / 背景や姿勢の変化が大きい
```

このため、Fashion-MNISTでSVD圧縮の基本パイプラインを確認したあと、CIFAR-10でより厳しい条件へ進む流れにした。

---

# 2. `ToTensor()`

`transforms.ToTensor()` は、典型的なuint8画像について、

```text
(H, W, C)
↓
(C, H, W)
```

へ軸順を変え、画素値も概ね、

```text
0 ... 255
↓
0.0 ... 1.0
```

へ変換する。

CIFAR-10なら、

```text
(32, 32, 3)
↓
(3, 32, 32)
```

となる。

PyTorchの `nn.Conv2d` は通常、batchを含めて、

```text
(N, C, H, W)
```

を受け取るので、このCHW化はCNN入力の基本である。

> [!note]
> `ToTensor()` と `Normalize()` は別の処理。`ToTensor()` はTensor化・軸順変更・典型的uint8画像の0–1化を行い、平均0付近へ寄せる処理は `Normalize()` が担当する。

---

# 3. `Normalize()`

今回の実験では、

```python
transforms.Normalize(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
```

を使用した。

各RGBチャネルについて、

$$
x' = \frac{x - 0.5}{0.5} = 2x - 1
$$

を適用するため、`ToTensor()` 後の値域 `[0,1]` は概ね `[-1,1]` になる。

これはCIFAR-10の実測mean/stdで厳密に標準化する方式ではなく、**0.5を使った単純な中心化・スケーリング**である。

重要なのは、学習時と評価時で同じNormalizeを使うこと。

```text
train時     [-1, 1] 相当
validation  [-1, 1] 相当
test        [-1, 1] 相当
```

と揃える。

学習時だけNormalizeして評価時に外すと、モデルが学習した入力分布と評価入力の分布が変わってしまう。

> [!important]
> 正規化条件は「モデル単体」ではなく、**モデルと学習時の前処理の組**として扱う。事前学習済みモデルを使う場合も、提供元が想定する前処理へ合わせる必要がある。

---

# 4. 学習時だけaugmentationを入れる理由

今回のtrain transformは、

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])
```

である。

## `RandomCrop(32, padding=4)`

32×32画像の上下左右に4 pixelずつpaddingするため、一時的には、

```text
32 + 4 + 4 = 40
```

で、40×40相当になる。

そこからランダムに32×32を切り出す。

```text
32×32
↓ padding=4
40×40
↓ random crop
32×32
```

物体の位置が少しずれても同じクラスとして認識できるようにする。

## `RandomHorizontalFlip()`

デフォルトでは確率0.5で左右反転する。

CIFAR-10の車・動物・船などでは、左右を反転しても通常クラスは変わらないため、クラスを保った見え方の変化として使える。

---

# 5. augmentationで「画像枚数」は増えるか

登録されているDatasetの枚数は増えない。

```text
len(train_dataset)
```

は元のtrain subsetの件数のまま。

一方、サンプルを取り出すたびにランダム変換が実行されるため、同じ元画像でもエポックごとに異なるTensorになり得る。

```text
元画像 x
  ├─ epoch 1: 左寄りcrop + flipなし
  ├─ epoch 2: 右寄りcrop + flipあり
  └─ epoch 3: 上寄りcrop + flipなし
```

したがって、これは**on-the-fly / online augmentation**として、実質的な入力バリエーションを増やす処理と考える。

```text
元画像ファイル数は増えない
Dataset長も増えない
しかし学習中に見るTensorの見え方は増える
```

---

# 6. `shuffle` とaugmentationは別物

混同しやすいので分ける。

```text
shuffle=True
→ 画像を取り出す順番を変える

RandomCrop / HorizontalFlip
→ 取り出した画像の内容を変える
```

DataLoaderがindexを選び、Datasetの `__getitem__()` がそのindexの画像を取り出すときにtransformが適用される。

概念的には、

```text
DataLoader
  ↓ indexを選ぶ
Dataset[index]
  ↓ 元画像を取得
transform
  ↓ augmentation / ToTensor / Normalize
変換済みTensor
  ↓ batch化
CNN
```

となる。

---

# 7. validation / testへランダムaugmentationを入れない理由

評価時には、

```python
evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])
```

を使用する。

`RandomCrop` や `RandomHorizontalFlip` を入れないのは、同じ評価画像が実行ごとに変化することを避けるため。

SVD圧縮前後を比較する際、評価入力まで変わってしまうと、

```text
accuracy差
=
モデル差？
入力のランダム差？
```

を切り分けにくい。

したがって、

```text
評価データ集合
評価transform
Normalize条件
DataLoaderのshuffle=False
```

を固定する。

---

# 8. 同じ公式trainを2つのDatasetオブジェクトとして読む理由

今回のように、同じ公式train 50,000枚からtrainとvalidationを作りたい場合、

```python
full_train_augmented = datasets.CIFAR10(
    ...,
    train=True,
    transform=train_transform,
)

full_train_evaluation = datasets.CIFAR10(
    ...,
    train=True,
    transform=evaluation_transform,
)
```

と、**元画像は同じだがtransformだけ違うDatasetオブジェクト**を2つ作る。

そのうえで同じindex体系を使い、

```text
train_indices
→ full_train_augmentedを参照

validation_indices
→ full_train_evaluationを参照
```

とする。

こうすると、

- trainはaugmentationあり
- validationはaugmentationなし
- 元画像集合の重複はindex分割で防ぐ

を同時に満たせる。

---

# 9. `Subset` の意味

例えば、

```python
train_dataset = Subset(
    full_train_augmented,
    train_indices,
)
```

は、元Datasetをコピーして新しい画像を作る処理ではない。

```text
train_dataset[0]
↓
train_indices[0] を見る
↓
full_train_augmented[そのindex]
```

という**参照するindexを制限したviewに近い役割**を持つ。

したがって、「どの画像を使うか」と「どのtransformを使うか」は別々に決まる。

```text
どの画像？
→ train_indices / validation_indices

どう前処理？
→ full_train_augmented / full_train_evaluation の transform
```

---

# 10. train / validation分割

専用Generatorを使ってindex順列を作る。

```python
split_generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(full_train_augmented),
    generator=split_generator,
).tolist()
```

例えば45,000 / 5,000へ分けるなら、

```python
train_indices = indices[:45_000]
validation_indices = indices[45_000:50_000]
```

となる。

`torch.randperm()` は0〜n-1を重複なしで並べ替えるので、同じ順列を前後に分割すれば、train / validationの重複を避けられる。

専用の `split_generator` を使うのは、データ分割の乱数を、

- weight初期化
- Dropout
- DataLoader shuffle
- augmentation

など別の乱数利用から切り離すためでもある。

---

# 11. DataLoader設定

概念的には、

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
```

とする。

## `num_workers`

DataLoaderがデータ取得を並列化するworker数。

```text
num_workers=0
→ main processで読み込む

num_workers>0
→ worker processを使って並列に準備する
```

Windows / Jupyterではmulti-process DataLoaderの扱いが環境依存になりやすいため、実験を単純化したい段階では0から始めるのが安全。

## `pin_memory`

GPUへTensorを転送する際の効率改善に使える設定。

ただし、SVD実験ではDataLoader高速化そのものとモデル推論latencyを混同しない。

---

# 12. SVD圧縮実験で固定するもの

圧縮前後の比較では、少なくとも、

```text
train / validation / testの分割
validation / testのevaluation_transform
Normalize
評価batch
model.eval()
inference_mode / no_grad
```

を揃える。

さらにrank選択ではtestを使わない。

```text
train
→ 重みを学習

validation
→ rank / candidate / fine-tuning条件を選択

test
→ 最終選択後に1回確認
```

この分離は、[[00_基礎理論/10_SVD圧縮モデルの評価設計]] と [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] へつながる。

---

# 13. Fashion-MNISTとの違い

Fashion-MNISTでもaugmentationは使えるが、今回の学習ではCIFAR-10ほど強く必要としなかった。

CIFAR-10は、

- RGB
- 背景の変化
- 物体位置の変化
- 左右方向の変化

が大きいため、RandomCrop + HorizontalFlipを使う意味が分かりやすい。

SVD圧縮の観点では、

```text
Fashion-MNIST
→ 圧縮パイプラインの原理確認

CIFAR-10
→ より難しい画像で複数Conv層のrank allocationを検証
```

という役割分担になった。

---

# 関連

- [[00_基礎理論/15_CNNとConv2dの基礎]]
- [[00_基礎理論/19_再現性と乱数管理]]
- [[00_基礎理論/20_Global Average PoolingとCIFAR10モデル設計]]
- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
