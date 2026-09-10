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

今回のcanonical CIFAR-10実験では、

```text
train
  RandomCrop(32, padding=4)
  RandomHorizontalFlip()
  ToTensor()
  Normalize((0.5,)*3, (0.5,)*3)

Early-Stopping Validation / Rank-Selection Validation / test
  ToTensor()
  Normalize((0.5,)*3, (0.5,)*3)
```

とした。

学習時だけランダムaugmentationを入れ、評価側では入力条件を固定する。

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

このため、Fashion-MNISTでSVD圧縮の基本パイプラインを確認したあと、CIFAR-10でより難しい入力と複数Conv層のrank allocationへ進んだ。

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

を受け取る。

> [!note]
> `ToTensor()` と `Normalize()` は別の処理。`ToTensor()` はTensor化・軸順変更・典型的uint8画像の0–1化を行い、中心化・スケーリングは `Normalize()` が担当する。

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

訓練集合のchannel別統計を使う一例は、

```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

である。統計量は算出方法や前処理に依存するため、0.5固定と実測統計のどちらを採用したかを実験条件として記録する。

重要なのは、学習時と評価時で同じNormalizeを使うこと。

```text
train
Early-Stopping Validation
Rank-Selection Validation
test
```

で入力スケールを揃える。

> [!important]
> 前処理条件はモデルと切り離して考えない。学習時と評価時でNormalizeが変わると、モデル差ではなく入力分布差が評価へ混ざる。

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

32×32画像の上下左右に4 pixelずつpaddingすると、一時的に40×40相当になる。
そこからランダムに32×32を切り出す。

```text
32×32
↓ padding=4
40×40
↓ random crop
32×32
```

物体位置の小さなずれに対する頑健性を学習させる。

## `RandomHorizontalFlip()`

デフォルトでは確率0.5で左右反転する。

CIFAR-10の車・動物・船などでは、左右反転しても通常クラスは変わらないため、ラベルを保った見え方の変化として利用できる。

Fashion-MNISTでもaugmentationは可能だが、今回のCIFAR-10では背景・姿勢・位置の変化が大きいため、RandomCrop + HorizontalFlipの意味がより分かりやすい。

---

# 5. augmentationで画像枚数は増えるか

登録されているDatasetの件数は増えない。

```text
len(train_dataset)
```

は元のtrain subsetの件数のまま。

一方、サンプルを取り出すたびにランダム変換が実行されるため、同じ元画像でもepochごとに異なるTensorになり得る。

```text
元画像 x
  ├─ epoch 1: 左寄りcrop + flipなし
  ├─ epoch 2: 右寄りcrop + flipあり
  └─ epoch 3: 上寄りcrop + flipなし
```

つまり、

```text
元画像ファイル数は増えない
Dataset長も増えない
学習中に見る入力バリエーションは増える
```

というon-the-fly / online augmentationである。

---

# 6. `shuffle` とaugmentationは別物

```text
shuffle=True
→ 画像を取り出す順番を変える

RandomCrop / HorizontalFlip
→ 取り出した画像の見え方を変える
```

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

DataLoaderが「どのindexをいつ読むか」を決め、Dataset側のtransformが「その画像をどう変換するか」を決める。

---

# 7. validation / testへランダムaugmentationを入れない理由

評価時は、

```python
evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])
```

を使う。

`RandomCrop` や `RandomHorizontalFlip` を入れないのは、同じ評価画像が実行ごとに変化することを避けるため。

SVD圧縮前後を比較する際、評価入力までランダムに変わると、

```text
accuracy差
=
モデル差？
入力差？
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

同じ公式train 50,000枚を、transformだけ変えて2つのDatasetオブジェクトとして読む。

```python
full_train_augmented = datasets.CIFAR10(
    root=data_dir,
    train=True,
    transform=train_transform,
    download=...,
)

full_train_evaluation = datasets.CIFAR10(
    root=data_dir,
    train=True,
    transform=evaluation_transform,
    download=...,
)
```

元画像は同じでも、

```text
train_indices
→ full_train_augmented
→ augmentationあり

validation indices
→ full_train_evaluation
→ augmentationなし
```

とできる。

この設計により、

- trainだけaugmentationあり
- validationは固定transform
- 元画像集合の重複はindex分割で防止

を同時に満たせる。

---

# 9. `Subset` の意味

```python
train_dataset = Subset(
    full_train_augmented,
    train_indices,
)
```

は、新しい画像をコピーして作る処理ではない。

```text
train_dataset[0]
↓
train_indices[0]
↓
full_train_augmented[そのindex]
```

のように、参照可能なindexを制限する。

したがって、

```text
どの画像を使うか
→ indices / Subset

どう前処理するか
→ 親Datasetのtransform
```

は別々の責務である。

---

# 10. split用Generatorと`randperm`

データ分割は専用Generatorで固定する。

```python
split_generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    len(full_train_augmented),
    generator=split_generator,
).tolist()
```

`torch.randperm()` は `0 ... n-1` を重複なしで並べ替える。
同じ順列を区間で切れば、split間で同じindexを重複させずに分けられる。

## 単純な45,000 / 5,000例

Perplexityで前処理を学んだ段階では、公式trainを、

```text
45,000 train
5,000 validation
```

へ分ける単純な例を使った。

```python
train_indices = indices[:45_000]
validation_indices = indices[45_000:50_000]
```

これはDataset / Subset / Generatorの仕組みを理解するための例として有効。

## canonical CIFAR-10実験は40,000 / 5,000 / 5,000

ただし、最終的なcorrected Notebookではモデル選択をより厳密に分離している。

```text
公式train 50,000
├─ Train                       40,000
├─ Early-Stopping Validation    5,000
└─ Rank-Selection Validation    5,000

公式test                         10,000
```

canonical Notebookでは、

```python
TRAIN_SIZE = 40_000
VALIDATION_SIZE = 5_000

train_indices, validation_indices_early_stop, validation_indices_rank = (
    shuffled_index_splits(
        len(full_train_augmented),
        (TRAIN_SIZE, VALIDATION_SIZE, VALIDATION_SIZE),
        seed=SEED,
    )
)
```

という役割分離を使う。

```text
Train
→ weight更新

Early-Stopping Validation
→ best epoch / best state

Rank-Selection Validation
→ rank sweep / Pareto / knee / candidate選択

Test
→ 最終モデル確定後だけ
```

したがって、**45k/5kは基礎説明用の例、40k/5k/5kが今回のcanonical実験条件**である。

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

```text
num_workers=0
→ main processで読み込む

num_workers>0
→ worker processを使って並列に準備する
```

Windows / Jupyterではmulti-process DataLoaderの扱いが環境依存になりやすいため、実験を単純化したい段階では0から始めるのが安全。

## `pin_memory`

GPUへTensorを転送する際の効率改善に使える設定。

ただし、DataLoader高速化とモデルforward latencyは別物なので、benchmarkでは何を計測対象に含めるかを明示する。

---

# 12. 学習用DataLoader Generator

`shuffle=True` の順序を再現可能にするには、DataLoader専用Generatorを使える。

```python
loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
)
```

candidate間でFine-tuning条件を揃える場合は、候補開始前に、

```python
set_seed(SEED)
loader_generator.manual_seed(SEED)
```

の両方を戻す。

また、train accuracyを測るために学習用 `shuffle=True` loaderを追加走査するとGenerator状態が進むため、評価には `shuffle=False` の `train_eval_loader` を使う。

詳細は [[00_基礎理論/04_実験設計/19_再現性と乱数管理]] を参照。

---

# 13. SVD圧縮実験で固定するもの

圧縮前後の比較では、少なくとも、

```text
train / validation / testの分割
validation / testのevaluation_transform
Normalize
評価input_batch
model.eval()
inference_mode / no_grad
benchmark warmup / repeats
```

を揃える。

rank選択にtestを使わない。

```text
Train
→ 学習

Validation
→ rank / candidate / Fine-tuning条件を選択

Test
→ 最終選択後にだけ評価
```

この分離は [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]] と [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] へつながる。

---

# 14. CIFAR-10のデータ取得で詰まったとき

Perplexityで実際に確認したデータ取得時の要点も、再現用メモとして残す。

## 一度取得できたら`download=False`

CIFAR-10が `data_dir` に正しく展開済みなら、以後は、

```python
full_train_augmented = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=False,
    transform=train_transform,
)

full_train_evaluation = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=False,
    transform=evaluation_transform,
)

test_dataset = datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=False,
    transform=evaluation_transform,
)
```

としてよい。

まだデータが存在しない状態で `download=False` にすると、Dataset not found / corrupted系のエラーになる。その場合だけ初回取得を行う。

## 展開後の配置

Torchvisionから読むときは、概念的に次の構成になっていることを確認する。

```text
data_dir/
└─ cifar-10-batches-py/
   ├─ data_batch_1
   ├─ data_batch_2
   ├─ data_batch_3
   ├─ data_batch_4
   ├─ data_batch_5
   ├─ test_batch
   └─ ...
```

## 自動downloadが極端に遅い場合

Notebookからの自動取得が長時間進まない場合は、コードを待ち続けるより、ブラウザ等でPython版アーカイブを手動取得して配置する方法もある。信頼できるmirrorから取得した場合も、ファイル内容が公式版と同一かをhashで確認する。

取得した `cifar-10-python.tar.gz` が正しいか確認するMD5は、Perplexityで確認した値では、

```text
c58f30108f718f92721af3b95e74349a
```

である。

Windows PowerShellなら、

```powershell
# Downloadsに保存したCIFAR-10 Python版archiveを検査する。
$archive = Join-Path `
    $env:USERPROFILE `
    'Downloads\cifar-10-python.tar.gz'
$expectedMd5 = 'c58f30108f718f92721af3b95e74349a'
$actualMd5 = (
    Get-FileHash -LiteralPath $archive -Algorithm MD5
).Hash.ToLowerInvariant()

if ($actualMd5 -ne $expectedMd5) {
    throw "MD5 mismatch: $actualMd5"
}

Write-Output "MD5 verified: $actualMd5"
```

MD5が一致したら展開し、`data_dir/cifar-10-batches-py/` になるよう配置して、以後 `download=False` で読む。hash不一致は不完全なダウンロードや別内容の可能性を示すため、そのarchiveは展開せず再取得する。mirrorの速度は環境依存なので、教材では特定mirrorを固定せず公式hashを判断基準にする。

> [!note]
> この節はSVD理論ではなく、実際のCIFAR-10データ準備で詰まった際の再現用トラブルシュート。実験結果そのものには含めない。

---

# 15. Fashion-MNISTとの違いと次の学習

```text
Fashion-MNIST
→ 圧縮パイプラインの原理確認
→ Linear / Convを個別に圧縮

CIFAR-10
→ RGB自然画像
→ augmentationと評価transformを分離
→ 複数Conv層のrank allocation
```

ここで、単一層のrank選択からモデル全体のrank配分へ問題が広がる。

---

# 関連

- [[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
