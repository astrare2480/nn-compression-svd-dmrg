# SVDによるNN重み行列圧縮に向けた学習順序

## 前提

- 『ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装』は読了済み
- AI実装検定A級を保有
- 現在はPythonデータ分析認定試験の勉強を優先
- 最終的には、SVD / Tensor Network / DMRG をNN圧縮に応用したい
- まずは、MNISTのMLPに対してSVDによる重み行列圧縮を試す

---

## 結論

最初にやるべきことは、テンソルネットワークやDMRGではなく、**PyTorchでMNISTのMLPを作り、Linear層の重みをSVDで低ランク圧縮すること**。

テンソルネットワーク本は、現時点では不要。  
SVD圧縮の実験が一通り動いてから読む方がよい。

---

# 全体ロードマップ

```text
Phase 0: Pythonデータ分析認定試験
↓
Phase 1: PyTorchの最低限
↓
Phase 2: MNIST / Fashion-MNISTでMLPを作る
↓
Phase 3: 学習済みLinear層の重みをSVDする
↓
Phase 4: Linear層を低ランク2層に置き換える
↓
Phase 5: rankごとの精度・パラメータ数を比較する
↓
Phase 6: fine-tuningで精度回復を確認する
↓
Phase 7: CNN / Fashion-MNIST / CIFAR-10へ拡張
↓
Phase 8: Tensor Train / MPS / DMRGへ進む
```

---

# Phase 0: Pythonデータ分析認定試験を優先

まずはPythonデータ分析認定試験を終わらせる。

この段階で重点的にやること。

```text
numpy:
- shape
- 行列積
- ブロードキャスト
- 統計量
- 線形代数の基礎

pandas:
- 実験結果CSVの読み書き
- DataFrame操作
- groupby
- 集計

matplotlib:
- rank vs accuracy
- compression ratio vs accuracy
- params vs accuracy
```

SVD圧縮の実験でも、最終的に以下のような表を作る。

| rank | params | compression_ratio | accuracy | accuracy_drop |
|---:|---:|---:|---:|---:|
| full | 100% | 1.0x | 98.20% | 0.00 |
| 256 | 70% | 1.4x | 98.15% | -0.05 |
| 128 | 40% | 2.5x | 98.00% | -0.20 |
| 64 | 22% | 4.5x | 97.60% | -0.60 |

## この段階で読むもの

| 優先度 | 教材 | 読み方 |
|---:|---|---|
| 最優先 | Pythonデータ分析認定試験の教材 | 試験対策として読む |
| 復習 | ゼロから作るDeep Learning | 誤差逆伝播、CNN、行列計算を軽く確認 |
| 不要 | PyTorch発展本 | まだ読まなくてよい |
| 不要 | Tensor Network / DMRG本 | まだ読まなくてよい |

---

# Phase 1: PyTorchの最低限をやる

Pythonデータ分析認定試験が終わったら、PyTorchに入る。

目的は、**PyTorchでMNISTのMLPを書けるようになること**。

## 公式チュートリアル日本語版でやる範囲

サイト: `https://yutaroogawa.github.io/pytorch_tutorials_jp/`

やる範囲は、まず **0. PyTorch入門（Learn the Basics）**。

順番は以下。

```text
0-[8] クイックスタート
0-[1] テンソル
0-[2] データセットとデータローダー
0-[3] データ変換
0-[4] モデル構築
0-[5] 自動微分
0-[6] 最適化
0-[7] モデルの保存・読み込み
```

## この段階で身につけるもの

```text
torch.Tensor
shape確認
nn.Module
nn.Linear
nn.Sequential
Dataset
DataLoader
training loop
model.eval()
torch.no_grad()
state_dict
torch.save
torch.load
```

## この段階で読む本

### 第一候補: 『動かしながら学ぶ PyTorchプログラミング入門』

用途: PyTorchの書き方に早く慣れるため。

読む範囲。

```text
Chapter 2 PyTorchの基本
- Tensor
- 自動微分
- ニューラルネットワークの定義
- 損失関数
- 最適化関数

Chapter 4 CNNによる画像分類
- 画像分類のPyTorch実装部分
```

全部読む必要はない。  
RNN、感情分析、アプリ開発系は後回しでよい。

### 第二候補: 『PyTorch実践入門』

用途: 長く使う辞書・補強用。

この段階では、Tensor、学習ループ、nn.Module周辺を参照する。

---

# Phase 2: MNIST / Fashion-MNISTでMLPを作る

最初はCNNではなくMLPでよい。

理由は、SVDをかける対象が明確だから。

```text
Linear層の重み: 2次元行列
Conv層の重み: 4次元テンソル
```

最初のモデル例。

```text
Input: 784
↓
Linear(784, 512)
↓
ReLU
↓
Linear(512, 256)
↓
ReLU
↓
Linear(256, 10)
```

## 最初の目標

```text
MNISTで Accuracy 97〜98%以上
```

余裕があれば次にFashion-MNISTへ進む。

```text
Fashion-MNISTで Accuracy 87〜90%以上
```

## この段階で読む本

### 『PyTorch実践入門』

この段階から、辞書・補強用として使う。

読む範囲の目安。

```text
第1章 ディープラーニングとPyTorchの概要
第3章 PyTorchにおけるテンソルの扱い方
第5章 ディープラーニングの学習メカニズム
第6章 ニューラルネットワーク入門
第7章 画像分類モデルの構築
第8章 畳み込み
```

ただし、すでにDL理論の基礎はあるので、精読ではなく、**PyTorchではどう書くか**に注目して読む。

---

# Phase 3: Linear層の重みを取り出してSVDする

学習済みモデルのLinear層から重みを取り出す。

PyTorchの `nn.Linear` の重み形状は以下。

```text
weight.shape = (out_features, in_features)
```

例えば、

```python
nn.Linear(784, 512)
```

なら、

```text
weight.shape = (512, 784)
```

SVDは以下。

```python
W = model.linear_relu_stack[0].weight.data
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
```

数学的には、

```text
W = U Σ V^T
```

PyTorchでは `V^T` が `Vh` として返る。

rank `r` で切る。

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
```

低ランク近似。

```python
W_r = U_r @ torch.diag(S_r) @ Vh_r
```

## ここで理解すること

SVDは損失関数を直接最小化しているわけではない。

SVDが解いているのは、

```text
W_r = argmin ||W - A||_F
```

つまり、重み行列 `W` をrank `r` の行列で最もよく近似する問題。

---

# Phase 4: Linear層を低ランク2層に置き換える

単に `W_r` を作って元の形に戻すだけでは、パラメータ数は減らない。

本当に圧縮するには、

```text
Linear(in, out)
```

を

```text
Linear(in, r, bias=False)
Linear(r, out, bias=True)
```

に置き換える。

元の層。

```text
y = Wx + b
```

SVD後。

```text
y ≈ U_r Σ_r V_r^T x + b
```

PyTorch上の対応。

```text
1層目 weight = Vh_r
2層目 weight = U_r @ diag(S_r)
2層目 bias = 元のbias
```

## パラメータ数

元。

```text
out × in + out
```

SVD後。

```text
r × in + out × r + out
```

例: `Linear(784, 512)` の場合。

```text
元:
784 × 512 = 401,408

rank 64:
784 × 64 + 512 × 64 = 82,944
```

重みだけなら約80%削減。

---

# Phase 5: rankごとの精度・圧縮率を比較する

最初に試すrank。

```text
r = 256
r = 128
r = 64
r = 32
r = 16
```

比較対象。

```text
Baseline
SVD rank=256
SVD rank=128
SVD rank=64
SVD rank=32
SVD rank=16
```

評価指標。

```text
Accuracy
パラメータ数
Compression ratio
Accuracy drop
```

余裕があれば追加。

```text
推論時間
モデルサイズ
```

ただし、推論時間は最初から強く主張しない方がよい。  
SVDで層を2つに分けると、理論上の演算量は減っても、実測では速くならないことがある。

---

# Phase 6: fine-tuningで精度回復を見る

SVD圧縮だけでは精度が落ちる可能性がある。

比較は必ず以下に分ける。

```text
A. Baseline
B. SVD圧縮のみ
C. SVD圧縮 + fine-tuning
```

fine-tuningは最初は短くてよい。

```text
epoch = 1
epoch = 3
epoch = 5
```

見たいこと。

```text
SVDで落ちた精度が、少数epochでどこまで戻るか
```

例。

```text
Baseline: 98.20%
rank 64 SVD only: 97.40%
rank 64 + fine-tuning: 98.05%
```

このような結果が出ると、実績として見せやすい。

---

# Phase 7: 実験結果をpandas / matplotlibで整理する

作るCSV例。

```text
results/svd_mnist_mlp.csv
```

中身の例。

```csv
model,rank,params,compression_ratio,accuracy,accuracy_drop,fine_tuned
baseline,full,669706,1.0,0.9820,0.0000,false
svd,256,450000,1.49,0.9815,-0.0005,false
svd,128,250000,2.67,0.9800,-0.0020,false
svd,64,140000,4.78,0.9760,-0.0060,false
svd_ft,64,140000,4.78,0.9805,-0.0015,true
```

作るグラフ。

```text
rank vs accuracy
compression ratio vs accuracy
params vs accuracy
fine-tuning有無によるaccuracy比較
```

ここはPythonデータ分析認定試験の知識がそのまま使える。

---

# Phase 8: READMEにまとめる

GitHub実績としてまとめる。

READMEに書く内容。

```text
目的:
学習済みMLPのLinear層をSVDで低ランク近似し、
精度を大きく落とさずにパラメータ数を削減できるかを検証する。

方法:
MNISTでMLPを学習し、Linear層をrank rの2層Linearに置換。
rankごとにAccuracy、パラメータ数、圧縮率を比較。
さらにSVD後fine-tuningによる精度回復を確認。

結果:
rank 64まで精度低下を0.5ポイント以内に抑えつつ、
パラメータ数をX%削減できた。
```

ここまでできれば、SVDを使用したNN重み行列圧縮の第一段階として十分な実績になる。

---

# Phase 9: Fashion-MNIST / CNN / CIFAR-10へ進む

MNISTは簡単なので、次はFashion-MNISTへ。

順番。

```text
Fashion-MNIST + MLP
↓
Fashion-MNIST + CNN
↓
CIFAR-10 + CNN
```

CNNでは、最初からConv層を圧縮しない。  
まずはCNNの最後の `Linear` 層だけをSVD圧縮する。

その後でConv層の圧縮に進む。

Conv層の重みは、

```text
(out_channels, in_channels, kernel_h, kernel_w)
```

なので、そのままSVDはできない。  
例えば、

```text
(out_channels, in_channels × kernel_h × kernel_w)
```

にreshapeしてSVDする方法がある。

---

# Phase 10: Tensor Train / MPS / DMRGへ進む

SVD圧縮が一通り動いた後で、Tensor Networkへ進む。

対応関係。

| 物理 / DMRG | NN圧縮 |
|---|---|
| MPS | Tensor Train層 |
| bond dimension | TT-rank |
| SVD truncation | 低ランク圧縮 |
| DMRG sweep | 局所fine-tuning |
| エネルギー最小化 | 損失関数最小化 |

この段階で初めて、DMRG経験が本格的に効いてくる。

---

# テンソルネットワーク本は必要か

## 結論

**今すぐは不要。**

理由は、現時点の最短ゴールが以下だから。

```text
PyTorchでMNISTのMLPを作る
↓
Linear層の重みをSVDする
↓
rankごとのAccuracyとパラメータ数を比較する
```

この範囲では、テンソルネットワーク本はほぼ使わない。

## 読むタイミング

テンソルネットワーク本を読むのは、以下が終わった後でよい。

```text
MNIST + MLP + SVD圧縮
SVD後fine-tuning
Fashion-MNISTで同様の検証
CNNのLinear層圧縮
```

その後に、

```text
Tensor Train層を作りたい
MPS表現のNNを試したい
DMRG的fine-tuningをしたい
```

となった段階で読む。

## 今買わなくてよい理由

```text
・SVD圧縮の初手には直接必要ない
・PyTorch実装が先
・MNIST実験が動かないと、Tensor Networkに進んでも検証できない
・読書だけ進んで実装が止まるリスクがある
```

## いつ必要になるか

必要になるのは以下の段階。

```text
SVD圧縮のベースラインができた
↓
Tensor Train / MPS層をPyTorchで実装したい
↓
bond dimensionと精度の関係を調べたい
↓
DMRG sweep的に局所テンソルを最適化したい
```

この段階なら、テンソルネットワーク本や資料を読む価値が高い。

---

# 本を読むタイミングまとめ

## 今

| タイミング | 読む本・教材 |
|---|---|
| 今 | Pythonデータ分析認定試験の教材 |
| 余裕があるとき | ゼロから作るDeep Learningの復習 |
| 読まない | PyTorch発展本、Tensor Network本 |

---

## 試験後すぐ

| タイミング | 読む本・教材 |
|---|---|
| 最初 | 公式チュートリアル日本語版「0. PyTorch入門」 |
| 補助 | 動かしながら学ぶ PyTorchプログラミング入門 |
| 辞書 | PyTorch実践入門 |

---

## MNIST MLPを作る段階

| タイミング | 読む本 |
|---|---|
| 実装前 | 公式チュートリアルのモデル構築・最適化・保存 |
| 実装中 | PyTorch実践入門のTensor / nn.Module / 画像分類 |
| 詰まったら | 動かしながら学ぶPyTorchのnn.Module周辺 |

---

## SVD圧縮を実装する段階

| タイミング | 読む本 |
|---|---|
| 実装前 | SVDの式を軽く確認 |
| 実装中 | PyTorch実践入門のTensor・nn.Module部分を参照 |
| 実験整理 | Pythonデータ分析教材を再利用 |

---

## CNNへ進む段階

| タイミング | 読む本 |
|---|---|
| Fashion-MNIST CNN | PyTorch実践入門の畳み込み周辺 |
| VGGなど既存モデル | つくりながら学ぶ! PyTorchによる発展ディープラーニング |
| 転移学習/応用 | つくりながら学ぶ! の画像分類・VGG周辺 |

---

## Tensor Network / DMRGへ進む段階

| タイミング | 読む本・資料 |
|---|---|
| SVD圧縮が一通り終わった後 | MPS / DMRG資料 |
| Tensor Train層を作りたくなったら | Tensor Network for ML系資料 |
| 研究寄りにしたくなったら | 論文・サーベイ |

---

# 買う優先度

## 今すぐ買うなら

### 1. PyTorch実践入門

理由。

```text
・完全初学者ではない人向けに使いやすい
・Tensor、nn.Module、画像分類、CNNを長く参照できる
・SVD圧縮の実装時にも辞書として使いやすい
```

## 早くPyTorchに慣れたいなら追加

### 2. 動かしながら学ぶ PyTorchプログラミング入門

理由。

```text
・PyTorchの書き方に早く慣れる
・前半だけ読めば十分
・最短コースでわかる PyTorch の代替として使いやすい
```

## 今はまだ買わなくてよい

### 3. つくりながら学ぶ! PyTorchによる発展ディープラーニング

理由。

```text
・良い本だが、最初のMNIST + SVD圧縮には重い
・CNN / VGG / 転移学習に進んでからでよい
```

### 4. Tensor Network / DMRG本

理由。

```text
・今の最初のゴールには不要
・SVD圧縮ベースラインができてから読む方が理解しやすい
・先に読むと実装が止まりやすい
```

---

# 最終的なおすすめ順

```text
1. Pythonデータ分析認定試験の教材
2. 公式チュートリアル日本語版「0. PyTorch入門」
3. PyTorch実践入門
4. 自作MNIST + MLP + SVD圧縮コード
5. 動かしながら学ぶ PyTorchプログラミング入門
   ※早くPyTorchに慣れたい場合のみ追加
6. つくりながら学ぶ! PyTorchによる発展ディープラーニング
   ※CNN/VGG圧縮へ進む段階
7. Tensor Network / MPS / DMRG資料
   ※SVD圧縮が一通り動いた後
```

---

# 最初の到達目標

まずは以下を達成する。

```text
PyTorchで学習したMNIST分類MLPに対して、
Linear層のSVD低ランク近似を適用し、
rankごとの精度低下とパラメータ削減率を評価する。
```

これができたら、SVDを使用したNN重み行列圧縮の第一段階は完了。

---

# その後の発展目標

```text
1. SVD後fine-tuningで精度回復を確認
2. Fashion-MNISTで再現
3. CNNのLinear層を圧縮
4. Conv層圧縮に進む
5. Tensor Train / MPS層を実装
6. DMRG的fine-tuningを検討
```

---

# 一言まとめ

現時点では、テンソルネットワーク本よりもPyTorch実装が先。

```text
今: Pythonデータ分析認定試験
次: PyTorchでMNIST MLP
その次: SVDでLinear層圧縮
その後: Tensor Network / DMRG
```

Tensor Networkは、SVD圧縮のベースラインが動いてから読めば十分。
