---
title: Linear層の2層置換
aliases:
  - DivideTwoLayer
  - SVDによるLinear層置換
  - LowRankLinearへの変換
tags:
  - PyTorch
  - nn.Linear
  - SVD
  - 低ランク分解
  - nn.Sequential
---

# Linear層の2層置換

## サマリー

このノートでは、学習済みの1つの `nn.Linear` を、SVD因子から作る2つの小さい `nn.Linear` へ置き換える。

2層化の数式、rank上限、パラメータ削減条件は [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]] にまとめる。ここでは、そのPyTorch操作を扱う。

元の層を、

$$
y
=
Wx+b
$$

とする。

rank $r$ の切り詰めSVDにより、

$$
W
\approx
W_r
=
U_r\Sigma_rV_r^{\mathsf{T}}
$$

とする。

特異値を後段へ吸収する場合、

$$
A
=
V_r^{\mathsf{T}}
$$

$$
B
=
U_r\Sigma_r
$$

として、

$$
W_r
=
BA
$$

と書ける。

これをPyTorchでは、

```text
Linear(D_in, r, bias=False)
Linear(r, D_out, bias=True)
```

へ対応させる。

```mermaid
flowchart LR
    X["入力 D_in"] --> A["first_layer<br/>D_in → r<br/>biasなし"]
    A --> B["second_layer<br/>r → D_out<br/>元bias"]
    B --> Y["出力 D_out"]
```

2層の間にはReLUやsigmoidを入れない。入れると、元のLinear層の低ランク近似ではなく、別の非線形モデルになる。

---

## 対応コード

現在のLinear低ランク置換は、次へモジュール化している。

```text
src/nn_compression/compression/linear_svd.py
```

主要関数：

```python
from nn_compression.compression import (
    factorize_linear_layer,
    factorize_named_linear,
    rebuild_linear_from_svd,
)
```

- `factorize_linear_layer`：任意の `nn.Linear` をrank $r$ の2層Linearへ分解する
- `factorize_named_linear`：モデルをdeepcopyし、指定Linear属性だけを置換する
- `rebuild_linear_from_svd`：比較用に低rankweightを元と同じdense Linearへ戻す

MLP固有の `fc1 / fc2` 組み立ては、

```text
src/nn_compression/compression/mlp_svd.py
```

へ分離している。

旧Notebook向けには `RebuildSVD` などのaliasを残しているが、新しいコードではsnake_caseの公開APIを使う。

Conv2d版の対応する処理は [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]] を参照する。

## 1. 中間次元とrankは同じか

この2因子分解では、中間次元はrank $r$ と同じにする。

$$
D_{\mathrm{in}}
\rightarrow
r
\rightarrow
D_{\mathrm{out}}
$$

理由は、切り詰めSVDで残した独立成分の数が $r$ 個だからである。

### 前段のshape

$$
A
=
V_r^{\mathsf{T}}
\in
\mathbb{R}^{r\times D_{\mathrm{in}}}
$$

これは、

```python
nn.Linear(D_in, r, bias=False)
```

のweight shapeと一致する。

### 後段のshape

$$
B
=
U_r\Sigma_r
\in
\mathbb{R}^{D_{\mathrm{out}}\times r}
$$

これは、

```python
nn.Linear(r, D_out, bias=True)
```

のweight shapeと一致する。

---

### shapeの0番目をin_featuresへ渡すと逆になる

`original` がPyTorchのweightであるとき、`Linear(original.shape[0], r)` とすると入力・出力の読み方が逆になる。
shape $(D_{\mathrm{out}},D_{\mathrm{in}})$ なので、
`D_out, D_in = original.shape` と取り出し、前段は `Linear(D_in, r)` にする。

小さい例として、biasなしの `Linear(3, 2)` を考える。

$$
W=\begin{pmatrix}9&0&0\\0&4&0\end{pmatrix},
\qquad
U=I_2,\quad
\Sigma=\begin{pmatrix}9&0\\0&4\end{pmatrix},\quad
V_h=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}.
$$

rank 1では

$$
A=V_{h,1}=\begin{pmatrix}1&0&0\end{pmatrix},
\qquad
B=U_1\Sigma_1=\begin{pmatrix}9\\0\end{pmatrix},
\qquad
BA=\begin{pmatrix}9&0&0\\0&0&0\end{pmatrix}.
$$

入力 $x=(x_1,x_2,x_3)^{\mathsf T}$ に対して

$$
h=Ax=x_1,\qquad
y=Bh=\begin{pmatrix}9x_1\\0\end{pmatrix}.
$$

中間次元は1だが、出力次元は元の2のままである。
**出力が2クラスだからrankも2固定、とはならない。**
MNISTの最終 `Linear(256, 10)` も、SVD初期化で選べる正のrankは1〜10であり、
最終出力は常に10次元へ戻す。ただし小さいrankで精度を保てるかは別に評価する。
rank 10も、ゼロ特異値があれば実際の行列rankが10とは限らない。

```python
import torch
from torch import nn

# モデルの学習ではなく、shapeの向きとrank 1のfactorコピーを確認する。
original = torch.tensor([[9.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64)
D_out, D_in = original.shape
rank = 1
U, S, Vh = torch.linalg.svd(original, full_matrices=False)
first = nn.Linear(D_in, rank, bias=False, dtype=original.dtype)
second = nn.Linear(rank, D_out, bias=False, dtype=original.dtype)
# Linearの生成時には初期値が入る。shapeを決めるだけの空箱ではない。
with torch.no_grad():
    first.weight.copy_(Vh[:rank, :])
    second.weight.copy_(U[:, :rank] * S[:rank].unsqueeze(0))
factors = nn.Sequential(first, second)
X = torch.tensor([[2.0, 3.0, 5.0]], dtype=original.dtype)
torch.testing.assert_close(factors(X), torch.tensor([[18.0, 0.0]], dtype=original.dtype))
assert first.weight.shape == (1, 3)
assert second.weight.shape == (2, 1)
```

逆に `Linear(original.shape[0], rank)` を作るとweightは $(1,2)$ となり、
$A$ の $(1,3)$ と一致しない。
SVDの上限を超えるsliceは指定数の列・行を増やしてくれるわけでもないため、
前段を作る前に `rank <= min(D_out, D_in)` を検査する。
関数を `divide_two_layer(layer, rank)` と定義したなら、呼び出しは `divide_two_layer(layer, 10)` または `divide_two_layer(layer, rank=10)` とし、定義にないkeywordは渡さない。
行列rankの上限とパラメータ削減の損益分岐は [[05_圧縮率とRank]] で分けて計算する。

---

## 2. `first_layer` と `second_layer`

`Linear(784, 512)` をrank 64へ分解する例：

```text
元の層
Linear(784, 512)
weight: (512, 784)
```

```text
前段
Linear(784, 64, bias=False)
weight: (64, 784)
```

```text
後段
Linear(64, 512, bias=True)
weight: (512, 64)
bias:   (512,)
```

SVD因子との対応：

```text
first_layer.weight  ← Vh_r
second_layer.weight ← U_r Σ_r
second_layer.bias   ← 元layer.bias
```

---

## 3. biasを後段だけへ置く理由

### forwardをbatchの要素から再構成重みまでつなぐ

この節の $A\in\mathbb R^{r\times D_{\mathrm{in}}}$、
$B\in\mathbb R^{D_{\mathrm{out}}\times r}$ を用いる。
第1層にbiasを持たせない場合

$$
H_{n,j}=\sum_{i=1}^{D_{\mathrm{in}}}X_{n,i}A_{j,i},
$$

$$
\begin{aligned}
(Y_r)_{n,o}
&=\sum_{j=1}^{r}H_{n,j}B_{o,j}+b_o\\
&=\sum_{j=1}^{r}
\left(\sum_iX_{n,i}A_{j,i}\right)B_{o,j}+b_o\\
&=\sum_iX_{n,i}\left(\sum_jB_{o,j}A_{j,i}\right)+b_o\\
&=\sum_iX_{n,i}(BA)_{o,i}+b_o.
\end{aligned}
$$

従って行列式は

$$
\begin{aligned}
H&=XA^{\mathsf T},\\
Y_r&=HB^{\mathsf T}+\mathbf 1_Nb^{\mathsf T}\\
&=XA^{\mathsf T}B^{\mathsf T}+\mathbf 1_Nb^{\mathsf T}\\
&=X(BA)^{\mathsf T}+\mathbf 1_Nb^{\mathsf T}\\
&=XW_r^{\mathsf T}+\mathbf 1_Nb^{\mathsf T}.
\end{aligned}
$$

$\mathbf 1_Nb^{\mathsf T}$ が実装のbias broadcastである。
第1層にbias $c$、第2層にbias $d$ を入れる場合は

$$
\begin{aligned}
Y_r
&=(XA^{\mathsf T}+\mathbf 1_Nc^{\mathsf T})B^{\mathsf T}
+\mathbf 1_Nd^{\mathsf T}\\
&=X(BA)^{\mathsf T}+\mathbf 1_N(Bc+d)^{\mathsf T}.
\end{aligned}
$$

元biasを保つには $Bc+d=b$ が必要となる。
$c=0,d=b$ は、この調整を不要にする選択である。

元の計算は、

$$
y
=
Wx+b
$$

である。

低ランク化後は、

$$
y
\approx
BAx+b
$$

とする。

したがって、biasは最終出力次元、

$$
D_{\mathrm{out}}
$$

を持つ後段へ置く。

前段にbiasを入れると、

$$
B(Ax+c)+b
=
BAx+Bc+b
$$

となり、元にはなかった追加項 $Bc$ が生じる。

### 元の層にbiasがない場合

元の `layer.bias is None` なら、後段も `bias=False` にする。

この条件を実装で分岐する。

---

## 4. 2層の間に活性化関数を入れない

通常のニューラルネットワークでは、Linear層の間へReLUを入れる。

しかし、今回の2層は、元の1つの線形写像を因子分解したものである。

正しい低ランク置換：

$$
x
\rightarrow
Ax
\rightarrow
BAx+b
$$

間にReLUを入れると、

$$
x
\rightarrow
Ax
\rightarrow
\operatorname{ReLU}(Ax)
\rightarrow
B\operatorname{ReLU}(Ax)+b
$$

となる。

これは、

$$
BAx+b
$$

とは異なる。

### 元から後ろにあったReLU

元モデルが、

```text
fc1 → relu1 → fc2
```

なら、置換後は、

```text
fc1.first → fc1.second → relu1 → fc2
```

とする。

元の `relu1` は残すが、2因子の間には追加しない。

---

## 5. `with torch.no_grad()` が必要な理由

新しく作った、

```python
first_layer.weight
second_layer.weight
```

は `nn.Parameter` であり、通常は `requires_grad=True` である。

初期値としてSVD因子をコピーする操作は、学習によって求める計算ではない。

したがって、

```python
with torch.no_grad():
    first_layer.weight.copy_(Vh_r)
    second_layer.weight.copy_(B)
```

とする。

### SVD部分との違い

```text
SVD入力として元重みを読む
└── layer.weight.detach()

新しいパラメータへ値を設定する
└── with torch.no_grad(): copy_()
```

詳細は [[05_SVD基礎実装検証/08_Linear層のSVD実装]] を参照する。

---

## 6. `copy_()` を使う理由

次のような代入は標準にしない。

```python
first_layer.weight = Vh_r
```

`weight` には `nn.Parameter` が必要であり、単純なTensorを代入するとModuleのパラメータ登録と整合しない。

また、

```python
first_layer.weight.data = Vh_r
```

もautograd管理を迂回する。

そのため、既存の `nn.Parameter` を保ったまま、

```python
first_layer.weight.copy_(Vh_r)
```

で値だけをコピーする。

---

## 7. `nn.Sequential` は何を返すか

```python
return nn.Sequential(
    first_layer,
    second_layer,
)
```

は、2つの層を順番に呼び出す1つの `nn.Module` を返す。

入力 `x` に対して、概念的には、

```python
x = first_layer(x)
x = second_layer(x)
return x
```

が実行される。

返り値はTensorではない。Tensorを受け取って計算できる、新しい小さなモデル部品である。

### Module tree

```text
Sequential(
  (0): Linear(D_in, r, bias=False)
  (1): Linear(r, D_out, bias=True)
)
```

---

## 8. `model.fc1 = divide_two_layer(...)` で何が起こるか

元モデルでは、

```text
model.fc1
└── Linear(784, 512)
```

である。

次を実行する。

```python
model.fc1 = divide_two_layer(
    model.fc1,
    64,
)
```

すると、属性 `fc1` が指すModuleが置き換わる。

```text
model.fc1
└── Sequential
    ├── 0: Linear(784, 64)
    └── 1: Linear(64, 512)
```

### 「元のfc1はfirstとsecondを持っていない」の整理

置換前の `fc1` は、確かに1つの `nn.Linear` であり、2つの子要素を持たない。

しかし、代入後の `model.fc1` は同じオブジェクトではない。

```text
置換前
model.fc1 → nn.Linear

置換後
model.fc1 → nn.Sequential
```

属性名は同じ `fc1` でも、中に入っているModuleが変わった。

### なぜmodelとして動くか

`nn.Sequential` 自体が `nn.Module` を継承しているためである。

元の `forward` に、

```python
x = self.fc1(x)
```

と書かれていても、`self.fc1` がLinearかSequentialかにかかわらず、Moduleとして呼び出せる。

Sequentialなら内部の2層が順番に実行される。

---

## 9. PyTorchが新しい層を登録する仕組み

`MNISTMLP` は `nn.Module` を継承している。

`nn.Module` の属性へ別の `nn.Module` を代入すると、そのModuleは子Moduleとして登録される。

その結果、次にも含まれる。

- `model.parameters()`
- `model.named_parameters()`
- `model.modules()`
- `model.state_dict()`
- `.to(device)`
- `.train()`
- `.eval()`

これは、単にPythonの変数へTensorを置いただけでは得られない `nn.Module` の機能である。

---

## 10. deviceを引き継ぐ

新しく、

```python
nn.Linear(...)
```

を作ると、通常はCPU上に作られる。

元の層がGPU上なら、そのままでは、

```text
Expected all tensors to be on the same device
```

のような不一致が起きる。

したがって、新しい2層を元層と同じdeviceへ移す。

```python
new_module = nn.Sequential(
    first_layer,
    second_layer,
).to(
    device=layer.weight.device,
    dtype=layer.weight.dtype,
)
```

### コピーの順序

安全な流れは、

1. 新しい層を作る
2. 元のdevice・dtypeへ移す
3. SVD因子をコピーする

である。

因子とコピー先のdevice・dtypeを一致させる。

---

## 11. train・eval状態の引き継ぎ

`nn.Linear` 自体はtrainとevalで計算が変わらない。

しかし、Module置換関数としては元層の状態を意識する。

元モデルが `eval()` 中なら、新しいModuleも評価状態へ合わせる。

```python
new_module.train(layer.training)
```

通常はモデルを構築・置換した後に、外側で、

```python
model.train()
```

または、

```python
model.eval()
```

を呼び直してもよい。

---

## 12. optimizerを作り直す理由

次の順番は避ける。

```text
1. optimizerを作る
2. model.fc1を新しい2層へ置換する
3. 同じoptimizerを使う
```

optimizerは、作成時点のパラメータオブジェクトへの参照を保持する。

層置換後は、

- 元の `fc1.weight`
- 元の `fc1.bias`

がモデルから外れ、

- `fc1.0.weight`
- `fc1.1.weight`
- `fc1.1.bias`

が新しいパラメータになる。

そのため、置換後に、

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=...,
)
```

として作り直す。

### fine-tuningの標準順序

```text
baselineを学習
↓
checkpointを保存
↓
モデルをコピー
↓
Linear層を置換
↓
圧縮直後を評価
↓
新しいoptimizerを作成
↓
fine-tuning
```

---

## 13. 元モデルを残す

比較実験では、学習済みbaselineを直接書き換えない。

推奨は、

```python
compressed_model = copy.deepcopy(
    baseline_model
)
```

でコピーを作り、そのコピーへ置換することである。

これにより、

- baseline出力
- 圧縮後出力
- logits差
- prediction agreement

を同じ入力で比較できる。

各rankは同じbaseline checkpointから作成する。

---

## 14. `state_dict` の構造変化

置換前：

```text
fc1.weight
fc1.bias
```

置換後：

```text
fc1.0.weight
fc1.1.weight
fc1.1.bias
```

となる。

そのため、未圧縮モデルの `state_dict` を、そのまま圧縮後構造へ `strict=True` で読み込むことはできない。

### 保存方針

- baseline checkpointは未圧縮構造で保存
- 圧縮後checkpointは同じ圧縮構造を構築してから読み込む
- rank設定もJSONなどへ保存

する。

---

## 15. rankの「上限」と「圧縮成立条件」は別

rankには2種類の条件がある。

### 数学的に使える上限

$$
r
\leq
\min
\left(
D_{\mathrm{in}},
D_{\mathrm{out}}
\right)
$$

### パラメータ数が減る条件

biasを後段だけへ残す場合、元のパラメータ数は、

$$
P_{\mathrm{original}}
=
D_{\mathrm{out}}D_{\mathrm{in}}
+
D_{\mathrm{out}}
$$

圧縮後は、

$$
P_{\mathrm{lowrank}}
=
rD_{\mathrm{in}}
+
rD_{\mathrm{out}}
+
D_{\mathrm{out}}
$$

である。

圧縮になる条件は、

$$
r
<
\frac{
D_{\mathrm{in}}D_{\mathrm{out}}
}{
D_{\mathrm{in}}+D_{\mathrm{out}}
}
$$

である。

`fc3 = Linear(256, 10)` では、

$$
r
<
\frac{2560}{266}
\approx
9.62
$$

なので、整数rankでは $r\leq9$ のときだけパラメータ数が減る。

$r=10$ は数学的には有効だが、圧縮にはならない。

---

## 16. 最大rankでの出力一致テスト

rankを最大値へ設定すれば、理論上、元の重みを再構成できる。

単体Linearへ同じ入力を与え、

```text
original_output
lowrank_output
```

を比較する。

確認対象：

- weight再構成
- bias
- batch入力
- device
- dtype

### ReLUを含めない単体テスト

まずLinear単体を比較する。

その後、モデル全体で、元から存在するReLUを含めて比較する。

---

## 17. 低rankでのテスト

低rankでは出力一致を要求しない。

確認するのは、

- 出力shapeが同じ
- NaN・Infがない
- rankを上げると重み誤差が小さくなる
- パラメータ数が式と一致する
- backwardが実行できる
- 新しいパラメータへgradientが入る

ことである。

圧縮後の2層は通常の `nn.Linear` なので、fine-tuning時は自動微分で学習できる。

---

## 18. `nn.Sequential` と専用クラスの比較

### `nn.Sequential`

利点：

- 短い
- Module登録が自動
- `forward` を書かなくてよい
- 初学者が構造を確認しやすい

欠点：

- 因子へ名前を付けにくい
- rankなどのメタデータを持たせにくい

### `LowRankLinear` クラス

利点：

- `first`、`second` と名前を付けられる
- `in_features`、`out_features`、`rank` を保持できる
- 再利用しやすい
- 保存・表示・検査を拡張しやすい

最初はSequentialで理解し、再利用段階で専用クラスへ移す。

---

## Pythonクラス記法の補足

`self`、`__init__()`、`super().__init__()`、`model(images)` と `forward(images)` の違いは、Linear層の置換だけに固有の話ではない。一般的なPython・PyTorchクラス記法として [[05_SVD基礎実装検証/13_PandasとPython実装メモ#23. PyTorchモデルを読むためのクラス記法]] に集約する。

このノートでは、`self.first` と `self.second` が子Moduleとして登録され、`model(images)` から2層が順に呼ばれることを押さえればよい。

## 19. よくある間違い

### 中間次元をrankと別の値にする

SVD因子のshapeと一致しなくなる。

### first_layerへbiasを付ける

元のLinearとは異なる定数項が追加される。

### 2層間へReLUを入れる

低ランク線形近似ではなく、新しい非線形ネットワークになる。

### 元biasをコピーしない

最大rankでも出力が一致しない。

### 新しい層をCPUのまま返す

GPU入力とのdevice不一致になる。

### optimizerを置換前のまま使う

新しい低ランク層が更新されない。

### 元モデルを直接変更する

baseline比較ができなくなる。

### rank最大なら必ず圧縮になると思う

数学的に有効でも、パラメータ数が増える場合がある。

---

## 20. このノートで押さえるポイント

- 中間次元はSVDで残したrank $r$ とする。
- 前段は `Linear(D_in, r, bias=False)` である。
- 後段は `Linear(r, D_out, bias=元と同じ)` である。
- 前段重みは `Vh_r` である。
- 後段重みは `U_r Σ_r` である。
- 元biasは後段へコピーする。
- 2層の間へ活性化関数を入れない。
- `nn.Sequential` は2つの層を持つ1つのModuleである。
- `model.fc1 = ...` によって `fc1` の中身がLinearからSequentialへ置き換わる。
- 置換後のModuleは `model.parameters()` と `state_dict()` へ自動登録される。
- deviceとdtypeを元層へ合わせる。
- 層置換後はoptimizerを作り直す。
- 数学的rank上限と、圧縮になるrank上限を区別する。

---

> [!note] 実測・実行結果は [[05_SVD基礎実装検証/02_Linear層2層置換の確認結果]] へ分離した。

## 次に読むノート

圧縮後モデルを、重み、出力、分類性能の段階に分けて評価する。

- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[05_SVD基礎実装検証/07_PyTorch実装]]
