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

実際の `code.zip` では、2層置換を次のNotebookで実装している。

| Notebook | 役割 |
|---|---|
| `notebooks/00_fundamentals/03_linear_svd_two_layer_replacement.ipynb` | 1つの `Linear(784,512)` をrank 64の2層へ置換する最小実験 |
| `notebooks/10_mnist_mlp/01_svd_compression.ipynb` | MNIST MLPの `fc1`・`fc2` をdense再構成版または2層版へ置換 |
| `notebooks/10_mnist_mlp/02_rank_accuracy_tradeoff.ipynb` | `fc1_rank` と `fc2_rank` を別々に指定して16条件を生成 |

実際の関数名は次のとおりである。

```python
def devidetwolayer(layer, r):
    ...

def make_one_layer_svd_model(model, r1, r2):
    ...

def make_two_layer_svd_model(model, r1, r2):
    ...
```

- `devidetwolayer`：1つのLinear層を2つのLinear層へ変換する
- `make_one_layer_svd_model`：SVDで近似した重みを、元と同じshapeのdense Linearへ戻す比較用モデル
- `make_two_layer_svd_model`：SVD因子を2つのLinearとして保持し、実際にパラメータ数を減らすモデル

> [!warning]
> 関数名 `devidetwolayer` は綴りに誤りがあるが、実コードとの対応を保つためここではそのまま記載する。  
> リファクタリング時は `divide_two_layer` へ改名する。

以前記載していた 以前の案にあった層置換用Pythonモジュール、`replace_linear_by_path`、`LowRankLinear` は、現在の `code.zip` には存在しない。

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

詳細は [[00_基礎理論/08_Linear層のSVD実装]] を参照する。

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

添付PDFでは、モデル定義を読むために必要なPython記法も確認した。  
SVDの数式とは別の話だが、層置換コードを理解するために必要なのでここへまとめる。

### `self` の意味

```python
self.fc1 = nn.Linear(784, 512)
```

`self` は、現在操作しているモデルのインスタンス自身を表す。

```text
self.fc1
=
このモデルが持っているfc1
```

`self.fc1` として登録した層は、`forward()` から利用できるだけでなく、`nn.Module` の子Moduleとして認識される。

そのため、

```python
model.parameters()
model.state_dict()
model.to(device)
model.train()
model.eval()
```

などの処理にも含まれる。

### `__init__()` の意味

```python
def __init__(self):
    ...
```

`__init__()` は、インスタンスを作るときに実行される初期化メソッドである。

モデルでは主に、

- Linear層
- ReLU
- Dropout
- BatchNorm
- その他の子Module

を定義する。

### `super().__init__()` の意味

```python
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
```

`MNISTMLP` は `nn.Module` を継承している。  
`super().__init__()` は、親クラスである `nn.Module` の初期化を呼ぶ。

これにより、PyTorchが子ModuleやParameterを正しく登録できる。

### `_` と `__init__` のアンダースコアは別物

```text
_
```

単独のアンダースコアは、Pythonで「値は受け取るが使わない」という意図を示す変数名として使われることが多い。

一方、

```text
__init__
```

の前後の2本のアンダースコアは、Pythonが定めた特殊メソッド名の一部である。  
単独の `_` のような捨て変数ではない。

```text
_          → 慣習的な未使用変数名
__init__   → Pythonの特殊メソッド名
```

### `model(images)` と `forward(images)`

```python
outputs = model(images)
```

と書くと、`nn.Module` の呼出し処理を経由して、自分で定義した `forward(images)` が実行される。

概念上は、

```python
outputs = model.forward(images)
```

に近いが、通常は `model(images)` を使う。

`model(images)` では、PyTorchのhookなどを含む `nn.Module` の呼出し機構が正しく働くためである。

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

- [[00_基礎理論/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/06_誤差評価]]
- [[00_基礎理論/07_PyTorch実装]]
