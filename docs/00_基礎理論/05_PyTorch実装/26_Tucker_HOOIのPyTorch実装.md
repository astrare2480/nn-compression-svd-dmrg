---
title: Tucker・HOOIのPyTorch実装
aliases:
  - Tucker src実装
  - HOOI src実装
  - Conv Tucker PyTorch
tags:
  - Tucker
  - HOOI
  - PyTorch
  - src
  - NN圧縮
---

# Tucker・HOOIのPyTorch実装

## サマリー

Tucker/HOOI編では、Notebookで自作して理解した処理を、最終的に `src/nn_compression/` へ責務分離した。

```text
Tensor演算
→ tensor/operations.py

Tensor入力contract
→ tensor/validation.py

HOSVD / Tucker再構成
→ compression/tucker.py

Tucker / HOOI固有contract
→ compression/tucker_validation.py

汎用HOOI
→ compression/hooi.py

Conv2d Tucker-2
→ compression/conv_tucker.py

Tensor近似誤差
→ metrics/tensor_approximation.py
```

学習Notebookは自作実装を残し、再利用用srcは汎用APIとして整理する。

---

## 1. Tensor演算

```text
src/nn_compression/tensor/operations.py
src/nn_compression/tensor/validation.py
```

主な関数：

```python
unfold(X, mode)
fold(unfolded, mode, shape)
mode_dot(X, matrix, mode)
```

`mode_dot` は

$$
\mathcal{Y}=\mathcal{X}\times_n A
$$

を実装し、HOSVD/HOOI双方から使う。

現行public contractでは、Tensor APIへ渡すshapeは2次元以上で各dimensionが正であることを要求する。`mode` はboolではない整数で、

$$
0\le n < \operatorname{ndim}(X)
$$

を満たす必要がある。

`fold` は単に総要素数が一致すればreshapeするのではなく、unfoldedの行数・列数が指定modeと元shapeに一致するかを確認する。転置されたunfoldedを黙って受理しない。

---

## 2. HOSVD / Tucker

```text
src/nn_compression/compression/tucker.py
src/nn_compression/compression/tucker_validation.py
```

主な関数：

```python
hosvd(X, ranks)
reconstruct_tucker(core, factors)
tucker_parameter_count(shape, ranks)
parameter_ratio(shape, ranks)
compression_factor(shape, ranks)
```

`hosvd` の重要な実装契約は、factorを各modeについて**元のXから独立に**求めること。

```text
X → mode0 unfold → SVD → U0
X → mode1 unfold → SVD → U1
...
```

factor計算中にXを順次projectしてはいけない。それをすると標準的な一回のHOSVDとは異なる処理になる。

### rank contract

`ranks` は `{mode: rank}` 形式の `Mapping` とする。空 `{}` はHOSVD/Tucker parameter計算ではidentity指定として扱えるが、HOOIでは更新対象が無いため拒否する。

mode-n unfolding

$$
X_{(n)}\in\mathbb R^{I_n\times\prod_{m\ne n}I_m}
$$

に対して取りうるSVD rank上限は、

$$
\boxed{
R_n
\le
\min\left(I_n,\prod_{m\ne n}I_m\right)
}
$$

である。現行srcではHOSVD、parameter count、ratio、compression factorでこの上限を共通contractとして使う。

boolはPython上 `int` のsubclassだが、rankやmodeとしては明示的に拒否する。

### dtype contract

現行HOSVD/HOOIはfactor射影に `U.T` を使う**実数Tensor向け実装**である。

複素Tensorでは本来共役転置 `U.mH` が必要であり、`torch.linalg.svd` 自体は複素dtypeを処理できてしまう。silent failureを避けるため、現行srcでは複素Tensorを入口で `TypeError` として拒否する。

---

## 3. 汎用HOOI

```text
src/nn_compression/compression/hooi.py
```

公開API：

```python
has_converged
hooi_sweep
core_from_factors
hooi
```

### `hooi_sweep`

下のloopで対象modeのfactorは射影に使わない。これから選び直す基底で先に情報を落とすと、候補を調べる局所問題そのものが変わってしまうためである。
毎回projectedを元の $X$ から作り、他modeについてはそのsweepで更新済みのfactorを使う。
前の対象modeで作ったprojectedをそのまま次へ渡すloopとは区別する。

元shapeが $(I_0,\ldots,I_{N-1})$、更新集合が $\mathcal M$ なら、対象mode $n$ のprojected unfoldingは

$$
I_n\times
\left(
\prod_{\substack{m\in\mathcal M\\m\ne n}}R_m
\prod_{m\notin\mathcal M}I_m
\right)
$$

になる。対象行数は $I_n$ のままだが、他modeの列数は既に $R_m$ へ変わっている。
例えばshape $(5,4,3)$、target ranks $(2,1,1)$ では、元mode 0 unfoldingは $5\times12$ だが、他modeへの射影後は $5\times1$ である。
後者のreduced SVDで得られる左特異ベクトルは1列なので、2列のfactorを要求する現行contractは満たせない。
「元unfoldingではrankが許される」と「全局所更新でもfactorを作れる」は違う、というfeasibility条件を数える例である。

`ranks` のkeyを更新対象modeとして扱う。

```python
for target_mode in ranks:
    projected = X
    for other_mode in ranks:
        if other_mode == target_mode:
            continue
        projected = mode_dot(
            projected,
            updated_factors[other_mode].T,
            other_mode,
        )
```

これにより、

```python
ranks = {0: 3, 1: 2, 2: 2}
```

なら3-mode HOOI、

```python
ranks = {0: rank_out, 1: rank_in}
```

なら4階Conv weightのmode 0 / 1だけを更新するpartial HOOIになる。

入力 `factors` はcloneし、破壊的に変更しない。1 sweep内では先に更新したfactorを後続mode更新に使うGauss-Seidel型である。

### HOOI入力validation

反復前に、

- `ranks` がMappingで空でない
- factor keyとrank keyが一致
- factorが2次元
- factor shapeが `(X.shape[mode], rank)`
- factorのdevice / dtypeがXと一致
- rankが元unfoldingだけでなく、他factorで射影した後のprojected unfoldingでも実現可能

を確認する。

このfeasibility確認は反復loopより前に行うため、`max_iter=0` でも不可能なrankを受理しない。

### `core_from_factors`

$$
G
=
X
\times_0U_0^{\mathsf T}
\times_1U_1^{\mathsf T}
\cdots
$$

を現在factorから計算する。

### `hooi`

```text
HOSVD初期化
→ factor / rank feasibility確認
→ initial errorをhistory[0]へ保存
→ hooi_sweep
→ core再計算
→ reconstruction
→ relative error
→ has_converged
```

を繰り返す。

historyはpandasに依存しない `list[float]` とし、Notebook側でDataFrame化する。

HOOIではrelative Frobenius errorを使うため、分母が0になるzero Tensorは明示的に拒否する。

`max_iter` はboolではない0以上の整数、`abs_tol / rel_tol` は有限かつ0以上を要求する。`has_converged()` 単体でも同じtolerance contractを検証する。

---

## 4. Conv Tucker-2

```text
src/nn_compression/compression/conv_tucker.py
```

主なAPI：

```python
tucker2_decompose_conv_weight
build_tucker2_conv
build_tucker2_conv_from_components
tucker2_hooi
tucker2_hooi_sweep
tucker2_effective_weight
```

### `build_tucker2_conv_from_components`

事前に計算した

```text
core
U_out
U_in
```

から

```text
C_in
→ 1x1 / U_in.T
→ R_in
→ kxk / core
→ R_out
→ 1x1 / U_out
→ C_out
```

を作る。

HOSVDとHOOIで分解法が違っても、3層を作る処理は共通化できる。

現行実装は `groups=1` の通常 `nn.Conv2d` を対象とする。中央core Convが元Convの `stride / padding / dilation / padding_mode` を継承し、元biasは最後の1x1 Convへ置く。

`rank_out / rank_in` はboolを拒否し、

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

を入口で要求する。

### `tucker2_effective_weight`

fine-tuning後の3層から

$$
\hat W
=
G\times_0U_{\mathrm{out}}\times_1U_{\mathrm{in}}
$$

を再構成し、元の1つのConv weightと比較できるようにする。

### 元モデルを保存したまま置換前後を確認する

圧縮前後を公平に比較するには、元モデルを直接書き換えず、deepcopyしたモデルの対象層だけを置換する。次の例では、同じ固定batchについてoutput shapeとlogits差を確認し、同時に3層が表す有効weightも元Conv weightと比較する。

```python
import copy

import torch

from nn_compression.compression import (
    build_tucker2_conv,
    tucker2_effective_weight,
)

# modelは学習済みモデル、input_batchは同じ入力で比較する固定batchとする。
device = next(model.parameters()).device
input_batch = input_batch.to(device)

# 元モデルが変化していないことを後で確認できるように、値を独立に保存する。
source_weight = model.conv2.weight.detach().clone()

baseline_model = copy.deepcopy(model).to(device).eval()
compressed_model = copy.deepcopy(model).to(device)

# deepcopy側だけをTucker-2の3層へ置換する。
compressed_model.conv2 = build_tucker2_conv(
    compressed_model.conv2,
    rank_out=rank_out,
    rank_in=rank_in,
).to(device)
compressed_model.eval()

with torch.no_grad():
    baseline_logits = baseline_model(input_batch)
    compressed_logits = compressed_model(input_batch)

# モデルの入出力interfaceは変えない。
assert baseline_logits.shape == compressed_logits.shape

# 元モデルは置換されず、元weightも変化していない。
assert isinstance(model.conv2, torch.nn.Conv2d)
assert torch.equal(model.conv2.weight.detach(), source_weight)

# 3層を1つの有効weightへ戻し、同じshapeで差を測る。
effective_weight = tucker2_effective_weight(compressed_model.conv2)
assert effective_weight.shape == source_weight.shape

weight_relative_error = (
    torch.linalg.vector_norm(source_weight - effective_weight)
    / torch.linalg.vector_norm(source_weight)
)
logits_rmse = torch.sqrt(
    torch.mean((baseline_logits - compressed_logits) ** 2)
)

print("weight relative error:", float(weight_relative_error))
print("logits RMSE:", float(logits_rmse))
```

`effective_weight` の一致は層の再構成に関する確認であり、`logits_rmse` は実データを通したモデル出力の確認である。この2つを分けることで、weight近似誤差とtask上の影響を混同しない。

### 元資料のdummy入力と、forward確認で分かる範囲

添付 `CNN CIFAR-10→Tuckerを考える (1).md` の1991–2236行では、
`copy.deepcopy(model)`、`conv2` の差し替え、dummy入力、forwardを一行ずつ説明している。
直前のコードのモデルを使う場合、元資料と同じ入力のサイズは

$$
\operatorname{shape}(\mathrm{dummy})=(N,C,H,W)=(4,3,32,32)
$$

である。4は独立な画像入力を4枚束ねたbatch、3は各画像のRGB channel、
二つの32は高さと幅である。第1軸だけを選ぶ `dummy[0]` は
shape `(3, 32, 32)` の1枚分になる。
元資料の `dummy[^18_0]` は引用マーカーが添字へ混入した表記なので、
コードの添字を `[0]` として読み直す。

```python
# 前の節で作ったコピー側を使い、元モデルやDataLoaderは変更しない。
parameter = next(compressed_model.parameters())
dummy = torch.randn(
    4, 3, 32, 32,
    device=parameter.device,
    dtype=parameter.dtype,
)
assert dummy[0].shape == (3, 32, 32)

# evalはDropout / BatchNormの動作を変え、no_gradは微分履歴を止める。
compressed_model.eval()
with torch.no_grad():
    dummy_logits = compressed_model(dummy)

# CIFAR-10の10クラス用モデルを前提としたinterfaceの確認である。
assert dummy_logits.shape == (4, 10)
```

`torch.randn` は平均0・分散1の正規分布から値を作る。
これは実際のCIFAR-10画像ではなく、入力のshapeを再現したテスト入力である。
batchを1枚へ変えるなら先頭の4を1へ、32枚なら32へ変える。
`copy.deepcopy(model)` はモデル自体を複製し、コピー側の `conv2` を置換しても
元の `model.conv2` は置換しない。ただし、特殊なTensor属性などを持つModuleでは、
deepcopyできるかを個別に確認する必要がある。

このforwardが成功すれば、少なくとも、その入力について前後層のchannelの接続、
空間shapeの接続、入力とparameterのdevice / dtypeの整合を確認できる。
成功だけで元の `stride / padding / dilation` が完全に同じと証明したことにはならない。
それらの設定は元Convと置換したspatial Convを直接比較する。

同じdummyをbaselineにも渡してlogits差を測ることはできるが、
ランダム入力での差は実データ上のloss / accuracyの代わりにはならない。
元資料がモデル全体のlogits差を「元のconv2の出力近似」と説明した箇所は、
測定対象を分けて読む。後続層も通したlogits差と、Conv2単体のactivation差は別の量である。
`no_grad()` は評価modeへ変更しないため、DropoutやBatchNormを評価動作にする
`eval()` と併用する。

---

## 5. Autograd境界：`detach()` と `no_grad()`

現行srcでは、**Tensor-level decomposition** と **Module構築** でautograd境界を分ける。

### Tensor-level HOOI

```python
tucker2_hooi(weight, ...)
```

は低レベルTensor APIなので、内部で入力 `weight` を `detach()` しない。入力が `requires_grad=True` なら、分解計算のgraphを不要に切らない。

### Module構築

```python
build_tucker2_conv(conv, ...)
```

は学習済みConvから新しい3層Moduleを作る**初期化処理**なので、ここで元 `conv.weight` を `detach()` して分解する。

新しい層への値コピーは、

```python
with torch.no_grad():
    input_layer.weight.copy_(...)
    core_layer.weight.copy_(...)
    output_layer.weight.copy_(...)
```

とする。

これはコピー操作のgradient historyを不要にするためであり、コピー後のParameterを学習不能にするものではない。`requires_grad` は元Convから引き継ぎ、fine-tuningでは通常どおり

```python
loss.backward()
optimizer.step()
```

で更新できる。

置換後の3層Parameterはleafであり、元ConvのParameter storageを共有しない。

### 元資料のleaf・in-placeエラーを、初期化と微分可能な計算に分ける

同じ添付の799–1052行で繰り返し説明した理由は、
「学習対象のParameterをコピーで書き換える操作」と「その後のfine-tuning」を分けることにある。
`nn.Parameter(..., requires_grad=True)` は通常leafであり、
`weight.copy_(...)` は値をその場で書き換えるin-place操作である。
勾配追跡が有効なまま、そのleafへコピーすると、通常は

```text
RuntimeError:
a leaf Variable that requires grad is being used in an in-place operation
```

になる。初期化なので、直前の `with torch.no_grad(): parameter.copy_(...)` を使う。
コピー元を `detach()` するだけでは、コピー先のleafを勾配追跡中に書き換える問題は解消しない。
また、`no_grad()` はコピー以前に既に作ったHOSVDのgraphを遡って消す機能ではない。
初期化用の分解を元Convから切り離す責務は、上で説明したModule構築側の `detach()` にある。

コピー後は新しい3層のparameterを学習の開始点として扱う。
`requires_grad=True` を維持したparameterには、その後のforwardとbackwardで勾配を計算できる。
`parameter.data.copy_(...)` でautogradの管理を迂回してエラーだけを避ける方法は使わない。
一方、SVD/HOSVDを含む変換自体を微分したい別の目的では、
新しいleaf Parameterへの初期値コピーと同じ構成にせず、Tensor-levelの計算として
graphを保持するかを設計する。
元資料の「履歴をコピーしない」は、本章ではこの初期化の境界を指す。

### effective weight評価

`tucker2_effective_weight()` はfine-tuning前後の3層を評価用の1つの4階weightへ戻すhelperなので、各layer weightをdetachして再構成する。

---

## 6. validationの責務

現行srcでは、中心アルゴリズムと入力contractを分ける。

```text
tensor/validation.py
→ Tensor ndim / positive shape / mode

compression/tucker_validation.py
→ Tucker shape / rank Mapping / unfolding最大rank
→ 実数dtype
→ tolerance / max_iter

conv_tucker.py内部validation
→ Conv2d 4階shape
→ rank_out / rank_in
→ component shape / dtype / device
→ groups=1 / Sequential構造
```

似た整数・shape validationが一部に重複しているが、現時点では依存方向を壊す大規模refactorを避けている。特に `compression.hooi` が `metrics` を利用するため、metrics側からcompression helperを安易にimportすると循環importを作る。

NumPy scalarをどのAPIまで受理するかは全体で完全統一しておらず、後続のAPI設計課題として残している。

---

## 7. 評価・学習側で固定したcontract

Tucker/HOOIの分解実装だけでなく、比較実験の共通基盤もsrc reviewで固定した。

- `evaluate()` は処理前後でroot + 全submoduleの `.training` 状態を個別に復元
- `agreement()` / `logits_rmse()` / `benchmark_inference()` / `collect_compression_metrics()` も同じ状態復元contractを利用
- `train_one_epoch()` / `evaluate()` はempty loaderを明示的に拒否
- `agreement()` / `logits_rmse()` は `len(loader)` に依存せず `IterableDataset` を扱える
- `benchmark_inference()` の `warmup` は0以上、`repeats` は1以上のboolではない整数
- baseline / compressedのlatency比較では同じ `input_batch` を使う

`collect_compression_metrics()` は複数指標でloaderを再走査するため、現状はone-shot iteratorではなく再走査可能なloaderを前提とする。

---

## 8. Notebookとsrcの役割分担

```text
Notebook
→ 学習過程
→ 自作アルゴリズムを残す
→ なぜそうなるかを確認する

src
→ 最終的な再利用実装
→ validation / docstring / 型を整理
→ 実験Notebookから呼び出す
```

そのため、

```text
02_hooi.ipynb
04_hooi_tucker2.ipynb
```

は自作実装を学習記録として残し、

```text
05_hosvd_vs_hooi_finetuning.ipynb
```

はsrc化後、重複HOOI処理をsrc呼び出しへ置換した。

## 9. TensorLy `partial_tucker` との対応

自作HOOIの検算では、同じweight、圧縮mode、rank、SVD初期化をTensorLyへ渡す。現行環境のTensorLy 0.9.0では、Conv weightのmode 0 / 1だけを圧縮する呼び出しは次の形になる。

```python
import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker, tucker
from tensorly.tenalg import mode_dot, multi_mode_dot
from tensorly.tucker_tensor import tucker_to_tensor

# TensorLy内部でもPyTorch Tensorを使う。
tl.set_backend("pytorch")

# unfold / foldは自作APIとのshape・値の照合に使える。
mode0_matrix = tl.unfold(weight, mode=0)
weight_roundtrip = tl.fold(
    mode0_matrix,
    mode=0,
    shape=weight.shape,
)
assert torch.equal(weight_roundtrip, weight)

modes = [0, 1]
(core_tl, factors_tl), reported_errors = partial_tucker(
    weight,
    rank=[rank_out, rank_in],
    modes=modes,
    init="svd",
    n_iter_max=20,
    tol=1e-6,
)

# factors_tl[0]とfactors_tl[1]が、指定したmode 0と1に対応する。
factors_tl_by_mode = dict(zip(modes, factors_tl))

# 1 modeだけを射影するmode productも個別に確認できる。
projected_mode0 = mode_dot(
    weight,
    factors_tl_by_mode[0].T,
    mode=0,
)
assert projected_mode0.shape[0] == rank_out

# partial Tuckerでは、圧縮したmodeだけfactorを掛け戻す。
weight_hat_tl = multi_mode_dot(
    core_tl,
    factors_tl,
    modes=modes,
)

relative_error_tl = (
    tl.norm(weight - weight_hat_tl, 2)
    / tl.norm(weight, 2)
)
```

全4 modeをfactorへ分解する通常のTucker分解なら、`tucker()`と`tucker_to_tensor()`を組み合わせる。

```python
# 空間mode 2, 3は完全rankのままにした例。
full_ranks = [
    rank_out,
    rank_in,
    weight.shape[2],
    weight.shape[3],
]

core_full, factors_full = tucker(
    weight,
    rank=full_ranks,
    init="svd",
)
weight_hat_full = tucker_to_tensor(
    (core_full, factors_full)
)

assert weight_hat_full.shape == weight.shape
```

`tucker_to_tensor()`は、全modeに対応するfactorを持つ完全なTucker表現を受け取る。`partial_tucker()`の返すfactorは指定mode分だけなので、その再構成には `multi_mode_dot(core, factors, modes=modes)` を使う方がmode対応を明示できる。

対応関係は

| 自作実装 | TensorLy | 比較対象 |
|---|---|---|
| `{0: rank_out, 1: rank_in}` | `modes=[0, 1]`, `rank=[rank_out, rank_in]` | 圧縮modeとrank |
| HOSVD初期化 | `init="svd"` | 初期部分空間 |
| `core` | `core_tl` | core shape |
| `factors[0]`, `factors[1]` | `factors_tl[0]`, `factors_tl[1]` | factor shape |
| 自作再構成 | `multi_mode_dot(...)` | 外部で再計算した誤差 |
| 全mode Tucker再構成 | `tucker_to_tensor(...)` | 元shapeへの再構成 |
| `unfold` / `fold` / `mode_dot` | `tl.unfold` / `tl.fold` / `mode_dot` | Tensor演算のshapeと値 |

となる。factor自体には符号や同じ部分空間内の回転の自由度があるため、要素の完全一致を合否条件にしない。最終的なshape、再構成weight、同じ式で外部再計算したrelative Frobenius errorを比較する。

---

## 10. 回帰テスト

Tucker/HOOI src化直後はローカル `99 passed` だったが、その後src全体のcontract reviewを複数回行った。

現行rv6では、

```text
252 passed
0 failed
```

をローカル全pytestで確認している。これはTucker専用テスト数ではなくリポジトリ全体のtest suiteであり、GitHub CIによる独立確認を意味しない。

主な確認項目：

- 3階Tensor HOOI / 4階Conv weightのpartial HOOI
- core / factor shape
- `hooi_sweep` が入力factorを破壊しない
- HOOI error history / convergence
- HOSVD buildとcomponents buildのforward一致
- Tensor ndim / positive shape / fold shape
- mode / rankのbool拒否
- Tucker ranks Mapping contract
- mode-n unfolding最大rank
- projected HOOI rank feasibility
- `max_iter=0` でもfeasibility検証
- 複素dtype拒否
- device / dtype / requires_grad
- Tucker-2 Tensor-level autograd保持
- Module構築後Parameterのleaf性・storage非共有
- bias / spatial config / groups contract
- empty loader / IterableDataset
- root + 全submoduleのtrain/eval状態復元
- benchmark `warmup / repeats` validation
- CUDA / device contract

検証詳細：[[06_Tucker基礎実装検証/01_Tucker_HOSVD基礎実装の確認結果]]
