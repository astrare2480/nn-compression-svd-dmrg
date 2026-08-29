---
title: Core API v1
aliases:
  - Public API freeze
  - API stability policy
---

# Core API v1

## 1. 目的

このファイルは、SVD〜Tucker/HOOI完了時点の `src/nn_compression/` に対して、**今後原則としてbreaking changeを行わないPublic API一覧**を定義する。

Core API v1はPyPI等へ公開したsemantic versionではなく、このリポジトリ内部の安定化境界である。

Public APIの基本判定は各sub-packageの `__all__` とする。

---

## 2. Stability Level

### A: Primary Stable API

新しいNotebook / TT/MPS / DMRG実装から利用してよいprimary API。

原則として以下を破壊的変更しない。

- 名前
- signature
- parameter semantics
- return structure
- 主要validation
- model / autograd / state semantics

### B: Experiment Support Stable API

Dataset、model fixture、Pareto/knee、path/seed等。

既存実験の再現性のため維持する。新しい手法の数学コアではないが、既存Notebookが依存するため安定APIとして扱う。

### C: Compatibility API

旧Notebook互換のため残している別名・legacy entry point。

新規コードではprimary名を使う。Core API v1の間は削除しないが、新機能は追加しない。

### Internal / Unfrozen

`__all__` に含まれない関数・private helper。

例：

```text
coerce_rank
coerce_integer_scalar
linear_max_rank
conv2d_max_rank
_validate_hooi_inputs
_preserve_module_training_modes
_coerce_positive_int_scalar
_coerce_macs_rank
_coerce_int_scalar
```

Public API contractを維持する限り変更可能。

---

# 3. `nn_compression.tensor`

Stability: **A**

```python
unfold(X: torch.Tensor, mode: int) -> torch.Tensor
fold(unfolded: torch.Tensor, mode: int, shape: tuple[int, ...]) -> torch.Tensor
mode_dot(X: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor
```

固定するcontract：

- Tensorは2階以上
- shapeの各dimensionは正
- modeはbool以外のint、`0 <= mode < ndim`
- `unfold` 出力shapeは `(X.shape[mode], prod(other dims))`
- `fold` は `unfold` のaxis規約を逆変換する
- `mode_dot` の `matrix.shape[1] == X.shape[mode]`
- `mode_dot` 出力は対象modeだけ `matrix.shape[0]` に置換される

詳細：[[07_src設計/03_tensor設計]]

---

# 4. `nn_compression.compression`

## 4.1 SVD core

Stability: **A**

```python
truncated_svd(matrix: torch.Tensor, rank: int)
retained_energy_from_matrix(matrix: torch.Tensor, rank: int) -> float
retained_energy(layer: nn.Module, rank: int) -> float
```

`truncated_svd` のreturn：

```text
U_r  : (out, rank)
S_r  : (rank,)
Vh_r : (rank, in)
```

rank contract：

```text
bool / Tensor scalar拒否
整数scalar
1 <= rank <= min(matrix.shape)
```

NumPy整数scalarはSVD側では受理する。

---

## 4.2 Linear SVD

Stability: **A**

```python
rebuild_linear_from_svd(U_r, S_r, Vh_r, layer: nn.Linear) -> nn.Linear
factorize_linear_layer(layer: nn.Linear, rank: int) -> nn.Sequential
factorize_named_linear(model: nn.Module, layer_name: str, rank: int) -> nn.Module
```

固定するcontract：

- 生成layerは元layerと同じdevice / dtype
- `requires_grad` を維持
- biasは出力側へ保持
- `factorize_linear_layer` は `in -> rank -> out`
- named APIは元modelを変更せずdeepcopyしたmodelを返す
- baselineとcompressedでParameterを共有しない

---

## 4.3 Conv2d SVD

Stability: **A**

```python
conv2d_weight_matrix(weight: torch.Tensor) -> torch.Tensor
factorize_conv2d_layer(conv: nn.Conv2d, rank: int) -> nn.Sequential
factorize_named_conv2d(model: nn.Module, layer_name: str, rank: int) -> nn.Module
```

`conv2d_weight_matrix`：

```text
(out_ch, in_ch, kH, kW)
→ (out_ch, in_ch*kH*kW)
```

factorized Conv：

```text
Conv(in -> rank, original kernel/stride/padding/dilation)
→ Conv(rank -> out, 1x1)
```

固定するcontract：

- `groups=1` の通常Conv2dのみ
- rank上限は `min(out_ch, in_ch*kH*kW)`
- 元biasは2層目
- 元 `padding_mode` は1層目へ継承
- device / dtype / requires_grad維持
- named APIはbaseline非破壊

---

## 4.4 複数named layer

Stability: **A**

```python
factorize_named_layers(
    model: nn.Module,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
) -> nn.Module
```

- Conv2d / Linearの辞書を分離
- 分解元は常に未分解のbaseline layer
- 元modelは変更しない

---

## 4.5 MLP組み立て

Stability: **B**

```python
make_one_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
)

make_two_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
)
```

- `fc1` / `fc2` を持つ現行MLP構造向け
- `r1` / `r2` はlegacy keyword
- primary名とlegacy名の同時指定はValueError
- deepcopyし、`fc3` を含めbaselineと共有しない

---

## 4.6 HOSVD / Tucker

Stability: **A**

```python
hosvd(
    X: torch.Tensor,
    ranks: dict[int, int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]

reconstruct_tucker(
    core: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor

tucker_parameter_count(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> int

parameter_ratio(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float

compression_factor(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float
```

固定するcontract：

```text
X / coreは2階以上
ranksはMapping
{mode: rank}
mode = bool以外のint
rank = bool以外の整数scalar
rank <= mode-n unfoldingの最大rank
```

`hosvd` / count系では空 `{}` をidentity指定として許可する。

`reconstruct_tucker(core,{})` もidentity再構成としてcoreを返すが、core自体は2階以上を要求する。

`hosvd` はfactorを**逐次projectしたcoreではなく、元のXの各unfoldingから独立に求める**。

HOSVD分解入口は実数Tensor限定。

---

## 4.7 HOOI

Stability: **A**

```python
has_converged(
    error: float,
    prev_error: float,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> bool

hooi_sweep(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> dict[int, torch.Tensor]

core_from_factors(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor

hooi(
    X: torch.Tensor,
    ranks: dict[int, int],
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]
```

固定するcontract：

- HOOIの `ranks` は空不可
- `hooi / hooi_sweep / core_from_factors` のXは2階以上・実数Tensor
- HOSVD初期化
- `history[0]` はHOSVD初期誤差
- `max_iter=0` は初期値だけ返す
- factor keys == rank keys（`hooi_sweep` / `hooi` のvalidation経路）
- factor shape `(X.shape[mode], rank)`
- factor dtype/device == X
- projected unfoldingの実現可能rankを反復前に検証
- `hooi_sweep` は入力factorsを破壊しない
- sweepは `ranks` の挿入順
- toleranceは有限・0以上、bool拒否

`core_from_factors` は単独utilityとしてfactor key集合とranksの完全一致までは検証しない。各factorは `mode_dot` のshape/mode contractに従う。

---

## 4.8 Tucker-2 Conv

Stability: **A**

```python
build_tucker2_conv(
    conv: nn.Conv2d,
    rank_out: int,
    rank_in: int,
) -> nn.Sequential

build_tucker2_conv_from_components(
    conv: nn.Conv2d,
    core: torch.Tensor,
    u_out: torch.Tensor,
    u_in: torch.Tensor,
) -> nn.Sequential

tucker2_decompose_conv_weight(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

tucker2_hooi(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]

tucker2_hooi_sweep(
    weight: torch.Tensor,
    factors: dict[int, torch.Tensor],
    rank_out: int,
    rank_in: int,
) -> dict[int, torch.Tensor]

tucker2_effective_weight(seq: nn.Sequential) -> torch.Tensor
```

構造contract：

```text
C_in
→ 1x1 / U_in.T
→ R_in
→ kHxkW / core
→ R_out
→ 1x1 / U_out
→ C_out
```

共通contract：

- `groups=1` の通常Conv2d
- core layerが元stride/padding/dilation/padding_modeを継承
- biasはoutput projectionのみ
- rankはbool以外の整数scalar
- component builderでは `rank_out <= C_out`, `rank_in <= C_in`
- componentsはconv.weightとdtype/device一致
- Module構築後Parameterはleafで元weight storageを共有しない
- `build_tucker2_conv` は元weightをdetachして初期化
- Tensor-level `tucker2_hooi` は入力weightをdetachしない
- `tucker2_effective_weight` は評価用effective weightを返す

### 分解APIの追加rank上限

`tucker2_decompose_conv_weight` / `tucker2_hooi` / `tucker2_hooi_sweep` と、それらを使う `build_tucker2_conv` では、generic HOSVD/HOOI側のmode-unfolding feasibilityも適用される。

4階weight `(C_out,C_in,kH,kW)` では、

```text
mode 0 max = min(C_out, C_in*kH*kW)
mode 1 max = min(C_in, C_out*kH*kW)
```

を超えるrankはrejectする。

`build_tucker2_conv_from_components` は「既に計算されたcomponentsをModuleへ写す」APIなので、channel shape整合を確認し、分解時のunfolding feasibilityを再計算する責務は持たない。

複素dtypeについては「分解API」が実数限定。componentsからのModule構築・再構成utility自体には全関数一律のcomplex guardを置いていない。詳細は [[07_src設計/04_compression設計]]。

---

## 4.9 rank sweep

Stability: **B**

```python
sweep_layer_ranks(
    model,
    layer_name,
    ranks,
    loader,
    criterion,
    device,
    *,
    factorize,
    baseline_acc,
    baseline_time_s,
    macs_fn=None,
    warmup=5,
    repeats=200,
    verbose=True,
    input_batch=None,
) -> list[dict]

sweep_conv2d_ranks(
    model,
    layer_name,
    ranks,
    out_hw,
    loader,
    criterion,
    device,
    *,
    baseline_acc,
    baseline_time_s,
    warmup=5,
    repeats=200,
    verbose=True,
    input_batch=None,
) -> list[dict]
```

- candidate model本体は既定で保持しない
- 元modelを変更しない
- 結果へ `layer / rank / retained_energy` と共通metricsを付与

---

# 5. `nn_compression.training`

Stability: **A/B**

```python
train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
) -> tuple[float, float]

evaluate(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float]

fit_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    max_epochs,
    patience,
    min_delta,
    *,
    reevaluate_train=True,
    log_every_epoch=False,
    train_eval_loader=None,
)

non_shuffling_loader(loader: DataLoader) -> DataLoader
```

`train_one_epoch` / `evaluate` はA、`fit_with_early_stopping` / `non_shuffling_loader` はBとして扱う。

固定するcontract：

- lossはサンプル数加重平均
- accuracyは全サンプル正解率
- empty loaderは `ValueError`
- `evaluate` は処理前のroot + 全submodule training状態を正常/例外時とも復元
- `train_one_epoch` は学習APIとして `model.train()` を設定
- Early Stoppingはbest validation lossのstate_dictへ復元
- train metric再評価ではshuffle付きtraining loaderを再走査しない

---

# 6. `nn_compression.metrics`

## 6.1 model comparison

Stability: **A**

```python
count_parameters(model)
parameters_reduction(baseline_model, compressed_model)
accuracy_drop(baseline_accuracy, compressed_accuracy, verbose=True)
agreement(baseline_model, compressed_model, loader, device)
logits_rmse(baseline_model, compressed_model, loader, device)
take_inference_batch(data_loader: DataLoader) -> torch.Tensor

benchmark_inference(
    model,
    data_loader=None,
    device=None,
    warmup=10,
    repeats=5000,
    *,
    input_batch=None,
    return_details: bool = False,
)

benchmark_inference_print(
    baseline_model,
    compressed_model,
    data_loader,
    device,
    verbose=True,
    *,
    warmup=10,
    repeats=5000,
    input_batch=None,
)

collect_compression_metrics(
    baseline_model,
    compressed_model,
    loader,
    criterion,
    device,
    *,
    baseline_acc: float,
    baseline_time_s: float,
    warmup: int = 5,
    repeats: int = 200,
    compressed_macs: int | None = None,
    compute_reduction: float | None = None,
    baseline_macs: int | None = None,
    include_model: bool = False,
    input_batch=None,
) -> dict
```

固定するcontract：

- `count_parameters` はrequires_gradに関係なく全Parameter要素数
- `agreement / logits_rmse` はloader長に依存せず走査し、emptyを明示拒否
- metrics実行前後で全submodule training状態を復元
- benchmarkはsame input batchで比較可能
- `warmup >= 0`, `repeats >= 1`, bool拒否、整数必須
- shuffle付きRandomSamplerからの `take_inference_batch` は拒否
- `collect_compression_metrics(include_model=False)` が既定

---

## 6.2 MACs

Stability: **A**

```python
linear_macs(layer) -> int
compressed_linear_macs(in_features: int, out_features: int, rank: int) -> int
conv2d_macs(conv, out_h: int, out_w: int) -> int
compressed_conv2d_macs(conv, rank: int, out_h: int, out_w: int) -> int
estimate_conv2d_macs(conv, rank: int, out_hw: tuple[int, int], verbose: bool = True)
```

Compatibility alias：

```python
factorized_linear_macs = compressed_linear_macs
factorized_conv2d_macs = compressed_conv2d_macs
```

固定する主contract：

- compressed rankはSVDと同じ数学的最大rankを超えない
- `conv2d_macs / compressed_conv2d_macs / estimate_conv2d_macs` の出力空間sizeはbool以外の正の整数
- compressed Convはgroups=1のみ
- baseline `conv2d_macs` はPyTorch weight shapeを使うためgrouped Convも計算可能
- MACsは理論積和回数でありlatencyではない

---

## 6.3 model-specific MACs

Stability: **B**

```python
estimate_mlp_macs(model, fc1_rank, fc2_rank, verbose=True)
estimate_cnn_linear_macs(model, fc1_rank, verbose=True)

estimate_cnn_conv2_macs(
    model,
    rank,
    verbose=True,
    *,
    layer_name="conv2",
    out_hw=None,
)

estimate_cnn_macs(
    model,
    conv2_rank=None,
    fc1_rank=None,
    verbose=True,
    *,
    conv_ranks=None,
    linear_ranks=None,
    conv_output_hw=None,
    linear_layer_names=("fc1", "fc2"),
)
```

既存Fashion-MNIST実験互換のdefaultを維持する。

---

## 6.4 Tensor approximation

Stability: **A**

```python
relative_frobenius_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
) -> torch.Tensor
```

- shape一致必須
- denominator `||X||` が0ならValueError
- Tensorで返し、必要な箇所で呼び出し側がdetach/float化する

---

# 7. `nn_compression.selection`

Stability: **B**

```python
is_pareto_candidate(df, row, x_column, y_column)
extract_pareto_frontier(df, x_column, y_column)
get_first_point(df, column, x, y)
get_endpoints(df, column, x, y)
line_equation(p0, p1)
find_knee_point(df, slope, intercept, x, y)
find_knee_point_numpy(df, slope, intercept, x, y)
```

- Paretoは2目的とも「小さいほど良い」前提
- knee入力は空 / NaN / infを拒否
- 1点だけならその点、距離0
- 垂直線は `(inf, x0)` 表現

---

# 8. `nn_compression.datasets`

Stability: **B**

```python
get_fashion_mnist_datasets(
    data_dir,
    *,
    transform=None,
    download=True,
)

split_fashion_mnist_dataset(
    dataset,
    split_lengths,
    *,
    seed,
)

make_fashion_mnist_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    *,
    train_batch_size,
    validation_batch_size,
    test_batch_size,
    rank_validation_dataset=None,
    train_generator=None,
)

shuffled_index_splits(
    n: int,
    lengths: tuple[int, ...],
    *,
    seed: int,
) -> tuple[list[int], ...]
```

- trainだけshuffle
- validation/test/rank-validation/train-evalはshuffleしない
- splitはseedで再現可能

---

# 9. `nn_compression.models`

Stability: **B**

```python
MNISTMLP()
FashionMNISTCNN()
CIFAR10CNN(
    *,
    in_channels=3,
    conv_channels=(32, 64, 128),
    hidden_dim=256,
    num_classes=10,
    dropout=0.5,
)
```

既存実験・checkpoint・named layer圧縮がlayer名に依存するため、default architectureと主要attribute名はCore API v1中に破壊的変更しない。

特に固定する名前：

```text
MNISTMLP: fc1, fc2, fc3
FashionMNISTCNN: conv1, conv2, fc1, fc2
CIFAR10CNN: conv1, conv2, conv3, fc1, fc2
```

`FashionMNISTCNN.inspect_shapes(x=None) -> int` も既存の学習用public methodとして維持する。

新しいarchitectureが必要なら既存classの意味を変更せず、新classまたは新explicit optionを追加する。

---

# 10. `nn_compression.utils`

Stability: **B**

```python
get_named_module(model: nn.Module, name: str) -> nn.Module
set_named_module(model: nn.Module, name: str, module: nn.Module) -> None
find_project_root(start_path) -> Path

get_experiment_dirs(
    project_root,
    method_name,
    case_name,
    experiment_name,
    *,
    create=True,
) -> tuple[Path, Path, Path]

set_seed(seed: int = 0) -> None
make_torch_generator(seed: int = 0) -> torch.Generator
```

- named moduleはdot path対応
- `set_named_module` は既存pathのみ置換
- experiment pathは `method/case/experiment`
- global seedとDataLoader generatorは別管理

---

# 11. Compatibility API

Stability: **C**

以下は既存Notebook互換のためCore API v1中は残す。

```python
SVD = truncated_svd
RebuildSVD = rebuild_linear_from_svd
factorize_Conv2d_layer = factorize_conv2d_layer
factorized_linear_macs = compressed_linear_macs
factorized_conv2d_macs = compressed_conv2d_macs
sweep_conv_svd_ranks(...)
```

また、MLP APIの

```text
r1
r2
```

もlegacy keywordとして維持する。

新規コードではprimary名を使用する。

---

# 12. Breaking changeの定義

次はbreaking changeとして扱う。

- Public API名の削除・改名
- positional / keyword-only境界の変更
- 引数名変更
- default値の意味変更
- return tuple / dict structureの非互換変更
- validだった入力を設計判断だけで突然rejectする
- baseline非破壊性を失う
- device/dtype/requires_gradの意味変更
- `history[0]` 等、実験結果解釈に影響するsemantics変更
- train/eval状態復元保証を失う

内部高速化・helper移動・重複整理は、上記を維持する限りbreaking changeではない。

---

# 13. API追加ルール

TT/MPS・DMRGで必要な処理は次の順に検討する。

```text
1. 既存Public APIで表現できるか
2. 既存private helperを一般化できるか
3. 新しいPublic APIを追加するか
4. breaking changeが不可避ならCore API v2として扱うか
```

Core API v1に新APIを追加すること自体は可能だが、追加時に

- `__all__`
- contract test
- 本設計書

を同時更新する。
