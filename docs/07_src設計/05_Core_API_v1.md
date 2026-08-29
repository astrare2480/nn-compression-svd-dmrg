---
title: Core API v1
aliases:
  - Public API freeze
  - API仕様
---

# Core API v1

## 1. 目的

各sub-packageの `__all__` を基準にPublic APIを一覧化し、すべて同じ観点で記載する。

| 項目 | 意味 |
|---|---|
| 責務 | その関数・クラスが何を担当するか |
| 引数 | 呼び出し側から何を渡すか |
| 戻り値 | 何が返るか |
| 使用場面 | どの処理・実験で使うか |
| Stability | A=Primary、B=Experiment Support、C=Compatibility |

Core API v1では、これらの**外部から見える意味を原則breaking changeしない**。

---

# 2. `nn_compression.tensor`

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `unfold(X, mode)` | 指定modeを行方向にしたmode-n unfoldingを作る | `X`: 2階以上Tensor、`mode`: axis | `(X.shape[mode], prod(other dims))` Tensor | HOSVD/HOOIでmodeごとにSVDする前処理 | A |
| `fold(unfolded, mode, shape)` | `unfold()` のaxis規約を逆変換する | 2D `unfolded`、`mode`、復元`shape` | `shape`を持つTensor | unfoldingの逆変換・round-trip検証 | A |
| `mode_dot(X, matrix, mode)` | Tensorと行列のmode-n積を計算する | `X`、`matrix(new_dim, X.shape[mode])`、`mode` | 対象modeだけ`new_dim`に置換したTensor | Tucker projection / reconstruction | A |

### 主なcontract

```text
Tensor ndim >= 2
各dimension > 0
modeはbool以外のint
0 <= mode < ndim
mode_dot: matrix.shape[1] == X.shape[mode]
```

---

# 3. `nn_compression.compression` — SVD / Linear / Conv2d

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `truncated_svd(matrix, rank)` | 上位rank成分だけのSVDを計算 | 2D `matrix`、`rank` | `U_r, S_r, Vh_r` | Linear/Conv SVD、HOSVD、HOOI更新 | A |
| `retained_energy_from_matrix(matrix, rank)` | 上位特異値のenergy保持率を計算 | 2D `matrix`、`rank` | `float` | rank候補のweight近似品質確認 | A |
| `retained_energy(layer, rank)` | layer weightのretained energyを計算 | weightを持つ`layer`、`rank` | `float` | layer rank sweepの補助指標 | A |
| `rebuild_linear_from_svd(U_r, S_r, Vh_r, layer)` | SVD成分から元shapeの単一Linearを再構築 | SVD成分、元`nn.Linear` | 新しい`nn.Linear` | parameter削減せず近似誤差だけ確認 | A |
| `factorize_linear_layer(layer, rank)` | Linearを`in→rank→out`へ2層化 | `nn.Linear`、`rank` | `nn.Sequential` | Linear SVD圧縮 | A |
| `factorize_named_linear(model, layer_name, rank)` | 指定Linearだけ圧縮したmodel copyを作る | model、dot path、rank | `nn.Module` copy | `fc1`等の単独圧縮 | A |
| `conv2d_weight_matrix(weight)` | 4D Conv weightをSVD用2D matrixへ変換 | `(C_out,C_in,kH,kW)` weight | `(C_out,C_in*kH*kW)` Tensor | Conv SVD前処理 | A |
| `factorize_conv2d_layer(conv, rank)` | Convを`k×k(in→r) + 1×1(r→out)`へ2層化 | `nn.Conv2d`、rank | `nn.Sequential` | Conv SVD圧縮 | A |
| `factorize_named_conv2d(model, layer_name, rank)` | 指定Convだけ圧縮したmodel copyを作る | model、dot path、rank | `nn.Module` copy | `conv1/conv2`等の単独圧縮 | A |
| `factorize_named_layers(model, *, conv_ranks=None, linear_ranks=None)` | 複数named Conv/Linearを一括圧縮 | model、`{name: rank}`辞書 | `nn.Module` copy | model-wide SVD圧縮 | A |
| `make_one_layer_svd_model(model, fc1_rank=None, fc2_rank=None, *, r1=None, r2=None)` | 現行MLPのfc1/fc2をshape不変のSVD再構築Linearへ交換 | model、fc1/fc2 rank、legacy r1/r2 | model copy | historical MLP近似誤差実験 | B |
| `make_two_layer_svd_model(model, fc1_rank=None, fc2_rank=None, *, r1=None, r2=None)` | 現行MLPのfc1/fc2をfactorized Linearへ交換 | model、fc1/fc2 rank、legacy r1/r2 | model copy | MNIST/Fashion-MNIST MLP圧縮・FT | B |

### SVD rank contract

```text
bool拒否
Tensor scalar拒否
整数scalar
1 <= rank <= min(matrix.shape)
NumPy整数scalarは受理
```

### Module contract

```text
入力layer/modelを破壊しない
device / dtype / requires_grad維持
Linear biasは最終層
Conv biasは1x1出力側
Conv SVDはgroups=1のみ
元stride/padding/dilation/padding_modeは空間Convへ継承
```

---

# 4. Tucker / HOSVD / HOOI

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `hosvd(X, ranks)` | 指定modeをtruncated HOSVDで低rank化 | `X`、`{mode: rank}` Mapping | `(core, factors)` | generic Tucker分解、HOOI初期化 | A |
| `reconstruct_tucker(core, factors)` | coreとfactorから近似Tensorを再構成 | `core`、`{mode: U}` | Tensor | 再構成誤差、effective weight生成 | A |
| `tucker_parameter_count(shape, ranks)` | Tucker表現の保存要素数を計算 | 元shape、ranks | `int` | Tucker rank候補のparameter比較 | A |
| `parameter_ratio(shape, ranks)` | Tucker要素数/元要素数を計算 | shape、ranks | `float` | 圧縮率を「割合」で比較 | A |
| `compression_factor(shape, ranks)` | 元要素数/Tucker要素数を計算 | shape、ranks | `float` | 圧縮率を「何倍」で比較 | A |
| `has_converged(error, prev_error, *, abs_tol=1e-8, rel_tol=1e-5)` | 誤差変化から収束判定 | 現在/前回error、tolerance | `bool` | HOOI停止判定 | A |
| `hooi_sweep(X, factors, ranks)` | HOOIの1 sweepでfactorを更新 | `X`、現在factors、ranks | 更新factor `dict` | sweep単位の検証・HOOI本体 | A |
| `core_from_factors(X, factors)` | 現factorからTucker coreを再計算 | `X`、factor dict | core Tensor | HOOI sweep後のcore更新 | A |
| `hooi(X, ranks, max_iter=30, abs_tol=1e-8, rel_tol=1e-5)` | HOSVD初期化からHOOIを反復 | `X`、ranks、iteration/tolerance | `(core, factors, history)` | HOSVDより再構成誤差を反復改善 | A |

### HOSVD/Tucker contract

```text
X/core ndim >= 2
HOSVD/HOOI decompositionは実数浮動小数点Tensor
ranksはMapping
rank上限はmode-n unfoldingの数学的最大rank
hosvd/count系の空{}はidentityとして許可
hosvd factorは逐次coreではなく元Xの各unfoldingから独立に求める
```

### HOOI contract

```text
ranksは空不可
factor keys == rank keys
factor shape == (X.shape[mode], rank)
factor dtype/device == X
projected unfolding feasibilityを反復前に検証
hooi_sweepは入力factorsを破壊しない
更新順はranksの挿入順
history[0]はHOSVD初期誤差
max_iter=0は初期値のみ返す
toleranceは有限・0以上・bool拒否
```

---

# 5. Tucker-2 Conv

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `tucker2_decompose_conv_weight(weight, rank_out, rank_in)` | Conv weightのchannel mode 0/1をHOSVD分解 | 4D weight、output/input rank | `(core, u_out, u_in)` | Tucker-2成分だけ取得、HOSVD/HOOI比較 | A |
| `build_tucker2_conv_from_components(conv, core, u_out, u_in)` | 既計算componentsを3層Convへ変換 | 元Conv、core、factor2枚 | `nn.Sequential` | HOSVD/HOOI共通Module builder | A |
| `build_tucker2_conv(conv, rank_out, rank_in)` | HOSVD分解と3層Module構築を一括実行 | Conv、channel ranks | `nn.Sequential` | HOSVD Tucker-2で直接layerを置換 | A |
| `tucker2_hooi(weight, rank_out, rank_in, max_iter=30, abs_tol=1e-8, rel_tol=1e-5)` | Conv channel modeへpartial HOOI適用 | weight、ranks、iteration/tolerance | `(core, factors, history)` | Tucker-2 HOOI分解 | A |
| `tucker2_hooi_sweep(weight, factors, rank_out, rank_in)` | Tucker-2 HOOIを1 sweep実行 | weight、現在factors、ranks | 更新factor `dict` | HOOI更新過程の確認 | A |
| `tucker2_effective_weight(seq)` | 3層Tucker-2 Moduleから等価4D weightを再構成 | Tucker-2 `nn.Sequential` | 4D Tensor | Fine-tuning後weight誤差評価 | A |

### 構造contract

```text
C_in
→ 1x1 input projection (C_in→R_in)
→ core Conv (R_in→R_out, original spatial config)
→ 1x1 output projection (R_out→C_out)
```

```text
groups=1
biasはoutput projectionのみ
Module構築後Parameterはleaf
元weightとstorage非共有
build_tucker2_convは元weightをdetachして初期化
tucker2_hooiはTensor-level APIなので入力weightをdetachしない
```

### rank contract

component builderのchannel shape制約に加え、分解APIでは4D weightに対し、

```text
rank_out <= min(C_out, C_in*kH*kW)
rank_in  <= min(C_in, C_out*kH*kW)
```

を適用する。

---

# 6. rank sweep

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `sweep_layer_ranks(model, layer_name, ranks, loader, criterion, device, *, factorize, baseline_acc, baseline_time_s, macs_fn=None, warmup=5, repeats=200, verbose=True, input_batch=None)` | 任意named layerの複数rankを同条件評価 | model、layer、rank列、評価条件、factorize callback等 | `list[dict]` | generic rank sweep | B |
| `sweep_conv2d_ranks(model, layer_name, ranks, out_hw, loader, criterion, device, *, baseline_acc, baseline_time_s, warmup=5, repeats=200, verbose=True, input_batch=None)` | Conv SVD rankをMACs込みで評価 | model、Conv名、ranks、out_hw、評価条件 | `list[dict]` | Conv SVD rank候補比較 | B |
| `sweep_conv_svd_ranks(model, layer_name, rank_list, out_hw, loader, criterion, device, *, baseline_acc, baseline_time_s, warmup=5, repeats=200, verbose=True, input_batch=None)` | 旧Notebook引数名のwrapper | historical引数 | `list[dict]` | 旧Notebook再現 | C |

### sweep contract

```text
各candidateはbaselineからdeepcopy
元modelは変更しない
結果へlayer/rank/retained_energy/共通metricsを付与
candidate model本体は既定で保持しない
```

---

# 7. `nn_compression.training`

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `train_one_epoch(model, train_loader, criterion, optimizer, device)` | 1 epochの学習・optimizer update | model、train loader、loss、optimizer、device | `(loss, accuracy)` | baseline学習・圧縮後Fine-tuning | A |
| `evaluate(model, loader, criterion, device)` | optimizer updateなしでdataset評価 | model、loader、loss、device | `(loss, accuracy)` | validation/test | A |
| `fit_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, max_epochs, patience, min_delta, *, reevaluate_train=True, log_every_epoch=False, train_eval_loader=None)` | Early Stopping付き学習とbest state復元 | 学習一式、停止条件、train再評価設定 | result `dict` | baseline学習・Fine-tuning | B |
| `non_shuffling_loader(loader)` | 同Datasetをshuffleせず走査するloader作成 | 元DataLoader | `DataLoader` | train Generatorを進めずtrain metric再評価 | B |

### training contract

```text
lossはsample-weighted average
accuracyは全sample正解率
train_one_epoch/evaluateはempty loaderをValueError
evaluateは正常/例外時とも全submodule training stateを復元
train_one_epochはmodel.train()を設定
fit_with_early_stoppingはbest validation loss時のstate_dictへ復元
```

`fit_with_early_stopping()` のresult dict：

```text
model
best_epoch
best_validation_loss
train_loss_history
validation_loss_history
history
```

---

# 8. `nn_compression.metrics` — model comparison

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `count_parameters(model)` | 全Parameter要素数を数える | model | `int` | model size比較 | A |
| `parameters_reduction(baseline_model, compressed_model)` | parameter削減率を計算 | baseline/compressed model | `float` | 圧縮率評価 | A |
| `accuracy_drop(baseline_accuracy, compressed_accuracy, verbose=True)` | baselineからのaccuracy低下量を計算 | 2つのaccuracy、表示flag | `float` | task性能変化の比較 | A |
| `agreement(baseline_model, compressed_model, loader, device)` | 予測class一致率を計算 | 2 model、loader、device | `float` | baselineの判断保持率を評価 | A |
| `logits_rmse(baseline_model, compressed_model, loader, device)` | logits全要素RMSEを計算 | 2 model、loader、device | `float` | 出力分布のずれ評価 | A |
| `take_inference_batch(data_loader)` | benchmark用固定inputを取得 | DataLoader | input Tensor | baseline/compressedへ同一inputを渡す | A |
| `benchmark_inference(model, data_loader=None, device=None, warmup=10, repeats=5000, *, input_batch=None, return_details=False)` | 1 batch平均forward時間を測定 | model、device、count、input source | `float` またはdetail `dict` | 実測latency評価 | A |
| `benchmark_inference_print(baseline_model, compressed_model, data_loader, device, verbose=True, *, warmup=10, repeats=5000, input_batch=None)` | 2 modelを同条件benchmark | baseline/compressed、loader/device/計測条件 | `(baseline_time_s, compressed_time_s)` | Notebookで簡易比較 | A |
| `collect_compression_metrics(baseline_model, compressed_model, loader, criterion, device, *, baseline_acc, baseline_time_s, warmup=5, repeats=200, compressed_macs=None, compute_reduction=None, baseline_macs=None, include_model=False, input_batch=None)` | 圧縮candidateの共通metricsを1 recordへ集約 | 2 model、評価条件、baseline値、optional MACs | `dict` | rank sweep・比較表 | A |

### model comparison contract

```text
count_parametersはrequires_grad=Falseも含む
agreement/logits_rmseはloader全体で集計しempty拒否
評価・benchmark前後で全submodule training stateを復元
benchmarkはsame input batchで比較可能
warmup >= 0
repeats >= 1
bool / Tensor scalar拒否
shuffle付きRandomSamplerからtake_inference_batchしない
collect_compression_metricsはinclude_model=Falseが既定
```

`benchmark_inference(return_details=True)`：

```text
time_s
time_ms
batch_size
input_shape
warmup
repeats
```

---

# 9. MACs

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `linear_macs(layer)` | baseline Linearの理論MACs | Linear layer | `int` | Linear計算量 | A |
| `compressed_linear_macs(in_features, out_features, rank)` | factorized Linearの理論MACs | input/output次元、rank | `int` | Linear SVD rank比較 | A |
| `conv2d_macs(conv, out_h, out_w)` | baseline Convの理論MACs | Conv、出力H/W | `int` | Conv計算量 | A |
| `compressed_conv2d_macs(conv, rank, out_h, out_w)` | SVD 2層Convの理論MACs | Conv、rank、出力H/W | `int` | Conv SVD rank比較 | A |
| `estimate_conv2d_macs(conv, rank, out_hw, verbose=True)` | baseline/compressed MACsと削減率をまとめる | Conv、rank、`(H,W)`、表示flag | `(baseline, compressed, reduction)` | 1 Convの比較・rank sweep | A |
| `factorized_linear_macs` | `compressed_linear_macs`旧名 | 同左 | `int` | 旧Notebook | C |
| `factorized_conv2d_macs` | `compressed_conv2d_macs`旧名 | 同左 | `int` | 旧Notebook | C |

### MACs contract

```text
compressed rankは分解可能上限以内
rankはbool/Tensor scalar拒否
Conv out_h/out_wはbool/Tensor scalar以外の正整数scalar
compressed Convはgroups=1のみ
baseline conv2d_macsはgrouped Convもweight shapeに基づき計算可能
MACsは理論積和回数でありlatencyではない
```

---

# 10. model-specific MACs

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `estimate_mlp_macs(model, fc1_rank, fc2_rank, verbose=True)` | 現行784→512→256→10 MLPのMACs集計 | model、fc1/fc2 rank、表示flag | `(baseline, compressed, reduction)` | MNIST/Fashion-MNIST MLP実験 | B |
| `estimate_cnn_linear_macs(model, fc1_rank, verbose=True)` | Fashion-MNIST CNN Linear部のMACs比較 | model、fc1 rank、表示flag | `(baseline, compressed, reduction)` | CNN Linear SVD | B |
| `estimate_cnn_conv2_macs(model, rank, verbose=True, *, layer_name='conv2', out_hw=None)` | 指定Conv 1層のMACs比較 | model、rank、layer名、out_hw | `(baseline, compressed, reduction)` | CNN Conv SVD | B |
| `estimate_cnn_macs(model, conv2_rank=None, fc1_rank=None, verbose=True, *, conv_ranks=None, linear_ranks=None, conv_output_hw=None, linear_layer_names=('fc1','fc2'))` | 指定Conv/Linear群のmodel-level MACs集計 | model、圧縮rank辞書、空間size辞書等 | `(baseline, compressed, reduction)` | 複数層SVDのmodel-wide比較 | B |

---

# 11. Tensor approximation

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `relative_frobenius_error(X, X_hat)` | `||X-X_hat||_F / ||X||_F`を計算 | 同shapeの元/近似Tensor | scalar Tensor | SVD/Tucker/HOOI weight再構成誤差 | A |

```text
shape一致必須
Xがzero tensorならValueError
関数内ではdetachしない
```

---

# 12. `nn_compression.selection`

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `is_pareto_candidate(df, row, x_column, y_column)` | 1候補が他候補に支配されていないか判定 | DataFrame、対象row、2列名 | `bool` | Pareto判定 | B |
| `extract_pareto_frontier(df, x_column, y_column)` | 非支配候補を抽出 | DataFrame、2列名 | DataFrame | rank候補絞り込み | B |
| `get_first_point(df, column, x, y)` | 指定列sort後の先頭座標取得 | DataFrame、sort列、x/y列 | `(x,y)` | knee端点処理 | B |
| `get_endpoints(df, column, x, y)` | sort後の先頭/末尾座標取得 | DataFrame、sort列、x/y列 | `(left,right)` | knee基準直線 | B |
| `line_equation(p0, p1)` | 2点を通る直線表現を作る | 2座標 | `(slope, intercept)` | knee距離計算 | B |
| `find_knee_point(df, slope, intercept, x, y)` | 直線から距離最大の点を返す | DataFrame、直線、x/y列 | `(point, distance)` | knee選択 | B |
| `find_knee_point_numpy(df, slope, intercept, x, y)` | knee計算のNumPy版 | 同上 | `(point, distance)` | vectorized knee計算 | B |

### selection前提

```text
Paretoは2目的とも小さいほど良い
kneeはempty / NaN / inf拒否
垂直線は(inf, x_constant)
1点だけならその点・距離0
```

---

# 13. `nn_compression.datasets`

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `get_fashion_mnist_datasets(data_dir, *, transform=None, download=True)` | Fashion-MNIST train全体/test取得 | data path、transform、download flag | `(full_train_dataset, test_dataset)` | Fashion-MNISTデータ取得 | B |
| `split_fashion_mnist_dataset(dataset, split_lengths, *, seed)` | Datasetをseed固定で分割 | Dataset、length tuple、seed | Subset tuple | train/val/rank-val分割 | B |
| `make_fashion_mnist_loaders(train_dataset, validation_dataset, test_dataset, *, train_batch_size, validation_batch_size, test_batch_size, rank_validation_dataset=None, train_generator=None)` | 共通shuffle方針でDataLoader群作成 | Dataset群、batch size、optional rank-val/Generator | loader `dict` | Fashion-MNIST実験準備 | B |
| `shuffled_index_splits(n, lengths, *, seed)` | index列をseed固定shuffleして指定長へ分割 | 件数、length tuple、seed | `tuple[list[int], ...]` | 異なるtransform Datasetへ同一split適用 | B |

### loader方針

```text
train_loader: shuffle=True
validation/test/train_eval/rank_validation: shuffle=False
```

---

# 14. `nn_compression.models`

| API / Class | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `MNISTMLP()` | 784→512→256→10 MLPを提供 | constructor引数なし | `nn.Module` instance | MNIST/Fashion-MNIST MLP SVD | B |
| `FashionMNISTCNN()` | 2 Conv + 2 Linear CNNを提供 | constructor引数なし | `nn.Module` instance | Fashion-MNIST CNN SVD | B |
| `FashionMNISTCNN.inspect_shapes(x=None)` | forward途中shapeを表示しflatten特徴数を確認 | optional input Tensor | flatten dimension `int` | `in_features`学習・確認 | B |
| `CIFAR10CNN(*, in_channels=3, conv_channels=(32,64,128), hidden_dim=256, num_classes=10, dropout=0.5)` | Conv×3 + GAP + 2 Linear CNNを提供 | architecture parameters | `nn.Module` instance | CIFAR-10 SVD/Tucker | B |

### 固定layer名

```text
MNISTMLP: fc1, fc2, fc3
FashionMNISTCNN: conv1, conv2, fc1, fc2
CIFAR10CNN: conv1, conv2, conv3, fc1, fc2
```

既存checkpoint・named compressionとの互換性のため、default architectureと主要attribute名を破壊的変更しない。

---

# 15. `nn_compression.utils`

| API | 責務 | 引数 | 戻り値 | 使用場面 | Stability |
|---|---|---|---|---|---|
| `get_named_module(model, name)` | dot pathの既存submodule取得 | model、name | `nn.Module` | generic named layer圧縮 | B |
| `set_named_module(model, name, module)` | 既存submoduleを置換 | model、name、新Module | `None` | named layer replacement | B |
| `find_project_root(start_path)` | 上位を辿り`.git` rootを探す | 開始Path/string | `Path` | Notebookからproject root解決 | B |
| `get_experiment_dirs(project_root, method_name, case_name, experiment_name, *, create=True)` | 実験保存directoryを共通規則で取得/作成 | root、method/case/experiment、create | `(data_dir, models_dir, results_dir)` | SVD/Tucker/TT/DMRG成果物保存 | B |
| `set_seed(seed=0)` | Python/NumPy/PyTorch/CUDA等の乱数設定 | seed | `None` | 実験開始時の再現性設定 | B |
| `make_torch_generator(seed=0)` | DataLoader/random_split用Generator作成 | seed | `torch.Generator` | mini-batch順・split再現 | B |

### utility contract

```text
set_named_moduleは存在しないpathを暗黙作成しない
experiment pathはmethod/case/experiment
set_seedは既存DataLoader Generatorの状態をresetしない
```

---

# 16. Compatibility API

| 旧API | Primary API | 責務/使用場面 | Stability |
|---|---|---|---|
| `SVD` | `truncated_svd` | 旧NotebookのSVD名互換 | C |
| `RebuildSVD` | `rebuild_linear_from_svd` | 旧Linear再構築名 | C |
| `factorize_Conv2d_layer` | `factorize_conv2d_layer` | 旧Conv factorization名 | C |
| `factorized_linear_macs` | `compressed_linear_macs` | 旧MACs名 | C |
| `factorized_conv2d_macs` | `compressed_conv2d_macs` | 旧MACs名 | C |
| `sweep_conv_svd_ranks` | `sweep_conv2d_ranks` | historical rank sweep | C |
| `r1/r2` keyword | `fc1_rank/fc2_rank` | historical MLP Notebook | C |

新規コードではPrimary APIを使用し、Compatibility APIへ新機能は追加しない。

---

# 17. Breaking changeの定義

以下はCore API v1のbreaking changeとして扱う。

```text
Public API名の削除・改名
引数名変更
positional / keyword-only境界変更
default値の意味変更
return tuple/dict structureの非互換変更
shape convention変更
baseline非破壊性の喪失
device/dtype/requires_grad semantics変更
HOOI history[0]等の意味変更
評価後のtraining state復元保証削除
```

TT/MPS・DMRG追加では、これらを変更するより新しいAPIを追加する。
