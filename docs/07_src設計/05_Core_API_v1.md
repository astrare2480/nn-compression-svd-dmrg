---
title: Core API v1
aliases:
  - Public API freeze
  - API仕様
---

# Core API v1

## 1. 目的

このファイルは、各sub-packageの `__all__` を基準にPublic APIを一覧化し、各APIについて次を定義する。

```text
責務
引数
戻り値
どういうときに使うか
主要contract / 注意事項
```

Core API v1では、これらの**外部から見える意味を原則breaking changeしない**。

内部helperや実装方法は、Public API contractを維持する限り変更可能。

### Stability

```text
A = Primary Stable API
B = Experiment Support Stable API
C = Compatibility API
```

---

# 2. `nn_compression.tensor`

## `unfold(X, mode) -> Tensor`  [A]

- **責務**: Tensorの指定modeを行方向へ移し、2次元mode-n unfoldingを作る。
- **引数**:
  - `X`: 2階以上のTensor。
  - `mode`: unfoldingするaxis。
- **戻り値**: shape `(X.shape[mode], prod(other dimensions))` のTensor。
- **使用場面**: HOSVD/HOOIでmodeごとにSVDを行う前処理。
- **主要contract**: `mode` はbool以外のint、dimensionはすべて正。

## `fold(unfolded, mode, shape) -> Tensor`  [A]

- **責務**: `unfold()` のaxis規約で作られた2次元行列を元Tensor shapeへ戻す。
- **引数**:
  - `unfolded`: 2次元行列。
  - `mode`: 行方向として扱われているmode。
  - `shape`: 復元後shape。
- **戻り値**: `shape` を持つTensor。
- **使用場面**: unfolding処理の逆変換が必要なTensor操作・検証。
- **主要contract**: 行列shapeと指定shapeが整合する必要がある。

## `mode_dot(X, matrix, mode) -> Tensor`  [A]

- **責務**: Tensor `X` と行列のmode-n積を計算する。
- **引数**:
  - `X`: 2階以上のTensor。
  - `matrix`: shape `(new_dim, X.shape[mode])` の2次元行列。
  - `mode`: 掛けるaxis。
- **戻り値**: 対象modeだけ `new_dim` に置き換わったTensor。
- **使用場面**: Tucker core生成、factorによるprojection/reconstruction。
- **主要contract**: `matrix.shape[1] == X.shape[mode]`。

---

# 3. `nn_compression.compression` — SVD

## `truncated_svd(matrix, rank) -> (U_r, S_r, Vh_r)`  [A]

- **責務**: 2次元行列の上位`rank`成分だけを残したSVDを返す。
- **引数**:
  - `matrix`: 2次元Tensor。
  - `rank`: 残す特異値数。
- **戻り値**:
  - `U_r`: `(m, rank)`。
  - `S_r`: `(rank,)`。
  - `Vh_r`: `(rank, n)`。
- **使用場面**: Linear/Conv SVD、HOSVD、HOOI更新の数学コア。
- **主要contract**: `1 <= rank <= min(matrix.shape)`。bool/Tensor scalarは拒否し、NumPy整数scalarは受理する。

## `retained_energy_from_matrix(matrix, rank) -> float`  [A]

- **責務**: 上位特異値が保持するFrobenius energy比率を計算する。
- **引数**: 2次元`matrix`、評価する`rank`。
- **戻り値**: `sum(S[:rank]^2) / sum(S^2)`。
- **使用場面**: rank候補のweight近似品質を確認するとき。
- **注意**: task accuracyを保証する指標ではない。ゼロ行列では`1.0`。

## `retained_energy(layer, rank) -> float`  [A]

- **責務**: NN layerのweightを2次元化しretained energyを計算する。
- **引数**: weightを持つ`layer`、`rank`。
- **戻り値**: retained energy。
- **使用場面**: Linear/Conv layerのrank sweep結果へweight側の近似指標を加えるとき。
- **注意**: 評価用なのでweightはdetachして扱う。

---

# 4. Linear SVD

## `rebuild_linear_from_svd(U_r, S_r, Vh_r, layer) -> nn.Linear`  [A]

- **責務**: truncated SVD成分から、元と同じ入出力shapeの単一Linearを再構築する。
- **引数**: `U_r`, `S_r`, `Vh_r`、元`nn.Linear`。
- **戻り値**: 元と同じ`in_features/out_features`の新しい`nn.Linear`。
- **使用場面**: parameter削減ではなく、truncated SVDによるweight近似誤差だけを確認するとき。
- **主要contract**: device/dtype/bias/trainabilityを元layerに合わせる。

## `factorize_linear_layer(layer, rank) -> nn.Sequential`  [A]

- **責務**: `Linear(in→out)` を `Linear(in→rank) → Linear(rank→out)` へ分解する。
- **引数**: 元`nn.Linear`、`rank`。
- **戻り値**: 2層`nn.Sequential`。
- **使用場面**: Linearのparameter/MACsを実際に削減するSVD圧縮。
- **主要contract**: biasは出力側、元layer非破壊、device/dtype/requires_grad維持。

## `factorize_named_linear(model, layer_name, rank) -> nn.Module`  [A]

- **責務**: model内の指定LinearだけをSVD分解したmodel copyを作る。
- **引数**: `model`、dot pathの`layer_name`、`rank`。
- **戻り値**: 圧縮済みmodel copy。
- **使用場面**: Notebookから特定の`fc1`等だけを圧縮したいとき。
- **主要contract**: 元modelを変更せずdeepcopyする。

---

# 5. Conv2d SVD

## `conv2d_weight_matrix(weight) -> Tensor`  [A]

- **責務**: Conv2d weight `(C_out,C_in,kH,kW)` をSVD用 `(C_out,C_in*kH*kW)` へreshapeする。
- **引数**: Conv2d weight Tensor。
- **戻り値**: 2次元matrix view/reshape結果。
- **使用場面**: Conv SVDの前処理、rank上限やretained energy確認。

## `factorize_conv2d_layer(conv, rank) -> nn.Sequential`  [A]

- **責務**: Conv2dを`kH×kW Conv(in→rank)`と`1×1 Conv(rank→out)`へ分解する。
- **引数**: `nn.Conv2d`、`rank`。
- **戻り値**: 2層Convの`nn.Sequential`。
- **使用場面**: Conv2dのSVD圧縮。
- **主要contract**: `groups=1`のみ。元stride/padding/dilation/padding_modeは1層目へ、biasは2層目へ保持。

## `factorize_named_conv2d(model, layer_name, rank) -> nn.Module`  [A]

- **責務**: model内の指定Conv2dだけを分解したmodel copyを返す。
- **引数**: `model`、`layer_name`、`rank`。
- **戻り値**: 圧縮済みmodel copy。
- **使用場面**: `conv1`/`conv2`等を単独で比較したいとき。
- **主要contract**: baseline非破壊。

## `factorize_named_layers(model, *, conv_ranks=None, linear_ranks=None) -> nn.Module`  [A]

- **責務**: 複数のnamed Conv2d/Linearを一度に圧縮する。
- **引数**:
  - `conv_ranks`: `{layer_name: rank}`。
  - `linear_ranks`: `{layer_name: rank}`。
- **戻り値**: 複数層を置換したmodel copy。
- **使用場面**: model-wide SVD圧縮。
- **主要contract**: 各分解元は圧縮途中のmodelではなく元baseline layer。

---

# 6. MLP実験支援SVD

## `make_one_layer_svd_model(model, fc1_rank=None, fc2_rank=None, *, r1=None, r2=None)`  [B]

- **責務**: 現行MLPのfc1/fc2を、shapeを変えないtruncated-SVD再構築Linearへ差し替える。
- **引数**: fc1/fc2 rank。`r1/r2`はlegacy名。
- **戻り値**: model copy。
- **使用場面**: parameter削減前にweight近似だけのaccuracy影響を確認するhistorical実験。
- **注意**: generic MLP APIではない。

## `make_two_layer_svd_model(...)`  [B]

- **責務**: 現行MLPのfc1/fc2を2層factorized Linearへ置換する。
- **引数**: `fc1_rank`, `fc2_rank` またはlegacy `r1/r2`。
- **戻り値**: 圧縮model copy。
- **使用場面**: MNIST/Fashion-MNIST MLPの実圧縮・Fine-tuning。
- **主要contract**: `fc3`を含めbaselineとParameter共有しない。

---

# 7. Tucker / HOSVD

## `hosvd(X, ranks) -> (core, factors)`  [A]

- **責務**: 指定modeをtruncated HOSVDで低rank化する。
- **引数**:
  - `X`: 2階以上の実数浮動小数点Tensor。
  - `ranks`: `{mode: rank}` Mapping。
- **戻り値**:
  - `core`: 圧縮modeをrankへ縮めたTensor。
  - `factors`: `{mode: U}`。
- **使用場面**: generic Tucker分解、Tucker-2 HOSVDの数学コア、HOOI初期化。
- **主要contract**: 各factorは元`X`の各unfoldingから独立に求める。空`{}`はidentityとして許可。

## `reconstruct_tucker(core, factors) -> Tensor`  [A]

- **責務**: coreへ各factorをmode_dotして元空間の近似Tensorを再構成する。
- **引数**: `core`、`{mode: U}`。
- **戻り値**: 再構成Tensor。
- **使用場面**: HOSVD/HOOIの再構成誤差評価、effective weight生成。
- **主要contract**: `core.ndim >= 2`。空factorsならidentity。

## `tucker_parameter_count(shape, ranks) -> int`  [A]

- **責務**: Tucker表現のcore+factor保存要素数を計算する。
- **引数**: 元`shape`、`ranks` Mapping。
- **戻り値**: 保存要素数。
- **使用場面**: Tucker rank候補のparameter圧縮量を理論比較するとき。
- **注意**: モデル全体parameter数ではなく1 Tensor表現の要素数。

## `parameter_ratio(shape, ranks) -> float`  [A]

- **責務**: `Tucker要素数 / 元要素数` を返す。
- **引数**: `shape`, `ranks`。
- **戻り値**: 比率。小さいほど圧縮。
- **使用場面**: rank sweepの圧縮率指標。

## `compression_factor(shape, ranks) -> float`  [A]

- **責務**: `元要素数 / Tucker要素数` を返す。
- **引数**: `shape`, `ranks`。
- **戻り値**: 圧縮倍率。大きいほど圧縮。
- **使用場面**: 「何倍小さいか」で比較したいとき。

---

# 8. HOOI

## `has_converged(error, prev_error, *, abs_tol=1e-8, rel_tol=1e-5) -> bool`  [A]

- **責務**: HOOIの誤差変化が許容範囲内か判定する。
- **引数**: 現在/前回誤差、絶対/相対tolerance。
- **戻り値**: 収束なら`True`。
- **使用場面**: iterative algorithmの停止判定。
- **主要contract**: toleranceは有限・0以上・bool拒否。

## `hooi_sweep(X, factors, ranks) -> dict[int, Tensor]`  [A]

- **責務**: 指定modeのfactorを1 sweep更新する。
- **引数**: 元Tensor `X`、現在の`factors`、`ranks`。
- **戻り値**: 更新後factor dict。
- **使用場面**: HOOIの反復1単位を検証・再利用するとき。
- **主要contract**: 入力factorsを破壊しない。更新順は`ranks`挿入順。factor shape/dtype/deviceを検証。

## `core_from_factors(X, factors) -> Tensor`  [A]

- **責務**: 現在のfactor集合からTucker coreを計算する。
- **引数**: `X`、`{mode: U}`。
- **戻り値**: core Tensor。
- **使用場面**: HOOI sweep後のcore更新、任意factorからcoreを再計算するとき。
- **注意**: factor keyとrank dictの一致を検証するAPIではない。

## `hooi(X, ranks, max_iter=30, abs_tol=1e-8, rel_tol=1e-5) -> (core, factors, history)`  [A]

- **責務**: HOSVD初期化からHOOIを反復し、最終分解と誤差履歴を返す。
- **引数**: `X`, `ranks`, 最大sweep数、tolerance。
- **戻り値**:
  - `core`
  - `factors`
  - `history`: relative Frobenius errorのlist。
- **使用場面**: HOSVDより再構成誤差を反復改善したいとき。
- **主要contract**: `history[0]`はHOSVD初期誤差。`max_iter=0`でもfeasibilityを検証し初期値だけ返す。

---

# 9. Tucker-2 Conv

## `tucker2_decompose_conv_weight(weight, rank_out, rank_in) -> (core, u_out, u_in)`  [A]

- **責務**: 4階Conv weightのchannel mode 0/1だけをHOSVDで圧縮する。
- **引数**: `(C_out,C_in,kH,kW)` weight、出力/入力channel rank。
- **戻り値**: `core`, `u_out`, `u_in`。
- **使用場面**: Moduleを作らずTucker-2成分だけ欲しいとき、HOSVD/HOOI比較。
- **主要contract**: 実数浮動小数点Tensor、mode-unfolding上限以内。

## `build_tucker2_conv_from_components(conv, core, u_out, u_in) -> nn.Sequential`  [A]

- **責務**: 既に計算されたTucker-2 componentsを3層Conv Moduleへ写す。
- **引数**: 元Conv、core、出力/input factor。
- **戻り値**: `1×1 input projection → core Conv → 1×1 output projection`。
- **使用場面**: HOSVD/HOOI等、分解方法を問わず同じModule構造へ変換したいとき。
- **主要contract**: component shape/device/dtype整合、`groups=1`。新Parameterはleafで元weight storageを共有しない。

## `build_tucker2_conv(conv, rank_out, rank_in) -> nn.Sequential`  [A]

- **責務**: Conv weightをHOSVD分解し、そのままTucker-2 Moduleを構築するconvenience API。
- **引数**: 元Conv、`rank_out`, `rank_in`。
- **戻り値**: 3層Conv `nn.Sequential`。
- **使用場面**: HOSVDで直接Conv layerを置換したいとき。
- **autograd**: 元weightをdetachして新Module初期値を作る。

## `tucker2_hooi(weight, rank_out, rank_in, max_iter=30, abs_tol=..., rel_tol=...)`  [A]

- **責務**: Conv weightのchannel modeにpartial HOOIを適用する。
- **引数**: weight、channel ranks、iteration/tolerance。
- **戻り値**: `(core, factors, history)`。
- **使用場面**: Tucker-2 HOSVDより再構成誤差を反復改善したいとき。
- **autograd**: Tensor-level APIなので入力weightをdetachしない。

## `tucker2_hooi_sweep(weight, factors, rank_out, rank_in) -> dict`  [A]

- **責務**: Tucker-2 HOOIを1 sweepだけ実行する。
- **引数**: weight、現在factors、channel ranks。
- **戻り値**: 更新factor dict。
- **使用場面**: sweep単位の挙動確認、学習用Notebookで更新過程を追うとき。

## `tucker2_effective_weight(seq) -> Tensor`  [A]

- **責務**: 3層Tucker-2 Convから等価な4階weightを再構成する。
- **引数**: Tucker-2構造の`nn.Sequential`。
- **戻り値**: `(C_out,C_in,kH,kW)` effective weight。
- **使用場面**: Fine-tuning後weightと元weightのrelative errorを測るとき。
- **注意**: 評価utilityであり、内部layer weightはdetachして扱う。

---

# 10. rank sweep

## `sweep_layer_ranks(...) -> list[dict]`  [B]

- **責務**: 任意named layerについて複数rank候補を同条件で評価するgeneric sweep。
- **主な引数**: model、layer名、rank列、loader、criterion、device、`factorize` callback、baseline metrics、optional MACs callback、benchmark条件。
- **戻り値**: rankごとのmetrics record list。
- **使用場面**: 新しいfactorization方式にも共通のrank sweep枠組みを使いたいとき。
- **主要contract**: candidateはbaselineからdeepcopy。結果にmodel本体は既定で保持しない。

## `sweep_conv2d_ranks(...) -> list[dict]`  [B]

- **責務**: Conv2d SVD用に`factorize_conv2d_layer`とMACs推定を組み合わせたrank sweep。
- **主な引数**: model、layer名、rank列、`out_hw`、評価条件。
- **戻り値**: rankごとのmetrics list。
- **使用場面**: Conv SVDのrank候補比較。

## `sweep_conv_svd_ranks(...) -> list[dict]`  [C]

- **責務**: 旧Notebook互換の引数名で`sweep_conv2d_ranks()`を呼ぶwrapper。
- **使用場面**: historical Notebookの再現。
- **新規コード**: `sweep_conv2d_ranks`を使う。

---

# 11. `nn_compression.training`

## `train_one_epoch(model, train_loader, criterion, optimizer, device) -> (loss, accuracy)`  [A]

- **責務**: 1 epochの学習を行う。
- **引数**: model、train DataLoader、loss、optimizer、device。
- **戻り値**: sample-weighted平均lossとaccuracy。
- **使用場面**: baseline/圧縮modelの通常学習・Fine-tuning。
- **主要contract**: `model.train()`を設定。empty loaderはValueError。

## `evaluate(model, loader, criterion, device) -> (loss, accuracy)`  [A]

- **責務**: optimizer updateなしでdataset全体を評価する。
- **引数**: model、loader、criterion、device。
- **戻り値**: sample-weighted平均lossとaccuracy。
- **使用場面**: validation/test評価。
- **主要contract**: 一時的にevalにし、正常/例外時とも全submoduleの元training stateを復元。empty loader拒否。

## `fit_with_early_stopping(...) -> dict`  [B]

- **責務**: validation loss基準のEarly Stopping付き学習を行い、best stateへ戻す。
- **主な引数**: model、train/val loader、criterion、optimizer、device、`max_epochs`, `patience`, `min_delta`、train再評価設定。
- **戻り値**: `model`, `best_epoch`, `best_validation_loss`, loss histories, `history` を含むdict。
- **使用場面**: baseline学習・圧縮後Fine-tuning。
- **主要contract**: best `state_dict`をdeepcopyし、終了時に復元。

## `non_shuffling_loader(loader) -> DataLoader`  [B]

- **責務**: 同じDatasetをshuffleせず評価するDataLoaderを作る。
- **引数**: 元DataLoader。
- **戻り値**: `shuffle=False`のDataLoader。
- **使用場面**: 学習用Generatorを進めずtrain metricを再評価したいとき。
- **注意**: 任意custom sampler/batch_samplerの完全cloneではない。

---

# 12. `nn_compression.metrics` — model comparison

## `count_parameters(model) -> int`  [A]

- **責務**: modelが保持する全Parameter要素数を数える。
- **引数**: model。
- **戻り値**: parameter数。
- **使用場面**: baseline/compressed modelのサイズ比較。
- **注意**: `requires_grad=False`も含む。

## `parameters_reduction(baseline_model, compressed_model) -> float`  [A]

- **責務**: parameter削減率を計算する。
- **戻り値**: `1 - compressed/baseline`。
- **使用場面**: 圧縮効果のmodel-level評価。

## `accuracy_drop(baseline_accuracy, compressed_accuracy, verbose=True) -> float`  [A]

- **責務**: baselineからのaccuracy低下量を計算する。
- **戻り値**: `baseline - compressed`。
- **使用場面**: 圧縮によるtask性能変化の簡易指標。

## `agreement(baseline_model, compressed_model, loader, device) -> float`  [A]

- **責務**: 2モデルの予測class一致率を計算する。
- **戻り値**: `[0,1]`の一致率。
- **使用場面**: accuracyとは別にbaselineの判断をどれだけ保持したか測るとき。
- **主要contract**: loader全体で集計、empty拒否、両modelのtraining state復元。

## `logits_rmse(...) -> float`  [A]

- **責務**: baseline/compressedのlogits全要素RMSEを計算する。
- **戻り値**: scalar RMSE。
- **使用場面**: argmax一致だけでなく出力分布のずれを測るとき。

## `take_inference_batch(data_loader) -> Tensor`  [A]

- **責務**: latency比較用の固定input batchを取得する。
- **戻り値**: input Tensor。
- **使用場面**: baseline/compressed benchmarkへ同一入力を渡す前処理。
- **主要contract**: shuffle付き`RandomSampler`はGenerator消費を避けるため拒否。

## `benchmark_inference(...) -> float | dict`  [A]

- **責務**: 1 batchあたりの平均forward時間を測定する。
- **主な引数**: model、device、`warmup`, `repeats`、`input_batch`またはDataLoader、`return_details`。
- **戻り値**:
  - 既定: 秒単位float。
  - `return_details=True`: time/batch/input条件を含むdict。
- **使用場面**: 理論MACsとは別に実測latencyを比較するとき。
- **主要contract**: `warmup>=0`, `repeats>=1`。bool/Tensor scalar拒否。CUDAは同期して計測。training state復元。

## `benchmark_inference_print(...) -> (baseline_time_s, compressed_time_s)`  [A]

- **責務**: baseline/compressedを同じinput条件で連続benchmarkするconvenience API。
- **戻り値**: 2モデルの平均秒数tuple。
- **使用場面**: Notebook上で簡単に比較・表示したいとき。

## `collect_compression_metrics(...) -> dict`  [A]

- **責務**: compressed candidateの共通評価指標を1 recordへ集約する。
- **主な引数**: baseline/compressed model、loader、criterion、device、baseline accuracy/time、benchmark条件、optional MACs。
- **戻り値**: parameters, validation, latency, agreement, RMSE等を含むdict。
- **使用場面**: rank sweepや比較表の共通record生成。
- **注意**: 同じloaderを複数回走査するためone-shot iteratorは未対応。`include_model=False`が既定。

---

# 13. MACs

## `linear_macs(layer) -> int`  [A]

- **責務**: Linear 1 sampleの理論MACsを計算する。
- **戻り値**: `in_features * out_features`。
- **使用場面**: baseline Linear計算量。

## `compressed_linear_macs(in_features, out_features, rank) -> int`  [A]

- **責務**: `in→rank→out` factorized LinearのMACsを計算する。
- **戻り値**: `in*rank + rank*out`。
- **使用場面**: Linear SVD candidateの理論計算量比較。

## `conv2d_macs(conv, out_h, out_w) -> int`  [A]

- **責務**: Conv2d 1 sampleの理論MACsを計算する。
- **引数**: Conv layerと出力空間size。
- **戻り値**: 理論MACs。
- **使用場面**: baseline Conv計算量。
- **主要contract**: `out_h/out_w`はbool/Tensor scalar以外の正整数。grouped Convもweight shapeに基づき計算可能。

## `compressed_conv2d_macs(conv, rank, out_h, out_w) -> int`  [A]

- **責務**: SVD 2層Convの理論MACsを計算する。
- **使用場面**: Conv SVD rank比較。
- **主要contract**: compressed Convは`groups=1`のみ、rankは分解可能上限以内。

## `estimate_conv2d_macs(conv, rank, out_hw, verbose=True) -> (baseline, compressed, reduction)`  [A]

- **責務**: 1 Convについてbaseline/compressed MACsと削減率をまとめて返す。
- **使用場面**: Notebookやrank sweepで1層の計算量を比較するとき。

## `factorized_linear_macs` / `factorized_conv2d_macs`  [C]

- **責務**: 上記`compressed_*_macs`のcompatibility alias。
- **使用場面**: 旧Notebookのみ。

---

# 14. model-specific MACs  [B]

## `estimate_mlp_macs(model, fc1_rank, fc2_rank, verbose=True)`

- **責務**: 現行784→512→256→10 MLPのbaseline/compressed MACsを集計する。
- **戻り値**: `(baseline_macs, compressed_macs, reduction)`。
- **使用場面**: MNIST/Fashion-MNIST MLP historical実験。

## `estimate_cnn_linear_macs(model, fc1_rank, verbose=True)`

- **責務**: Fashion-MNIST CNNのLinear部でfc1を圧縮した場合のMACsを比較する。
- **使用場面**: CNN Linear SVD実験。

## `estimate_cnn_conv2_macs(model, rank, verbose=True, *, layer_name='conv2', out_hw=None)`

- **責務**: 指定Conv1層のbaseline/compressed MACsを比較する。
- **使用場面**: Fashion-MNISTのconv2、または任意named Convの1層評価。

## `estimate_cnn_macs(...)`

- **責務**: 指定したConv/Linear群を合計しmodel-level理論MACsを比較する。
- **主な引数**: `conv_ranks`, `linear_ranks`, `conv_output_hw`, `linear_layer_names`。位置引数はFashion-MNIST互換。
- **戻り値**: `(baseline_macs, compressed_macs, reduction)`。
- **使用場面**: 複数層SVD圧縮のmodel-wide MACs比較。

---

# 15. Tensor approximation

## `relative_frobenius_error(X, X_hat) -> Tensor`  [A]

- **責務**: `||X-X_hat||_F / ||X||_F` を計算する。
- **引数**: 同shapeの元Tensor/近似Tensor。
- **戻り値**: scalar Tensor。
- **使用場面**: SVD/Tucker/HOOIのweight再構成誤差評価。
- **主要contract**: shape一致必須、`X`がzero tensorならValueError、関数内ではdetachしない。

---

# 16. `nn_compression.selection`  [B]

## `is_pareto_candidate(df, row, x_column, y_column) -> bool`

- **責務**: 1候補が他候補に支配されていないか判定する。
- **使用場面**: 2目的rank候補のPareto判定。
- **前提**: x/yとも小さいほど良い。

## `extract_pareto_frontier(df, x_column, y_column) -> DataFrame`

- **責務**: 非支配候補だけを抽出しx順に返す。
- **使用場面**: parameters vs validation loss等の候補絞り込み。

## `get_first_point(df, column, x, y) -> (x, y)`

- **責務**: 指定columnでsortした先頭点を返す。
- **使用場面**: knee用端点処理。

## `get_endpoints(df, column, x, y) -> (left, right)`

- **責務**: sort後の先頭/末尾座標を返す。
- **使用場面**: knee基準直線の端点取得。

## `line_equation(p0, p1) -> (slope, intercept)`

- **責務**: 2点を通る直線表現を返す。
- **使用場面**: knee距離計算。
- **注意**: 垂直線は `(inf, x_constant)`。

## `find_knee_point(...) -> (point, distance)`

- **責務**: 端点直線から距離最大の候補をkneeとして返す。
- **使用場面**: Pareto frontierから代表rankを選ぶ補助。

## `find_knee_point_numpy(...) -> (point, distance)`

- **責務**: 上記knee計算のNumPy版。
- **使用場面**: 同じ意味でvectorized計算したいとき。

---

# 17. `nn_compression.datasets`  [B]

## `get_fashion_mnist_datasets(data_dir, *, transform=None, download=True)`

- **責務**: Fashion-MNIST train全体とtest Datasetを取得する。
- **戻り値**: `(full_train_dataset, test_dataset)`。
- **使用場面**: Fashion-MNIST実験のデータ取得。
- **注意**: transform未指定時は`ToTensor()`。

## `split_fashion_mnist_dataset(dataset, split_lengths, *, seed)`

- **責務**: Datasetをseed固定で再現可能に分割する。
- **戻り値**: `random_split`のSubset群。
- **使用場面**: train/validation/rank-validation分割。

## `make_fashion_mnist_loaders(...) -> dict`

- **責務**: 学習・validation・test・train-eval等のDataLoaderを統一方針で作る。
- **主な引数**: 各Dataset、batch size、optional rank-validation Dataset、train Generator。
- **戻り値**: loader名をkeyにしたdict。
- **使用場面**: Fashion-MNIST実験のloader準備。
- **主要contract**: trainだけshuffle=True。

## `shuffled_index_splits(n, lengths, *, seed) -> tuple[list[int], ...]`

- **責務**: `0..n-1`をseed固定でshuffleし、指定長ごとにindex listへ分ける。
- **使用場面**: train/evalでtransformの違うDatasetへ同じindex splitを適用するとき。
- **主要contract**: `sum(lengths) <= n`, 各length>=0。

---

# 18. `nn_compression.models`  [B]

## `MNISTMLP()`

- **責務**: MNIST/Fashion-MNIST共通の784→512→256→10 MLPを提供する。
- **戻り値**: `nn.Module` instance。
- **使用場面**: MLP SVD学習・圧縮実験。
- **固定layer名**: `fc1`, `fc2`, `fc3`。

## `FashionMNISTCNN()`

- **責務**: Fashion-MNIST用2 Conv + 2 Linear CNNを提供する。
- **使用場面**: CNN Linear/Conv SVD実験。
- **固定layer名**: `conv1`, `conv2`, `fc1`, `fc2`。

### `FashionMNISTCNN.inspect_shapes(x=None) -> int`

- **責務**: forward途中shapeを表示しFlatten後特徴数を返す。
- **使用場面**: 学習用shape確認、Linear `in_features`の検証。
- **戻り値**: flatten dimension。

## `CIFAR10CNN(*, in_channels=3, conv_channels=(32,64,128), hidden_dim=256, num_classes=10, dropout=0.5)`

- **責務**: CIFAR-10実験用Conv×3 + GAP + 2 Linear CNNを提供する。
- **使用場面**: CIFAR-10 SVD/Tucker比較。
- **固定layer名**: `conv1`, `conv2`, `conv3`, `fc1`, `fc2`。
- **注意**: default architectureは既存checkpoint/resultsと対応するため破壊的変更しない。

---

# 19. `nn_compression.utils`  [B]

## `get_named_module(model, name) -> nn.Module`

- **責務**: dot pathで既存submoduleを取得する。
- **使用場面**: `conv2`, `block.0`, `block1.conv`等をgenericに圧縮するとき。

## `set_named_module(model, name, module) -> None`

- **責務**: 既存named submoduleを新Moduleへ置換する。
- **使用場面**: named layer compression。
- **主要contract**: 存在しないpathを暗黙作成しない。

## `find_project_root(start_path) -> Path`

- **責務**: 上位directoryを辿り`.git`があるproject rootを探す。
- **使用場面**: Notebookのcwdに依存せずdata/models/results pathを解決するとき。
- **戻り値**: project root `Path`。

## `get_experiment_dirs(project_root, method_name, case_name, experiment_name, *, create=True) -> (data_dir, models_dir, results_dir)`

- **責務**: 実験成果物directoryを共通規則で生成/取得する。
- **使用場面**: SVD/Tucker/TT/MPS/DMRG Notebookの保存先統一。
- **構造**: `method / case / experiment`。

## `set_seed(seed=0) -> None`

- **責務**: Python/NumPy/PyTorch/CUDA等の乱数・deterministic設定をまとめて行う。
- **使用場面**: 実験開始時の再現性設定。
- **注意**: 既に存在するDataLoader専用Generatorの状態はresetしない。

## `make_torch_generator(seed=0) -> torch.Generator`

- **責務**: DataLoader shuffle/random_split用の専用Generatorを作る。
- **使用場面**: candidate間でmini-batch順やsplitを再現したいとき。

---

# 20. Compatibility API  [C]

以下はhistorical Notebook互換のため残す。

| 旧API | Primary API | 使用方針 |
|---|---|---|
| `SVD` | `truncated_svd` | 新規コードではPrimaryを使う |
| `RebuildSVD` | `rebuild_linear_from_svd` | 同上 |
| `factorize_Conv2d_layer` | `factorize_conv2d_layer` | 同上 |
| `factorized_linear_macs` | `compressed_linear_macs` | 同上 |
| `factorized_conv2d_macs` | `compressed_conv2d_macs` | 同上 |
| `sweep_conv_svd_ranks` | `sweep_conv2d_ranks` | historical Notebook向け |
| `r1/r2` keyword | `fc1_rank/fc2_rank` | historical MLP Notebook向け |

Compatibility APIへ新しい機能は追加しない。

---

# 21. Breaking change

次はCore API v1のbreaking changeとして扱う。

```text
Public API名の削除・改名
引数名やkeyword-only境界の変更
defaultの意味変更
return tuple/dict structureの非互換変更
shape convention変更
baseline非破壊性の喪失
device/dtype/requires_grad semantics変更
HOOI history[0]等の意味変更
評価後のtraining state復元保証の削除
```

TT/MPS・DMRG追加では、これらを変更するより新しいAPIを追加する。
