# Core API v1 — 関数別仕様

## 目的

`src/nn_compression/` のPublic APIを、**1関数または1クラスにつき1ファイル**で確認できるようにする。

各ファイルは原則として次を共通項目として持つ。

```text
責務
Signature
引数
戻り値
使用場面
処理概要
フローチャート / シーケンス図 / 構造図（必要な場合のみ）
主なcontract / 注意事項
関連API
```

## 引数・戻り値の記述ルール

- identifier・型・shapeはコードと対応させるが、**型やshapeだけで終わらせず、その値が何を意味し、処理のどこで使われるかを日本語で説明する**。
- 複数引数を「model、loader、device」のように一文でまとめず、役割が異なる場合は原則として引数ごとに説明する。
- rankは「整数」とだけ書かず、どのdimension / channel / modeを何次元へ縮約する値かを書く。
- Tensor componentはshapeに加え、core / factor / weightなど**何を表し、どの演算・Moduleへ対応するか**を書く。
- callback・flag・tolerance・seedは、値そのものより**何の挙動を制御するか**を書く。
- 戻り値がtuple / dictの場合は、各要素・主要keyの意味を個別に説明する。
- scalar指標は式だけで終わらせず、値が大きい/小さい/正/負のとき何を意味するかを必要に応じて書く。
- class constructorの戻り値は単に`nn.Module instance`とせず、どの入力を何へ変換するbaseline modelか、安定して参照するnamed layerは何かを書く。
- 引数が無いAPIは`なし`と明記し、固定architectureやdefault semanticsがある場合はその意味を補足する。

## 処理説明の記述ルール

- 関数名・変数名・PyTorch API名はコード上のidentifierをそのまま残す。
- ただし処理内容は英語identifierの羅列だけにせず、**各段階で何をしているかを日本語で説明する**。
- 処理順は現行`src`の実装順・意味と一致させる。
- アルゴリズム系APIは、validation・中間表現・更新・再構成・収束判定などの段階を分けて書く。
- Module builderは、Parameterをどこへ配置するか、bias/device/dtype/requires_grad/autogradをどう扱うかを書く。
- wrapper / convenience APIは、自身が持つ処理と、どの下位APIへ委譲しているかを分けて書く。
- 数行で終わる単純utilityでも、入力解釈・内部操作・副作用・戻り値が分かる粒度を保つ。

## Mermaid図を入れる基準

図は**必要なAPIだけ**に追加する。図を入れる目的を先に決め、同じ情報を別形式で重複させない。

### フローチャート

**処理の制御**を理解するために使う。

- 反復処理がある。
- 条件分岐・異常終了・収束判定が重要である。
- 複数段階のbuild処理など、処理順そのものを視覚化する価値がある。

単純な一本道utilityを、番号付き`処理概要`と同じ内容の箱へ置き換えるだけなら追加しない。

### シーケンス図

**API / object間の呼び出しと責務分担**を理解するために使う。

- wrapperが複数の下位APIへ順に委譲する。
- baseline / candidate / builderなど複数object間でデータの受け渡しがある。
- 「誰が分解し、誰が評価し、誰が置換するか」が設計上重要である。

### 構造図 / Component配置図

**Tensor / Parameter / Moduleの構造や配置**を理解するために使う。

- 1つのModuleを複数Moduleへ変換する。
- core / factor / biasなどが構築後のどの層へ入るかが重要である。
- baseline model architectureやfeature dimensionの変化を視覚化する価値がある。

### 複数図を同じファイルへ入れる場合

複数図は、それぞれが別の問いに答える場合だけ併記する。

```text
フローチャート   → どの順番・分岐で処理するか
シーケンス図     → 誰が誰を呼び、何を受け渡すか
構造図           → 何がどこに配置され、構造がどう変わるか
```

原則1図、必要なら2図、3図は役割が明確に分かれる場合だけとする。

現在、図を付ける代表API / classは次のとおり。

| API / class | 図 | 主な目的 |
|---|---|---|
| `factorize_conv2d_layer` | フローチャート | SVD成分から2層Convを作る流れ |
| `factorize_named_layers` | フローチャート + シーケンス図 + コンポーネント図 | 制御・責務分担・model置換構造 |
| `sweep_layer_ranks` | フローチャート + シーケンス図 | rank反復とcallback/metrics委譲 |
| `hosvd` | フローチャート | modeごとのfactor計算 |
| `hooi_sweep` | フローチャート | factor更新順 |
| `hooi` | フローチャート | 反復・収束判定 |
| `build_tucker2_conv` | シーケンス図 | Tensor分解とModule builderの責務分離 |
| `build_tucker2_conv_from_components` | フローチャート + Component配置図 | build順とcomponent配置 |
| `tt_svd_exact` | フローチャート | sequential TT-SVDと零rank特殊処理 |
| `tt_svd` | フローチャート | max_rank検証とrank制限付きsequential sweep |
| `fit_with_early_stopping` | フローチャート | epoch反復・best state・停止条件 |
| `benchmark_inference` | フローチャート | fixed input・同期・計測分岐 |
| `collect_compression_metrics` | フローチャート + シーケンス図 | optional分岐と評価API委譲 |
| `MNISTMLP` | 構造図 | layerとfeature dimension |
| `FashionMNISTCNN` | 構造図 | Conv/Pool/Linear構造 |
| `CIFAR10CNN` | 構造図 | Conv/GAP/Linear構造 |

逆に、単純な数式計算、1回の委譲だけのwrapper、短い一本道utilityにはMermaid図を付けない。

### Stability

- **A — Primary Stable API**: TT/MPS以降からも再利用する主要API。
- **B — Experiment Support Stable API**: 既存実験再現のため維持する支援API。
- **C — Compatibility API**: historical Notebook互換用。

---

## tensor

- [[07_src設計/05_Core_API_v1/tensor/unfold]]
- [[07_src設計/05_Core_API_v1/tensor/fold]]
- [[07_src設計/05_Core_API_v1/tensor/mode_dot]]

## compression — SVD / model replacement

- [[07_src設計/05_Core_API_v1/compression/truncated_svd]]
- [[07_src設計/05_Core_API_v1/compression/retained_energy_from_matrix]]
- [[07_src設計/05_Core_API_v1/compression/retained_energy]]
- [[07_src設計/05_Core_API_v1/compression/rebuild_linear_from_svd]]
- [[07_src設計/05_Core_API_v1/compression/factorize_linear_layer]]
- [[07_src設計/05_Core_API_v1/compression/factorize_named_linear]]
- [[07_src設計/05_Core_API_v1/compression/conv2d_weight_matrix]]
- [[07_src設計/05_Core_API_v1/compression/factorize_conv2d_layer]]
- [[07_src設計/05_Core_API_v1/compression/factorize_named_conv2d]]
- [[07_src設計/05_Core_API_v1/compression/factorize_named_layers]]
- [[07_src設計/05_Core_API_v1/compression/make_one_layer_svd_model]]
- [[07_src設計/05_Core_API_v1/compression/make_two_layer_svd_model]]

## compression — Tucker / HOOI / Tucker-2

- [[07_src設計/05_Core_API_v1/compression/hosvd]]
- [[07_src設計/05_Core_API_v1/compression/reconstruct_tucker]]
- [[07_src設計/05_Core_API_v1/compression/tucker_parameter_count]]
- [[07_src設計/05_Core_API_v1/compression/parameter_ratio]]
- [[07_src設計/05_Core_API_v1/compression/compression_factor]]
- [[07_src設計/05_Core_API_v1/compression/has_converged]]
- [[07_src設計/05_Core_API_v1/compression/hooi_sweep]]
- [[07_src設計/05_Core_API_v1/compression/core_from_factors]]
- [[07_src設計/05_Core_API_v1/compression/hooi]]
- [[07_src設計/05_Core_API_v1/compression/tucker2_decompose_conv_weight]]
- [[07_src設計/05_Core_API_v1/compression/build_tucker2_conv_from_components]]
- [[07_src設計/05_Core_API_v1/compression/build_tucker2_conv]]
- [[07_src設計/05_Core_API_v1/compression/tucker2_hooi_sweep]]
- [[07_src設計/05_Core_API_v1/compression/tucker2_hooi]]
- [[07_src設計/05_Core_API_v1/compression/tucker2_effective_weight]]
- [[07_src設計/05_Core_API_v1/compression/sweep_layer_ranks]]
- [[07_src設計/05_Core_API_v1/compression/sweep_conv2d_ranks]]
- [[07_src設計/05_Core_API_v1/compression/sweep_conv_svd_ranks]]

## compression — TT / MPS

- [[07_src設計/05_Core_API_v1/compression/tt_unfold]]
- [[07_src設計/05_Core_API_v1/compression/tt_svd_exact]]
- [[07_src設計/05_Core_API_v1/compression/tt_svd]]
- [[07_src設計/05_Core_API_v1/compression/tt_reconstruct]]
- [[07_src設計/05_Core_API_v1/compression/tt_num_parameters]]

## training

- [[07_src設計/05_Core_API_v1/training/train_one_epoch]]
- [[07_src設計/05_Core_API_v1/training/evaluate]]
- [[07_src設計/05_Core_API_v1/training/fit_with_early_stopping]]
- [[07_src設計/05_Core_API_v1/training/non_shuffling_loader]]

## metrics — model comparison

- [[07_src設計/05_Core_API_v1/metrics/count_parameters]]
- [[07_src設計/05_Core_API_v1/metrics/parameters_reduction]]
- [[07_src設計/05_Core_API_v1/metrics/accuracy_drop]]
- [[07_src設計/05_Core_API_v1/metrics/agreement]]
- [[07_src設計/05_Core_API_v1/metrics/logits_rmse]]
- [[07_src設計/05_Core_API_v1/metrics/take_inference_batch]]
- [[07_src設計/05_Core_API_v1/metrics/benchmark_inference]]
- [[07_src設計/05_Core_API_v1/metrics/benchmark_inference_print]]
- [[07_src設計/05_Core_API_v1/metrics/collect_compression_metrics]]

## metrics — MACs / Tensor error

- [[07_src設計/05_Core_API_v1/metrics/linear_macs]]
- [[07_src設計/05_Core_API_v1/metrics/compressed_linear_macs]]
- [[07_src設計/05_Core_API_v1/metrics/conv2d_macs]]
- [[07_src設計/05_Core_API_v1/metrics/compressed_conv2d_macs]]
- [[07_src設計/05_Core_API_v1/metrics/estimate_conv2d_macs]]
- [[07_src設計/05_Core_API_v1/metrics/estimate_mlp_macs]]
- [[07_src設計/05_Core_API_v1/metrics/estimate_cnn_linear_macs]]
- [[07_src設計/05_Core_API_v1/metrics/estimate_cnn_conv2_macs]]
- [[07_src設計/05_Core_API_v1/metrics/estimate_cnn_macs]]
- [[07_src設計/05_Core_API_v1/metrics/relative_frobenius_error]]

## selection

- [[07_src設計/05_Core_API_v1/selection/is_pareto_candidate]]
- [[07_src設計/05_Core_API_v1/selection/extract_pareto_frontier]]
- [[07_src設計/05_Core_API_v1/selection/get_first_point]]
- [[07_src設計/05_Core_API_v1/selection/get_endpoints]]
- [[07_src設計/05_Core_API_v1/selection/line_equation]]
- [[07_src設計/05_Core_API_v1/selection/find_knee_point]]
- [[07_src設計/05_Core_API_v1/selection/find_knee_point_numpy]]

## datasets

- [[07_src設計/05_Core_API_v1/datasets/get_fashion_mnist_datasets]]
- [[07_src設計/05_Core_API_v1/datasets/split_fashion_mnist_dataset]]
- [[07_src設計/05_Core_API_v1/datasets/make_fashion_mnist_loaders]]
- [[07_src設計/05_Core_API_v1/datasets/shuffled_index_splits]]

## models

- [[07_src設計/05_Core_API_v1/models/MNISTMLP]]
- [[07_src設計/05_Core_API_v1/models/FashionMNISTCNN]]
- [[07_src設計/05_Core_API_v1/models/FashionMNISTCNN.inspect_shapes]]
- [[07_src設計/05_Core_API_v1/models/CIFAR10CNN]]

`forward()`はPyTorch標準の呼び出し境界なので個別ファイルへ分離せず、各class仕様の処理説明で扱う。

## utils

- [[07_src設計/05_Core_API_v1/utils/get_named_module]]
- [[07_src設計/05_Core_API_v1/utils/set_named_module]]
- [[07_src設計/05_Core_API_v1/utils/find_project_root]]
- [[07_src設計/05_Core_API_v1/utils/get_experiment_dirs]]
- [[07_src設計/05_Core_API_v1/utils/set_seed]]
- [[07_src設計/05_Core_API_v1/utils/make_torch_generator]]

## Compatibility

- [[07_src設計/05_Core_API_v1/Compatibility_API]]

---

## 更新ルール

Public APIを追加・変更するときは、

```text
src実装
→ __all__
→ contract test
→ 対応する関数別md
→ 引数 / 戻り値の意味を日本語で確認
→ 図が本当に必要か・図種が適切か判断
→ 必要なら設計本文
```

を同時に更新する。