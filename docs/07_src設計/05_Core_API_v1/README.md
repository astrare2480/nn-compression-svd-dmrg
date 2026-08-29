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
フローチャート / 構造図（必要な場合のみ）
主なcontract / 注意事項
関連API
```

## 処理説明の記述ルール

- 関数名・変数名・PyTorch API名はコード上のidentifierをそのまま残す。
- ただし処理内容は英語identifierの羅列だけにせず、**各段階で何をしているかを日本語で説明する**。
- 処理順は現行`src`の実装順・意味と一致させる。
- アルゴリズム系APIは、validation・中間表現・更新・再構成・収束判定などの段階を分けて書く。
- Module builderは、Parameterをどこへ配置するか、bias/device/dtype/requires_grad/autogradをどう扱うかを書く。
- wrapper / convenience APIは、自身が持つ処理と、どの下位APIへ委譲しているかを分けて書く。
- 数行で終わる単純utilityでも、入力解釈・内部操作・副作用・戻り値が分かる粒度を保つ。

## Mermaid図を入れる基準

図は**必要なAPIだけ**に追加する。以下のいずれかに当てはまり、文章だけより処理構造が理解しやすくなる場合を対象とする。

1. 反復処理がある。
2. 条件分岐が処理結果や状態遷移に影響する。
3. 複数の下位APIをまたいで状態や中間表現が変化する。
4. 1つのModuleを複数Moduleへ変換するなど、構造変換を視覚化する価値がある。

逆に、単純な数式計算、1回の委譲だけのwrapper、短い一本道utilityにはMermaid図を付けない。

現在、図を付ける代表APIは次のとおり。

```text
factorize_conv2d_layer
factorize_named_layers
sweep_layer_ranks
hosvd
hooi_sweep
hooi
build_tucker2_conv_from_components
fit_with_early_stopping
benchmark_inference
collect_compression_metrics
```

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
→ 必要なら設計本文
```

を同時に更新する。
