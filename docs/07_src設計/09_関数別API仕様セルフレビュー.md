# 関数別API仕様 セルフレビュー

## 1. 目的

Core API v1を関数・クラス単位の仕様へ分割し、各APIの処理説明を日本語で詳細化したうえで、必要なAPIにだけMermaid図を追加した。次を照合した。

```text
各sub-packageの __all__
↕
現行srcのsignature / docstring / behavior
↕
05_Core_API_v1/ の関数別md
```

## 2. レビュー結果

### Critical

なし。

### Major

なし。

### Documentation Minor

`FashionMNISTCNN.inspect_shapes()`は引数省略時にCPU Tensorを作るため、modelをGPUへ移動した状態ではdummy inputとdeviceが一致しない。この利用条件を個別仕様へ明記した。

## 3. Public API coverage

関数別仕様の対象：

```text
tensor
compression
training
metrics
selection
datasets
models
utils
```

各sub-packageの`__all__`にあるprimary / experiment-support APIを個別ファイル化した。

単純aliasは重複説明を避け、`05_Core_API_v1/Compatibility_API.md`へ集約した。

追加で、`__all__`そのものではないが利用者が直接呼ぶpublic methodとして、`FashionMNISTCNN.inspect_shapes()`も個別仕様化した。

PyTorch標準の`forward()`とprivate helperは個別ファイル化せず、class仕様または関連Public APIの処理説明内で扱う。

## 4. 各ファイルの基本項目

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

`処理概要`は番号付き日本語を基本とし、英語identifierだけの短いフローを本文の代わりにしない。

## 5. Mermaid図の要否判断

全APIへ機械的に図を付けると、単純関数まで大きくなり可読性が落ちる。そのため次の場合だけ図を付ける。

1. 反復がある。
2. 条件分岐が重要である。
3. 複数の下位APIをまたいで状態・中間表現が変わる。
4. Module構造変換を視覚化する価値がある。

今回、図を追加したAPIは次の10個。

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

一方、`count_parameters`, `has_converged`, `linear_macs`, `get_named_module`のような短い計算・判定・委譲utilityには図を追加していない。

## 6. 図と文章の重複整理

以前は、

```text
処理の流れ（日本語）
処理フロー（短縮版）
```

を併記しており意味が重複していた。

図を入れるAPIではこれを、

```text
処理概要
→ 詳細な日本語説明

フローチャート / 構造図
→ 視覚的な補助
```

へ整理した。Mermaidは文章の代替ではなく、反復・分岐・構造変換を一目で確認するための補助とする。

## 7. 代表APIの再照合

現行srcと特に次を再照合した。

- `factorize_conv2d_layer`: 2層構造、factor配置、bias、device/dtype/requires_grad
- `factorize_named_layers`: baseline deepcopy、元baseline layerを分解元にすること
- `sweep_layer_ranks`: rank loop、candidate独立性、optional MACs、metrics収集
- `hosvd`: factorは逐次coreではなく元`X`から求める
- `hooi_sweep`: factor clone、挿入順、最新factorを使うGauss-Seidel型更新
- `hooi`: HOSVD初期化、`max_iter=0`、`history[0]`、収束分岐
- `build_tucker2_conv_from_components`: 3層構造とfactor/core配置
- `fit_with_early_stopping`: train再評価、best state、patience分岐
- `benchmark_inference`: fixed input、CUDA同期、return形式
- `collect_compression_metrics`: loader複数走査、optional model/MACs

図の矢印・分岐は現行srcの処理意味と一致することを確認した。

## 8. 今回の変更範囲

今回のMermaid図追加と見出し整理では、`docs/07_src設計/`配下のMarkdownだけを変更する。`src/tests/Notebook/results`は変更しない。

PR内に既に存在するCore API v1 freeze前のvalidation修正は、それ以前のself reviewで追加した別変更である。

## 9. 結論

関数別API仕様は、関数名から直接、

```text
何のための関数か
何を渡すか
何が返るか
内部で何をどの順番で行うか
どこで分岐・反復するか
どの処理を下位APIへ任せるか
何を壊してはいけないか
```

を追える構成になった。

今後Public APIを追加する場合は、

```text
src実装
→ __all__
→ contract test
→ 対応する関数別md
→ 図が本当に必要か判断
→ 必要なら上位設計書
```

を同時に更新する。
