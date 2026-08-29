# 関数別API仕様 セルフレビュー

## 1. 目的

Core API v1を関数・クラス単位の仕様へ分割し、さらに各APIの処理説明を日本語で詳細化した後、次を照合した。

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

追加で、`__all__`そのものではないが利用者が直接呼ぶpublic methodとして、

```text
FashionMNISTCNN.inspect_shapes()
```

も個別仕様化した。

PyTorch標準の`forward()`とprivate helperは個別ファイル化せず、class仕様または関連Public APIの処理説明内で扱う。

## 4. 各ファイルの必須項目

全API仕様は原則として、

```text
責務
Signature
引数
戻り値
使用場面
処理の流れ（日本語）
処理フロー（短縮版。必要な場合）
主なcontract / 注意事項
関連API
```

を持つ構成へ統一した。

## 5. 処理説明のレビュー方針

以前の`ざっくりした処理`は、例えば

```text
validation → SVD → return
```

のように短く、コードを開かないと中間処理や責務境界を判断しにくかった。

今回、次の基準へ変更した。

1. **処理順を番号付きの日本語で説明する。**
2. **何をするかだけでなく、重要な箇所はなぜ行うかも書く。**
3. **Tensor/shapeが途中でどう変化するかを必要に応じて書く。**
4. **下位APIへ委譲するwrapperは、自身の処理と委譲先の責務を分ける。**
5. **Module builderはweight/factor配置、bias、device/dtype、requires_grad、autograd境界を説明する。**
6. **学習・評価系はmodel state、DataLoader走査、empty loader、RNG影響を説明する。**
7. **短縮版フローは補助として残すが、日本語本文の代わりにはしない。**

## 6. 代表APIの再照合

現行srcと特に次を再照合した。

- `unfold / fold / mode_dot`: validation、axis移動、reshape、逆変換
- `truncated_svd`: rank validation、economy SVD、slice
- `factorize_linear_layer / factorize_conv2d_layer`: factor配置、bias、device/dtype/requires_grad
- `hosvd`: factorは逐次coreではなく元`X`から求める
- `hooi_sweep`: factor clone、挿入順、最新factorを使うGauss-Seidel型更新
- `hooi`: HOSVD初期化、`max_iter=0`でもfeasibility検証、`history[0]`、収束判定
- Tucker-2 APIs: channel mode対応、3層構造、Tensor-levelとModule-levelのautograd境界
- `evaluate / benchmark_inference`: 全submodule状態復元、CUDA同期、same input
- `collect_compression_metrics`: loaderを複数回走査することとone-shot iterator制約
- MACs helpers: 実際に生成するfactorized layer構造と式の対応
- selection: Pareto支配条件、knee直線・距離計算
- datasets: split RNGとtraining shuffle RNGの分離
- utils: named moduleのstrict置換、project root、seed/Generatorの副作用

上記について、現行実装とのCritical / Majorな不一致は見つからなかった。

## 7. 設計本文との分離

```text
01〜04
→ architecture / module responsibility / flow / common design policy

05_Core_API_v1/
→ function/class-level API specification

06〜09
→ tests / known constraints / review records
```

とし、「なぜこのパッケージ構造なのか」と「個々の関数が何をするか」を混ぜない。

## 8. 今回の詳細化でのsrc変更

今回の**処理説明の日本語詳細化ではsrc/tests/Notebook/resultsを変更していない**。

PR内に既に存在するCore API v1 freeze前のvalidation修正は、それ以前のself reviewで追加した別変更である。

## 9. 結論

関数別API仕様は、関数名から直接、

```text
何のための関数か
何を渡すか
何が返るか
内部で何をどの順番で行うか
どの処理を下位APIへ任せるか
何を壊してはいけないか
```

を日本語で追える構成になった。

今後Public APIを追加する場合は、

```text
src実装
→ __all__
→ contract test
→ 対応する関数別md
→ 必要なら上位設計書
```

を同時に更新する。
