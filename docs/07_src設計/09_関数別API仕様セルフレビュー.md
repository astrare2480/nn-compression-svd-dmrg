# 関数別API仕様 セルフレビュー

## 1. 目的

Core API v1をモノリシックな一覧から関数・クラス単位の仕様へ分割した後、次を照合した。

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

1件。

`FashionMNISTCNN.inspect_shapes()` は引数省略時にCPU Tensorを作るため、modelをGPUへ移動した状態ではdummy inputとdeviceが一致しない。この利用条件を個別仕様へ追記した。

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

各sub-packageの`__all__`にあるprimary/experiment-support APIを個別ファイル化した。

単純aliasは重複説明を避け、`05_Core_API_v1/Compatibility_API.md`へ集約した。

追加で、`__all__`そのものではないが利用者が直接呼ぶpublic methodとして、

```text
FashionMNISTCNN.inspect_shapes()
```

も個別仕様化した。

PyTorch標準の`forward()`とprivate helperは個別化していない。

## 4. 各ファイルの必須項目

全API仕様は原則として、

```text
責務
Signature
引数
戻り値
使用場面
ざっくりした処理
主なcontract / 注意事項
関連API
```

を持つ構成へ統一した。

## 5. 設計本文との分離

```text
01〜04
→ architecture / module responsibility / flow / common design policy

05_Core_API_v1/
→ function/class-level API specification

06〜09
→ tests / known constraints / review records
```

とし、「なぜこの構造か」と「この関数をどう使うか」を同じファイルへ混ぜない。

## 6. src変更

今回の**関数別ファイル化そのものではsrcを変更していない**。

PR内に既に存在するCore API v1 freeze前のvalidation修正は別のself reviewで追加したものであり、本再編はdocumentation構造の変更のみ。

## 7. 結論

関数別API仕様への再編により、利用者は関数名から直接、引数・戻り値・使用場面・処理概要・contractを確認できる状態になった。

今後Public APIを追加する場合は、`__all__`・contract test・対応する関数別mdを同時に更新する。
