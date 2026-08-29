# 関数別API仕様 セルフレビュー

## 1. 目的

Core API v1の関数・クラス別仕様について、今回あらためて**全77仕様**を横断確認した。

確認対象は次の対応関係。

```text
各sub-packageの __all__ / public method
↕
現行srcのsignature / behavior
↕
05_Core_API_v1/ の関数・クラス別md
```

今回の重点は、

1. `引数`が型・shape・identifierの列挙だけでなく、**その値の日本語での意味と役割**を説明しているか。
2. `戻り値`がtuple / dict / scalar / Moduleの**中身と読み方**を説明しているか。
3. Mermaid図が単なる文章の描き直しではなく、**制御・責務分担・構造**のいずれかを追加で理解できるか。

の3点。

`Compatibility_API.md`はalias対応表のため77件の個別仕様数には含めず、互換名の集約資料として別扱いする。

## 2. レビュー結果

### Critical

なし。

### Major

なし。

### Documentation Minor — 修正済み

今回、次の傾向を修正した。

- `2モデル、評価loader、device`のように異なる役割の引数を一文へまとめていた箇所。
- `shape`や`(R_out, R_in, ...)`だけで、Tensorが**何を表すか**を書いていなかった箇所。
- tuple / dictを返すAPIで、各要素・主要keyの意味が弱かった箇所。
- model classの戻り値が単に`nn.Module instance`となっていた箇所。
- `factorize_named_layers()`のフローチャートを簡略化しすぎ、番号付き`処理概要`の縦並びに近くなっていた箇所。

`factorize_named_layers()`は、Conv/Linearの反復・型判定・`TypeError`・分解API・copy側置換を別工程として追える粒度へ戻した。

## 3. Public API coverage

個別仕様の内訳は次のとおり。

```text
tensor        3
compression  30
training      4
metrics      19
selection     7
datasets      4
models        4
utils         6
----------------
合計         77
```

READMEのリンク一覧と実ファイルinventoryを正本として全77件を確認した。

単純aliasは重複説明を避け、`05_Core_API_v1/Compatibility_API.md`へ集約する。

`FashionMNISTCNN.inspect_shapes()`は`__all__`そのものではないが、利用者が直接呼ぶpublic methodとして個別仕様化している。

PyTorch標準`forward()`とprivate helperは個別ファイル化せず、class仕様または関連Public APIの処理説明内で扱う。

## 4. 引数・戻り値レビュー

全77仕様について`## 引数`と`## 戻り値`を再確認した。

### 修正した代表例

- `truncated_svd`: `matrix / rank`が何を決めるか、`U_r / S_r / Vh_r`各成分の意味。
- `rebuild_linear_from_svd`: 3つのSVD componentと元`layer`の役割。
- Tucker系: `shape / ranks / core / factors / rank_out / rank_in`がどのmode・channelを表すか。
- `build_tucker2_conv_from_components`: `core / u_out / u_in`が構築後のどのConv weightへ入るか。
- rank sweep系: baseline値、callback、MACs callback、fixed input batchの意味。
- metrics系: baseline/compressed model、loader、device、benchmark条件、optional MACsの役割。
- Dataset系: split用seedとtrain shuffle用Generatorの責務分離、loader dict各keyの用途。
- selection系: x/y列、基準直線、endpoint、knee戻り値の意味。
- model class: constructorが返すbaseline architectureと安定named layer。
- `get_experiment_dirs`: `method / case / experiment`各階層と3つの戻りdirectoryの意味。

### 無変更とした仕様

すでに、

- 引数の役割が日本語で個別に書かれている。
- 戻り値のshapeだけでなく意味も本文・戻り値節から一意に分かる。
- 短いscalar utilityでも式と値の読み方が十分明確である。

ものは、文章量だけを増やす変更を行っていない。

今回、77件中42件を実際に更新し、残り35件は現状の説明で十分と判断した。

## 5. 図の種類と要否判断

Mermaidは次の3種類に分けて判断した。

### フローチャート

反復・分岐・異常終了・収束、または複数段階のbuild順など**制御フロー**を示す。

### シーケンス図

wrapper / orchestratorが複数API・objectへ処理を委譲する場合に、**誰が誰を呼び、何を受け渡すか**を示す。

### 構造図 / Component配置図

Tensor factor・Parameter・Moduleがどこへ配置されるか、model architectureがどう接続されるかという**構造**を示す。

複数図を同じファイルへ入れる場合は、各図が別の問いに答えることを条件とした。

## 6. 最終的に図を使う代表仕様

```text
factorize_conv2d_layer
  フローチャート

factorize_named_layers
  フローチャート + シーケンス図 + コンポーネント図

sweep_layer_ranks
  フローチャート + シーケンス図

hosvd / hooi_sweep / hooi
  フローチャート

build_tucker2_conv
  シーケンス図

build_tucker2_conv_from_components
  フローチャート + Component配置図

fit_with_early_stopping
  フローチャート

benchmark_inference
  フローチャート

collect_compression_metrics
  フローチャート + シーケンス図

MNISTMLP / FashionMNISTCNN / CIFAR10CNN
  構造図
```

一方、`count_parameters`, `has_converged`, `linear_macs`, `get_named_module`, Pareto/kneeの小helperなどは、図を追加しても文章・式の描き直しになるため追加していない。

## 7. 図の重複セルフレビュー

### `factorize_named_layers`

3図すべてを残す価値がある。

- フローチャート: Conv/Linear loop、型分岐、異常終了、処理順。
- シーケンス図: baselineを分解元、copyを置換先とする責務分担。
- コンポーネント図: model内の指定層だけがfactorized Moduleへ変わる構造。

フローチャート内の`factorize_*`とcopy側置換は別工程として残した。ここを1箱へまとめると、シーケンス図との重複軽減よりも処理理解の損失が大きいと判断した。

### `build_tucker2_conv_from_components`

- フローチャート: component検証 → Module生成 → Parameter配置 → returnの組み立て順。
- Component配置図: `u_in / core / u_out / bias / spatial config`が3層のどこへ反映されるか。

同じ3層を二度描くことが目的ではなく、**処理順と配置関係**を分離している。

### `sweep_layer_ranks` / `collect_compression_metrics`

フローチャートはloop・optional分岐、シーケンス図はcallback・評価APIへの委譲を担当し、役割を分けた。

## 8. 今回の変更範囲

今回の修正は`docs/07_src設計/`配下のMarkdownのみ。

- 個別API仕様77件をレビュー。
- そのうち説明不足・図の改善余地があった42件を更新。
- `05_Core_API_v1/README.md`で記述ルールを明文化。
- 本セルフレビューを更新。

`src / tests / Notebook / results`は変更しない。

## 9. テスト・実装影響

今回はdocumentation-only変更なので、src挙動・実験値・API signatureは変更していない。

このレビューに対してpytestは新規実行していない。PR内の以前のsrc validation差分については別途最終merge前のローカルpytest確認が必要であり、今回のdocumentation reviewをもって「現PR全体のtests green」とはみなさない。

## 10. 結論

全77の関数・クラス別仕様について、利用者がファイル単体から、

```text
何のためのAPIか
各引数は何を意味するか
戻り値の各要素は何を表すか
内部で何をどの順番で行うか
どこで分岐・反復するか
どの処理を下位APIへ任せるか
Tensor / Parameter / Moduleがどう配置されるか
何を壊してはいけないか
```

を追えることを確認した。

今後Public APIを追加・変更する場合は、

```text
src実装
→ __all__
→ contract test
→ 対応する関数別md
→ 引数 / 戻り値の日本語意味を確認
→ 図の要否・図種・重複を確認
→ 必要なら上位設計書
```

を同時に更新する。
