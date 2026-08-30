# 関数別API仕様 セルフレビュー

## 1. 目的

Core API v1の関数・クラス別仕様について、**全77仕様**を横断確認した。

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

さらにmainへCore API v1をmergeした後、図についてもう一段厳しく再レビューし、**処理段階が多いだけの一本道をフローチャートにしない**という基準へ修正した。

`Compatibility_API.md`はalias対応表のため77件の個別仕様数には含めず、互換名の集約資料として別扱いする。

## 2. レビュー結果

### Critical

なし。

### Major

なし。

### Documentation Minor — 修正済み

これまでのレビューで次を修正した。

- `2モデル、評価loader、device`のように異なる役割の引数を一文へまとめていた箇所。
- `shape`や`(R_out, R_in, ...)`だけで、Tensorが**何を表すか**を書いていなかった箇所。
- tuple / dictを返すAPIで、各要素・主要keyの意味が弱かった箇所。
- model classの戻り値が単に`nn.Module instance`となっていた箇所。
- `factorize_named_layers()`のフローチャートを簡略化しすぎ、番号付き`処理概要`の縦並びに近くなっていた箇所。
- `build_tucker2_conv_from_components()`で、一本道のbuilder処理をフローチャート化していた箇所。
- `factorize_conv2d_layer()`の図を制御フローとして分類していた箇所。

`factorize_named_layers()`は、Conv/Linearの反復・型判定・`TypeError`・分解API・copy側置換を別工程として追える粒度を維持する。

`build_tucker2_conv_from_components()`はフローチャートを削除し、component配置図だけを残す。

`factorize_conv2d_layer()`は図自体は有用だが、意味は制御フローではなく**SVD成分から2層Conv weightへの分解・配置関係**なので、分解・配置図として分類し直す。

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

すでに意味が十分明確な仕様には、文章量だけを増やす変更は行っていない。

## 5. 図の種類と要否判断

Mermaidは次の3分類で判断する。

### フローチャート

**実際の制御構造**を示す。

対象となるのは、

- ループと戻り先
- 条件分岐
- 異常終了 / 早期終了
- 収束判定 / 停止条件

など。

処理段階が多いだけでは追加理由にしない。

```text
検証 → 生成 → copy → return
```

のような一本道は、番号付き`処理概要`の方が簡潔なのでフローチャートを置かない。

### シーケンス図

wrapper / orchestratorが複数API・objectへ処理を委譲する場合に、**誰が誰を呼び、何を受け渡すか**を示す。

1つの下位APIを1回呼ぶだけの薄いwrapperには付けない。

### 構造図 / Component配置図 / データフロー図

Tensor factor・Parameter・Moduleがどこへ配置されるか、model architectureがどう接続されるか、入力表現がどの構造へ変換されるかを示す。

Mermaid記法として`flowchart`を使っていても、図が示す意味によって分類する。

## 6. 図対象の全体再レビュー結果

現在、図を使う代表仕様と判定は次のとおり。

```text
factorize_conv2d_layer
  分解・配置図
  → SVD成分と2層Conv weightの対応を示すため残す

factorize_named_layers
  フローチャート + シーケンス図 + コンポーネント図
  → loop/type error、責務分担、置換構造で役割が分かれるため残す

sweep_layer_ranks
  フローチャート + シーケンス図
  → rank loop/optional MACs と callback/metrics委譲で役割が分かれるため残す

hosvd
  フローチャート
  → mode反復と「factorは元Xから求める」を視覚化するため残す

hooi_sweep
  フローチャート
  → target mode反復と更新済みfactorを次modeで使う流れを示すため残す

hooi
  フローチャート
  → iterative sweepと収束/最大反復停止を示すため残す

build_tucker2_conv
  シーケンス図
  → Tensor分解APIとModule builderの責務分離を示すため残す

build_tucker2_conv_from_components
  Component配置図のみ
  → 一本道フローチャートは削除。component/属性の配置先だけ図示する

fit_with_early_stopping
  フローチャート
  → loader選択、best更新、patience、epoch loopを示すため残す

benchmark_inference
  フローチャート
  → input有無、CUDA同期、return形式の分岐を示すため残す

collect_compression_metrics
  フローチャート + シーケンス図
  → optional分岐と複数評価APIへの委譲で役割が分かれるため残す

MNISTMLP / FashionMNISTCNN / CIFAR10CNN
  構造図
  → named layer位置とfeature/channel dimension変化を示すため残す
```

一方、`count_parameters`, `has_converged`, `linear_macs`, `get_named_module`, Pareto/kneeの小helper、一本道のbuilder / wrapperなどは、図を追加しても文章・式の描き直しになるため追加しない。

## 7. 複数図の重複セルフレビュー

### `factorize_named_layers`

3図すべてを残す。

- フローチャート: Conv/Linear loop、型分岐、異常終了、処理順。
- シーケンス図: baselineを分解元、copyを置換先とする責務分担。
- コンポーネント図: model内の指定層だけがfactorized Moduleへ変わる構造。

フローチャート内の`factorize_*`とcopy側置換は別工程として残す。ここを1箱へまとめると処理理解の損失が大きい。

### `sweep_layer_ranks` / `collect_compression_metrics`

フローチャートはloop・optional分岐、シーケンス図はcallback・評価APIへの委譲を担当し、役割が異なる。

### `build_tucker2_conv_from_components`

以前はフローチャートとComponent配置図を併記していたが、再レビューでフローチャートは削除した。

理由は、

```text
validation
→ Module生成
→ component copy
→ return
```

が一本道であり、番号付き`処理概要`と同じ情報しか持たなかったため。

Component配置図は、`u_in / core / u_out / bias / spatial config / device / dtype / requires_grad`の配置先という文章より把握しやすい情報を持つので残す。

## 8. 今回の再レビューで変更するもの

今回のMermaid再レビューでは、documentationのみを変更する。

- `build_tucker2_conv_from_components.md`
  - 一本道のフローチャートを削除。
  - Component配置図を残す。
- `factorize_conv2d_layer.md`
  - 図を「フローチャート」から「分解・配置図」へ分類し直す。
  - SVD componentと2層Conv weightの対応が見える形へ整理。
- `05_Core_API_v1/README.md`
  - フローチャート条件を「実際の制御構造」に限定。
  - 処理段階数だけでは図を追加しないことを明文化。
- 本セルフレビュー
  - 全図対象の再判定結果を記録。

`src / tests / Notebook / results`は変更しない。

## 9. テスト・実装影響

今回はdocumentation-only変更なので、src挙動・実験値・API signatureは変更していない。

Mermaidの意味・分類・重複を見直す作業であり、pytestの実行結果には影響しない。

## 10. 結論

図を入れるかどうかは、APIが複雑か、手順が長いかではなく、**図によって文章だけでは把握しにくい情報が追加されるか**で判断する。

今後Public APIを追加・変更する場合は、

```text
src実装
→ __all__
→ contract test
→ 対応する関数別md
→ 引数 / 戻り値の日本語意味を確認
→ 図が文章に無い情報を追加するか確認
→ 図種が制御 / 責務分担 / 構造のどれか確認
→ 複数図なら役割重複がないか確認
→ 必要なら上位設計書
```

を同時に更新する。
