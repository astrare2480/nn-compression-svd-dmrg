---
title: src設計書
aliases:
  - Core API v1
  - nn_compression設計
  - src architecture
---

# src設計書

## 目的

`src/nn_compression/` の**設計判断とPublic API contract**を固定し、TT/MPS・DMRG追加時に既存SVD/Tucker/HOOI実装を不用意に壊さないための設計書。

設計本文と関数仕様を分ける。

```text
設計本文
→ なぜこの構造・責務分離にしているか

Core API v1
→ 各Public APIが何を受け取り、何を返し、何を保証するか
```

## 構成

```text
docs/07_src設計/
├─ README.md
├─ 01_アーキテクチャ設計.md
├─ 02_モジュール設計.md
├─ 03_主要処理フロー.md
├─ 04_共通設計方針.md
├─ 05_Core_API_v1.md                 # API仕様の入口
├─ 05_Core_API_v1/                   # 関数・クラス単位の詳細
│  ├─ README.md
│  ├─ tensor/
│  ├─ compression/
│  ├─ training/
│  ├─ metrics/
│  ├─ selection/
│  ├─ datasets/
│  ├─ models/
│  ├─ utils/
│  └─ Compatibility_API.md
├─ 06_テスト設計.md
├─ 07_既知の制約と拡張方針.md
├─ 08_Core_API_v1セルフレビュー.md
└─ 09_関数別API仕様セルフレビュー.md
```

## 読む順番

1. [[07_src設計/01_アーキテクチャ設計]]
2. [[07_src設計/02_モジュール設計]]
3. [[07_src設計/03_主要処理フロー]]
4. [[07_src設計/04_共通設計方針]]
5. [[07_src設計/05_Core_API_v1/README]]
6. 必要な関数の個別md
7. [[07_src設計/06_テスト設計]]
8. [[07_src設計/07_既知の制約と拡張方針]]
9. [[07_src設計/08_Core_API_v1セルフレビュー]]
10. [[07_src設計/09_関数別API仕様セルフレビュー]]

## Core API v1

Core API v1で固定するのは内部実装そのものではなく、Public APIの外部契約。

原則として維持するもの：

- 公開名とsignature
- 引数の意味・default・keyword-only境界
- 戻り値構造
- shape/rank/mode validation
- device/dtype/requires_grad semantics
- baseline非破壊性
- autograd境界
- train/eval状態復元
- empty input/loaderの扱い

private helper、内部refactor、高速化はPublic contractを維持する限り変更可能。

## Public API判定

各sub-packageの`__all__`を基本境界とする。単なるcompatibility aliasは個別ファイルを増やさず、[[07_src設計/05_Core_API_v1/Compatibility_API]] にまとめる。

## 設計書と既存ノートの役割

```text
00_基礎理論
→ 数学・PyTorch・実験設計を理解する

05/06_基礎実装検証
→ 実装時に何を確認・修正したかを残す

07_src設計
→ 現在のsrc構造とPublic contractを固定する

10/20/30...
→ 各実験の条件・結果を残す
```

実験結果の数値を設計書へ重複して持ち込まず、srcの責務とcontractへ集中する。
