---
title: src設計書
aliases:
  - Core API v1
  - nn_compression設計
  - src architecture
---

# src設計書

## 目的

このディレクトリは、`src/nn_compression/` の**アーキテクチャ・モジュール責務・主要処理フロー・共通設計方針・Public API contract**をまとめた設計書である。

SVD → Tucker / HOSVD / HOOIまでの実装とcontract reviewが一段落した時点を **Core API v1** として区切り、以後のTT/MPS・DMRG追加で既存APIを不用意に壊さないために作成した。

設計書は次の2層に分ける。

```text
設計書
→ なぜこの構造か
→ 各moduleの責務は何か
→ 処理がどう流れるか
→ 何を共通ルールにするか

API仕様
→ 各関数の責務
→ 引数
→ 戻り値
→ 使用場面
→ contract
```

関数ごとの詳細を設計本文へ散らさず、`05_Core_API_v1.md` をPublic API仕様の正本とする。

---

## Core API v1の意味

固定するのは内部実装そのものではなく、**Public APIの外部contract**である。

原則維持するもの：

- 公開関数・公開クラス名
- 引数名、位置引数/keyword-only、既定値
- 引数の意味
- 戻り値構造
- shape/rank/mode contract
- device/dtype/`requires_grad`
- baseline非破壊性
- autograd境界
- train/eval状態復元
- empty input等の主要例外条件

変更可能：

- private helper
- Public contractを変えない内部refactor
- validation helperの配置
- 高速化
- 型注釈・コメント・docstring
- テスト追加

---

## Public APIの判定

各sub-packageの `__all__` をPublic APIの基本境界とする。

```text
nn_compression.tensor.__all__
nn_compression.compression.__all__
nn_compression.training.__all__
nn_compression.metrics.__all__
nn_compression.selection.__all__
nn_compression.datasets.__all__
nn_compression.models.__all__
nn_compression.utils.__all__
```

利用側は責務単位のimportを基本とする。

```python
from nn_compression.compression import hooi
from nn_compression.metrics import relative_frobenius_error
```

---

## 読む順番

1. [[07_src設計/01_アーキテクチャ設計]]
2. [[07_src設計/02_モジュール設計]]
3. [[07_src設計/03_主要処理フロー]]
4. [[07_src設計/04_共通設計方針]]
5. [[07_src設計/05_Core_API_v1]]
6. [[07_src設計/06_テスト設計]]
7. [[07_src設計/07_既知の制約と拡張方針]]
8. [[07_src設計/08_Core_API_v1セルフレビュー]]

---

## ファイル構成

```text
docs/07_src設計/
├─ README.md
├─ 01_アーキテクチャ設計.md
├─ 02_モジュール設計.md
├─ 03_主要処理フロー.md
├─ 04_共通設計方針.md
├─ 05_Core_API_v1.md
├─ 06_テスト設計.md
├─ 07_既知の制約と拡張方針.md
└─ 08_Core_API_v1セルフレビュー.md
```

### `01_アーキテクチャ設計`

src全体の構造、Notebookとの境界、レイヤ構成、拡張方針を定義する。

### `02_モジュール設計`

`tensor / compression / training / metrics / selection / datasets / models / utils` の責務、入力、出力、依存方向を定義する。

### `03_主要処理フロー`

SVD、HOSVD、HOOI、Tucker-2、Fine-tuning、評価の処理シーケンスを示す。

### `04_共通設計方針`

非破壊性、device/dtype、autograd、validation、state復元、DataLoader、compatibility等の横断ルールを定義する。

### `05_Core_API_v1`

Public API仕様書。各関数について、

```text
責務
引数
戻り値
どういうときに使うか
主要contract
```

を記載する。

### `06_テスト設計`

数学テスト、contract test、状態・autograd・互換性テストの役割を定義する。

### `07_既知の制約と拡張方針`

現時点で意図的に残す制約とTT/MPS・DMRGの追加方針を記録する。

### `08_Core_API_v1セルフレビュー`

freeze前のセルフレビューで確認した項目・修正内容・残課題を記録する。

---

## 既存ノートとの役割分担

```text
00_基礎理論
→ 数学・PyTorch・実験設計を理解する

05_SVD基礎実装検証 / 06_Tucker基礎実装検証
→ 実装・実験時に何を確認・修正したかを残す

07_src設計
→ 現在のsrcをどう設計し、何を保証するかを定義する

10/20/30...
→ 個別実験の条件・結果を残す
```

設計書へ実験結果の数値を重複して持ち込まず、srcの構造とcontractへ集中する。

---

## 変更ルール

Core API v1の変更が必要になった場合は、次の順で検討する。

```text
1. private helperの変更だけで実現できないか
2. 新しいPublic API追加で実現できないか
3. 既存signature/return/exception contractを維持できないか
4. breaking changeならCore API v2として扱うべきか
```

TT/MPS・DMRG追加では既存APIを書き換えるより、新しい責務を新APIとして追加する。
