---
title: src設計書
aliases:
  - Core API v1
  - nn_compression設計
  - src architecture
---

# src設計書

## 目的

このディレクトリは、`src/nn_compression/` の**現在の実装責務とPublic API contractを固定する設計書**である。

SVD → Tucker / HOSVD / HOOI までの学習・実験・src共通化・contract reviewが一段落した時点を **Core API v1** として区切り、以後のTT/MPS・DMRG実装で既存APIを不用意に壊さないために作成した。

設計書のsource of truthは、作成時点の `main` のsrc実装である。

```text
src baseline commit: 49836bc
review state: src codex rv6
local regression tests: 252 passed / 0 failed
```

この設計書を追加する変更自体は `src/` を変更しない。

---

## Core API v1の意味

Core API v1で固定するのは、**内部実装そのものではなくPublic APIの外部契約**である。

原則として維持するもの：

- 公開関数・公開クラス名
- 引数名、位置引数 / keyword-only の区別、既定値
- 入力shape / rank / mode等の意味
- 戻り値の構造
- 明示的なvalidationと主要な例外条件
- device / dtype / `requires_grad` の扱い
- baseline modelを破壊・共有しない契約
- autograd境界
- train/eval状態の保存・復元契約
- empty input / empty loaderの扱い

変更してよいもの：

- `_` から始まるprivate helper
- Public API contractを変えない内部refactor
- validation helperの配置・共通化
- アルゴリズムの数値結果を変えない高速化
- コメント・docstring・型注釈の改善
- テストの追加

TT/MPS・DMRG追加では、既存Public APIへのbreaking changeより**新しいAPIの追加**を優先する。

---

## Public APIの判定基準

この設計書では、各サブパッケージの `__init__.py` にある `__all__` をPublic APIの基本境界とする。

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

`nn_compression` 直下の `__init__.py` は現在サブパッケージAPIを再exportしていないため、利用側は原則として

```python
from nn_compression.compression import ...
from nn_compression.metrics import ...
```

のように責務単位でimportする。

`__all__` に含まれない関数は、名前が `_` で始まらなくても内部実装またはサブモジュール内補助APIとして扱い、Core API v1の固定対象にはしない。

---

## 読む順番

1. [[07_src設計/01_全体構成と設計方針]]
2. [[07_src設計/02_Core_API_v1]]
3. [[07_src設計/03_tensor設計]]
4. [[07_src設計/04_compression設計]]
5. [[07_src設計/05_training_metrics設計]]
6. [[07_src設計/06_実験支援API設計]]
7. [[07_src設計/07_既知の制約と拡張方針]]

---

## ファイル構成

```text
docs/07_src設計/
├─ README.md
├─ 01_全体構成と設計方針.md
├─ 02_Core_API_v1.md
├─ 03_tensor設計.md
├─ 04_compression設計.md
├─ 05_training_metrics設計.md
├─ 06_実験支援API設計.md
└─ 07_既知の制約と拡張方針.md
```

### 01_全体構成と設計方針

`src/nn_compression/` のパッケージ構成、責務、依存方向、Notebookとの役割分担を定義する。

### 02_Core_API_v1

Public APIの凍結リスト。既存関数・クラスのprimary API、互換API、固定範囲を一覧化する。

### 03_tensor設計

`unfold / fold / mode_dot` とtensor validationのshape / mode contractを扱う。

### 04_compression設計

SVD、Linear / Conv2d分解、HOSVD、HOOI、Tucker-2 Conv、rank sweepの設計を扱う。

### 05_training_metrics設計

学習・評価、状態復元、モデル比較、MACs、benchmark、empty loader等を扱う。

### 06_実験支援API設計

models / datasets / selection / utilsを扱う。実験結果の再現性とNotebookとの接続もここに置く。

### 07_既知の制約と拡張方針

現時点で意図的に残している制約と、TT/MPS・DMRGを既存APIを壊さず追加する方針を定義する。

---

## 設計書と既存ノートの役割分担

```text
00_基礎理論
→ 数学・PyTorch・実験設計を理解する

05_SVD基礎実装検証 / 06_Tucker基礎実装検証
→ 実装時に何を確認・修正したかを残す

07_src設計
→ 現在のsrcが何を保証するかを固定する

10/20/30...
→ 各実験の条件・結果を残す
```

設計書へ実験結果の表を重複して持ち込まず、srcの責務とcontractに集中する。

---

## 変更ルール

Core API v1を変更したくなった場合は、まず次を確認する。

```text
1. private helperの変更だけで実現できないか
2. 新規API追加で実現できないか
3. 既存signature / return / exception contractを維持できないか
4. contract testを壊すbreaking changeか
```

breaking changeが本当に必要なら、Notebook都合でその場変更せず、Core API v2相当の変更として設計書・テスト・利用Notebookを同時に更新する。
