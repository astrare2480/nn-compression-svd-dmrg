---
title: Core API v1 セルフレビュー
aliases:
  - Core API v1 RV
  - src設計セルフレビュー
---

# Core API v1 セルフレビュー

## 1. レビュー目的

`docs/07_src設計/` を作成した後、設計書をそのまま確定せず、次の4点を相互に照合した。

```text
Public export (__all__)
↕
実装 signature / behavior
↕
contract test
↕
設計書・既存ノート
```

目的は、TT/MPSへ進んだ後で既存SVD/Tucker APIを何度も書き直さなくて済むよう、**Core API v1のfreeze前に小さいcontract gapを潰すこと**。

レビュー基準となったmain：

```text
49836bc  src codex rv6
baseline pytest: 252 passed / 0 failed
```

この252件はself review前のbaseline結果であり、以下の追加差分に対する実行結果ではない。

---

# 2. 最終判定の分類

## Critical

なし。

## Major

なし。

HOSVD/HOOI/Tucker-2の中心アルゴリズム、SVD因子分解、既存実験結果を無効化する問題は見つからなかった。

## Minor / freeze前に修正した項目

6件。

1. 設計書の依存方向記述
2. Conv MACs出力空間sizeの型contract
3. standalone Tucker utilityのndim contract
4. Tensor scalarを整数として受ける穴
5. Tucker/HOOIの非浮動小数点dtype
6. Tucker-2 rank上限の設計書表現

---

# 3. Finding 1: 依存方向の記述ミス

### 問題

初稿の `01_全体構成と設計方針.md` で、`training` と `metrics.model_comparison` の依存方向を逆に読める図になっていた。

### 実装の事実

```text
metrics.model_comparison
→ training.loops
```

であり、`evaluate` とtraining-state保存helperをmetrics側が利用する。

さらに、

```text
compression.hooi
→ metrics.relative_frobenius_error
```

という依存もある。

### 修正

依存図を実装どおりに修正し、「矢印 = importして利用する側 → 依存先」と明記した。

### 重要性

validation helper共通化で循環importを作らないための前提になる。

---

# 4. Finding 2: Conv MACs `out_h / out_w` のcontract不足

### 問題

以前の `compressed_conv2d_macs()` は、

```text
out_h > 0
out_w > 0
```

は確認していたが、「正の整数」というメッセージに対して型を厳密には検証していなかった。

`conv2d_macs()` はさらに型・正値validationを持たなかった。

そのためfloat等でも偶発的にMACs値を返し得た。

### 修正

`metrics/macs.py` にprivate helperを追加し、

```text
conv2d_macs
compressed_conv2d_macs
estimate_conv2d_macs
```

で、

```text
bool拒否
Tensor scalar拒否
integer scalar必須
> 0
```

へ統一した。

NumPy integer scalar等の `__index__` 対応scalarは受理する。

### 影響

通常の整数 `out_h / out_w` を使う既存Notebookの挙動は変わらない。

---

# 5. Finding 3: standalone Tucker utilityのndim不一致

### 問題

`hosvd()` は `X.ndim >= 2` を要求する一方、

```python
reconstruct_tucker(core, {})
core_from_factors(X, {})
```

ではfactorが空だとmode演算が一度も呼ばれず、1階Tensorをそのまま返せた。

### 修正

Public Tucker APIのshape contractを揃えるため、

```text
reconstruct_tucker: core.ndim >= 2
core_from_factors: X.ndim >= 2
```

を入口で明示的にvalidationする。

### 影響

2階以上の既存Tucker/HOOI利用には影響しない。

中心数式は変更していない。

---

# 6. Finding 4: Tensor scalar整数の抜け道

### 問題

Pythonでは `operator.index(torch.tensor(1))` が整数として評価できる場合があるため、

```text
Python boolを拒否
→ operator.index(...)
```

だけではSVD側の「Tensor scalarは拒否」と同じcontractになっていなかった。

対象：

```text
metrics MACs rank
Conv MACs out_h / out_w
benchmark warmup / repeats
```

### 修正

metrics側でも `torch.is_tensor(value)` を明示的に拒否した。

これにより、整数scalar policyは少なくとも以下で一致する。

```text
SVD rank / max_iter
MACs rank
MACs output spatial size
benchmark warmup / repeats

→ Tensor scalarは入力しない
```

NumPy scalarの受理方針はmode系と完全統一していないため、別の既知制約として残す。

---

# 7. Finding 5: Tucker/HOOIの非浮動小数点dtype

### 問題

従来の `validate_real_dtype()` は複素dtypeだけを拒否していた。

そのため整数/bool TensorはPublic API境界を通過し、`torch.linalg.svd()` や `torch.linalg.vector_norm()` の内部例外へ到達する可能性があった。

### 仕様判断

このリポジトリのHOSVD/HOOI対象はNN weightであり、正式対応範囲を

```text
実数浮動小数点Tensor
```

とする。

### 修正

`validate_real_dtype()` を、

```text
complex
→ TypeError（複素数未対応）

integer / bool
→ TypeError（実数浮動小数点dtypeが必要）

floating-point real
→ supported
```

へ明確化した。

`hooi()` ではzero-norm計算より先にdtype validationを行い、PyTorch内部例外よりPublic API contractを優先する。

### 影響

既存のfloat32 / float64 NN weight実験には影響しない。

---

# 8. Finding 6: Tucker-2 rank上限の説明不足

### 問題

設計書初稿では、Tucker-2 rankを

```text
rank_out <= C_out
rank_in <= C_in
```

だけで説明しており、generic HOSVD/HOOIから継承するmode-unfolding feasibilityが読み取りにくかった。

### 正しい整理

component shapeとしては、

```text
R_out <= C_out
R_in  <= C_in
```

が必要。

さらに分解APIでは、weight `(C_out,C_in,kH,kW)` に対して、

```text
mode 0 max = min(C_out, C_in*kH*kW)
mode 1 max = min(C_in, C_out*kH*kW)
```

を適用する。

### 修正

`02_Core_API_v1.md` と `04_compression設計.md` で、

```text
component builderのshape contract
≠
decomposition APIのrank feasibility
```

を分離して記載した。

---

# 9. 追加したcontract test

`tests/test_core_api_v1_contract.py` を追加した。

主な追加確認：

```text
Conv MACs out_h/out_w
- bool拒否
- 0/負数拒否
- float拒否
- Tensor scalar拒否
- 正の整数境界を受理

compressed MACs rank
- Tensor scalar拒否

benchmark warmup/repeats
- Tensor scalar拒否

Tucker standalone utility
- 1階Tensor拒否

HOSVD / HOOI dtype
- integer Tensor拒否
- bool Tensor拒否
```

既存complex dtype testは、複素数向けの明示エラーメッセージを維持する。

---

# 10. 意図的に修正しなかった項目

以下はCore API v1を止める問題ではないため、既知制約として残す。

### NumPy scalar受理方針

rank系はNumPy整数scalarを受理し、mode系はPython int中心。この統一はAPI policyを決めてから行う。

### `collect_compression_metrics()` のone-shot iterable

同じloaderを複数回走査するため、re-iterable loader前提。

### validation helper重複

`compression.hooi → metrics` という依存があるため、単純にmetricsからcompression helperをimportすると循環しやすい。

### `non_shuffling_loader()` の完全clone

custom sampler / batch_samplerまで完全再現するAPIではない。

### `take_inference_batch()` のcustom sampler

標準 `RandomSampler` は拒否するが、任意custom samplerの乱数消費までは自動判定しない。

### `fit_with_early_stopping()` のscalar validation

experiment-support APIとして現在のsignature/semanticsを固定し、HOOI/benchmark並みの全面validation統一は別課題とする。

---

# 11. 変更していないもの

self reviewによるsrc変更はPublic API境界のvalidation追加だけ。

変更していない：

```text
truncated SVDの中心計算
HOSVD factor計算式
HOOI projection式
HOOI sweep順序
convergence式
Tucker-2 3層構造
保存済み実験結果
Notebookのcanonical実験値
```

---

# 12. テスト実行状況

設計書作成前のmain `49836bc` では、

```text
252 passed
0 failed
```

が確認済み。

一方、self reviewで追加したsrc/test差分については、この作業環境からGitHubリポジトリを実行環境へ取得できず、GitHub Actionsも設定されていないため、**全pytestの実行確認は未完了**。

Core API v1の最終freeze条件は、ローカルで

```bash
python -m pytest tests -q --tb=short
git diff --check
```

がgreenになること。

この実行確認までは、

```text
設計書作成: 完了
静的self review: 完了
freeze前の修正: 完了
全pytest確認: 未完了
Core API v1最終確定: pytest確認待ち
```

と扱う。

---

# 13. 結論

静的レビュー上、Core API v1の設計書作成を止めるCritical/Major問題は残っていない。

freeze前に修正すべき小さなcontract gapは上記の通り修正した。

全pytestとdiff checkがgreenであることを確認できれば、SVD〜Tucker/HOOIの既存Public APIを **Core API v1として固定し、TT/MPS実装へ進んでよい**。
