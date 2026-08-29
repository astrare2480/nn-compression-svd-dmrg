---
title: Core API v1 セルフレビュー
aliases:
  - Core API v1 RV
  - src設計セルフレビュー
---

# Core API v1 セルフレビュー

## 1. レビュー目的

設計書作成後、次を相互照合した。

```text
Public export (__all__)
↕
実装 signature / behavior
↕
contract test
↕
設計書・既存ノート
```

目的は、TT/MPSへ進んだ後で既存SVD/Tucker APIを繰り返し書き直さなくて済むよう、**Core API v1 freeze前にcontract gapを潰すこと**。

レビュー基準となったmain：

```text
49836bc  src codex rv6
baseline pytest: 252 passed / 0 failed
```

252件はself review前baselineの結果であり、以下の追加差分に対する実行結果ではない。

---

## 2. 設計書の再編レビュー

初稿は関数ごとの数式・validation説明が設計本文へ入り込み、API仕様書寄りになっていた。

そのため設計書を次の役割へ再編した。

```text
01_アーキテクチャ設計
→ 全体構造・境界・拡張方針

02_モジュール設計
→ package責務・入力/出力・依存方向

03_主要処理フロー
→ SVD / HOSVD / HOOI / Tucker-2 / trainingの流れ

04_共通設計方針
→ 非破壊性・dtype・autograd・state・validation

05_Core_API_v1
→ 各Public APIの責務 / 引数 / 戻り値 / 使用場面 / contract

06_テスト設計
→ contractを実行可能に固定するテスト方針
```

これにより、**設計判断と関数仕様を分離**した。

---

## 3. 最終判定の分類

### Critical

なし。

### Major

なし。

HOSVD/HOOI/Tucker-2の中心アルゴリズム、SVD因子分解、既存実験結果を無効化する問題は見つからなかった。

### Minor / freeze前に修正した項目

6件。

1. 依存方向の記述
2. Conv MACs出力空間sizeの型contract
3. standalone Tucker utilityのndim contract
4. Tensor scalarを整数として受ける穴
5. Tucker/HOOIの非浮動小数点dtype
6. Tucker-2 rank上限の説明

---

## 4. Finding 1: 依存方向

### 問題

初稿でtrainingとmetricsの依存方向が逆に読める表現があった。

### 実装の事実

```text
metrics.model_comparison
→ training.loops

compression.hooi
→ metrics.relative_frobenius_error
```

### 修正

`02_モジュール設計.md` で実装どおりの依存方向を明記した。

`metrics → compression` を追加しないことも共通ルールとした。

---

## 5. Finding 2: Conv MACs `out_h/out_w`

### 問題

以前は正値確認だけで、float等を厳密には拒否していなかった。

### 修正

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

---

## 6. Finding 3: standalone Tucker utilityのndim

### 問題

`hosvd()` は2階以上を要求する一方、

```python
reconstruct_tucker(core, {})
core_from_factors(X, {})
```

は空factorだと1階Tensorを素通しできた。

### 修正

```text
reconstruct_tucker: core.ndim >= 2
core_from_factors: X.ndim >= 2
```

へ統一した。

中心数式は変更していない。

---

## 7. Finding 4: Tensor scalar整数

### 問題

`operator.index(torch.tensor(1))` が通る場合があり、Python boolだけの拒否ではSVD側のpolicyと一致しなかった。

### 修正

```text
MACs rank
Conv MACs out_h/out_w
benchmark warmup/repeats
```

でTensor scalarを明示拒否した。

NumPy scalarの受理方針はmode系と完全統一していないため既知制約として残す。

---

## 8. Finding 5: Tucker/HOOI dtype

### 問題

従来のdtype guardはcomplexだけを拒否し、integer/bool TensorがPyTorch内部例外まで流れる可能性があった。

### 仕様判断

HOSVD/HOOIの正式対応範囲を、

```text
実数浮動小数点Tensor
```

とする。

### 修正

```text
complex
→ TypeError

integer / bool
→ TypeError

floating-point real
→ supported
```

`hooi()` でもzero norm計算より前にdtype validationを行う。

---

## 9. Finding 6: Tucker-2 rank上限

### 問題

初稿は、

```text
rank_out <= C_out
rank_in <= C_in
```

だけで説明し、decomposition時のmode-unfolding feasibilityが見えにくかった。

### 整理

component builderではchannel shape整合を確認する。

一方decomposition APIでは4階weight `(C_out,C_in,kH,kW)` に対して、

```text
mode 0 max = min(C_out, C_in*kH*kW)
mode 1 max = min(C_in, C_out*kH*kW)
```

も適用する。

`05_Core_API_v1.md` と `03_主要処理フロー.md` で役割を分けて記載した。

---

## 10. API仕様レビュー

Public APIは各sub-packageの `__all__` を基準に棚卸しした。

`05_Core_API_v1.md` では各関数に、

```text
責務
引数
戻り値
使用場面
主要contract
```

を記載した。

対象は、

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

のPublic exportと、exportされたmodel classの主要public method。

Compatibility APIはPrimary APIと分離し、新規コードで使うべき名前が分かるようにした。

---

## 11. 追加contract test

`tests/test_core_api_v1_contract.py` を追加した。

主な追加確認：

```text
Conv MACs out_h/out_w
- bool / 0 / 負数 / float / Tensor scalar拒否

compressed MACs rank
- Tensor scalar拒否

benchmark warmup/repeats
- Tensor scalar拒否

Tucker standalone utility
- 1階Tensor拒否

HOSVD / HOOI dtype
- integer / bool Tensor拒否
```

---

## 12. 意図的に修正しなかった項目

Core API v1を止める問題ではないため、`07_既知の制約と拡張方針.md` に残した。

```text
NumPy scalar受理方針の完全統一
collect_compression_metricsのone-shot iterable
validation helper重複
non_shuffling_loaderのcustom sampler完全clone
take_inference_batchのcustom sampler自動判定
fit_with_early_stoppingの全面scalar validation
```

---

## 13. 変更していないもの

```text
truncated SVD中心計算
HOSVD factor計算式
HOOI projection式
HOOI sweep順序
convergence式
Tucker-2 3層構造
保存済み実験結果
canonical Notebook実験値
```

self reviewのsrc変更はPublic API境界のvalidation補完に限定した。

---

## 14. テスト実行状況

self review前のmain `49836bc` では、

```text
252 passed
0 failed
```

を確認済み。

self reviewで追加したsrc/test差分についてはGitHub Actionsが設定されていないため、**全pytest実行確認は別途必要**。

最終freeze条件：

```bash
python -m pytest tests -q --tb=short
git diff --check
```

がgreenであること。

現状は、

```text
設計書再編: 完了
Public API棚卸し: 完了
静的self review: 完了
freeze前validation修正: 完了
contract test追加: 完了
全pytest確認: 未完了
Core API v1最終freeze: pytest確認待ち
```

と扱う。

---

## 15. 結論

静的レビュー上、Core API v1を止めるCritical/Major問題は残っていない。

設計書は「アーキテクチャ・責務・処理フロー」と「関数単位API仕様」を分離した構成へ再編済み。

全pytestとdiff checkがgreenなら、SVD〜Tucker/HOOIの既存Public APIをCore API v1として固定し、TT/MPSへ進んでよい。
