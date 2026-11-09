# 本実験の評価コードレビュー結果

レビュー日：2026-11-09  
レビュー担当：Codex Bugbot  
対象：Qwen3-0.6B-BaseのTT対行列比較における評価コード  
基準：`design.md` §7・§8、および `01_evaluation_code_review.md` の確認事項

## 0. 結論

保存済みの数値・分類を誤らせたと確認できる不具合は、確認できた範囲では0件だった。

ただし、レビュー時点のフル評価は未完了だった。

- `evaluation/raw/L14.pt`：225 checks、失敗0
- `evaluation/raw/L7.pt`：169 checks、失敗0
- `evaluation/raw/L21.pt`：未生成
- `evaluation/checks.json`：未生成
- `evaluation/aggregate.json`：未生成
- `evaluation_smoke/checks.json`：48 checks、失敗0
- `logs/evaluation.log`：L21のNLL計算中、`L21_k_50 M4`で終了

この状態から必要なのは、L21の評価を再開して完了し、その後に最終集計を実行することである。L14・L7を最初から再実行する必要はない。

## (A) 結果の数値や分類が変わる誤り

確認できた範囲では0件。

分類、paired bootstrap、単位ごとの二次形式、サブ層出力、NLLの途中層からの再計算について、現在保存されている数値を誤らせたと断定できる実装ミスは見つからなかった。

## (B) 現在の結果は変わらない指摘

### B-1. フル集計が欠損層をエラーにせず、部分結果を書ける

- 重要度：P1
- 確かさ：確実
- 場所：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py:293-299, 382-387`
- 入力例：`L14.pt`と`L7.pt`だけがあり、`L21.pt`がない状態で`--stage summarize`を実行する。
- 起きること：28セルだけでCSVを作成し、`aggregate.main`を`skipped`として正常終了する。フル評価の欠損を失敗扱いしない。
- 誤った結果の例：`checks_passed: true`を含む部分集計が、完了した本実験の結果に見える。
- 現在への影響：レビュー時点ではfull summarizeは未実行であり、既存数値は変わっていない。

### B-2. 指定テストは成功表示になるが、モデル順伝播を使う7テストがskipされる

- 重要度：P1
- 確かさ：確実
- 場所：`tests/test_qwen3_main_evaluation.py:102-117, 125-197`
- 実行結果：`13 passed, 7 skipped in 1.88s`
- 入力例：`transformers`がない現在の環境で、指定されたpytestコマンドを実行する。
- 起きること：NLL全体一致、unit Gram、サブ層再計算のテストが走らない。
- 誤った結果の例：`13 passed, 7 skipped`を、評価コード全体のテストが完了したと読む。
- 現在への影響：保存済みL14・L7では実モデル上の検算がすべて通っているため、数値を直ちに疑う証拠ではない。

### B-3. 再開時に既存rawと現在のモデル・データ・方式の互換性を検査しない

- 重要度：P1
- 確かさ：確実
- 場所：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py:112-114, 267-273`
- 入力例：`units.json`、モデルrevision、方式checkpointのいずれかを変更し、既存の`original_nll.pt`、`L14.pt`、`L7.pt`を残して再実行する。
- 起きること：既存ファイルを無条件で再利用し、新しいL21結果と古いL14・L7結果を混在できる。
- 誤った結果の例：単位数が同じでも記事順が変わった場合、同じbootstrap indexが別の記事を指す層結果を混ぜる。
- 現在への影響：今回のL14・L7は同一実行で作られており、異なる条件の混在を示す証拠はない。

### B-4. rank・parameter検算がALS方式にしか掛からない

- 重要度：P2
- 確かさ：確実
- 場所：`src/nn_compression/llm/qwen3_main_evaluation.py:172-191`
- 入力例：`m3_cores`または`m4_cores`が設計外rankである、あるいはpayloadの`out_modes`・`in_modes`が設計表と異なる。
- 起きること：`ranks_and_parameters`はM5・M3Fなどの保存ALS結果にしか出ず、M3・M4・M4rのrankとmodeを設計表へ直接照合しない。
- 誤った結果の例：`methods_summary`も同じ誤ったpayloadから作られていれば、目的値の再現検査を通過したまま、誤った予算条件を評価できる。
- 現在への影響：42 payloadすべてについて、M3・M4・M4rのmodeとrankは`EXPECTED_MODES_AND_RANKS`に一致していたため、現結果は変わらない。

### B-5. `M5_M4r`が開始点M4rより改善したかを検査しない

- 重要度：P2
- 確かさ：確実
- 場所：`src/nn_compression/llm/qwen3_main_evaluation.py:192-209`
- 入力例：M4rの開始目的値が100、保存された`M5_M4r`の最終目的値が101であり、`jobs_status`にも101が記録されている。
- 起きること：保存値との一致検査は通り、`M5_M4r`ではM3以下を要求しないため、目的値の増加も許される。
- 誤った結果の例：ALSが開始点を悪化させた補助結果でも、検算を通過する。
- 現在への影響：実測12本はすべて`Q(final) / Q(M4r) = 0.10〜0.58`で改善しているため、再実行は不要。

### B-6. 代替λ・補助評価へ`local_solver_unresolved`を伝播しない

- 重要度：P2
- 確かさ：確実
- 場所：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py:339-377`
- 入力例：`M5_lam1e-07`が`local_solver_unresolved`で停止する。
- 起きること：主M5と異なり`solver_unresolved=True`を渡さず、代替λを`high`・`mid`・`low`へ分類し、`lambda_sensitive`判定にも使う。
- 誤った結果の例：未解決の代替λ結果が`high`、主λが`mid`となり、`lambda_sensitive=True`になる。
- 現在への影響：M5のλ感度16本はすべて`study_resolution_reached`であり、主M5 42本にもunresolvedはないため、現結果は変わらない。

### B-7. 情報を持たない分母にも`1-rho`を出す

- 重要度：P3
- 確かさ：確実
- 場所：`src/nn_compression/llm/qwen3_main_statistics.py:153-159`
- 入力例：`D / e(M3) = 0.04`、`D_lo > 0`、`rho = 0.8`。
- 起きること：分類は`denominator_not_informative`だが、`one_minus_rho = 0.2`を出力する。
- 誤った結果の例：設計 §7.4の「1-rhoは分母が情報を持つ場合に報告する」に反し、情報を持たないセルへ有限値が記録される。
- 現在への影響：修正する場合もGPU評価は不要であり、集計だけをやり直せばよい。

### B-8. 「全方式で同じbootstrap再標本」のcheckが実質的な検査になっていない

- 重要度：P3
- 確かさ：確実
- 場所：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py:299-301`
- 入力例：将来、特定方式だけ別のcountsを使うように実装を変更する。
- 起きること：`bootstrap_same_counts_for_all_methods`は実際の使用状況を比較せず、常に`ok: true`になる。
- 誤った結果の例：方式ごとの再標本が異なっていても、`checks.json`では合格する。
- 現在への影響：現コードは同じ`counts[domain]`を各セル・方式へ渡す構造になっているため、現結果は正しい。

## 3. 解釈の確認

### 3.1 `not_classified`と`denominator_not_informative`の優先順位

コードは、`not_classified`を先に適用し、その後に`denominator_not_informative`を適用する。これはレビュー依頼書 §2の解釈と一致する。

一方、design §7.4の表はdenominator判定を「先に当てはめる」とも読める。両方へ該当するセルをどちらへ分類するかは、文書間で曖昧である。今回の結果には両方へ該当するセルがないため、数値・分類への影響はない。

### 3.2 「Dの区間が0をまたぐ」の境界

コードは`D_lo <= 0`を使用している。これはレビュー依頼書に明記された解釈と一致しており、誤りではない。

### 3.3 `lambda_sensitive`の対象

コードは、M2とM5の両方を別λにした8セルだけを`lambda_sensitive`の対象とし、他セルのM2-only変更は報告列だけに出す。これはレビュー依頼書の解釈と一致する。

ただし、design §7.7の「分類がλで変わったセル・入力グループ」だけを読むと、M2-onlyの分類変化も含めるように読める。現在の狭い定義を正式採用するか確認が必要である。

### 3.4 `M5_M4r`とM3の比較

`M5_M4r`はM4rから開始するため、M3以下を必須にしないこと自体は妥当である。ただし、少なくとも開始点M4r以下になったことは検算できる。

## 4. 設計どおりだった主要点

- 各bootstrap再標本で、単位ごとの分子・分母を合計してから比を計算する。
- WikiText-2とC4では別のcountsを使い、どちらもseed 0とする。
- 同じdomainでは、全方式・全セルへ同じcountsを渡す。
- `H_0.2`・`H_0.5`によるセル分類を使う。
- W・B・Fの26/34規則を実装している。
- `unit_quadratic_pass`で`<Delta^T Delta, G_u>`を計算する。
- attentionとMLPのサブ層出力は残差加算前を測定する。
- NLLは記事単位のsliding windowを使い、対象層以降だけを再計算する。
- NLLの対象方式と差の組み合わせは設計どおりである。
- M5のλ感度は8セル×2代替λの16本である。
- 目的値再計算の相対許容値は`1e-9`である。

## 5. 実行したテスト

```text
python -m pytest tests/test_qwen3_main_evaluation.py -q
13 passed, 7 skipped in 1.88s
```

7件のskipは、現在の環境に`transformers`がないためである。モデルのダウンロードや長時間計算は実施していない。

## 6. レビュー時の制約

- コードは変更していない。
- `results/`と`models/`は読み取りだけに使用した。
- GPU評価は実行していない。
- モデルのダウンロードは行っていない。
