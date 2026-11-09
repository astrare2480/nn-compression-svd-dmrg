# Codex へのレビュー依頼：本実験の評価のコード

作成：2026-11-09　実装と実行：Claude（今回は Cursor を使わず、Claude が書いて実行した）
設計：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md` の §7（以下「設計」）

## 0. お願いしたいこと

Claude が書いた評価のコードを、**書いた本人とは別の目で**読んで、誤りを探してほしい。直さずに、見つけた点を報告してほしい（直すかどうかはユーザーと Claude で決める）。

## 1. 読んでほしいファイル

| ファイル | 中身 |
|---|---|
| `src/nn_compression/llm/qwen3_main_statistics.py` | ブートストラップ、e・D・N・H_c の区間、セルの分類（設計 §7.3〜§7.4）、W・B・F（§7.5）、NLL の差の区間（§7.6）。GPU を使わない |
| `src/nn_compression/llm/qwen3_main_evaluation.py` | 方式の行列の作り直しと検算、単位ごとの誤差、サブ層の出力の誤差、置き換えた層から後ろだけを計算する NLL |
| `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py` | 実行の流れ（層ごとに GPU で計算して保存 → 集計して CSV・JSON を書く） |
| `tests/test_qwen3_main_evaluation.py` | テスト（統計の小さな例と、小さい Qwen3 での順伝播の部品） |

## 2. 特に確かめてほしい点

1. **分類の規則が設計 §7.4 と一致しているか**（`classify_from_intervals`）。
   - 「D の区間が 0 をまたぐ」は「D の区間の下限が 0 以下」と解釈した。
   - `not_classified` は、`denominator_not_informative` より先に当てはめている。
2. **ブートストラップ**（`bootstrap_counts`・`ratio_point_and_boot`）。
   - 各再標本で、単位ごとの和を足し合わせてから比を取っているか。
   - 全方式・全セルで同じ再標本の行列を使っているか。WikiText-2 と C4 は別の行列で、どちらも seed 0。
3. **単位ごとの誤差**（`unit_quadratic_pass`）。
   - $\|\Delta X_u\|_F^2 = \langle \Delta^\top \Delta,\, G_u \rangle$ として計算している。
   - 単位の区切り（同じ単位の区間が続けて並ぶこと）と、単位の境目で足し込みを切り替える処理が正しいか。
4. **サブ層の出力**（`sublayer_pass`・`run_attention`）。
   - 残差を足す前の出力（o_proj のあとの attention、down_proj のあとの MLP）を測っているか。
   - 保存した層の入力・MLP の入力から計算し直すことが、元のモデルの計算と同じになっているか。
   - 位置と mask は、区間の長さごとに 1 つ保存して使い回している。
5. **NLL**（`cache_window_inputs`・`nll_from_layer`）。
   - 置き換えた層より前を計算しない近道が、モデル全体を通した値と一致するか。
   - 窓の切り方と、数えるトークンが `sliding_window_perplexity` と同じか。
   - 保存した mask を後ろの層にも使っている。全層が `full_attention` であることを前提にしている（`assert_uniform_attention`）。
6. **λ の扱い**（集計の `lambda_sensitive`）。
   - 印は、M5 を別の λ で回した 8 セルにだけつけている。
   - そのほかのセルでは、M2 だけを別の λ にしたときの分類を報告用の列（`class_M2lam…_M5main_*`）に出し、印にはしていない。設計の読み方として妥当か。
7. **方式の検算**（`build_cell_matrices`）。
   - 許容値は相対 1e-9。
   - M5_M4r だけは、M3 以下であることを確かめていない（M4r から始めるため）。

## 3. 実行の結果（参考）

- 検算の一覧：`results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/evaluation/checks.json`
- ログ：`results/.../04_main_experiment/logs/evaluation.log`
