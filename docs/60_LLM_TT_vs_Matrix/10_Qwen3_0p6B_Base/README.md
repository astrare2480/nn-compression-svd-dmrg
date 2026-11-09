# Qwen3-0.6B-Baseの単層圧縮実験

Qwen3-0.6B-BaseのLinear層を対象に、行列低rank近似とTT-matrix近似を、保存parameter予算と入力データ計量をそろえて比較する。

## 結果ノート

1. [[60_LLM_TT_vs_Matrix/10_Qwen3_0p6B_Base/03_長いALSの収束確認と初期値依存]]
2. [[60_LLM_TT_vs_Matrix/10_Qwen3_0p6B_Base/04_本実験_複数の層と射影でのTTと行列の比較]]
3. [[60_LLM_TT_vs_Matrix/10_Qwen3_0p6B_Base/05_考察_TTは行列にどこまで迫れたか]]（01〜04 と追加分析をまとめた考察）

対応するNotebookは `notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/`、保存結果は `results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/` にある。

## 技術ノート

- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/81_計量付きTT_ALSの局所方程式と停留性]]
