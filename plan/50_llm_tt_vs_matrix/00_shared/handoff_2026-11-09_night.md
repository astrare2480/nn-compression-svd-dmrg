# 引き継ぎ（2026-11-09 夜。利用上限の前に Claude が中断）

方針：論文化を優先する（ChatGPT の回答による）。順番は 候補 1 → Pythia（本線）と v の続き計算を並行 → 必要なら候補 2。
TT の切り方を変える比較は今後の課題に回す。実装は Claude、レビューは Codex。

## 動いている・終わった計算（未コミット）

1. **候補 3（v_proj 25% × 3 セルの続き計算）**：`notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_followup_v_continue.py`
   - 04 の M5 から +10,000 掃引。分解能では止めない。100 掃引ごとにコアと e_wt2・e_c4 を保存。
   - 保存先：`models/.../04_main_experiment/followup/v_continue/<cell>/`、記録 `results/.../followup_analysis/v_continue/`、`log_v_continue.txt`
   - 検算：続きの最初の Q_λ は 04 の最終値とビット単位で一致。終了見込み 11-10 の 3〜4 時
   - **終了（3 セルとも +10,000 掃引、max_sweeps）**：最終 $s_{\max}$ は L7 4.2e-8、L14 1.2e-8、L21 1.0e-8 で、停留の基準 1e-6 を下回った（03 の外挿 1e5 掃引より早い）。Q_λ の減少は 04 の最終値から 0.04%（L7）、0.08%（L14）、0.06%（L21）。
   - **11-10 計算済み**（`04_followup_v_continue_first_token.py`）：先頭の v の誤差はほぼ横ばい（射影・先頭 ±3% 程度、head・sink ±4% 以内）。o の後はむしろ少し増える。→ 最適化不足ではなく目的の重みづけ。詳細は followup_analysis/README.md の分析 6 の続き。
2. **候補 1（先頭の入力の固有方向の帯）**：`04_followup_first_token_bands.py`、結果 `followup_analysis/first_token_bands/`、`log_first_token_bands.txt`
   - 途中の値（L7・L14）：先頭の入力の量の 35〜62% が 257 位以降の帯にある（先頭以外は 9〜17%）。「まれな向き」仮説に合う方向。L21 と、方式ごとの誤差の帯分解・検算は、終了後に確認する。
   - 帯の区切りを変える確認（32・128・512 など）と、非対角を含む分解はまだ書いていない。

## Pythia（途中）

- ダウンロード済み（ユーザー許可あり）：`EleutherAI/pythia-410m`、commit `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`（main = step143000）。記録 `data/hf_cache/download_record_pythia.json`（safetensors の sha256 つき）。
- config を確認：24 層、hidden 1024、16 head（head 64）、RoPE は head の 25%、attention と MLP は並列、バイアスあり。重みは float16。
- qkv の出力は head ごとに [q(64), k(64), v(64)] の順（§2.6 で q・k・v に分けるときに使う）。transformers 5.22 では出力層は `lm_head`、最後の正規化は `gpt_neox.final_layer_norm`。
- 書いたもの（未実行・未テスト）：`src/nn_compression/llm/pythia_external_check.py`（定数、方式の作成、データ準備、仕事一覧、τ_warn の決め方）。射影名は `qkv`・`dense`・`hto4h`・`4htoh`（`_` を含めない）。
- まだのもの：
  - スクリプト（`notebooks/50_llm_tt_vs_matrix/20_pythia_410m/`）：データ準備、方式の作成（設計 §2.2 の形と一致するかで止まる）、ALS の実行（`qwen3_main_run.worker_process` をそのまま使える）、試運転のまとめと τ_warn の決定。
  - 評価のモジュール（Qwen3 の `qwen3_main_evaluation.py` の Pythia 版：`gpt_neox.layers`、並列の attention / MLP、`lm_head`）と評価スクリプト（§7.5 の Pythia の項：内訳の報告と「同じ傾向」の条件）。
  - §2.6 の分析スクリプト。
  - Codex へのレビュー依頼書。
- 本実行（長い ALS）はユーザーと相談してから。
