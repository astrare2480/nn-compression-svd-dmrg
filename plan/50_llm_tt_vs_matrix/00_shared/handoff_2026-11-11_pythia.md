# 引き継ぎ：Pythia-410M の外部確認（2026-11-11、新しいチャット向け）

## 方針と役割

- 論文化を優先。Pythia-410M は「Qwen3 で見た傾向が別のモデルでも出るか」の外部確認で、いまの本線。
- 実装は Claude、レビューは Codex、判断はユーザー。長い ALS の本実行は、試運転の結果を見てユーザーと相談してから。
- 設計書：`plan/50_llm_tt_vs_matrix/20_pythia_410m/01_external_check/design.md`（§2.2 形とランク、§2.6 first-token と固有帯の指標、§3.2 τ_warn、§7.5 報告）。

## Qwen3 側で分かったこと（Pythia で確かめる対象）

詳細は `results/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/followup_analysis/README.md`（分析 0〜7）。

- M5（較正の TT の ALS）は、較正でまれな向き（固有値 257 位以降）で M2 より多く外し、C4 ではその差が広がる。
- v で M5 が弱いのは attention sink の先頭トークンのため。先頭の入力はまれな向きに偏り、元の $W$ は先頭をほぼ 0 に写す。M2・M3 はこの性質を保つが、M5 は保たない（まれな向きで、重みなしの相対誤差が 1 を超える）。
- 停留点のごく近くまで解き足しても（$s_{\max}\approx10^{-8}$）先頭の誤差は残る。別の初期値・TT の形・解き方は未確認。
- 次の Qwen3 側の候補（今はやらない）：attention で重みづけした較正の ALS（候補 2）。Pythia の試運転の後に、v の 3 セルだけでやるか決める。

## Pythia の現状

- ダウンロード済み（ユーザー許可あり）：`EleutherAI/pythia-410m`、commit `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`（main = step143000）。記録 `data/hf_cache/download_record_pythia.json`（safetensors の sha256 つき、`data/` は git 管理外）。
- config：24 層、hidden 1024、16 head（head 64）、RoPE は head の 25%（rotary_ndims 16）、attention と MLP は並列（それぞれ別の LayerNorm）、全線形層にバイアス（圧縮しない・予算の外）。重みは float16。
- qkv の出力は head ごとに [q(64), k(64), v(64)] の順。transformers 5.22 では層は `model.gpt_neox.layers`、最後の正規化は `gpt_neox.final_layer_norm`、出力層は `lm_head`。
- 書いたもの（**未実行・未テスト・未コミット**）：`src/nn_compression/llm/pythia_external_check.py`
  - 定数：LAYER=12、射影 `qkv`・`dense`・`hto4h`・`4htoh`（`_` を含めない名前）、予算 50・25%、λ=1e-6（掃引用 1e-7・1e-5）、設計 §2.2 の形とランク（EXPECTED_MODES_AND_RANKS）、50% の qkv 以外は near_saturated。
  - 関数：layout、仕事一覧（16 本：M5 と M3F、M3 から、λ=1e-6）、方式の作成（Qwen3 版の写し＋バイアス・設計との一致の確認）、データ準備（float16 で読み float32 に）、jobs.json、τ_warn の決め方（§3.2）。
  - Qwen3 の `qwen3_main_run` / `qwen3_main_experiment` の補助関数を使う。src 化の約束：「src に足すだけ、元（Qwen3 のモジュール）は触らない」。

## やること（順に）

1. `pythia_external_check.py` の短い動作確認（import、定数と設計の一致、config の確認）。
2. スクリプト（`notebooks/50_llm_tt_vs_matrix/20_pythia_410m/`）：データ準備 → 方式の作成（§2.2 と一致しなければ止まる）→ ALS の実行（`qwen3_main_run.worker_process` を流用）→ 試運転のまとめと τ_warn の決定。
3. 評価のモジュール（`qwen3_main_evaluation.py` の Pythia 版：`gpt_neox.layers`、並列の attention / MLP、`lm_head`）と評価スクリプト（§7.5）。
4. §2.6 の分析（first-token と固有帯の指標。Qwen3 の分析 3・5・7 に対応）。
5. 検算：元のモジュールと自作の attention の出力が一致すること、qkv の並び、並列残差、バイアス、`lm_head` の扱い。
6. Codex へのレビュー依頼書。
7. 試運転で時間とメモリを測り、本実行の計画をユーザーと相談。

## 環境のメモ

- Python：`C:\Users\astra\anaconda3\envs\pytorch-svd\python.exe`（`python` は PATH にない）。`PYTHONIOENCODING=utf-8`、`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`。
- GPU：RTX 5070 Ti 16 GB。GPU を前提に効率を設計する。
- コミットはユーザーが言ったときだけ。main にいるときはブランチを切る。
