# 20_pythia_410m / 01 外部確認（Pythia-410M の中間層）

- 状態（2026年11月8日）：**設計は確定**（正本に従う）。実装・ダウンロードはまだ
- **規則の正本**：`plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment/design.md`。この文書には Pythia に固有の点だけを書く。**正本の規則を上書きしない。** 食い違いがあれば正本が優先する

## 1. 正本から受け継ぐもの

正本の §3（ソルバーと解き直し）、§4（停止の規則）、§5（初期値。Pythia は M3 から始める本だけで、M4r は回さない）、§6（GPU・試運転・中断と再開。時間の上限 48 時間は Qwen3 と合わせた全体の値）、§7（方式・検証データの作り方・区間・分類・報告だけの指標・検算）をすべてそのまま使う。全体の結論の書き方は正本の §7.5 の「Pythia」の項に従う（合否を決めず、内訳を報告する）。

## 2. Pythia に固有の点

### 2.1 モデル

- `EleutherAI/pythia-410m` の標準の版（deduped ではない）の最終チェックポイント（`main`）。
- 選ぶ理由：Pythia の系列の標準のチェックポイントで外部確認をするため。deduped 版を選んでも評価データとの重複の懸念はなくならないので、両方は回さない。
- 実行のときに記録するもの：repository ID、解決した commit SHA、チェックポイントの名前と step、tokenizer の revision、model config、重みの dtype。
- 24 層・hidden 1024 などの構成は、ダウンロードしてから config で確かめる（今は Claude の知識と ChatGPT が挙げた公式の config による。Claude は未確認）。
- ダウンロード（約 0.8 GB）は、この段階に入るときにユーザーの許可を得る。キャッシュは Qwen3 と同じ `data/hf_cache/`。

### 2.2 層と射影

- 層：**0 から数えて 12（13 番目のブロック）**。24 層の真ん中は 11 層と 12 層の間なので、「真ん中に隣り合う 2 層のうち、前もって選んだ一方」とする（Qwen3 の 14/28 と同じ規則）。
- 射影：qkv（1024→3072）、attention の dense（1024→1024）、h_to_4h（1024→4096）、4h_to_h（4096→1024）の 4 つ × 予算 50%・25% ＝ 8 セル。
- qkv は 3 つの射影をまとめた 1 つの行列のまま圧縮する。
- 形とランク、結合の飽和の分類は正本の §2.2・§2.3 の表のとおり。50% の dense・h_to_4h・4h_to_h は `near_saturated_bond` で、`multi_cut` は 50% の qkv と 25% の 4 射影の 5 セルだけ。
- 入力のグループ：attention の入力（qkv）、attention の出力側の入力（dense）、MLP の入力（h_to_4h）、MLP の中間の入力（4h_to_h）。Pythia は attention と MLP を並列に計算し、それぞれ別の LayerNorm を通すので、qkv と h_to_4h の入力の Gram 行列は別になる（実装で確かめる）。

### 2.3 バイアス

- Pythia の線形層にはバイアスがある。バイアスは圧縮せず、元の値を全方式で保持し、予算の外に置く。
- 誤差 e は (W − Ŵ)x で測るので、バイアスは打ち消し合い、目的関数にも判定にも入らない。
- パラメータ数は「圧縮する重みだけ」と「バイアスを含む射影全体」の 2 通りを報告する。
- サブ層の出力と NLL の評価でも、バイアスは元の値を使う。

### 2.4 データ

- 較正・検証の作り方は正本の §7.2 と同じ規則を、Pythia のトークナイザーで当てはめる（較正は WikiText-103 train の 128 系列 × 1,024 トークン、WikiText-2 validation の 1,024 トークン以上の記事を全文、C4 validation の 1,024 トークン以上の文書から 128 文書）。
- トークナイザーが違うので、「1,024 トークン以上」の記事・文書の数や、選ばれる文書が Qwen3 と変わりうる。規則は変えずに、数と文書の hash を記録する。
- 学習データとの重複：照合しない。書き方は正本の §2.1 のとおり（「較正に使っていない固定の文書で評価した。事前学習のデータとの文書の重複は照合していない」）。

### 2.5 Qwen3 と違う点のまとめ

| 項目 | Qwen3（正本） | Pythia（この文書） |
|---|---|---|
| 層 | 7・14・21 | 12 だけ |
| 射影 | 7 つ | 4 つ（qkv はまとめた 1 行列） |
| バイアス | なし | あり（保持し、予算の外） |
| 初期値 | 全セル M3、layer 14 だけ M4r も | M3 だけ |
| 正規化 | RMSNorm | LayerNorm（attention と MLP が並列） |
| 結論の書き方 | §7.5 の W・B・F | 内訳だけ（§7.5 の「Pythia」の項） |
| λ の確認 | §7.7 | しない（正本の §7.7 は Qwen3 の layer 14 だけ） |

## 3. 置き場所

- パス：`get_experiment_dirs(PROJECT, "50_llm_tt_vs_matrix", "20_pythia_410m", "01_external_check")`
- `notebooks/50_llm_tt_vs_matrix/20_pythia_410m/01_external_check.ipynb`
- `results/50_llm_tt_vs_matrix/20_pythia_410m/01_external_check/`
- `models/50_llm_tt_vs_matrix/20_pythia_410m/01_external_check/`
- `data/50_llm_tt_vs_matrix/20_pythia_410m/01_external_check/`（git の管理対象外）
