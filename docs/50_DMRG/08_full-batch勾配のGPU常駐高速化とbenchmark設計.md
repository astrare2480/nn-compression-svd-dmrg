---
title: full-batch勾配のGPU常駐高速化とbenchmark設計
aliases:
  - GPU resident full-batch benchmark
tags:
  - full-batch gradient
  - DataLoader
  - GPU resident
  - batch size
  - benchmark
  - performance
---

# full-batch勾配のGPU常駐高速化とbenchmark設計

## このノートの役割

同じ学習データ全体を局所更新ごとに何度も使うfull-batch学習では、forward/backward以外のデータ供給が大きな負荷になり得る。本ノートは、DataLoader方式と前処理済みGPU tensor再利用方式を比較するときの一般技術、数値同等性、測定区間、memoryとの交換条件を整理する。

Fashion-MNISTの実測値と採用判断は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]へ分ける。

---

## 1. full-batch局所更新のデータ経路

### 1.1 DataLoader方式

各局所更新で全データを走査する場合、DataLoader経路は概ね次になる。

```text
Dataset.__getitem__
→ transform
→ collateしてCPU batchを作る
→ CPUからGPUへ転送
→ forward/backward
```

局所更新を$K$回行うと、同じsampleに対する取得・決定的前処理・batch構築を$K$回繰り返す。

### 1.2 GPU常駐方式

データがGPU memoryへ収まり、前処理結果を固定してよい場合は、

```text
最初にDatasetをCPU tensorへ積む
→ GPUへ一度転送
→ 各更新ではGPU tensorをslice
→ forward/backward
```

とできる。

これはCPUで行っていたtransformをGPU kernelへ移植する方法ではない。**毎更新繰り返していた準備を初期化時の1回へまとめ、結果を再利用する方法**である。

---

## 2. 何が省かれ、何が残るか

GPU常駐で省ける候補は次である。

- sampleごとのDatasetアクセス。
- 決定的transformの再実行。
- Python/collateによるCPU batch構築。
- batchごとのhost-to-device転送。

一方、modelが各更新で変わるため、次は省けない。

- forward。
- loss計算。
- backwardと勾配蓄積。
- parameter候補作成。
- TT pairのSVD再分割。
- 候補検査とcommit。

したがって、GPU常駐は学習sampleやgradient計算を削減する近似ではない。データ供給経路の重複作業を減らす最適化である。

---

## 3. batch sizeと全データ平均勾配

batch sizeを$B$としても、全データ目的を

$$
L=\sum_b\frac{n_b}{N}L_b
$$

で計算すれば、sample重みは同じである。batch size変更で変わるのは主に、

- batch loop回数。
- forward/backward呼出し回数。
- kernel launchとPython overhead。
- 勾配加算の順序。
- peak memory。

である。

浮動小数点加算は結合則を厳密には満たさないので、batch sizeを変えるとbitwise identicalにはならない。速度比較では、同じ数学的目的を許容差内で再現するか確認する。

### 3.1 ボトルネック依存性

データ準備が支配的なら、batch loopを減らしてもsampleごとの準備が残るため、batch拡大の効果は小さいことがある。データ供給を軽くした後は、batch loop、同期、forward/backward呼出しの削減が目立つ。

最適化の効果は施策単独で決まらず、現在残っているボトルネックに依存する。

---

## 4. 4条件factorialで原因を切り分ける

データ供給方式とbatch sizeを同時に変えた2条件だけでは、どちらが効いたか分からない。少なくとも次の$2\times2$を比較する。

| | small batch | large batch |
|---|---|---|
| DataLoader | baseline | batch拡大だけ |
| GPU常駐 | 常駐だけ | 両方 |

この比較から、

- DataLoader行：batch拡大の効果。
- small batch列：GPU常駐の効果。
- GPU常駐行：供給負荷を除いた後のbatch拡大効果。
- 対角：最終候補の総合効果。

を分けて読める。

ただし、2水準だけでは最適batch sizeを求めたことにはならない。

---

## 5. 数値的同等性を何で確認するか

速度だけで採用すると、sample欠落、loss重みの違い、更新対象の違いを見逃す。単一局所更新では次をbaselineと比較する。

- 使用sample数とsample順。
- 全データ平均loss。
- 結合テンソル勾配。
- 各通常parameterの勾配。
- SGD後・SVD前のmerged tensor。
- SVD後に再結合したpair。
- 再構成した実効weight。
- 更新後の通常parameter。
- retained rank、特異値、discarded norm。
- 非有限値。

相対Frobenius差は

$$
\delta_{\mathrm{rel}}
=
\frac{\lVert A-B\rVert_F}{\lVert B\rVert_F}
$$

で測れる。ただし、$\lVert B\rVert_F$がほぼ0なら相対差は不安定なので、最大絶対差とFrobenius差の絶対基準を使う。

1往復後は、validation loss、accuracy、prediction agreement、logits RMSE、最終center、pair順、更新数を確認する。core要素はSVDの符号・縮退部分空間の非一意性で直接一致しない場合があるため、実効pairや実効weightも比較する。

短期の許容差合格は、長時間trajectoryの一致を保証しない。採用候補を長時間学習へ移す前にsmokeを行う。

---

## 6. wall-clock測定の境界

速度比を報告するときは、timerが何を含むかを明示する。

### 6.1 分けて記録する時間

- CPU上でのdataset stack作成。
- GPUへの一回のpreload。
- checkpoint restore。
- 定常的な局所更新。
- 往復後validation。
- checkpoint・CSV保存。

GPU常駐方式はpreloadという一回限りの費用を払う。長い反復では償却できるが、1更新だけなら無視できない。

総時間は概念的に

$$
T_{\mathrm{total}}
=
T_{\mathrm{prepare}}
+T_{\mathrm{preload}}
+K T_{\mathrm{update}}
+T_{\mathrm{evaluation}}
+T_{\mathrm{save}}
$$

と分けられる。$K$が大きいほど、$T_{\mathrm{update}}$の短縮が支配的になる。

### 6.2 CUDA同期

CUDA演算は非同期なので、単純なCPU壁時計ではkernel完了を待たず短く見える。区間の前後で`torch.cuda.synchronize()`するか、CUDA eventを使う。

同期自体にもoverheadがある。batchごとに同期する測定では、batch数削減が「本来の演算効率」と「計測同期回数削減」の両方を含む場合がある。

### 6.3 反復と代表値

- 最初にwarm-upを行う。
- 複数反復を測る。
- 外れ値に強い中央値を代表値にする。
- baselineと候補を同じ環境で測る。
- 実測範囲を超えて速度比を一般化しない。

---

## 7. speedとmemoryの交換条件

large batchとGPU常駐は通常、peak allocated memoryを増やす。採用基準には速度だけでなく、

$$
\frac{M_{\mathrm{peak}}}{M_{\mathrm{GPU}}}<\alpha
$$

のようなmemory上限を入れる。余裕を残すのは、model更新、workspace、validation、長時間実行時のfragmentationへ備えるためである。

GPU全量常駐が使えない場合は、次を検討する。

- CPU側の前処理結果だけをcacheする。
- pinned memoryとnon-blocking転送。
- DataLoader workerとprefetch。
- 一部だけGPU cacheする。
- device上のchunkを再利用する。

これらはGPU全量常駐と同じ方式ではないため、別条件として数値と速度を確認する。

---

## 8. 適用できない・注意が必要な条件

### 8.1 ランダムaugmentation

毎epoch・毎sampleでランダム変換する学習では、前処理済みtensorを固定して再利用するとaugmentation分布が変わる。GPU常駐するなら、raw dataだけを置いてdevice側でaugmentationするなど、学習条件を保つ設計が必要である。

### 8.2 データがGPUへ収まらない

大規模画像・音声・系列では全量常駐できない。Fashion-MNISTで有効だったことを、そのまま大規模datasetへ一般化しない。

### 8.3 DataLoader baselineが未最適化

`num_workers=0`、`pin_memory=False`、同期転送のbaselineに対する速度比は、最適化済みDataLoaderに対する速度比ではない。報告ではbaseline設定を必ず併記する。

### 8.4 最速batchが一般最適とは限らない

比較した候補中で最速でも、未測定の中間・大batchがよりよい可能性がある。目的が候補採用なら探索範囲を明記して閉じ、最適化研究へ無制限に広げない。

---

## 9. 実装構造

データ供給と局所更新を分けると、同じ更新処理へ異なるsourceを渡せる。

```python
# 条件ごとに、データ供給源だけを準備する。
source = prepare_data_source(mode, batch_size)

# 局所更新側は、どの供給源でも同じbatch interfaceを使う。
for images, labels in source.iter_batches():
    weighted_loss = criterion(model(images), labels) * (len(labels) / n_samples)
    weighted_loss.backward()

# 全batchの勾配を集めてから、一度だけ候補更新する。
apply_candidate_update()
```

この分離により、DataLoaderとGPU常駐で勾配計算・SVD・commitの実装を二重化せずに済む。速度比較で変更する範囲も明確になる。

---

## 10. benchmarkチェックリスト

- 変更因子をデータ供給方式とbatch sizeに限定する。
- 同じcheckpoint、sample順、loss、更新対象、pair順を使う。
- sample欠落・重複を確認する。
- $n_b/N$で全データ平均を維持する。
- warm-upを保存結果へ混ぜない。
- CUDA測定区間を同期する。
- 単一更新と1往復を分ける。
- preload、restore、validation、saveを別時間にする。
- 複数反復の中央値を報告する。
- 勾配、候補、実効weight、validationを許容差で比較する。
- 非有限値と更新順を確認する。
- peak allocatedとreservedを分ける。
- 速度比のbaseline設定を明記する。
- 短期同等性を長時間同一性へ一般化しない。
- ランダムaugmentationとGPU容量の適用条件を明記する。

## 関連

- [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]

