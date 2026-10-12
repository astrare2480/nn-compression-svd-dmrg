---
title: Epoch 0を含むcheckpoint選択と保存モデルの再現確認
tags:
  - model-compression
  - fine-tuning
  - checkpoint-selection
  - validation
  - reproducibility
---

# Epoch 0を含むcheckpoint選択と保存モデルの再現確認

## 1. このノートの目的

圧縮モデルのfine-tuningでは、学習後のepochだけでなく、学習前のepoch 0もcheckpoint候補である。このノートでは次の一般論を整理する。

- epoch 0を候補に含める理由
- 選択指標と診断指標を分ける理由
- accuracyとCross Entropy lossが逆方向へ動き得る理由
- `min_delta`と同点時の扱い
- 保存モデルの再評価と再学習の再現性の違い
- post-hocな再選択の限界

Fashion-MNIST MLPの具体的な保存結果は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/05_epoch0を含むcheckpoint再選択と再現確認]]で扱う。

## 2. epoch 0も有効なcheckpointである

圧縮直後のモデルを$\theta_0$、fine-tuning後の各epochを$\theta_1,\ldots,\theta_T$とする。validation lossで選ぶなら、本来の候補集合は、

$$
\mathcal{C}
=
\{\theta_0,\theta_1,\ldots,\theta_T\}
$$

である。

epoch 1以降だけを候補にすると、すべての更新がepoch 0より悪化していても、その中から1つをbestとして保存してしまう。特に高rankや弱い圧縮では、圧縮直後から十分な性能を持ち、追加更新の改善余地が小さい場合がある。

従って、fine-tuningが常に必要だと仮定せず、`no update`も比較候補に含める。

## 3. `min_delta`を含む選択規則

現在のbest lossを$L_{\mathrm{best}}$、新しい候補のlossを$L_t$、意味のある最小改善量を$\delta$とする。新しい候補へ更新する条件を、

$$
L_t < L_{\mathrm{best}}-\delta
$$

とする。

これにより、丸め誤差や極小の変動だけでcheckpointを置き換えない。同点または改善が$\delta$以下なら、より早いepochを残す規則は、同等性能なら追加学習の少ないモデルを選ぶという解釈を持つ。

### 小さい数値例

`min_delta = 0.0001`とする。

| epoch | validation loss | epoch 0からの改善 | 選択 |
| ---: | ---: | ---: | :-- |
| 0 | 0.30000 | - | 初期best |
| 1 | 0.29995 | 0.00005 | 改善が小さいためepoch 0を維持 |
| 2 | 0.29970 | 0.00030 | epoch 2へ更新 |

## 4. 選択指標と診断指標を分ける

checkpoint選択をvalidation lossに固定したなら、accuracy、agreement、logits RMSEなどは結果を理解するための診断指標である。結果を見た後で候補ごとに選択指標を切り替えると、選択規則自体が新たな自由度になり、都合のよい結果を選びやすくなる。

必要なら最初から、

- accuracy最大化
- loss最小化
- 複数指標の制約付き選択
- Pareto selection

のどれを使うか決める。異なる指標で別のモデルが良くても矛盾ではなく、目的関数が異なる。

## 5. accuracyとCross Entropy lossは異なる量である

多クラス分類で、sampleの正解ラベルを$y$、softmax確率を$p_k$とすると、Cross Entropyは、

$$
\ell
=
-\log p_y
$$

である。一方、accuracyは、

$$
\operatorname*{arg\,max}_k p_k = y
$$

であるかだけを見る。

### 小さい数値例

正解クラスをclass 1とする。

| 確率ベクトル | 判定 | Cross Entropy |
| :-- | :-- | ---: |
| $(0.90,0.05,0.05)$ | 正解 | 約0.105 |
| $(0.51,0.49,0.00)$ | 正解 | 約0.673 |
| $(0.10,0.89,0.01)$ | 不正解 | 約2.303 |
| $(0.01,0.98,0.01)$ | 不正解 | 約4.605 |

上の2行はaccuracyでは同じ1件の正解だが、lossは異なる。下の2行もaccuracyでは同じ1件の不正解だが、正解クラスへの確率が低いほどlossが大きい。

従って、少数のsampleが新たに正解へ変わってaccuracyが上昇しても、他のsampleで正解確率が下がれば平均lossは悪化し得る。集計されたaccuracyとlossだけでは、その内訳やcalibrationの良否までは分からない。

## 6. post-hocな二者比較と完全なearly stoppingは異なる

学習後にepoch 0と保存済みbestだけを比較する方法は、既存artifactを使って選択漏れを検査できる。しかし、これは候補集合、

$$
\{\theta_0,\theta_{\mathrm{saved\ best}}\}
$$

だけの二者比較である。

完全な再選択には全epochのcheckpointまたは履歴が必要である。また、epoch 0を最初からbestとして登録すると、`patience`の起点や停止epochが変わる場合がある。従って、post-hocな二者比較を「epoch 0込みでearly stoppingを再実行した」と表現しない。

## 7. 候補集合の拡張は過去の測定値の訂正ではない

epoch 1以降だけから選んだbestと、epoch 0を加えて選んだbestは、異なる候補集合に対する答えである。保存値が正しく再現されるなら、以前の結果は当時の規則の下で有効であり、新しい結果は候補集合を拡張した判断である。

この履歴を残すと、

1. 学習後の候補を評価した。
2. 学習前より悪化する場合に気づいた。
3. 保存・復元・再評価を検証した。
4. epoch 0を候補へ加えた。

という改善過程を追跡できる。

## 8. 保存checkpointの再現確認

保存値$m_{\mathrm{saved}}$と、checkpointを読み直して同じ評価手順で得た値$m_{\mathrm{fresh}}$を比較する。例えば絶対差、

$$
\Delta_m
=
\left|m_{\mathrm{fresh}}-m_{\mathrm{saved}}\right|
$$

が指標ごとの許容差以内か確認する。

一致すれば、次の可能性を小さくできる。

- 指標の単純な転記ミス
- checkpoint読み込みによる大きな復元不良
- 元の実行時だけの一時的な評価異常
- 許容差を超える再評価の揺れ

ただし、同じコードに共通の不具合があれば同じ誤値を再現し得る。また、保存済みモデルの再評価は、新しい初期化・shuffle・seedで学習を繰り返すことではない。

## 9. 3種類の再現性を区別する

| 種類 | 確認するもの | 確認できないもの |
| :-- | :-- | :-- |
| artifact再現性 | 同じcheckpointから同じ指標が得られるか | 学習を繰り返して同じcheckpointになるか |
| training再現性 | 同じseed・環境で学習結果を再生成できるか | seedを変えて傾向が維持されるか |
| statistical robustness | 複数seedで効果とばらつきが安定するか | 未使用testへの最終的な汎化 |

「再現した」という語だけでは範囲が曖昧になるため、どの再現性を確認したか明示する。

## 10. 実験設計上の実用的な手順

1. fine-tuning前にepoch 0を評価する。
2. epoch 0をbest checkpointとして登録する。
3. 事前に固定したvalidation指標と`min_delta`で更新する。
4. accuracyやdense再現性は診断指標として併記する。
5. checkpointを保存後に読み直し、指標を再現する。
6. validationで全条件を固定するまでtest setを使わない。
7. 重要な結論は複数seedで確認する。

## 11. まとめ

- epoch 0は「fine-tuningしない」という有効なcheckpoint候補である。
- checkpointは事前に固定したvalidation指標で選び、他の指標は分けて報告する。
- accuracyはargmax、Cross Entropyは正解ラベルへの確率を見るため、逆方向へ動き得る。
- post-hocなepoch 0対saved best比較は有用だが、全epochやpatience trajectoryの再計算ではない。
- 保存モデルの再評価一致はartifactの整合性を支持するが、学習再現性や実装全体の正しさは保証しない。
