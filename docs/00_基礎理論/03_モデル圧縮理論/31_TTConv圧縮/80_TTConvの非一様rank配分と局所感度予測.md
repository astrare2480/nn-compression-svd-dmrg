---
title: TTConvの非一様rank配分と局所感度予測
aliases:
  - nonuniform TTConv rank allocation
  - local sensitivity rank allocation for TTConv
tags:
  - TTConv
  - CNN
  - nonuniform rank
  - local sensitivity
  - model selection
---

# TTConvの非一様rank配分と局所感度予測

## このノートの目的

学習済みTTConvの内部bond rankを事後的に削るとき、全bondを同じrankへ切る必要はない。本ノートでは、

1. 非一様rank候補を同じ費用予算で比較する方法
2. Weight・Activation・Task基準の違い
3. 単一bondの局所感度から複数bond配分を予測する方法
4. 候補評価数と実時間を分ける理由
5. MLPとCNNで共通する点・異なる点

を一般論として整理する。

実測結果は次へ分ける。

- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/05_事後非一様TTConv_rank配分と選択基準比較]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/06_単一bond局所感度によるrank配分予測]]

TT全般のrank選択は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/80_適応bond_rank選択と非一様rank配分]]も参照する。

---

## 1. sourceモデルを固定する

候補比較では、全候補を同じsource checkpointから生成する。sourceには少なくとも次を記録する。

- tensorization
- core shapesとactual ranks
- fine-tuning済みか、TT-SVD直後か
- 更新範囲
- 選択epochとdata split
- source accuracy、loss、parameter数、理論MACs

fine-tuning済みsourceと未学習sourceでは、spectrumと分類感度が異なり得る。MLPとCNNの結果を比べるときも、source学習条件をそろえずにモデル種別だけの差へ帰属しない。

---

## 2. TTConvの非一様rank候補

3-core TTConvで内部rankが二つなら、候補を、

$$
\mathbf r=(r_1,r_2)
$$

と書ける。各bondの候補集合が、

$$
\mathcal R_1=\mathcal R_2=\{12,16,20,24\}
$$

なら、全候補数は、

$$
|\mathcal R_1|\,|\mathcal R_2|=4^2=16
$$

である。

一様rank $(20,20)$と非一様rank $(16,24)$は、同じ「平均rank」では比較しない。TT core要素数とdirect contractionの理論MACsを実際のshapeから計算する。

---

## 3. 費用予算を先に固定する

sourceより小さい候補を探すだけでは、性能と費用の交換関係を比較できない。基準候補$\mathbf r_{mathrm{base}}$の費用を$B$として、

$$
\mathcal C_B
=
\left\{
\mathbf r
\mid
C(\mathbf r)\le C(\mathbf r_{mathrm{base}})
\right\}
$$

から選ぶ。

$C$には、

- TTConv対象層parameter数
- モデル総parameter数
- 対象層理論MACs
- モデル総理論MACs

があり、同じ順位になるとは限らない。主目的に合う費用軸を事前に決める。

理論MAC削減は実測latency削減を保証しない。kernel launch、メモリアクセス、中間Tensor、batch size、実装最適化が影響する。

---

## 4. Weight・Activation・Task基準

### Weight-aware

dense kernelまたは実効TTConv kernelの相対誤差を使う。

$$
S_W(\mathbf r)
=
\frac{\|W_{\mathbf r}-W_0\|_F}{\|W_0\|_F}
$$

入力不要だが、実画像のpatch分布を見ない。

### Activation-aware

calibration画像に対する対象conv出力を使う。

$$
S_A(\mathbf r)
=
\frac{
\|Z_{\mathbf r}-Z_0\|_F
}{
\|Z_0\|_F
}
$$

Convでは、入力画像そのものではなく`unfold`後のpatch分布とkernelの作用が反映される。

### Task-aware

calibrationのcross-entropy増加を使う。

$$
S_T(\mathbf r)
=
\mathcal L_{mathrm{cal}}(\mathbf r)
-
\mathcal L_{mathrm{cal}}(\mathrm{source})
$$

後段とlabelを含むが、calibrationへ選択が適応する。validationで選択規則を評価し、testは最終候補固定後まで使わない。

三基準が同じ候補を選んでも一般的な同値性は示さない。逆に異なる候補を選んでも、測っている段階が違うため矛盾ではない。

---

## 5. no-cut control

候補生成処理がsourceを壊していないことを確認するため、元rankをそのまま指定したno-cut候補を作る。

確認項目は、

- effective kernel
- 対象層出力
- logits
- prediction
- parameter shapes

である。丸め誤差範囲を事前に定める。

no-cut controlの一致は、rank削減候補の妥当性や分類性能を保証するものではない。候補生成経路の実装controlである。

---

## 6. 単一bond局所感度

source rankを$(R_1,R_2)$とする。bond 1だけを削った候補$(r_1,R_2)$と、bond 2だけを削った候補$(R_1,r_2)$を測る。

単一bond scoreを、

$$
s_1(r_1),\qquad s_2(r_2)
$$

とする。複数bond候補$(r_1,r_2)$のscoreを直接加算で、

$$
\widehat S_{mathrm{sum}}(r_1,r_2)
=
s_1(r_1)+s_2(r_2)
$$

と近似できるかを調べる。

scoreが相対L2のようなノルムなら、独立な二乗誤差を仮定した、

$$
\widehat S_{mathrm{sq}}(r_1,r_2)
=
\sqrt{s_1(r_1)^2+s_2(r_2)^2}
$$

も候補になる。

どちらも厳密式ではない。複数bondを切ると、

- 最初の切り詰めが次のspectrumを変える。
- 誤差方向間の交差項がある。
- ReLUと後段によりtask lossは非線形に変わる。

ためである。

---

## 7. 予測scoreの評価を三つに分ける

### 7.1 score再現

全候補の実測scoreと予測scoreの順位相関や誤差を見る。

### 7.2 選択一致

予算内で、予測scoreが選んだrank配分と実測scoreが選んだ配分が一致するかを見る。

### 7.3 validation性能

選択した配分が、一様対照やvalidation oracleと比べてどのaccuracy・lossを持つかを見る。

予測scoreを正確に再現しても、そのscore自体がtask性能と弱くしか対応しなければ、validation性能は良くならない。逆にscore選択が一致しなくても、選択候補同士のvalidation性能が近いことはある。

---

## 8. 評価候補数と実時間

各bondにsource以外の削減候補が$m$個、対象bondが$b$本あるとする。

全組合せ評価は、

$$
(m+1)^b
$$

候補である。sourceと各単一bond削減だけなら、

$$
1+bm
$$

候補で済む。

$b=2,m=3$なら、全組合せ16候補に対して局所診断は7候補である。

ただし、候補数削減はそのままwall time短縮ではない。候補生成、kernel再構成、calibration forward、保存、GPU同期の費用が異なるため、速度向上を主張するには実測時間が必要である。

---

## 9. 結果の解釈

### 非一様候補が一様候補を上回った場合

そのsource・予算・候補集合では、bond位置に応じたrank配分が有効だった。非一様rank一般の優越性ではない。

### ActivationとTaskが同じ候補を選んだ場合

その条件で、対象層出力の保存とtask lossの保存が同じ配分を支持した。両指標の一般的同値性ではない。

### 局所予測が成功した場合

その候補集合内では、単一bond情報から有用な配分を選べた。候補集合の外、強い削減、別sourceへの外挿は未検証である。

### Weight基準が外れた場合

重み誤差が無意味なのではない。実入力のpatch分布、後段、classification marginを含まないため、task性能の順位を直接保証しなかったと読む。

---

## 10. チェックリスト

- source checkpointと学習履歴を固定する。
- 全候補を同じsourceから独立に生成する。
- no-cut controlを入れる。
- rank配分と切り詰め順序を保存する。
- 対象層費用とモデル全体費用を分ける。
- calibration、validation、testを分ける。
- 一様対照とoracleを残す。
- score再現、選択一致、validation性能を分ける。
- 評価候補数と実時間を分ける。
- MLPとCNNをsource条件なしに直接比較しない。

## 関連

- [[00_基礎理論/03_モデル圧縮理論/31_TTConv圧縮/79_Conv2dのTT-matrix表現と直接forward]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/80_適応bond_rank選択と非一様rank配分]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/05_事後非一様TTConv_rank配分と選択基準比較]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/06_単一bond局所感度によるrank配分予測]]
